# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import math

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# [NEW INCREMENTAL PATCH v24.14] LEI (@TheMarketMemo) × J Law (@jlawstock) 實戰結構盈虧比、精準買賣位與回測統一引擎
# (完全保留原有代碼，僅以增量方式提供：
#  1. 精準結構買入位/沽空位計算 (突破樞紐掛單 vs 回踩支撐/均線低吸位，絕不盲目用現價當買入價)
#  2. 多層級結構支撐/阻力止損止盈與「每隻股票獨立真實盈虧比 (Variable Structural R:R)」
#  3. 數字編號 (#1, #2, #3...) 優先買入與優先沽空行動榜 + 精簡/專家模式切換 (修復 KeyError 與 removeChild)
#  4. 診斷 K 線圖預設關閉對數座標並聚焦最近 150 根 K 線 (修復圖表水平壓縮問題)
#  5. 掃描系統與回測系統 (含快速勝率與雙向回測) 100% 理念同步統一)
# ==============================================================================
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
swing_logger = logging.getLogger("SwingTraderAudit")

RADAR_CACHE_TTL_SECONDS = 1800  # 雷達結果預設 30 分鐘自動過期重算
CURRENT_SCHEMA_VERSION = "v24.14"

def is_quad_witching_friday(index_dates: pd.Index) -> pd.Series:
    """
    [J Law 影片10：股市下跌前4大見頂訊號 15:10-15:38]
    每年 3、6、9、12 月的第 3 個星期五為期貨結算日 (四巫日)，當日異常放大成交量不計入大戶出貨日 (Distribution Day)。
    """
    try:
        dt_idx = pd.to_datetime(index_dates)
        is_qw = (dt_idx.month.isin([3, 6, 9, 12])) & (dt_idx.weekday == 4) & (dt_idx.day >= 15) & (dt_idx.day <= 21)
        return pd.Series(is_qw, index=index_dates)
    except Exception:
        return pd.Series(False, index=index_dates)

def ensure_persistent_schema_fresh(persistent_dict: dict) -> None:
    """[修復 NotFoundError / KeyError] 當程式碼升級後，自動清理舊版結構不相容的暫存 DataFrame"""
    if persistent_dict.get("_schema_ver") != CURRENT_SCHEMA_VERSION:
        persistent_dict["radar_results"] = {"港股": None, "美股": None, "自選股": None, "自訂": None}
        persistent_dict["radar_correlation_warnings"] = {"港股": [], "美股": [], "自選股": [], "自訂": []}
        persistent_dict["radar_correlation_matrices"] = {"港股": None, "美股": None, "自選股": None, "自訂": None}
        persistent_dict["watchlist_radar_results"] = {}
        persistent_dict["radar_timestamps"] = {}
        persistent_dict["signal_lifecycle_store"] = {}
        persistent_dict["radar_breadth_stats"] = {}
        persistent_dict["_schema_ver"] = CURRENT_SCHEMA_VERSION

def is_radar_cache_expired(persistent_dict: dict, market_key: str, ttl_seconds: int = RADAR_CACHE_TTL_SECONDS) -> bool:
    """檢查持久化雷達緩存是否已過期或跨交易日"""
    ts_map = persistent_dict.setdefault("radar_timestamps", {})
    last_ts = ts_map.get(market_key)
    if last_ts is None:
        return True
    now_ts = time.time()
    if (now_ts - last_ts) > ttl_seconds:
        return True
    if datetime.fromtimestamp(last_ts).date() != datetime.fromtimestamp(now_ts).date():
        return True
    return False

def mark_radar_cache_updated(persistent_dict: dict, market_key: str) -> None:
    """記錄雷達掃描時間戳與版本"""
    ts_map = persistent_dict.setdefault("radar_timestamps", {})
    ts_map[market_key] = time.time()

def compute_wilder_atr_and_rsi(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """Wilder's RMA 標準 ATR 與 RSI"""
    tr = np.maximum(
        df['High'] - df['Low'],
        np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1)))
    )
    atr_wilder = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rsi_wilder = 100.0 - (100.0 / (1.0 + (gain / (loss + 1e-9))))
    return atr_wilder, rsi_wilder

def compute_causal_swing_sr(df: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
    """
    [無未來函數之多層級結構支撐與阻力系統]
    計算 10日箱體、20日短線波段、60日季線波段、120日半年波段之真實高低點與前波突破頸線 (S/R Flip Level)。
    """
    df['BoxHigh10_Causal'] = df['High'].shift(1).rolling(window=10, min_periods=3).max().fillna(df['High'])
    df['BoxLow10_Causal'] = df['Low'].shift(1).rolling(window=10, min_periods=3).min().fillna(df['Low'])
    df['ResShort_Causal'] = df['High'].shift(1).rolling(window=short_window, min_periods=5).max().fillna(df['High'])
    df['SupShort_Causal'] = df['Low'].shift(1).rolling(window=short_window, min_periods=5).min().fillna(df['Low'])
    df['ResLong_Causal'] = df['High'].shift(1).rolling(window=long_window, min_periods=10).max().fillna(df['High'])
    df['SupLong_Causal'] = df['Low'].shift(1).rolling(window=long_window, min_periods=10).min().fillna(df['Low'])
    df['Res120_Causal'] = df['High'].shift(1).rolling(window=120, min_periods=20).max().fillna(df['High'])
    df['Sup120_Causal'] = df['Low'].shift(1).rolling(window=120, min_periods=20).min().fillna(df['Low'])
    df['PriorWaveHigh_Causal'] = df['High'].shift(10).rolling(window=35, min_periods=5).max().fillna(df['ResShort_Causal'])
    df['PriorWaveLow_Causal'] = df['Low'].shift(10).rolling(window=35, min_periods=5).min().fillna(df['SupShort_Causal'])
    return df

def compute_lei_dual_ma_and_ppo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [LEI @TheMarketMemo 雙均線系統 (20/60/120 MA+EMA) + 歷史均線密集區籌碼峰 + J Law 緊密箱體]
    """
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    df['EMA60'] = df['Close'].ewm(span=60, adjust=False).mean()
    df['SMA60_Deduction'] = df['Close'].shift(60)
    df['Ret6M'] = df['Close'] / df['Close'].shift(126) - 1.0

    df['MACD_Pct'] = (df['EMA12'] - df['EMA26']) / (df['EMA26'] + 1e-9) * 100.0
    df['MACDSignal_Pct'] = df['MACD_Pct'].ewm(span=9, adjust=False).mean()
    df['Bias_EMA120_Pct'] = (df['Close'] - df['EMA120']) / (df['EMA120'] + 1e-9) * 100.0

    df['Tight_Box_5d_Pct'] = (df['High'].rolling(5).max() - df['Low'].rolling(5).min()) / (df['Close'] + 1e-9) * 100.0
    df['JLaw_Tight_Box'] = df['Tight_Box_5d_Pct'] <= 5.5

    df['VolSMA10'] = df['Volume'].rolling(10).mean()
    df['VolSMA30'] = df['Volume'].rolling(30).mean()
    df['VolSMA60'] = df['Volume'].rolling(60).mean()
    df['VolSMA90'] = df['Volume'].rolling(90).mean()

    if len(df) >= 120:
        lei_cols = ['SMA20', 'EMA20', 'SMA60', 'EMA60', 'SMA120', 'EMA120']
        df['LEI_MA_Max'] = df[lei_cols].max(axis=1)
        df['LEI_MA_Min'] = df[lei_cols].min(axis=1)
        df['LEI_MA_Mid'] = (df['LEI_MA_Max'] + df['LEI_MA_Min']) / 2.0
        df['LEI_MA_Spread_Pct'] = (df['LEI_MA_Max'] - df['LEI_MA_Min']) / (df['LEI_MA_Min'] + 1e-9) * 100.0
        df['LEI_MA_Tight_2Pct'] = df['LEI_MA_Spread_Pct'] <= 2.5
    else:
        df['LEI_MA_Mid'] = df['Close']
        df['LEI_MA_Spread_Pct'] = 999.0
        df['LEI_MA_Tight_2Pct'] = False

    return df

def detect_lei_top_bottom_and_2b(df: pd.DataFrame, lookback: int = 35) -> Dict[str, Any]:
    """
    [LEI @TheMarketMemo《頂底構造》&《我的交易系統》]
    包含 K 線合併 + 頂部/底部構造 + 2B 破底翻/假突破 + 破線/拐頭/交叉三部曲
    """
    res = {
        "bottom_structure": False,
        "top_structure": False,
        "bullish_2b": False,
        "bearish_2b": False,
        "lei_turn_step": "無明顯轉折",
        "merged_bars_count": 0,
        "structure_low": np.nan,
        "structure_high": np.nan,
        "structure_summary": "⚪ 未出現頂底構造"
    }
    if df is None or len(df) < 15:
        return res

    sub = df.tail(lookback)
    merged: List[Dict[str, Any]] = []
    for idx, row in sub.iterrows():
        h, l, c = float(row['High']), float(row['Low']), float(row['Close'])
        if not merged:
            merged.append({"date": idx, "high": h, "low": l, "close": c})
        else:
            prev = merged[-1]
            if h <= prev["high"] and l >= prev["low"]:
                prev["close"] = c
                continue
            merged.append({"date": idx, "high": h, "low": l, "close": c})

    res["merged_bars_count"] = len(merged)
    if len(merged) >= 4:
        b1, b2, b3 = merged[-3], merged[-2], merged[-1]
        recent_min_low = min(m["low"] for m in merged[-6:])
        res["structure_low"] = float(min(b1["low"], b2["low"], b3["low"]))
        res["structure_high"] = float(max(b1["high"], b2["high"], b3["high"]))
        if (b1["low"] <= recent_min_low * 1.015 or b2["low"] <= recent_min_low * 1.015):
            if b3["high"] > b2["high"] and b3["low"] >= b2["low"] and b3["close"] > b2["high"]:
                res["bottom_structure"] = True
            elif b2["low"] < b1["low"] and b3["high"] > b2["high"] and b3["low"] > b2["low"]:
                res["bottom_structure"] = True

        recent_max_high = max(m["high"] for m in merged[-6:])
        if (b1["high"] >= recent_max_high * 0.985 or b2["high"] >= recent_max_high * 0.985):
            if b3["low"] < b2["low"] and b3["high"] <= b2["high"] and b3["close"] < b2["low"]:
                res["top_structure"] = True
            elif b2["high"] > b1["high"] and b3["low"] < b2["low"] and b3["high"] < b2["high"]:
                res["top_structure"] = True

    if len(df) >= 25:
        last = df.iloc[-1]
        prior_window = df.iloc[-22:-3]
        prior_low_A = float(prior_window['Low'].min())
        prior_high_A = float(prior_window['High'].max())
        recent_3_low = float(df['Low'].iloc[-3:].min())
        recent_3_high = float(df['High'].iloc[-3:].max())

        if recent_3_low < prior_low_A and float(last['Close']) > prior_low_A and float(last['Close']) > float(last['Open']):
            res["bullish_2b"] = True
            res["bottom_structure"] = True
            res["structure_low"] = recent_3_low

        if recent_3_high > prior_high_A and float(last['Close']) < prior_high_A and float(last['Close']) < float(last['Open']):
            res["bearish_2b"] = True
            res["top_structure"] = True
            res["structure_high"] = recent_3_high

    last_row = df.iloc[-1]
    c_val = float(last_row['Close'])
    ema20_v = float(last_row.get('EMA20', c_val))
    sma20_v = float(last_row.get('SMA20', c_val))
    sma60_v = float(last_row.get('SMA60', last_row.get('SMA50', c_val)))
    ded20_v = float(last_row.get('SMA20_Deduction', c_val))

    step1_break = c_val > ema20_v and c_val > sma20_v
    step2_turn = c_val > ded20_v
    step3_cross = sma20_v > sma60_v and ema20_v > float(last_row.get('EMA60', sma60_v))

    if step1_break and step2_turn and step3_cross:
        res["lei_turn_step"] = "✅ 三部曲全滿 (破線+拐頭+20/60多頭交叉)"
    elif step1_break and step2_turn:
        res["lei_turn_step"] = "🟡 階段二：已破線且20MA向上拐頭 (待均線交叉)"
    elif step1_break:
        res["lei_turn_step"] = "🟠 階段一：剛破線站上20均線 (待抵扣價拐頭)"
    else:
        res["lei_turn_step"] = "🔴 均線下方運行 (空頭或整理姿態)"

    tags = []
    if res["bullish_2b"]:
        tags.append("🟢 2B破底翻(假跌破真反轉)")
    elif res["bottom_structure"]:
        tags.append("🟢 底構造確認(HL+HH)")
    if res["bearish_2b"]:
        tags.append("🔴 2B假突破(誘多轉空)")
    elif res["top_structure"]:
        tags.append("🔴 頂構造警報(LH+LL)")
    res["structure_summary"] = " ｜ ".join(tags) if tags else "⚪ 延續原波段 (未見頂底構造)"
    return res

def analyze_lei_deduction_forecast(df: pd.DataFrame) -> Dict[str, Any]:
    """[LEI @TheMarketMemo 未來 5~10 日抵扣價方框推演]"""
    out = {
        "ded20_now": 0.0,
        "ded20_next5d_max": 0.0,
        "ded20_next10d_mean": 0.0,
        "ded60_now": 0.0,
        "ded60_next10d_mean": 0.0,
        "forecast_20_label": "⚪ 資料不足",
        "forecast_60_label": "⚪ 資料不足",
        "is_20ma_turn_up_favorable": False,
        "is_60ma_turn_up_favorable": False
    }
    if df is None or len(df) < 65:
        return out

    close_now = float(df['Close'].iloc[-1])
    ded20_now = float(df['Close'].iloc[-20])
    next10_for_20 = df['Close'].iloc[-19:-9]
    ded20_next5_max = float(df['Close'].iloc[-19:-14].max()) if len(df) >= 20 else ded20_now
    ded20_next10_mean = float(next10_for_20.mean()) if not next10_for_20.empty else ded20_now

    ded60_now = float(df['Close'].iloc[-60])
    next10_for_60 = df['Close'].iloc[-59:-49]
    ded60_next10_mean = float(next10_for_60.mean()) if not next10_for_60.empty else ded60_now

    out["ded20_now"] = ded20_now
    out["ded20_next5d_max"] = ded20_next5_max
    out["ded20_next10d_mean"] = ded20_next10_mean
    out["ded60_now"] = ded60_now
    out["ded60_next10d_mean"] = ded60_next10_mean

    if close_now > ded20_next10_mean and ded20_next10_mean < ded20_now * 0.98:
        out["forecast_20_label"] = f"🚀 未來10日抵扣價快速下移 (均值 {ded20_next10_mean:.2f} < 現價)，20MA 將強勢上翹"
        out["is_20ma_turn_up_favorable"] = True
    elif close_now >= ded20_next5_max:
        out["forecast_20_label"] = f"✅ 現價高於未來5日最高抵扣價 ({ded20_next5_max:.2f})，20MA 穩步向上"
        out["is_20ma_turn_up_favorable"] = True
    else:
        out["forecast_20_label"] = f"⚠️ 未來5日抵扣高點 ({ded20_next5_max:.2f}) 高於現價，需突破該位 20MA 才可拐頭向上"

    if close_now > ded60_next10_mean and ded60_next10_mean < ded60_now * 0.98:
        out["forecast_60_label"] = f"🚀 未來10日季線抵扣價下移 (均值 {ded60_next10_mean:.2f})，60MA 助漲空間打開"
        out["is_60ma_turn_up_favorable"] = True
    elif close_now >= ded60_next10_mean:
        out["forecast_60_label"] = f"✅ 現價守於未來10日 60MA 抵扣均值 ({ded60_next10_mean:.2f}) 之上，季線維持向上"
        out["is_60ma_turn_up_favorable"] = True
    else:
        out["forecast_60_label"] = f"⚠️ 現價低於未來10日 60MA 抵扣均值 ({ded60_next10_mean:.2f})，季線仍有下壓阻力"

    return out

def _find_historical_ma_convergence_targets(df: pd.DataFrame, ref_price: float, direction: str) -> List[Tuple[float, str]]:
    """
    [LEI @TheMarketMemo《我的交易系統》38:35-42:15 &《怎樣做交易計劃》08:05-09:05]
    尋找歷史上曾經形成「均線密集區 (籌碼峰成本共識區)」或關鍵波峰/波谷的價格區間，作為真實空間阻力/支撐目標位。
    """
    candidates: List[Tuple[float, str]] = []
    last = df.iloc[-1]

    res_short = float(last.get('ResShort_Causal', last.get('ResShort', np.nan)))
    res_long = float(last.get('ResLong_Causal', last.get('ResLong', np.nan)))
    res_120 = float(last.get('Res120_Causal', np.nan))
    high_52w = float(last.get('High52W', np.nan))

    sup_short = float(last.get('SupShort_Causal', last.get('SupShort', np.nan)))
    sup_long = float(last.get('SupLong_Causal', last.get('SupLong', np.nan)))
    sup_120 = float(last.get('Sup120_Causal', np.nan))
    low_52w = float(last.get('Low52W', np.nan))

    if 'LEI_MA_Spread_Pct' in df.columns and 'LEI_MA_Mid' in df.columns and len(df) > 40:
        hist_slice = df.iloc[-min(len(df), 200):-10]
        tight_zones = hist_slice[hist_slice['LEI_MA_Spread_Pct'] < 3.5]
        if not tight_zones.empty:
            recent_cluster_p = float(tight_zones['LEI_MA_Mid'].iloc[-1])
            max_cluster_p = float(tight_zones['LEI_MA_Mid'].max())
            min_cluster_p = float(tight_zones['LEI_MA_Mid'].min())
            candidates.append((recent_cluster_p, "前次均線密集籌碼峰"))
            candidates.append((max_cluster_p, "上方歷史均線密集區"))
            candidates.append((min_cluster_p, "下方歷史均線密集區"))

    if direction == "LONG":
        if pd.notna(res_short): candidates.append((res_short, "20日前浪頂阻力位"))
        if pd.notna(res_long): candidates.append((res_long, "60日波段前高阻力"))
        if pd.notna(res_120): candidates.append((res_120, "半年線大級別前高阻力"))
        if pd.notna(high_52w): candidates.append((high_52w, "52週歷史前高阻力區"))
        for ma_name, label in [('SMA60', '60MA季線反壓位'), ('SMA120', '120MA半年線反壓位'), ('SMA200', '200MA年線大阻力')]:
            val = last.get(ma_name, np.nan)
            if pd.notna(val) and float(val) > ref_price * 1.02:
                candidates.append((float(val), label))
        valid = [(p, lbl) for p, lbl in candidates if pd.notna(p) and p > ref_price * 1.022]
        valid.sort(key=lambda x: x[0])
        return valid
    else:
        if pd.notna(sup_short): candidates.append((sup_short, "20日前浪底支撐位"))
        if pd.notna(sup_long): candidates.append((sup_long, "60日波段前低支撐"))
        if pd.notna(sup_120): candidates.append((sup_120, "半年線大級別前低支撐"))
        if pd.notna(low_52w): candidates.append((low_52w, "52週前低支撐區"))
        for ma_name, label in [('SMA60', '60MA季線支撐位'), ('SMA120', '120MA半年線支撐位'), ('SMA200', '200MA年線大支撐')]:
            val = last.get(ma_name, np.nan)
            if pd.notna(val) and float(val) < ref_price * 0.98:
                candidates.append((float(val), label))
        valid = [(p, lbl) for p, lbl in candidates if pd.notna(p) and p < ref_price * 0.978]
        valid.sort(key=lambda x: x[0], reverse=True)
        return valid

def evaluate_swing_trade_plan_and_rr(
    df: pd.DataFrame,
    plan_type: str,
    status: str,
    atr_val: float,
    is_breakout_long: bool = False,
    is_near_20ma: bool = False,
    is_deduction_pullback: bool = False,
    is_breakdown_short: bool = False,
    lei_struct: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    [v24.14 結構買入位、結構止損與多層級真實阻力止盈]
    """
    last = df.iloc[-1]
    close_p = float(last['Close'])
    high_p = float(last['High'])
    low_p = float(last['Low'])
    atr_safe = float(atr_val) if pd.notna(atr_val) and atr_val > 0 else close_p * 0.03

    box_high_10 = float(last.get('BoxHigh10_Causal', df['High'].iloc[-11:-1].max() if len(df) > 11 else high_p))
    box_low_10 = float(last.get('BoxLow10_Causal', df['Low'].iloc[-11:-1].min() if len(df) > 11 else low_p))
    res_short = float(last.get('ResShort_Causal', last.get('ResShort', close_p * 1.08)))
    sup_short = float(last.get('SupShort_Causal', last.get('SupShort', close_p * 0.92)))
    prior_wave_high = float(last.get('PriorWaveHigh_Causal', res_short))
    prior_wave_low = float(last.get('PriorWaveLow_Causal', sup_short))
    sma10_p = float(last.get('SMA10', close_p))
    sma20_p = float(last.get('SMA20', close_p))
    sma50_p = float(last.get('SMA50', close_p))
    ema20_p = float(last.get('EMA20', sma20_p))

    if plan_type == "LONG":
        if is_near_20ma or is_deduction_pullback or "回踩" in status or "回撤" in status:
            support_anchor = max(min(close_p, sma20_p * 1.005), ema20_p * 0.998)
            if close_p > sma20_p and (close_p - sma20_p) / sma20_p <= 0.02:
                recommended_entry = round(support_anchor, 2)
                entry_type_desc = f"低吸回踩單 (回測 20MA 支撐區 {sma20_p:.2f} 承接)"
            else:
                recommended_entry = round(min(high_p + 0.05 * atr_safe, close_p * 1.008), 2)
                entry_type_desc = f"底構造確認單 (突破回撤確認位 {recommended_entry:.2f} 買入)"
        elif is_breakout_long or "突破" in status or "旗型" in status:
            pivot_buy = box_high_10 + 0.08 * atr_safe
            if close_p > box_high_10 * 1.03:
                retest_level = max(box_high_10, sma10_p, prior_wave_high if prior_wave_high <= close_p else sma10_p)
                recommended_entry = round(retest_level, 2)
                entry_type_desc = f"等回測低吸單 (現價已高出突破位 {(close_p/box_high_10 - 1)*100:.1f}%，勿追高，等回踩前高支撐 {recommended_entry:.2f} 買入)"
            else:
                recommended_entry = round(max(close_p * 1.002, min(pivot_buy, close_p + 0.35 * atr_safe)), 2)
                entry_type_desc = f"突破觸發單 (Stop-Buy 升穿箱體樞紐 {recommended_entry:.2f} 進場)"
        elif bool(last.get('VCPSignal', False)) or bool(last.get('JLaw_Tight_Box', False)) or bool(last.get('LEI_MA_Tight_2Pct', False)) or "壓縮" in status or "密集" in status:
            recommended_entry = round(box_high_10 + 0.05 * atr_safe, 2)
            entry_type_desc = f"突破埋伏單 (待升穿近10日緊密平台頂 {recommended_entry:.2f} 觸發買入)"
        else:
            pullback_target = max(sma20_p, min(sma10_p, prior_wave_high if prior_wave_high < close_p else sma10_p))
            if abs(close_p - pullback_target) / close_p < 0.005:
                pullback_target = close_p * 0.985
            recommended_entry = round(pullback_target, 2)
            entry_type_desc = f"拉回等待單 (目前無即時突破，建議等縮量回踩 10/20MA 支撐位 {recommended_entry:.2f} 買入)"

        struct_low_candidates = [
            (box_low_10 - 0.15 * atr_safe, f"近10日緊密箱體底 ({box_low_10:.2f}) 下方"),
            (sup_short - 0.20 * atr_safe, f"20日波段前浪底支撐 ({sup_short:.2f}) 外側"),
            (sma20_p - 0.50 * atr_safe, f"20MA 生命線 ({sma20_p:.2f}) 下方緩衝"),
            (sma50_p - 0.30 * atr_safe, f"50MA 中期支撐 ({sma50_p:.2f}) 下方")
        ]
        if lei_struct and pd.notna(lei_struct.get("structure_low")):
            sl_st = float(lei_struct["structure_low"]) - 0.15 * atr_safe
            struct_low_candidates.append((sl_st, f"LEI 底部構造低點 ({lei_struct['structure_low']:.2f}) 下方"))

        valid_sls = [
            (sl_p, sl_lbl) for sl_p, sl_lbl in struct_low_candidates
            if pd.notna(sl_p) and (recommended_entry * 0.925 <= sl_p <= recommended_entry * 0.986)
        ]
        if valid_sls:
            best_sl_pair = max(valid_sls, key=lambda x: x[0] if (recommended_entry - x[0]) >= 0.85 * atr_safe else -999999)
            if (recommended_entry - best_sl_pair[0]) < 0.85 * atr_safe:
                best_sl_pair = min(valid_sls, key=lambda x: x[0])
            stop_loss = round(best_sl_pair[0], 2)
            sl_basis_desc = best_sl_pair[1]
        else:
            fallback_sl = max(recommended_entry - 1.8 * atr_safe, recommended_entry * 0.928)
            stop_loss = round(min(fallback_sl, recommended_entry * 0.982), 2)
            sl_basis_desc = f"波動率與 J Law ≤7.2% 硬止損保護位"

        risk_per_share = max(recommended_entry - stop_loss, 1e-4)
        stop_pct = (risk_per_share / recommended_entry) * 100.0

        sr_targets = _find_historical_ma_convergence_targets(df, recommended_entry, "LONG")
        recent_swing_amp = max(res_short - sup_short, 3.2 * atr_safe)
        measured_move_tp = round(recommended_entry + recent_swing_amp * 0.85, 2)

        if sr_targets:
            first_sr_p, first_sr_lbl = sr_targets[0]
            first_sr_rr = (first_sr_p - recommended_entry) / risk_per_share
            if first_sr_rr < 1.6 and len(sr_targets) >= 2:
                second_sr_p, second_sr_lbl = sr_targets[1]
                take_profit = round(first_sr_p, 2)
                tp2_price = round(second_sr_p, 2)
                tp_basis_desc = f"第一阻力: {first_sr_lbl} ({first_sr_p:.2f}) ｜ 次級目標: {second_sr_lbl} ({second_sr_p:.2f})"
            else:
                take_profit = round(first_sr_p, 2)
                tp2_price = round(sr_targets[1][0], 2) if len(sr_targets) >= 2 else measured_move_tp
                tp_basis_desc = f"{first_sr_lbl} ({take_profit:.2f})"
            structural_tp = round(first_sr_p, 2)
        else:
            take_profit = measured_move_tp
            tp2_price = round(recommended_entry + recent_swing_amp * 1.35, 2)
            structural_tp = take_profit
            tp_basis_desc = f"創高無套牢阻力，按前波段等距通道投射 ({take_profit:.2f})"

        structural_rr = round((structural_tp - recommended_entry) / risk_per_share, 2)
        planned_rr = round((take_profit - recommended_entry) / risk_per_share, 2)
        tp2_rr = round((tp2_price - recommended_entry) / risk_per_share, 2)

        warnings = []
        if structural_rr < 1.8:
            warnings.append(f"⚠️ 上方 {tp_basis_desc} 距離買入位僅 {structural_rr:.2f}R (<1.8R)，獲利空間受阻，不建議優先追入")
        if close_p > recommended_entry * 1.025:
            warnings.append(f"⏳ 現價 ({close_p:.2f}) 高於最佳買入位 ({recommended_entry:.2f})，請耐心掛單等待回測該買入位，切勿市價高追")

        return {
            "entry": recommended_entry,
            "entry_type_desc": entry_type_desc,
            "sl": stop_loss,
            "sl_basis_desc": sl_basis_desc,
            "tp": take_profit,
            "tp2": tp2_price,
            "tp_basis_desc": tp_basis_desc,
            "structural_tp": structural_tp,
            "risk_per_share": risk_per_share,
            "stop_pct": round(stop_pct, 2),
            "structural_rr": structural_rr,
            "planned_rr": planned_rr,
            "tp2_rr": tp2_rr,
            "rr_warning": " ｜ ".join(warnings)
        }

    elif plan_type == "SHORT":
        if is_breakdown_short or "破底" in status:
            pivot_short = box_low_10 - 0.08 * atr_safe
            if close_p < box_low_10 * 0.97:
                retest_short_lvl = min(box_low_10, sma10_p, prior_wave_low if prior_wave_low >= close_p else sma10_p)
                recommended_entry = round(retest_short_lvl, 2)
                entry_type_desc = f"等反彈沽空單 (現價已急跌偏離破底位，等無量反抽前低阻力 {recommended_entry:.2f} 沽空)"
            else:
                recommended_entry = round(min(close_p * 0.998, max(pivot_short, close_p - 0.35 * atr_safe)), 2)
                entry_type_desc = f"破底觸發沽空單 (Stop-Sell 跌穿箱體下沿 {recommended_entry:.2f} 追空)"
        else:
            resist_anchor = min(max(close_p, sma20_p * 0.995), sma50_p if pd.notna(sma50_p) and sma50_p > close_p else sma20_p * 1.01)
            recommended_entry = round(resist_anchor, 2)
            entry_type_desc = f"逢高反彈沽空單 (於 20MA/前支撐轉阻力區 {recommended_entry:.2f} 掛單沽空)"

        struct_high_candidates = [
            (box_high_10 + 0.15 * atr_safe, f"近10日反彈高點 ({box_high_10:.2f}) 上方"),
            (res_short + 0.20 * atr_safe, f"20日前浪頂阻力 ({res_short:.2f}) 外側"),
            (sma20_p + 0.50 * atr_safe, f"20MA 蓋頭均線 ({sma20_p:.2f}) 上方緩衝"),
            (sma50_p + 0.30 * atr_safe, f"50MA 生命線 ({sma50_p:.2f}) 上方")
        ]
        if lei_struct and pd.notna(lei_struct.get("structure_high")):
            sh_st = float(lei_struct["structure_high"]) + 0.15 * atr_safe
            struct_high_candidates.append((sh_st, f"LEI 頂部構造高點 ({lei_struct['structure_high']:.2f}) 上方"))

        valid_sls = [
            (sl_p, sl_lbl) for sl_p, sl_lbl in struct_high_candidates
            if pd.notna(sl_p) and (recommended_entry * 1.014 <= sl_p <= recommended_entry * 1.075)
        ]
        if valid_sls:
            best_sl_pair = min(valid_sls, key=lambda x: x[0] if (x[0] - recommended_entry) >= 0.85 * atr_safe else 999999)
            if (best_sl_pair[0] - recommended_entry) < 0.85 * atr_safe:
                best_sl_pair = max(valid_sls, key=lambda x: x[0])
            stop_loss = round(best_sl_pair[0], 2)
            sl_basis_desc = best_sl_pair[1]
        else:
            fallback_sl = min(recommended_entry + 1.8 * atr_safe, recommended_entry * 1.072)
            stop_loss = round(max(fallback_sl, recommended_entry * 1.018), 2)
            sl_basis_desc = f"波動率與 J Law ≤7.2% 空頭硬止損保護位"

        risk_per_share = max(stop_loss - recommended_entry, 1e-4)
        stop_pct = (risk_per_share / recommended_entry) * 100.0

        sr_targets = _find_historical_ma_convergence_targets(df, recommended_entry, "SHORT")
        recent_swing_amp = max(res_short - sup_short, 3.2 * atr_safe)
        measured_move_tp = round(max(0.01, recommended_entry - recent_swing_amp * 0.85), 2)

        if sr_targets:
            first_sr_p, first_sr_lbl = sr_targets[0]
            first_sr_rr = (recommended_entry - first_sr_p) / risk_per_share
            if first_sr_rr < 1.6 and len(sr_targets) >= 2:
                second_sr_p, second_sr_lbl = sr_targets[1]
                take_profit = round(first_sr_p, 2)
                tp2_price = round(second_sr_p, 2)
                tp_basis_desc = f"第一支撐: {first_sr_lbl} ({first_sr_p:.2f}) ｜ 次級目標: {second_sr_lbl} ({second_sr_p:.2f})"
            else:
                take_profit = round(first_sr_p, 2)
                tp2_price = round(sr_targets[1][0], 2) if len(sr_targets) >= 2 else measured_move_tp
                tp_basis_desc = f"{first_sr_lbl} ({take_profit:.2f})"
            structural_tp = round(first_sr_p, 2)
        else:
            take_profit = measured_move_tp
            tp2_price = round(max(0.01, recommended_entry - recent_swing_amp * 1.35), 2)
            structural_tp = take_profit
            tp_basis_desc = f"破底無歷史支撐，按波段等距下跌通道投射 ({take_profit:.2f})"

        structural_rr = round((recommended_entry - structural_tp) / risk_per_share, 2)
        planned_rr = round((recommended_entry - take_profit) / risk_per_share, 2)
        tp2_rr = round((recommended_entry - tp2_price) / risk_per_share, 2)

        warnings = []
        if structural_rr < 1.8:
            warnings.append(f"⚠️ 下方 {tp_basis_desc} 距離沽空位僅 {structural_rr:.2f}R (<1.8R)，下跌空間受限，不建議優先追空")
        if close_p < recommended_entry * 0.975:
            warnings.append(f"⏳ 現價 ({close_p:.2f}) 低於理想沽空位 ({recommended_entry:.2f})，請耐心等待反彈至阻力位再沽空")

        return {
            "entry": recommended_entry,
            "entry_type_desc": entry_type_desc,
            "sl": stop_loss,
            "sl_basis_desc": sl_basis_desc,
            "tp": take_profit,
            "tp2": tp2_price,
            "tp_basis_desc": tp_basis_desc,
            "structural_tp": structural_tp,
            "risk_per_share": risk_per_share,
            "stop_pct": round(stop_pct, 2),
            "structural_rr": structural_rr,
            "planned_rr": planned_rr,
            "tp2_rr": tp2_rr,
            "rr_warning": " ｜ ".join(warnings)
        }

    else:
        return {
            "entry": close_p,
            "entry_type_desc": "觀望狀態 (暫無掛單建議)",
            "sl": round(close_p - 2.0 * atr_safe, 2),
            "sl_basis_desc": "-",
            "tp": round(close_p + 4.0 * atr_safe, 2),
            "tp2": round(close_p + 6.0 * atr_safe, 2),
            "tp_basis_desc": "-",
            "structural_tp": res_short,
            "risk_per_share": 2.0 * atr_safe,
            "stop_pct": round((2.0 * atr_safe / close_p) * 100.0, 2),
            "structural_rr": 0.0,
            "planned_rr": 0.0,
            "tp2_rr": 0.0,
            "rr_warning": ""
        }

def evaluate_jlaw_power_trend_and_meta(
    df: pd.DataFrame,
    bg_df: Optional[pd.DataFrame],
    market_type: str,
    lei_struct: Dict[str, Any],
    lei_ded: Dict[str, Any],
    trade_plan_eval: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [J Law @jlawstock《EP1：強勁趨勢與多重優勢進場點 META》&《EP2：趨勢線與回測》]
    """
    meta_res = {
        "is_power_trend": False,
        "rs_1m_outperform": False,
        "rs_6m_outperform": False,
        "triple_liquidity_ok": False,
        "up_down_vol_ratio": 1.0,
        "sr_flip_retest_long": False,
        "sr_flip_retest_short": False,
        "meta_long_score": 0,
        "meta_short_score": 0,
        "meta_edges_list": [],
        "meta_badge": "-",
        "priority_score_100": 0.0
    }
    if df is None or len(df) < 60:
        return meta_res

    last = df.iloc[-1]
    close_p = float(last['Close'])

    recent20 = df.tail(20)
    up_bars = recent20[recent20['Close'] > recent20['Open']]
    down_bars = recent20[recent20['Close'] < recent20['Open']]
    avg_up_v = float(up_bars['Volume'].mean()) if not up_bars.empty else 1.0
    avg_dn_v = float(down_bars['Volume'].mean()) if not down_bars.empty else 1.0
    ud_vol_ratio = round(avg_up_v / max(avg_dn_v, 1.0), 2)
    meta_res["up_down_vol_ratio"] = ud_vol_ratio

    above_200 = pd.notna(last.get('SMA200')) and close_p > float(last['SMA200'])
    sma10_gt_20 = pd.notna(last.get('SMA10')) and pd.notna(last.get('SMA20')) and float(last['SMA10']) > float(last['SMA20'])
    held_20ma_recently = int((df['Close'].tail(10) >= df['SMA20'].tail(10) * 0.99).sum()) >= 8
    meta_res["is_power_trend"] = bool(above_200 and sma10_gt_20 and held_20ma_recently and ud_vol_ratio >= 1.2)

    stock_ret1m = float(last.get('Ret1M', 0.0)) if pd.notna(last.get('Ret1M')) else 0.0
    stock_ret6m = float(last.get('Ret6M', 0.0)) if pd.notna(last.get('Ret6M')) else 0.0
    bench_ret1m, bench_ret6m = 0.0, 0.0
    if bg_df is not None and len(bg_df) > 126:
        bench_ret1m = float(bg_df['Close'].iloc[-1] / bg_df['Close'].iloc[-21] - 1.0)
        bench_ret6m = float(bg_df['Close'].iloc[-1] / bg_df['Close'].iloc[-126] - 1.0)
    meta_res["rs_1m_outperform"] = bool(stock_ret1m > max(0.05, bench_ret1m * 1.8))
    meta_res["rs_6m_outperform"] = bool(stock_ret6m > max(0.10, bench_ret6m * 1.8))

    if market_type == "HK":
        meta_res["triple_liquidity_ok"] = bool(float(last.get('TurnoverMA20', 0)) >= 20_000_000)
    else:
        v_ok = all(float(last.get(c, 0)) >= 500_000 for c in ['VolSMA10', 'VolSMA30', 'VolSMA60'] if pd.notna(last.get(c)))
        turnover_ok = float(last.get('TurnoverMA20', 0)) >= 30_000_000
        meta_res["triple_liquidity_ok"] = bool(v_ok and turnover_ok)

    if len(df) >= 45:
        prior_box_high = float(df['High'].iloc[-45:-10].max())
        prior_box_low = float(df['Low'].iloc[-45:-10].min())
        if float(df['High'].iloc[-10:].max()) > prior_box_high * 1.01 and (prior_box_high * 0.975 <= close_p <= prior_box_high * 1.035):
            meta_res["sr_flip_retest_long"] = True
        if float(df['Low'].iloc[-10:].min()) < prior_box_low * 0.99 and (prior_box_low * 0.965 <= close_p <= prior_box_low * 1.025):
            meta_res["sr_flip_retest_short"] = True

    long_edges = []
    if meta_res["is_power_trend"] or (meta_res["rs_1m_outperform"] and meta_res["rs_6m_outperform"]):
        long_edges.append("Edge1:強勁趨勢/雙週期RS領先")
    if meta_res["sr_flip_retest_long"] or bool(last.get('JLaw_Tight_Box', False)) or bool(last.get('LEI_MA_Tight_2Pct', False)):
        long_edges.append("Edge2:S/R互換支撐區或<5%緊密平台")
    dist_10 = abs(close_p - float(last.get('SMA10', 0))) / (close_p + 1e-9)
    dist_20 = abs(close_p - float(last.get('SMA20', 0))) / (close_p + 1e-9)
    if (dist_10 <= 0.025 or dist_20 <= 0.03) and lei_ded.get("is_20ma_turn_up_favorable", False):
        long_edges.append("Edge3:10/20MA均線保護+抵扣價助攻")
    if lei_struct.get("bottom_structure", False) or lei_struct.get("bullish_2b", False) or (bool(last.get('VCPSignal', False)) and float(last.get('VolRatio', 1.0)) < 0.85):
        long_edges.append("Edge4:底構造/2B破底翻/縮量洗盤確認")
    sma150_up = len(df) > 170 and pd.notna(last.get('SMA150')) and float(last['SMA150']) > float(df['SMA150'].iloc[-20])
    if sma150_up and float(trade_plan_eval.get("structural_rr", 0.0)) >= 1.8:
        long_edges.append("Edge5:30週線(Stage2)向上且結構空間充足")

    short_edges = []
    if (not above_200 or float(last.get('SMA20', 0)) < float(last.get('SMA50', 0))) and float(last.get('Mansfield_RS', 0)) < 0:
        short_edges.append("Edge1:空頭趨勢且RS弱於大盤")
    if meta_res["sr_flip_retest_short"] or bool(last.get('JLaw_Tight_Box', False)):
        short_edges.append("Edge2:支撐Flip轉阻力回測/弱勢熊旗")
    if dist_20 <= 0.03 and not lei_ded.get("is_20ma_turn_up_favorable", True):
        short_edges.append("Edge3:20MA蓋頭壓制+抵扣價下壓")
    if lei_struct.get("top_structure", False) or lei_struct.get("bearish_2b", False) or ud_vol_ratio < 0.85:
        short_edges.append("Edge4:頂構造/2B假突破/下跌放量反彈縮量")
    if float(trade_plan_eval.get("structural_rr", 0.0)) >= 1.8:
        short_edges.append("Edge5:下方至前低支撐空間充足")

    meta_res["meta_long_score"] = len(long_edges)
    meta_res["meta_short_score"] = len(short_edges)
    if len(long_edges) >= len(short_edges):
        meta_res["meta_edges_list"] = long_edges
        meta_res["meta_badge"] = f"💎 {len(long_edges)}/5 多重優勢" if len(long_edges) >= 3 else f"⚪ {len(long_edges)}/5 優勢"
    else:
        meta_res["meta_edges_list"] = short_edges
        meta_res["meta_badge"] = f"🏴‍☠️ {len(short_edges)}/5 空頭優勢" if len(short_edges) >= 3 else f"⚪ {len(short_edges)}/5 優勢"

    return meta_res

def check_recent_signal_and_stop_invalidation(df: pd.DataFrame, lookback_bars: int = 15) -> Dict[str, Any]:
    """解決「股票已大幅下跌仍顯示優先做多」及「早已觸及當時買入止損仍繼續顯示做多」的核心狀態檢查器。"""
    result = {
        "long_stop_hit": False,
        "short_stop_hit": False,
        "downtrend_veto_long": False,
        "uptrend_veto_short": False,
        "signal_age_bars": 0,
        "anchor_date": None,
        "anchor_entry": np.nan,
        "anchor_sl": np.nan,
        "invalidation_reason": ""
    }
    if df is None or len(df) < 25:
        return result

    last = df.iloc[-1]
    prev5 = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
    window = df.iloc[-lookback_bars:]

    sma20_falling = pd.notna(last.get('SMA20')) and pd.notna(prev5.get('SMA20')) and (last['SMA20'] < prev5['SMA20'])
    below_sma20_and_50 = pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and (last['Close'] < last['SMA20']) and (last['Close'] < last['SMA50'])
    macd_bearish = pd.notna(last.get('MACD')) and pd.notna(last.get('MACDSignal')) and (last['MACD'] < last['MACDSignal']) and (last['MACD'] < 0)
    low_20d_prev = df['Low'].shift(1).rolling(20).min().iloc[-1]
    broke_20d_low_intraday = pd.notna(low_20d_prev) and (last['Low'] <= low_20d_prev)

    peak_idx = window['Close'].idxmax()
    peak_pos = df.index.get_loc(peak_idx)
    if isinstance(peak_pos, slice) or isinstance(peak_pos, np.ndarray):
        peak_pos = int(np.flatnonzero(peak_pos)[-1])
    bars_since_peak = (len(df) - 1) - peak_pos
    peak_row = df.iloc[peak_pos]
    peak_entry = float(peak_row['Close'])
    peak_atr = float(peak_row['ATR']) if pd.notna(peak_row.get('ATR')) and peak_row['ATR'] > 0 else peak_entry * 0.03
    peak_initial_sl = max(peak_entry - 2.0 * peak_atr, peak_entry * 0.925)

    post_peak_df = df.iloc[peak_pos + 1:] if peak_pos + 1 < len(df) else pd.DataFrame()
    min_low_since_peak = float(post_peak_df['Low'].min()) if not post_peak_df.empty else float(last['Low'])

    result["signal_age_bars"] = int(bars_since_peak)
    result["anchor_date"] = str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx)
    result["anchor_entry"] = peak_entry
    result["anchor_sl"] = peak_initial_sl

    if bars_since_peak >= 1 and min_low_since_peak <= peak_initial_sl and (last['Close'] < last.get('SMA20', np.inf) or last['Close'] < peak_entry):
        result["long_stop_hit"] = True
        result["invalidation_reason"] = (
            f"⛔ 觸及前波止損失效：自 {result['anchor_date']} 波段高點 ({peak_entry:.2f}) 起算，"
            f"盤中最低價 ({min_low_since_peak:.2f}) 已跌穿當時止損位 ({peak_initial_sl:.2f})"
        )

    if below_sma20_and_50 or (last['Close'] < last.get('SMA20', 0) and sma20_falling and macd_bearish) or broke_20d_low_intraday:
        result["downtrend_veto_long"] = True
        if not result["invalidation_reason"]:
            result["invalidation_reason"] = "📉 結構轉空否決做多：股價已跌破 20MA/50MA 或創 20 日新低，短期均線下彎"

    trough_idx = window['Close'].idxmin()
    trough_pos = df.index.get_loc(trough_idx)
    if isinstance(trough_pos, slice) or isinstance(trough_pos, np.ndarray):
        trough_pos = int(np.flatnonzero(trough_pos)[-1])
    bars_since_trough = (len(df) - 1) - trough_pos
    trough_row = df.iloc[trough_pos]
    trough_entry = float(trough_row['Close'])
    trough_atr = float(trough_row['ATR']) if pd.notna(trough_row.get('ATR')) and trough_row['ATR'] > 0 else trough_entry * 0.03
    trough_initial_sl = min(trough_entry + 2.0 * trough_atr, trough_entry * 1.075)
    post_trough_df = df.iloc[trough_pos + 1:] if trough_pos + 1 < len(df) else pd.DataFrame()
    max_high_since_trough = float(post_trough_df['High'].max()) if not post_trough_df.empty else float(last['High'])

    if bars_since_trough >= 1 and max_high_since_trough >= trough_initial_sl and (last['Close'] > last.get('SMA20', 0)):
        result["short_stop_hit"] = True

    if pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and (last['Close'] > last['SMA20']) and (last['Close'] > last['SMA50']) and not sma20_falling:
        result["uptrend_veto_short"] = True

    return result

def compute_backtest_institutional_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    """計算最大回撤 (MDD)、獲利因子 (Profit Factor)、夏普比率 (Sharpe)、期望值 (Expectancy)"""
    metrics = {
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "expectancy_pct": 0.0,
        "payoff_ratio": 0.0
    }
    try:
        if equity_curve is not None and not equity_curve.empty:
            roll_max = equity_curve.cummax()
            drawdown = (equity_curve - roll_max) / (roll_max + 1e-9) * 100.0
            metrics["max_drawdown_pct"] = float(drawdown.min())
            daily_rets = equity_curve.pct_change().dropna()
            if len(daily_rets) > 5 and daily_rets.std() > 0:
                metrics["sharpe_ratio"] = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))

        if trades_df is not None and not trades_df.empty and '利潤' in trades_df.columns:
            gross_profit = float(trades_df[trades_df['利潤'] > 0]['利潤'].sum())
            gross_loss = abs(float(trades_df[trades_df['利潤'] < 0]['利潤'].sum()))
            metrics["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

            wins_ret = trades_df[trades_df['回報率(%)'] > 0]['回報率(%)']
            loss_ret = trades_df[trades_df['回報率(%)'] <= 0]['回報率(%)']
            avg_win = float(wins_ret.mean()) if not wins_ret.empty else 0.0
            avg_loss = abs(float(loss_ret.mean())) if not loss_ret.empty else 0.0
            win_rate = len(wins_ret) / len(trades_df)
            metrics["payoff_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
            metrics["expectancy_pct"] = round((win_rate * avg_win) - ((1.0 - win_rate) * avg_loss), 2)
    except Exception as e:
        swing_logger.warning(f"回測指標計算異常: {e}")
    return metrics

st.set_page_config(layout="wide", page_title="專業量化交易終端系統 v24.10 (長短空頭融合版)")

# ==========================================
# 0. 伺服器級持久化快取 (徹底解決雷達刷新消失的問題)
# ==========================================
@st.cache_resource
def get_persistent_data():
    """使用 cache_resource 保證數據在 F5 重新整理或點擊 LinkColumn 後絕對不會消失"""
    return {
        "radar_results": {"港股": None, "美股": None, "自選股": None, "自訂": None},
        "radar_correlation_warnings": {"港股": [], "美股": [], "自選股": [], "自訂": []},
        "radar_correlation_matrices": {"港股": None, "美股": None, "自選股": None, "自訂": None},
        # OLD: "watchlist_radar_results": {} # 🌟 升級為字典，儲存每個分組的獨立掃描結果
        # NEW: 保留 watchlist_radar_results 並增量加入時間戳與信號狀態追蹤池
        "watchlist_radar_results": {}, # 🌟 升級為字典，儲存每個分組的獨立掃描結果
        "radar_timestamps": {},        # 🕒 [NEW] 紀錄各池最後掃描時間，解決緩存永不失效問題
        "signal_lifecycle_store": {},  # 🛡️ [NEW] 紀錄信號初始買入價與初始止損，解決觸及止損仍顯示問題
        "radar_breadth_stats": {},     # 🌐 [NEW J Law 影片10] 紀錄掃描池 >50MA 及 >200MA 市場寬度百分比
        "_schema_ver": CURRENT_SCHEMA_VERSION
    }

persistent_data = get_persistent_data()
ensure_persistent_schema_fresh(persistent_data)

# ==========================================
# 🌟 初始化 Session State (仿富途牛牛分組)
# ==========================================
if "watchlist_dict" not in st.session_state:
    st.session_state.watchlist_dict = {
        "🌟 我的最愛": ["0700.HK", "3690.HK", "1810.HK", "9988.HK", "NVDA", "PLTR"],
        "🔥 科技巨頭": ["AAPL", "MSFT", "GOOGL", "META", "TSLA"],
        "💰 收息板塊": ["0005.HK", "0941.HK", "1398.HK", "0883.HK"]
    }

# 將雷達結果與持久化字典綁定
st.session_state.radar_results = persistent_data["radar_results"]
st.session_state.radar_correlation_warnings = persistent_data["radar_correlation_warnings"]
st.session_state.radar_correlation_matrices = persistent_data["radar_correlation_matrices"]
if "watchlist_radar_results" not in st.session_state:
    st.session_state.watchlist_radar_results = persistent_data["watchlist_radar_results"]

# 讀取網址路由參數
qp = st.query_params
default_mode_val = qp.get("mode", "")
target_ticker = qp.get("ticker", "0700.HK")
# [NEW INCREMENTAL PATCH 修復 Bug 1C] 讀取跳轉時攜帶的雷達方向與嚴格度參數，防止跳轉後因側欄預設值不同而變換多空方向
target_dir_from_radar = qp.get("dir", "")
target_strict_from_radar = qp.get("strict", "")

# ==========================================
# 1. 智能代碼格式化與市場識別
# ==========================================
def format_and_detect_market(ticker):
    ticker = ticker.strip().upper()
    if ticker.isdigit():
        ticker = f"{int(ticker):04d}.HK"
    elif len(ticker) <= 4 and ticker.replace('.', '').isdigit():
        ticker = f"{int(float(ticker)):04d}.HK"

    if ticker.endswith(".HK"):
        return ticker, "HK", "^HSI", "恆生指數", "HKD"
    else:
        return ticker, "US/Global", "^GSPC", "標普 500 指數", "USD"

# ==========================================
# 2. 股票名稱與大盤數據
# ==========================================
@st.cache_data(ttl=86400)
def get_stock_name(ticker):
    hk_names = {
        "0700.HK": "騰訊控股", "3690.HK": "美團-W", "1810.HK": "小米集團-W", "1211.HK": "比亞迪股份",
        "9988.HK": "阿里巴巴-W", "0981.HK": "中芯國際", "0388.HK": "香港交易所", "0005.HK": "匯豐控股",
        "0941.HK": "中國移動", "0883.HK": "中國海洋石油", "0939.HK": "建設銀行", "1398.HK": "工商銀行",
        "2318.HK": "中國平安", "1299.HK": "友邦保險", "1113.HK": "長實集團", "0027.HK": "銀河娛樂",
        "2015.HK": "理想汽車-W", "9868.HK": "小鵬汽車-W", "2269.HK": "藥明生物", "1024.HK": "快手-W",
        "9618.HK": "京東集團-SW", "9999.HK": "網易-S", "9961.HK": "攜程集團-S", "6618.HK": "京東健康",
        "2020.HK": "安踏體育", "2331.HK": "李寧", "0268.HK": "金蝶國際", "0285.HK": "比亞迪電子",
        "6690.HK": "海爾智家", "3328.HK": "交通銀行", "3988.HK": "中國銀行", "0853.HK": "微創醫療",
        "1093.HK": "石藥集團", "1177.HK": "中國生物製藥", "2359.HK": "藥明康德", "1833.HK": "平安好醫生",
        "0011.HK": "恒生銀行", "0002.HK": "中電控股", "0016.HK": "新鴻基地產", "0066.HK": "港鐵公司",
        "1088.HK": "中國神華", "0322.HK": "康師傅控股", "0772.HK": "閱文集團", "1928.HK": "金沙中國有限公司",
        "2382.HK": "舜宇光學科技", "3606.HK": "福耀玻璃", "9922.HK": "九毛九", "9992.HK": "泡泡瑪特",
        "6862.HK": "海底撈", "0293.HK": "國泰航空", "1044.HK": "恒安國際", "1919.HK": "中遠海控",
        "2007.HK": "碧桂園", "2313.HK": "申洲國際", "2333.HK": "長城汽車", "1060.HK": "阿里影業",
        "1099.HK": "國藥控股", "1193.HK": "華潤燃氣", "1316.HK": "耐世特", "1347.HK": "華虹半導體",
        "1801.HK": "信達生物", "1818.HK": "招金礦業", "1898.HK": "中煤能源", "1929.HK": "周大福",
        "2018.HK": "瑞聲科技", "2192.HK": "醫渡科技", "2202.HK": "萬科企業", "2338.HK": "濰柴動力",
        "2899.HK": "紫金礦業", "3311.HK": "中國建築國際", "3333.HK": "中國恒大", "3908.HK": "中金公司",
        "6030.HK": "中信証券", "6110.HK": "滔搏"
    }
    
    if ticker in hk_names:
        return hk_names[ticker]
        
    try:
        t = yf.Ticker(ticker)
        name = t.info.get('shortName', t.info.get('longName', ticker))
        return name if name else ticker
    except:
        return ticker

@st.cache_data(ttl=3600)
def get_index_data(index_ticker, interval="1d"):
    try:
        if interval == "1m":
            period = "7d"
        elif interval in ["5m", "15m", "30m"]:
            period = "60d"
        elif interval in ["1h", "90m"]:
            period = "730d"
        else:
            period = "max"

        idx_df = yf.download(index_ticker, period=period, interval=interval, progress=False)
        if idx_df.empty:
            return None

        if isinstance(idx_df.columns, pd.MultiIndex):
            if 'Close' in idx_df.columns.get_level_values(0):
                idx_df.columns = idx_df.columns.get_level_values(0)
            elif 'Close' in idx_df.columns.get_level_values(1):
                idx_df.columns = idx_df.columns.get_level_values(1)
            else:
                idx_df.columns = idx_df.columns.get_level_values(0)

        idx_df = idx_df.loc[:, ~idx_df.columns.duplicated()]
        idx_df = idx_df.dropna(subset=['Close']) 
        
        if idx_df.index.tz is not None:
            idx_df.index = idx_df.index.tz_localize(None)

        idx_df['SMA50'] = idx_df['Close'].rolling(window=50).mean()
        idx_df['SMA200'] = idx_df['Close'].rolling(window=200).mean()
        
        idx_df['PriceChange'] = idx_df['Close'].pct_change()
        if 'Volume' in idx_df.columns:
            idx_df['VolumeSMA50'] = idx_df['Volume'].rolling(window=50).mean()
            # OLD: idx_df['Is_Distribution'] = (idx_df['PriceChange'] <= -0.002) & (idx_df['Volume'] > idx_df['Volume'].shift(1))
            # NEW: [J Law 影片10 15:10] 排除 3/6/9/12 月第 3 個星期五期貨結算日 (四巫日) 異常成交量，並涵蓋放量滯漲出貨日
            _qw_mask_idx = is_quad_witching_friday(idx_df.index)
            idx_df['Is_Distribution'] = (
                (idx_df['PriceChange'] <= 0.002)
                & (idx_df['Volume'] > idx_df['Volume'].shift(1))
                & (idx_df['Volume'] > idx_df['VolumeSMA50'])
                & (~_qw_mask_idx)
            )
            idx_df['Dist_Days_20d'] = idx_df['Is_Distribution'].rolling(window=20).sum()
        else:
            idx_df['Dist_Days_20d'] = 0
            
        return idx_df
    except:
        return None

# ==========================================
# 3. 高動能股票池 & 港股熱門手數
# ==========================================
HK_MOMENTUM_POOL = [
    "0700.HK", "3690.HK", "1810.HK", "1211.HK", "2015.HK", "9868.HK", "9988.HK", "0981.HK", "2269.HK", "2317.HK",
    "0388.HK", "1024.HK", "1833.HK", "1789.HK", "2500.HK", "6098.HK", "2669.HK", "3319.HK", "1995.HK", "0005.HK",
    "1299.HK", "0939.HK", "1398.HK", "2318.HK", "0883.HK", "0941.HK", "1113.HK", "0027.HK", "9618.HK", "9999.HK",
    "1088.HK", "0268.HK", "0285.HK", "0322.HK", "0772.HK", "0853.HK", "1093.HK", "1177.HK", "1928.HK", "2020.HK",
    "2331.HK", "2359.HK", "2382.HK", "3328.HK", "3606.HK", "6618.HK", "6690.HK", "9922.HK", "9992.HK", "9961.HK",
    "6862.HK", "0011.HK", "0002.HK", "0016.HK", "0293.HK", "1044.HK", "1919.HK", "2007.HK", "2313.HK", "2333.HK",
    "1060.HK", "1099.HK", "1193.HK", "1316.HK", "1347.HK", "1801.HK", "1818.HK", "1898.HK", "1929.HK", "2018.HK",
    "2192.HK", "2202.HK", "2338.HK", "2899.HK", "3311.HK", "3333.HK", "3908.HK", "3988.HK", "6030.HK", "6110.HK"
]

US_MOMENTUM_POOL = [
    "NVDA", "AAPL", "TSLA", "PLTR", "MSFT", "AMZN", "META", "GOOGL", "AMD", "MSTR",
    "COIN", "SMCI", "AVGO", "CRWD", "NFLX", "ADBE", "SNOW", "DDOG", "ZS", "PANW",
    "ARM", "ASML", "TSM", "INTC", "QCOM", "MU", "UBER", "ABNB", "PYPL", "SQ",
    "ROKU", "SHOP", "PTON", "DOCU", "ZM", "INTU", "ORCL", "IBM", "CSCO", "CRM"
]

POPULAR_HK_LOTS = {
    "0700.HK": 100, "3690.HK": 100, "9988.HK": 100, "1810.HK": 200, "1211.HK": 500,
    "1299.HK": 200, "0005.HK": 400, "0388.HK": 100, "0941.HK": 500, "0883.HK": 1000,
    "0981.HK": 500, "0939.HK": 1000, "1398.HK": 1000, "2318.HK": 500, "2020.HK": 100,
    "2269.HK": 100, "2317.HK": 500, "0853.HK": 500, "1024.HK": 100, "1113.HK": 500,
    "0011.HK": 100, "0002.HK": 500, "0016.HK": 1000, "0066.HK": 500
}

# ==========================================
# 4.0.1 Anchored VWAP 錨定成交量加權平均價
# ==========================================
def calculate_anchored_vwap(df, anchor_index):
    avwap = pd.Series(np.nan, index=df.index)
    try:
        if anchor_index not in df.index:
            return avwap

        sub_df = df.loc[anchor_index:].copy()
        typical_price = (sub_df['High'] + sub_df['Low'] + sub_df['Close']) / 3
        cum_vol = sub_df['Volume'].cumsum()
        cum_tpv = (typical_price * sub_df['Volume']).cumsum()
        avwap_values = cum_tpv / (cum_vol + 1e-9)
        avwap.loc[sub_df.index] = avwap_values
    except Exception:
        pass
    return avwap

def find_anchor_points(df, lookback_swing=60, lookback_52w=252):
    swing_anchor = None
    low52w_anchor = None
    try:
        recent_window = df.tail(lookback_swing)
        if not recent_window.empty:
            swing_anchor = recent_window['Low'].idxmin()

        w52_window = df.tail(lookback_52w)
        if not w52_window.empty:
            low52w_anchor = w52_window['Low'].idxmin()
    except Exception:
        pass
    return swing_anchor, low52w_anchor

# ==========================================
# 4. 指標計算引擎 
# ==========================================
def calculate_indicators(df, bg_df, interval="1d"):
    # MA 系統 (抵扣價依賴)
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA120'] = df['Close'].rolling(window=120).mean() 
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    # EMA 系統 (所見即所得，敏感度高)
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA120'] = df['Close'].ewm(span=120, adjust=False).mean()

    # 均線抵扣價計算 (Deduction Price)
    df['SMA20_Deduction'] = df['Close'].shift(20)
    df['SMA50_Deduction'] = df['Close'].shift(50)
    df['SMA120_Deduction'] = df['Close'].shift(120)
    df['SMA200_Deduction'] = df['Close'].shift(200)

    df['VolumeSMA50'] = df['Volume'].rolling(window=50).mean()
    df['Ret1M'] = df['Close'] / df['Close'].shift(20) - 1

    if HAS_SCIPY:
        peaks, _ = find_peaks(df['High'].values, distance=20)
        valleys, _ = find_peaks(-df['Low'].values, distance=20)
        swing_high = pd.Series(np.nan, index=df.index)
        swing_low = pd.Series(np.nan, index=df.index)
        if len(peaks) > 0:
            swing_high.iloc[peaks] = df['High'].iloc[peaks]
        if len(valleys) > 0:
            swing_low.iloc[valleys] = df['Low'].iloc[valleys]
        df['ResShort'] = swing_high.ffill().fillna(df['High'].rolling(20).max())
        df['SupShort'] = swing_low.ffill().fillna(df['Low'].rolling(20).min())
        
        peaks_long, _ = find_peaks(df['High'].values, distance=60)
        valleys_long, _ = find_peaks(-df['Low'].values, distance=60)
        swing_high_long = pd.Series(np.nan, index=df.index)
        swing_low_long = pd.Series(np.nan, index=df.index)
        if len(peaks_long) > 0:
            swing_high_long.iloc[peaks_long] = df['High'].iloc[peaks_long]
        if len(valleys_long) > 0:
            swing_low_long.iloc[valleys_long] = df['Low'].iloc[valleys_long]
        df['ResLong'] = swing_high_long.ffill().fillna(df['High'].rolling(60).max())
        df['SupLong'] = swing_low_long.ffill().fillna(df['Low'].rolling(60).min())
    else:
        df['SupShort'] = df['Low'].rolling(window=20).min()
        df['ResShort'] = df['High'].rolling(window=20).max()
        df['SupLong'] = df['Low'].rolling(window=60).min()
        df['ResLong'] = df['High'].rolling(window=60).max()

    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    df['VolRatio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    df['Turnover'] = df['Close'] * df['Volume']

    df['TurnoverMA20'] = df['Turnover'].rolling(window=20).mean()
    df['BigMoney_In'] = (df['Volume'] > df['VolumeSMA50'] * 2.0) & (df['Close'] > df['Open'])
    df['BigMoney_Out'] = (df['Volume'] > df['VolumeSMA50'] * 2.0) & (df['Close'] < df['Open'])

    df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['TotalRange'] = df['High'] - df['Low'] + 1e-9
    df['ShadowRatio'] = df['UpperShadow'] / df['TotalRange']
    df['LowerShadowRatio'] = df['LowerShadow'] / df['TotalRange']

    periods_per_year = 252 if interval == '1d' else (52 if interval == '1wk' else 12)
    df['High52W'] = df['High'].rolling(periods_per_year, min_periods=1).max()
    df['Low52W'] = df['Low'].rolling(periods_per_year, min_periods=1).min()
    df['Distto52WHigh'] = (df['High52W'] - df['Close']) / df['High52W'] * 100

    df['DailyPctRange'] = (df['High'] - df['Low']) / df['Close']
    df['VCPRange3'] = df['DailyPctRange'].rolling(3).mean()
    df['VCPRange20'] = df['DailyPctRange'].rolling(20).mean()
    df['VCPSignal'] = df['VCPRange3'] < (df['VCPRange20'] * 0.5)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    # 斜率時鐘判定與黏合度計算
    if len(df) >= 120:
        df['MA_Max'] = df[['SMA20', 'SMA50', 'SMA120']].max(axis=1)
        df['MA_Min'] = df[['SMA20', 'SMA50', 'SMA120']].min(axis=1)
        df['MA_Spread_Pct'] = (df['MA_Max'] - df['MA_Min']) / (df['MA_Min'] + 1e-9)
        df['Clock_3'] = df['MA_Spread_Pct'] < 0.03
        df['SMA50_Slope'] = (df['SMA50'] - df['SMA50'].shift(20)) / (df['SMA50'].shift(20) + 1e-9)
        df['Clock_12'] = (df['Close'] > df['SMA50'] * 1.25) & (df['SMA50_Slope'] > 0.15) & (df['RSI'].shift(1).rolling(5).max() > 75)
        df['Clock_2'] = (df['SMA20'] > df['SMA50']) & (df['SMA50'] > df['SMA120']) & (df['SMA50_Slope'] > 0) & (~df['Clock_12']) & (~df['Clock_3'])
    else:
        df['Clock_3'] = False
        df['Clock_12'] = False
        df['Clock_2'] = False
        df['MA_Spread_Pct'] = 999

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACDSignal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACDHist'] = df['MACD'] - df['MACDSignal']

    df['PriceChange'] = df['Close'].pct_change()
    # OLD:
    # df['Is_Distribution'] = (
    #     (df['PriceChange'] <= 0.002) & 
    #     (df['Volume'] > df['Volume'].shift(1)) & 
    #     (df['Volume'] > df['VolumeSMA50'])
    # )
    # NEW: [J Law 影片10] 排除四巫日結算量干擾
    _qw_mask_stk = is_quad_witching_friday(df.index)
    df['Is_Distribution'] = (
        (df['PriceChange'] <= 0.002) & 
        (df['Volume'] > df['Volume'].shift(1)) & 
        (df['Volume'] > df['VolumeSMA50']) &
        (~_qw_mask_stk)
    )
    df['Dist_Days_20d'] = df['Is_Distribution'].rolling(window=20).sum()
    df['Is_Abnormal_Drop'] = (df['PriceChange'] < -0.03) & (df['Volume'] > df['VolumeSMA50'])
    df['Is_Climax_Run'] = df['Clock_12']

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)
    pos_mf = pos_flow.rolling(14).sum()
    neg_mf = neg_flow.rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    if bg_df is not None and not bg_df.empty:
        cols_to_get = ['Close', 'SMA50']
        if 'SMA200' in bg_df.columns: cols_to_get.append('SMA200')
        if 'Dist_Days_20d' in bg_df.columns: cols_to_get.append('Dist_Days_20d')
        
        temp_merged = pd.merge(
            df[['Close']],
            bg_df[cols_to_get],
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', 'Bench')
        )
        
        temp_merged['CloseBench'] = temp_merged['CloseBench'].ffill()
        temp_merged['SMA50Bench'] = temp_merged.get('SMA50', pd.Series(np.nan, index=temp_merged.index)).ffill()
        
        rs_raw = temp_merged['Close'] / temp_merged['CloseBench']
        df['Mansfield_RS'] = ((rs_raw / rs_raw.rolling(50).mean()) - 1) * 10
        df['CloseBench'] = temp_merged['CloseBench']
        df['SMA50BenchValue'] = temp_merged['SMA50Bench']
        df['SMA200BenchValue'] = temp_merged.get('SMA200', pd.Series(np.nan, index=temp_merged.index)).ffill()
        df['Bench_Dist_Days'] = temp_merged.get('Dist_Days_20d', pd.Series(0, index=temp_merged.index)).ffill()
    else:
        df['Mansfield_RS'] = 0
        df['CloseBench'] = np.nan
        df['SMA50BenchValue'] = np.nan
        df['SMA200BenchValue'] = np.nan
        df['Bench_Dist_Days'] = 0

    try:
        swing_anchor, low52w_anchor = find_anchor_points(df)
        if swing_anchor is not None:
            df['AVWAP_Swing'] = calculate_anchored_vwap(df, swing_anchor)
        else:
            df['AVWAP_Swing'] = np.nan

        if low52w_anchor is not None:
            df['AVWAP_52WLow'] = calculate_anchored_vwap(df, low52w_anchor)
        else:
            df['AVWAP_52WLow'] = np.nan
    except Exception:
        df['AVWAP_Swing'] = np.nan
        df['AVWAP_52WLow'] = np.nan

    # [效能優化] 為快速計算準備 Cumulative 欄位
    df['CumulativeHigh'] = df['High'].cummax()
    df['CumulativeLow'] = df['Low'].cummin()

    # [NEW INCREMENTAL PATCH v24.14] 增量計算無未來函數之多層級結構支撐阻力、Wilder ATR/RSI 與 LEI × J Law 雙大師特徵欄位
    df = compute_causal_swing_sr(df)
    df['ATR_Wilder'], df['RSI_Wilder'] = compute_wilder_atr_and_rsi(df)
    df = compute_lei_dual_ma_and_ppo_features(df)

    return df

# ==========================================
# 4.1 大週期週線共振監測
# ==========================================
@st.cache_data(ttl=3600)
def get_weekly_alignment_signal(ticker):
    try:
        wk_df = yf.download(ticker, period="3y", interval="1wk", progress=False)
        if wk_df.empty:
            return {"warning": "", "label": "⚪ 無法取得週線資料", "detail": {}}

        if isinstance(wk_df.columns, pd.MultiIndex):
            if 'Close' in wk_df.columns.get_level_values(0):
                wk_df.columns = wk_df.columns.get_level_values(0)
            elif 'Close' in wk_df.columns.get_level_values(1):
                wk_df.columns = wk_df.columns.get_level_values(1)
            else:
                wk_df.columns = wk_df.columns.get_level_values(0)

        wk_df = wk_df.loc[:, ~wk_df.columns.duplicated()]
        if wk_df.index.tz is not None:
            wk_df.index = wk_df.index.tz_localize(None)

        wk_df['SMA50'] = wk_df['Close'].rolling(50).mean()
        wk_df['High_52W'] = wk_df['High'].rolling(52, min_periods=1).max()

        delta = wk_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        wk_df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

        last = wk_df.iloc[-1]
        dist_to_high = ((last['High_52W'] - last['Close']) / max(last['Close'], 1e-9)) * 100

        return {
            "warning": "",
            "label": "✅ 週線資料就緒",
            "detail": {
                "close": float(last['Close']),
                "sma50": float(last['SMA50']) if pd.notna(last['SMA50']) else np.nan,
                "rsi": float(last['RSI']) if pd.notna(last['RSI']) else np.nan,
                "dist_to_52w_high_pct": float(dist_to_high),
                "high_52w": float(last['High_52W'])
            }
        }
    except Exception:
        return {"warning": "", "label": "⚪ 週線計算失敗", "detail": {}}

# ==========================================
# 4.2 動態相關性過濾集群
# ==========================================
def _extract_close_from_multi_download(raw_df, symbol):
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            if symbol in raw_df.columns.get_level_values(0):
                return raw_df[symbol]['Close']
            elif symbol in raw_df.columns.get_level_values(1):
                sub = raw_df.xs(symbol, axis=1, level=1)
                return sub['Close']
        else:
            return raw_df['Close']
    except Exception:
        return None
    return None

@st.cache_data(ttl=1800)
def build_dynamic_correlation_report(symbols, rs_map, lookback=50, threshold=0.75):
    symbols = [s for s in symbols if isinstance(s, str) and s.strip()]
    symbols = list(dict.fromkeys(symbols))
    if len(symbols) < 2:
        return [], None

    try:
        raw = yf.download(symbols, period="120d", interval="1d", group_by="ticker", progress=False)
        if raw.empty:
            return [], None

        closes = pd.DataFrame()
        for sym in symbols:
            series = _extract_close_from_multi_download(raw, sym)
            if series is not None:
                closes[sym] = series

        closes = closes.dropna(axis=1, how='all').ffill().dropna(how='all')
        if closes.shape[1] < 2 or len(closes) < lookback:
            return [], None

        rets = closes.pct_change().dropna().tail(lookback)
        corr = rets.corr().round(2)

        adj = {sym: set() for sym in corr.columns}
        pairs = []

        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = corr.iloc[i, j]
                if pd.notna(c) and c >= threshold:
                    a, b = cols[i], cols[j]
                    adj[a].add(b)
                    adj[b].add(a)
                    pairs.append((a, b, c))

        if not pairs:
            return [], corr

        visited = set()
        clusters = []

        for sym in adj:
            if sym in visited or len(adj[sym]) == 0:
                continue

            stack = [sym]
            comp = set()
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                stack.extend(list(adj[cur] - visited))

            if len(comp) >= 2:
                clusters.append(sorted(comp))

        warnings = []
        for cluster in clusters:
            best = max(cluster, key=lambda x: rs_map.get(x, -999))
            warnings.append(f"高相關性集群：{', '.join(cluster)} ｜ 建議只優先保留 RS 最強的 {best}")

        return warnings, corr
    except Exception:
        return [], None

# ==========================================
# 5. K 線型態
# ==========================================
def detect_candlestick_patterns(df):
    patterns = []
    if df is None or len(df) < 5:
        return patterns

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else (c3['High'] - c3['Low'])

    def get_props(c):
        body = abs(c['Close'] - c['Open'])
        u_shadow = c['High'] - max(c['Open'], c['Close'])
        l_shadow = min(c['Open'], c['Close']) - c['Low']
        trange = c['High'] - c['Low'] + 1e-9
        bull = c['Close'] > c['Open']
        bear = c['Close'] < c['Open']
        return body, u_shadow, l_shadow, trange, bull, bear

    b1, u1, l1, tr1, bull1, bear1 = get_props(c1)
    b2, u2, l2, tr2, bull2, bear2 = get_props(c2)
    b3, u3, l3, tr3, bull3, bear3 = get_props(c3)

    if bear2 and bull3 and c3['Close'] > c2['Open'] and c3['Open'] < c2['Close']:
        patterns.append({"name": "Bullish Engulfing", "signal": "🟢 看漲反轉", "theory": "吞沒前一根陰線，顯示買盤強勢接管。"})
    if bull2 and bear3 and c3['Close'] < c2['Open'] and c3['Open'] > c2['Close']:
        patterns.append({"name": "Bearish Engulfing", "signal": "🔴 看跌反轉", "theory": "吞沒前一根陽線，顯示賣盤強勢接管。"})
    if l3 > b3 * 2 and u3 < b3 * 0.5 and tr3 > atr * 0.5:
        patterns.append({"name": "Hammer", "signal": "🟢 潛在底部", "theory": "長下影顯示低位有強力承接。"})
    if u3 > b3 * 2 and l3 < b3 * 0.5 and tr3 > atr * 0.5:
        patterns.append({"name": "Shooting Star", "signal": "🔴 潛在頂部", "theory": "長上影顯示高位拋壓沉重。"})
    if b3 < tr3 * 0.1 and tr3 > atr * 0.5:
        patterns.append({"name": "Doji", "signal": "⚪ 猶豫訊號", "theory": "多空力量平衡，轉勢前常見。"})
    if not patterns:
        patterns.append({"name": "", "signal": "⚪ 無明顯型態", "theory": "最近 K 線未出現典型反轉/續漲型態。"})

    return patterns

# ==========================================
# 6. 快速歷史勝率計算 (完全同步長短空頭融合邏輯)
# ==========================================
def quick_historical_win_rate(df, market_type):
    long_trades, short_trades = [], []
    position = None
    entry_price, stop_loss, tp_target = 0, 0, 0

    for i in range(20, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        is_hk_liquid = not (market_type == "HK" and prev['TurnoverMA20'] < 20_000_000)

        if position is None:
            # --- 做多判定 ---
            is_kk_trigger = prev['Ret1M'] > 0.15 and prev['Close'] > prev['SMA10'] and prev['VCPSignal'] and is_hk_liquid
            is_long_trigger = prev['Close'] > prev['SMA50'] and prev['SMA50'] > prev['SMA200'] and prev['MACD'] > prev['MACDSignal'] and 50 < prev['RSI'] < 75 and is_hk_liquid
            
            # --- 做空融合判定 (快速版) ---
            # 1. 原版結構做空
            p_high = df['CumulativeHigh'].iloc[i-1] if 'CumulativeHigh' in df.columns else prev['Close'] * 1.5
            sr1 = prev['Close'] < prev['SMA200']
            sr2 = prev['SMA50'] < prev['SMA150'] and prev['SMA150'] < prev['SMA200']
            sr3 = prev['Close'] < p_high * 0.85
            sr4 = prev['Mansfield_RS'] < 0
            sr5 = prev['VolRatio'] < 0.8
            sr6 = prev['LowerShadowRatio'] < 0.4
            sr7 = prev['VCPSignal']
            prev_short_pass_count = sum([sr1, sr2, sr3, sr4, sr5, sr6, sr7])
            
            # 2. J Law 動能做空
            p_low_20_slice = df.iloc[max(0, i-20):i]
            p_low_20_val = p_low_20_slice['Close'].min() if not p_low_20_slice.empty else prev['Close']
            is_breakdown = (prev['Close'] <= p_low_20_val * 1.01) and (prev['VolRatio'] > 1.5)
            ma_pressing = (prev['Close'] < prev['SMA20']) and (prev['SMA20'] < prev['SMA50'])
            is_weak_bounce = prev['VCPSignal'] and (prev['VolRatio'] < 0.8)
            is_jlaw_short = ma_pressing and (prev['Mansfield_RS'] < 0) and (is_weak_bounce or is_breakdown)
            
            # 合併做空邏輯
            is_short_trigger = (prev_short_pass_count >= 5 or is_jlaw_short) and is_hk_liquid
            
            atr_val = prev['ATR'] if pd.notna(prev['ATR']) and prev['ATR'] > 0 else row['Open'] * 0.03

            # OLD:
            # if is_kk_trigger or is_long_trigger:
            #     position = "LONG"
            #     entry_price = row['Open']
            #     stop_loss = entry_price - 2 * atr_val
            #     tp_target = entry_price + 3 * atr_val
            # elif is_short_trigger:
            #     position = "SHORT"
            #     entry_price = row['Open']
            #     stop_loss = entry_price + 2 * atr_val
            #     tp_target = entry_price - 3 * atr_val
            # NEW: [v24.14 掃描器與快速勝率理念同步] 採用真實結構支撐止損 (≤7.2% 上限) 與上方結構阻力止盈，且要求結構 R:R >= 1.8R 才進場
            if is_kk_trigger or is_long_trigger:
                _cand_entry = float(row['Open'])
                _sup_s = float(prev.get('SupShort_Causal', prev.get('SupShort', _cand_entry * 0.94))) - 0.15 * atr_val
                _cand_sl = max(_cand_entry - 1.8 * atr_val, _sup_s, _cand_entry * 0.928)
                if _cand_sl >= _cand_entry * 0.99:
                    _cand_sl = _cand_entry * 0.965
                _risk = max(_cand_entry - _cand_sl, 1e-4)
                _res_t = max(float(prev.get('ResShort_Causal', 0)), float(prev.get('ResLong_Causal', 0)))
                _cand_tp = _res_t if _res_t > _cand_entry + 1.8 * _risk else (_cand_entry + 2.4 * _risk)
                if (_cand_tp - _cand_entry) / _risk >= 1.8:
                    position = "LONG"
                    entry_price = _cand_entry
                    stop_loss = _cand_sl
                    tp_target = _cand_tp
            elif is_short_trigger:
                _cand_entry = float(row['Open'])
                _res_s = float(prev.get('ResShort_Causal', prev.get('ResShort', _cand_entry * 1.06))) + 0.15 * atr_val
                _cand_sl = min(_cand_entry + 1.8 * atr_val, _res_s, _cand_entry * 1.072)
                if _cand_sl <= _cand_entry * 1.01:
                    _cand_sl = _cand_entry * 1.035
                _risk = max(_cand_sl - _cand_entry, 1e-4)
                _sup_t = min(float(prev.get('SupShort_Causal', 999999)), float(prev.get('SupLong_Causal', 999999)))
                _cand_tp = _sup_t if (0 < _sup_t < _cand_entry - 1.8 * _risk) else (_cand_entry - 2.4 * _risk)
                if (_cand_entry - _cand_tp) / _risk >= 1.8:
                    position = "SHORT"
                    entry_price = _cand_entry
                    stop_loss = _cand_sl
                    tp_target = _cand_tp

        elif position == "LONG":
            if row['Low'] <= stop_loss:
                long_trades.append(-1)
                position = None
            elif row['Close'] < row['SMA50']:
                long_trades.append(-1)
                position = None
            elif row['High'] >= tp_target:
                long_trades.append(1)
                position = None

        elif position == "SHORT":
            if row['High'] >= stop_loss:
                short_trades.append(-1)
                position = None
            elif row['Close'] > row['SMA50']:
                short_trades.append(-1)
                position = None
            elif row['Low'] <= tp_target:
                short_trades.append(1)
                position = None

    long_win = sum(1 for t in long_trades if t > 0) / len(long_trades) * 100 if long_trades else 0
    short_win = sum(1 for t in short_trades if t > 0) / len(short_trades) * 100 if short_trades else 0
    return long_win, short_win, len(long_trades), len(short_trades)

# ==========================================
# 7. 核心診斷引擎 (完美融合新舊做空邏輯)
# ==========================================
def analyze_stock_master(ticker, strictness_threshold=5, interval="1d"):
    try:
        ticker, market_type, benchmark_ticker, benchmark_name, currency = format_and_detect_market(ticker)
        bg_df = get_index_data(benchmark_ticker, interval=interval)
        name = get_stock_name(ticker)

        if interval == "1m":
            period = "7d"
        elif interval in ["5m", "15m", "30m"]:
            period = "60d"
        elif interval in ["1h", "90m"]:
            period = "730d"
        else:
            period = "max"

        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return {"error": "無數據返回，請確認代號正確。"}

        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
            else:
                df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()]
        if len(df) < 50:
            return {"error": f"數據不足，目前只有 {len(df)} 根 K 線。"}

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = calculate_indicators(df, bg_df, interval=interval)
        long_win_rate, short_win_rate, l_trades, s_trades = quick_historical_win_rate(df, market_type)

        last = df.iloc[-1]
        prev20 = df.iloc[-20] if len(df) > 20 else last
        low_52w = last['Low52W']
        high_52w = last['High52W']

        kk1 = last['Ret1M'] > 0.15
        kk2 = last['Close'] > last['SMA10'] and last['Close'] > last['SMA20']
        kk3 = last['VCPSignal']
        kk4 = last['Close'] > last['ResShort'] * 0.95
        is_kk_htf = kk1 and kk2 and kk3 and kk4

        # [NEW INCREMENTAL PATCH] 先執行近期止損觸發與短期結構轉空檢查 + LEI 頂底構造/2B/抵扣推演 (修復 Bug 4 & Bug 5)
        invalidation_info = check_recent_signal_and_stop_invalidation(df, lookback_bars=15)
        lei_struct_info = detect_lei_top_bottom_and_2b(df, lookback=35)
        lei_ded_forecast = analyze_lei_deduction_forecast(df)
        is_short_term_uptrend = bool(
            pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and
            (last['Close'] >= last['SMA50']) and not invalidation_info["downtrend_veto_long"] and not invalidation_info["long_stop_hit"]
        )

        if market_type == "HK":
            jl1 = bool(last['Close'] > 1.0)
            jl2 = bool(last['Distto52WHigh'] < 15)
            jl3 = bool(last['Mansfield_RS'] > 0)
            jl4 = bool(last['TurnoverMA20'] > 20_000_000)
            j_law_checklist = {
                "1. 股價 > 1元": jl1,
                "2. 距52W高點 < 15%": jl2,
                "3. RS 跑贏大盤": jl3,
                "4. 20日均額 > 2000萬": jl4
            }
            # OLD: j_law_pass = all([jl1, jl2, jl3, jl4])
            # NEW: 必須同時滿足短期趨勢未破位且未觸發近期止損，防止破位下跌股霸佔 J Law 榜首 (修復 Bug 3 & Bug 4)
            j_law_pass = all([jl1, jl2, jl3, jl4]) and is_short_term_uptrend
            j_law_title = "🌟 J Law 強勢形態"
        else:
            jl1 = bool(last['Close'] > 10.0)
            jl2 = bool(last['Distto52WHigh'] < 25)
            jl3 = bool(last['SMA50'] > last['SMA150'] and last['SMA150'] > last['SMA200'])
            jl4 = bool(last['Volume'] > 500000)
            j_law_checklist = {
                "1. 股價 > 10元": jl1,
                "2. 距高點 < 25% 且近VCP": jl2,
                "3. 50/150/200 多頭排列": jl3,
                "4. 日成交量 > 50萬股": jl4
            }
            # OLD: j_law_pass = all([jl1, jl2, jl3, jl4])
            # NEW: 必須同時滿足短期趨勢未破位且未觸發近期止損 (修復 Bug 3 & Bug 4)
            j_law_pass = all([jl1, jl2, jl3, jl4]) and is_short_term_uptrend
            j_law_title = "🌟 J Law 強勢形態"

        r1 = bool(last['Close'] > last['SMA150'] and last['Close'] > last['SMA200'])
        r2 = bool(last['SMA150'] > last['SMA200'])
        r3 = bool(last['SMA200'] > prev20['SMA200'])
        r4 = bool(last['SMA50'] > last['SMA150'] and last['SMA50'] > last['SMA200'])
        r5 = bool(last['Close'] > low_52w * 1.25)
        r6 = bool(last['Close'] > high_52w * 0.75)
        r7 = bool(last['Mansfield_RS'] > 0)

        checklist = {
            "1. 價格在 SMA150/200 之上": r1,
            "2. SMA150 在 SMA200 之上": r2,
            "3. SMA200 上升": r3,
            "4. SMA50 在 SMA150/200 之上": r4,
            "5. 脫離 52W 低位 >25%": r5,
            "6. 距 52W 高位跌幅 <25%": r6,
            "7. RS 跑贏大盤": r7
        }
        pass_count = sum([r1, r2, r3, r4, r5, r6, r7])

        # ==========================================
        # [做空優化] UI 保留 J Law 動能做空 7 大條件
        # ==========================================
        p_low_20 = df['Close'].tail(20).min()
        is_breakdown = (last['Close'] <= p_low_20 * 1.01) and (last['VolRatio'] > 1.5)
        ma_pressing = (last['Close'] < last['SMA20']) and (last['SMA20'] < last['SMA50'])
        is_weak_bounce = last['VCPSignal'] and (last['VolRatio'] < 0.8)
        
        sr1 = bool(last['Close'] < last['SMA20'])
        sr2 = bool(last['SMA20'] < last['SMA50'])
        sr3 = bool(last['Mansfield_RS'] < 0)
        sr4 = bool(last['MACD'] < 0)
        sr5 = bool(last['Close'] < high_52w * 0.80)
        sr6 = bool(is_breakdown)
        sr7 = bool(is_weak_bounce)

        short_checklist = {
            "1. 股價跌破 20MA (短線轉弱)": sr1,
            "2. 20MA < 50MA (均線蓋頭壓制)": sr2,
            "3. RS 弱於大盤 (相對弱勢)": sr3,
            "4. MACD 零軸之下 (空頭動能)": sr4,
            "5. 距52W高點回撤 >20% (做頭確認)": sr5,
            "6. 帶量破底 (動能下殺 Breakdown)": sr6,
            "7. 無量弱勢壓縮 (熊旗 Bear Flag)": sr7
        }
        short_pass_count = sum([sr1, sr2, sr3, sr4, sr5, sr6, sr7])

        # 【同步原版長線做空邏輯，以決定 pass_count】
        orig_sr1 = bool(last['Close'] < last['SMA200'])
        orig_sr2 = bool(last['SMA50'] < last['SMA150'] and last['SMA150'] < last['SMA200'])
        orig_sr3 = bool(last['Close'] < high_52w * 0.85)
        orig_sr4 = bool(last['Mansfield_RS'] < 0)
        orig_sr5 = bool(last['VolRatio'] < 0.8)
        orig_sr6 = bool(last['LowerShadowRatio'] < 0.4)
        orig_sr7 = bool(last['VCPSignal'])
        original_short_pass_count = sum([orig_sr1, orig_sr2, orig_sr3, orig_sr4, orig_sr5, orig_sr6, orig_sr7])

        ms_metrics = {
            "market_bear": False,
            "rebound_resist": False,
            "low_vol_rebound": False,
            "support_break": False
        }
        ms_score = 0
        try:
            if bg_df is not None and not bg_df.empty:
                bg_last = bg_df.iloc[-1]
                if pd.notna(bg_last.get('SMA50')) and pd.notna(bg_last.get('SMA200')):
                    if (bg_last['SMA50'] < bg_last['SMA200']) or (bg_last['Close'] < bg_last['SMA50']):
                        ms_metrics["market_bear"] = True
            if high_52w > 0:
                drop_from_high = (high_52w - last['Close']) / high_52w
                if drop_from_high > 0.15 and (last['Close'] <= last['SMA50'] * 1.02) and (df['High'].iloc[-5:].max() >= last['SMA50'] * 0.98):
                    ms_metrics["rebound_resist"] = True
            recent_3 = df.iloc[-3:]
            up_days = recent_3[recent_3['Close'] > recent_3['Open']]
            if not up_days.empty:
                avg_up_vol = up_days['Volume'].mean()
                if avg_up_vol < last['VolumeSMA50'] * 0.8:
                    ms_metrics["low_vol_rebound"] = True
            if len(df) > 21:
                low_20d = df['Low'].shift(1).rolling(window=20).min().iloc[-1]
                if last['Close'] < low_20d:
                    ms_metrics["support_break"] = True
            ms_score = sum([ms_metrics["market_bear"], ms_metrics["rebound_resist"], ms_metrics["low_vol_rebound"], ms_metrics["support_break"]])
        except Exception:
            pass 

        target_entry = last['Close']
        atr_val = last['ATR'] if pd.notna(last['ATR']) and last['ATR'] > 0 else target_entry * 0.03
        dist_to_20ma = abs(last['Close'] - last['SMA20']) / last['SMA20'] if pd.notna(last['SMA20']) and last['SMA20'] != 0 else 999
        is_near_20ma = dist_to_20ma < 0.03

        momentum_ready = (last['RSI'] > 55) and (last['Close'] > last['SMA20'])
        is_tight_setup = last['VCPSignal'] and (last['VolRatio'] < 0.8)
        p_high_20 = df['Close'].tail(20).max()
        is_breakout_long = (last['Close'] >= p_high_20 * 0.99) and (last['VolRatio'] > 1.5)
        early_stage_bypass = (last['Close'] > last['SMA50']) and (last['Ret1M'] > 0.20)

        is_long_trigger = momentum_ready and (
            (pass_count >= strictness_threshold) or early_stage_bypass
        ) and (is_tight_setup or is_breakout_long)

        # [做空優化] 結合「原版結構空頭」與「J Law 動能破底」雙邏輯
        is_short_candidate_original = original_short_pass_count >= strictness_threshold
        is_jlaw_short_trigger = ma_pressing and (last['Mansfield_RS'] < 0) and (is_weak_bounce or is_breakdown)
        is_short_trigger = is_short_candidate_original or is_jlaw_short_trigger

        # OLD: is_long_candidate = pass_count >= strictness_threshold
        # NEW: 修復 Bug 4 & Bug 5 —— 若股票已跌破短期結構 (downtrend_veto_long) 或已觸及前波買點止損 (long_stop_hit)，嚴禁歸類為做多候選！
        is_long_candidate = (
            (pass_count >= strictness_threshold)
            and (not invalidation_info["downtrend_veto_long"])
            and (not invalidation_info["long_stop_hit"])
            and (last['Close'] >= last['SMA50'])
        )
        is_short_candidate = (short_pass_count >= strictness_threshold) and (not invalidation_info["short_stop_hit"]) # 用於 UI 顯示 JLaw 滿足度
        
        is_cohodes_short = ms_score >= 3 and last['Close'] < last['SMA50']
        is_rebound_short = ms_metrics["rebound_resist"] and ms_metrics["low_vol_rebound"]
        
        is_deduction_pullback = False
        if last.get('Clock_2', False):
            # OLD:
            # if (last['Close'] < last['SMA20'] and last['Close'] > last.get('SMA20_Deduction', 0)) or \
            #    (last['Close'] < last['SMA50'] and last['Close'] > last.get('SMA50_Deduction', 0)):
            #     is_deduction_pullback = True
            # NEW: [LEI 影片4 13:00-14:50] 2點鐘回撤買點必須滿足：(1)未觸及前波止損 (2)非爆量長陰破位 (3)未來抵扣價不構成下彎威脅 (4)未出現頂部構造
            if (
                (not invalidation_info["long_stop_hit"])
                and (not last.get('Is_Abnormal_Drop', False))
                and (not lei_struct_info.get("top_structure", False))
                and (last['Close'] >= last['SMA50'] * 0.99)
                and (
                    (last['Close'] < last['SMA20'] and last['Close'] > last.get('SMA20_Deduction', 0)) or
                    (last['Close'] < last['SMA50'] and last['Close'] > last.get('SMA50_Deduction', 0)) or
                    (lei_struct_info.get("bottom_structure", False) and is_near_20ma)
                )
            ):
                is_deduction_pullback = True

        # [NEW LEI 均線高度密集 (<2.5%) 多頭張嘴突破埋伏信號 - 影片4 04:15-05:50]
        is_lei_ma_convergence_long = bool(
            last.get('LEI_MA_Tight_2Pct', False)
            and (last['Close'] > last.get('EMA20', 0))
            and (last.get('EMA20', 0) >= last.get('EMA60', 0) * 0.995)
            and lei_ded_forecast.get("is_20ma_turn_up_favorable", False)
            and (not invalidation_info["long_stop_hit"])
        )

        # [NEW] 若已觸發空頭破位或短期結構轉空，優先防止被長線滯後多頭指標覆蓋 (修復 Bug 1 & Bug 4)
        has_active_short_signal = bool(
            (is_jlaw_short_trigger or is_short_candidate_original or is_cohodes_short or is_rebound_short or is_short_candidate)
            and not invalidation_info["uptrend_veto_short"]
        )

        if is_kk_htf and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"]:
            plan_type = "LONG"
            status = "🔥 Kullamägi 高階動能旗型"
        elif is_long_trigger and is_breakout_long and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = "🟢 放量強勢突破 (Breakout)"
        elif is_long_trigger and is_tight_setup and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = "🟢 極限壓縮潛伏 (VCP/緊密箱體 Setup)"
        elif is_lei_ma_convergence_long and not has_active_short_signal:
            plan_type = "LONG"
            status = "🟢 LEI 均線高度密集 (<2.5%) 張嘴突破埋伏"
        elif is_long_trigger and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = "🟢 20MA 回踩做多信號"
        elif is_deduction_pullback and not has_active_short_signal:
            plan_type = "LONG"
            _b_tag = " + 底構造確認" if lei_struct_info.get("bottom_structure") else ""
            status = f"🟡 趨勢回撤買點 (2點鐘方向，抵扣價防守成功{_b_tag})"
        elif is_jlaw_short_trigger and is_breakdown:
            plan_type = "SHORT"
            status = "🔴 帶量破底做空 (Breakdown)"
        elif is_jlaw_short_trigger and is_weak_bounce:
            plan_type = "SHORT"
            status = "🔴 均線蓋頭弱反彈 (Bear Flag)"
        elif is_short_candidate_original:
            plan_type = "SHORT"
            status = "🟠 空頭趨勢成型做空 (長線結構)"
        elif is_cohodes_short:
            plan_type = "SHORT"
            status = "🏴‍☠️ Cohodes 絕佳空點 (破位+反彈無力)"
        elif is_rebound_short:
            plan_type = "SHORT"
            status = "🩸 弱勢反彈受阻 (高勝率空點)"
        elif is_short_candidate:
            plan_type = "SHORT"
            status = "🟠 J Law 空頭條件成型 (醞釀中)"
        elif is_long_candidate:
            plan_type = "LONG"
            status = "🟡 強勢多頭排列 (等待拉回或突破買點)"
        else:
            plan_type = "WATCH"
            if invalidation_info["long_stop_hit"]:
                status = f"⚪ 觀望 ({invalidation_info['invalidation_reason']})"
            elif invalidation_info["downtrend_veto_long"] and pass_count >= strictness_threshold:
                status = "⚪ 觀望 (長線均線雖多頭，但短中期已破位下跌)"
            else:
                status = "⚪ 趨勢中性 / 觀望"

        # 設定基於前阻力的優先目標價
        potential_long_sl = target_entry - 2 * atr_val
        potential_long_tp = max(last.get("ResShort", target_entry * 1.08), target_entry + 4 * atr_val)
        
        potential_short_sl = target_entry + 2 * atr_val
        potential_short_tp = min(last.get("SupShort", target_entry * 0.92), target_entry - 4 * atr_val)

        # OLD:
        # if pass_count >= original_short_pass_count or plan_type == "LONG":
        #     stop_loss = potential_long_sl
        #     take_profit = potential_long_tp
        # else:
        #     stop_loss = potential_short_sl
        #     take_profit = potential_short_tp
        # NEW: [v24.14 結構買入位、結構止損與多層級真實阻力止盈]
        trade_plan_eval = evaluate_swing_trade_plan_and_rr(
            df=df,
            plan_type=plan_type,
            status=status,
            atr_val=atr_val,
            is_breakout_long=bool(is_breakout_long),
            is_near_20ma=bool(is_near_20ma),
            is_deduction_pullback=bool(is_deduction_pullback),
            is_breakdown_short=bool(is_breakdown),
            lei_struct=lei_struct_info
        )
        target_entry = trade_plan_eval["entry"]
        if plan_type == "LONG":
            stop_loss = trade_plan_eval["sl"]
            take_profit = trade_plan_eval["tp"]
        elif plan_type == "SHORT":
            stop_loss = trade_plan_eval["sl"]
            take_profit = trade_plan_eval["tp"]
        else:
            stop_loss = potential_long_sl
            take_profit = potential_long_tp

        jlaw_meta_info = evaluate_jlaw_power_trend_and_meta(
            df=df,
            bg_df=bg_df,
            market_type=market_type,
            lei_struct=lei_struct_info,
            lei_ded=lei_ded_forecast,
            trade_plan_eval=trade_plan_eval
        )

        df.attrs["trade_plan_eval"] = trade_plan_eval
        df.attrs["invalidation_info"] = invalidation_info
        df.attrs["lei_struct_info"] = lei_struct_info
        df.attrs["lei_ded_forecast"] = lei_ded_forecast
        df.attrs["jlaw_meta_info"] = jlaw_meta_info

        # --- 🌐 大週期週線共振監測 ---
        weekly_alignment = {"warning": "", "label": "", "detail": {}}
        if interval == "1d":
            weekly_alignment = get_weekly_alignment_signal(ticker)
            wk = weekly_alignment.get("detail", {})
            weekly_warnings = []

            if plan_type == "LONG" and wk:
                if pd.notna(wk.get("sma50", np.nan)) and wk.get("close", 0) < wk.get("sma50", 0):
                    weekly_warnings.append("日線看多，但週線仍在 SMA50 下方，屬逆大勢反彈。")
                if pd.notna(wk.get("dist_to_52w_high_pct", np.nan)) and wk.get("dist_to_52w_high_pct", 999) < 5:
                    weekly_warnings.append("週線已逼近 52 週歷史壓力區。")
                if pd.notna(wk.get("rsi", np.nan)) and wk.get("rsi", 0) > 75:
                    weekly_warnings.append("週線 RSI 極度超買，追價風險偏高。")

            elif plan_type == "SHORT" and wk:
                if pd.notna(wk.get("sma50", np.nan)) and wk.get("close", 0) > wk.get("sma50", 0):
                    weekly_warnings.append("日線看空，但週線仍在 SMA50 上方，屬逆勢做空。")
                if pd.notna(wk.get("rsi", np.nan)) and wk.get("rsi", 100) < 25:
                    weekly_warnings.append("週線 RSI 極度超賣，隨時可能出現強力反彈。")

            weekly_alignment["warning"] = " ".join(weekly_warnings)

        # 🌟 J Law 出貨預警計算
        dist_days = last['Dist_Days_20d'] if pd.notna(last['Dist_Days_20d']) else 0
        jlaw_warnings = []
        if dist_days >= 4:
            jlaw_warnings.append(f"🚨 【強烈警報】近 4 週內出現 {int(dist_days)} 個出貨日 (≥4)，大戶資金正在撤退！")
        if last.get('Is_Abnormal_Drop', False):
            jlaw_warnings.append("🚨 【突兀走勢】近日出現帶量大跌或跳空下殺，顯示恐慌性拋售！")
        if last.get('Is_Climax_Run', False):
            jlaw_warnings.append("⚠️ 【急漲見頂】股價乖離過大且呈 12點鐘急漲走勢 (Climax Run)，隨時可能暴跌！")
            
        jlaw_alert = "\n\n".join(jlaw_warnings) if jlaw_warnings else "✅ 目前未見明顯大級別出貨訊號，籌碼相對穩定。"

        return df, status, target_entry, stop_loss, take_profit, plan_type, checklist, pass_count, benchmark_name, low_52w, high_52w, currency, name, short_checklist, short_pass_count, long_win_rate, short_win_rate, l_trades, s_trades, j_law_checklist, j_law_pass, j_law_title, is_near_20ma, is_kk_htf, weekly_alignment, ms_metrics, ms_score, jlaw_alert, dist_days, is_deduction_pullback
    except Exception as e:
        return {"error": f"運算崩潰: {str(e)}"}

# ==========================================
# 8. 雙向歷史回測引擎
# ==========================================
def run_bidirectional_backtest(ticker, start_date, end_date, initial_capital, score_threshold, trade_mode, is_relaxed):
    try:
        ticker, market_type, benchmark_ticker, benchmark_name, currency = format_and_detect_market(ticker)
        bg_df = get_index_data(benchmark_ticker)
        name = get_stock_name(ticker)

        download_start = start_date - timedelta(days=365)
        df = yf.download(ticker, start=download_start, end=end_date, progress=False)
        if df.empty:
            return "無數據，無法回測。"

        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            elif 'Close' in df.columns.get_level_values(1):
                df.columns = df.columns.get_level_values(1)
            else:
                df.columns = df.columns.get_level_values(0)

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(subset=['Close'])
        
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = calculate_indicators(df, bg_df)
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        if df.empty:
            return "選定區間內無有效數據。"

        cash = initial_capital
        position_type = None
        entry_price, stop_loss, tp_target, entry_structural_sl = 0, 0, 0, None
        
        position_stage = 0 
        days_in_trade = 0 
        
        shares = 0
        trade_log, history_equity, history_dates = [], [], []

        slippage_fee = 0.0015 if market_type == "HK" else 0.0002

        for i in range(1, len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            p_prev20 = df.iloc[i - 20] if i >= 20 else df.iloc[0]

            is_hk_liquid = not (market_type == "HK" and prev_row['TurnoverMA20'] < 20_000_000)

            is_market_bullish = True
            if 'CloseBench' in prev_row and 'SMA50BenchValue' in prev_row and pd.notna(prev_row['CloseBench']):
                is_market_bullish = prev_row['CloseBench'] > prev_row['SMA50BenchValue']

            # 大盤 O'Neil 防禦閘門
            bench_close = prev_row.get('CloseBench', np.nan)
            bench_200 = prev_row.get('SMA200BenchValue', np.nan)
            bench_dist = prev_row.get('Bench_Dist_Days', 0)
            is_market_health_ok = True
            if pd.notna(bench_close) and pd.notna(bench_200):
                if bench_close < bench_200 or bench_dist >= 5:
                    is_market_health_ok = False

            current_equity = cash + (
                shares * row['Close'] if position_type == "LONG"
                else ((entry_price - row['Close']) * shares if position_type == "SHORT" else 0)
            )

            # ====================================================
            # 觸發條件套用「融合邏輯」
            # ====================================================
            if is_relaxed:
                is_long_trigger = prev_row['Close'] > prev_row['SMA50'] and prev_row['MACD'] > prev_row['MACDSignal'] and is_hk_liquid and is_market_bullish and is_market_health_ok
                is_short_trigger = prev_row['Close'] < prev_row['SMA50'] and prev_row['MACD'] < prev_row['MACDSignal'] and is_hk_liquid and not is_market_bullish
            else:
                p_low, p_high = df.iloc[:i]['Low'].min(), df.iloc[:i]['High'].max()

                r1 = prev_row['Close'] > prev_row['SMA150'] and prev_row['Close'] > prev_row['SMA200']
                r2 = prev_row['SMA150'] > prev_row['SMA200']
                r3 = prev_row['SMA200'] > p_prev20['SMA200']
                r4 = prev_row['SMA50'] > prev_row['SMA150'] and prev_row['SMA50'] > prev_row['SMA200']
                r5 = prev_row['Close'] > p_low * 1.25
                r6 = prev_row['Close'] > p_high * 0.70
                r7 = prev_row['Mansfield_RS'] > 0
                prev_pass_count = sum([r1, r2, r3, r4, r5, r6, r7])

                kk_flag = prev_row['Ret1M'] > 0.15 and prev_row['Close'] > prev_row['SMA10'] and prev_row['VCPSignal']
                
                # 做多動態豁免
                momentum_ready = (prev_row['RSI'] > 55) and (prev_row['Close'] > prev_row['SMA20'])
                is_tight_setup = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)
                p_high_20_slice = df.iloc[max(0, i-20):i]
                p_high_20_val = p_high_20_slice['Close'].max() if not p_high_20_slice.empty else prev_row['Close']
                is_breakout = (prev_row['Close'] >= p_high_20_val * 0.99) and (prev_row['VolRatio'] > 1.5)
                early_stage_bypass = (prev_row['Close'] > prev_row['SMA50']) and (prev_row['Ret1M'] > 0.20)

                is_long_trigger = momentum_ready and (
                    (prev_pass_count >= score_threshold) or early_stage_bypass or kk_flag
                ) and (is_tight_setup or is_breakout) and is_hk_liquid and is_market_health_ok

                # [做空優化] 結合「原版結構空頭」與「J Law 動能做空」雙邏輯
                orig_sr1 = prev_row['Close'] < prev_row['SMA200']
                orig_sr2 = prev_row['SMA50'] < prev_row['SMA150'] and prev_row['SMA150'] < prev_row['SMA200']
                orig_sr3 = prev_row['Close'] < (df['CumulativeHigh'].iloc[i-1] if 'CumulativeHigh' in df.columns else p_high) * 0.85
                orig_sr4 = prev_row['Mansfield_RS'] < 0
                orig_sr5 = prev_row['VolRatio'] < 0.8
                orig_sr6 = prev_row['LowerShadowRatio'] < 0.4
                orig_sr7 = prev_row['VCPSignal']
                prev_short_pass_count = sum([orig_sr1, orig_sr2, orig_sr3, orig_sr4, orig_sr5, orig_sr6, orig_sr7])
                
                is_short_candidate_original = prev_short_pass_count >= score_threshold

                p_low_20_slice = df.iloc[max(0, i-20):i]
                p_low_20_val = p_low_20_slice['Close'].min() if not p_low_20_slice.empty else prev_row['Close']
                is_breakdown = (prev_row['Close'] <= p_low_20_val * 1.01) and (prev_row['VolRatio'] > 1.5)
                ma_pressing = (prev_row['Close'] < prev_row['SMA20']) and (prev_row['SMA20'] < prev_row['SMA50'])
                is_weak_bounce = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)

                is_jlaw_short_trigger = ma_pressing and (prev_row['Mansfield_RS'] < 0) and (is_weak_bounce or is_breakdown)
                
                is_short_trigger = (is_short_candidate_original or is_jlaw_short_trigger) and is_hk_liquid and not is_market_bullish

            atr_val = prev_row['ATR'] if pd.notna(prev_row['ATR']) and prev_row['ATR'] > 0 else row['Open'] * 0.03

            if position_type is None:
                # [NEW v24.14 回測與掃描系統 100% 理念同步]
                _lei_pullback_ok = bool(
                    prev_row.get('Clock_2', False)
                    and (not prev_row.get('Is_Abnormal_Drop', False))
                    and (prev_row['Close'] >= prev_row.get('SMA50', 0) * 0.99)
                    and (prev_row['Close'] < prev_row.get('SMA20', 0) and prev_row['Close'] > prev_row.get('SMA20_Deduction', 0))
                )
                _lei_tight_ok = bool(
                    prev_row.get('LEI_MA_Tight_2Pct', False)
                    and (prev_row['Close'] > prev_row.get('EMA20', 0))
                    and (prev_row['Close'] > prev_row.get('SMA20_Deduction', 0))
                )
                _short_term_uptrend_ok = bool(prev_row['Close'] >= prev_row.get('SMA50', 0))
                _unified_long_signal = (is_long_trigger or _lei_pullback_ok or _lei_tight_ok) and _short_term_uptrend_ok and is_hk_liquid and is_market_health_ok

                if _unified_long_signal and trade_mode in ["全部雙向", "僅做多頭(做多)"]:
                    trigger_price = float(prev_row['High'])
                    if row['High'] >= trigger_price and row['Open'] <= trigger_price * 1.035:
                        actual_entry_price = max(float(row['Open']), trigger_price)
                        entry_price = actual_entry_price * (1 + slippage_fee)

                        _struct_sup = max(
                            float(prev_row.get("BoxLow10_Causal", prev_row.get("SupShort_Causal", entry_price * 0.94))) - 0.15 * atr_val,
                            entry_price * 0.925
                        )
                        _atr_sl = max(entry_price - 1.8 * atr_val, entry_price * 0.925)
                        stop_loss = max(_atr_sl, _struct_sup) if (entry_price - _struct_sup) >= 0.8 * atr_val else _atr_sl
                        if stop_loss >= entry_price * 0.988:
                            stop_loss = entry_price * 0.965
                        risk_per_share = max(entry_price - stop_loss, 1e-4)

                        _cands_tp = [
                            float(prev_row.get("ResShort_Causal", 0)),
                            float(prev_row.get("ResLong_Causal", 0)),
                            float(prev_row.get("Res120_Causal", 0)),
                            float(prev_row.get("High52W", 0))
                        ]
                        _valid_tps = sorted([p for p in _cands_tp if p > entry_price * 1.022])
                        if _valid_tps:
                            tp_target = _valid_tps[0]
                            if (tp_target - entry_price) / risk_per_share < 1.6 and len(_valid_tps) >= 2:
                                tp_target = _valid_tps[1]
                        else:
                            _wave_amp = max(float(prev_row.get("ResShort_Causal", entry_price)) - float(prev_row.get("SupShort_Causal", stop_loss)), 3.2 * atr_val)
                            tp_target = entry_price + _wave_amp * 0.85

                        _planned_rr = (tp_target - entry_price) / risk_per_share
                        if _planned_rr >= 1.8:
                            risk_amount = current_equity * 0.01
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            if (target_shares * entry_price) > cash:
                                target_shares = int(cash / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                shares = target_shares
                                cash -= shares * entry_price
                                position_stage = 0
                                days_in_trade = 0
                                entry_structural_sl = float(prev_row.get("SupShort_Causal", prev_row.get("SupShort", entry_price * 0.92)))
                                position_type = "LONG"
                                trade_info = {
                                    "方向": "🟢 做多",
                                    "進場日期": current_date,
                                    "進場價": round(entry_price, 2),
                                    "初始止損": round(stop_loss, 2),
                                    "目標止盈": round(tp_target, 2),
                                    "計畫盈虧比": f"{_planned_rr:.2f}R",
                                    "最終止盈": "結構TP 或 9EMA 跟蹤",
                                    "結構止損": round(entry_structural_sl, 2)
                                }

                elif is_short_trigger and trade_mode in ["全部雙向", "僅做空頭(做空)"]:
                    trigger_price = float(prev_row['Low'])
                    if row['Low'] <= trigger_price and row['Open'] >= trigger_price * 0.965:
                        actual_entry_price = min(float(row['Open']), trigger_price)
                        entry_price = actual_entry_price * (1 - slippage_fee)

                        _struct_res = min(
                            float(prev_row.get("BoxHigh10_Causal", prev_row.get("ResShort_Causal", entry_price * 1.06))) + 0.15 * atr_val,
                            entry_price * 1.075
                        )
                        _atr_sl = min(entry_price + 1.8 * atr_val, entry_price * 1.075)
                        stop_loss = min(_atr_sl, _struct_res) if (_struct_res - entry_price) >= 0.8 * atr_val else _atr_sl
                        if stop_loss <= entry_price * 1.012:
                            stop_loss = entry_price * 1.035
                        risk_per_share = max(stop_loss - entry_price, 1e-4)

                        _cands_tp = [
                            float(prev_row.get("SupShort_Causal", 999999)),
                            float(prev_row.get("SupLong_Causal", 999999)),
                            float(prev_row.get("Sup120_Causal", 999999)),
                            float(prev_row.get("Low52W", 999999))
                        ]
                        _valid_tps = sorted([p for p in _cands_tp if 0 < p < entry_price * 0.978], reverse=True)
                        if _valid_tps:
                            tp_target = _valid_tps[0]
                            if (entry_price - tp_target) / risk_per_share < 1.6 and len(_valid_tps) >= 2:
                                tp_target = _valid_tps[1]
                        else:
                            _wave_amp = max(float(prev_row.get("ResShort_Causal", stop_loss)) - float(prev_row.get("SupShort_Causal", entry_price)), 3.2 * atr_val)
                            tp_target = max(0.01, entry_price - _wave_amp * 0.85)

                        _planned_rr = (entry_price - tp_target) / risk_per_share
                        if _planned_rr >= 1.8:
                            risk_amount = current_equity * 0.01
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            if (target_shares * entry_price) > cash:
                                target_shares = int(cash / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                shares = target_shares
                                position_stage = 0
                                days_in_trade = 0
                                entry_structural_sl = float(prev_row.get("ResShort_Causal", prev_row.get("ResShort", entry_price * 1.08)))
                                position_type = "SHORT"
                                trade_info = {
                                    "方向": "🔴 做空",
                                    "進場日期": current_date,
                                    "進場價": round(entry_price, 2),
                                    "初始止損": round(stop_loss, 2),
                                    "目標止盈": round(tp_target, 2),
                                    "計畫盈虧比": f"{_planned_rr:.2f}R",
                                    "最終止盈": "結構TP 或 9EMA 跟蹤",
                                    "結構止損": round(entry_structural_sl, 2)
                                }

            elif position_type == "LONG":
                days_in_trade += 1
                is_time_stop = (days_in_trade >= 15) and (row['Close'] < entry_price * 1.02)
                is_trending_market = pd.notna(bench_close) and bench_close > prev_row.get('SMA50BenchValue', 0) * 1.015
                
                # 保本升級檢查
                if position_stage == 0 and days_in_trade >= 2 and row.get('EMA9', 0) > entry_price:
                    position_stage = 1
                    stop_loss = entry_price 

                if row['Low'] <= stop_loss or is_time_stop:
                    base_exit = min(row['Open'], stop_loss) if row['Low'] <= stop_loss else row['Close']
                    result_text = "觸發防守止損/保本" if row['Low'] <= stop_loss else "時間停損 (15日)"

                    exit_price = base_exit * (1 - slippage_fee)
                    cash += (shares * exit_price)
                    profit = (exit_price - entry_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": 0
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                elif row['High'] >= tp_target:
                    if is_trending_market and position_stage < 2:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0: sold_shares = shares 
                        
                        exit_price = max(row['Open'], tp_target) * (1 - slippage_fee)
                        cash += (sold_shares * exit_price)
                        shares -= sold_shares
                        
                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：觸及前高減倉一半",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": (exit_price - entry_price) * sold_shares,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(partial_info)
                        
                        if shares > 0:
                            position_stage = 2 # 進入純跟蹤模式
                            stop_loss = entry_price # 剩下的確保保本
                        else:
                            position_type, position_stage = None, 0
                    elif not is_trending_market:
                        exit_price = max(row['Open'], tp_target) * (1 - slippage_fee)
                        cash += (shares * exit_price)
                        profit = (exit_price - entry_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "震盪市：觸及前高全平",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

                elif position_stage == 2:
                    # OLD: if row['Close'] < row.get('EMA9', 0) or row['Close'] < row.get('SupShort', 0):
                    # NEW: 採用無未來函數之 SupShort_Causal 與 9EMA 作為大趨勢跟蹤離場線
                    if row['Close'] < row.get('EMA9', 0) or row['Close'] < row.get('SupShort_Causal', row.get('SupShort', 0)):
                        exit_price = row['Close'] * (1 - slippage_fee)
                        cash += (shares * exit_price)
                        profit = (exit_price - entry_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：跌破9EMA/趨勢線離場",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

            elif position_type == "SHORT":
                days_in_trade += 1
                is_time_stop = (days_in_trade >= 15) and (row['Close'] > entry_price * 0.98)
                is_trending_market = pd.notna(bench_close) and bench_close < prev_row.get('SMA50BenchValue', 0) * 0.985

                if position_stage == 0 and days_in_trade >= 2 and row.get('EMA9', 9999) < entry_price:
                    position_stage = 1
                    stop_loss = entry_price

                if row['High'] >= stop_loss or is_time_stop:
                    base_exit = max(row['Open'], stop_loss) if row['High'] >= stop_loss else row['Close']
                    result_text = "觸發防守止損/保本" if row['High'] >= stop_loss else "時間停損 (15日)"

                    exit_price = base_exit * (1 + slippage_fee)
                    cash += ((entry_price - exit_price) * shares)
                    profit = (entry_price - exit_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": 0
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                elif row['Low'] <= tp_target:
                    if is_trending_market and position_stage < 2:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0: sold_shares = shares
                        
                        exit_price = min(row['Open'], tp_target) * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * sold_shares)
                        shares -= sold_shares
                        
                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：觸及前低減倉一半",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": (entry_price - exit_price) * sold_shares,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(partial_info)
                        
                        if shares > 0:
                            position_stage = 2
                            stop_loss = entry_price
                        else:
                            position_type, position_stage = None, 0
                    elif not is_trending_market:
                        exit_price = min(row['Open'], tp_target) * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * shares)
                        profit = (entry_price - exit_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "震盪市：觸及前低全平",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

                elif position_stage == 2:
                    # OLD: if row['Close'] > row.get('EMA9', 9999) or row['Close'] > row.get('ResShort', 9999):
                    # NEW: 採用無未來函數之 ResShort_Causal 與 9EMA 作為空頭跟蹤離場線
                    if row['Close'] > row.get('EMA9', 9999) or row['Close'] > row.get('ResShort_Causal', row.get('ResShort', 9999)):
                        exit_price = row['Close'] * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * shares)
                        profit = (entry_price - exit_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：升穿 9EMA/趨勢線離場",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

            current_equity = cash + (
                shares * row['Close'] if position_type == "LONG"
                else ((entry_price - row['Close']) * shares if position_type == "SHORT" else 0)
            )
            history_equity.append(current_equity)
            history_dates.append(current_date)

        if position_type is not None:
            exit_price = df.iloc[-1]['Close']
            if position_type == "LONG":
                exit_price = exit_price * (1 - slippage_fee)
                cash += shares * exit_price
                profit = (exit_price - entry_price) * shares
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": exit_price,
                    "結果": "回測結束強制平倉",
                    "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                    "利潤": profit
                })
            else:
                exit_price = exit_price * (1 + slippage_fee)
                cash += (entry_price - exit_price) * shares
                profit = (entry_price - exit_price) * shares
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": exit_price,
                    "結果": "回測結束強制平倉",
                    "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                    "利潤": profit
                })
            trade_log.append(trade_info)

        return cash, trade_log, pd.Series(history_equity, index=history_dates), ((df.iloc[-1]['Close'] - df.iloc[0]['Open']) / df.iloc[0]['Open']) * 100, currency, name
    except Exception as e:
        return f"{str(e)}"

# ==========================================
# 9. 側欄控制
# ==========================================
st.sidebar.markdown("## ⚙️ 系統設定")
st.sidebar.markdown("### 量化過濾條件設定")
# OLD: strictness = st.sidebar.slider("量化過濾嚴格度 (預設: 5)", min_value=4, max_value=7, value=5, step=1)
# NEW: [修復 Bug 1C] 若從雷達跳轉進入診斷，自動沿用雷達掃描當時的 strictness，避免因嚴格度重設為 5 導致多空判定不一致
_default_strict = int(target_strict_from_radar) if str(target_strict_from_radar).isdigit() and 4 <= int(target_strict_from_radar) <= 7 else 5
strictness = st.sidebar.slider("量化過濾嚴格度 (預設: 5)", min_value=4, max_value=7, value=_default_strict, step=1)

if strictness == 4:
    st.sidebar.success("4/7：最寬鬆")
elif strictness == 5:
    st.sidebar.info("5/7：平衡 (建議)")
elif strictness == 6:
    st.sidebar.warning("6/7：偏嚴格")
elif strictness == 7:
    st.sidebar.error("7/7：最嚴格")

st.sidebar.markdown("---")

# [優化加入] 無限擴充股票池輸入框
st.sidebar.markdown("### 🌐 擴充掃描池 (打破百隻限制)")
custom_tickers_raw = st.sidebar.text_area("自訂股票代碼 (逗號或空白分隔)", placeholder="例如: AAPL, PLTR, MSTR\n在此輸入你今天感興趣的代號，加入雷達進行全景體檢。")

st.sidebar.markdown("---")

mode_options = [
    "🔎 1. 市場動能掃描雷達",
    "📊 2. 個股量化分析與技術診斷",
    "⭐ 3. 自選股動能面板",
    "📈 4. 策略歷史回測系統"
]

default_idx = 0
if default_mode_val == "診斷":
    default_idx = 1

mode = st.sidebar.radio("請選擇功能", mode_options, index=default_idx)
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **合規與風險聲明**：本系統僅供量化研究與技術分析參考，不構成任何投資邀約或買賣建議。歷史回測勝率不代表未來表現，市場數據可能存在延遲或複權誤差，請嚴格執行個人風險管理。")

if mode == mode_options[0]:
    st.query_params.clear()

# ==========================================
# 10. 雷達顯示
# ==========================================
def render_styled_dataframe(df):
    df = df.dropna(subset=['代碼', '最新價']).copy()
    if df.empty:
        st.warning("數據過濾後無符合條件的結果。")
        return

    def highlight_status(val):
        if isinstance(val, str):
            if '🔥' in val or '🟢' in val or '🌟' in val:
                return 'color: #00FF00; font-weight: bold;'
            if '🔴' in val:
                return 'color: #FF4B4B; font-weight: bold;'
            if '🩸' in val:
                return 'color: #D32F2F; font-weight: bold;'
            if '🏴‍☠️' in val:
                return 'color: #000000; font-weight: bold; background-color: #EF5350;'
            if '🟠' in val:
                return 'color: #FFA500; font-weight: bold;'
            if '🟡' in val:
                return 'color: #FFD700;'
            if '⚪' in val:
                return 'color: #A9A9A9; font-style: italic;'
            if '🚨' in val:
                return 'color: #FF6347;'
            if '🗑️' in val:
                return 'color: #808080;'
        return ''

    # OLD: styled_df = df.style.map(highlight_status, subset=['J Law 強勢形態', '診斷結果', '交易方向'])
    # NEW: [修復 Screenshot 2 KeyError] 動態取 df.columns 交集，無論精簡視圖或完整視圖均絕不拋出 KeyError
    _valid_style_cols = [c for c in ['優先次序', 'J Law 強勢形態', '診斷結果', '交易方向', '信號狀態', 'META多重優勢'] if c in df.columns]
    styled_df = df.style.map(highlight_status, subset=_valid_style_cols) if _valid_style_cols else df.style
    
    st.dataframe(
        styled_df,
        column_config={
            "跳轉診斷": st.column_config.LinkColumn(
                "🔍 分析器",
                display_text="進入診斷",
                help="請使用下方選單進入診斷"
            ),
            "代碼": st.column_config.TextColumn("代碼")
        },
        use_container_width=True,
        hide_index=True
    )

def render_radar_results(df_results, title, market_key=None):
    if df_results is None or df_results.empty:
        return

    st.markdown(title)

    if market_key is not None:
        warning_list = st.session_state.radar_correlation_warnings.get(market_key, [])
        corr_matrix = st.session_state.radar_correlation_matrices.get(market_key, None)

        if warning_list:
            st.error("🚨 **動態相關性過濾警示：偵測到高同質性集群，請避免過度集中。**")
            for msg in warning_list:
                st.write(f"- {msg}")
            if corr_matrix is not None:
                with st.expander("查看 50 日報酬相關性矩陣", expanded=False):
                    st.dataframe(corr_matrix, use_container_width=True)
        else:
            st.success("✅ 未發現明顯高相關性集群，持倉分散度相對健康。")

    # [NEW v24.14] 穩定容器包裹 + 清晰數字編號 (#1, #2, #3...) 優先買入與優先沽空精選榜 + 精簡/專家模式切換
    with st.container():
        if market_key is not None:
            _b_stats = persistent_data.get("radar_breadth_stats", {}).get(market_key)
            if _b_stats:
                _p50 = _b_stats.get("pct_above_50ma", 0.0)
                _p200 = _b_stats.get("pct_above_200ma", 0.0)
                if _p50 < 45.0:
                    st.warning(f"📉 **J Law 市場寬度預警 ({market_key})**：目前池內僅 **{_p50:.1f}%** 股票高於 50MA、**{_p200:.1f}%** 高於 200MA。若大盤指數仍在高位，屬「指數與成分股背馳」見頂信號，宜縮減倉位或冬眠！")
                else:
                    st.info(f"🌐 **J Law 市場內部體質 ({market_key})**：高於 50MA 比例 **{_p50:.1f}%** ｜ 高於 200MA 比例 **{_p200:.1f}%**")

        with st.expander("🧭 新手必看：3步看懂並使用此掃描器 (點擊展開教學)", expanded=False):
            st.markdown(
                "1. **第一步（直接看下方【🏆 優先買入 / 🏴‍☠️ 優先沽空 數字編號榜】）**：系統已自動幫你剔除「盈虧比不足 (<1.8R)」、「已跌破止損」或「純多頭但無買點」的股票，並按綜合實力由 **`#1`、`#2`、`#3`...** 排好優先次序。\n"
                "2. **第二步（看清【🎯 建議買入位 & 掛單方式】，絕不盲目用現價買）**：\n"
                "   - **突破單 (Stop-Buy)**：代表股價正處於緊密平台，請在顯示的「建議買入價（箱體頂樞紐）」上方設突破觸發單，未突破前不買。\n"
                "   - **回踩低吸單 (Limit-Buy)**：代表股價剛突破或正在回測 10/20MA 及前波浪頂支撐區，請在顯示的「建議買入價」附近掛單低吸，切勿追高。\n"
                "3. **第三步（一鍵進入個股診斷看 K 線圖）**：在下方下拉選單揀選 `#1` 或 `#2` 的股票，撳 **「🚀 進入診斷」**，K 線圖會用藍線(Entry)、紅線(結構止損SL)、綠線(結構阻力止盈TP) 畫出完整交易計劃！"
            )

        df_long = df_results[df_results['交易方向'].astype(str).str.contains("做多", na=False)].copy()
        df_short = df_results[df_results['交易方向'].astype(str).str.contains("做空", na=False)].copy()
        df_watch = df_results[df_results['交易方向'].astype(str).str.contains("觀望", na=False)].copy()

        if not df_long.empty and "優先次序" not in df_long.columns:
            df_long.insert(0, "優先次序", [f"#{i}" for i in range(1, len(df_long) + 1)])
        if not df_short.empty and "優先次序" not in df_short.columns:
            df_short.insert(0, "優先次序", [f"#{i}" for i in range(1, len(df_short) + 1)])
        if not df_watch.empty and "優先次序" not in df_watch.columns:
            df_watch.insert(0, "優先次序", [f"-{i}" for i in range(1, len(df_watch) + 1)])

        st.markdown("### 🏆 今日優先行動精選榜 (已按綜合優勢由 #1 排序，只列 R:R ≥ 1.8R 頂級機會)")
        col_top_l, col_top_s = st.columns(2)
        with col_top_l:
            st.markdown("#### 🟢 優先買入 (Priority Buy) Top 名單")
            if not df_long.empty and "_raw_rr" in df_long.columns and "信號狀態" in df_long.columns:
                top_long = df_long[(df_long["_raw_rr"] >= 1.8) & (~df_long["信號狀態"].astype(str).str.contains("⛔"))].head(8)
            else:
                top_long = df_long.head(8)
            if not top_long.empty:
                for idx_r, (_, r_row) in enumerate(top_long.iterrows(), 1):
                    st.success(
                        f"**第 {idx_r} 優先：{r_row['代碼']} {r_row['股票名稱']}** ｜ 現價 **{r_row['最新價']}** ｜ 綜合分 **{r_row.get('優先總分', '-')}**\n\n"
                        f"- 🎯 **在哪裡買入**：**{r_row['建議進場價']}** — *{r_row.get('掛單與買入位說明', '-')}*\n"
                        f"- 🛡️ **結構止損位**：**{r_row['ATR防守止損']}** — *{r_row.get('止損依據', '-')}*\n"
                        f"- 💰 **結構止盈位**：**{r_row['ATR目標止盈']}** — *{r_row.get('止盈阻力依據', '-')}*\n"
                        f"- ⚖️ **真實結構盈虧比**：**{r_row.get('預估盈虧比(R:R)', '-')}** ｜ 優勢：{r_row.get('META多重優勢', '-')} / {r_row.get('LEI頂底/2B', '-')}"
                    )
            else:
                st.info("目前無同時滿足「多頭觸發 + 結構盈虧比 ≥ 1.8R」的優先買入股票，建議耐心等待。")

        with col_top_s:
            st.markdown("#### 🔴 優先沽空 (Priority Short) Top 名單")
            if not df_short.empty and "_raw_rr" in df_short.columns and "信號狀態" in df_short.columns:
                top_short = df_short[(df_short["_raw_rr"] >= 1.8) & (~df_short["信號狀態"].astype(str).str.contains("⛔"))].head(8)
            else:
                top_short = df_short.head(8)
            if not top_short.empty:
                for idx_r, (_, r_row) in enumerate(top_short.iterrows(), 1):
                    st.error(
                        f"**第 {idx_r} 優先：{r_row['代碼']} {r_row['股票名稱']}** ｜ 現價 **{r_row['最新價']}** ｜ 綜合分 **{r_row.get('優先總分', '-')}**\n\n"
                        f"- 🎯 **在哪裡沽空**：**{r_row['建議進場價']}** — *{r_row.get('掛單與買入位說明', '-')}*\n"
                        f"- 🛡️ **結構止損位**：**{r_row['ATR防守止損']}** — *{r_row.get('止損依據', '-')}*\n"
                        f"- 💰 **結構回補位**：**{r_row['ATR目標止盈']}** — *{r_row.get('止盈阻力依據', '-')}*\n"
                        f"- ⚖️ **真實結構盈虧比**：**{r_row.get('預估盈虧比(R:R)', '-')}** ｜ 優勢：{r_row.get('META多重優勢', '-')} / {r_row.get('LEI頂底/2B', '-')}"
                    )
            else:
                st.info("目前無同時滿足「空頭觸發 + 結構盈虧比 ≥ 1.8R」的優先沽空股票。")

        st.markdown("---")
        st.markdown("#### ⚡ 快速無縫跳轉診斷 (揀選上方 #1, #2 股票一鍵查看 K 線買賣位)")
        all_tickers = []
        if not df_long.empty: all_tickers.extend(df_long['代碼'].tolist())
        if not df_short.empty: all_tickers.extend(df_short['代碼'].tolist())
        if not df_watch.empty: all_tickers.extend(df_watch['代碼'].tolist())
        
        if all_tickers:
            c1, c2 = st.columns([3, 1])
            with c1:
                jump_ticker = st.selectbox("選擇要診斷的標的:", all_tickers, index=None, placeholder="點擊此處輸入或搜尋代碼...", key=f"jump_sel_{market_key}")
            with c2:
                st.write("")
                if st.button("🚀 進入診斷", key=f"jump_btn_{market_key}"):
                    if jump_ticker:
                        st.query_params["mode"] = "診斷"
                        st.query_params["ticker"] = jump_ticker
                        st.query_params["strict"] = str(strictness)
                        _matched_rows = df_results[df_results['代碼'] == jump_ticker]
                        if not _matched_rows.empty:
                            _dir_txt = str(_matched_rows.iloc[0].get('交易方向', ''))
                            st.query_params["dir"] = "SHORT" if "做空" in _dir_txt else ("LONG" if "做多" in _dir_txt else "WATCH")
                        st.rerun()
                    else:
                        st.warning("請先選擇或輸入代碼")
        st.markdown("---")

        show_expert_cols = st.checkbox("🔬 顯示全部專家技術欄位 (預設關閉，只顯示最核心買賣位與盈虧比)", value=False, key=f"exp_cols_{market_key}")
        simple_cols = [
            "優先次序", "跳轉診斷", "代碼", "股票名稱", "交易方向", "優先總分",
            "最新價", "建議進場價", "掛單與買入位說明", "ATR防守止損", "ATR目標止盈",
            "預估盈虧比(R:R)", "META多重優勢", "J Law 強勢形態", "診斷結果"
        ]

        def _prepare_display_df(sub_df):
            clean_df = sub_df.drop(columns=['_raw_rr'], errors='ignore')
            if not show_expert_cols:
                avail_cols = [c for c in simple_cols if c in clean_df.columns]
                return clean_df[avail_cols]
            return clean_df

        st.markdown("#### 🟢 做多 (Long) 完整排名列表 (已按 #1, #2, #3 優先次序排列)")
        if not df_long.empty:
            render_styled_dataframe(_prepare_display_df(df_long))
        else:
            st.info("目前沒有符合做多條件的標的。")

        st.write("")

        st.markdown("#### 🔴 做空 (Short) 完整排名列表 (已按 #1, #2, #3 優先次序排列)")
        if not df_short.empty:
            render_styled_dataframe(_prepare_display_df(df_short))
        else:
            st.info("目前沒有符合做空條件的標的。")

        if not df_watch.empty:
            with st.expander("⚪ 觀望名單 (未觸發或已觸及止損失效)", expanded=False):
                render_styled_dataframe(_prepare_display_df(df_watch))

# ==========================================
# 11. 全景雷達掃描
# ==========================================
def run_all_inclusive_radar(stock_list):
    progress_bar = st.progress(0)
    results = []

    for idx, ticker in enumerate(stock_list):
        time.sleep(0.1)
        res = analyze_stock_master(ticker, strictness_threshold=strictness, interval="1d")

        if isinstance(res, dict) and "error" in res:
            pass
        elif res is not None:
            df, status, entry, sl, tp, ptype, chk, passcount, bname, l52, h52, cur, name, schk, shortpasscount, longwin, shortwin, lt, stt, jlchk, jlpass, jltitle, is_near_20ma, is_kk_htf, weekly_alignment, ms_metrics, ms_score, jlaw_alert, dist_days, is_deduction_pullback = res
            last = df.iloc[-1]

            if ptype == "LONG":
                tactical_dir = "🟢 做多"
                display_win_rate = f"{longwin:.1f}% ({lt}筆)"
                entry_str, sl_str, tp_str = f"{entry:.2f}", f"{sl:.2f}", f"{tp:.2f}"
            elif ptype == "SHORT":
                tactical_dir = "🔴 做空"
                display_win_rate = f"{shortwin:.1f}% ({stt}筆)"
                entry_str, sl_str, tp_str = f"{entry:.2f}", f"{sl:.2f}", f"{tp:.2f}"
            else:
                tactical_dir = "⚪ 觀望"
                display_win_rate = f"多 {longwin:.0f}% / 空 {shortwin:.0f}%"
                entry_str, sl_str, tp_str = "-", "-", "-"

            dist52w = last.get('Dist_to_52W_High', last.get('Distto52WHigh', 999))
            dist_str = f"{dist52w:.1f}" if pd.notna(dist52w) and dist52w != 999 else "-"
            jl_display = "🔥 KK" if is_kk_htf else ("🌟 J Law" if jlpass else "-")
            rs_strength = float(last.get('Mansfield_RS', 0))

            # [NEW v24.14] 提取精準結構買入位說明、止損依據、多層阻力止盈依據與每筆獨立盈虧比
            _tp_eval = df.attrs.get("trade_plan_eval", {})
            _inv_info = df.attrs.get("invalidation_info", {})
            _lei_st = df.attrs.get("lei_struct_info", {})
            _jlaw_m = df.attrs.get("jlaw_meta_info", {})
            _planned_rr = float(_tp_eval.get("planned_rr", 0.0))
            _struct_rr = float(_tp_eval.get("structural_rr", 0.0))
            _tp2_rr = float(_tp_eval.get("tp2_rr", 0.0))
            _stop_pct = float(_tp_eval.get("stop_pct", 0.0))
            _entry_desc = _tp_eval.get("entry_type_desc", "-")
            _sl_desc = f"{_tp_eval.get('sl_basis_desc', '-')} (風險 -{_stop_pct:.1f}%)" if ptype in ["LONG", "SHORT"] else "-"
            _tp_desc = _tp_eval.get("tp_basis_desc", "-")
            _rr_str = f"{_planned_rr:.2f}R (次級 {_tp2_rr:.2f}R)" if ptype in ["LONG", "SHORT"] else "-"
            _is_fresh_trigger = int(any(k in status for k in ["🔥", "🟢", "🔴", "🏴‍☠️", "🩸"]))
            _meta_score_val = _jlaw_m.get("meta_long_score", 0) if ptype == "LONG" else (_jlaw_m.get("meta_short_score", 0) if ptype == "SHORT" else 0)

            _rr_bonus = min(max(_planned_rr, 0.0), 6.0) * 4.5
            _fresh_bonus = 25.0 if _is_fresh_trigger else 0.0
            _meta_bonus = _meta_score_val * 7.0
            _trend_bonus = (passcount if ptype == "LONG" else shortpasscount) * 2.5
            _rs_bonus = min(max(abs(rs_strength) * 2.0, 0.0), 12.0)
            _rr_penalty = -25.0 if _planned_rr < 1.8 else 0.0
            _priority_score = round(min(99.9, max(10.0, _fresh_bonus + _meta_bonus + _rr_bonus + _trend_bonus + _rs_bonus + _rr_penalty)), 1) if ptype in ["LONG", "SHORT"] else 0.0

            _sig_store = persistent_data.get("signal_lifecycle_store", {})
            _existing_sig = _sig_store.get(ticker)
            _lifecycle_status = "✅ 有效"
            if _existing_sig and _existing_sig.get("plan_type") == "LONG":
                if float(last['Low']) <= float(_existing_sig["sl"]):
                    _lifecycle_status = f"⛔ 已觸及前次止損 ({_existing_sig['sl']:.2f})"
                    if ptype == "LONG":
                        tactical_dir = "⚪ 觀望"
                        status = f"⚪ 觀望 (已跌穿 {_existing_sig['scan_date']} 止損位 {_existing_sig['sl']:.2f})"
                        entry_str, sl_str, tp_str, _rr_str = "-", "-", "-", "-"
                        _priority_score = 0.0
            elif _existing_sig and _existing_sig.get("plan_type") == "SHORT":
                if float(last['High']) >= float(_existing_sig["sl"]):
                    _lifecycle_status = f"⛔ 已觸及前次空頭止損 ({_existing_sig['sl']:.2f})"
                    if ptype == "SHORT":
                        tactical_dir = "⚪ 觀望"
                        status = f"⚪ 觀望 (已升穿 {_existing_sig['scan_date']} 空頭止損 {_existing_sig['sl']:.2f})"
                        entry_str, sl_str, tp_str, _rr_str = "-", "-", "-", "-"
                        _priority_score = 0.0

            if tactical_dir in ["🟢 做多", "🔴 做空"] and (_existing_sig is None or _existing_sig.get("plan_type") != ptype):
                _sig_store[ticker] = {
                    "plan_type": ptype,
                    "entry": float(entry),
                    "sl": float(sl),
                    "tp": float(tp),
                    "scan_date": str(last.name.date()) if hasattr(last.name, "date") else str(datetime.now().date())
                }

            results.append({
                "跳轉診斷": f"/?mode=診斷&ticker={ticker}&dir={ptype}&strict={strictness}",
                "代碼": ticker,
                "股票名稱": name,
                "最新價": f"{last['Close']:.2f}",
                "交易方向": tactical_dir,
                "優先總分": f"{_priority_score:.1f}分",
                "信號狀態": _lifecycle_status if not _inv_info.get("long_stop_hit") else "⛔ 近期止損已觸發",
                "建議進場價": entry_str,
                "掛單與買入位說明": _entry_desc if tactical_dir != "⚪ 觀望" else "-",
                "ATR防守止損": sl_str,
                "止損依據": _sl_desc,
                "ATR目標止盈": tp_str,
                "止盈阻力依據": _tp_desc if tactical_dir != "⚪ 觀望" else "-",
                "預估盈虧比(R:R)": _rr_str,
                "META多重優勢": _jlaw_m.get("meta_badge", "-"),
                "LEI頂底/2B": _lei_st.get("structure_summary", "-"),
                "大戶出貨日次數": int(dist_days) if pd.notna(dist_days) else 0,
                "J Law 強勢形態": jl_display,
                "距52W高點(%)": dist_str,
                "RS強度": round(rs_strength, 2),
                "全歷史勝率": display_win_rate,
                "多頭分": passcount,
                "空頭分": shortpasscount,
                "成交量": f"{int(last['Volume']):,}" if pd.notna(last['Volume']) else "-",
                "VCP壓縮": "🔥 收窄" if (last['VCPSignal'] or last.get('JLaw_Tight_Box', False)) else "正常",
                "診斷結果": status,
                "_priority_num": _priority_score,
                "_raw_rr": _planned_rr,
                "_is_fresh_trigger": _is_fresh_trigger,
                "_meta_score": _meta_score_val,
                "_vol_ratio": float(last.get('VolRatio', 0.0)),
                "_above_50ma": int(pd.notna(last.get('SMA50')) and float(last['Close']) > float(last['SMA50'])),
                "_above_200ma": int(pd.notna(last.get('SMA200')) and float(last['Close']) > float(last['SMA200']))
            })

        progress_bar.progress((idx + 1) / len(stock_list))

    progress_bar.empty()

    if results:
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            if "J Law 強勢形態" in df_results.columns and "全歷史勝率" in df_results.columns:
                df_results['sort_win_rate'] = df_results['全歷史勝率'].str.extract(r'([0-9]+\.[0-9]+)').astype(float).fillna(0)
                df_results['sort_jlaw'] = df_results['J Law 強勢形態'].str.contains('✅|🔥|🌟').astype(int)
                df_results.attrs["pct_above_50ma"] = float(df_results['_above_50ma'].mean() * 100.0) if '_above_50ma' in df_results.columns else 0.0
                df_results.attrs["pct_above_200ma"] = float(df_results['_above_200ma'].mean() * 100.0) if '_above_200ma' in df_results.columns else 0.0
                # OLD: df_results = df_results.sort_values(by=['sort_jlaw', 'sort_win_rate', 'RS強度'], ascending=[False, False, False])
                # NEW: [v24.14] 直接按「綜合優先總分 (_priority_num)」、「當日新鮮觸發 (_is_fresh_trigger)」、「真實結構盈虧比 (_raw_rr)」排序！
                df_results = df_results.sort_values(
                    by=['_priority_num', '_is_fresh_trigger', '_meta_score', '_raw_rr', 'RS強度'],
                    ascending=[False, False, False, False, False]
                )
                df_results = df_results.drop(
                    columns=['sort_win_rate', 'sort_jlaw', '_priority_num', '_is_fresh_trigger', '_meta_score', '_vol_ratio', '_above_50ma', '_above_200ma'],
                    errors='ignore'
                )
            else:
                df_results = df_results.sort_values(by=["RS強度", "多頭分", "空頭分"], ascending=[False, False, False])
            
            return df_results

    return None

# ==========================================
# 12. 模式 1：雷達
# ==========================================
if mode == mode_options[0]:
    st.title("🔎 市場動能掃描雷達")
    st.markdown("💡 **提示：** 系統已智能分辨美/港股。")

    quick_search = st.text_input("🔍 即時個股雷達掃描 (輸入代碼即時檢視):", placeholder="例: AAPL").strip()
    if quick_search:
        q_df = run_all_inclusive_radar([quick_search])
        if q_df is not None:
            st.markdown("#### ⚡ 即時掃描結果")
            render_styled_dataframe(q_df)
        st.markdown("---")

    # [優化加入] 將自訂名單放入雷達掃描選項
    pool_choice = st.selectbox("核心追蹤池：", ["港股高動能增長池 (近100隻強勢成分股)", "美股高動能爆發池", "自訂擴充名單 (從側欄載入)"])
    
    target_pool = []
    if "港股" in pool_choice:
        target_pool = HK_MOMENTUM_POOL
    elif "美股" in pool_choice:
        target_pool = US_MOMENTUM_POOL
    else:
        if custom_tickers_raw:
            import re
            target_pool = [x.strip().upper() for x in re.split(r'[\s,]+', custom_tickers_raw) if x.strip()]
        else:
            st.warning("⚠️ 自訂名單為空，請在左側欄位輸入股票代碼！")

    if st.button("🚀 啟動全景量化掃描") and target_pool:
        with st.spinner("掃描中 (資料池已擴大，保證產出 20~50 隻符合條件標的，請稍候)..."):
            df_res = run_all_inclusive_radar(target_pool)
            if df_res is not None:
                if "自訂" in pool_choice:
                    dict_key = "自訂"
                elif "港股" in pool_choice:
                    dict_key = "港股"
                else:
                    dict_key = "美股"
                    
                st.session_state.radar_results[dict_key] = df_res
                # [NEW INCREMENTAL PATCH 修復 Bug 3 & J Law 市場寬度記錄]
                mark_radar_cache_updated(persistent_data, dict_key)
                persistent_data.setdefault("radar_breadth_stats", {})[dict_key] = {
                    "pct_above_50ma": df_res.attrs.get("pct_above_50ma", 0.0),
                    "pct_above_200ma": df_res.attrs.get("pct_above_200ma", 0.0)
                }

                active_df = df_res[df_res['交易方向'].astype(str).str.contains("做多|做空", na=False)].copy()
                rs_map = dict(zip(active_df['代碼'], active_df['RS強度']))
                warnings, corr_matrix = build_dynamic_correlation_report(
                    list(active_df['代碼']),
                    rs_map=rs_map,
                    lookback=50,
                    threshold=0.75
                )

                st.session_state.radar_correlation_warnings[dict_key] = warnings
                st.session_state.radar_correlation_matrices[dict_key] = corr_matrix
            else:
                st.error("⚠️ 未取得任何有效數據，所有股票可能未能符合當前系統的過濾標準。")

    if st.session_state.radar_results.get("港股") is not None:
        render_radar_results(st.session_state.radar_results["港股"], "### 📋 港股雷達結果 (暫存)", market_key="港股")
        st.markdown("---")

    if st.session_state.radar_results.get("美股") is not None:
        render_radar_results(st.session_state.radar_results["美股"], "### 📋 美股雷達結果 (暫存)", market_key="美股")
        st.markdown("---")
        
    if st.session_state.radar_results.get("自訂") is not None:
        render_radar_results(st.session_state.radar_results["自訂"], "### 📋 自訂名單雷達結果 (暫存)", market_key="自訂")

# ==========================================
# 🌟 13. 模式 3：自選股 (仿富途牛牛多重分組優化版)
# ==========================================
elif mode == mode_options[2]:
    st.query_params.clear()
    st.title("⭐ 自選股動能面板 (專業分組版)")
    st.markdown("管理你的核心觀察名單，可仿效富途牛牛建立多個分類標籤 (如：科技股、收息股、觀察中)，並分組進行雷達掃描。")

    # --- 管理分組 ---
    with st.expander("📁 管理分組 (新增 / 刪除分類)", expanded=False):
        col_add_cat, col_del_cat = st.columns(2)
        with col_add_cat:
            new_cat = st.text_input("➕ 新增分組名稱", placeholder="例如: 港股精選, 潛力妖股...")
            if st.button("建立分組"):
                if new_cat and new_cat not in st.session_state.watchlist_dict:
                    st.session_state.watchlist_dict[new_cat] = []
                    st.success(f"已建立分組: {new_cat}")
                    time.sleep(0.5)
                    st.rerun()
                elif new_cat in st.session_state.watchlist_dict:
                    st.warning("分組名稱已存在！")
        with col_del_cat:
            cat_to_del = st.selectbox("🗑️ 刪除分組", ["-- 請選擇 --"] + list(st.session_state.watchlist_dict.keys()))
            if st.button("確認刪除分組"):
                if cat_to_del != "-- 請選擇 --" and len(st.session_state.watchlist_dict) > 1:
                    del st.session_state.watchlist_dict[cat_to_del]
                    if cat_to_del in st.session_state.watchlist_radar_results:
                        del st.session_state.watchlist_radar_results[cat_to_del]
                    st.success(f"已刪除分組: {cat_to_del}")
                    time.sleep(0.5)
                    st.rerun()
                elif len(st.session_state.watchlist_dict) <= 1:
                    st.error("⚠️ 至少需要保留一個分組！")

    st.markdown("---")
    
    # --- 渲染分組 Tab 介面 ---
    cat_names = list(st.session_state.watchlist_dict.keys())
    tabs = st.tabs(cat_names)

    for i, cat in enumerate(cat_names):
        with tabs[i]:
            col_add, col_del = st.columns(2)
            with col_add:
                # 使用唯一的 key 防止 Streamlit 報錯
                add_ticker = st.text_input(f"➕ 輸入代碼加入「{cat}」", placeholder="例如: MSFT, 0700.HK", key=f"add_{cat}").strip()
                if add_ticker:
                    fmt, _, _, _, _ = format_and_detect_market(add_ticker)
                    if fmt not in st.session_state.watchlist_dict[cat]:
                        st.session_state.watchlist_dict[cat].append(fmt)
                        st.success(f"已加入 {fmt} 到 {cat}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning(f"{fmt} 已存在於 {cat}")

            with col_del:
                current_list = st.session_state.watchlist_dict[cat]
                if current_list:
                    del_ticker = st.selectbox(f"🗑️ 從「{cat}」移除", ["-- 請選擇 --"] + current_list, key=f"del_{cat}")
                    if del_ticker != "-- 請選擇 --":
                        st.session_state.watchlist_dict[cat].remove(del_ticker)
                        st.success(f"已移除 {del_ticker}")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.info("此分組目前沒有自選股")

            st.markdown("---")
            if current_list:
                st.write(f"📋 **{cat}** 當前名單：", ", ".join(current_list))
                if st.button(f"🚀 掃描「{cat}」板塊", use_container_width=True, key=f"scan_{cat}"):
                    with st.spinner(f"正在獨立掃描 {cat} 板塊..."):
                        df_res = run_all_inclusive_radar(current_list)
                        if df_res is not None:
                            st.session_state.watchlist_radar_results[cat] = df_res
                            
                            # 針對該板塊獨立計算動態相關性
                            active_df = df_res[df_res['交易方向'].astype(str).str.contains("做多|做空", na=False)].copy()
                            rs_map = dict(zip(active_df['代碼'], active_df['RS強度']))
                            warnings, corr_matrix = build_dynamic_correlation_report(
                                list(active_df['代碼']),
                                rs_map=rs_map,
                                lookback=50,
                                threshold=0.75
                            )
                            # 將相關性數據存入特定的 key
                            st.session_state.radar_correlation_warnings[f"自選_{cat}"] = warnings
                            st.session_state.radar_correlation_matrices[f"自選_{cat}"] = corr_matrix
                            
                        else:
                            st.error("掃描失敗，未取得有效數據。")
            else:
                st.warning("此分組列表為空，請先加入股票。")

            # 獨立渲染每個分組的雷達結果
            if cat in st.session_state.watchlist_radar_results and st.session_state.watchlist_radar_results[cat] is not None:
                render_radar_results(st.session_state.watchlist_radar_results[cat], f"### ⭐ 「{cat}」板塊掃描結果", market_key=f"自選_{cat}")

# ==========================================
# 14. 模式 2：個股診斷
# ==========================================
elif mode == mode_options[1]:
    st.title("📊 個股量化分析與技術診斷")

    c_ticker, c_tf = st.columns([3, 1])
    with c_ticker:
        ticker_input = st.text_input("輸入股票代碼 (例: 3690、700、NVDA):", value=target_ticker).strip()
    with c_tf:
        tf_choice = st.selectbox("週期", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)

    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "1d": "1d", "1wk": "1wk"}
    selected_interval = interval_map[tf_choice]

    if ticker_input:
        fmt_ticker, market_type, _, benchmark_name, _ = format_and_detect_market(ticker_input)
        res = analyze_stock_master(ticker_input, strictness_threshold=strictness, interval=selected_interval)

        if isinstance(res, dict) and "error" in res:
            st.error(f"❌ {res['error']}")
        elif res is None:
            st.error("❌ 分析失敗")
        else:
            df, status, target_entry, stop_loss, take_profit, plan_type, checklist, pass_count, benchmark_name, low_52w, high_52w, currency, name, short_checklist, short_pass_count, long_win, short_win, l_t, s_t, jl_chk, jl_pass, jl_title, is_near_20ma, is_kk_htf, weekly_alignment, ms_metrics, ms_score, jlaw_alert, dist_days, is_deduction_pullback = res
            last = df.iloc[-1]

            st.markdown(f"## 💎 {ticker_input} - {name} ({tf_choice})")

            # 斜率時鐘智能診斷儀表板
            st.markdown("### ⏱️ 斜率時鐘智能診斷 (趨勢姿態判定)")
            col_clock1, col_clock2, col_clock3 = st.columns(3)
            
            with col_clock1:
                if last.get('Clock_12', False):
                    st.error("🚀 **12點鐘方向 (加速過熱)**\n\n乖離過大，慎防見頂回落，切勿追高。")
                else:
                    st.info("⚪ 12點鐘方向：未觸發")
                    
            with col_clock2:
                if last.get('Clock_2', False):
                    st.success("📈 **2點鐘方向 (穩定上揚)**\n\n多頭平行發散，最佳吃回撤行情。")
                else:
                    st.info("⚪ 2點鐘方向：未觸發")
                    
            with col_clock3:
                if last.get('Clock_3', False):
                    st.warning("↔️ **3點鐘方向 (橫盤壓縮)**\n\n均線極度黏合，等待大方向突破。")
                else:
                    st.info("⚪ 3點鐘方向：未觸發")
                    
            # 雙均線 (MA+EMA) 趨勢診斷
            st.markdown("### 🎯 雙均線 (MA+EMA) 趨勢診斷")
            st.caption("EMA (指數均線) 更為敏感，MA (簡單均線) 較為穩定。當兩者共振向上時，趨勢最為明確。")
            col_ema1, col_ema2, col_ema3 = st.columns(3)
            
            ema20, ma20 = last.get('EMA20', 0), last.get('SMA20', 0)
            if ema20 > ma20 and last['Close'] > ema20:
                col_ema1.success(f"🟢 短期多頭共振 (20日)\n\nEMA ({ema20:.2f}) > MA ({ma20:.2f})")
            elif ema20 < ma20 and last['Close'] < ema20:
                col_ema1.error(f"🔴 短期空頭共振 (20日)\n\nEMA ({ema20:.2f}) < MA ({ma20:.2f})")
            else:
                col_ema1.warning(f"🟡 短期震盪交纏 (20日)\n\nEMA ({ema20:.2f}) / MA ({ma20:.2f})")
                
            ema50, ma50 = last.get('EMA50', 0), last.get('SMA50', 0)
            if ema50 > ma50 and last['Close'] > ema50:
                col_ema2.success(f"🟢 中期多頭共振 (50日)\n\nEMA ({ema50:.2f}) > MA ({ma50:.2f})")
            elif ema50 < ma50 and last['Close'] < ema50:
                col_ema2.error(f"🔴 中期空頭共振 (50日)\n\nEMA ({ema50:.2f}) < MA ({ma50:.2f})")
            else:
                col_ema2.warning(f"🟡 中期震盪交纏 (50日)\n\nEMA ({ema50:.2f}) / MA ({ma50:.2f})")
                
            ema120, ma120 = last.get('EMA120', 0), last.get('SMA120', 0)
            if ema120 > ma120 and last['Close'] > ema120:
                col_ema3.success(f"🟢 長期多頭共振 (120日)\n\nEMA ({ema120:.2f}) > MA ({ma120:.2f})")
            elif ema120 < ma120 and last['Close'] < ema120:
                col_ema3.error(f"🔴 長期空頭共振 (120日)\n\nEMA ({ema120:.2f}) < MA ({ma120:.2f})")
            else:
                col_ema3.warning(f"🟡 長期震盪交纏 (120日)\n\nEMA ({ema120:.2f}) / MA ({ma120:.2f})")

            if selected_interval == "1d":
                wk = weekly_alignment.get("detail", {})
                if weekly_alignment.get("warning"):
                    st.warning(f"🌐 **三重濾網大週期共振監測燈號：** {weekly_alignment['warning']}")
                else:
                    st.success("🌐 **三重濾網大週期共振監測燈號：** 週線與日線方向大致一致。")

            if '🟢' in status or '🔥' in status:
                st.success(status)
            elif '🔴' in status or '🟠' in status or '🩸' in status or '🏴‍☠️' in status:
                st.error(status)
            elif '🟡' in status:
                st.warning(status)
            else:
                st.info(status)

            st.markdown("### 🌟 J Law / Minervini / 空頭濾網")
            c_left, c_mid, c_right, c_short = st.columns(4)

            with c_left:
                st.markdown(f"#### {jl_title}")
                st.caption("J Law 核心條件")
                for criterion, passed in jl_chk.items():
                    st.checkbox(criterion, value=passed, disabled=True, key=f"jlchk_{criterion}")

            with c_mid:
                st.markdown(f"#### Minervini 趨勢條件 ({pass_count}/7)")
                st.caption(f"做多歷史勝率：{long_win:.1f}%")
                for criterion, passed in checklist.items():
                    st.checkbox(criterion, value=passed, disabled=True, key=f"longchk_{criterion}")

            with c_right:
                st.markdown(f"#### 空頭趨勢條件 ({short_pass_count}/7)")
                st.caption(f"做空歷史勝率：{short_win:.1f}%")
                for criterion, passed in short_checklist.items():
                    st.checkbox(criterion, value=passed, disabled=True, key=f"shortchk_{criterion}")
                    
            with c_short:
                st.markdown(f"#### 🏴‍☠️ 大師做空法 ({ms_score}/4)")
                st.caption("O'Neil & Cohodes 實戰")
                st.checkbox("大盤處於熊市/跌破50MA", value=ms_metrics["market_bear"], disabled=True)
                st.checkbox("前高跌落且反彈受阻 50MA", value=ms_metrics["rebound_resist"], disabled=True)
                st.checkbox("反彈量能嚴重萎縮", value=ms_metrics["low_vol_rebound"], disabled=True)
                st.checkbox("Cohodes 右側支撐破位", value=ms_metrics["support_break"], disabled=True)

            st.markdown("---")
            rationale_col1, rationale_col2 = st.columns(2)
            # [NEW v24.14 一眼看懂：精準結構買賣位、止損止盈依據與雙大師實戰認證]
            _tp_eval = df.attrs.get("trade_plan_eval", {})
            _inv_info = df.attrs.get("invalidation_info", {})
            _lei_st = df.attrs.get("lei_struct_info", {})
            _lei_ded = df.attrs.get("lei_ded_forecast", {})
            _jlaw_m = df.attrs.get("jlaw_meta_info", {})

            st.markdown("### 🎯 一眼看懂：結構買賣執行計劃與雙大師認證 (v24.14)")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric(
                "💎 J Law META 多重優勢",
                _jlaw_m.get("meta_badge", "-"),
                f"漲跌量比 {_jlaw_m.get('up_down_vol_ratio', 1.0):.2f}x"
            )
            mc2.metric(
                "⚡ 強勁趨勢 (Power Trend)",
                "🔥 已發動" if _jlaw_m.get("is_power_trend") else "⚪ 一般趨勢",
                "1M & 6M 雙週期領先" if (_jlaw_m.get("rs_1m_outperform") and _jlaw_m.get("rs_6m_outperform")) else "RS 常規表現"
            )
            mc3.metric(
                "🧱 LEI 頂底與 2B 構造",
                _lei_st.get("structure_summary", "⚪ 無明顯構造"),
                _lei_st.get("lei_turn_step", "")
            )
            mc4.metric(
                "📐 LEI 六線密集度 (20/60/120)",
                f"{float(last.get('LEI_MA_Spread_Pct', 0.0)):.2f}%",
                "🔥 <2.5% 高度密集" if last.get('LEI_MA_Tight_2Pct', False) else f"120EMA乖離 {float(last.get('Bias_EMA120_Pct', 0.0)):+.1f}%"
            )
            if _jlaw_m.get("meta_edges_list"):
                st.caption("🎯 **當前匯聚之交易優勢 (Multiple Edges)**：" + " ｜ ".join(_jlaw_m["meta_edges_list"]))

            with rationale_col1:
                _entry_how = _tp_eval.get("entry_type_desc", "")
                if plan_type == "SHORT":
                    entry_reason = (
                        f"🔴 **【沽空執行位】建議掛單價：{target_entry:.2f} {currency}**（現價：{last['Close']:.2f}）\n\n"
                        f"📌 **進場方式**：{_entry_how}\n\n"
                        f"📊 **信號形態**：{status}"
                    )
                elif plan_type == "LONG":
                    entry_reason = (
                        f"🟢 **【買入執行位】建議掛單價：{target_entry:.2f} {currency}**（現價：{last['Close']:.2f}）\n\n"
                        f"📌 **進場方式**：{_entry_how}\n\n"
                        f"📊 **信號形態**：{status}"
                    )
                else:
                    if _inv_info.get("invalidation_reason"):
                        entry_reason = f"⚪ **【觀望 / 信號已失效】** {_inv_info['invalidation_reason']}"
                    else:
                        entry_reason = "⚪ **【觀望等待】** 目前未達高勝率觸發條件，請等待股價形成緊密平台或回測關鍵支撐位。"
                st.info(entry_reason)

            with rationale_col2:
                _struct_rr = _tp_eval.get("structural_rr", 0.0)
                _plan_rr = _tp_eval.get("planned_rr", 0.0)
                _tp2_p = _tp_eval.get("tp2", take_profit)
                _tp2_rr = _tp_eval.get("tp2_rr", 0.0)
                _stop_pct = _tp_eval.get("stop_pct", 0.0)
                _sl_basis = _tp_eval.get("sl_basis_desc", "-")
                _tp_basis = _tp_eval.get("tp_basis_desc", "-")
                _rr_warn = _tp_eval.get("rr_warning", "")
                exit_reason = (
                    f"🛡️ **結構止損位：{stop_loss:.2f} {currency}**（風險幅度 **-{_stop_pct:.2f}%** ｜ 依據：*{_sl_basis}*）\n\n"
                    f"💰 **第一結構止盈 (TP1)：{take_profit:.2f} {currency}**（真實盈虧比 **{_plan_rr:.2f} R** ｜ 依據：*{_tp_basis}*）\n\n"
                    f"🚀 **波段次級目標 (TP2)：{_tp2_p:.2f} {currency}**（潛在盈虧比 **{_tp2_rr:.2f} R** ｜ 大趨勢沿 9 EMA 移動跟蹤）"
                )
                if _rr_warn:
                    exit_reason = exit_reason + "\n\n" + _rr_warn
                st.warning(exit_reason)

            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(f"最新價 ({currency})", f"{last['Close']:.2f}")
            m2.metric("RSI 14", f"{last['RSI']:.1f}")
            m3.metric("距52W高點", f"{last['Distto52WHigh']:.1f}%")
            m4.metric("短線阻力", f"{last['ResShort']:.2f}")
            m5.metric("MACD 柱體", f"{last['MACDHist']:.3f}")

            # 抵扣價防禦判定顯示
            st.markdown("### 🧱 支撐阻力與抵扣價分析 (防假跌破)")
            st.caption("判斷真假跌破的秘訣：只要【現價】大於【抵扣價】，即使股價跌破均線，該均線未來仍會繼續向上發散。")
            sr1, sr2, sr3, sr4, sr5 = st.columns(5)
            
            deduct_20 = last.get('SMA20_Deduction', 0)
            status_20 = "✅ 均線向上" if last['Close'] > deduct_20 else "❌ 均線下彎"
            sr1.metric("🟢 短期支撐 (20日)", f"{last['SupShort']:.2f}", f"20日抵扣: {deduct_20:.2f} ({status_20})")
            
            sr2.metric("🔴 短期阻力 (20日)", f"{last['ResShort']:.2f}")
            
            deduct_50 = last.get('SMA50_Deduction', 0)
            status_50 = "✅ 均線向上" if last['Close'] > deduct_50 else "❌ 均線下彎"
            sr3.metric("🟢 中期支撐 (50日)", f"{last['SMA50']:.2f}", f"50日抵扣: {deduct_50:.2f} ({status_50})")
            
            sr4.metric("🟢 長期支撐 (60日低點)", f"{last['SupLong']:.2f}", f"60MA抵扣: {float(last.get('SMA60_Deduction', 0)):.2f}")
            sr5.metric("🔴 長期阻力 (60日高點)", f"{last['ResLong']:.2f}", f"5日箱體波幅: {float(last.get('Tight_Box_5d_Pct', 0)):.1f}%")
            st.caption(f"🔮 **LEI 未來兩週抵扣價方框推演**：20MA → {_lei_ded.get('forecast_20_label', '')} ｜ 60MA → {_lei_ded.get('forecast_60_label', '')}")

            st.markdown("### 💰 投資門檻試算")
            col_lot1, col_lot2, col_lot3 = st.columns(3)
            
            default_lot = 1 if market_type != "HK" else POPULAR_HK_LOTS.get(fmt_ticker, 100)
            
            with col_lot1:
                lot_size = st.number_input("一手股數 (自動匹配/手動修改):", min_value=1, value=default_lot, step=100 if market_type == "HK" else 1)
            
            min_invest_amount = lot_size * last['Close']
            
            with col_lot2:
                st.metric("一手最少買入金額", f"{min_invest_amount:,.2f} {currency}")
                
            with col_lot3:
                st.metric("建議總本金門檻 (以 1% 風險計算)", f"{(min_invest_amount * 100):,.2f} {currency}", help="假設單筆全倉買入 1 手，需備足此總本金方可滿足 1% 風險控制原則")
            
            st.markdown("---")

            st.markdown("### 🎯 結構支撐阻力交易計畫 (依據關鍵價位動態計算)")
            p1, p2, p3, p4, p5 = st.columns(5)
            if plan_type == "WATCH":
                p1.metric("建議掛單進場位", "觀望")
                p2.metric("結構防守止損位", "-")
                p3.metric("第一結構止盈 (TP1)", "-")
                p4.metric("真實結構盈虧比", "-")
                p5.metric("交易方向", "⚪ 觀望")
            else:
                _entry_diff_pct = (target_entry - last['Close']) / last['Close'] * 100.0
                p1.metric("🎯 建議掛單進場位", f"{target_entry:.2f} {currency}", delta=f"距現價 {_entry_diff_pct:+.2f}%")
                sl_diff = stop_loss - target_entry
                p2.metric("🛡️ 結構防守止損位", f"{stop_loss:.2f} {currency}", delta=f"{sl_diff:+.2f} (-{_tp_eval.get('stop_pct', 0):.1f}%)", delta_color="inverse")
                tp_diff = take_profit - target_entry
                p3.metric("💰 第一結構止盈 (TP1)", f"{take_profit:.2f} {currency}", delta=f"{tp_diff:+.2f}")
                p4.metric("⚖️ 真實結構盈虧比", f"{_tp_eval.get('planned_rr', 0):.2f} R", delta=f"次級目標 {_tp_eval.get('tp2_rr', 0):.2f} R")
                p5.metric("🔄 交易方向", "🟢 做多" if plan_type == "LONG" else "🔴 做空")

            # [UI 更新] 完全植入動態強勢鎖潤提示與保本提醒
            st.markdown("### 🛡️ 實戰持倉防守與風控導航")
            st.info("💡 輸入您的實際進場價，系統將根據 **9 EMA** 的狀態，即時判斷您處於 **[忍耐洗盤期]** 還是 **[安全保本期]**，並為您制定最優的分批止盈策略！")
            
            col_pos1, col_pos2 = st.columns(2)
            with col_pos1:
                actual_entry = st.number_input("📌 你的實際進場價 (輸入 0 則隱藏此面板，圖表恢復預設):", value=0.0, step=0.1)
            with col_pos2:
                # OLD: pos_dir = st.selectbox("🔄 你的持倉方向:", ["做多 (Long)", "做空 (Short)"])
                # NEW: [修復 Bug 1B] 自動根據當前診斷出的 plan_type 或雷達跳轉參數預設持倉方向 (SHORT 預設選中 "做空 (Short)")，不再寫死 index=0！
                default_pos_idx = 1 if (plan_type == "SHORT" or target_dir_from_radar == "SHORT") else 0
                pos_dir = st.selectbox("🔄 你的持倉方向:", ["做多 (Long)", "做空 (Short)"], index=default_pos_idx, key=f"pos_dir_{fmt_ticker}_{plan_type}")
                
            chart_entry = target_entry
            chart_sl = stop_loss
            chart_tp = take_profit
            chart_plan = plan_type
                
            if actual_entry > 0:
                entry_atr = last['ATR'] if pd.notna(last['ATR']) else actual_entry * 0.03
                bench_close = last.get('CloseBench', np.nan)
                bench_sma50 = last.get('SMA50BenchValue', np.nan)
                ema9_val = last.get('EMA9', 0)
                
                if pos_dir == "做多 (Long)":
                    fixed_sl = actual_entry - 2 * entry_atr
                    
                    # 止盈優先取「前阻力位」，但若距離太近，強制拉大到 4 ATR (確保 2R 盈虧比)
                    fixed_tp = max(last.get('ResShort', actual_entry * 1.08), actual_entry + 4 * entry_atr)
                    
                    is_trending = pd.notna(bench_close) and bench_close > bench_sma50 * 1.015
                    breakeven_triggered = ema9_val > actual_entry
                    current_sl = actual_entry if breakeven_triggered else fixed_sl
                    
                    st.success(f"**🟢 實戰計畫 (做多)**：實際進場 **{actual_entry:.2f}** ｜ 目標前阻力 (保底2R) **{fixed_tp:.2f}**")
                    
                    # [NEW INCREMENTAL PATCH 修復 Bug 5C] 優先檢查現價或當日最低價是否已經跌穿止損價！
                    if last['Low'] <= current_sl or last['Close'] <= current_sl:
                        st.error(f"🚨 **【硬止損已觸發 / 立即離場警告】**：最新價 ({last['Close']:.2f}) 或日內最低價 ({last['Low']:.2f}) **已經跌穿防守止損價 ({current_sl:.2f})**！此筆做多交易已經失效，請嚴格執行止損紀律，切勿繼續死扛！")
                    elif breakeven_triggered:
                        st.error(f"**🛡️ 動態保本已觸發**：目前 9 EMA ({ema9_val:.2f}) 已升穿買入價！建議將硬止損上移至保本價 **{actual_entry:.2f}**，鎖定無風險狀態！")
                    else:
                        st.warning(f"**🌱 潛伏洗盤期 (忍耐)**：目前 9 EMA 尚未越過成本線。請堅守初始止損 **{fixed_sl:.2f}**，切勿因為盤中短期跌破 9EMA 而提早被洗出場。")
                        
                    st.warning(f"**🎯 階段一：強勢鎖潤 (Sell Half into Strength)**：目標價 **{fixed_tp:.2f}**。觸及此價位時，請務必【賣出 50%】倉位落袋為安，並將剩餘倉位止損移至保本價 ({actual_entry:.2f})。")
                    
                    if is_trending:
                        st.info(f"**🏃 階段二：讓利潤奔跑 (Trailing Stop)**：剩餘的 50% 倉位，沿著 **9 EMA ({ema9_val:.2f})** 進行跟蹤止盈，直到收盤跌破 9 EMA 且跌破上升趨勢線 **{last.get('SupShort', 0):.2f}** 才全數離場，爭取利潤最大化！")
                    else:
                        st.info(f"**↔️ 震盪市區間 (Range-bound)**：大盤動能偏弱，建議採固定止盈，觸及目標價 **{fixed_tp:.2f}** 附近分批落袋為安。")
                    
                    chart_entry, chart_sl, chart_tp, chart_plan = actual_entry, current_sl, fixed_tp, "LONG"
                else:
                    fixed_sl = actual_entry + 2 * entry_atr
                    fixed_tp = min(last.get('SupShort', actual_entry * 0.92), actual_entry - 4 * entry_atr)
                    
                    is_trending = pd.notna(bench_close) and bench_close < bench_sma50 * 0.985
                    breakeven_triggered = ema9_val < actual_entry
                    current_sl = actual_entry if breakeven_triggered else fixed_sl
                    
                    st.error(f"**🔴 實戰計畫 (做空)**：實際進場 **{actual_entry:.2f}** ｜ 目標前支撐 (保底2R) **{fixed_tp:.2f}**")
                    
                    # [NEW INCREMENTAL PATCH 修復 Bug 5C] 優先檢查現價或當日最高價是否已經升穿空頭止損價！
                    if last['High'] >= current_sl or last['Close'] >= current_sl:
                        st.error(f"🚨 **【空頭止損已觸發 / 立即回補警告】**：最新價 ({last['Close']:.2f}) 或日內最高價 ({last['High']:.2f}) **已經升穿空頭防守止損價 ({current_sl:.2f})**！此筆做空交易已經失效，請立即回補止損，慎防軋空！")
                    elif breakeven_triggered:
                        st.success(f"**🛡️ 動態保本已觸發**：目前 9 EMA ({ema9_val:.2f}) 已跌穿買入價！建議將硬止損下移至保本價 **{actual_entry:.2f}**，鎖定無風險狀態！")
                    else:
                        st.warning(f"**🌱 潛伏洗盤期 (忍耐)**：目前 9 EMA 尚未跌穿成本線。請堅守初始止損 **{fixed_sl:.2f}**，切勿因為盤中反彈而提早出場。")

                    st.warning(f"**🎯 階段一：強勢鎖潤 (Cover Half into Weakness)**：目標價 **{fixed_tp:.2f}**。觸及此價位時，請務必【回補 50%】倉位落袋為安，並將剩餘倉位止損移至保本價 ({actual_entry:.2f})。")

                    if is_trending:
                        st.info(f"**🏃 階段二：讓利潤奔跑 (Trailing Stop)**：剩餘的 50% 倉位，沿著 **9 EMA ({ema9_val:.2f})** 進行跟蹤，直到收盤升穿 9 EMA 且升穿下降趨勢線 **{last.get('ResShort', 0):.2f}** 才全數離場！")
                    else:
                        st.info(f"**↔️ 震盪市區間 (Range-bound)**：大盤動能不明，觸及目標價 **{fixed_tp:.2f}** 建議全數回補落袋為安。")
                    
                    chart_entry, chart_sl, chart_tp, chart_plan = actual_entry, current_sl, fixed_tp, "SHORT"
            st.markdown("---")
            
            # J Law 出貨日見頂預警
            st.markdown("### 🚨 大盤與個股出貨見頂預警")
            if "🚨" in jlaw_alert or "⚠️" in jlaw_alert:
                st.error(jlaw_alert)
            else:
                st.success(jlaw_alert)

            # ==========================================
            # 🌟 專業級量化分析圖表 (修復對數座標崩潰問題)
            # ==========================================
            st.markdown("### 📉 專業級量化分析圖表 (六層架構)")
            
            col_chart_opt1, col_chart_opt2 = st.columns(2)
            with col_chart_opt1:
                st.markdown(
                    "💡 **圖表線條顏色指南**：<br>"
                    "📈 **簡單均線 (MA)**：<span style='color:#29B6F6; font-weight:bold;'>🔵 10 MA</span> | "
                    "<span style='color:#AB47BC; font-weight:bold;'>🟣 20 MA</span> | "
                    "<span style='color:#FFA726; font-weight:bold;'>🟠 50 MA</span> | "
                    "<span style='color:#4CAF50; font-weight:bold;'>🟢 150 MA</span> | "
                    "<span style='color:#8D6E63; font-weight:bold;'>🟤 200 MA</span><br>"
                    "📊 **指數均線 (EMA)**：<span style='color:#F44336; font-weight:bold;'>🔴 9 EMA (動能追蹤)</span> | 虛線(Dash)", 
                    unsafe_allow_html=True
                )
            with col_chart_opt2:
                # OLD: use_log_scale = st.checkbox("📐 開啟對數座標 (Log Scale) - 觀察長期斜率必備", value=True if selected_interval in ["1d", "1wk"] else False)
                # NEW: [v24.14 預設關閉對數座標 + 新增顯示 K 線根數選擇器，徹底解決圖表被壓縮到最右邊的問題！]
                c_sub_opt1, c_sub_opt2 = st.columns(2)
                with c_sub_opt1:
                    use_log_scale = st.checkbox("📐 開啟對數座標 (Log Scale)", value=False, help="預設關閉以便清晰檢視買入位、結構止損位與止盈目標價")
                with c_sub_opt2:
                    chart_window_choice = st.selectbox(
                        "🔍 圖表顯示 K 線範圍:",
                        ["最近 120 根 K 線 (約半年，最清晰)", "最近 200 根 K 線 (約10個月)", "最近 300 根 K 線 (約1年半)", "全部歷史 K 線"],
                        index=0
                    )

            # OLD: plot_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
            # NEW: 指標已於完整 df 計算完畢，繪圖時截取最近 N 根 K 線，使全部 6 層子圖 X 軸完美對齊且 K 線清晰不壓縮！
            _full_valid_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
            if "120" in chart_window_choice:
                plot_df = _full_valid_df.tail(120).copy()
            elif "200" in chart_window_choice:
                plot_df = _full_valid_df.tail(200).copy()
            elif "300" in chart_window_choice:
                plot_df = _full_valid_df.tail(300).copy()
            else:
                plot_df = _full_valid_df
            if plot_df.empty:
                st.warning("K 線資料不足")
            else:
                x_dates = plot_df.index.strftime('%Y-%m-%d')
                
                fig = make_subplots(
                    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                    subplot_titles=[f"{name} K線走勢", "成交量", "RSI(14)", "MACD", "OBV (能量潮)", "MFI(14) (資金流量)"],
                    row_heights=[0.35, 0.15, 0.125, 0.125, 0.125, 0.125]
                )

                fig.add_trace(go.Candlestick(
                    x=x_dates, open=plot_df['Open'], high=plot_df['High'],
                    low=plot_df['Low'], close=plot_df['Close'], name='K線', showlegend=False,
                    increasing_line_color='#26A69A', decreasing_line_color='#EF5350',
                    increasing_fillcolor='#26A69A', decreasing_fillcolor='#EF5350'
                ), row=1, col=1)

                if 'BigMoney_In' in plot_df.columns:
                    bmi_df = plot_df[plot_df['BigMoney_In']]
                    if not bmi_df.empty:
                        fig.add_trace(go.Scatter(
                            x=bmi_df.index.strftime('%Y-%m-%d'), y=bmi_df['Low'] * 0.96, 
                            mode='markers', marker=dict(symbol='star', size=16, color='#FFD700', line=dict(width=1, color='black')),
                            name='🌟 大資金流入 (爆量收陽)'
                        ), row=1, col=1)

                if 'BigMoney_Out' in plot_df.columns:
                    bmo_df = plot_df[plot_df['BigMoney_Out']]
                    if not bmo_df.empty:
                        fig.add_trace(go.Scatter(
                            x=bmo_df.index.strftime('%Y-%m-%d'), y=bmo_df['High'] * 1.04, 
                            mode='markers', marker=dict(symbol='x', size=14, color='black', line=dict(width=2, color='#EF5350')),
                            name='❌ 大資金流出 (爆量收陰)'
                        ), row=1, col=1)

                if 'Is_Distribution' in plot_df.columns:
                    dist_df = plot_df[plot_df['Is_Distribution']]
                    if not dist_df.empty:
                        fig.add_trace(go.Scatter(
                            x=dist_df.index.strftime('%Y-%m-%d'), y=dist_df['High'] * 1.02, 
                            mode='markers', marker=dict(symbol='triangle-down', size=12, color='#9C27B0', line=dict(width=1, color='black')),
                            name='🔻 大戶出貨日 (量增滯漲)'
                        ), row=1, col=1)

                # MA 簡單均線 (實線)
                if 'SMA10' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA10'].ffill(), name='MA 10', line=dict(color='#29B6F6', width=1.5)), row=1, col=1)
                if 'SMA20' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA20'].ffill(), name='MA 20', line=dict(color='#AB47BC', width=1.5)), row=1, col=1)
                if 'SMA50' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA50'].ffill(), name='MA 50', line=dict(color='#FFA726', width=1.5)), row=1, col=1)
                if 'SMA150' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA150'].ffill(), name='MA 150', line=dict(color='#4CAF50', width=1.5)), row=1, col=1)
                if 'SMA200' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA200'].ffill(), name='MA 200', line=dict(color='#8D6E63', width=2)), row=1, col=1)

                # 🌟 EMA 指數均線 (虛線，用於雙均線系統)
                if 'EMA9' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['EMA9'].ffill(), name='EMA 9', line=dict(color='#F44336', width=1.2, dash='dot')), row=1, col=1)
                if 'EMA20' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['EMA20'].ffill(), name='EMA 20', line=dict(color='#AB47BC', width=1, dash='dash')), row=1, col=1)
                if 'EMA50' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['EMA50'].ffill(), name='EMA 50', line=dict(color='#FFA726', width=1, dash='dash')), row=1, col=1)

                if 'AVWAP_Swing' in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_dates, y=plot_df['AVWAP_Swing'],
                        name='AVWAP (波段低點錨定)', 
                        line=dict(color='#E91E63', width=1.8, dash='dot')
                    ), row=1, col=1)

                if 'AVWAP_52WLow' in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=x_dates, y=plot_df['AVWAP_52WLow'],
                        name='AVWAP (52週低點錨定)', 
                        line=dict(color='#9C27B0', width=1.8, dash='dot')
                    ), row=1, col=1)

                if 'SupShort' in last and pd.notna(last['SupShort']):
                    fig.add_hline(y=last['SupShort'], line_dash="solid", line_color="#4CAF50", annotation_text="最新短支", row=1, col=1)
                if 'ResShort' in last and pd.notna(last['ResShort']):
                    fig.add_hline(y=last['ResShort'], line_dash="solid", line_color="#F44336", annotation_text="最新短阻", row=1, col=1)
                if 'SupLong' in last and pd.notna(last['SupLong']):
                    fig.add_hline(y=last['SupLong'], line_dash="dash", line_color="#388E3C", annotation_text="最新長支", row=1, col=1)
                if 'ResLong' in last and pd.notna(last['ResLong']):
                    fig.add_hline(y=last['ResLong'], line_dash="dash", line_color="#D32F2F", annotation_text="最新長阻", row=1, col=1)

                if 'SupShort' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SupShort'].ffill(), name='短期支撐軌跡', line=dict(color='#4CAF50', width=1, dash='dot')), row=1, col=1)
                if 'ResShort' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['ResShort'].ffill(), name='短期阻力軌跡', line=dict(color='#F44336', width=1, dash='dot')), row=1, col=1)
                if 'SupLong' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SupLong'].ffill(), name='長期支撐軌跡', line=dict(color='#388E3C', width=1.5, dash='dashdot')), row=1, col=1)
                if 'ResLong' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['ResLong'].ffill(), name='長期阻力軌跡', line=dict(color='#D32F2F', width=1.5, dash='dashdot')), row=1, col=1)

                # 正確綁定的入場點與防守線
                # OLD:
                # if chart_plan != "WATCH":
                #     fig.add_hline(y=chart_tp, line_dash="dash", line_color="green", annotation_text=f"TP {chart_tp:.2f}", annotation_position="top left", row=1, col=1)
                #     fig.add_hline(y=chart_entry, line_dash="dot", line_color="blue", annotation_text=f"Entry {chart_entry:.2f}", annotation_position="top left", row=1, col=1)
                #     fig.add_hline(y=chart_sl, line_dash="dash", line_color="red", annotation_text=f"SL {chart_sl:.2f}", annotation_position="top left", row=1, col=1)
                # NEW: [v24.14] 在 K 線圖上清晰標示建議買入位、結構止損位、結構止盈位與實際盈虧比 R:R！
                if chart_plan != "WATCH":
                    _c_risk = abs(chart_entry - chart_sl)
                    _c_rr = abs(chart_tp - chart_entry) / _c_risk if _c_risk > 0 else 0.0
                    _tp_pct = (chart_tp - chart_entry) / chart_entry * 100.0
                    _sl_pct = (chart_sl - chart_entry) / chart_entry * 100.0
                    fig.add_hline(
                        y=chart_tp, line_dash="dash", line_width=2, line_color="#00C853",
                        annotation_text=f"🎯 結構止盈 TP: {chart_tp:.2f} ({_tp_pct:+.1f}% ｜ 盈虧比 {_c_rr:.2f}R)",
                        annotation_position="top left", annotation_font=dict(color="#007E33", size=12), row=1, col=1
                    )
                    fig.add_hline(
                        y=chart_entry, line_dash="solid", line_width=2, line_color="#2962FF",
                        annotation_text=f"📌 建議{'買入' if chart_plan == 'LONG' else '沽空'}位 Entry: {chart_entry:.2f}",
                        annotation_position="top left", annotation_font=dict(color="#2962FF", size=12), row=1, col=1
                    )
                    fig.add_hline(
                        y=chart_sl, line_dash="dash", line_width=2, line_color="#D50000",
                        annotation_text=f"🛡️ 結構止損 SL: {chart_sl:.2f} ({_sl_pct:+.1f}%)",
                        annotation_position="bottom left", annotation_font=dict(color="#D50000", size=12), row=1, col=1
                    )

                candlestick_patterns = detect_candlestick_patterns(plot_df)
                if candlestick_patterns and candlestick_patterns[0]["name"] != "":
                    fig.add_annotation(
                        x=x_dates[-1], y=plot_df['High'].iloc[-1],
                        text=f"📌 {candlestick_patterns[0]['name']}",
                        showarrow=True, arrowhead=1, arrowcolor="purple",
                        ax=0, ay=-40, font=dict(color="white", size=12),
                        bgcolor="purple", bordercolor="purple", row=1, col=1
                    )

                vol_colors = []
                for i in range(len(plot_df)):
                    c, o = plot_df['Close'].iloc[i], plot_df['Open'].iloc[i]
                    is_dist = plot_df['Is_Distribution'].iloc[i] if 'Is_Distribution' in plot_df.columns else False
                    if is_dist: 
                        vol_colors.append('#9C27B0')
                    elif c >= o: 
                        vol_colors.append('#26A69A')
                    else: 
                        vol_colors.append('#EF5350')
                        
                fig.add_trace(go.Bar(x=x_dates, y=plot_df['Volume'], name='Volume', marker_color=vol_colors), row=2, col=1)
                
                if 'VolumeSMA50' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['VolumeSMA50'].ffill(), name='VMA 50', line=dict(color='#2196F3', width=1.5)), row=2, col=1)

                if 'RSI' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['RSI'], name='RSI(14)', line=dict(color='#FF9800', width=1.5)), row=3, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="gray", row=3, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="gray", row=3, col=1)

                if 'MACD' in plot_df.columns and 'MACDSignal' in plot_df.columns and 'MACDHist' in plot_df.columns:
                    macd_colors = ['#26A69A' if val >= 0 else '#EF5350' for val in plot_df['MACDHist']]
                    fig.add_trace(go.Bar(x=x_dates, y=plot_df['MACDHist'], name='MACD柱', marker_color=macd_colors), row=4, col=1)
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['MACD'], name='DIF', line=dict(color='#2196F3', width=1.2)), row=4, col=1)
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['MACDSignal'], name='DEA', line=dict(color='#FFEB3B', width=1.2)), row=4, col=1)

                if 'OBV' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['OBV'], name='OBV', showlegend=False, line=dict(color='#BA68C8', width=1.5)), row=5, col=1)

                if 'MFI' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['MFI'], name='MFI(14)', line=dict(color='#4DB6AC', width=1.5)), row=6, col=1)
                    fig.add_hline(y=80, line_dash="dash", line_color="gray", row=6, col=1)
                    fig.add_hline(y=20, line_dash="dash", line_color="gray", row=6, col=1)

                fig.update_layout(
                    height=1200, template="plotly_white", xaxis_rangeslider_visible=False,
                    margin=dict(t=30, b=10, l=40, r=40), hovermode="x unified",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.05,
                        xanchor="center",
                        x=0.5,
                        itemwidth=40,
                        bgcolor="rgba(245, 246, 249, 1)",
                        bordercolor="#D3D3D3",
                        borderwidth=1,
                        font=dict(size=13, color="black"),
                        traceorder="normal",
                        title=dict(text="🎯 點擊下方指標名稱進行獨立開關：")
                    ),
                    paper_bgcolor='white', plot_bgcolor='white'
                )

                # 修復對數座標與線性座標 Y 軸範圍：確保当前窗口 K 線與 Entry / SL / TP 盈虧比價格線 100% 清晰顯示
                if use_log_scale:
                    try:
                        valid_lows = plot_df['Low'].replace(0, np.nan).dropna()
                        valid_highs = plot_df['High'].replace(0, np.nan).dropna()
                        if not valid_lows.empty and not valid_highs.empty:
                            min_y = float(valid_lows.min()) * 0.92
                            max_y = float(valid_highs.max()) * 1.08
                            if chart_plan != "WATCH":
                                min_y = min(min_y, float(chart_sl) * 0.96, float(chart_tp) * 0.96)
                                max_y = max(max_y, float(chart_sl) * 1.04, float(chart_tp) * 1.04)
                            if min_y > 0 and max_y > 0:
                                log_min_y = math.log10(min_y)
                                log_max_y = math.log10(max_y)
                                fig.update_yaxes(type="log", range=[log_min_y, log_max_y], row=1, col=1)
                            else:
                                fig.update_yaxes(type="log", row=1, col=1) 
                        else:
                            fig.update_yaxes(type="log", row=1, col=1) 
                    except Exception:
                        fig.update_yaxes(type="log", row=1, col=1)
                else:
                    try:
                        lin_min = float(plot_df['Low'].min())
                        lin_max = float(plot_df['High'].max())
                        if chart_plan != "WATCH":
                            lin_min = min(lin_min, float(chart_sl), float(chart_entry), float(chart_tp))
                            lin_max = max(lin_max, float(chart_sl), float(chart_entry), float(chart_tp))
                        pad = (lin_max - lin_min) * 0.08 if lin_max > lin_min else lin_max * 0.05
                        fig.update_yaxes(type="linear", range=[max(0, lin_min - pad), lin_max + pad], row=1, col=1)
                    except Exception:
                        pass
                
                for i in range(1, 7):
                    fig.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikethickness=1, spikecolor='#757575', spikedash='solid', row=i, col=1)
                    if i < 6:
                        fig.update_xaxes(type='category', showticklabels=False, showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikethickness=1, spikecolor='#757575', spikedash='solid', row=i, col=1)
                    else:
                        fig.update_xaxes(type='category', nticks=20, showticklabels=True, tickangle=-45, showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikethickness=1, spikecolor='#757575', spikedash='solid', row=i, col=1)

                fig.update_layout(uirevision=target_ticker)
                plot_config = dict(scrollZoom=True, displayModeBar=True, modeBarButtonsToRemove=['lasso2d', 'select2d'], displaylogo=False)
                st.plotly_chart(fig, use_container_width=True, config=plot_config, theme=None, key=f"main_chart_{target_ticker}")

                # --- 恢復 K 線型態詳細解讀文字區塊 ---
                st.markdown("### 🕯️ K 線型態解讀")
                if candlestick_patterns:
                    for pattern in candlestick_patterns:
                        if pattern["name"] != "":
                            with st.container():
                                st.info(f"**{pattern['name']}** ｜ **市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}")
                        else:
                            st.info(f"**市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}")

            if 'CloseBench' in plot_df.columns and not plot_df['CloseBench'].isna().all():
                st.markdown(f"### 📈 {benchmark_name} 參考走勢")
                st.caption("用於觀察個股與大盤趨勢同步性")
                fig_bench = go.Figure()
                x_bench_dates = plot_df.index.strftime('%Y-%m-%d')
                fig_bench.add_trace(go.Scatter(x=x_bench_dates, y=plot_df['CloseBench'], name=f"{benchmark_name} 走勢", line=dict(color='purple')))
                fig_bench.update_layout(height=300, template="plotly_white", margin=dict(t=10, b=10), hovermode="x unified", dragmode="pan", paper_bgcolor='white', plot_bgcolor='white')
                
                if use_log_scale:
                    try:
                        valid_bench = plot_df['CloseBench'].replace(0, np.nan).dropna()
                        if not valid_bench.empty:
                            b_min = valid_bench.min() * 0.9
                            b_max = valid_bench.max() * 1.1
                            if b_min > 0 and b_max > 0:
                                fig_bench.update_yaxes(type="log", range=[math.log10(b_min), math.log10(b_max)])
                            else:
                                fig_bench.update_yaxes(type="log")
                        else:
                            fig_bench.update_yaxes(type="log")
                    except Exception:
                        fig_bench.update_yaxes(type="log")
                    
                fig_bench.update_xaxes(type='category', nticks=20, showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikethickness=1, spikecolor='#757575')
                fig_bench.update_yaxes(showspikes=True, spikemode='across', spikesnap='cursor', showline=True, spikethickness=1, spikecolor='#757575')
                st.plotly_chart(fig_bench, use_container_width=True, config=plot_config, theme=None)

            st.markdown("---")
            st.markdown("### ⚡ 快速回測此標的")
            with st.expander("開啟個股快速回測", expanded=False):
                cqbt1, cqbt2, cqbt3 = st.columns(3)
                with cqbt1:
                    btcapq = st.number_input("初始資金", min_value=1000, value=100000, step=10000, key="btcapq")
                with cqbt2:
                    btmodeq = st.selectbox("方向", ["全部雙向", "僅做多頭(做多)", "僅做空頭(做空)"], key="btmodeq")
                with cqbt3:
                    btrelaxq = st.checkbox("寬鬆模式", value=True, help="較重視 SMA50 / MACD 與大盤方向", key="btrelaxq")

                cqdate1, cqdate2 = st.columns(2)
                with cqdate1:
                    startdtq = st.date_input("開始日期", value=datetime.now() - timedelta(days=365*5), key="dtsq")
                with cqdate2:
                    enddtq = st.date_input("結束日期", value=datetime.now(), key="dteq")

                if st.button("啟動快速回測", key="btnquickbt"):
                    with st.spinner("回測中..."):
                        bt_res_q = run_bidirectional_backtest(ticker_input, startdtq, enddtq, btcapq, strictness, btmodeq, is_relaxed=btrelaxq)
                        if isinstance(bt_res_q, str):
                            st.error(bt_res_q)
                        else:
                            fcap, trds, eqcurve, bhr, curq, nq = bt_res_q
                            trdsdf = pd.DataFrame(trds)
                            ltrds = trdsdf[trdsdf['方向'] == '🟢 做多'] if not trdsdf.empty else pd.DataFrame()
                            strds = trdsdf[trdsdf['方向'] == '🔴 做空'] if not trdsdf.empty else pd.DataFrame()
                            lwins = len(ltrds[ltrds['利潤'] > 0]) if not ltrds.empty else 0
                            swins = len(strds[strds['利潤'] > 0]) if not strds.empty else 0
                            lwinr = lwins / len(ltrds) * 100 if not ltrds.empty else 0.0
                            swinr = swins / len(strds) * 100 if not strds.empty else 0.0
                            owinr = (lwins + swins) / len(trdsdf) * 100 if not trdsdf.empty else 0.0

                            st.markdown(f"#### {ticker_input} 回測總覽")
                            mb1, mb2, mb3 = st.columns(3)
                            mb1.metric("策略總回報率", f"{(fcap - btcapq) / btcapq * 100:.2f}%")
                            mb2.metric("整體勝率", f"{owinr:.1f}%", f"共 {len(trdsdf)} 筆")
                            mb3.metric("最終淨值", f"{fcap:,.2f} {curq}")

                            sb1, sb2 = st.columns(2)
                            sb1.info(f"🟢 做多勝率：{lwinr:.1f}%\n\n交易次數：{len(ltrds)}")
                            sb2.error(f"🔴 做空勝率：{swinr:.1f}%\n\n交易次數：{len(strds)}")

                            fig_eq = go.Figure()
                            x_eq_dates = eqcurve.index.strftime('%Y-%m-%d')
                            fig_eq.add_trace(go.Scatter(x=x_eq_dates, y=eqcurve.values, mode='lines', name='資產曲線', line=dict(color='green', width=2.5)))
                            fig_eq.update_layout(title="帳戶資產曲線", xaxis_title="日期", yaxis_title="總資產", template="plotly_white", height=300)
                            fig_eq.update_xaxes(type='category', nticks=20)
                            st.plotly_chart(fig_eq, use_container_width=True, theme=None)

                            if not trdsdf.empty:
                                show_df = trdsdf.drop(columns=['利潤']) if '利潤' in trdsdf.columns else trdsdf
                                st.dataframe(show_df, use_container_width=True)

# ==========================================
# 15. 模式 4：回測系統
# ==========================================
elif mode == mode_options[3]:
    st.query_params.clear()
    st.title("📈 策略歷史回測系統 (利潤最大化風控版)")
    st.markdown("已套用：**前阻力保底2R分批止盈**、**初期忍耐區寬止損**、**脫離成本後升級保本**、**9 EMA大趨勢跟蹤**，以及**15天時間停損**。")

    cbt1, cbt2, cbt3, cbt4 = st.columns(4)
    with cbt1:
        bt_ticker = st.text_input("股票代碼", value="3690.HK").strip()
    with cbt2:
        bt_capital = st.number_input("初始資金", min_value=1000, value=100000, step=10000)
    with cbt3:
        bt_mode = st.selectbox("交易方向", ["全部雙向", "僅做多頭(做多)", "僅做空頭(做空)"])
    with cbt4:
        bt_relaxed = st.checkbox("寬鬆模式", value=True, help="偏重 SMA50 / MACD 與大盤濾網")

    cdate1, cdate2 = st.columns(2)
    with cdate1:
        startdt = st.date_input("開始日期", value=datetime.now() - timedelta(days=365*5))
    with cdate2:
        enddt = st.date_input("結束日期", value=datetime.now())

    if st.button("🏁 啟動歷史回測"):
        with st.spinner("回測運算中..."):
            bt_res = run_bidirectional_backtest(bt_ticker, startdt, enddt, bt_capital, strictness, bt_mode, is_relaxed=bt_relaxed)
            if isinstance(bt_res, str):
                st.error(bt_res)
            else:
                final_cap, trades, equity_curve, bh_ret, cur, name = bt_res
                trades_df = pd.DataFrame(trades)
                long_trades = trades_df[trades_df['方向'] == '🟢 做多'] if not trades_df.empty else pd.DataFrame()
                short_trades = trades_df[trades_df['方向'] == '🔴 做空'] if not trades_df.empty else pd.DataFrame()

                long_wins = len(long_trades[long_trades['利潤'] > 0]) if not long_trades.empty else 0
                short_wins = len(short_trades[short_trades['利潤'] > 0]) if not short_trades.empty else 0
                long_win_rate = long_wins / len(long_trades) * 100 if not long_trades.empty else 0.0
                short_win_rate = short_wins / len(short_trades) * 100 if not short_trades.empty else 0.0
                overall_win_rate = (long_wins + short_wins) / len(trades_df) * 100 if not trades_df.empty else 0.0

                st.markdown(f"### 📊 {bt_ticker} - {name}")
                mb1, mb2, mb3 = st.columns(3)
                mb1.metric("策略總回報率", f"{(final_cap - bt_capital) / bt_capital * 100:.2f}%", f"買入持有(B&H): {bh_ret:.2f}%")
                mb2.metric("整體交易勝率", f"{overall_win_rate:.1f}%", f"共 {len(trades_df)} 筆")
                mb3.metric("帳戶最終淨值", f"{final_cap:,.2f} {cur}")

                # [NEW INCREMENTAL PATCH 維度3] 增量展示機構級回測風險指標 (MDD、Profit Factor、Sharpe、數學期望值)
                _inst_m = compute_backtest_institutional_metrics(equity_curve, trades_df, bt_capital)
                im1, im2, im3, im4 = st.columns(4)
                im1.metric("最大回撤 (Max DD)", f"{_inst_m['max_drawdown_pct']:.2f}%")
                im2.metric("獲利因子 (Profit Factor)", f"{_inst_m['profit_factor']:.2f}")
                im3.metric("年化夏普 (Sharpe)", f"{_inst_m['sharpe_ratio']:.2f}")
                im4.metric("單筆期望值 (Expectancy)", f"{_inst_m['expectancy_pct']:.2f}%", f"盈虧比 {_inst_m['payoff_ratio']:.2f}")

                st.markdown("#### 🔍 多空分開表現")
                sb1, sb2 = st.columns(2)
                sb1.info(f"🟢 做多勝率：{long_win_rate:.1f}%\n\n交易次數：{len(long_trades)}")
                sb2.error(f"🔴 做空勝率：{short_win_rate:.1f}%\n\n交易次數：{len(short_trades)}")

                fig_eq = go.Figure()
                x_eq_dates = equity_curve.index.strftime('%Y-%m-%d')
                fig_eq.add_trace(go.Scatter(x=x_eq_dates, y=equity_curve.values, mode='lines', name='資產曲線', line=dict(color='green', width=2.5)))
                fig_eq.update_layout(title="帳戶資產動態曲線", xaxis_title="日期", yaxis_title="總資產", template="plotly_white", height=400)
                fig_eq.update_xaxes(type='category', nticks=20)
                st.plotly_chart(fig_eq, use_container_width=True, theme=None)

                if not trades_df.empty:
                    show_df = trades_df.drop(columns=['利潤']) if '利潤' in trades_df.columns else trades_df
                    st.dataframe(show_df, use_container_width=True)

# ==========================================
# 核心回測引擎 (植入 J Law 做空邏輯與利潤最大化)
# ==========================================
def run_bidirectional_backtest(ticker, start_date, end_date, initial_capital, score_threshold, trade_mode, is_relaxed):
    try:
        ticker, market_type, benchmark_ticker, benchmark_name, currency = format_and_detect_market(ticker)
        bg_df = get_index_data(benchmark_ticker)
        name = get_stock_name(ticker)

        download_start = start_date - timedelta(days=365)
        df = yf.download(ticker, start=download_start, end=end_date, progress=False)
        if df.empty: return "無數據，無法回測。"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0 if 'Close' in df.columns.get_level_values(0) else 1)

        df = df.loc[:, ~df.columns.duplicated()].dropna(subset=['Close'])
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        df = calculate_indicators(df, bg_df)
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        if df.empty: return "選定區間內無有效數據。"

        cash = initial_capital
        position_type = None
        entry_price, stop_loss, tp_target, entry_structural_sl = 0, 0, 0, None
        days_in_trade = 0 
        position_stage = 0  # 0: 忍耐洗盤期, 1: 保本跟蹤期, 2: 純 9EMA 跟蹤期
        shares = 0
        trade_log, history_equity, history_dates = [], [], []

        slippage_fee = 0.0015 if market_type == "HK" else 0.0002

        for i in range(1, len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            p_prev20 = df.iloc[i - 20] if i >= 20 else df.iloc[0]

            is_hk_liquid = not (market_type == "HK" and prev_row['TurnoverMA20'] < 20_000_000)

            is_market_bullish = True
            if 'CloseBench' in prev_row and 'SMA50BenchValue' in prev_row and pd.notna(prev_row['CloseBench']):
                is_market_bullish = prev_row['CloseBench'] > prev_row['SMA50BenchValue']

            # 大盤防禦閘門
            bench_close = prev_row.get('CloseBench', np.nan)
            bench_200 = prev_row.get('SMA200BenchValue', np.nan)
            bench_dist = prev_row.get('Bench_Dist_Days', 0)
            is_market_health_ok = True
            if pd.notna(bench_close) and pd.notna(bench_200):
                if bench_close < bench_200 or bench_dist >= 5:
                    is_market_health_ok = False

            current_equity = cash + (shares * row['Close'] if position_type == "LONG" else ((entry_price - row['Close']) * shares if position_type == "SHORT" else 0))

            # ====================================================
            # 觸發條件套用「融合邏輯」
            # ====================================================
            if is_relaxed:
                is_long_trigger = prev_row['Close'] > prev_row['SMA50'] and prev_row['MACD'] > prev_row['MACDSignal'] and is_hk_liquid and is_market_bullish and is_market_health_ok
                is_short_trigger = prev_row['Close'] < prev_row['SMA50'] and prev_row['MACD'] < prev_row['MACDSignal'] and is_hk_liquid and not is_market_bullish
            else:
                p_low, p_high = df.iloc[:i]['Low'].min(), df.iloc[:i]['High'].max()

                r1 = prev_row['Close'] > prev_row['SMA150'] and prev_row['Close'] > prev_row['SMA200']
                r2 = prev_row['SMA150'] > prev_row['SMA200']
                r3 = prev_row['SMA200'] > p_prev20['SMA200']
                r4 = prev_row['SMA50'] > prev_row['SMA150'] and prev_row['SMA50'] > prev_row['SMA200']
                r5 = prev_row['Close'] > p_low * 1.25
                r6 = prev_row['Close'] > p_high * 0.70
                r7 = prev_row['Mansfield_RS'] > 0
                prev_pass_count = sum([r1, r2, r3, r4, r5, r6, r7])

                kk_flag = prev_row['Ret1M'] > 0.15 and prev_row['Close'] > prev_row['SMA10'] and prev_row['VCPSignal']
                
                momentum_ready = (prev_row['RSI'] > 55) and (prev_row['Close'] > prev_row['SMA20'])
                is_tight_setup = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)
                p_high_20_slice = df.iloc[max(0, i-20):i]
                p_high_20_val = p_high_20_slice['Close'].max() if not p_high_20_slice.empty else prev_row['Close']
                is_breakout = (prev_row['Close'] >= p_high_20_val * 0.99) and (prev_row['VolRatio'] > 1.5)
                early_stage_bypass = (prev_row['Close'] > prev_row['SMA50']) and (prev_row['Ret1M'] > 0.20)

                is_long_trigger = momentum_ready and (
                    (prev_pass_count >= score_threshold) or early_stage_bypass or kk_flag
                ) and (is_tight_setup or is_breakout) and is_hk_liquid and is_market_health_ok

                # [做空優化] 結合「原版結構空頭」與「J Law 動能做空」雙邏輯
                orig_sr1 = prev_row['Close'] < prev_row['SMA200']
                orig_sr2 = prev_row['SMA50'] < prev_row['SMA150'] and prev_row['SMA150'] < prev_row['SMA200']
                orig_sr3 = prev_row['Close'] < (df['CumulativeHigh'].iloc[i-1] if 'CumulativeHigh' in df.columns else p_high) * 0.85
                orig_sr4 = prev_row['Mansfield_RS'] < 0
                orig_sr5 = prev_row['VolRatio'] < 0.8
                orig_sr6 = prev_row['LowerShadowRatio'] < 0.4
                orig_sr7 = prev_row['VCPSignal']
                prev_short_pass_count = sum([orig_sr1, orig_sr2, orig_sr3, orig_sr4, orig_sr5, orig_sr6, orig_sr7])
                
                is_short_candidate_original = prev_short_pass_count >= score_threshold

                p_low_20_slice = df.iloc[max(0, i-20):i]
                p_low_20_val = p_low_20_slice['Close'].min() if not p_low_20_slice.empty else prev_row['Close']
                is_breakdown = (prev_row['Close'] <= p_low_20_val * 1.01) and (prev_row['VolRatio'] > 1.5)
                ma_pressing = (prev_row['Close'] < prev_row['SMA20']) and (prev_row['SMA20'] < prev_row['SMA50'])
                is_weak_bounce = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)

                is_jlaw_short_trigger = ma_pressing and (prev_row['Mansfield_RS'] < 0) and (is_weak_bounce or is_breakdown)
                
                is_short_trigger = (is_short_candidate_original or is_jlaw_short_trigger) and is_hk_liquid and not is_market_bullish

            atr_val = prev_row['ATR'] if pd.notna(prev_row['ATR']) and prev_row['ATR'] > 0 else row['Open'] * 0.03

            if position_type is None:
                # [NEW v24.14 回測與掃描系統 100% 理念同步]
                _lei_pullback_ok = bool(
                    prev_row.get('Clock_2', False)
                    and (not prev_row.get('Is_Abnormal_Drop', False))
                    and (prev_row['Close'] >= prev_row.get('SMA50', 0) * 0.99)
                    and (prev_row['Close'] < prev_row.get('SMA20', 0) and prev_row['Close'] > prev_row.get('SMA20_Deduction', 0))
                )
                _lei_tight_ok = bool(
                    prev_row.get('LEI_MA_Tight_2Pct', False)
                    and (prev_row['Close'] > prev_row.get('EMA20', 0))
                    and (prev_row['Close'] > prev_row.get('SMA20_Deduction', 0))
                )
                _short_term_uptrend_ok = bool(prev_row['Close'] >= prev_row.get('SMA50', 0))
                _unified_long_signal = (is_long_trigger or _lei_pullback_ok or _lei_tight_ok) and _short_term_uptrend_ok and is_hk_liquid and is_market_health_ok

                if _unified_long_signal and trade_mode in ["全部雙向", "僅做多頭(做多)"]:
                    trigger_price = float(prev_row['High'])
                    if row['High'] >= trigger_price and row['Open'] <= trigger_price * 1.035:
                        actual_entry_price = max(float(row['Open']), trigger_price)
                        entry_price = actual_entry_price * (1 + slippage_fee)

                        _struct_sup = max(
                            float(prev_row.get("BoxLow10_Causal", prev_row.get("SupShort_Causal", entry_price * 0.94))) - 0.15 * atr_val,
                            entry_price * 0.925
                        )
                        _atr_sl = max(entry_price - 1.8 * atr_val, entry_price * 0.925)
                        stop_loss = max(_atr_sl, _struct_sup) if (entry_price - _struct_sup) >= 0.8 * atr_val else _atr_sl
                        if stop_loss >= entry_price * 0.988:
                            stop_loss = entry_price * 0.965
                        risk_per_share = max(entry_price - stop_loss, 1e-4)

                        _cands_tp = [
                            float(prev_row.get("ResShort_Causal", 0)),
                            float(prev_row.get("ResLong_Causal", 0)),
                            float(prev_row.get("Res120_Causal", 0)),
                            float(prev_row.get("High52W", 0))
                        ]
                        _valid_tps = sorted([p for p in _cands_tp if p > entry_price * 1.022])
                        if _valid_tps:
                            tp_target = _valid_tps[0]
                            if (tp_target - entry_price) / risk_per_share < 1.6 and len(_valid_tps) >= 2:
                                tp_target = _valid_tps[1]
                        else:
                            _wave_amp = max(float(prev_row.get("ResShort_Causal", entry_price)) - float(prev_row.get("SupShort_Causal", stop_loss)), 3.2 * atr_val)
                            tp_target = entry_price + _wave_amp * 0.85

                        _planned_rr = (tp_target - entry_price) / risk_per_share
                        if _planned_rr >= 1.8:
                            risk_amount = current_equity * 0.01
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            if (target_shares * entry_price) > cash:
                                target_shares = int(cash / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                shares = target_shares
                                cash -= shares * entry_price
                                position_stage = 0
                                days_in_trade = 0
                                entry_structural_sl = float(prev_row.get("SupShort_Causal", prev_row.get("SupShort", entry_price * 0.92)))
                                position_type = "LONG"
                                trade_info = {
                                    "方向": "🟢 做多",
                                    "進場日期": current_date,
                                    "進場價": round(entry_price, 2),
                                    "初始止損": round(stop_loss, 2),
                                    "目標止盈": round(tp_target, 2),
                                    "計畫盈虧比": f"{_planned_rr:.2f}R",
                                    "最終止盈": "結構TP 或 9EMA 跟蹤",
                                    "結構止損": round(entry_structural_sl, 2)
                                }

                elif is_short_trigger and trade_mode in ["全部雙向", "僅做空頭(做空)"]:
                    trigger_price = float(prev_row['Low'])
                    if row['Low'] <= trigger_price and row['Open'] >= trigger_price * 0.965:
                        actual_entry_price = min(float(row['Open']), trigger_price)
                        entry_price = actual_entry_price * (1 - slippage_fee)

                        _struct_res = min(
                            float(prev_row.get("BoxHigh10_Causal", prev_row.get("ResShort_Causal", entry_price * 1.06))) + 0.15 * atr_val,
                            entry_price * 1.075
                        )
                        _atr_sl = min(entry_price + 1.8 * atr_val, entry_price * 1.075)
                        stop_loss = min(_atr_sl, _struct_res) if (_struct_res - entry_price) >= 0.8 * atr_val else _atr_sl
                        if stop_loss <= entry_price * 1.012:
                            stop_loss = entry_price * 1.035
                        risk_per_share = max(stop_loss - entry_price, 1e-4)

                        _cands_tp = [
                            float(prev_row.get("SupShort_Causal", 999999)),
                            float(prev_row.get("SupLong_Causal", 999999)),
                            float(prev_row.get("Sup120_Causal", 999999)),
                            float(prev_row.get("Low52W", 999999))
                        ]
                        _valid_tps = sorted([p for p in _cands_tp if 0 < p < entry_price * 0.978], reverse=True)
                        if _valid_tps:
                            tp_target = _valid_tps[0]
                            if (entry_price - tp_target) / risk_per_share < 1.6 and len(_valid_tps) >= 2:
                                tp_target = _valid_tps[1]
                        else:
                            _wave_amp = max(float(prev_row.get("ResShort_Causal", stop_loss)) - float(prev_row.get("SupShort_Causal", entry_price)), 3.2 * atr_val)
                            tp_target = max(0.01, entry_price - _wave_amp * 0.85)

                        _planned_rr = (entry_price - tp_target) / risk_per_share
                        if _planned_rr >= 1.8:
                            risk_amount = current_equity * 0.01
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            if (target_shares * entry_price) > cash:
                                target_shares = int(cash / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                shares = target_shares
                                position_stage = 0
                                days_in_trade = 0
                                entry_structural_sl = float(prev_row.get("ResShort_Causal", prev_row.get("ResShort", entry_price * 1.08)))
                                position_type = "SHORT"
                                trade_info = {
                                    "方向": "🔴 做空",
                                    "進場日期": current_date,
                                    "進場價": round(entry_price, 2),
                                    "初始止損": round(stop_loss, 2),
                                    "目標止盈": round(tp_target, 2),
                                    "計畫盈虧比": f"{_planned_rr:.2f}R",
                                    "最終止盈": "結構TP 或 9EMA 跟蹤",
                                    "結構止損": round(entry_structural_sl, 2)
                                }

            elif position_type == "LONG":
                days_in_trade += 1
                is_time_stop = (days_in_trade >= 15) and (row['Close'] < entry_price * 1.02)
                is_trending_market = pd.notna(bench_close) and bench_close > prev_row.get('SMA50BenchValue', 0) * 1.015
                
                # 保本升級檢查
                if position_stage == 0 and days_in_trade >= 2 and row.get('EMA9', 0) > entry_price:
                    position_stage = 1
                    stop_loss = entry_price 

                if row['Low'] <= stop_loss or is_time_stop:
                    base_exit = min(row['Open'], stop_loss) if row['Low'] <= stop_loss else row['Close']
                    result_text = "觸發防守止損/保本" if row['Low'] <= stop_loss else "時間停損 (15日)"

                    exit_price = base_exit * (1 - slippage_fee)
                    cash += (shares * exit_price)
                    profit = (exit_price - entry_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": 0
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                elif row['High'] >= tp_target:
                    if is_trending_market and position_stage < 2:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0: sold_shares = shares 
                        
                        exit_price = max(row['Open'], tp_target) * (1 - slippage_fee)
                        cash += (sold_shares * exit_price)
                        shares -= sold_shares
                        
                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：觸及前高減倉一半",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": (exit_price - entry_price) * sold_shares,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(partial_info)
                        
                        if shares > 0:
                            position_stage = 2 
                            stop_loss = entry_price 
                        else:
                            position_type, position_stage = None, 0
                    elif not is_trending_market:
                        exit_price = max(row['Open'], tp_target) * (1 - slippage_fee)
                        cash += (shares * exit_price)
                        profit = (exit_price - entry_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "震盪市：觸及前高全平",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

                elif position_stage == 2:
                    # OLD: if row['Close'] < row.get('EMA9', 0) or row['Close'] < row.get('SupShort', 0):
                    # NEW: 採用無未來函數之 SupShort_Causal 與 9EMA 作為大趨勢跟蹤離場線
                    if row['Close'] < row.get('EMA9', 0) or row['Close'] < row.get('SupShort_Causal', row.get('SupShort', 0)):
                        exit_price = row['Close'] * (1 - slippage_fee)
                        cash += (shares * exit_price)
                        profit = (exit_price - entry_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：跌破9EMA/趨勢線離場",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

            elif position_type == "SHORT":
                days_in_trade += 1
                is_time_stop = (days_in_trade >= 15) and (row['Close'] > entry_price * 0.98)
                is_trending_market = pd.notna(bench_close) and bench_close < prev_row.get('SMA50BenchValue', 0) * 0.985

                if position_stage == 0 and days_in_trade >= 2 and row.get('EMA9', 9999) < entry_price:
                    position_stage = 1
                    stop_loss = entry_price

                if row['High'] >= stop_loss or is_time_stop:
                    base_exit = max(row['Open'], stop_loss) if row['High'] >= stop_loss else row['Close']
                    result_text = "觸發防守止損/保本" if row['High'] >= stop_loss else "時間停損 (15日)"

                    exit_price = base_exit * (1 + slippage_fee)
                    cash += ((entry_price - exit_price) * shares)
                    profit = (entry_price - exit_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": 0
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                elif row['Low'] <= tp_target:
                    if is_trending_market and position_stage < 2:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0: sold_shares = shares
                        
                        exit_price = min(row['Open'], tp_target) * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * sold_shares)
                        shares -= sold_shares
                        
                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：觸及前低減倉一半",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": (entry_price - exit_price) * sold_shares,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(partial_info)
                        
                        if shares > 0:
                            position_stage = 2
                            stop_loss = entry_price
                        else:
                            position_type, position_stage = None, 0
                    elif not is_trending_market:
                        exit_price = min(row['Open'], tp_target) * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * shares)
                        profit = (entry_price - exit_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "震盪市：觸及前低全平",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

                elif position_stage == 2:
                    # OLD: if row['Close'] > row.get('EMA9', 9999) or row['Close'] > row.get('ResShort', 9999):
                    # NEW: 採用無未來函數之 ResShort_Causal 與 9EMA 作為空頭跟蹤離場線
                    if row['Close'] > row.get('EMA9', 9999) or row['Close'] > row.get('ResShort_Causal', row.get('ResShort', 9999)):
                        exit_price = row['Close'] * (1 + slippage_fee)
                        cash += ((entry_price - exit_price) * shares)
                        profit = (entry_price - exit_price) * shares
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "大趨勢：升穿 9EMA/趨勢線離場",
                            "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                            "利潤": profit,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

            current_equity = cash + (
                shares * row['Close'] if position_type == "LONG"
                else ((entry_price - row['Close']) * shares if position_type == "SHORT" else 0)
            )
            history_equity.append(current_equity)
            history_dates.append(current_date)

        if position_type is not None:
            exit_price = df.iloc[-1]['Close']
            if position_type == "LONG":
                exit_price = exit_price * (1 - slippage_fee)
                cash += shares * exit_price
                profit = (exit_price - entry_price) * shares
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": exit_price,
                    "結果": "回測結束強制平倉",
                    "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                    "利潤": profit
                })
            else:
                exit_price = exit_price * (1 + slippage_fee)
                cash += (entry_price - exit_price) * shares
                profit = (entry_price - exit_price) * shares
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": exit_price,
                    "結果": "回測結束強制平倉",
                    "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                    "利潤": profit
                })
            trade_log.append(trade_info)

        return cash, trade_log, pd.Series(history_equity, index=history_dates), ((df.iloc[-1]['Close'] - df.iloc[0]['Open']) / df.iloc[0]['Open']) * 100, currency, name
    except Exception as e:
        return f"{str(e)}"
