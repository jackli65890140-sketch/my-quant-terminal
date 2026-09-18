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

st.set_page_config(layout="wide", page_title="專業量化交易終端系統 v24.2 (全寬表格優化版)")

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
        "watchlist_radar_results": {} # 🌟 升級為字典，儲存每個分組的獨立掃描結果
    }

persistent_data = get_persistent_data()

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
        
        # [優化加入] 大盤出貨日統計 (加入防呆機制避免 Volume 丟失導致報錯)
        idx_df['PriceChange'] = idx_df['Close'].pct_change()
        if 'Volume' in idx_df.columns:
            idx_df['VolumeSMA50'] = idx_df['Volume'].rolling(window=50).mean()
            idx_df['Is_Distribution'] = (idx_df['PriceChange'] <= -0.002) & (idx_df['Volume'] > idx_df['Volume'].shift(1))
            idx_df['Dist_Days_20d'] = idx_df['Is_Distribution'].rolling(window=20).sum()
        else:
            idx_df['Dist_Days_20d'] = 0
            
        return idx_df
    except:
        return None

# ==========================================
# 3. 高動能股票池 & 港股整手數配置
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

# [優化加入] 港股熱門整手數定義，供實盤資金試算與回測使用
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
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean() # [優化加入] 用作動能股退場防守線
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

    # J Law 出貨日
    df['PriceChange'] = df['Close'].pct_change()
    df['Is_Distribution'] = (
        (df['PriceChange'] <= 0.002) & 
        (df['Volume'] > df['Volume'].shift(1)) & 
        (df['Volume'] > df['VolumeSMA50'])
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

    # [修復崩潰點] 確保合併時只向真正存在的欄位添加後綴，徹底避免 KeyError
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
        
        # 由於只有 'Close' 欄位重疊，其他欄位不會被加上 'Bench' 尾碼，這裡使用 get 安全取值
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
# 6. 快速歷史勝率計算
# ==========================================
def quick_historical_win_rate(df, market_type):
    long_trades, short_trades = [], []
    position = None
    entry_price, stop_loss, tp_target, entry_structural_sl = 0, 0, 0, None

    for i in range(200, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        is_hk_liquid = not (market_type == "HK" and prev['TurnoverMA20'] < 20_000_000)

        if position is None:
            is_kk_trigger = prev['Ret1M'] > 0.15 and prev['Close'] > prev['SMA10'] and prev['VCPSignal'] and is_hk_liquid
            is_long_trigger = prev['Close'] > prev['SMA50'] and prev['SMA50'] > prev['SMA200'] and prev['MACD'] > prev['MACDSignal'] and 50 < prev['RSI'] < 75 and is_hk_liquid
            is_short_trigger = prev['Close'] < prev['SMA50'] and prev['SMA50'] < prev['SMA200'] and prev['MACD'] < prev['MACDSignal'] and 25 < prev['RSI'] < 50 and is_hk_liquid
            atr_val = prev['ATR'] if pd.notna(prev['ATR']) and prev['ATR'] > 0 else row['Open'] * 0.03

            if is_kk_trigger or is_long_trigger:
                position = "LONG"
                entry_price = row['Open']
                stop_loss = entry_price - 2 * atr_val
                tp_target = entry_price + 3 * atr_val
            elif is_short_trigger:
                position = "SHORT"
                entry_price = row['Open']
                stop_loss = entry_price + 2 * atr_val
                tp_target = entry_price - 3 * atr_val

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
# 7. 核心診斷引擎
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
            j_law_pass = all([jl1, jl2, jl3, jl4])
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
            j_law_pass = all([jl1, jl2, jl3, jl4])
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

        sr1 = bool(last['Close'] < last['SMA200'])
        sr2 = bool(last['SMA50'] < last['SMA150'] and last['SMA150'] < last['SMA200'])
        sr3 = bool(last['Close'] < high_52w * 0.85)
        sr4 = bool(last['Mansfield_RS'] < 0)
        sr5 = bool(last['VolRatio'] < 0.8)
        sr6 = bool(last['LowerShadowRatio'] < 0.4)
        sr7 = bool(last['VCPSignal'])

        short_checklist = {
            "1. 價格跌破 SMA200": sr1,
            "2. SMA50/150/200 空頭排列": sr2,
            "3. 距高位崩塌 >15%": sr3,
            "4. RS 弱於大盤": sr4,
            "5. 量能收縮": sr5,
            "6. 無明顯下影承接": sr6,
            "7. 弱勢壓縮": sr7
        }
        short_pass_count = sum([sr1, sr2, sr3, sr4, sr5, sr6, sr7])

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

        # [優化加入] 提早豁免與突破分流邏輯
        momentum_ready = (last['RSI'] > 55) and (last['Close'] > last['SMA20'])
        is_tight_setup = last['VCPSignal'] and (last['VolRatio'] < 0.8)
        p_high_20 = df['Close'].tail(20).max()
        is_breakout = (last['Close'] >= p_high_20 * 0.99) and (last['VolRatio'] > 1.5)
        early_stage_bypass = (last['Close'] > last['SMA50']) and (last['Ret1M'] > 0.20)

        # [優化加入] is_long_trigger 整合新邏輯
        is_long_trigger = momentum_ready and (
            (pass_count >= strictness_threshold) or early_stage_bypass
        ) and (is_tight_setup or is_breakout)

        is_short_trigger = last['Close'] < last['SMA50'] and last['SMA50'] < last['SMA200'] and last['MACD'] < last['MACDSignal'] and 25 < last['RSI'] < 50
        is_long_candidate = pass_count >= strictness_threshold
        is_short_candidate = short_pass_count >= strictness_threshold
        is_cohodes_short = ms_score >= 3 and last['Close'] < last['SMA50']
        is_rebound_short = ms_metrics["rebound_resist"] and ms_metrics["low_vol_rebound"]
        
        is_deduction_pullback = False
        if last.get('Clock_2', False):
            if (last['Close'] < last['SMA20'] and last['Close'] > last.get('SMA20_Deduction', 0)) or \
               (last['Close'] < last['SMA50'] and last['Close'] > last.get('SMA50_Deduction', 0)):
                is_deduction_pullback = True

        if is_kk_htf:
            plan_type = "LONG"
            status = "🔥 Kullamägi 高階動能旗型"
        elif is_deduction_pullback:
            plan_type = "LONG"
            status = "🟡 趨勢回撤買點 (2點鐘方向，抵扣價防守成功)"
        # [優化加入] 獨立的 Breakout 與 Setup 分流判斷
        elif is_long_trigger and is_breakout:
            plan_type = "LONG"
            status = "🟢 放量強勢突破 (Breakout)"
        elif is_long_trigger and is_tight_setup:
            plan_type = "LONG"
            status = "🟢 極限壓縮潛伏 (Setup)"
        elif is_long_trigger:
            plan_type = "LONG"
            status = "🟢 20MA 回踩做多信號"
        elif is_long_candidate:
            plan_type = "LONG"
            status = "🟡 強勢多頭排列"
        elif is_cohodes_short:
            plan_type = "SHORT"
            status = "🏴‍☠️ Cohodes 絕佳空點 (破位+反彈無力)"
        elif is_rebound_short:
            plan_type = "SHORT"
            status = "🩸 弱勢反彈受阻 (高勝率空點)"
        elif is_short_trigger:
            plan_type = "SHORT"
            status = "🔴 空頭破位做空信號"
        elif is_short_candidate:
            plan_type = "SHORT"
            status = "🟠 強勢空頭排列"
        else:
            plan_type = "WATCH"
            status = "⚪ 趨勢中性 / 觀望"

        potential_long_sl = target_entry - 2 * atr_val
        potential_long_tp = target_entry + 3 * atr_val
        potential_short_sl = target_entry + 2 * atr_val
        potential_short_tp = target_entry - 3 * atr_val

        if pass_count >= short_pass_count or plan_type == "LONG":
            stop_loss = potential_long_sl
            take_profit = potential_long_tp
        else:
            stop_loss = potential_short_sl
            take_profit = potential_short_tp

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
        
        # [優化加入] 階梯式持倉狀態追蹤
        position_stage = 0 
        partial_tp_target = 0
        
        shares = 0
        trade_log, history_equity, history_dates = [], [], []

        slippage_fee = 0.0025 if market_type == "HK" else 0.0005

        def apply_intraday_risk_penalty(raw_exit_price, row, side):
            intraday_range = max(float(row['High'] - row['Low']), 0.0)
            penalty = intraday_range * 0.15
            if side == "LONG":
                return max(0.01, raw_exit_price - penalty), penalty
            else:
                return raw_exit_price + penalty, penalty

        for i in range(1, len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            p_prev20 = df.iloc[i - 20] if i >= 20 else df.iloc[0]

            is_hk_liquid = not (market_type == "HK" and prev_row['TurnoverMA20'] < 20_000_000)

            is_market_bullish = True
            if 'CloseBench' in prev_row and 'SMA50BenchValue' in prev_row and pd.notna(prev_row['CloseBench']):
                is_market_bullish = prev_row['CloseBench'] > prev_row['SMA50BenchValue']

            # [優化加入] 大盤 O'Neil 防禦閘門
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

            if is_relaxed:
                is_long_trigger = prev_row['Close'] > prev_row['SMA50'] and prev_row['MACD'] > prev_row['MACDSignal'] and is_hk_liquid and is_market_bullish and is_market_health_ok
                is_short_trigger = prev_row['Close'] < prev_row['SMA50'] and prev_row['MACD'] < prev_row['MACDSignal'] and is_hk_liquid and not is_market_bullish
            else:
                p_low = df.iloc[:i]['Low'].min()
                p_high = df.iloc[:i]['High'].max()

                r1 = prev_row['Close'] > prev_row['SMA150'] and prev_row['Close'] > prev_row['SMA200']
                r2 = prev_row['SMA150'] > prev_row['SMA200']
                r3 = prev_row['SMA200'] > p_prev20['SMA200']
                r4 = prev_row['SMA50'] > prev_row['SMA150'] and prev_row['SMA50'] > prev_row['SMA200']
                r5 = prev_row['Close'] > p_low * 1.25
                r6 = prev_row['Close'] > p_high * 0.70
                r7 = prev_row['Mansfield_RS'] > 0
                prev_pass_count = sum([r1, r2, r3, r4, r5, r6, r7])

                kk_flag = prev_row['Ret1M'] > 0.15 and prev_row['Close'] > prev_row['SMA10'] and prev_row['VCPSignal']
                
                # [優化加入] 動態與提早豁免邏輯 (針對 Backtest)
                momentum_ready = (prev_row['RSI'] > 55) and (prev_row['Close'] > prev_row['SMA20'])
                is_tight_setup = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)
                p_high_20_slice = df.iloc[max(0, i-20):i]
                p_high_20_val = p_high_20_slice['Close'].max() if not p_high_20_slice.empty else prev_row['Close']
                is_breakout = (prev_row['Close'] >= p_high_20_val * 0.99) and (prev_row['VolRatio'] > 1.5)
                early_stage_bypass = (prev_row['Close'] > prev_row['SMA50']) and (prev_row['Ret1M'] > 0.20)

                is_long_trigger = momentum_ready and (
                    (prev_pass_count >= score_threshold) or early_stage_bypass or kk_flag
                ) and (is_tight_setup or is_breakout) and is_hk_liquid and is_market_health_ok

                is_short_trigger = prev_row['Close'] < prev_row['SMA150'] and prev_row['Mansfield_RS'] < 0 and is_hk_liquid and not is_market_bullish

            atr_val = prev_row['ATR'] if pd.notna(prev_row['ATR']) and prev_row['ATR'] > 0 else row['Open'] * 0.03

            if position_type is None:
                if is_long_trigger and trade_mode in ["全部雙向", "僅做多頭(做多)"]:
                    position_type = "LONG"
                    entry_price = row['Open'] * (1 + slippage_fee)

                    risk_per_trade = 0.01
                    risk_amount = current_equity * risk_per_trade
                    risk_per_share = 2 * atr_val
                    target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                    if (target_shares * entry_price) > cash:
                        target_shares = int(cash / entry_price)

                    # [優化加入] 港股實盤整手計算，避免虛假過度擬合
                    if market_type == "HK":
                        lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                        target_shares = (target_shares // lot_sz) * lot_sz
                        if target_shares == 0:
                            position_type = None # 資金不足以承擔一手風險，跳過該筆交易
                            continue

                    shares = target_shares
                    cash = cash - shares * entry_price
                    stop_loss = entry_price - risk_per_share
                    
                    # [優化加入] 取代原始單一止盈，轉用 KK 階梯式持倉
                    position_stage = 0
                    partial_tp_target = entry_price + (2.5 * atr_val)
                    
                    entry_structural_sl = prev_row.get("SupShort", entry_price * 0.9)
                    trade_info = {"方向": "🟢 做多", "進場日期": current_date, "進場價": entry_price, "初始止損": stop_loss, "最終止盈": "動態 EMA 追蹤", "結構止損": entry_structural_sl}

                elif is_short_trigger and trade_mode in ["全部雙向", "僅做空頭(做空)"]:
                    position_type = "SHORT"
                    entry_price = row['Open'] * (1 - slippage_fee)

                    risk_per_trade = 0.01
                    risk_amount = current_equity * risk_per_trade
                    risk_per_share = 2 * atr_val
                    target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

                    if market_type == "HK":
                        lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                        target_shares = (target_shares // lot_sz) * lot_sz
                        if target_shares == 0:
                            position_type = None
                            continue

                    max_short_shares = int((current_equity * 0.05) / entry_price)
                    target_shares = min(target_shares, max_short_shares)

                    if (target_shares * entry_price) > cash:
                        target_shares = int(cash / entry_price)

                    shares = target_shares
                    stop_loss = entry_price + risk_per_share
                    
                    oneil_tp = entry_price * 0.8
                    atr_tp = entry_price - (3 * atr_val)
                    tp_target = max(atr_tp, oneil_tp)
                    
                    entry_structural_sl = prev_row.get("ResShort", entry_price * 1.1)
                trade_info = {"方向": "🔴 做空", "進場日期": current_date, "進場價": entry_price, "初始止損": stop_loss, "最終止盈": tp_target, "結構止損": entry_structural_sl}

            elif position_type == "LONG":
                intraday_ma_break = pd.notna(row['SMA50']) and row['Low'] <= row['SMA50']

                # [優化加入] 情況 1：觸發防守止損 或 盤中跌穿 50MA (僅在第一階段防守)
                if row['Low'] <= stop_loss or (intraday_ma_break and position_stage == 0):
                    if row['Low'] <= stop_loss:
                        base_exit = min(row['Open'], stop_loss)
                        result_text = "觸發防守止損" if position_stage == 0 else "保本出局"
                    else:
                        base_exit = min(row['Open'], row['SMA50'])
                        result_text = "盤中跌穿 SMA50 提前平倉"

                    punished_exit, penalty = apply_intraday_risk_penalty(base_exit, row, "LONG")
                    exit_price = punished_exit * (1 - slippage_fee)

                    cash += (shares * exit_price)
                    profit = (exit_price - entry_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": penalty
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                # [優化加入] 情況 2：達到 2.5 ATR，賣出 40%，止損推至保本點 (Breakeven)
                elif position_stage == 0 and row['High'] >= partial_tp_target:
                    position_stage = 1
                    stop_loss = entry_price 
                    sold_shares = int(shares * 0.4)
                    
                    if market_type == "HK":
                        lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                        sold_shares = (sold_shares // lot_sz) * lot_sz
                        
                    if sold_shares > 0:
                        exit_price = max(row['Open'], partial_tp_target) * (1 - slippage_fee)
                        cash += (sold_shares * exit_price)
                        shares -= sold_shares
                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": exit_price,
                            "結果": "階段一止盈 (減倉鎖定)",
                            "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                            "利潤": (exit_price - entry_price) * sold_shares,
                            "盤中洗盤懲罰": 0
                        })
                        trade_log.append(partial_info)

                # [優化加入] 情況 3：主升浪跌破 EMA10 全部離場
                elif position_stage == 1 and row['Close'] < row.get('EMA10', row.get('SMA20', 0)):
                    exit_price = row['Close'] * (1 - slippage_fee)
                    cash += (shares * exit_price)
                    profit = (exit_price - entry_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": "主升浪破線動態止盈 (全平)",
                        "回報率(%)": ((exit_price - entry_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": 0
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

            elif position_type == "SHORT":
                intraday_ma_break = pd.notna(row['SMA50']) and row['High'] >= row['SMA50']

                if row['High'] >= stop_loss or intraday_ma_break:
                    if row['High'] >= stop_loss:
                        base_exit = max(row['Open'], stop_loss)
                        result_text = "盤中急拉觸發止損"
                    else:
                        base_exit = max(row['Open'], row['SMA50'])
                        result_text = "盤中升穿 SMA50 提前回補"

                    punished_exit, penalty = apply_intraday_risk_penalty(base_exit, row, "SHORT")
                    exit_price = punished_exit * (1 + slippage_fee)

                    cash += ((entry_price - exit_price) * shares)
                    profit = (entry_price - exit_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": result_text,
                        "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                        "利潤": profit,
                        "盤中洗盤懲罰": penalty
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0

                elif row['Low'] <= tp_target:
                    exit_price = min(row['Open'], tp_target)
                    exit_price = exit_price * (1 + slippage_fee)
                    cash += ((entry_price - exit_price) * shares)
                    profit = (entry_price - exit_price) * shares

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": exit_price,
                        "結果": "策略止盈",
                        "回報率(%)": ((entry_price - exit_price) / entry_price) * 100,
                        "利潤": profit
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0

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
strictness = st.sidebar.slider("量化過濾嚴格度 (預設: 5)", min_value=4, max_value=7, value=5, step=1)

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

    styled_df = df.style.map(highlight_status, subset=['J Law 強勢形態', '診斷結果', '交易方向'])
    
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

    df_long = df_results[df_results['交易方向'].astype(str).str.contains("做多", na=False)]
    df_short = df_results[df_results['交易方向'].astype(str).str.contains("做空", na=False)]
    df_watch = df_results[df_results['交易方向'].astype(str).str.contains("觀望", na=False)]

    st.markdown("---")
    st.markdown("#### ⚡ 快速無縫跳轉診斷 (防刷新防丟失)")
    st.caption("請使用下方的選單無縫跳轉，以防部分瀏覽器直接點擊表格後遺失暫存畫面。")
    
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
                    st.rerun()
                else:
                    st.warning("請先選擇或輸入代碼")
    st.markdown("---")
    
    # [佈局優化] 移除 st.columns(2)，改為全寬度上下顯示
    st.markdown("#### 🟢 做多 (Long) 候選名單")
    if not df_long.empty:
        render_styled_dataframe(df_long)
    else:
        st.info("目前沒有符合做多條件的標的。")

    st.markdown("<br>", unsafe_allow_html=True) # 增加視覺間距

    st.markdown("#### 🔴 做空 (Short) 候選名單")
    if not df_short.empty:
        render_styled_dataframe(df_short)
    else:
        st.info("目前沒有符合做空條件的標的。")

    if not df_watch.empty:
        with st.expander("⚪ 觀望名單", expanded=False):
            render_styled_dataframe(df_watch)

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

            results.append({
                "跳轉診斷": f"/?mode=診斷&ticker={ticker}",
                "代碼": ticker,
                "股票名稱": name,
                "最新價": f"{last['Close']:.2f}",
                "交易方向": tactical_dir,
                "大戶出貨日次數": int(dist_days) if pd.notna(dist_days) else 0,
                "J Law 強勢形態": jl_display,
                "距52W高點(%)": dist_str,
                "RS強度": round(rs_strength, 2),
                "全歷史勝率": display_win_rate,
                "建議進場價": entry_str,
                "ATR防守止損": sl_str,
                "ATR目標止盈": tp_str,
                "多頭分": passcount,
                "空頭分": shortpasscount,
                "成交量": f"{int(last['Volume']):,}" if pd.notna(last['Volume']) else "-",
                "VCP壓縮": "🔥 收窄" if last['VCPSignal'] else "正常",
                "診斷結果": status
            })

        progress_bar.progress((idx + 1) / len(stock_list))

    progress_bar.empty()

    if results:
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            if "J Law 強勢形態" in df_results.columns and "全歷史勝率" in df_results.columns:
                df_results['sort_win_rate'] = df_results['全歷史勝率'].str.extract(r'([0-9]+\.[0-9]+)').astype(float).fillna(0)
                df_results['sort_jlaw'] = df_results['J Law 強勢形態'].str.contains('✅|🔥|🌟').astype(int)
                df_results = df_results.sort_values(by=['sort_jlaw', 'sort_win_rate', 'RS強度'], ascending=[False, False, False])
                df_results = df_results.drop(columns=['sort_win_rate', 'sort_jlaw'])
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
            with rationale_col1:
                if is_kk_htf:
                    entry_reason = f"建議進場價：{target_entry:.2f}。屬 Kullamägi 15% 動能 + VCP 壓縮型態。"
                elif is_deduction_pullback:
                    entry_reason = f"建議進場價：{target_entry:.2f}。處於 2 點鐘穩定上漲趨勢，雖跌破短期均線，但**現價仍高於抵扣價**，均線持續向上，此為絕佳**回撤買點**！"
                elif is_near_20ma and last['VolRatio'] < 0.8:
                    entry_reason = f"建議進場價：{target_entry:.2f}。屬 J Law 20MA 回踩低量承接。"
                elif last['Close'] > last['ResShort'] * 0.98 and last['VolRatio'] > 1.2:
                    entry_reason = f"建議進場價：{target_entry:.2f}。接近突破點且量能放大。"
                elif pass_count >= strictness:
                    entry_reason = "符合 Minervini 多頭條件，RS / MACD / 趨勢排列支持。"
                elif ms_score >= 3 and last['Close'] < last['SMA50']:
                    entry_reason = "符合 Cohodes 做空法，股價跌破生命線且反彈無力，絕佳高空點。"
                elif ms_metrics["rebound_resist"] and ms_metrics["low_vol_rebound"]:
                    entry_reason = "弱勢反彈且量能萎縮，受阻於50MA，高勝率做空點。"
                elif short_pass_count >= strictness:
                    entry_reason = "符合空頭趨勢條件，可留意反彈做空機會。"
                else:
                    entry_reason = "未達最優觸發條件，建議觀察等待。"
                st.info(entry_reason)

            with rationale_col2:
                exit_reason = f"止損 {stop_loss:.2f}；止盈 {take_profit:.2f}。以 ATR 模型推算，並結合當前趨勢條件。"
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
            
            sr4.metric("🟢 長期支撐 (60日低點)", f"{last['SupLong']:.2f}")
            sr5.metric("🔴 長期阻力 (60日高點)", f"{last['ResLong']:.2f}")

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

            st.markdown("### 🎯 ATR 交易計畫 (系統理論值)")
            p1, p2, p3, p4 = st.columns(4)
            if plan_type == "WATCH":
                p1.metric("建議進場", "觀望")
                p2.metric("ATR 防守止損", "-")
                p3.metric("ATR 目標止盈", "-")
                p4.metric("交易方向", "-")
            else:
                p1.metric("建議進場", f"{target_entry:.2f} {currency}")
                sl_diff = stop_loss - target_entry
                p2.metric("ATR 防守止損", f"{stop_loss:.2f} {currency}", delta=f"{sl_diff:.2f}", delta_color="inverse")
                tp_diff = take_profit - target_entry
                p3.metric("ATR 目標止盈", f"{take_profit:.2f} {currency}", delta=f"{tp_diff:.2f}")
                p4.metric("交易方向", "🟢 做多" if plan_type == "LONG" else "🔴 做空")

            # 實戰持倉防守與風控設定
            st.markdown("### 🛡️ 實戰持倉防守與風控設定")
            st.info("💡 系統預設顯示的是『今日進場』的理論止盈損。如果你**已經建倉**，請在下方輸入你的實際進場價，圖表上的紅綠虛線將會立刻為你錨定固定風控！")
            
            col_pos1, col_pos2 = st.columns(2)
            with col_pos1:
                actual_entry = st.number_input("📌 你的實際進場價 (輸入 0 則隱藏此面板，圖表恢復預設):", value=0.0, step=0.1)
            with col_pos2:
                pos_dir = st.selectbox("🔄 你的持倉方向:", ["做多 (Long)", "做空 (Short)"])
                
            chart_entry = target_entry
            chart_sl = stop_loss
            chart_tp = take_profit
            chart_plan = plan_type
                
            if actual_entry > 0:
                entry_atr = last['ATR'] if pd.notna(last['ATR']) else actual_entry * 0.03
                if pos_dir == "做多 (Long)":
                    fixed_sl = actual_entry - 2 * entry_atr
                    fixed_tp = actual_entry + 3 * entry_atr
                    trailing_sl = max(last.get('SMA20', 0), last.get('SupShort', 0))
                    
                    st.success(f"**🟢 錨定計畫 (做多)**：固定防守止損 **{fixed_sl:.2f}** ｜ 目標止盈 **{fixed_tp:.2f}**")
                    st.warning(f"**🏃 動態跟蹤止損 (Trailing Stop)**：目前趨勢防守底線為 **{trailing_sl:.2f}** (20MA或近期支撐)，跌破建議減倉或離場。")
                    
                    chart_entry, chart_sl, chart_tp, chart_plan = actual_entry, fixed_sl, fixed_tp, "LONG"
                else:
                    fixed_sl = actual_entry + 2 * entry_atr
                    fixed_tp = actual_entry - 3 * entry_atr
                    trailing_sl = min(last.get('SMA20', 99999), last.get('ResShort', 99999))
                    
                    st.error(f"**🔴 錨定計畫 (做空)**：固定防守止損 **{fixed_sl:.2f}** ｜ 目標止盈 **{fixed_tp:.2f}**")
                    st.warning(f"**🏃 動態跟蹤止損 (Trailing Stop)**：目前趨勢防守上限為 **{trailing_sl:.2f}** (20MA或近期阻力)，突破建議減倉或離場。")
                    
                    chart_entry, chart_sl, chart_tp, chart_plan = actual_entry, fixed_sl, fixed_tp, "SHORT"
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
                    "<span style='color:#8D6E63; font-weight:bold;'>🟤 200 MA</span><br>"
                    "📊 **指數均線 (EMA)**：同色系但為**虛線(Dash)**", 
                    unsafe_allow_html=True
                )
            with col_chart_opt2:
                # 對數座標切換，預設日線開啟
                use_log_scale = st.checkbox("📐 開啟對數座標 (Log Scale) - 觀察長期斜率必備", value=True if selected_interval in ["1d", "1wk"] else False)
            
            plot_df = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
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
                if 'SMA200' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['SMA200'].ffill(), name='MA 200', line=dict(color='#8D6E63', width=2)), row=1, col=1)

                # 🌟 EMA 指數均線 (虛線，用於雙均線系統)
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
                if chart_plan != "WATCH":
                    fig.add_hline(y=chart_tp, line_dash="dash", line_color="green", annotation_text=f"TP {chart_tp:.2f}", annotation_position="top left", row=1, col=1)
                    fig.add_hline(y=chart_entry, line_dash="dot", line_color="blue", annotation_text=f"Entry {chart_entry:.2f}", annotation_position="top left", row=1, col=1)
                    fig.add_hline(y=chart_sl, line_dash="dash", line_color="red", annotation_text=f"SL {chart_sl:.2f}", annotation_position="top left", row=1, col=1)

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

                # 修復對數座標崩潰：動態計算合理的 Y 軸上下界，強制限制顯示範圍
                if use_log_scale:
                    try:
                        valid_lows = plot_df['Low'].replace(0, np.nan).dropna()
                        valid_highs = plot_df['High'].replace(0, np.nan).dropna()
                        if not valid_lows.empty and not valid_highs.empty:
                            min_y = valid_lows.min() * 0.8
                            max_y = valid_highs.max() * 1.2
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
    st.title("📈 策略歷史回測系統")
    st.markdown("以 1% 風險倉位控制、滑點、Benchmark Filter、**盤中極限洗盤懲罰**進行更貼近實戰的歷史驗證。")

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
                mb1.metric("策略總回報率", f"{(final_cap - bt_capital) / bt_capital * 100:.2f}%")
                mb2.metric("整體交易勝率", f"{overall_win_rate:.1f}%", f"共 {len(trades_df)} 筆")
                mb3.metric("帳戶最終淨值", f"{final_cap:,.2f} {cur}")

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
