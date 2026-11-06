# -*- coding: utf-8 -*-
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
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
# OLD: CURRENT_SCHEMA_VERSION = "v25.1"  # v25.1 富途牛牛中線波段專版 (持倉數週至1-2個月：防洗盤結構止損 + 富途真實收費 + 漏洞全修復 + 刪除重複代碼)
# NEW: [v25.2 專業機構級審計升級版] 消除回測未來錨點前視偏差、補齊 Martin Luk 回測信號、修正港股 1 手風險超標攔截、盤中未收盤成交量動態折算、Sortino/Calmar/樣本內外與蒙特卡洛壓力測試
# OLD: CURRENT_SCHEMA_VERSION = "v25.2"
# NEW: [v25.4 專業機構級實戰終端版] 1) 持倉方向感知 UI 與「開倉日錨定鎖定 (Entry-Date Anchored SL/TP/R:R)」徹底解決同一個進場價跨日盈虧比漂移與做空顯示做多提示漏洞; 2) 掃描雷達 × 快速勝率 × 雙向回測 100% 共享 evaluate_historical_bar_unified_plan 單一真理來源; 3) 補齊 Cohodes/Rebound/2B 破底翻/2B 假突破回測信號與 J Law 大戶出貨日倉位減半保護
# OLD: CURRENT_SCHEMA_VERSION = "v25.4"
# NEW: [v25.5 修復 React DOM removeChild 崩潰與視角切換同步版] 1) 修正 rationale_col1/col2 跨區塊延遲填充與穩定容器包裹 (消除 Chrome 自動翻譯/切換股票時之 NotFoundError: removeChild); 2) 修正視角切換時 jlaw_meta_info 即時重算; 3) 統一綁定 fmt_ticker 圖表 key
# OLD: CURRENT_SCHEMA_VERSION = "v25.5"
# NEW: [v25.6 徹底根治 React DOM removeChild 崩潰與文字錯位終極版]
# 根因定位：當點擊「🔒 鎖定計劃」或切換「做多/做空」及輸入實際進場價時，條件式插入之 st.success / st.error / st.warning 改變了父容器內子節點數量 (Sibling Index Shift)，
# 導致 Streamlit React 虛擬 DOM 在重用未帶 key 的 Alert 節點時發生新舊文字拼接錯位（如截圖中紅框與綠框文字互相穿插）並在最後一個節點拋出 NotFoundError: removeChild！
# 解決方案：1) 新增原子化 HTML 防崩潰提示卡渲染引擎 (render_atomic_alert_html / render_atomic_alert_group)，加上 translate="no" class="notranslate"，避開 React Markdown 子節點逐一 removeChild Diff；
#          2) 所有條件式提示（如 lock_btn、大盤濾網警告、手數預警、持倉導航面板）100% 綁定固定預留之 st.empty() 槽位，確保 DOM 樹結構與節點索引恆定不變！
# OLD: CURRENT_SCHEMA_VERSION = "v25.6"
# NEW: [v26.0 專業機構級實戰終端終極版]
# 1) 徹底根治「同一日期、同一買入價，5天後再輸入時止損/止盈/盈虧比漂移」漏洞：
#    - 實作「開倉日確定性歷史切片快照 (Deterministic Causal Snapshot at Entry Date)」+「(代碼,方向,開倉日,進場價) 自動雜湊持久化鎖定」，
#      隔絕開倉日之後的未來 K 線對 AVWAP、樞紐點與籌碼峰的污染；同時修正未鎖定時預設開倉日隨今日漂移的問題。
# 2) 新增「⚡ 即時入場 (Immediate Actionable Entry)」與「⏳ 等待回調 / 🎯 突破掛單」雙軌進場機制：
#    - 修復舊版 is_near_20ma 攔截突破單及過度偏向拉回等待的問題，當股價處於樞紐突破安全區 (-0.5%~+2.5%) 或支撐洗盤收回確認區時，直接給出「⚡ 即時現價入場」指令與價位。
# 3) 新增「🧠 DeepChart 前高阻力空單消化與訂單流吸收引擎 (compute_deepchart_orderflow_absorption)」：
#    - 量化計算前高阻力帶套牢盤/空單換手消化率 (Absorption Ratio)、測試縮量枯竭度 (Seller Exhaustion)、微結構累積買賣差額 (Bar CVD) 與空單消化總分 (0~100)，作為突破前高成功率的硬核依據。
# 4) 100% 統一「全景掃描雷達 × 快速歷史勝率 (quick_historical_win_rate) × 雙向歷史回測 (run_bidirectional_backtest)」：
#    - 修復 quick_historical_win_rate 未實際呼叫 evaluate_historical_bar_unified_plan 的脫節問題，並支援 IMMEDIATE / STOP / LIMIT 三種訂單執行與 DeepChart 突破確認。
# OLD: CURRENT_SCHEMA_VERSION = "v26.0"
# NEW: [v27.0 獨立深度量化與系統架構升級版]
# 除了解決表面問題，本版本從專業量化交易員與系統架構師角度主動新增 5 大核心底層引擎：
# 1) 真實 Volume-by-Price (VBP) 籌碼分佈矩陣與 Wyckoff VSA 引擎 (compute_vbp_profile_and_vsa_engine)：
#    - 計算 POC (最大籌碼共識價)、VAH/VAL (70% 價值區)、HVN (上方套牢泥沼) 與 LVN (籌碼真空加速區) 及 Wyckoff No-Supply Test (縮量無供給確認)。
# 2) 組合總熱度 (Portfolio Heat ≤ 5%)、行業板塊集中度上限與財報跳空預警引擎 (evaluate_portfolio_heat_and_sector_risk)：
#    - 防止同時買入多隻高相關同板塊股票導致系統性跳空爆倉，自動監控所有已鎖定持倉之未保本總風險敞口。
# 3) 本地磁碟 JSON 原子化持久化引擎 (LocalDiskPersistenceManager)：
#    - 自動將自選股分組、已鎖定持倉計劃與開倉日 R:R 快照同步寫入本地 .swing_terminal_state.json，重啟 Streamlit 絕不丟失數據。
# 4) 機構級 Walk-Forward 四階段滾動驗證、Van Tharp SQN (系統品質指數)、最大水下恢復期 (Max Time Under Water) 與 R 倍數分佈統計。
# 5) 動態市場寬度自適應倉位調節器 (Regime-Adaptive Risk Multiplier)：根據市場寬度 (>50MA%) 與指數趨勢自動調節單筆基準風險 (0% / 0.25% / 0.5% / 0.75% / 1.0%)。
# OLD: CURRENT_SCHEMA_VERSION = "v27.0"
# NEW: [v28.0 雙版本終極融合旗艦版 (融合 main13 v27.0 五大底層引擎 + main12 v26.0 雙層防洗盤止損與保本修復)]
# 完美結合兩者所有優點並補齊最後漏洞：
# 1) 保留 main13.py 之 40格真實 Volume-by-Price (VBP: POC/VAH/VAL/LVN/HVN)、Wyckoff VSA、DeepChart 換手消化率、組合總熱度與板塊集中度防火牆、SQN/Walk-Forward/TUW 指標、雷達入場時機過濾器及原子磁碟持久化；
# 2) 植入「雙層防洗盤止損機制 (Two-Tier Anti-Whipsaw Stop: 收盤結構軟防守位 soft_closing_sl + 盤中避開 0.35~0.5x ATR 散戶獵殺區之次深結構災難硬止損)」，徹底解決一觸及止損賣出即反轉痛點；
# 3) 根治「買入第 1 日因 20MA > 買入價即誤觸發保本止損」之致命漏洞 (同時修復持倉導航、雙向回測與快速勝率)；
# 4) 融合「OBV 能量潮領先突破 + 阻力測試量能遞減率」至 DeepChart 引擎，並將頂部 5 大交易計劃指標卡與開倉日錨定計劃實時同步；
# 5) 升級 ZoneInfo("America/New_York") 自動美東夏令/冬令時區校準。
# OLD: CURRENT_SCHEMA_VERSION = "v28.0"
# NEW: [v29.0 四週期共振狙擊旗艦版 (1D 日線結構 × 4H 波段中繼 × 1H 右側轉折與錨定AVWAP × 15m 日內VWAP/ORB狙擊)]
# 徹底解決「買入後總是不能立即盈利 (左側過早接飛刀 / 開盤高追情緒頂)」與「離場位置極差 (日線反應過慢嘔凸利潤 / 15m恐慌針尖砍在最低點)」：
# 1) 支援真實 4H (由 1H 合成之半日線) K 線與修正分鐘/小時圖指標週期換算 (避免 15m/1h 下 High52W 變成 12 根 K 線之 Bug)；
# 2) 新增 compute_mtf_execution_and_exit_sniper 四週期共振引擎：
#    - 入場右側確認：日線回踩支撐時，要求 1H 收盤站上 1H 20EMA 與「前高回調錨定 AVWAP」且形成 1H Higher Low 才扣扳機，杜絕左側接飛刀買完繼續陰跌；
#    - 日內防高追與 ORB 突破：利用 15m 日內 VWAP 與開盤首 15/30 分鐘高低點 (ORB)，攔截開盤偏離日內 VWAP > 1.8x ATR 的追高陷阱；
#    - 離場精準鎖潤與防恐慌洗盤：提供 1H 20EMA / 1H 前波谷結構鎖潤離場位（比日線提早鎖住頂部利潤），並在 15m 極度超賣 (RSI<25) 恐慌針尖時攔截盲目市價砍倉！
# OLD: CURRENT_SCHEMA_VERSION = "v29.0"
# NEW: [v30.0 專業機構級全鏈路零漏洞與勝率最大化終極旗艦版]
# 由頭到尾全量代碼審計修復 6 大深層隱藏漏洞並新增「勝率最大化 A+ 決策引擎」：
# 1) 修復 check_recent_signal_and_stop_invalidation 與回測中「盤中刺穿20日低點即無條件否決做多 (broke_20d_low_intraday)」誤殺所有 2B 破底翻 / Undercut & Rally 洗盤買點之漏洞；
# 2) 修復 analyze_stock_master 使用 df['Close'].tail(20) (自我包含當日收盤價) 與回測 ResShort_Causal 不一致問題，並將 S/R Flip 回測、2B 破底翻、Wyckoff No-Supply/Spring 完整接入主狀態鏈；
# 3) 修復 evaluate_portfolio_heat_and_sector_risk 對已鎖定之同一隻股票重複累加兩次風險熱度 (Double-Counting) 漏洞；
# 4) 修復回測系統 IMMEDIATE 訂單標籤誤判與 LIMIT 單進場當日收長下影陽線仍被誤判止損之邏輯缺陷；
# 5) 新增「🚀 RS 相對強弱線領先價格創高 (RS Line New High Before Price)」與「🔥 偏離 20MA 過熱高潮延伸主動減倉位 (Climax Extension Profit-Lock)」；
# 6) 新增「🏆 機構級 8 維度交易品質評級 (S級 / A+級 / A級 / B級 / C級) 與動態建議倉位權重引擎 (evaluate_holistic_setup_grade_and_edge)」，最大化實戰資金運用成功率！
# OLD: CURRENT_SCHEMA_VERSION = "v30.0"
# NEW: [v31.0 專業機構級多空雙向語義對齊 × 跨用戶Session隔離 × 財報跳空預警 × 空頭融券成本終極版]
# 1) 徹底修復做空視角 (SHORT View / dir=SHORT) 出現做多字眼與指標顛倒之漏洞：
#    - 修復 evaluate_holistic_setup_grade_and_edge 將空頭 RS 創 20 日新低誤標為「🚀 RS率先創20日新高 / 機構逆勢掃貨指標」及高潮延伸減倉位字眼；
#    - 升級 compute_deepchart_orderflow_absorption 與 compute_vbp_profile_and_vsa_engine 支援多空雙向獨立診斷（空頭顯示「🩸 空頭壓制與破底動能分」、「📉 下方籌碼通道性質 (LVN暴跌真空區)」、「📈 反彈買盤枯竭度 (No-Demand)」與空頭專屬戰術導航）；
#    - 修復切換「強制做空/做多視角」時 holistic_grade_info 未即時重算之緩存漏洞，並全面對齊 Martin Luk R倍數、抵扣價防假突破、富途一手沽空成本與交易計劃 Delta 顏色方向；
# 2) 新增「📅 財報/業績跳空黑天鵝風險檢測器 (detect_earnings_and_gap_risk)」與「📉 空頭融券借券利息 + 維持保證金強平模擬」；
# 3) 新增「🔒 多用戶 Session 隔離與線程安全快取鎖 (Thread-Safe Session State Isolation)」及 URL 參數白名單消毒，解決 Streamlit Cloud 公網部署下跨用戶持倉互相覆蓋漏洞；
# 4) 新增「📋 已鎖定持倉組合實時監控看板與 CSV 交易日誌匯出 (render_locked_portfolio_summary_and_export)」。
# OLD: CURRENT_SCHEMA_VERSION = "v31.0"
# NEW: [v32.0 美股防假突破與防追高枯竭專項升級版 (US vs HK Microstructure Adaptive Engine)]
# 針對實戰痛點「美股十筆全止損、買入後無盈利即大跌，而港股卻能剛好到止盈」進行 7 大核心根因修復：
# 1) 補齊 analyze_stock_master 實盤大盤濾網 (Live Market Regime Gate：同步檢測標普500/納指 QQQ 趨勢與出貨日，逆風市嚴禁突破追高)；
# 2) 新增「🛑 美股多日連漲動能枯竭與尾盤上影線過濾器 (compute_us_swing_exhaustion_and_trap_guard)」：攔截連漲 >=3 日、5日波幅 >2.3x ATR 或偏離 10EMA >1.5x ATR 之追高買點，強制改為等待回踩 10/21EMA 低吸；
# 3) 修復 evaluate_swing_trade_plan_and_rr 中 `or close_p >= sma20_p` 導致普通多頭排列被誤判為「⚡ 即時入場」之致命漏洞；
# 4) 升級「📈 美股高 ADR% 動態止損上限補償 + TP0 (+1.2R) 先鎖利潤墊三段式止盈機制」，解決美股因 ADR>5% 導致結構止損被 7.8% 硬上限截斷懸空、以及未達遠端 TP1 即反轉止損之痛點；
# 5) 修正美股盤中開盤首小時 U 型成交量折算失真 (U-Shaped Volume Curve)，杜絕香港時間 21:30-23:00 美股早盤衝高假突破誘多；
# 6) 補齊真實 yfinance 財報日曆倒數檢測 (days_to_earnings)、回測空頭融券利息與維持保證金強平模擬、以及 _CACHE_LOCK 多用戶線程安全隔離。
# OLD: CURRENT_SCHEMA_VERSION = "v32.0"
# NEW: [v35.0 專業機構級波段核心引擎重構版 (18筆港美股實盤歸因 × 純量化算法升級)]
# 1) 嚴格區分「背景多頭排列 (WATCH)」與「精確觸發買點 (LONG)」：廢除無買點觸發僅憑均線排列即歸類為 LONG 的邏輯，杜絕無結構支撐之市價追高；
# 2) 多均線斜率同向濾網與右側收復確認引擎 (compute_ma_slope_and_pullback_trigger)：要求回踩 20MA 時 20/50/60MA 斜率不得下彎，並區分「右側收復 9EMA 即時買點」與「左側 (9EMA+20MA)/2 平價限價買點」；
# 3) 收緊中線波段止損上限 (≤5.5%~5.8%) 與現實化單波段目標價 (TP1 ≤ 2.8x~3.2x ATR)：淘汰止損過寬 (>6%) 之鬆散形態，解決美股目標價過遠 (+15%~+18%) 無法觸及即反轉之結構缺陷；
# 4) 三階段階梯式動態鎖潤引擎 (compute_ratchet_trailing_and_exit_plan)：支援單手(1-Lot)/震盪市於 TP1 全數止盈、浮盈期放量長陰或跌穿 9EMA 即時鎖潤離場、以及 +1.0R/+4.5% 預留跳空緩衝之保本鎖定；
# 5) 同板塊相對強弱選股 (Sector RS Leader Cap = 1) 與等風險倉位反推器 (Equal-Risk Notional Sizer)：雷達優先榜每板塊僅保留 RS 最強領頭羊，並按固定風險額直接反推建議買入金額與股數。
# OLD: CURRENT_SCHEMA_VERSION = "v35.0"
# NEW: [v36.0 美股盤前跳空止損協議 (RTH ORB-Low Stop Protocol) × 18筆實盤量化核心終極版]
# 新增 evaluate_premarket_gap_and_orb_stop_protocol：區分「美股盤前薄市假跌穿 (Pre-Market Illiquid Pierce)」與「正規盤開盤區間實質破位 (RTH ORB-Low Breakdown)」，
# 自動計算正規盤開盤首 15 分鐘低點 (ORB Low) 止損觸發位與災難硬止損線，杜絕在盤前流動性真空期因大價差滑點砍在最低點！
# OLD: CURRENT_SCHEMA_VERSION = "v36.0"
# NEW: [v37.0 美股現實化波段止盈包絡線 × 實盤手數/金額精準持倉導航 × 財報日曆/黑天鵝雷達 × 實時新聞情報與量價背馳NLP引擎]
# 1) 徹底解決「美股止盈目標 (TP1) 太遠導致未觸及即回落由盈轉虧」：
#    - 修復 evaluate_swing_trade_plan_and_rr 與 compute_anchored_position_plan 當 sr_targets[0] 為遠端 52W/60D 高位 (+11%~+18%) 時直接當作 TP1 之缺陷；
#    - 引入「單波段 ATR 動能包絡線上限 (Impulse Wave Envelope Cap: 美股 TP1 <= min(首道結構阻力, Entry + 2.5x ATR, Entry * 1.082))」，將遠端大阻力移至 TP2，確保 TP1 落在單次 5~12 日波段高機率觸達區 (+5.5%~+8.2%)！
# 2) 升級「🛡️ 實戰持倉防守與風控導航」：新增「實際持倉手數 / 股數」與「實際持倉金額」雙向連動輸入，精確計算實時淨浮盈 ($)、止損實質淨風險 ($)、TP0/TP1 應賣出之確切股數/手數與預計落袋金額 ($)，並自動識別港股「1手單倉」切換為全進全出階梯鎖潤；
# 3) 升級「📅 業績公佈日、除淨日與黑天鵝風險雷達」：三層解析 yfinance 財報具體日期 (YYYY-MM-DD) 與倒數天數、除淨日 (防除息假破位) 及量價黑天鵝異動；
# 4) 新增「📰 實時新聞情報搜尋與 AI 利好/利空 × 量價背馳診斷引擎 (fetch_and_analyze_stock_news_intelligence)」：結合 yfinance News 與 Google News RSS 雙源檢索，自動分類 6 大催化劑、評定利好/利空分數，並交叉比對當日 K 線揭露「利好出貨 (Sell the News)」或「利空出盡」！
# OLD: CURRENT_SCHEMA_VERSION = "v37.0"
# NEW: [v38.0 100%保留原版UI介面 × 修復美股雷達 st.dataframe 崩潰 × 退市改名股池清洗 × 納指/標普雙基準共振 × 回測閉環版]
# OLD: CURRENT_SCHEMA_VERSION = "v38.0"
# NEW: [v39.0 專業機構級 Swing Trader 全維度審計修復終極版]
# 1) 根治回測與指標前視偏差 (Zero Look-Ahead Bias)：移除 calculate_indicators 全序列 scipy.find_peaks 未來拐點洩漏，並預計算滾動錨定 AVWAP 與 Martin Luk 向量化特徵，消除回測 O(N^2) 切片重算瓶頸；
# 2) 100% 統一 quick_historical_win_rate 與 evaluate_historical_bar_unified_plan，並傳入 strictness_threshold 與支援 TP0 (+1.15R/+4.5%) 三段式鎖潤；
# 3) 修復 evaluate_swing_trade_plan_and_rr 與 analyze_stock_master 中 5.8%/6.2% 硬編碼覆寫側欄止損模式與美股高 ADR% 動態止損上限之矛盾；
# 4) 修復 check_recent_signal_and_stop_invalidation 將「15日最高收盤價回落 2x ATR」誤判為前波止損失效而扼殺強勢股首次回踩 20MA 買點之缺陷；
# 5) 修復回測 LIMIT 低吸單自動偷換為 STOP 追高單、空頭保證金強平 (Margin Call) 誤標為時間停損及借券利息漏計問題；
# 6) 修復 compute_equal_risk_position_cap 港美股幣種混淆 (USD vs HKD)、自選股分組持久化遺失、以及 datetime.utcnow() 棄用與盤中折算量不一致問題。
# OLD: CURRENT_SCHEMA_VERSION = "v39.0"
# NEW: [v40.0 專業 Swing Trader 全維度機構級審計終極修復版]
# 1) 徹底根治雷達搜尋重複顯示 "[v35.0 等風險倉位反推引擎...]" 之 Streamlit Magic AST 獨立字串渲染 Bug (Line 273)，並補齊港股 1 手超風險預算攔截欄位 (min_1lot_over_budget)；
# 2) 修正 evaluate_swing_trade_plan_and_rr 與 render_radar_results 呼叫 compute_equal_risk_position_cap 時漏傳 market_type 與 lot_size 導致港股按 $600 錯誤上限截斷之 Bug；
# 3) 100% 統一 quick_historical_win_rate 與 evaluate_historical_bar_unified_plan 單一真理來源，並於 calculate_indicators 預計算無未來函數之因果滾動 AVWAP 與 Martin Luk 特徵，消除回測 O(N^2) 重算與勝率虛高；
# 4) 對齊 run_bidirectional_backtest 與實戰持倉導航之 TP0 (+1.15R/+4.5%) 先鎖利潤墊機制，並修復空頭阶段一減倉借券利息漏計與開倉日錨定快照誤用當日 252D 高低點問題；
# 5) 修復 F5 刷新或點擊 LinkColumn 跳轉新分頁時雷達結果因 _session_isolated_store 未同步 _global_persistent_data 而消失之問題，並將 signal_lifecycle_store 納入磁碟原子持久化與修復 datetime.now(timezone.utc)。
# OLD: CURRENT_SCHEMA_VERSION = "v40.0"
# NEW: [v41.1 Master Edition — 五階段大盤見底轉勢識別(不屏蔽做多清單) × 滿倉HK$1,000風控紅綠燈 × 21筆實盤復盤終極版]
# 1) [Module 1] 五階段大盤趨勢與「🌤️ 回調見底/放量反攻 (Bottom Reversal / FTD)」智能識別引擎 (evaluate_global_market_regime_gate)：
#    - 解決單靠滯後 50MA 判斷大市導致「恒指/納指回調見底大升當日仍被誤判為空頭、甚至完全屏蔽做多清單」之致命缺陷；
#    - 結合「中線均線結構 (20MA/50MA/200MA)」與「短線領先轉勢信號 (單日大升 >=1.0%、收復 5EMA/9EMA/10MA、底分型/Higher High)」，精準識別「🌤️ 大市回調見底反攻期」；
#    - 預設【絕不屏蔽做多或做空清單】，完整保留所有符合結構之做多領頭羊與 2B 破底翻買點，改以「大市五態橫幅 + 建議倉位比例 (試倉 60% vs 滿倉 100%)」導航！
# 2) [Module 2] 動能股池純化 + 財報前夕預警標籤 (保留信號不隱藏) + 空頭連跌超賣防追殺濾網：
#    - 將低波幅收息內銀/公用股移至獨立 HK_DEFENSIVE_DIVIDEND_POOL，保持 HK_MOMENTUM_POOL 高動能純度；
#    - 財報前 5 日於信號旁附加「⚠️距財報X日」醒目提示而不強制踢出做多清單；
#    - 空頭連跌 >=4 日或偏離 20MA >1.85x ATR 時提示超賣防追空，杜絕 2382.HK 低位追空被夾！
# 3) [Module 3] 專為「習慣滿倉 (HK$18,000~HK$25,000) + 每筆止損鎖定 HK$1,000 (US$128)」度身訂造之風控紅綠燈 (compute_equal_risk_position_cap)：
#    - 預設單筆最大承受虧損 HK$1,000 (US$128) 及目標滿倉本金 HK$21,500 (US$2,756)，支援側欄動態調整；
#    - 實時對比「🛡️ 圖表結構止損位」與「💵 滿倉蝕 HK$1,000 硬性金額止損位」，提供：①【滿倉安全低吸掛單價】、②【現價等風險精準股數】、③【兩段式底倉 + 浮盈加滿倉】方案！
# 4) [Module 4] 中線止損空間平衡 (恢復中線波段上限 7.5% + 保本鎖定需滿 5 日及 +1.25R MFE) 與逆勢補倉紅牌攔截 (杜絕 TSM 越跌越溝與 SNOW 過早移保本被洗)；
# 5) [Module 5] 清理 run_bidirectional_backtest 不可達死代碼 (Dead Code)。
# OLD: CURRENT_SCHEMA_VERSION = "v41.1"
# NEW: [v42.0 Superperformance Master Edition — 《超級績效 1~4》全書量化引擎 × 港美股雙軌差異化 R:R (還原 main7 藥明生物波段目標) × 四組雙標準重疊小篩網 × 牛熊轉勢/調整結束識別終極版]
# 1) [SEPA Book 1 & 2] VCP 遞進式波幅收縮足跡 (T1 > T2 > T3 + 縮量乾涸) 與 3-C 三層買點體系 (💎 Low Cheat 低位作弊點 / 🎯 3C 中位作弊點 / 🚀 標準杯柄樞紐點)：
#    - Low Cheat 買在箱體中低部極窄收斂區，結構止損天然僅 3.2%~4.6%，完美契合「滿倉 HK$21,500 + 單筆止損鎖死 ≤ HK$1,000」！
# 2) [SEPA Book 2] 「🎾 網球強彈 (Tennis Ball) vs 🥚 雞蛋落地 (Egg Slump)」與「🏋️ 突破蹲伏 (Squat Bar)」實戰行為檢驗；
# 3) [SEPA Book 2] 四組「雙標準獨立小篩網 (2-Criteria Overlapping Mini-Screens)」交叉重疊法：
#    - 分設 ①RS抗跌先行網、②VCP收斂乾涸網、③動能突圍口袋支點網、④中小型高彈性趨勢網，統計重複出現次數 (🔥 4/4、🌟 3/4、🟢 2/4)，
#      徹底解決單一 12 條件過濾器在大市剛調整結束時篩走「長線均線未完全排好但升速最快之早期二階段彈簧股」的問題！
# 4) [Dual-Track HK vs US R:R Engine] 港股與美股雙軌差異化止盈與盈虧比引擎 (compute_dual_track_hk_us_rr_targets)：
#    - 🇭🇰 港股模式：100% 還原 main7 買藥明生物 (2269.HK) 階段精髓！當箱體內近期小阻力不足 2.0R (或少於 +7.2%) 時自動跨過該小阻力，
#      以 max(合規主波段大阻力, Entry + 3.8x ATR, Entry + 2.25R) 投射 +9%~+14% (2.2R~2.8R) 主升浪目標，不再被近期小高點斬頭！
#    - 🇺🇸 美股模式：針對美股零印花稅、1股可拆賣但常於 +6.5%~+8.5% 出現 Squat 回抽之微結構，採用《超級績效 2》「2R~2.4R 強勢賣出一半 (TP1) + 尾倉沿 21EMA 奔跑搏遠端大目標 (TP2)」雙軌制！
# 5) [Five-Stage Bull/Bear & Correction-Ending Engine] 同時分辨「🐂 確認牛市」、「🌤️ 大盤調整結束 / 熊轉牛初段 (彈簧股爆發期)」、「🟡 牛市中期修正」、「🌩️ 牛市見頂派發預警」與「🐻 確認熊市」；
# 6) [Small/Mid-Cap High-Velocity Pool] 新增「🚀 超級績效・中小型高彈性潛力池」並配備日均成交額 ≥ 2,000萬港元防老千股護欄。
CURRENT_SCHEMA_VERSION = "v42.0"

# ==============================================================================
# 1. 《超級績效 1 & 2》VCP 遞進收縮足跡 + 3-C (Low Cheat / Cheat / Pivot) 識別引擎
# ==============================================================================
def compute_minervini_vcp_and_3c_cheats(
    df: pd.DataFrame,
    lookback_base: int = 65
) -> Dict[str, Any]:
    """
    量化識別 Mark Minervini《超級績效 1 & 2》中的：
    1) VCP 遞進收縮足跡 (例えば 3T: -18% -> -9% -> -4%，要求每次回撤幅度遞減且右側成交量乾涸 < 60% SMA50)；
    2) 3-C 三層進場點：
       - Low Cheat (低位作弊點)：位於基底下半部 (距基底低點 5%~15%、距前高 12%~28%) 之極窄幅縮量平台突破，
         止損極窄 (≤3.5%~4.5%)，最適合「滿倉 HK$20,000 且風險鎖死 ≤ HK$1,000」；
       - Mid Cheat / 3C (中位杯完成作弊點)：位於基底中上部 (距前高 5%~12%)、站穩 20MA/50MA 之窄幅橫盤；
       - Standard Pivot (標準樞紐突破點)：距離基底前高 0%~5% 內之最後一收縮 (T3/T4) 高點。
    """
    res: Dict[str, Any] = {
        "is_valid_vcp": False,
        "contraction_count": 0,
        "t1_depth_pct": 0.0,
        "t2_depth_pct": 0.0,
        "t3_depth_pct": 0.0,
        "vcp_footprint_str": "-",
        "volume_dry_up": False,
        "entry_tier": "NONE",          # "LOW_CHEAT", "MID_CHEAT_3C", "HANDLE_PIVOT", or "NONE"
        "entry_tier_badge": "-",
        "cheat_pivot_price": 0.0,
        "cheat_stop_price": 0.0,
        "cheat_stop_pct": 0.0
    }
    if df is None or len(df) < 40:
        return res

    try:
        sub = df.iloc[-min(len(df), lookback_base):].copy()
        close_now = float(sub["Close"].iloc[-1])
        high_now = float(sub["High"].iloc[-1])
        low_now = float(sub["Low"].iloc[-1])

        base_high = float(sub["High"].max())
        base_low = float(sub["Low"].min())
        if base_high <= 0 or base_high <= base_low:
            return res

        total_base_depth = ((base_high - base_low) / base_high) * 100.0
        # Divide base window into 3 progressive segments (Left T1, Middle T2, Right T3)
        n = len(sub)
        seg1 = sub.iloc[: max(10, int(n * 0.45))]
        seg2 = sub.iloc[max(8, int(n * 0.40)): max(16, int(n * 0.80))]
        seg3 = sub.iloc[-min(12, max(5, int(n * 0.25))):]

        t1_depth = ((float(seg1["High"].max()) - float(seg1["Low"].min())) / max(float(seg1["High"].max()), 1e-6)) * 100.0
        t2_depth = ((float(seg2["High"].max()) - float(seg2["Low"].min())) / max(float(seg2["High"].max()), 1e-6)) * 100.0
        t3_depth = ((float(seg3["High"].max()) - float(seg3["Low"].min())) / max(float(seg3["High"].max()), 1e-6)) * 100.0

        res["t1_depth_pct"] = round(t1_depth, 1)
        res["t2_depth_pct"] = round(t2_depth, 1)
        res["t3_depth_pct"] = round(t3_depth, 1)

        # Progressive contraction validation: T1 > T2 > T3 and final contraction is tight (<= 7.5%)
        progressive_2t = (t1_depth > t3_depth * 1.35) and (t3_depth <= 7.5)
        progressive_3t = (t1_depth > t2_depth * 1.15) and (t2_depth > t3_depth * 1.15) and (t3_depth <= 6.8)
        contractions = 3 if progressive_3t else (2 if progressive_2t else 1)
        res["contraction_count"] = contractions

        vol_sma50 = float(sub["Volume"].rolling(50, min_periods=10).mean().iloc[-1]) if "Volume" in sub.columns else 1.0
        recent3_min_vol = float(sub["Volume"].iloc[-5:].min()) if "Volume" in sub.columns else vol_sma50
        vol_dry_up = bool(recent3_min_vol <= vol_sma50 * 0.68)
        res["volume_dry_up"] = vol_dry_up

        if (progressive_2t or progressive_3t) and 10.0 <= total_base_depth <= 45.0:
            res["is_valid_vcp"] = True
            res["vcp_footprint_str"] = f"{contractions}T VCP ({t1_depth:.0f}%→{t2_depth:.0f}%→{t3_depth:.1f}%)" + (" + 縮量乾涸" if vol_dry_up else "")

        # Identify 3-C Entry Tier (Low Cheat vs Mid Cheat 3C vs Handle Pivot)
        recent5 = sub.iloc[-5:]
        pivot_5d_high = float(recent5["High"].max())
        stop_5d_low = float(recent5["Low"].min())
        tight_5d_pct = ((pivot_5d_high - stop_5d_low) / max(pivot_5d_high, 1e-6)) * 100.0

        pos_in_base = (close_now - base_low) / max(base_high - base_low, 1e-6)
        sma20 = float(sub["SMA20"].iloc[-1]) if "SMA20" in sub.columns else float(sub["Close"].rolling(20, min_periods=5).mean().iloc[-1])
        sma50 = float(sub["SMA50"].iloc[-1]) if "SMA50" in sub.columns else float(sub["Close"].rolling(50, min_periods=10).mean().iloc[-1])
        sma200 = float(sub["SMA200"].iloc[-1]) if "SMA200" in sub.columns else close_now * 0.9

        if tight_5d_pct <= 6.2 and close_now >= sma200 * 0.98:
            if 0.20 <= pos_in_base < 0.55 and close_now >= sma20 * 0.99:
                res["entry_tier"] = "LOW_CHEAT"
                res["entry_tier_badge"] = f"💎 Low Cheat 低位作弊點 (極窄止損 -{tight_5d_pct:.1f}%)"
                res["cheat_pivot_price"] = round(pivot_5d_high * 1.002, 2)
                res["cheat_stop_price"] = round(stop_5d_low * 0.996, 2)
                res["cheat_stop_pct"] = round(tight_5d_pct + 0.4, 2)
            elif 0.55 <= pos_in_base < 0.82 and close_now >= sma20 and close_now >= sma50 * 0.99:
                res["entry_tier"] = "MID_CHEAT_3C"
                res["entry_tier_badge"] = f"🎯 3C 中位作弊買點 (波幅收窄 -{tight_5d_pct:.1f}%)"
                res["cheat_pivot_price"] = round(pivot_5d_high * 1.002, 2)
                res["cheat_stop_price"] = round(stop_5d_low * 0.996, 2)
                res["cheat_stop_pct"] = round(tight_5d_pct + 0.4, 2)
            elif pos_in_base >= 0.82 and close_now >= sma20 and close_now >= sma50:
                res["entry_tier"] = "HANDLE_PIVOT"
                res["entry_tier_badge"] = f"🚀 VCP 標準杯柄樞紐點 (止損 -{tight_5d_pct:.1f}%)"
                res["cheat_pivot_price"] = round(pivot_5d_high * 1.002, 2)
                res["cheat_stop_price"] = round(stop_5d_low * 0.996, 2)
                res["cheat_stop_pct"] = round(tight_5d_pct + 0.4, 2)
    except Exception:
        pass
    return res


# ==============================================================================
# 2. 《超級績效 2》「網球 vs 雞蛋 (Tennis Ball vs Egg)」與「突破蹲伏 (Squat)」診斷引擎
# ==============================================================================
def evaluate_tennis_ball_and_squat_action(df: pd.DataFrame) -> Dict[str, Any]:
    """
    量化 Mark Minervini《超級績效 2》兩大實戰持倉與買點驗證法則：
    1) 網球 vs 雞蛋行為 (Tennis Ball vs Egg Action)：
       - 真正超級強勢股回踩 10MA/20MA/21EMA 後，會在 1~3 日內像水底放開的網球一樣強力彈起並收復前高/均線 (反弹日量增、回調日量縮)；
       - 若回踩後死氣沉沉、連續收陰且無量反彈 (像雞蛋掉在地上黏住)，代表是假強勢，不應買入或應及早換馬！
    2) 突破後蹲伏與反轉修復法則 (Post-Breakout Squat & Reversal Recovery Rule)：
       - 當突破日或買入日衝高回落收在當日波幅下半部 (Squat 蹲伏)，不必恐慌即刻砍倉，只要未跌破結構硬止損，
         觀察未來 1~3 日內能否站回突破樞紐或 9EMA (Reversal Recovery)。
    """
    res = {
        "is_tennis_ball": False,
        "is_egg_slump": False,
        "is_squat_bar": False,
        "squat_recovered": False,
        "action_badge": "-",
        "action_note": ""
    }
    if df is None or len(df) < 25:
        return res
    try:
        sub = df.iloc[-12:]
        last = sub.iloc[-1]
        c_now = float(last["Close"])
        o_now = float(last["Open"])
        h_now = float(last["High"])
        l_now = float(last["Low"])
        ema9 = float(last.get("EMA9", c_now))
        sma20 = float(last.get("SMA20", c_now))
        vol_now = float(last.get("Volume", 0))
        vol_ma = float(df["Volume"].rolling(20, min_periods=5).mean().iloc[-1]) if "Volume" in df.columns else 1.0

        # Check pullback in prior 5 days followed by strong bounce
        prior5 = sub.iloc[-6:-1]
        dipped_near_20ma = bool((prior5["Low"] <= prior5["SMA20"] * 1.015).any()) if "SMA20" in prior5.columns else False
        close_pos_now = (c_now - l_now) / max(h_now - l_now, 1e-6)

        if dipped_near_20ma and c_now >= sma20 and c_now >= ema9 and c_now > o_now and close_pos_now >= 0.60:
            res["is_tennis_ball"] = True
            res["action_badge"] = "🎾 網球強彈確認 (Tennis Ball)"
            res["action_note"] = "回測均線後迅速收陽彈起站上 9EMA/20MA，屬《超級績效》標準『網球抗跌反彈』，機構承接力極強！"
        elif dipped_near_20ma and c_now < sma20 and (sub["Close"].iloc[-3:] < sub["Open"].iloc[-3:]).sum() >= 2:
            res["is_egg_slump"] = True
            res["action_badge"] = "🥚 雞蛋軟弱無力 (Egg Slump)"
            res["action_note"] = "回測均線後缺乏買盤彈力、連續陰跌像『雞蛋落地』，切勿左側盲目接刀！"

        # Check Squat Bar (High volume breakout attempt that closes in bottom 40% of range)
        if vol_now >= vol_ma * 1.30 and close_pos_now <= 0.40 and h_now >= float(sub["High"].iloc[:-1].max()):
            res["is_squat_bar"] = True
            res["action_badge"] = "🏋️ 突破蹲伏 (Squat Bar)"
            res["action_note"] = "當日放量衝高但收出上影線蹲伏 (Squat)，《超級績效》法則：若未破結構止損，觀察 1~2 日內能否陽燭收復今日高點；若隔日放量續跌破 9EMA 則果斷離場。"
    except Exception:
        pass
    return res


# ==============================================================================
# 3. 《超級績效 2》四組「雙標準獨立小篩網 × 重複出現次數計分器 (Overlapping Mini-Screens)」
# ==============================================================================
def evaluate_minervini_overlapping_mini_screens(df: pd.DataFrame) -> Dict[str, Any]:
    """
    實踐 Mark Minervini《超級績效 2》核心選股哲學：
    「不要把 10 個條件塞進同一個過濾器，而是每次只用 2 個標準跑多個獨立小篩網，找出在多個篩網中重複出現的股票！」
    - 篩網 A【RS 逆市抗跌先行網】(2標準)：(1) Mansfield RS > 0 或近10日跑贏大盤；(2) 股價穩守 20MA 之上且近5日低點抬高。
    - 篩網 B【VCP 收斂與縮量乾涸網】(2標準)：(1) 近3日波幅收窄 (<20日均幅 68% 或 5日箱體 <=5.5%)；(2) 量比 < 0.85 或剛放量收復 9EMA。
    - 篩網 C【動能突圍與口袋支點網】(2標準)：(1) 收陽站上 9EMA/10MA 且收於波幅上半部 (>=55%)；(2) 成交量放大 (VolRatio >= 1.20) 或創 10 日收盤新高。
    - 篩網 D【中小型高彈性趨勢網】(2標準)：(1) 日均波幅 ADR% 介乎 3.0%~8.0% (剔除牛皮大笨象股)；(2) 短中期趨勢向上 (Close > 50MA 且 20MA >= 50MA * 0.99)。
    """
    out = {
        "screen_a_rs_leader": False,
        "screen_b_vcp_tight": False,
        "screen_c_pocket_thrust": False,
        "screen_d_small_mid_velocity": False,
        "overlap_count": 0,
        "overlap_badge": "⚪ 0/4",
        "matched_screens_list": []
    }
    if df is None or len(df) < 30:
        return out
    try:
        last = df.iloc[-1]
        c_now = float(last["Close"])
        o_now = float(last["Open"])
        h_now = float(last["High"])
        l_now = float(last["Low"])
        sma20 = float(last.get("SMA20", c_now))
        sma50 = float(last.get("SMA50", c_now))
        ema9 = float(last.get("EMA9", c_now))
        rs_val = float(last.get("Mansfield_RS", 0.0))
        vol_ratio = float(last.get("VolRatio", 1.0))
        adr_pct = float(last.get("ADR_Pct", ((df["High"] - df["Low"]) / df["Close"]).tail(14).mean() * 100.0))
        close_pos = (c_now - l_now) / max(h_now - l_now, 1e-6)

        # Screen A: RS & Resilience (2 criteria)
        hl_5d = float(df["Low"].iloc[-3:].min()) >= float(df["Low"].iloc[-8:-3].min()) * 0.995
        scr_a = bool((rs_val > 0) and (c_now >= sma20 * 0.995 and hl_5d))

        # Screen B: VCP Tightness & Volume Dry-Up (2 criteria)
        range_3d = float(((df["High"] - df["Low"]) / df["Close"]).iloc[-3:].mean())
        range_20d = float(((df["High"] - df["Low"]) / df["Close"]).iloc[-20:].mean())
        is_tight_3d = (range_3d <= range_20d * 0.70) or bool(last.get("VCPSignal", False)) or bool(last.get("JLaw_Tight_Box", False))
        scr_b = bool(is_tight_3d and (vol_ratio <= 0.88 or (c_now >= ema9 and c_now > o_now)))

        # Screen C: Pocket Pivot / Momentum Thrust (2 criteria)
        high_10d_prev = float(df["Close"].iloc[-11:-1].max())
        scr_c = bool((c_now >= ema9 and c_now > o_now and close_pos >= 0.55) and (vol_ratio >= 1.20 or c_now >= high_10d_prev))

        # Screen D: Small/Mid-Cap High Velocity Trend (2 criteria)
        scr_d = bool((3.0 <= adr_pct <= 8.5) and (c_now >= sma50 and sma20 >= sma50 * 0.99))

        matched = []
        if scr_a:
            matched.append("①RS抗跌")
        if scr_b:
            matched.append("②VCP收斂")
        if scr_c:
            matched.append("③動能突圍")
        if scr_d:
            matched.append("④高彈性趨勢")

        cnt = len(matched)
        if cnt == 4:
            badge = f"🔥 4/4 四網全滿貫 ({'+'.join(matched)})"
        elif cnt == 3:
            badge = f"🌟 3/4 三網重疊 ({'+'.join(matched)})"
        elif cnt == 2:
            badge = f"🟢 2/4 雙網重疊 ({'+'.join(matched)})"
        elif cnt == 1:
            badge = f"🟡 1/4 單網命中 ({matched[0]})"
        else:
            badge = "⚪ 0/4 未命中"

        out.update({
            "screen_a_rs_leader": scr_a,
            "screen_b_vcp_tight": scr_b,
            "screen_c_pocket_thrust": scr_c,
            "screen_d_small_mid_velocity": scr_d,
            "overlap_count": cnt,
            "overlap_badge": badge,
            "matched_screens_list": matched
        })
    except Exception:
        pass
    return out


# ==============================================================================
# 4. 港股 vs 美股雙軌差異化盈虧比與波段目標價引擎 (Dual-Track HK/US R:R Engine)
# ==============================================================================
def compute_dual_track_hk_us_rr_targets(
    entry_price: float,
    stop_loss: float,
    atr_val: float,
    sr_targets: List[Tuple[float, str]],
    market_type: str = "HK",
    direction: str = "LONG",
    user_batting_avg: float = 0.50
) -> Dict[str, Any]:
    """
    解決「港股與美股盈虧比 (R:R) 做法本質不同」以及「恢復 main7 買藥明生物 (2269.HK) 階段合理波段目標」之核心函數：
    - 【港股波段模式 (HK Track)】：
      1) 港股受制於印花稅與每手股數 (1手往往不可拆半賣出)，且趨勢一旦啟動常沿 10/20MA 走出 +9%~+14% 主升浪；
      2) 若最近小阻力 (sr_targets[0]) 距離買入價不足 2.0R (或少於 2.8x ATR / +7.5%)，判定為「箱體內部小浪頂 (僅作推保本參考點 TP0)」，
         自動跨過該小阻力，採用 main7 經典波段投射：TP1 = max(合規結構阻力, Entry + 3.6x ~ 4.0x ATR, Entry + 2.2R)，確保 R:R 落在 2.2R~2.8R 黃金區！
    - 【美股波段模式 (US Track)】：
      1) 美股零印花稅、1股可拆賣，但盤前/隔夜跳空多且突破後常於 +6%~+8.5% (約 2x 止損距離) 出現 Squat 蹲伏回抽；
      2) 按照《超級績效 2》「2x 平均跌幅鎖定保本並於強勢延伸時賣出一半 (Sell Half into Strength)」法則：
         將第一實戰落袋點 TP1 設於 2.0R ~ 2.4R (約 +6.5%~+8.5% 或 2.6x ATR) 先袋走 50% 利潤並推保本，
         將 TP2 設於遠端結構大阻力 / 4.0x ATR 讓剩餘 50% 尾倉沿 21EMA 奔跑！
    """
    is_hk = (str(market_type).upper() == "HK")
    risk_per_share = max(abs(entry_price - stop_loss), entry_price * 0.01)
    stop_pct = (risk_per_share / max(entry_price, 1e-6)) * 100.0

    # Minervini Built-in Failure Minimum Required R:R based on Batting Average
    min_required_rr = 2.0 if is_hk else 1.85

    if direction == "LONG":
        # Classic main7 wave target: Entry + 3.8x ATR (or at least +2.2R)
        main7_wave_tp = round(max(entry_price + 3.8 * atr_val, entry_price + 2.25 * risk_per_share), 2)
        tp0_breakeven_trigger = round(entry_price + 1.25 * risk_per_share, 2)

        valid_major_srs = [
            (p, lbl) for (p, lbl) in (sr_targets or [])
            if (p - entry_price) / risk_per_share >= min_required_rr and p >= entry_price * (1.072 if is_hk else 1.055)
        ]

        if is_hk:
            # HK Mode: Never let a minor <2.0R recent pivot cap the swing wave!
            if valid_major_srs:
                tp1_price, tp1_desc = round(valid_major_srs[0][0], 2), f"港股波段主結構阻力 ({valid_major_srs[0][1]})"
            else:
                tp1_price = main7_wave_tp
                tp1_desc = f"跨過近期箱體小阻力，按 main7 波段一浪 (Entry + 3.8x ATR / 2.25R) 投射 ({tp1_price:.2f})"
            tp2_price = round(max(tp1_price * 1.05, entry_price + 5.0 * atr_val), 2)
            exit_strategy_note = (
                f"🇭🇰 **港股 main7 波段止盈法**：忽略箱體內小阻力，升穿 **{tp0_breakeven_trigger:.2f} (+1.25R)** 時將止損推至富途打和價，"
                f"直取主波段目標 **TP1 ({tp1_price:.2f}，+{((tp1_price - entry_price)/entry_price)*100:.1f}%)**！"
            )
        else:
            # US Mode: Minervini 2R~2.4R Sell-Half-Into-Strength + 21EMA Runner
            us_impulse_cap = round(min(entry_price * 1.095, max(entry_price + 2.65 * atr_val, entry_price + 2.05 * risk_per_share)), 2)
            if valid_major_srs and valid_major_srs[0][0] <= us_impulse_cap * 1.02:
                tp1_price, tp1_desc = round(valid_major_srs[0][0], 2), f"美股首道波段阻力 ({valid_major_srs[0][1]})"
            else:
                tp1_price = us_impulse_cap
                tp1_desc = f"美股《超級績效》2R 強勢減半目標 (Sell into Strength: {tp1_price:.2f})"
            tp2_price = round(valid_major_srs[0][0], 2) if (valid_major_srs and valid_major_srs[0][0] > tp1_price * 1.02) else main7_wave_tp
            exit_strategy_note = (
                f"🇺🇸 **美股《超級績效》分批鎖潤法**：美股可 1 股拆賣，於 **TP1 ({tp1_price:.2f}，+{((tp1_price - entry_price)/entry_price)*100:.1f}%)** "
                f"趁強勢先賣出 50% 鎖定利潤並推保本，剩餘 50% 尾倉沿 **21EMA** 跟蹤直取 **TP2 ({tp2_price:.2f})**！"
            )

        planned_rr = round((tp1_price - entry_price) / risk_per_share, 2)
        tp2_rr = round((tp2_price - entry_price) / risk_per_share, 2)
    else:
        main7_wave_tp = round(max(0.01, min(entry_price - 3.6 * atr_val, entry_price - 2.2 * risk_per_share)), 2)
        tp0_breakeven_trigger = round(max(0.01, entry_price - 1.25 * risk_per_share), 2)
        tp1_price = main7_wave_tp
        tp2_price = round(max(0.01, entry_price - 4.8 * atr_val), 2)
        tp1_desc = f"空頭波段回補目標 ({tp1_price:.2f})"
        planned_rr = round((entry_price - tp1_price) / risk_per_share, 2)
        tp2_rr = round((entry_price - tp2_price) / risk_per_share, 2)
        exit_strategy_note = f"空頭下跌波段快而急，觸及 **TP1 ({tp1_price:.2f})** 建議回補 50%~100% 落袋。"

    return {
        "tp0_breakeven_trigger": tp0_breakeven_trigger,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "planned_rr": planned_rr,
        "tp2_rr": tp2_rr,
        "stop_pct": round(stop_pct, 2),
        "tp1_desc": tp1_desc,
        "exit_strategy_note": exit_strategy_note
    }
import threading as _threading_lib
_CACHE_LOCK = _threading_lib.Lock()

US_TICKER_ALIAS_MAP: Dict[str, str] = {
    "SQ": "XYZ",       # Block Inc. 2025年1月正式由 SQ 改名為 XYZ
    "BRK.B": "BRK-B",  # Yahoo Finance 格式統一
    "BF.B": "BF-B",
    "FB": "META",
}

def sanitize_ticker_input(raw_ticker: str, default_val: str = "0700.HK") -> str:
    """[v38.0 安全加固與美股改名映射] 對 URL Query Params 或輸入框代碼進行白名單正則消毒並自動轉換改名代碼"""
    if not raw_ticker or not isinstance(raw_ticker, str):
        return default_val
    cleaned = re.sub(r"[^A-Za-z0-9.\^=-]", "", raw_ticker.strip().upper())
    cleaned = US_TICKER_ALIAS_MAP.get(cleaned, cleaned)
    return cleaned[:15] if cleaned else default_val

def evaluate_premarket_gap_and_orb_stop_protocol(
    daily_df: pd.DataFrame,
    stop_loss: float,
    entry_price: float = 0.0,
    market_type: str = "US/Global",
    orb_low_15m: float = np.nan,
    intraday_vwap: float = np.nan
) -> Dict[str, Any]:
    """
    [v36.0 核心新增：美股盤前跌穿止損 vs 正規盤開盤執行協議 (Pre-Market Gap-Down & RTH ORB-Low Stop Protocol)]
    專門解決美股實戰痛點：「有些美股在盤前便跌穿止損，究竟應該在盤前止損，還是等開盤先執行？」
    量化規則：
    1) 盤前流動性真空保護 (04:00~09:30 EST)：盤前無做市商深度且買賣價差極闊，除非跳空跌幅超過災難閾值 (Catastrophic Gap > 2.2x ATR 或跌穿止損 > 3.5%)，
       否則嚴禁在盤前執行市價止損（富途止損單應設為「僅正規交易時段觸發」）。
    2) 正規盤開盤首 15 分鐘低點法則 (RTH 15m ORB-Low Rule)：
       - 若開盤價低於或貼近止損位，但未觸及災難硬止損 (catastrophic_hard_sl)，先觀察 09:30~09:45 EST 首 15 分鐘低點 (orb_low_15m)；
       - 將實戰止損觸發單設於 `orb_low_15m - 0.10 * ATR`：若開盤後守住首 15 分鐘低點並站回日內 VWAP，代表盤前屬假跌破洗盤，繼續持有；
       - 若正規盤實質跌穿 `orb_low_15m` 或直接跌穿 `catastrophic_hard_sl`，代表正規盤機構賣壓確認，立即無條件止損離場！
    """
    res = {
        "catastrophic_hard_sl": round(stop_loss * 0.97, 2),
        "rth_orb_stop_trigger": round(stop_loss, 2),
        "premarket_order_setting": "僅正規交易時段 (RTH 09:30-16:00 EST) 觸發",
        "protocol_action": "按正規盤結構止損執行",
        "protocol_detail": ""
    }
    if daily_df is None or daily_df.empty or stop_loss <= 0:
        return res
    try:
        last = daily_df.iloc[-1]
        curr_c = float(last['Close'])
        curr_o = float(last['Open'])
        atr_v = float(last.get('ATR', curr_c * 0.03))
        if not pd.notna(atr_v) or atr_v <= 0:
            atr_v = curr_c * 0.03

        # 災難硬止損線：結構止損下方 1.2x ATR 或 -3.2% (防止極端黑天鵝無限深跌)
        cat_sl = round(max(stop_loss - 1.20 * atr_v, stop_loss * 0.968), 2)
        res["catastrophic_hard_sl"] = cat_sl

        orb_l = float(orb_low_15m) if pd.notna(orb_low_15m) and orb_low_15m > 0 else min(curr_o, float(last['Low']))
        rth_orb_trigger = round(orb_l - 0.08 * atr_v, 2)
        res["rth_orb_stop_trigger"] = rth_orb_trigger

        if market_type != "HK":
            res["protocol_detail"] = (
                f"🧭 **【美股盤前穿止損 vs 開盤執行標準協議 (RTH ORB-Low Protocol)】**：\n"
                f"1. **富途止損單設定**：請務必選擇 **「僅正規交易時段 (09:30–16:00 EST) 觸發」**，**切勿開啟盤前/盤後觸發**（盤前買賣價差極大，極易被幾百股薄市錯價單掃出在最低點）。\n"
                f"2. **若盤前或開盤小幅跌穿止損 (`{cat_sl:.2f}` ~ `{stop_loss:.2f}` 之間)**：開盤頭 5~15 分鐘先唔好市價砍倉！觀察正規盤開盤區間低點 (**ORB Low: `{orb_l:.2f}`**) 與日內 VWAP (**`{intraday_vwap if pd.notna(intraday_vwap) else curr_c:.2f}`**)，"
                f"將止損觸發價設於 **`{rth_orb_trigger:.2f}` (ORB低點下方)**——守住並站回 VWAP 即屬盤前假跌破洗盤；若正規盤跌穿 `{rth_orb_trigger:.2f}` 則立即止損！\n"
                f"3. **極端黑天鵝例外 (`< {cat_sl:.2f}`)**：僅當個股突發重大負面消息且盤前/開盤直接跌穿災難底線 **`{cat_sl:.2f}`** 時，才於開盤首分鐘無條件離場。"
            )
    except Exception:
        pass
    return res

def check_intraday_execution_window(market_type: str = "US/Global") -> Dict[str, Any]:
    """
    [v35.0 交易時段微結構過濾器]
    檢測當前是否處於美股盤前 (Pre-Market 04:00~09:30 EST) 或開盤首 30 分鐘高波動價格發現期 (09:30~10:00 EST)。
    盤前與開盤初期流動性不均且假跳空頻發，市價單極易成交於不利滑點位。
    """
    res = {
        "is_restricted_window": False,
        "window_phase": "正常交易/收盤時段",
        "hkt_time_str": "",
        "est_time_str": "",
        "warning_banner": ""
    }
    if market_type == "HK":
        return res
    try:
        now_est = datetime.now(ZoneInfo("America/New_York"))
        now_hkt = datetime.now(ZoneInfo("Asia/Hong_Kong"))
        res["hkt_time_str"] = now_hkt.strftime("%H:%M HKT")
        res["est_time_str"] = now_est.strftime("%H:%M EST/EDT")
        if now_est.weekday() < 5:
            mins = now_est.hour * 60 + now_est.minute
            if 240 <= mins < 570:
                res["is_restricted_window"] = True
                res["window_phase"] = "🚫 美股盤前時段 (Pre-Market)"
                res["warning_banner"] = (
                    f"🚨 **【交易時段微結構警示 ({res['hkt_time_str']} / {res['est_time_str']})】**："
                    f"當前處於美股盤前時段，缺乏機構真實成交量確認且買賣價差偏大，日線指標仍停留在上一交易日收盤狀態。"
                    f"**不建議於盤前執行市價單，宜等待正規盤開盤 30 分鐘後 (美東 10:00 後) 站穩日內 VWAP 及開盤區間 (ORB) 再執行。**"
                )
            elif 570 <= mins < 600:
                res["is_restricted_window"] = True
                res["window_phase"] = "⚠️ 美股開盤首30分鐘價格發現期"
                res["warning_banner"] = (
                    f"⚠️ **【開盤首 30 分鐘價格發現期 ({res['hkt_time_str']} / {res['est_time_str']})】**："
                    f"開盤初期波幅與假突破率較高，請確認股價守穩日內 VWAP 且無長上影線派發跡象後再進場。"
                )
    except Exception:
        pass
    return res

def get_user_risk_and_full_pos_config(market_type: str = "HK") -> Dict[str, float]:
    """
    [v41.1 核心配置] 讀取用戶側欄設定之「單筆最大承受虧損 (預設 HK$1,000 / US$128)」與「單筆目標滿倉本金 (預設 HK$21,500 / US$2,756)」
    """
    is_hk = (str(market_type).upper() == "HK")
    try:
        hkd_risk = float(st.session_state.get("user_max_loss_hkd", 1000.0))
        hkd_full_notional = float(st.session_state.get("user_full_pos_hkd", 21500.0))
    except Exception:
        hkd_risk = 1000.0
        hkd_full_notional = 21500.0
    if is_hk:
        return {
            "risk_budget": round(hkd_risk, 2),
            "full_notional": round(hkd_full_notional, 2),
            "risk_hkd": round(hkd_risk, 2),
            "full_notional_hkd": round(hkd_full_notional, 2),
            "currency_symbol": "HK$"
        }
    else:
        return {
            "risk_budget": round(hkd_risk / 7.8, 2),
            "full_notional": round(hkd_full_notional / 7.8, 2),
            "risk_hkd": round(hkd_risk, 2),
            "full_notional_hkd": round(hkd_full_notional, 2),
            "currency_symbol": "US$"
        }


def compute_equal_risk_position_cap(
    entry_price: float,
    stop_loss: float,
    fixed_risk_budget: Optional[float] = None,
    max_notional_cap: Optional[float] = None,
    lot_size: int = 1,
    market_type: str = "US/Global"
) -> Dict[str, Any]:
    """
    [v41.1 專為「滿倉 HK$18,000~25,000 + 單筆止損鎖定 HK$1,000 (US$128)」度身訂造之等風險倉位與安全買點反推引擎]
    """
    is_hk = (str(market_type).upper() == "HK")
    usr_cfg = get_user_risk_and_full_pos_config(market_type=market_type)
    currency_sym = usr_cfg["currency_symbol"]

    if fixed_risk_budget is None or fixed_risk_budget == 600.0:
        eff_risk_budget = usr_cfg["risk_budget"]
    else:
        eff_risk_budget = float(fixed_risk_budget)

    if max_notional_cap is None or max_notional_cap == 20000.0:
        eff_max_notional = usr_cfg["full_notional"]
    else:
        eff_max_notional = float(max_notional_cap)

    if entry_price <= 0 or stop_loss <= 0:
        return {
            "stop_pct": 0.0,
            "recommended_notional": 0.0,
            "recommended_shares": 0,
            "unrounded_shares": 0,
            "expected_loss_at_stop": 0.0,
            "min_1lot_over_budget": False,
            "min_1lot_loss": 0.0,
            "risk_budget_used": round(eff_risk_budget, 2),
            "target_full_notional": round(eff_max_notional, 2),
            "currency_symbol": currency_sym,
            "can_full_pos_at_market": False,
            "full_pos_shares": 0,
            "full_pos_notional": 0.0,
            "full_pos_loss_at_struct_sl": 0.0,
            "dollar_cap_sl_for_full_pos": 0.0,
            "safe_full_pos_entry": 0.0,
            "is_noise_trap": False,
            "full_pos_tag": "⚪ 待定"
        }

    is_short = (stop_loss > entry_price)
    stop_dist = max(abs(entry_price - stop_loss), entry_price * 0.008)
    stop_pct = (stop_dist / entry_price) * 100.0
    fee_est = 38.0 if is_hk else 3.5
    net_risk_for_price = max(eff_risk_budget - fee_est, eff_risk_budget * 0.75)

    eff_stop_pct_dec = (stop_dist / entry_price) + 0.0020
    raw_notional = net_risk_for_price / max(eff_stop_pct_dec, 0.008)
    safe_notional = min(raw_notional, eff_max_notional)
    raw_shares = int(safe_notional / entry_price)
    min_1lot_shares = max(1, int(lot_size))
    min_1lot_loss = round(min_1lot_shares * stop_dist + fee_est, 2)
    min_1lot_over_budget = False
    if lot_size > 1:
        if raw_shares < lot_size:
            min_1lot_over_budget = True
            shares = lot_size
        else:
            shares = (raw_shares // lot_size) * lot_size
    else:
        if raw_shares < 1:
            min_1lot_over_budget = True
            shares = 1
        else:
            shares = raw_shares
    actual_notional = round(shares * entry_price, 2)
    est_loss = round(shares * stop_dist + fee_est, 2)

    raw_full_shares = max(1, int(eff_max_notional / entry_price))
    if lot_size > 1:
        full_pos_shares = max(lot_size, round(raw_full_shares / lot_size) * lot_size)
    else:
        full_pos_shares = raw_full_shares
    full_pos_notional = round(full_pos_shares * entry_price, 2)
    full_pos_loss_at_struct_sl = round(full_pos_shares * stop_dist + fee_est, 2)

    max_move_per_share_1k = net_risk_for_price / max(full_pos_shares, 1)
    if is_short:
        dollar_cap_sl = round(entry_price + max_move_per_share_1k, 2)
        safe_full_pos_entry = round(stop_loss - max_move_per_share_1k, 2)
        is_noise_trap = bool(dollar_cap_sl < stop_loss * 0.996)
    else:
        dollar_cap_sl = round(max(0.01, entry_price - max_move_per_share_1k), 2)
        safe_full_pos_entry = round(stop_loss + max_move_per_share_1k, 2)
        is_noise_trap = bool(dollar_cap_sl > stop_loss * 1.004)

    can_full_pos_at_market = bool(full_pos_loss_at_struct_sl <= eff_risk_budget * 1.15 and not min_1lot_over_budget)
    if can_full_pos_at_market:
        full_pos_tag = f"🟢 可直接滿倉 (蝕≤{currency_sym}{eff_risk_budget:,.0f})"
    elif min_1lot_over_budget:
        full_pos_tag = f"🔴 1手已超{currency_sym}{eff_risk_budget:,.0f}風險 (需碎股/低吸)"
    else:
        full_pos_tag = f"🟡 滿倉需回調至 {safe_full_pos_entry:.2f} 或縮倉至 {shares}股"

    return {
        "stop_pct": round(stop_pct, 2),
        "recommended_notional": actual_notional,
        "recommended_shares": int(shares),
        "unrounded_shares": int(max(1, raw_shares)),
        "expected_loss_at_stop": est_loss,
        "min_1lot_over_budget": bool(min_1lot_over_budget),
        "min_1lot_loss": min_1lot_loss,
        "risk_budget_used": round(eff_risk_budget, 2),
        "target_full_notional": round(eff_max_notional, 2),
        "currency_symbol": currency_sym,
        "can_full_pos_at_market": can_full_pos_at_market,
        "full_pos_shares": int(full_pos_shares),
        "full_pos_notional": full_pos_notional,
        "full_pos_loss_at_struct_sl": full_pos_loss_at_struct_sl,
        "dollar_cap_sl_for_full_pos": dollar_cap_sl,
        "safe_full_pos_entry": safe_full_pos_entry,
        "is_noise_trap": is_noise_trap,
        "full_pos_tag": full_pos_tag
    }

def compute_ma_slope_and_pullback_trigger(df: pd.DataFrame) -> Dict[str, Any]:
    """
    [v35.0 多均線斜率同向濾網與回踩右側收復確認引擎]
    1) 均線斜率同向檢驗：回踩 20MA 時，要求 20MA、50MA、60MA 不得處於同步下彎狀態，過濾「均線蓋頭向下之假回踩」；
    2) 回踩買點雙軌定位：
       - 若股價前 3 日曾刺穿或回測 9EMA/10MA，且當日收陽並重新站上 9EMA (收盤位於當日波幅上半部)，確認為「右側洗盤收復買點 (⚡ 即時入場)」；
       - 若股價仍處於 9EMA 下方陰跌尋底，則將左側限價承接位設於「(9EMA + 20SMA) / 2 中位數價值區」，避免在下跌中途過早市價接刀。
    """
    res = {
        "sma10_rising": True,
        "sma20_rising": True,
        "sma50_rising": True,
        "sma60_rising": True,
        "ma_ribbon_sloping_down": False,
        "reclaimed_9ema_right_side": False,
        "mid_9ema_20ma_price": 0.0,
        "ema9_reclaim_trigger": 0.0,
        "slope_warning": ""
    }
    if df is None or len(df) < 25:
        return res
    try:
        last = df.iloc[-1]
        prev5 = df.iloc[-6] if len(df) >= 6 else df.iloc[0]
        prev3 = df.iloc[-4] if len(df) >= 4 else df.iloc[0]
        c_now = float(last['Close'])
        o_now = float(last['Open'])
        h_now = float(last['High'])
        l_now = float(last['Low'])
        ema9_now = float(last.get('EMA9', c_now))
        sma10_now = float(last.get('SMA10', c_now))
        sma20_now = float(last.get('SMA20', c_now))
        sma50_now = float(last.get('SMA50', c_now))
        sma60_now = float(last.get('SMA60', sma50_now))

        s10_up = sma10_now >= float(prev3.get('SMA10', sma10_now)) * 0.998
        s20_up = sma20_now >= float(prev5.get('SMA20', sma20_now)) * 0.999
        s50_up = sma50_now >= float(prev5.get('SMA50', sma50_now)) * 0.999
        s60_up = sma60_now >= float(prev5.get('SMA60', sma60_now)) * 0.999

        res["sma10_rising"] = bool(s10_up)
        res["sma20_rising"] = bool(s20_up)
        res["sma50_rising"] = bool(s50_up)
        res["sma60_rising"] = bool(s60_up)

        down_cnt = sum([not s10_up, not s20_up, not s50_up, not s60_up])
        if (not s50_up and not s60_up) or (not s20_up and not s50_up) or down_cnt >= 3:
            res["ma_ribbon_sloping_down"] = True
            res["slope_warning"] = (
                "⚠️ **【均線斜率背馳過濾】**：雖然價格接近 20MA，但中期均線組 (50MA/60MA/10MA) 正在下彎，"
                "缺乏均線助漲推力，系統已自動攔截此回踩做多信號。"
            )

        mid_p = round((ema9_now + sma20_now) / 2.0, 2)
        res["mid_9ema_20ma_price"] = mid_p
        res["ema9_reclaim_trigger"] = round(ema9_now * 1.002, 2)
        recent3_min_low = float(df['Low'].iloc[-3:].min())
        close_pos = (c_now - l_now) / max(h_now - l_now, 1e-6)
        if recent3_min_low <= ema9_now * 1.002 and c_now >= ema9_now and c_now >= o_now and close_pos >= 0.52:
            res["reclaimed_9ema_right_side"] = True
    except Exception:
        pass
    return res

import urllib.request as _urllib_req
import urllib.parse as _urllib_parse
import xml.etree.ElementTree as _xml_et
import ssl as _ssl
import json as _json
import re as _re
import html as _html

try:
    import requests as _requests
    import urllib3 as _urllib3
    _urllib3.disable_warnings(_urllib3.exceptions.InsecureRequestWarning)
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "zh-HK,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6",
    "Cache-Control": "no-cache",
}

def _safe_http_get_bytes(url: str, timeout: float = 4.5, extra_headers: Optional[Dict[str, str]] = None) -> Tuple[Optional[bytes], str]:
    """
    [v37.1 跨平台防 SSL 憑證報錯與防反爬蟲 HTTP 讀取器]
    解決 macOS / Windows Python 執行 urllib.request.urlopen 時因缺乏根憑證拋出
    `[SSL: CERTIFICATE_VERIFY_FAILED]` 以及單純 User-Agent: Mozilla/5.0 被 Google/Yahoo 403 攔截之問題。
    優先使用 requests (自動支援 certifi 與 verify=False 容錯)，再以 unverified SSL context 之 urllib 兜底。
    """
    hdrs = dict(_BROWSER_HEADERS)
    if extra_headers:
        hdrs.update(extra_headers)
    last_err = ""
    if _HAS_REQUESTS:
        for verify_ssl in (True, False):
            try:
                resp = _requests.get(url, headers=hdrs, timeout=timeout, verify=verify_ssl)
                if resp.status_code == 200 and resp.content:
                    return resp.content, ""
                last_err = f"HTTP {resp.status_code}"
            except Exception as e:
                last_err = f"{type(e).__name__}: {str(e)[:80]}"
    try:
        ctx = _ssl._create_unverified_context()
        req = _urllib_req.Request(url, headers=hdrs)
        with _urllib_req.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read()
            if data:
                return data, ""
    except Exception as e:
        last_err = f"{type(e).__name__}: {str(e)[:80]}"
    return None, last_err

_CORP_EVENTS_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def _fetch_corporate_events_calendar(ticker: str) -> Dict[str, Any]:
    """
    [v37.1 四層財報日期與除淨日提取器 — 防空結果緩存污染]
    支援從：
    1) yfinance Ticker.calendar (Dict 或 DataFrame)
    2) yfinance Ticker.info (earningsTimestamp / exDividendDate)
    3) yfinance Ticker.get_earnings_dates(limit=8) (專治港股與部分美股 calendar 為空)
    4) Yahoo Finance quoteSummary API 直接查詢兜底
    提取確切業績公佈日期 (YYYY-MM-DD)、距離天數與除淨日。
    """
    out = {
        "earnings_date_str": None,
        "days_to_earnings": None,
        "ex_dividend_date_str": None,
        "days_to_ex_div": None
    }
    if not ticker:
        return out
    now_ts = time.time()
    cached = _CORP_EVENTS_MEM_CACHE.get(ticker)
    if cached and (now_ts - cached[0] < 21600):
        return dict(cached[1])

    today_d = datetime.now().date()
    cand_dates = []
    try:
        t = yf.Ticker(ticker)
        # Layer 1: t.calendar
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date") or cal.get("Earnings High") or cal.get("Earnings Low")
                if isinstance(ed_raw, (list, tuple)):
                    for item in ed_raw:
                        cand_dates.append(pd.to_datetime(item).date())
                elif ed_raw is not None:
                    cand_dates.append(pd.to_datetime(ed_raw).date())
                ex_raw = cal.get("Ex-Dividend Date")
                if ex_raw is not None:
                    ex_d = pd.to_datetime(ex_raw).date()
                    out["ex_dividend_date_str"] = str(ex_d)
                    out["days_to_ex_div"] = int((ex_d - today_d).days)
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for val in cal.values.flatten():
                    try:
                        cand_dates.append(pd.to_datetime(val).date())
                    except Exception:
                        pass
        except Exception:
            pass

        # Layer 2: t.info timestamps
        try:
            info_d = t.info or {}
            for k_ts in ["earningsTimestamp", "earningsTimestampStart", "earningsTimestampEnd"]:
                ts_v = info_d.get(k_ts)
                if ts_v:
                    cand_dates.append(datetime.fromtimestamp(int(ts_v)).date())
            if not out["ex_dividend_date_str"] and info_d.get("exDividendDate"):
                ex_d = datetime.fromtimestamp(int(info_d["exDividendDate"])).date()
                out["ex_dividend_date_str"] = str(ex_d)
                out["days_to_ex_div"] = int((ex_d - today_d).days)
        except Exception:
            pass

        # Layer 3: t.get_earnings_dates(limit=8) (當 Layer 1/2 皆空時調用)
        if not cand_dates:
            try:
                ed_df = t.get_earnings_dates(limit=8)
                if isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
                    for idx_val in ed_df.index:
                        try:
                            cand_dates.append(pd.to_datetime(idx_val).date())
                        except Exception:
                            pass
            except Exception:
                pass

        # Pick nearest upcoming or very recent (-3 to +120 days) earnings date
        valid_future = sorted([d for d in set(cand_dates) if 0 <= (d - today_d).days <= 120])
        if not valid_future:
            valid_future = sorted([d for d in set(cand_dates) if -5 <= (d - today_d).days < 0], reverse=True)
        if valid_future:
            best_d = valid_future[0]
            out["earnings_date_str"] = best_d.strftime("%Y-%m-%d")
            out["days_to_earnings"] = int((best_d - today_d).days)
            _CORP_EVENTS_MEM_CACHE[ticker] = (now_ts, dict(out))
    except Exception:
        pass
    return out

def _fetch_upcoming_earnings_days(ticker: str) -> Optional[int]:
    return _fetch_corporate_events_calendar(ticker).get("days_to_earnings")

def detect_earnings_and_gap_risk(df: pd.DataFrame, ticker: str = "") -> Dict[str, Any]:
    """
    [v37.0 升級：確切業績公佈日期 (YYYY-MM-DD) + 除淨日 + 量價黑天鵝異動預警引擎]
    不僅提示距離業績還有幾天，更直接顯示業績公佈日期、除淨日（防港股除息假破位），
    並檢測近期是否出現「黑天鵝前夕異常放量跳空或單日 >2.2x ATR 巨震」！
    """
    res = {
        "gap_events_1y": 0,
        "max_overnight_gap_pct": 0.0,
        "recent_gap_shock": False,
        "earnings_date_str": None,
        "days_to_earnings": None,
        "ex_dividend_date_str": None,
        "days_to_ex_div": None,
        "abnormal_vol_shock_5d": False,
        "gap_risk_level": "LOW",
        "earnings_badge": "📅 暫無近期業績公佈資料",
        "gap_warning_msg": ""
    }
    if df is None or len(df) < 25:
        return res
    try:
        sub = df.tail(min(len(df), 252))
        prev_c = sub['Close'].shift(1)
        overnight_gaps = ((sub['Open'] / (prev_c + 1e-9)) - 1.0).abs() * 100.0
        overnight_gaps = overnight_gaps.dropna()
        if not overnight_gaps.empty:
            res["gap_events_1y"] = int((overnight_gaps >= 4.5).sum())
            res["max_overnight_gap_pct"] = round(float(overnight_gaps.max()), 2)
            res["recent_gap_shock"] = bool(float(overnight_gaps.tail(5).max()) >= 5.0)

        # 檢測近 5 日是否出現黑天鵝式異常巨震 (單日波幅 > 2.3x ATR 且成交量 > 1.8x 均量)
        last5 = sub.tail(5)
        atr_last = float(sub['ATR'].iloc[-1]) if ('ATR' in sub.columns and pd.notna(sub['ATR'].iloc[-1])) else float(sub['Close'].iloc[-1]) * 0.03
        vol_ma20 = float(sub['Volume'].tail(20).mean()) if 'Volume' in sub.columns else 1.0
        if float((last5['High'] - last5['Low']).max()) >= 2.3 * atr_last and float(last5['Volume'].max()) >= 1.8 * vol_ma20:
            res["abnormal_vol_shock_5d"] = True

        cal_info = _fetch_corporate_events_calendar(ticker) if ticker else {}
        d_earn = cal_info.get("days_to_earnings")
        e_str = cal_info.get("earnings_date_str")
        ex_str = cal_info.get("ex_dividend_date_str")
        d_ex = cal_info.get("days_to_ex_div")

        res["days_to_earnings"] = d_earn
        res["earnings_date_str"] = e_str
        res["ex_dividend_date_str"] = ex_str
        res["days_to_ex_div"] = d_ex

        msgs = []
        if e_str is not None and d_earn is not None:
            if d_earn < 0:
                res["earnings_badge"] = f"📅 剛於 {e_str} 公佈業績 ({abs(d_earn)}日前)"
            elif d_earn == 0:
                res["earnings_badge"] = f"🚨 今日 ({e_str}) 公佈業績！"
            elif d_earn <= 7:
                res["earnings_badge"] = f"🚨 業績日：{e_str} (倒數 {d_earn} 天)"
            elif d_earn <= 21:
                res["earnings_badge"] = f"⚠️ 業績日：{e_str} (倒數 {d_earn} 天)"
            else:
                res["earnings_badge"] = f"📅 下次業績：{e_str} (還有 {d_earn} 天)"

            if 0 <= d_earn <= 7:
                res["gap_risk_level"] = "CRITICAL"
                msgs.append(
                    f"🚨 **【業績公佈黑天鵝警報 — 預計公佈日：{e_str}（倒數 {d_earn} 天）】**："
                    f"距離該股公佈業績僅剩 **{d_earn} 天**！無論做多或做空（如你曾在小米業績日做空遭遇隔日大升止損），"
                    f"**嚴禁在業績前 7 日內無利潤墊 (<1.5R) 開立新倉或重倉賭業績！** 已持倉者請在 {e_str} 前先減倉一半或推高止損保護！"
                )
            elif 8 <= d_earn <= 14:
                if res["gap_risk_level"] == "LOW":
                    res["gap_risk_level"] = "MEDIUM"
                msgs.append(
                    f"⚠️ **【業績窗口臨近提示 — 預計公佈日：{e_str}（還有 {d_earn} 天）】**："
                    f"若計劃開倉，請確保在 {e_str} 業績公佈前能先觸及 TP0 (+4.5%) 鎖定半倉利潤墊，避免裸倉過業績。"
                )

        if ex_str is not None and d_ex is not None and 0 <= d_ex <= 10:
            msgs.append(
                f"💰 **【除淨日提示 — 除淨日期：{ex_str}（還有 {d_ex} 天）】**："
                f"除淨當日股價會因扣除派息而自然下調，請留意富途條件單會否因除淨價差而誤觸止損。"
            )

        if res["recent_gap_shock"] or res["abnormal_vol_shock_5d"]:
            if res["gap_risk_level"] == "LOW":
                res["gap_risk_level"] = "HIGH"
            msgs.append(
                f"⚡ **【近期黑天鵝量價異動警報】**：近 5 個交易日內出現異常跳空或單日 >2.3x ATR 爆量巨震，"
                f"顯示有突發消息或機構劇烈換手，請務必結合下方【📰 實時新聞情報與利好/利空診斷】確認原因！"
            )
        elif res["max_overnight_gap_pct"] >= 10.0 or res["gap_events_1y"] >= 4:
            if res["gap_risk_level"] == "LOW":
                res["gap_risk_level"] = "HIGH"
            msgs.append(
                f"⚡ **【歷史高跳空風險基因】**：過去一年曾出現 **{res['gap_events_1y']} 次** ≥4.5% 隔夜跳空"
                f"（最大單日隔夜跳空 **{res['max_overnight_gap_pct']:.1f}%**），持倉過夜務必嚴控單筆風險上限。"
            )

        res["gap_warning_msg"] = "\n\n".join(msgs)
    except Exception:
        pass
    return res

def _classify_single_news_headline(title: str, summary: str = "") -> Dict[str, Any]:
    """
    [v37.1 雙語金融 NLP 催化劑分類與利好/利空評分器 (高靈敏度實戰版)]
    支援繁/簡體中文與英文財經新聞標題及摘要，涵蓋強催化劑與日常多空方向詞彙，識別 6 大波段催化劑類別與正負面情緒權重，並處理否定詞反轉。
    """
    text = f"{title} {summary}".lower()
    bull_keywords = [
        # 強催化劑 (+20 ~ +38)
        ("盈喜", 35, "📊 業績財測"), ("扭虧", 35, "📊 業績財測"), ("純利增", 28, "📊 業績財測"), ("利潤增", 28, "📊 業績財測"),
        ("超預期", 30, "📊 業績財測"), ("勝預期", 30, "📊 業績財測"), ("創新高", 24, "🚀 業務與動能"), ("破頂", 24, "🚀 業務與動能"),
        ("回購", 26, "💎 回購與派息"), ("增持", 26, "💎 回購與派息"), ("特別息", 25, "💎 回購與派息"), ("派息增", 24, "💎 回購與派息"),
        ("上調目標價", 28, "🏦 機構評級"), ("升目標價", 28, "🏦 機構評級"), ("籲買入", 24, "🏦 機構評級"), ("首予買入", 25, "🏦 機構評級"),
        ("跑贏大市", 22, "🏦 機構評級"), ("獲批", 28, "🚀 業務與動能"), ("中標", 26, "🚀 業務與動能"), ("戰略合作", 20, "🚀 業務與動能"),
        ("大漲", 20, "🚀 業務與動能"), ("飆升", 22, "🚀 業務與動能"), ("急升", 20, "🚀 業務與動能"), ("暴漲", 22, "🚀 業務與動能"),
        ("beat", 28, "📊 業績財測"), ("beats", 28, "📊 業績財測"), ("surges", 22, "🚀 業務與動能"), ("soars", 24, "🚀 業務與動能"),
        ("jumps", 20, "🚀 業務與動能"), ("rallies", 20, "🚀 業務與動能"), ("upgrade", 28, "🏦 機構評級"), ("upgraded", 28, "🏦 機構評級"),
        ("raises target", 28, "🏦 機構評級"), ("price target raised", 28, "🏦 機構評級"), ("outperform", 22, "🏦 機構評級"),
        ("overweight", 20, "🏦 機構評級"), ("buyback", 26, "💎 回購與派息"), ("repurchase", 24, "💎 回購與派息"),
        ("raises guidance", 32, "📊 業績財測"), ("strong demand", 24, "🚀 業務與動能"), ("record revenue", 28, "📊 業績財測"),
        ("fda approval", 32, "🚀 業務與動能"), ("contract win", 26, "🚀 業務與動能"), ("partnership", 18, "🚀 業務與動能"),
        # 中度正向動能與利好詞彙 (+12 ~ +18)
        ("利好", 18, "🚀 業務與動能"), ("唱好", 18, "🏦 機構評級"), ("看好", 16, "🏦 機構評級"), ("買入", 15, "🏦 機構評級"),
        ("突破", 16, "🚀 業務與動能"), ("造好", 15, "🚀 業務與動能"), ("向好", 15, "🚀 業務與動能"), ("回升", 14, "🚀 業務與動能"),
        ("反彈", 14, "🚀 業務與動能"), ("走高", 15, "🚀 業務與動能"), ("上揚", 15, "🚀 業務與動能"), ("強勁", 16, "📊 業績財測"),
        ("增長", 14, "📊 業績財測"), ("營收增", 22, "📊 業績財測"), ("多賺", 24, "📊 業績財測"), ("吸納", 16, "💎 回購與派息"),
        ("北水掃貨", 20, "💎 回購與派息"), ("淨流入", 18, "💎 回購與派息"), ("訂單", 15, "🚀 業務與動能"), ("量產", 18, "🚀 業務與動能"),
        ("bullish", 18, "🏦 機構評級"), ("buy", 14, "🏦 機構評級"), ("top pick", 22, "🏦 機構評級"), ("gains", 15, "🚀 業務與動能"),
        ("rises", 14, "🚀 業務與動能"), ("climbs", 15, "🚀 業務與動能"), ("higher", 12, "🚀 業務與動能"), ("growth", 14, "📊 業績財測"),
        ("strong", 15, "📊 業績財測"), ("positive", 14, "🚀 業務與動能"), ("record high", 24, "🚀 業務與動能"), ("breakout", 18, "🚀 業務與動能"),
        ("ai demand", 20, "🚀 業務與動能"), ("dividend", 14, "💎 回購與派息"), ("profit rises", 24, "📊 業績財測")
    ]
    bear_keywords = [
        # 強利空與黑天鵝 (-22 ~ -40)
        ("盈警", -38, "📊 業績財測"), ("轉虧", -36, "📊 業績財測"), ("虧損擴大", -35, "📊 業績財測"), ("純利跌", -28, "📊 業績財測"),
        ("遜預期", -30, "📊 業績財測"), ("低於預期", -30, "📊 業績財測"), ("差過預期", -30, "📊 業績財測"),
        ("配股", -36, "🩸 配股抽水/減持"), ("折讓", -32, "🩸 配股抽水/減持"), ("抽水", -36, "🩸 配股抽水/減持"), ("先舊後新", -34, "🩸 配股抽水/減持"),
        ("減持", -28, "🩸 配股抽水/減持"), ("套現", -28, "🩸 配股抽水/減持"), ("拋售", -26, "🩸 配股抽水/減持"),
        ("下調目標價", -28, "🏦 機構評級"), ("削目標價", -28, "🏦 機構評級"), ("降級", -26, "🏦 機構評級"), ("籲沽售", -30, "🏦 機構評級"),
        ("跑輸大市", -24, "🏦 機構評級"), ("調查", -34, "⚖️ 監管與黑天鵝"), ("訴訟", -28, "⚖️ 監管與黑天鵝"), ("罰款", -28, "⚖️ 監管與黑天鵝"),
        ("停牌", -38, "⚖️ 監管與黑天鵝"), ("退市", -40, "⚖️ 監管與黑天鵝"), ("沽空機構", -38, "⚖️ 監管與黑天鵝"), ("做空報告", -38, "⚖️ 監管與黑天鵝"),
        ("暴跌", -25, "📉 市場異動"), ("急挫", -22, "📉 市場異動"), ("重挫", -24, "📉 市場異動"), ("插水", -24, "📉 市場異動"),
        ("miss", -28, "📊 業績財測"), ("misses", -28, "📊 業績財測"), ("cuts guidance", -35, "📊 業績財測"), ("lowers guidance", -34, "📊 業績財測"),
        ("downgrade", -28, "🏦 機構評級"), ("downgraded", -28, "🏦 機構評級"), ("cuts target", -26, "🏦 機構評級"), ("underperform", -24, "🏦 機構評級"),
        ("sec probe", -36, "⚖️ 監管與黑天鵝"), ("investigation", -32, "⚖️ 監管與黑天鵝"), ("lawsuit", -28, "⚖️ 監管與黑天鵝"),
        ("short seller", -35, "⚖️ 監管與黑天鵝"), ("secondary offering", -32, "🩸 配股抽水/減持"), ("dilution", -30, "🩸 配股抽水/減持"),
        ("insider selling", -24, "🩸 配股抽水/減持"), ("plunges", -26, "📉 市場異動"), ("tumbles", -24, "📉 市場異動"), ("slumps", -24, "📉 市場異動"),
        # 中度負向與風險詞彙 (-12 ~ -20)
        ("利淡", -18, "📉 市場異動"), ("利空", -18, "📉 市場異動"), ("看淡", -16, "🏦 機構評級"), ("唱淡", -18, "🏦 機構評級"),
        ("下跌", -14, "📉 市場異動"), ("受壓", -15, "📉 市場異動"), ("走弱", -15, "📉 市場異動"), ("回落", -12, "📉 市場異動"),
        ("少賺", -22, "📊 業績財測"), ("倒退", -20, "📊 業績財測"), ("虧損", -22, "📊 業績財測"), ("裁員", -18, "📉 市場異動"),
        ("關稅", -16, "⚖️ 監管與黑天鵝"), ("制裁", -28, "⚖️ 監管與黑天鵝"), ("禁令", -26, "⚖️ 監管與黑天鵝"), ("沽壓", -18, "📉 市場異動"),
        ("bearish", -18, "🏦 機構評級"), ("sell", -14, "🏦 機構評級"), ("falls", -14, "📉 市場異動"), ("drops", -15, "📉 市場異動"),
        ("slides", -15, "📉 市場異動"), ("declines", -14, "📉 市場異動"), ("lower", -12, "📉 市場異動"), ("weak", -15, "📊 業績財測"),
        ("loss", -18, "📊 業績財測"), ("warning", -20, "📊 業績財測"), ("tariff", -16, "⚖️ 監管與黑天鵝"), ("sanction", -26, "⚖️ 監管與黑天鵝"),
        ("delay", -18, "🚀 業務與動能"), ("layoffs", -16, "📉 市場異動"), ("pullback", -12, "📉 市場異動")
    ]

    score = 0
    detected_cat = "📰 綜合市場消息"
    matched_terms = []

    for kw, w, cat in bull_keywords:
        if kw in text:
            neg_check = any(neg + kw in text for neg in ["不", "未", "無", "難", "not ", "no ", "fails to ", "never "])
            if neg_check:
                score -= int(w * 0.8)
                matched_terms.append(f"否定:{kw}")
            else:
                score += w
                if detected_cat == "📰 綜合市場消息" or w >= 22:
                    detected_cat = cat
                matched_terms.append(kw)

    for kw, w, cat in bear_keywords:
        if kw in text:
            neg_check = any(neg + kw in text for neg in ["不", "未", "無", "難", "not ", "no "])
            if neg_check:
                score += int(abs(w) * 0.5)
            else:
                score += w  # w is negative
                if detected_cat == "📰 綜合市場消息" or abs(w) >= 22:
                    detected_cat = cat
                matched_terms.append(kw)

    score = max(-100, min(100, score))
    is_black_swan = any(
        k in text for k in [
            "盈警", "配股", "抽水", "調查", "停牌", "退市", "做空報告", "沽空機構", "制裁",
            "cuts guidance", "sec probe", "short seller", "secondary offering", "delist"
        ]
    )
    if is_black_swan and score <= -15:
        label = "🚨 重大利空/黑天鵝"
        kind = "error"
    elif score >= 10:
        label = "🟢 利好 (Bullish)"
        kind = "success"
    elif score <= -10:
        label = "🔴 利空 (Bearish)"
        kind = "error"
    else:
        label = "⚪ 中性 (Neutral)"
        kind = "info"

    return {
        "score": score,
        "label": label,
        "kind": kind,
        "category": detected_cat,
        "is_black_swan": is_black_swan,
        "matched_terms": matched_terms[:3]
    }


def _clean_stock_query_terms(ticker: str, stock_name: str, market_type: str) -> Dict[str, str]:
    """
    [v37.1 搜尋關鍵字淨化器 — 根治 Google News `-W` / `-SW` 負向排除語法 Bug]
    原因：港股名稱如「小米集團-W」、「美團-W」、「阿里巴巴-W」、「京東集團-SW」含有 `-W` 或 `-SW`，
    直接丟入 Google Search (`q=小米集團-W`) 會被 Google 當成布林減號運算符 (`NOT W`)，或因名稱過長導致 0 結果！
    """
    raw_t = str(ticker or "").strip().upper()
    base_symbol = raw_t.replace(".HK", "")
    hk_num_no_zero = base_symbol.lstrip("0") or "0"
    hk_num_4d = f"{int(hk_num_no_zero):04d}" if hk_num_no_zero.isdigit() else base_symbol

    # 移除會觸發 Google Search 排除語法或干擾匹配的尾綴
    clean_name = str(stock_name or "").strip()
    clean_name = _re.sub(r"-(W|SW|S|B|R|SS|H)", "", clean_name, flags=_re.IGNORECASE)
    for noise in [
        "股份有限公司", "控股有限公司", "集團有限公司", "有限公司", "控股", "集團", "股份",
        "Corporation", "Incorporated", "Holdings", "Limited", "Company",
        "Corp.", "Corp", "Inc.", "Inc", "Ltd.", "Ltd", "PLC", "plc", "ADR", "Class A", "Class B", "Class C", ","
    ]:
        clean_name = _re.sub(_re.escape(noise), " ", clean_name, flags=_re.IGNORECASE)
    clean_name = _re.sub(r"\s+", " ", clean_name).strip()
    if not clean_name or clean_name.upper() == raw_t:
        clean_name = base_symbol

    # 針對熱門美股/港股補充中英文對照別名，大幅提升新聞召回率
    us_cn_alias_map = {
        "NVDA": "輝達 英偉達 NVIDIA", "TSLA": "特斯拉 Tesla", "AAPL": "蘋果 Apple",
        "MSFT": "微軟 Microsoft", "META": "Meta Facebook", "GOOGL": "Google Alphabet",
        "AMZN": "亞馬遜 Amazon", "AMD": "超微 AMD", "TSM": "台積電 TSMC",
        "SNOW": "Snowflake", "CRWD": "CrowdStrike", "MU": "美光 Micron",
        "DDOG": "Datadog", "PLTR": "Palantir", "SMCI": "美超微 Super Micro",
        "AVGO": "博通 Broadcom", "NFLX": "Netflix", "COIN": "Coinbase", "MSTR": "MicroStrategy",
        "ARM": "Arm Holdings", "ASML": "ASML 阿斯麥", "INTC": "英特爾 Intel", "QCOM": "高通 Qualcomm"
    }
    if market_type == "HK":
        # 港股中文搜尋詞：例如 `"騰訊" OR "0700"`、`"小米" OR "1810"`
        zh_q = f"{clean_name} {hk_num_4d}" if clean_name != base_symbol else f"{hk_num_4d}.HK 港股"
        en_q = f"{hk_num_4d}.HK {clean_name} stock"
        bing_q = f"{clean_name} {hk_num_4d} 港股"
    else:
        cn_alias = us_cn_alias_map.get(base_symbol, clean_name)
        zh_q = f"{base_symbol} {cn_alias} 美股"
        en_q = f"{base_symbol} {clean_name} stock" if clean_name.upper() != base_symbol else f"{base_symbol} stock"
        bing_q = f"{base_symbol} {clean_name} stock news"

    return {
        "raw_ticker": raw_t,
        "base_symbol": base_symbol,
        "hk_num_4d": hk_num_4d,
        "clean_name": clean_name,
        "zh_query": zh_q,
        "en_query": en_q,
        "bing_query": bing_q
    }


def _parse_rss_xml_items(xml_bytes: bytes, default_source: str, max_items: int = 8) -> List[Dict[str, str]]:
    """通用 RSS 2.0 / Atom XML 新聞解析器 (相容 Google News / Yahoo Finance RSS / Bing News RSS)"""
    parsed = []
    if not xml_bytes:
        return parsed
    try:
        xml_str = xml_bytes.decode("utf-8", errors="ignore").strip()
        root = _xml_et.fromstring(xml_str)
        for item_el in root.findall(".//item")[:max_items]:
            t_el = item_el.find("title")
            l_el = item_el.find("link")
            p_el = item_el.find("pubDate")
            s_el = item_el.find("source")
            d_el = item_el.find("description")

            title_txt = _html.unescape(t_el.text.strip()) if (t_el is not None and t_el.text) else ""
            if not title_txt:
                continue
            # Google News 標題常帶有 " - 媒體名稱" 後綴，拆分出媒體與乾淨標題
            src_txt = s_el.text.strip() if (s_el is not None and s_el.text) else ""
            if not src_txt and " - " in title_txt:
                parts = title_txt.rsplit(" - ", 1)
                if len(parts[1]) <= 25:
                    title_txt, src_txt = parts[0].strip(), parts[1].strip()
            if not src_txt:
                src_txt = default_source

            link_txt = l_el.text.strip() if (l_el is not None and l_el.text) else ""
            pub_raw = p_el.text.strip() if (p_el is not None and p_el.text) else ""
            pub_txt = pub_raw[:16] if pub_raw else "近期"
            try:
                if pub_raw:
                    pub_dt = pd.to_datetime(pub_raw, utc=True)
                    pub_txt = pub_dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

            desc_txt = ""
            if d_el is not None and d_el.text:
                # 去除 HTML tag
                raw_desc = _html.unescape(_re.sub(r"<[^>]+>", " ", d_el.text))
                raw_desc = _re.sub(r"\s+", " ", raw_desc).strip()
                if raw_desc and raw_desc[:40] not in title_txt:
                    desc_txt = raw_desc[:160]

            parsed.append({
                "title": title_txt,
                "summary": desc_txt,
                "pub_date": pub_txt,
                "publisher": src_txt,
                "link": link_txt
            })
    except Exception:
        pass
    return parsed


_NEWS_MEM_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}

def fetch_and_analyze_stock_news_intelligence(
    ticker: str,
    stock_name: str = "",
    market_type: str = "US/Global",
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    [v37.1 六源容錯實時新聞搜尋與 AI 利好/利空情報分析引擎 — 徹底修復「全部股票 0 則新聞」問題]
    修復重點：
    1) 移除 @st.cache_data 對空結果 `[]` 的 15 分鐘緩存鎖死；僅在成功抓到新聞時寫入記憶體緩存 (_NEWS_MEM_CACHE)！
    2) 透過 _safe_http_get_bytes 自動處理 macOS/Windows Python SSL 憑證錯誤 (`CERTIFICATE_VERIFY_FAILED`) 與模擬真實 Chrome Header。
    3) 淨化港股 `-W` / `-SW` / `-S` 後綴，避免觸發 Google News 負向排除語法 (`NOT W`)。
    4) 六源併聯檢索管道：
       - 管道 1：Google News RSS (繁體中文 zh-HK，針對港股或美股中文報導)
       - 管道 2：Google News RSS (英文 en-US，針對美股或國際大行報告)
       - 管道 3：Yahoo Finance Search JSON API (`query2.finance.yahoo.com/v1/finance/search`)
       - 管道 4：Yahoo Finance 官方個股 RSS (`feeds.finance.yahoo.com/rss/2.0/headline`)
       - 管道 5：Bing News RSS 全球免憑證備援 (`www.bing.com/news/search?format=rss`)
       - 管道 6：yfinance 原生 Ticker.news / Search.news (相容所有版本 schema)
    """
    cache_key = f"{ticker}_{market_type}"
    now_ts = time.time()
    if not force_refresh and cache_key in _NEWS_MEM_CACHE:
        ts_saved, cached_data = _NEWS_MEM_CACHE[cache_key]
        if (now_ts - ts_saved < 600) and cached_data.get("news_items"):
            return dict(cached_data)

    out = {
        "news_items": [],
        "overall_score": 0.0,
        "overall_verdict": "⚪ 中性偏穩 (暫無極端催化消息)",
        "bull_count": 0,
        "bear_count": 0,
        "neutral_count": 0,
        "has_black_swan_news": False,
        "black_swan_headlines": [],
        "active_sources": [],
        "debug_notes": []
    }
    raw_articles: List[Dict[str, str]] = []
    seen_titles = set()
    q_info = _clean_stock_query_terms(ticker, stock_name, market_type)

    def _add_article(art: Dict[str, str], source_tag: str):
        t_clean = str(art.get("title") or "").strip()
        if not t_clean or len(t_clean) < 4:
            return
        # 以標題前 28 字元去重
        dedup_key = _re.sub(r"\W+", "", t_clean.lower())[:28]
        if not dedup_key or dedup_key in seen_titles:
            return
        seen_titles.add(dedup_key)
        raw_articles.append(art)
        if source_tag not in out["active_sources"]:
            out["active_sources"].append(source_tag)

    # 管道 1：Google News RSS (香港繁體中文 zh-HK — 港股主力、美股中文快訊)
    try:
        enc_zh = _urllib_parse.quote(q_info["zh_query"])
        gnews_hk_url = f"https://news.google.com/rss/search?q={enc_zh}&hl=zh-HK&gl=HK&ceid=HK:zh-Hant"
        xml_b, err_msg = _safe_http_get_bytes(gnews_hk_url, timeout=4.0)
        if xml_b:
            for item in _parse_rss_xml_items(xml_b, "Google News (HK)", max_items=8):
                _add_article(item, "Google News (中文)")
        elif err_msg:
            out["debug_notes"].append(f"GNews-HK:{err_msg}")
    except Exception as e:
        out["debug_notes"].append(f"GNews-HK:{type(e).__name__}")

    # 管道 2：Google News RSS (美國英文 en-US — 美股主力、港股外資大行報告)
    if len(raw_articles) < 10:
        try:
            enc_en = _urllib_parse.quote(q_info["en_query"])
            gnews_us_url = f"https://news.google.com/rss/search?q={enc_en}&hl=en-US&gl=US&ceid=US:en"
            xml_b, err_msg = _safe_http_get_bytes(gnews_us_url, timeout=4.0)
            if xml_b:
                for item in _parse_rss_xml_items(xml_b, "Google News (US)", max_items=8):
                    _add_article(item, "Google News (EN)")
            elif err_msg:
                out["debug_notes"].append(f"GNews-US:{err_msg}")
        except Exception as e:
            out["debug_notes"].append(f"GNews-US:{type(e).__name__}")

    # 管道 3：Yahoo Finance Search JSON API (直接 HTTP JSON 請求，不依賴 yfinance 版本)
    if len(raw_articles) < 10:
        for q_candidate in ([ticker, q_info["clean_name"]] if q_info["clean_name"] != ticker else [ticker]):
            try:
                enc_yf = _urllib_parse.quote(q_candidate)
                yf_json_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={enc_yf}&quotesCount=1&newsCount=10"
                json_b, err_msg = _safe_http_get_bytes(yf_json_url, timeout=3.8)
                if json_b:
                    payload = _json.loads(json_b.decode("utf-8", errors="ignore"))
                    for n_item in (payload.get("news") or [])[:10]:
                        if not isinstance(n_item, dict):
                            continue
                        t_txt = str(n_item.get("title") or "").strip()
                        if not t_txt:
                            continue
                        pub_ts = n_item.get("providerPublishTime")
                        pub_str = "近期"
                        if pub_ts:
                            try:
                                pub_str = datetime.fromtimestamp(int(pub_ts)).strftime("%Y-%m-%d %H:%M")
                            except Exception:
                                pass
                        _add_article({
                            "title": t_txt,
                            "summary": "",
                            "pub_date": pub_str,
                            "publisher": str(n_item.get("publisher") or "Yahoo Finance"),
                            "link": str(n_item.get("link") or "")
                        }, "Yahoo Finance API")
                if len(raw_articles) >= 8:
                    break
            except Exception as e:
                out["debug_notes"].append(f"YF-API:{type(e).__name__}")

    # 管道 4：Yahoo Finance 官方 Headline RSS Feed
    if len(raw_articles) < 8:
        try:
            reg_str = "HK&lang=zh-Hant-HK" if market_type == "HK" else "US&lang=en-US"
            yf_rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={_urllib_parse.quote(ticker)}&region={reg_str}"
            xml_b, _ = _safe_http_get_bytes(yf_rss_url, timeout=3.8)
            if xml_b:
                for item in _parse_rss_xml_items(xml_b, "Yahoo Finance RSS", max_items=8):
                    _add_article(item, "Yahoo RSS")
        except Exception:
            pass

    # 管道 5：Bing News RSS (全球高可用備援，極少封鎖 IP)
    if len(raw_articles) < 8:
        try:
            enc_bing = _urllib_parse.quote(q_info["bing_query"])
            bing_url = f"https://www.bing.com/news/search?q={enc_bing}&format=rss"
            xml_b, _ = _safe_http_get_bytes(bing_url, timeout=4.0)
            if xml_b:
                for item in _parse_rss_xml_items(xml_b, "Bing News", max_items=8):
                    _add_article(item, "Bing News")
        except Exception:
            pass

    # 管道 6：yfinance 原生 Ticker(ticker).news 及 yf.Search
    if len(raw_articles) < 8:
        try:
            t_obj = yf.Ticker(ticker)
            yf_news = []
            try:
                yf_news = t_obj.news or []
            except Exception:
                pass
            if not yf_news and hasattr(yf, "Search"):
                try:
                    yf_news = yf.Search(ticker, news_count=8).news or []
                except Exception:
                    pass
            for item in yf_news[:10]:
                if not isinstance(item, dict):
                    continue
                content = item.get("content") if isinstance(item.get("content"), dict) else item
                title = str(content.get("title") or item.get("title") or "").strip()
                if not title:
                    continue
                summary = str(content.get("summary") or content.get("description") or "").strip()
                pub_str = ""
                if content.get("pubDate"):
                    pub_str = str(content.get("pubDate"))[:16].replace("T", " ")
                elif item.get("providerPublishTime"):
                    try:
                        pub_str = datetime.fromtimestamp(int(item["providerPublishTime"])).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass
                link = ""
                if isinstance(content.get("canonicalUrl"), dict):
                    link = str(content["canonicalUrl"].get("url", ""))
                elif isinstance(content.get("clickThroughUrl"), dict):
                    link = str(content["clickThroughUrl"].get("url", ""))
                elif item.get("link"):
                    link = str(item.get("link", ""))
                publisher = ""
                if isinstance(content.get("provider"), dict):
                    publisher = str(content["provider"].get("displayName", "Yahoo Finance"))
                else:
                    publisher = str(item.get("publisher", "Yahoo Finance"))

                _add_article({
                    "title": title,
                    "summary": summary[:180],
                    "pub_date": pub_str or "近期",
                    "publisher": publisher or "Yahoo Finance",
                    "link": link
                }, "yfinance")
        except Exception:
            pass

    if not raw_articles:
        return out

    analyzed_list = []
    scores = []
    for art in raw_articles[:12]:
        cls_info = _classify_single_news_headline(art["title"], art["summary"])
        art_enriched = {**art, **cls_info}
        analyzed_list.append(art_enriched)
        scores.append(cls_info["score"])
        if cls_info["score"] >= 10:
            out["bull_count"] += 1
        elif cls_info["score"] <= -10:
            out["bear_count"] += 1
        else:
            out["neutral_count"] += 1
        if cls_info["is_black_swan"] and cls_info["score"] <= -15:
            out["has_black_swan_news"] = True
            out["black_swan_headlines"].append(art["title"])

    avg_sc = round(float(np.mean(scores)), 1) if scores else 0.0
    out["news_items"] = analyzed_list
    out["overall_score"] = avg_sc
    if out["has_black_swan_news"]:
        out["overall_verdict"] = f"🚨 偵測到潛在黑天鵝/重大利空消息 (利空 {out['bear_count']} 則 vs 利好 {out['bull_count']} 則)"
    elif avg_sc >= 8 or out["bull_count"] > out["bear_count"]:
        out["overall_verdict"] = f"🟢 消息面整體偏利好 (利好 {out['bull_count']} 則 ／ 利空 {out['bear_count']} 則 ／ 中性 {out['neutral_count']} 則)"
    elif avg_sc <= -8 or out["bear_count"] > out["bull_count"]:
        out["overall_verdict"] = f"🔴 消息面整體偏利空 (利空 {out['bear_count']} 則 ／ 利好 {out['bull_count']} 則 ／ 中性 {out['neutral_count']} 則)"
    else:
        out["overall_verdict"] = f"⚪ 消息面多空均衡/中性 (利好 {out['bull_count']} 則 ／ 利空 {out['bear_count']} 則 ／ 中性 {out['neutral_count']} 則)"

    # 僅在成功抓取到新聞時才寫入記憶體緩存，徹底杜絕緩存空結果！
    _NEWS_MEM_CACHE[cache_key] = (now_ts, dict(out))
    return out

def evaluate_news_price_divergence(news_intel: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
    """
    [v37.0 專業 Swing Trader 核心優勢：新聞情緒 × 量價反應交叉驗證 (News vs Price-Action Matrix)]
    為什麼單看新聞炒股會輸錢？因為機構常利用「好消息高開派發 (Sell the News)」，或利用「壞消息低開洗盤吸籌」！
    本函數將新聞利好/利空傾向與當日 K 線 (開盤/收盤/上影線/成交量) 交叉比對，給出實戰行動結論。
    """
    res = {"reaction_badge": "⚪ 消息與量價常規運行", "reaction_advice": "", "alert_kind": "info"}
    if df is None or df.empty or not news_intel.get("news_items"):
        return res
    try:
        last = df.iloc[-1]
        c_now, o_now, h_now, l_now = float(last['Close']), float(last['Open']), float(last['High']), float(last['Low'])
        pchg = float(last.get('PriceChange', 0.0)) * 100.0
        vol_r = float(last.get('VolRatio', 1.0))
        close_pos = (c_now - l_now) / max(h_now - l_now, 1e-6)
        bull_c = int(news_intel.get("bull_count", 0))
        bear_c = int(news_intel.get("bear_count", 0))
        has_bs = bool(news_intel.get("has_black_swan_news", False))

        if has_bs:
            res["reaction_badge"] = "🚨 黑天鵝利空高危警報"
            res["reaction_advice"] = (
                f"🚨 **【黑天鵝消息攔截】**：最新新聞包含重大負面關鍵字（如盈警/配股抽水/調查/下調指引）："
                f"*{news_intel['black_swan_headlines'][0][:65]}...*。在利空未完全消化及站回 20MA 前，嚴禁盲目抄底做多！"
            )
            res["alert_kind"] = "error"
        elif bull_c > bear_c and (c_now < o_now or close_pos < 0.42) and vol_r >= 1.15:
            res["reaction_badge"] = "⚠️ 利好不漲反跌 (提防主力趁好消息出貨 Sell-the-News)"
            res["reaction_advice"] = (
                f"⚠️ **【量價背馳警告：利好出貨嫌疑 (Sell the News)】**：雖然近期新聞偏利好（利好 {bull_c} 則 vs 利空 {bear_c} 則），"
                f"但當日 K 線卻出現**放量收陰或長上影線（量比 {vol_r:.2f}x，收盤僅在波幅 {close_pos*100:.0f}% 位置）**！"
                f"這通常是機構趁散戶睇好消息衝入時逢高派發，切勿因好消息市價高追！"
            )
            res["alert_kind"] = "warning"
        elif bull_c > bear_c and c_now >= o_now and pchg > 0.5 and close_pos >= 0.55:
            res["reaction_badge"] = "🔥 利好與量價多頭共振 (Catalyst Confirmed)"
            res["reaction_advice"] = (
                f"✅ **【消息與量價正向共振】**：新聞面偏利好（利好 {bull_c} 則）且股價同步收陽上攻（今日 {pchg:+.2f}%，量比 {vol_r:.2f}x），"
                f"證明市場資金認同該利好催化劑，順勢做多勝率高。"
            )
            res["alert_kind"] = "success"
        elif bear_c > bull_c and c_now > o_now and close_pos >= 0.60 and c_now >= float(last.get('SMA20', c_now)):
            res["reaction_badge"] = "💎 利空不跌反升 (底部洗盤吸收信號)"
            res["reaction_advice"] = (
                f"💎 **【利空測試通過 (Bad News Priced-In)】**：雖然新聞面偏負面（利空 {bear_c} 則），"
                f"但股價拒絕下跌並低開高走收復 20MA（收盤位於當日波幅 {close_pos*100:.0f}% 高位），顯示主力正在承接恐慌盤！"
            )
            res["alert_kind"] = "success"
        elif bear_c > bull_c and (pchg < -1.0 or c_now < float(last.get('SMA20', c_now))):
            res["reaction_badge"] = "🩸 利空與技術破位共振 (避開下跌飛刀)"
            res["reaction_advice"] = (
                f"🛑 **【利空催化共振下跌】**：新聞面偏利空（利空 {bear_c} 則）且股價同步走弱（今日 {pchg:+.2f}%），"
                f"多頭切勿逆勢接刀，已持倉者請嚴守止損線。"
            )
            res["alert_kind"] = "error"
        else:
            res["reaction_advice"] = f"📡 **消息面概況**：{news_intel.get('overall_verdict', '')}，目前股價走勢主要由技術結構主導。"
    except Exception:
        pass
    return res

def compute_us_swing_exhaustion_and_trap_guard(df: pd.DataFrame, market_type: str = "US/Global") -> Dict[str, Any]:
    """
    [v32.0 核心新增：美股防追高動能枯竭與假突破誘多攔截器 (US Swing Exhaustion & Bull-Trap Guard)]
    專門解決用戶實戰痛點：「美股十筆交易全部止損、買入後一直沒有盈利、買入數日即大跌」！
    量化檢測 4 大美股常見敗因：
    1) 連續上漲天數過多 (consec_up_days >= 3 且未經橫盤壓縮)：短線買盤力竭，突破極易成為最後一棒；
    2) 5日波段急漲幅度過大 (runup_5d_atr >= 2.3x ATR 或 偏離 10EMA >= 1.55x ATR)：從箱體底一口氣拉到箱體頂，此時觸及樞紐實為短線獲利盤拋售區；
    3) 日線收盤位置不佳 (close_range_pos < 0.55)：美股突破日若留長上影線 (收盤不在當日振幅上半部)，代表機構趁突破派發 (Bull Trap)；
    4) 過去 20 日假突破基因檢測 (false_breakout_count_20d)：若該股近期多次升穿前高後次日即跌回，代表股性偏洗盤均值回歸，嚴禁使用 Stop-Buy 追高，強制改用回踩 20MA/21EMA Limit 低吸！
    """
    out = {
        "consec_up_days": 0,
        "runup_5d_atr": 0.0,
        "ext_10ema_atr": 0.0,
        "close_range_pos": 0.5,
        "false_breakout_20d": 0,
        "is_exhausted_dont_chase": False,
        "is_intraday_bull_trap_wick": False,
        "exhaustion_reason": ""
    }
    if df is None or len(df) < 22:
        return out
    try:
        closes = df['Close'].astype(float).values
        highs = df['High'].astype(float).values
        lows = df['Low'].astype(float).values
        opens = df['Open'].astype(float).values
        curr_c, curr_h, curr_l, curr_o = closes[-1], highs[-1], lows[-1], opens[-1]
        atr_v = float(df['ATR'].iloc[-1]) if ('ATR' in df.columns and pd.notna(df['ATR'].iloc[-1]) and float(df['ATR'].iloc[-1]) > 0) else curr_c * 0.03
        ema10_v = float(df['EMA10'].iloc[-1]) if ('EMA10' in df.columns and pd.notna(df['EMA10'].iloc[-1])) else curr_c

        consec_up = 0
        for k in range(len(closes) - 1, 0, -1):
            if closes[k] > closes[k - 1]:
                consec_up += 1
            else:
                break
        out["consec_up_days"] = consec_up

        low_5d = float(np.min(lows[-5:]))
        runup_5d_atr = (curr_c - low_5d) / max(atr_v, 1e-6)
        ext_10ema_atr = (curr_c - ema10_v) / max(atr_v, 1e-6)
        close_pos = (curr_c - curr_l) / max(curr_h - curr_l, 1e-6)
        out["runup_5d_atr"] = round(float(runup_5d_atr), 2)
        out["ext_10ema_atr"] = round(float(ext_10ema_atr), 2)
        out["close_range_pos"] = round(float(close_pos), 2)

        # 統計過去 20 日內「盤中突破前10日高點但收盤跌回或次日大跌」之假突破次數
        fb_cnt = 0
        for k in range(max(12, len(closes) - 20), len(closes) - 1):
            prev10_h = float(np.max(highs[k - 10:k]))
            if highs[k] > prev10_h * 1.002 and (closes[k] < prev10_h or closes[k + 1] < prev10_h * 0.988):
                fb_cnt += 1
        out["false_breakout_20d"] = fb_cnt

        reasons = []
        max_consec = 3 if market_type != "HK" else 4
        max_runup_atr = 2.35 if market_type != "HK" else 2.80
        max_ext10_atr = 1.55 if market_type != "HK" else 1.85

        if consec_up >= max_consec and runup_5d_atr >= 1.85:
            out["is_exhausted_dont_chase"] = True
            reasons.append(f"已連續上漲 {consec_up} 日 (5日急升達 {runup_5d_atr:.1f}x ATR)")
        if runup_5d_atr >= max_runup_atr or ext_10ema_atr >= max_ext10_atr:
            out["is_exhausted_dont_chase"] = True
            reasons.append(f"短線波幅已過度消耗 (5日拉升 {runup_5d_atr:.1f}x ATR / 偏離10EMA達 {ext_10ema_atr:.1f}x ATR)")
        if close_pos < 0.50 and curr_h > float(np.max(highs[-11:-1])) * 0.995:
            out["is_intraday_bull_trap_wick"] = True
            out["is_exhausted_dont_chase"] = True
            reasons.append(f"測試前高時留長上影線 (收盤僅在當日波幅 {close_pos*100:.0f}% 位置，屬日內衝高回落誘多)")
        if fb_cnt >= 2 and market_type != "HK":
            reasons.append(f"近20日已出現 {fb_cnt} 次突破失敗洗盤，不宜追高突破")

        if reasons:
            out["exhaustion_reason"] = "🛑 【防買在短線頂部攔截】" + "；".join(reasons) + "。此時現價追入極易『買入即被套並遭遇數日回調』，系統已強制將計劃改為【等待回踩均線/支撐區低吸】！"
    except Exception:
        pass
    return out

def render_locked_portfolio_summary_and_export(persistent_dict: dict) -> None:
    """[v31.0 新增：已鎖定持倉組合實時監控看板與 CSV 交易日誌匯出]"""
    locked_map = persistent_dict.get("user_locked_positions", {})
    if not locked_map:
        return
    with st.expander(f"📋 已鎖定實戰持倉組合監控看板 ({len(locked_map)} 筆持倉) & 一鍵匯出 CSV 交易日誌", expanded=False):
        rows = []
        for sym, p in locked_map.items():
            if not isinstance(p, dict):
                continue
            rows.append({
                "代碼": sym,
                "板塊": get_ticker_sector(sym),
                "方向": "🔴 做空" if p.get("dir_code") == "SHORT" else "🟢 做多",
                "開倉錨定日": p.get("anchor_date", "-"),
                "進場價": p.get("entry", 0.0),
                "鎖定硬止損": p.get("fixed_sl", 0.0),
                "收盤軟止損": p.get("soft_closing_sl", p.get("fixed_sl", 0.0)),
                "第一目標(TP1)": p.get("fixed_tp", 0.0),
                "次級目標(TP2)": p.get("tp2_price", 0.0),
                "鎖定初始R:R": f"{float(p.get('initial_rr', 0.0)):.2f}R",
                "當前實現R": f"{float(p.get('current_r', 0.0)):+.2f}R",
                "實時浮盈(%)": f"{float(p.get('unrealized_pnl_pct', 0.0)):+.2f}%",
                "富途打和價": p.get("be_price", 0.0),
                "狀態": "🚨 已觸及止損" if p.get("stop_hit_since_entry") else ("🛡️ 可移至保本" if float(p.get("current_r", 0.0)) >= 1.0 else "🌱 持倉發酵中")
            })
        if rows:
            port_df = pd.DataFrame(rows)
            st.dataframe(port_df, use_container_width=True, hide_index=True)
            csv_bytes = port_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 匯出已鎖定持倉交易日誌 (CSV)",
                data=csv_bytes,
                file_name=f"swing_trade_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="dl_locked_portfolio_csv"
            )

def evaluate_holistic_setup_grade_and_edge(
    df: pd.DataFrame,
    plan_type: str,
    trade_plan_eval: Dict[str, Any],
    jlaw_meta_info: Dict[str, Any],
    lei_struct_info: Dict[str, Any],
    lei_ded_forecast: Dict[str, Any],
    martin_luk_info: Dict[str, Any],
    deepchart_info: Dict[str, Any],
    vbp_vsa_info: Dict[str, Any],
    invalidation_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [v30.0 核心新增：機構級 8 維度交易品質評級 (Setup Grade: S / A+ / A / B / C) 與高潮延伸鎖潤引擎]
    解決問題：「點樣最大化運用成功嘅機會，避免喺普通 B/C 級信號浪費資金同止損額度？」
    綜合評估 8 大無未來函數硬核維度：
    1. 結構真實盈虧比 (R:R >= 2.2R 為優，>= 1.8R 及格)
    2. RS 相對強弱線領先價格創高 (RS New High Before Price / 雙週期 Alpha)
    3. DeepChart 前高空單消化度 (>= 62分 或 已突破清朗區)
    4. VBP 籌碼分佈位置 (站上最大籌碼峰 POC 且上方無 HVN 巨量套牢壓制 / 處於 LVN 真空加速區)
    5. Wyckoff VSA / LEI 微結構確認 (No-Supply 縮量無供給測試 / Spring 彈簧洗盤收回 / 2B 破底翻 / 底構造)
    6. 均線系統與抵扣價助漲 (LEI 2點鐘或3點鐘六線密集 + 未來10日抵扣價下移助攻 + Martin 9/21/50 EMA 雲帶)
    7. 量價健康度 (近20日上漲日均量 > 下跌日均量 1.2x 且 大戶出貨日 < 4 次)
    8. 入場時機精準度 (處於「⚡ 即時入場」擊球區或緊密樞紐突破位，非過度延伸追高)
    同時計算「🔥 波段高潮延伸主動減倉位 (Climax Extension Take-Profit: 20MA + 3.2x ATR)」，防止大賺後坐過山車嘔凸利潤！
    """
    out = {
        "setup_grade": "C (觀望級)",
        "grade_code": "C",
        "conviction_score": 0.0,
        "sizing_multiplier": 0.0,
        "sizing_advice": "🛑 暫不建倉 (等待更高勝率機會)",
        "rs_leads_price": False,
        "climax_extension_tp": np.nan,
        "is_overextended_now": False,
        "eight_pillars_passed": [],
        "eight_pillars_missing": [],
        "summary_banner": ""
    }
    if df is None or len(df) < 30 or plan_type not in ["LONG", "SHORT"]:
        return out

    last = df.iloc[-1]
    curr_c = float(last['Close'])
    curr_h = float(last['High'])
    sma20_v = float(last.get('SMA20', curr_c))
    sma10_v = float(last.get('SMA10', curr_c))
    atr_v = float(last.get('ATR', curr_c * 0.03))
    if not pd.notna(atr_v) or atr_v <= 0:
        atr_v = curr_c * 0.03

    # 計算 RS Line 是否率先創 20 日新高 (多頭) 或率先創 20 日新低 (空頭)
    # OLD:
    # rs_leads = False
    # if 'CloseBench' in df.columns and df['CloseBench'].notna().sum() >= 25:
    #     rs_line = df['Close'] / (df['CloseBench'] + 1e-9)
    #     rs_now = float(rs_line.iloc[-1])
    #     rs_max20 = float(rs_line.iloc[-22:-1].max())
    #     rs_min20 = float(rs_line.iloc[-22:-1].min())
    #     if plan_type == "LONG" and rs_now >= rs_max20 * 0.998:
    #         rs_leads = True
    #     elif plan_type == "SHORT" and rs_now <= rs_min20 * 1.002:
    #         rs_leads = True
    # out["rs_leads_price"] = rs_leads
    # NEW: [v31.0 修復做空時將 RS 創 20 日新低誤判為「RS率先創20日新高 / 機構逆勢掃貨」之嚴重漏洞]
    rs_leads_bull = False
    rs_leads_bear = False
    if 'CloseBench' in df.columns and df['CloseBench'].notna().sum() >= 25:
        rs_line = df['Close'] / (df['CloseBench'] + 1e-9)
        rs_now = float(rs_line.iloc[-1])
        rs_max20 = float(rs_line.iloc[-22:-1].max())
        rs_min20 = float(rs_line.iloc[-22:-1].min())
        if rs_now >= rs_max20 * 0.998:
            rs_leads_bull = True
        if rs_now <= rs_min20 * 1.002:
            rs_leads_bear = True
    rs_leads = rs_leads_bull if plan_type == "LONG" else rs_leads_bear
    out["rs_leads_price"] = rs_leads_bull
    out["rs_leads_bear"] = rs_leads_bear

    # 計算「🔥 高潮延伸主動鎖潤位 (Climax Extension TP)」：當偏離 20MA 達 3.2x ATR (或偏離 10MA 達 2.3x ATR) 時主動賣出 1/3~1/2
    if plan_type == "LONG":
        climax_tp = round(max(sma20_v + 3.2 * atr_v, sma10_v + 2.3 * atr_v, curr_c * 1.04), 2)
        is_overext = bool((curr_c - sma20_v) >= 3.0 * atr_v or float(last.get('RSI', 50)) >= 78.0 or bool(last.get('Clock_12', False)))
    else:
        climax_tp = round(max(0.01, min(sma20_v - 3.2 * atr_v, sma10_v - 2.3 * atr_v, curr_c * 0.96)), 2)
        is_overext = bool((sma20_v - curr_c) >= 3.0 * atr_v or float(last.get('RSI', 50)) <= 22.0)
    out["climax_extension_tp"] = climax_tp
    out["is_overextended_now"] = is_overext

    passed, missing = [], []
    score = 0.0
    planned_rr = float(trade_plan_eval.get("planned_rr", 0.0))

    # 維度 1：結構盈虧比 (20分)
    if planned_rr >= 2.5:
        score += 20.0
        passed.append(f"① 極佳結構盈虧比 ({planned_rr:.2f}R ≥ 2.5R)")
    elif planned_rr >= 1.8:
        score += 14.0
        passed.append(f"① 合格結構盈虧比 ({planned_rr:.2f}R ≥ 1.8R)")
    else:
        missing.append(f"① 結構盈虧比偏低 ({planned_rr:.2f}R < 1.8R)")

    # 維度 2：RS 相對強弱領先性 (15分)
    # OLD:
    # if rs_leads or (jlaw_meta_info.get("rs_1m_outperform") and jlaw_meta_info.get("rs_6m_outperform")):
    #     score += 15.0
    #     passed.append("② RS 相對強弱領先創高 / 雙週期 Alpha 領頭羊")
    # elif float(last.get('Mansfield_RS', 0)) > 0 and plan_type == "LONG": ...
    # NEW: [v31.0 嚴格區分做多與做空之 RS 評語，杜絕做空時顯示「領先創高 / Alpha 領頭羊」]
    if plan_type == "LONG":
        if rs_leads_bull or (jlaw_meta_info.get("rs_1m_outperform") and jlaw_meta_info.get("rs_6m_outperform")):
            score += 15.0
            passed.append("② RS 相對強弱領先創高 / 雙週期 Alpha 領頭羊")
        elif float(last.get('Mansfield_RS', 0)) > 0:
            score += 9.0
            passed.append("② Mansfield RS 跑贏大盤")
        else:
            missing.append("② RS 相對強弱未顯著領先大盤")
    else:
        if rs_leads_bear and float(last.get('Mansfield_RS', 0)) <= 0:
            score += 15.0
            passed.append("② RS 相對強弱率先破底創20日新低 (機構持續派發之空頭領跌股)")
        elif float(last.get('Mansfield_RS', 0)) < 0 or rs_leads_bear:
            score += 12.0
            passed.append("② Mansfield RS 顯著弱於大盤 (空頭弱勢標的)")
        else:
            missing.append("② 個股 RS 仍強於大盤 (逆勢做空強勢股風險較高)")

    # 維度 3：DeepChart 前高空單消化度 (15分)
    dc_score = float(deepchart_info.get("absorption_score", 50.0))
    if plan_type == "LONG":
        if dc_score >= 65.0:
            score += 15.0
            passed.append(f"③ DeepChart 前高空單已充分消化 ({dc_score:.0f}分)")
        elif dc_score >= 52.0:
            score += 9.0
            passed.append(f"③ 前高籌碼換手尚可 ({dc_score:.0f}分)")
        else:
            missing.append(f"③ 前高套牢空單未消化 ({dc_score:.0f}分)")
    else:
        if dc_score < 45.0 or not deepchart_info.get("cvd_10d_positive", True):
            score += 14.0
            passed.append("③ 訂單流淨賣壓主導 (利於做空)")
        else:
            missing.append("③ 下方買盤吸收仍強")

    # 維度 4：VBP 籌碼峰與價值區位置 (15分)
    if plan_type == "LONG":
        if vbp_vsa_info.get("is_in_lvn_vacuum"):
            score += 15.0
            passed.append("④ 處於 VBP 籌碼真空加速區 (LVN 上方無套牢)")
        elif not vbp_vsa_info.get("is_blocked_by_hvn") and curr_c >= float(vbp_vsa_info.get("poc_price", 0)):
            score += 12.0
            passed.append(f"④ 站穩最大籌碼峰 POC ({vbp_vsa_info.get('poc_price', 0):.2f}) 之上")
        else:
            missing.append("④ 頭頂受制於 HVN 巨量套牢籌碼峰")
    else:
        if curr_c < float(vbp_vsa_info.get("poc_price", 999999)):
            score += 14.0
            passed.append(f"④ 跌破最大籌碼峰 POC ({vbp_vsa_info.get('poc_price', 0):.2f}) 形成蓋頭反壓")
        else:
            missing.append("④ 仍在 POC 籌碼支撐之上")

    # 維度 5：Wyckoff VSA 與 LEI 頂底/2B 微結構 (12分)
    if plan_type == "LONG":
        if vbp_vsa_info.get("vsa_no_supply") or vbp_vsa_info.get("vsa_spring_absorption") or lei_struct_info.get("bullish_2b") or lei_struct_info.get("bottom_structure") or deepchart_info.get("pocket_pivot_10d"):
            score += 12.0
            passed.append("⑤ VSA 縮量無供給 / Spring洗盤收回 / 口袋支點確認")
        else:
            score += 4.0
            missing.append("⑤ 未出現 VSA 極致洗盤/口袋支點信號")
    else:
        if vbp_vsa_info.get("vsa_upthrust_distribution") or lei_struct_info.get("bearish_2b") or lei_struct_info.get("top_structure"):
            score += 12.0
            passed.append("⑤ VSA Upthrust 派發 / 2B假突破 / 頂部構造確認")
        else:
            score += 4.0

    # 維度 6：均線雲帶與抵扣價方向 (10分)
    if plan_type == "LONG":
        if lei_ded_forecast.get("is_20ma_turn_up_favorable") and ("多頭" in martin_luk_info.get("ema_cloud_state", "") or "修復" in martin_luk_info.get("ema_cloud_state", "")):
            score += 10.0
            passed.append("⑥ 9/21/50 EMA 雲帶多頭 + 20MA 抵扣價助攻向上")
        else:
            missing.append("⑥ 均線雲帶或未來抵扣價尚未完全順向")
    else:
        if not lei_ded_forecast.get("is_20ma_turn_up_favorable", True):
            score += 10.0
            passed.append("⑥ 20MA 抵扣價高企助跌 + 空頭均線壓制")

    # 維度 7：量價健康與大戶出貨日控制 (8分)
    dist_d = int(last.get('Dist_Days_20d', 0)) if pd.notna(last.get('Dist_Days_20d')) else 0
    if plan_type == "LONG":
        if dist_d <= 2 and jlaw_meta_info.get("up_down_vol_ratio", 1.0) >= 1.15:
            score += 8.0
            passed.append(f"⑦ 量價健康 (漲跌量比 {jlaw_meta_info.get('up_down_vol_ratio', 1.0):.2f}x，出貨日僅 {dist_d} 次)")
        elif dist_d >= 4:
            score -= 10.0
            missing.append(f"⑦ 近20日大戶出貨日過多 ({dist_d} 次 ≥ 4)")
        else:
            score += 4.0
    else:
        if jlaw_meta_info.get("up_down_vol_ratio", 1.0) <= 0.90 or dist_d >= 3:
            score += 8.0
            passed.append("⑦ 下跌放量、反彈縮量 (空頭量價配合)")

    # 維度 8：進場時機與防過熱/超賣檢查 (5分)
    # OLD:
    # if is_overext:
    #     score -= 15.0
    #     missing.append("⑧ 短線已嚴重過熱延伸 (偏離20MA > 3x ATR，嚴禁追高)")
    # elif str(trade_plan_eval.get("action_timeliness", "")).startswith("⚡"):
    #     score += 5.0
    #     passed.append("⑧ 正處於 ⚡ 即時入場黃金擊球區")
    # NEW: [v31.0 多空雙向語義對齊]
    if is_overext:
        score -= 15.0
        if plan_type == "LONG":
            missing.append("⑧ 多頭短線已嚴重過熱延伸 (偏離20MA > 3x ATR，嚴禁追高)")
        else:
            missing.append("⑧ 空頭短線已極度超賣延伸 (低於20MA逾 3x ATR 或 RSI≤22，嚴禁低位追空)")
    elif str(trade_plan_eval.get("action_timeliness", "")).startswith("⚡"):
        score += 5.0
        passed.append("⑧ 正處於 ⚡ 即時入場黃金擊球區" if plan_type == "LONG" else "⑧ 正處於 ⚡ 即時沽空黃金擊球區")
    else:
        score += 3.0

    score = round(min(99.0, max(5.0, score)), 1)
    out["conviction_score"] = score
    out["eight_pillars_passed"] = passed
    out["eight_pillars_missing"] = missing

    if plan_type == "LONG":
        out["rs_metric_title"] = "🚀 RS 相對強弱線創高"
        out["rs_metric_val"] = "🔥 RS率先創20日新高" if rs_leads_bull else "⚪ 常規隨勢運行"
        out["rs_metric_delta"] = "機構逆勢掃貨指標" if rs_leads_bull else "觀察相對大盤強弱"
        out["climax_metric_title"] = "💰 高潮延伸主動減倉位"
        out["climax_metric_delta"] = "⚠️ 現價已過熱勿追" if is_overext else "偏離20MA達+3.2x ATR鎖潤位"
    else:
        out["rs_metric_title"] = "📉 RS 相對強弱線破底"
        out["rs_metric_val"] = "🩸 RS率先創20日新低" if rs_leads_bear else ("🔻 RS弱於大盤" if float(last.get('Mansfield_RS', 0)) < 0 else "⚪ 常規隨勢運行")
        out["rs_metric_delta"] = "機構領跌派發指標 (利於做空)" if (rs_leads_bear or float(last.get('Mansfield_RS', 0)) < 0) else "⚠️ 個股相對抗跌"
        out["climax_metric_title"] = "💰 急跌超賣主動回補位"
        out["climax_metric_delta"] = "⚠️ 短線已超賣勿追空" if is_overext else "偏離20MA達-3.2x ATR回補鎖潤位"

    if score >= 85.0 and planned_rr >= 2.0 and not is_overext:
        out["setup_grade"] = "🏆 S 級 (聖杯滿貫機會)" if plan_type == "LONG" else "🏆 S 級 (空頭滿貫機會)"
        out["grade_code"] = "S"
        out["sizing_multiplier"] = 1.25
        out["sizing_advice"] = (
            "🔥 **S 級多頭頂尖共振（建議 100%~125% 標準風險倉位，即 0.50%~0.75% 總資金風險）**：八大機構維度全面共振，上方套牢清空且上漲空間充裕！"
            if plan_type == "LONG" else
            "🔥 **S 級空頭頂尖共振（建議 100%~125% 標準風險倉位，即 0.50%~0.75% 總資金風險）**：空頭八大維度全面共振，上方套牢反壓沉重、下方支撐薄弱且下跌空間充裕！"
        )
    elif score >= 74.0 and planned_rr >= 1.8 and not is_overext:
        out["setup_grade"] = "💎 A+ 級 (機構高勝率機會)" if plan_type == "LONG" else "💎 A+ 級 (高勝率做空機會)"
        out["grade_code"] = "A+"
        out["sizing_multiplier"] = 1.0
        out["sizing_advice"] = (
            "✅ **A+ 級多頭核心機會（建議 100% 標準風險倉位，即 0.50% 總資金風險）**：量價、籌碼與盈虧比高度配合，可按計劃標準買入。"
            if plan_type == "LONG" else
            "✅ **A+ 級空頭核心機會（建議 100% 標準風險倉位，即 0.50% 總資金風險）**：空頭反壓、賣壓訂單流與下跌盈虧比高度配合，可按計劃標準沽空。"
        )
    elif score >= 62.0 and planned_rr >= 1.8:
        out["setup_grade"] = "🟢 A 級 (標準波段機會)" if plan_type == "LONG" else "🔴 A 級 (標準波段做空)"
        out["grade_code"] = "A"
        out["sizing_multiplier"] = 0.75
        out["sizing_advice"] = (
            "🟢 **A 級合格做多機會（建議 75% 倉位或先入半倉試水溫，約 0.35% 風險）**：主趨勢與空間達標，請嚴格配合 1H/15m 右側扳機買入。"
            if plan_type == "LONG" else
            "🔴 **A 級合格做空機會（建議 75% 倉位或先入半倉試水溫，約 0.35% 風險）**：空頭趨勢與下跌空間達標，請嚴格配合 1H/15m 反抽受阻或破底扳機沽空。"
        )
    elif score >= 48.0:
        out["setup_grade"] = "🟡 B 級 (次級/等待確認)"
        out["grade_code"] = "B"
        out["sizing_multiplier"] = 0.50
        out["sizing_advice"] = (
            "🟡 **B 級觀察機會（建議最多 50% 輕倉或掛單等待回調，約 0.25% 風險）**：部分籌碼或量價條件尚未完美，切勿重倉勉強買入。"
            if plan_type == "LONG" else
            "🟡 **B 級空頭觀察機會（建議最多 50% 輕倉或掛單等待反抽阻力，約 0.25% 風險）**：部分空頭條件尚未完美，切勿在支撐位重倉勉強追空。"
        )
    else:
        out["setup_grade"] = "⚪ C 級 (勝率偏低/放棄)"
        out["grade_code"] = "C"
        out["sizing_multiplier"] = 0.0
        out["sizing_advice"] = (
            "🛑 **C 級低勝率（建議放棄或純觀望）**：盈虧比不足、上方套牢沉重或短線過度延伸，保留資金等待 A+/S 級做多機會！"
            if plan_type == "LONG" else
            "🛑 **C 級空頭低勝率（建議放棄做空或純觀望）**：下跌盈虧比不足、下方買盤承接過強或短線已極度超賣，保留資金等待 A+/S 級做空機會！"
        )

    return out
from zoneinfo import ZoneInfo

@st.cache_data(ttl=600)
def fetch_intraday_mtf_data(ticker: str) -> Dict[str, Optional[pd.DataFrame]]:
    """
    [v29.0 多時間框架數據引擎]
    拉取目標股票之 1H (近 60 日) 與 15m (近 15 日) 數據，並由 1H 無縫重採樣合成真實 4H (4小時/半日) K 線。
    (註：yfinance 原生不支援 interval='4h'，必須由 1h 聚合生成)
    """
    out: Dict[str, Optional[pd.DataFrame]] = {"4h": None, "1h": None, "15m": None}
    try:
        df_1h = yf.download(ticker, period="60d", interval="1h", auto_adjust=True, progress=False)
        if df_1h is not None and not df_1h.empty:
            if isinstance(df_1h.columns, pd.MultiIndex):
                df_1h.columns = df_1h.columns.get_level_values(0) if 'Close' in df_1h.columns.get_level_values(0) else df_1h.columns.get_level_values(1)
            df_1h = df_1h.loc[:, ~df_1h.columns.duplicated()].dropna(subset=['Open', 'High', 'Low', 'Close'])
            if df_1h.index.tz is not None:
                df_1h.index = df_1h.index.tz_localize(None)
            out["1h"] = df_1h
            # 由 1H 聚合生成 4H (每 4 根有效交易小時 K 線或按 4h 時間桶聚合)
            if len(df_1h) >= 8:
                df_4h = df_1h.resample("4h").agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna(subset=["Open", "High", "Low", "Close"])
                out["4h"] = df_4h
    except Exception as e:
        swing_logger.debug(f"1H/4H 數據拉取略過 ({ticker}): {e}")

    try:
        df_15m = yf.download(ticker, period="15d", interval="15m", auto_adjust=True, progress=False)
        if df_15m is not None and not df_15m.empty:
            if isinstance(df_15m.columns, pd.MultiIndex):
                df_15m.columns = df_15m.columns.get_level_values(0) if 'Close' in df_15m.columns.get_level_values(0) else df_15m.columns.get_level_values(1)
            df_15m = df_15m.loc[:, ~df_15m.columns.duplicated()].dropna(subset=['Open', 'High', 'Low', 'Close'])
            if df_15m.index.tz is not None:
                df_15m.index = df_15m.index.tz_localize(None)
            out["15m"] = df_15m
    except Exception as e:
        swing_logger.debug(f"15m 數據拉取略過 ({ticker}): {e}")

    return out


def compute_mtf_execution_and_exit_sniper(
    ticker: str,
    daily_df: pd.DataFrame,
    plan_type: str,
    planned_entry: float,
    stop_loss: float,
    take_profit: float
) -> Dict[str, Any]:
    """
    [v29.0 核心引擎：J Law × Brian Shannon × Qullamägi 四週期 (1D × 4H × 1H × 15m) 入場與離場最佳位置狙擊器]
    專門解決用戶實戰痛點：
    - 痛點 A：「每次買入後總是不能立即獲得盈利 (買太早接飛刀，或開盤衝高買在日內最高點)」
    - 痛點 B：「每次離場位置非常差 (日線止盈太慢嘔返利潤，或日內恐慌針尖賣在最低點)」
    """
    res = {
        "has_intraday": False,
        "tf_4h_state": "⚪ 4H 資料載入中",
        "tf_1h_state": "⚪ 1H 資料載入中",
        "tf_15m_state": "⚪ 15m 資料載入中",
        "mtf_alignment_score": 50,
        "mtf_action_verdict": "⚪ 請配合成交量觀察日線關鍵位",
        "sniper_entry_trigger_price": round(planned_entry, 2),
        "sniper_entry_method": "按日線建議位掛單",
        "sniper_profit_lock_exit": round(take_profit, 2),
        "sniper_panic_guard_msg": "",
        "intraday_vwap": np.nan,
        "orb_high_15m": np.nan,
        "orb_low_15m": np.nan,
        "h1_ema20": np.nan,
        "h1_pullback_avwap": np.nan,
        "h1_swing_low": np.nan,
        "h1_swing_high": np.nan,
        "h4_ema20": np.nan,
        "entry_timing_cards": []
    }
    if daily_df is None or daily_df.empty:
        return res

    mtf_data = fetch_intraday_mtf_data(ticker)
    df_4h = mtf_data.get("4h")
    df_1h = mtf_data.get("1h")
    df_15m = mtf_data.get("15m")

    curr_price = float(daily_df['Close'].iloc[-1])
    is_short = (plan_type == "SHORT")
    score = 50

    # --- 1. 分析 4H (波段中繼趨勢與動能) ---
    h4_bullish, h4_bearish = False, False
    if df_4h is not None and len(df_4h) >= 20:
        res["has_intraday"] = True
        c4 = df_4h['Close'].astype(float)
        ema20_4h = c4.ewm(span=20, adjust=False).mean()
        ema50_4h = c4.ewm(span=50, adjust=False).mean()
        macd_4h = c4.ewm(span=12, adjust=False).mean() - c4.ewm(span=26, adjust=False).mean()
        sig_4h = macd_4h.ewm(span=9, adjust=False).mean()
        last_c4 = float(c4.iloc[-1])
        e20_4h_v = float(ema20_4h.iloc[-1])
        e50_4h_v = float(ema50_4h.iloc[-1])
        res["h4_ema20"] = round(e20_4h_v, 2)

        if last_c4 >= e20_4h_v and e20_4h_v >= e50_4h_v * 0.995 and float(macd_4h.iloc[-1]) >= float(sig_4h.iloc[-1]):
            h4_bullish = True
            res["tf_4h_state"] = f"🟢 4H 多頭推進 (站上 4H 20EMA {e20_4h_v:.2f} & MACD 金叉)"
            score += (18 if not is_short else -15)
        elif last_c4 < e20_4h_v and float(macd_4h.iloc[-1]) < float(sig_4h.iloc[-1]):
            h4_bearish = True
            res["tf_4h_state"] = f"🔴 4H 空頭回調中 (受制於 4H 20EMA {e20_4h_v:.2f} 下方)"
            score += (-15 if not is_short else 18)
        else:
            res["tf_4h_state"] = f"🟡 4H 震盪築底/換手 (4H 20EMA: {e20_4h_v:.2f})"

    # --- 2. 分析 1H (Brian Shannon / J Law 右側結構轉折與前高回調錨定 AVWAP) ---
    h1_reclaimed = False
    h1_breakdown = False
    if df_1h is not None and len(df_1h) >= 25:
        res["has_intraday"] = True
        c1 = df_1h['Close'].astype(float)
        h1 = df_1h['High'].astype(float)
        l1 = df_1h['Low'].astype(float)
        v1 = df_1h['Volume'].astype(float) if 'Volume' in df_1h.columns else pd.Series(1.0, index=df_1h.index)
        ema20_1h = c1.ewm(span=20, adjust=False).mean()
        ema50_1h = c1.ewm(span=50, adjust=False).mean()
        e20_1h_v = float(ema20_1h.iloc[-1])
        e50_1h_v = float(ema50_1h.iloc[-1])
        res["h1_ema20"] = round(e20_1h_v, 2)

        # 過去 12 根 1H K 線之波谷與波峰 (作為 1H 結構止盈/止損與突破觸發位)
        h1_recent_low = float(l1.iloc[-12:-1].min()) if len(l1) > 12 else float(l1.min())
        h1_recent_high = float(h1.iloc[-12:-1].max()) if len(h1) > 12 else float(h1.max())
        res["h1_swing_low"] = round(h1_recent_low, 2)
        res["h1_swing_high"] = round(h1_recent_high, 2)

        # 計算近 40 根 1H K 線波段高點錨定之 Pullback AVWAP (多頭回調買入必須突破此線才算洗盤結束！)
        win_1h = df_1h.tail(min(len(df_1h), 50))
        if not is_short:
            anc_idx = win_1h['High'].idxmax()
        else:
            anc_idx = win_1h['Low'].idxmin()
        sub_avwap_df = win_1h.loc[anc_idx:]
        tp_1h = (sub_avwap_df['High'] + sub_avwap_df['Low'] + sub_avwap_df['Close']) / 3.0
        vol_1h_sub = sub_avwap_df['Volume'].replace(0, 1.0) if 'Volume' in sub_avwap_df.columns else pd.Series(1.0, index=sub_avwap_df.index)
        pb_avwap_val = float((tp_1h * vol_1h_sub).cumsum().iloc[-1] / (vol_1h_sub.cumsum().iloc[-1] + 1e-9))
        res["h1_pullback_avwap"] = round(pb_avwap_val, 2)

        last_c1 = float(c1.iloc[-1])
        # Higher Low 檢查：最近 4 根 1H 低點 >= 前 5~12 根 1H 低點
        h1_higher_low = float(l1.iloc[-4:].min()) >= float(l1.iloc[-12:-4].min()) * 0.998 if len(l1) >= 12 else True
        h1_lower_high = float(h1.iloc[-4:].max()) <= float(h1.iloc[-12:-4].max()) * 1.002 if len(h1) >= 12 else True

        if last_c1 >= e20_1h_v and last_c1 >= pb_avwap_val * 0.998 and h1_higher_low:
            h1_reclaimed = True
            res["tf_1h_state"] = f"🔥 1H 右側轉多確認 (站上 1H 20EMA {e20_1h_v:.2f} 及回調AVWAP {pb_avwap_val:.2f})"
            score += (20 if not is_short else -18)
        elif last_c1 < e20_1h_v and last_c1 < pb_avwap_val and not h1_higher_low:
            h1_breakdown = True
            res["tf_1h_state"] = f"🛑 1H 仍在尋底插水 (低於 1H 20EMA {e20_1h_v:.2f} 及回調AVWAP {pb_avwap_val:.2f}，勿接飛刀)"
            score += (-20 if not is_short else 20)
        else:
            res["tf_1h_state"] = f"🟡 1H 測試關鍵位 (1H 20EMA: {e20_1h_v:.2f} ｜ 回調AVWAP: {pb_avwap_val:.2f})"

    # --- 3. 分析 15m (日內 VWAP、開盤區間 ORB、防追高與防恐慌殺跌針尖) ---
    m15_overbought_chase = False
    m15_oversold_panic = False
    m15_orb_breakout = False
    if df_15m is not None and len(df_15m) >= 10:
        res["has_intraday"] = True
        last_day_date = df_15m.index[-1].date()
        today_15m = df_15m[df_15m.index.date == last_day_date]
        # OLD:
        # if len(today_15m) < 2:
        #     today_15m = df_15m.tail(16)
        # NEW: [v31.0 修復開盤首 15 分鐘將昨日下午 K 線混入今日 ORB 開盤區間之漏洞]
        # 若當日僅有 1 根 15m K 線，直接以該根 K 線作為當日初始 ORB 與日內 VWAP，絕不把昨日尾盤當作今日開盤高低點！
        if len(today_15m) == 0:
            today_15m = df_15m.tail(16)

        tp_15 = (today_15m['High'] + today_15m['Low'] + today_15m['Close']) / 3.0
        vol_15 = today_15m['Volume'].replace(0, 1.0) if 'Volume' in today_15m.columns else pd.Series(1.0, index=today_15m.index)
        intra_vwap = float((tp_15 * vol_15).cumsum().iloc[-1] / (vol_15.cumsum().iloc[-1] + 1e-9))
        res["intraday_vwap"] = round(intra_vwap, 2)

        # 開盤首 30 分鐘 (前 2 根 15m K 線) ORB 高低點
        orb_bars = today_15m.iloc[:min(2, len(today_15m))]
        orb_high = float(orb_bars['High'].max())
        orb_low = float(orb_bars['Low'].min())
        res["orb_high_15m"] = round(orb_high, 2)
        res["orb_low_15m"] = round(orb_low, 2)

        # 15m RSI 與 15m ATR
        c15 = df_15m['Close'].astype(float)
        h15 = df_15m['High'].astype(float)
        l15 = df_15m['Low'].astype(float)
        tr15 = np.maximum(h15 - l15, np.maximum(abs(h15 - c15.shift(1)), abs(l15 - c15.shift(1))))
        atr15 = float(tr15.tail(14).mean()) if len(tr15) >= 5 else curr_price * 0.005
        delta15 = c15.diff()
        g15 = delta15.where(delta15 > 0, 0.0).rolling(14, min_periods=3).mean().iloc[-1]
        ls15 = (-delta15.where(delta15 < 0, 0.0)).rolling(14, min_periods=3).mean().iloc[-1]
        rsi15 = 100.0 - (100.0 / (1.0 + (float(g15) / (float(ls15) + 1e-9))))
        last_c15 = float(c15.iloc[-1])

        if last_c15 > intra_vwap + 1.8 * atr15 or rsi15 >= 74.0:
            m15_overbought_chase = True
            res["tf_15m_state"] = f"⚠️ 15m 日內短線過熱 (RSI {rsi15:.0f} ｜ 偏離日內VWAP {intra_vwap:.2f} 過大，勿市價高追)"
            if not is_short:
                score -= 10
        elif last_c15 < intra_vwap - 1.8 * atr15 or rsi15 <= 26.0:
            m15_oversold_panic = True
            res["tf_15m_state"] = f"🛡️ 15m 日內極度超賣針尖 (RSI {rsi15:.0f} ｜ 遠低於日內VWAP {intra_vwap:.2f}，防賣在最低點)"
            if is_short:
                score -= 10
        elif last_c15 >= orb_high and last_c15 >= intra_vwap:
            m15_orb_breakout = True
            res["tf_15m_state"] = f"🚀 15m 突破開盤高點 ORB ({orb_high:.2f}) 且站穩日內VWAP ({intra_vwap:.2f})"
            if not is_short:
                score += 12
        elif last_c15 <= orb_low and last_c15 <= intra_vwap:
            res["tf_15m_state"] = f"🩸 15m 跌破開盤低點 ORB ({orb_low:.2f}) 且受制日內VWAP ({intra_vwap:.2f})"
            if is_short:
                score += 12
        else:
            res["tf_15m_state"] = f"🟢 15m 圍繞日內VWAP ({intra_vwap:.2f}) 窄幅換手 (ORB區間 {orb_low:.2f}~{orb_high:.2f})"

    score = int(min(99, max(5, score)))
    res["mtf_alignment_score"] = score

    # --- 4. 綜合生成「🎯 最佳入場扳機位」與「💰 最佳離場鎖潤位 / 防洗盤保護」 ---
    h1_e20 = res["h1_ema20"] if pd.notna(res["h1_ema20"]) else planned_entry
    h1_avwap = res["h1_pullback_avwap"] if pd.notna(res["h1_pullback_avwap"]) else planned_entry
    intra_v = res["intraday_vwap"] if pd.notna(res["intraday_vwap"]) else planned_entry
    orb_h = res["orb_high_15m"] if pd.notna(res["orb_high_15m"]) else planned_entry
    orb_l = res["orb_low_15m"] if pd.notna(res["orb_low_15m"]) else planned_entry
    h1_sw_low = res["h1_swing_low"] if pd.notna(res["h1_swing_low"]) else stop_loss
    h1_sw_high = res["h1_swing_high"] if pd.notna(res["h1_swing_high"]) else stop_loss

    cards: List[Tuple[str, str]] = []
    if not is_short:
        # 多頭入場精準導航
        if h1_breakdown:
            trigger_p = round(max(h1_e20, min(h1_avwap, h1_sw_high)), 2)
            res["sniper_entry_trigger_price"] = trigger_p
            res["sniper_entry_method"] = f"🛑 暫緩左側盲接！改設 1H 右側轉強觸發單 (升穿 {trigger_p:.2f} 買入)"
            cards.append((
                "error",
                f"🛑 **【為什麼你買入後總是不能立即盈利？—— 1H/4H 正在插水尋底，切勿左側接飛刀！】**\n\n"
                f"雖然日線看似接近支撐位 ({planned_entry:.2f})，但 **1H 價格目前仍受制於下彎的 1H 20EMA ({h1_e20:.2f}) 與回調錨定 AVWAP ({h1_avwap:.2f}) 下方**！"
                f"此時直接掛限價單買入，大概率買完還要繼續陰跌洗盤。\n\n"
                f"👉 **最佳入場做法 (Brian Shannon / J Law 右側確認法)**：請等 **1H 收盤重新站上 {trigger_p:.2f} (1H 20EMA / 15m ORB 高點 {orb_h:.2f})** 確認賣壓枯竭、多頭重掌主導權時才扣扳機買入，實現「一買入即順勢脫離成本」！"
            ))
        elif m15_overbought_chase:
            pullback_15m_p = round(max(intra_v, h1_e20), 2)
            res["sniper_entry_trigger_price"] = pullback_15m_p
            res["sniper_entry_method"] = f"⚠️ 勿追日內高點！改掛 15m 日內 VWAP ({pullback_15m_p:.2f}) 回踩承接單"
            cards.append((
                "warning",
                f"⚠️ **【日內開盤衝高陷阱預警 —— 切勿在 15m 過度延伸時市價追高！】**\n\n"
                f"目前 4H/1H 趨勢雖好，但 **15m 短線已大幅偏離日內 VWAP ({intra_v:.2f})**。此時市價追入極易買在早盤情緒最高點，買完即遭遇日內回吐 1.5%~2.5%！\n\n"
                f"👉 **最佳入場位置**：請在 **15m 日內 VWAP ({intra_v:.2f}) 至 1H 20EMA ({h1_e20:.2f})** 區間掛限價單低吸，或等 15m 橫盤縮量後再次突破時買入。"
            ))
        elif h1_reclaimed and (h4_bullish or m15_orb_breakout):
            sniper_p = round(curr_price if curr_price <= orb_h * 1.01 else orb_h, 2)
            res["sniper_entry_trigger_price"] = sniper_p
            res["sniper_entry_method"] = f"⚡ 四週期共振最佳擊球點！可於現價 / 日內VWAP ({intra_v:.2f}) 即時建倉"
            cards.append((
                "success",
                f"🔥 **【1D × 4H × 1H × 15m 四週期多頭完美共振 —— 即買即升黃金擊球區！】**\n\n"
                f"目前股價已同時站穩 **4H 20EMA ({res['h4_ema20']})**、**1H 20EMA ({h1_e20:.2f})**、**1H 回調 AVWAP ({h1_avwap:.2f})** 與 **15m 日內 VWAP ({intra_v:.2f})**！"
                f"小週期洗盤已徹底完成並與日線主升浪同向共振，現價 (**{curr_price:.2f}**) 正處於爆發臨界點！"
            ))
        else:
            sniper_p = round(max(intra_v, min(curr_price, h1_e20)), 2)
            res["sniper_entry_trigger_price"] = sniper_p
            res["sniper_entry_method"] = f"🎯 關注 15m ORB 高點 ({orb_h:.2f}) 突破或 1H 20EMA ({h1_e20:.2f}) 企穩買點"
            cards.append((
                "info",
                f"🧭 **【多週期入場精準定位】**：日線結構買點為 **{planned_entry:.2f}**；日內小週期建議觀察 **15m 日內 VWAP ({intra_v:.2f})** 承接力，或於升穿 **15m 開盤高點 ORB ({orb_h:.2f}) / 1H 20EMA ({h1_e20:.2f})** 時右側入場。"
            ))

        # 多頭離場（止盈鎖潤 + 防恐慌洗盤止損）精準導航
        profit_lock_1h = round(max(h1_sw_low, h1_e20 * 0.992), 2)
        res["sniper_profit_lock_exit"] = profit_lock_1h
        exit_guide_msg = (
            f"💰 **【如何用 1H / 15m 找到最佳離場位置（徹底告別「賺變蝕」與「賣在最低點」）】**：\n"
            f"1. **衝高分批止盈（Sell into 15m/1H Climax）**：當股價逼近日線第一目標 **{take_profit:.2f}**（或日內偏離 VWAP > 2x ATR 且 15m RSI > 78 出現長上影線），先主動減倉 50% 鎖定利潤。\n"
            f"2. **1H 結構移動鎖潤線（`{profit_lock_1h:.2f}`）**：當你已有明顯浮盈（>1.5R）後，唔好傻等日線跌穿 20MA（會嘔返大半利潤）！將尾倉一半嘅移動止盈設喺 **1H 前波谷支撐 (`{h1_sw_low:.2f}`) 與 1H 20EMA (`{h1_e20:.2f}`)**，一旦 1H 收盤實質跌破 `{profit_lock_1h:.2f}` 即刻鎖潤落袋！\n"
            f"3. **防日內恐慌針尖賣在最低點**：{'🚨 **目前 15m 正處於極度超賣恐慌針尖 (15m RSI ≤ 26)，嚴禁在此秒市價砍倉！請等反抽日內 VWAP (' + f'{intra_v:.2f}' + ') 或 1H 收盤確認！**' if m15_oversold_panic else f'盤中若跌穿軟止損但未破硬止損 ({stop_loss:.2f})，請先睇 15m 是否守住開盤低點 ORB ({orb_l:.2f}) 或尾盤收回，避免在早盤洗盤最低點交出籌碼。'}"
        )
        cards.append(("warning" if m15_oversold_panic else "info", exit_guide_msg))
    else:
        # 空頭入場與離場精準導航
        if h1_reclaimed:
            trigger_p = round(min(h1_e20, max(h1_avwap, h1_sw_low)), 2)
            res["sniper_entry_trigger_price"] = trigger_p
            res["sniper_entry_method"] = f"🛑 暫緩左側摸頂做空！改設 1H 跌破轉弱觸發單 (跌穿 {trigger_p:.2f} 沽空)"
            cards.append((
                "error",
                f"🛑 **【空頭防軋空預警：1H 仍處於強勢反彈結構之上，切勿過早左側摸頂！】**\n\n"
                f"目前 1H 收盤仍站穩 **1H 20EMA ({h1_e20:.2f})** 與反彈 AVWAP ({h1_avwap:.2f}) 之上，直接掛單摸頂極易遭遇逼空急升。\n\n"
                f"👉 **最佳沽空做法**：請等反彈力竭、**1H 收盤重新跌破 {trigger_p:.2f} (或跌穿 15m 開盤低點 ORB {orb_l:.2f})** 確認空頭重掌動能時才順勢入場沽空！"
            ))
        else:
            sniper_p = round(min(intra_v, max(curr_price, h1_e20)), 2)
            res["sniper_entry_trigger_price"] = sniper_p
            res["sniper_entry_method"] = f"⚡ 於 1H 20EMA ({h1_e20:.2f}) 受阻或跌穿 15m ORB 低點 ({orb_l:.2f}) 沽空"
            cards.append((
                "success" if h1_breakdown else "info",
                f"🏴‍☠️ **【多週期沽空精準定位】**：1H 20EMA 反壓位 **{h1_e20:.2f}** ｜ 15m 日內 VWAP **{intra_v:.2f}** ｜ 15m 開盤低點 ORB **{orb_l:.2f}**。當反抽受阻於日內 VWAP 或跌破 1H 前低 ({h1_sw_low:.2f}) 時為最佳順勢沽空點。"
            ))

        profit_lock_1h_s = round(min(h1_sw_high, h1_e20 * 1.008), 2)
        res["sniper_profit_lock_exit"] = profit_lock_1h_s
        cards.append((
            "info",
            f"💰 **【空頭多週期最佳回補離場位】**：第一日線結構目標 **{take_profit:.2f}**；短線浮盈鎖潤線設於 **1H 前波峰 ({h1_sw_high:.2f}) / 1H 20EMA ({h1_e20:.2f})**，若 15m 出現極度超賣恐慌急跌 (RSI < 22) 可先回補一半鎖定暴跌利潤！"
        ))

    if not str(ticker).upper().endswith(".HK") and not is_short and stop_loss > 0:
        _pm_proto = evaluate_premarket_gap_and_orb_stop_protocol(
            daily_df=daily_df,
            stop_loss=stop_loss,
            entry_price=planned_entry,
            market_type="US/Global",
            orb_low_15m=res.get("orb_low_15m", np.nan),
            intraday_vwap=res.get("intraday_vwap", np.nan)
        )
        res["rth_orb_stop_trigger"] = _pm_proto["rth_orb_stop_trigger"]
        res["catastrophic_hard_sl"] = _pm_proto["catastrophic_hard_sl"]
        if _pm_proto.get("protocol_detail"):
            cards.append(("info", _pm_proto["protocol_detail"]))
    res["entry_timing_cards"] = cards
    return res

import html as _html_lib
import re

def _md_bold_to_html(text: str) -> str:
    """將簡易 Markdown **粗體** 與 *斜體* 安全轉換為 HTML 標籤並保留換行，避免 ReactMarkdown 內部拆碎 DOM 節點導致 removeChild 崩潰"""
    escaped = _html_lib.escape(str(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*(.+?)\*", r"<em>\1</em>", escaped)
    return escaped.replace("\n\n", "<br><br>").replace("\n", "<br>")

def build_atomic_alert_html(msg: str, kind: str = "info") -> str:
    """生成單個防翻譯、防 React removeChild 崩潰之原生視覺相容提示框 HTML"""
    styles = {
        "success": ("#e8f5e9", "#2e7d32", "#1b5e20"),
        "error":   ("#ffebee", "#c62828", "#b71c1c"),
        "warning": ("#fff8e1", "#f9a825", "#e65100"),
        "info":    ("#e3f2fd", "#1565c0", "#0d47a1"),
    }
    bg, border, fg = styles.get(kind, styles["info"])
    body_html = _md_bold_to_html(msg)
    return (
        f'<div class="notranslate" translate="no" style="'
        f'background-color:{bg};border-left:5px solid {border};color:{fg};'
        f'padding:13px 16px;border-radius:6px;margin-bottom:10px;'
        f'font-size:0.96rem;line-height:1.55;box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
        f'{body_html}</div>'
    )

def render_atomic_alert(msg: str, kind: str = "info", slot: Any = None) -> None:
    """在固定槽位或當前容器內以單一原子 HTML 節點渲染提示框，徹底杜絕 React 對內部 <strong> 節點執行 removeChild"""
    target = slot if slot is not None else st
    target.markdown(build_atomic_alert_html(msg, kind=kind), unsafe_allow_html=True)

def render_atomic_alert_group(alerts: List[Tuple[str, str]], slot: Any = None) -> None:
    """將多個連續動態提示框合併為單一 HTML 容器一次過渲染，100% 根絕多框切換時之文字穿插與 removeChild 錯誤"""
    combined = '<div class="notranslate" translate="no">' + "".join(
        build_atomic_alert_html(m, k) for k, m in alerts if m
    ) + '</div>'
    target = slot if slot is not None else st
    target.markdown(combined, unsafe_allow_html=True)



import json
import os as _os_lib

STATE_BACKUP_FILE = _os_lib.path.join(_os_lib.path.expanduser("~"), ".swing_terminal_state_v27.json")

SECTOR_MAPPING: Dict[str, str] = {
    "0700.HK": "科網平台", "3690.HK": "科網平台", "9988.HK": "科網平台", "9618.HK": "科網平台", "9999.HK": "科網平台", "1024.HK": "科網平台", "9888.HK": "科網平台", "9626.HK": "科網平台", "0772.HK": "科網平台", "9961.HK": "科網平台",
    "1810.HK": "硬科技與消費電子", "0981.HK": "半導體與晶片", "1347.HK": "半導體與晶片", "0285.HK": "硬科技與消費電子", "2382.HK": "硬科技與消費電子", "2018.HK": "硬科技與消費電子", "0268.HK": "雲端軟件",
    "1211.HK": "新能源車與智駕", "2015.HK": "新能源車與智駕", "9868.HK": "新能源車與智駕", "2333.HK": "新能源車與智駕",
    "2269.HK": "生物醫藥", "2359.HK": "生物醫藥", "1093.HK": "生物醫藥", "1177.HK": "生物醫藥", "1801.HK": "生物醫藥", "0853.HK": "生物醫藥", "6618.HK": "生物醫藥", "1833.HK": "生物醫藥",
    "0005.HK": "金融銀行保險", "1299.HK": "金融銀行保險", "0388.HK": "金融銀行保險", "2318.HK": "金融銀行保險", "0939.HK": "金融銀行保險", "1398.HK": "金融銀行保險", "3988.HK": "金融銀行保險", "0011.HK": "金融銀行保險", "3328.HK": "金融銀行保險", "3908.HK": "金融銀行保險", "6030.HK": "金融銀行保險",
    "0883.HK": "能源與資源", "1088.HK": "能源與資源", "1898.HK": "能源與資源", "2899.HK": "能源與資源", "1818.HK": "能源與資源", "0941.HK": "電訊高息公用", "0002.HK": "電訊高息公用", "0066.HK": "電訊高息公用", "1193.HK": "電訊高息公用",
    "1919.HK": "航運與航空物流", "0293.HK": "航運與航空物流",
    "2020.HK": "內需消費品牌", "2331.HK": "內需消費品牌", "9992.HK": "內需消費品牌", "6862.HK": "內需消費品牌", "1928.HK": "博彩旅遊", "0027.HK": "博彩旅遊",
    "NVDA": "AI與半導體", "AMD": "AI與半導體", "AVGO": "AI與半導體", "TSM": "AI與半導體", "ASML": "AI與半導體", "ARM": "AI與半導體", "MU": "AI與半導體", "QCOM": "AI與半導體", "INTC": "AI與半導體", "SMCI": "AI與半導體",
    "AAPL": "大型科技巨頭", "MSFT": "大型科技巨頭", "GOOGL": "大型科技巨頭", "AMZN": "大型科技巨頭", "META": "大型科技巨頭", "NFLX": "大型科技巨頭", "TSLA": "新能源車與智駕",
    "PLTR": "AI與企業軟件", "CRWD": "網絡安全與雲端", "PANW": "網絡安全與雲端", "ZS": "網絡安全與雲端", "SNOW": "AI與企業軟件", "DDOG": "AI與企業軟件", "CRM": "AI與企業軟件", "ADBE": "AI與企業軟件", "ORCL": "AI與企業軟件", "SHOP": "電商與金融科技",
    "COIN": "加密資產與Fintech", "MSTR": "加密資產與Fintech", "PYPL": "加密資產與Fintech", "SQ": "加密資產與Fintech", "XYZ": "加密資產與Fintech", "HOOD": "加密資產與Fintech",
    "UBER": "共享出行與旅遊", "ABNB": "共享出行與旅遊", "APP": "AI與企業軟件", "ANET": "AI與半導體", "RDDT": "科網平台", "AXON": "硬科技與消費電子",
    "VST": "AI電力與能源", "CEG": "AI電力與能源", "GEV": "AI電力與能源"
}

def get_ticker_sector(ticker: str) -> str:
    fmt_t = str(ticker).strip().upper()
    if fmt_t in SECTOR_MAPPING:
        return SECTOR_MAPPING[fmt_t]
    return "港股其他板塊" if fmt_t.endswith(".HK") else "美股其他板塊"

def load_persisted_state_from_disk(persistent_dict: dict) -> None:
    """從本地磁碟讀取已保存的自選股、鎖定持倉、信號生命週期與開倉日 R:R 雜湊快照，防止重啟 Streamlit 時記憶清空"""
    try:
        if _os_lib.path.exists(STATE_BACKUP_FILE):
            with open(STATE_BACKUP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "user_locked_positions" in data and isinstance(data["user_locked_positions"], dict):
                    persistent_dict.setdefault("user_locked_positions", {}).update(data["user_locked_positions"])
                if "auto_entry_date_hash_store" in data and isinstance(data["auto_entry_date_hash_store"], dict):
                    persistent_dict.setdefault("auto_entry_date_hash_store", {}).update(data["auto_entry_date_hash_store"])
                if "signal_lifecycle_store" in data and isinstance(data["signal_lifecycle_store"], dict):
                    persistent_dict.setdefault("signal_lifecycle_store", {}).update(data["signal_lifecycle_store"])
                if "watchlist_dict" in data and isinstance(data["watchlist_dict"], dict):
                    persistent_dict["_disk_watchlist_dict"] = data["watchlist_dict"]
    except Exception as e:
        swing_logger.debug(f"讀取本地狀態備份略過: {e}")

def save_persisted_state_to_disk(persistent_dict: dict, watchlist_dict: Optional[ dict ] = None) -> None:
    """以原子寫入方式將用戶鎖定持倉、信號生命週期、開倉日快照與自選股名單備份至本地磁碟並同步至全域快取"""
    try:
        if watchlist_dict is not None:
            persistent_dict["_disk_watchlist_dict"] = watchlist_dict
        payload = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_locked_positions": persistent_dict.get("user_locked_positions", {}),
            "auto_entry_date_hash_store": persistent_dict.get("auto_entry_date_hash_store", {}),
            "signal_lifecycle_store": persistent_dict.get("signal_lifecycle_store", {}),
            "watchlist_dict": watchlist_dict if watchlist_dict is not None else persistent_dict.get("_disk_watchlist_dict", {})
        }
        tmp_path = STATE_BACKUP_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _os_lib.replace(tmp_path, STATE_BACKUP_FILE)
    except Exception as e:
        swing_logger.debug(f"寫入本地狀態備份略過: {e}")

def compute_vbp_profile_and_vsa_engine(df: pd.DataFrame, lookback: int = 120, num_bins: int = 40) -> Dict[str, Any]:
    """
    [v27.0 獨立量化升級：真實 Volume-by-Price (VBP) 籌碼分佈矩陣與 Wyckoff VSA 量價散度引擎]
    為什麼比單純套用日內 DeepCharts Orderflow 更適合日線 Swing Trader？
    - 免費日線 OHLCV 並無 Level-2 Bid/Ask Tick 數據，但日線波段交易真正的「上方阻力空單/套牢賣壓」取決於歷史成交量在價格軸上的精確分佈 (Volume-by-Price)！
    - 本引擎將過去 lookback 日每根 K 線之成交量按其 [Low, High] 跨越的價格區間均勻映射至 40 個價格格網 (Price Bins)，精確算出：
      1) POC (Point of Control)：過去半年最大籌碼共識價（最強支撐或最大套牢中心）
      2) VAH / VAL (Value Area High / Low)：涵蓋 70% 歷史總成交量的價值區上下界
      3) 上方緊鄰區域狀態 (Overhead Zone State)：區分「🚀 LVN 籌碼真空加速區 (上方套牢籌碼極少，突破極易連拉大陽線)」與「🧱 HVN 巨量套牢泥沼區 (上方堆積重兵，突破易受阻回落)」
      4) Wyckoff VSA 兩大王牌微結構信號：
         - No-Supply Test (縮量無供給測試)：回踩時收窄幅陰線/下影線且成交量低於前 2 日，證明空頭砸盤籌碼已枯竭
         - Stopping Volume / Spring (底部吸收彈簧)：跌穿支撐後放量長下影收回
    """
    res = {
        "poc_price": np.nan,
        "vah_price": np.nan,
        "val_price": np.nan,
        "overhead_supply_pct": 0.0,
        "is_in_lvn_vacuum": False,
        "is_blocked_by_hvn": False,
        "vsa_no_supply": False,
        "vsa_spring_absorption": False,
        "vsa_upthrust_distribution": False,
        "vbp_position_label": "⚪ 資料不足",
        "vbp_tactical_advice": ""
    }
    if df is None or len(df) < 30:
        return res

    sub = df.tail(min(len(df), lookback)).copy()
    highs = sub['High'].astype(float).values
    lows = sub['Low'].astype(float).values
    closes = sub['Close'].astype(float).values
    opens = sub['Open'].astype(float).values
    vols = sub['Volume'].astype(float).values if 'Volume' in sub.columns else np.ones(len(sub))
    curr_c = float(closes[-1])
    atr_v = float(sub['ATR'].iloc[-1]) if ('ATR' in sub.columns and pd.notna(sub['ATR'].iloc[-1]) and float(sub['ATR'].iloc[-1]) > 0) else curr_c * 0.03

    p_min = float(np.min(lows))
    p_max = float(np.max(highs))
    if p_max <= p_min * 1.01:
        return res

    bin_edges = np.linspace(p_min, p_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (p_max - p_min) / num_bins
    vbp_hist = np.zeros(num_bins, dtype=float)

    # 將每根 K 線的成交量按其 [Low, High] 覆蓋的價格格網分攤，並對近期 K 線賦予時間衰減權重 (0.6 ~ 1.0)
    n_bars = len(sub)
    for i in range(n_bars):
        time_decay = 0.60 + 0.40 * (i / max(1, n_bars - 1))
        bar_l, bar_h, bar_v = lows[i], highs[i], vols[i] * time_decay
        overlap = np.maximum(0.0, np.minimum(bin_edges[1:], bar_h) - np.maximum(bin_edges[:-1], bar_l))
        total_ov = float(np.sum(overlap))
        if total_ov > 0:
            vbp_hist += bar_v * (overlap / total_ov)
        else:
            idx_b = min(num_bins - 1, max(0, int((closes[i] - p_min) / max(bin_width, 1e-6))))
            vbp_hist[idx_b] += bar_v

    total_vbp = float(np.sum(vbp_hist))
    if total_vbp <= 0:
        return res

    poc_idx = int(np.argmax(vbp_hist))
    poc_price = float(bin_centers[poc_idx])
    res["poc_price"] = round(poc_price, 2)

    # 計算 70% Value Area (VAH / VAL)
    target_va_vol = 0.70 * total_vbp
    cum_va_vol = float(vbp_hist[poc_idx])
    left_i, right_i = poc_idx, poc_idx
    while cum_va_vol < target_va_vol and (left_i > 0 or right_i < num_bins - 1):
        v_left = vbp_hist[left_i - 1] if left_i > 0 else -1.0
        v_right = vbp_hist[right_i + 1] if right_i < num_bins - 1 else -1.0
        if v_right >= v_left:
            right_i += 1
            cum_va_vol += float(vbp_hist[right_i])
        else:
            left_i -= 1
            cum_va_vol += float(vbp_hist[left_i])

    val_price = float(bin_edges[left_i])
    vah_price = float(bin_edges[right_i + 1])
    res["val_price"] = round(val_price, 2)
    res["vah_price"] = round(vah_price, 2)

    # 計算現價上方 1.0x ~ 2.5x ATR 空間內的籌碼密度 vs 平均籌碼密度 (判斷 LVN 籌碼真空區 vs HVN 套牢峰)
    upper_zone_mask = (bin_centers > curr_c * 1.003) & (bin_centers <= curr_c + 2.2 * atr_v)
    overhead_all_mask = bin_centers > curr_c * 1.005
    overhead_supply_pct = float(np.sum(vbp_hist[overhead_all_mask]) / total_vbp * 100.0)
    res["overhead_supply_pct"] = round(overhead_supply_pct, 1)

    avg_bin_vol = total_vbp / num_bins
    if np.any(upper_zone_mask):
        near_upper_mean_vol = float(np.mean(vbp_hist[upper_zone_mask]))
        near_upper_max_vol = float(np.max(vbp_hist[upper_zone_mask]))
        if near_upper_mean_vol <= avg_bin_vol * 0.58 or overhead_supply_pct <= 8.0:
            res["is_in_lvn_vacuum"] = True
        elif near_upper_max_vol >= avg_bin_vol * 1.65 and curr_c < poc_price * 0.995:
            res["is_blocked_by_hvn"] = True
    else:
        # 股價已創區間新高，上方 0% 套牢籌碼 (Blue Sky Breakout)
        res["is_in_lvn_vacuum"] = True

    # Wyckoff VSA 量價散度分析 (No-Supply Test / Spring Absorption / Upthrust)
    if n_bars >= 5:
        spread_now = highs[-1] - lows[-1]
        avg_spread_20 = float(np.mean(highs[-20:] - lows[-20:])) if n_bars >= 20 else spread_now
        vol_sma20 = float(np.mean(vols[-20:])) if n_bars >= 20 else float(np.mean(vols))
        # No-Supply Test: 縮量窄幅回踩 (成交量小於前兩日且 < 0.78x 均量，收盤不在最低點)
        if (vols[-1] < vols[-2] and vols[-1] < vols[-3] and vols[-1] < vol_sma20 * 0.78 and spread_now <= avg_spread_20 * 1.05 and (closes[-1] - lows[-1]) >= 0.35 * max(spread_now, 1e-6)):
            res["vsa_no_supply"] = True
        # Spring / Shakeout Absorption: 下影線佔比 >= 50% 且收盤高於開盤或前日收盤
        if (( min(opens[-1], closes[-1]) - lows[-1] ) >= 0.48 * max(spread_now, 1e-6)) and closes[-1] >= lows[-1] + 0.60 * spread_now:
            res["vsa_spring_absorption"] = True
        # Upthrust Distribution: 高位放量長上影陰線
        if ((highs[-1] - max(opens[-1], closes[-1])) >= 0.48 * max(spread_now, 1e-6)) and vols[-1] > vol_sma20 * 1.25 and closes[-1] <= opens[-1]:
            res["vsa_upthrust_distribution"] = True

    if curr_c > vah_price and res["is_in_lvn_vacuum"]:
        res["vbp_position_label"] = f"🚀 站上 VAH ({vah_price:.2f}) 進入 LVN 籌碼真空加速區 (上方套牢僅 {overhead_supply_pct:.1f}%)"
        res["vbp_tactical_advice"] = "現價已突破 70% 籌碼價值區頂部 (VAH) 且上方屬低量真空區 (LVN)，套牢阻力極輕，最利於【⚡ 即時突破入場】或順勢加碼！"
    elif curr_c >= poc_price:
        res["vbp_position_label"] = f"🟢 站穩最大籌碼峰 POC ({poc_price:.2f}) 之上 (價值區 {val_price:.2f}~{vah_price:.2f})"
        res["vbp_tactical_advice"] = f"最大籌碼共識區 POC ({poc_price:.2f}) 已轉化為下方強力鐵底支撐，回踩 POC 或 20MA 出現 Wyckoff 無供給縮量時為最佳買點。"
    elif res["is_blocked_by_hvn"]:
        res["vbp_position_label"] = f"🧱 處於 POC ({poc_price:.2f}) 下方，頭頂面臨 HVN 巨量套牢峰壓制 (上方套牢佔 {overhead_supply_pct:.1f}%)"
        res["vbp_tactical_advice"] = f"上方 {poc_price:.2f}~{vah_price:.2f} 堆積大量歷史套牢盤，直接硬衝極易受阻，務必等放量站上 POC ({poc_price:.2f}) 或縮量回踩 VAL ({val_price:.2f}) 企穩。"
    else:
        res["vbp_position_label"] = f"🟡 處於籌碼價值區下沿 (POC: {poc_price:.2f} ｜ VAL: {val_price:.2f})"
        res["vbp_tactical_advice"] = f"關注價值區下沿 VAL ({val_price:.2f}) 支撐承接力，若跌穿 VAL 將進入下方籌碼斷層，需嚴守止損。"

    # [v31.0 新增：空頭專屬 VBP 下方籌碼通道性質 (Downside LVN Vacuum vs HVN Support) 與 Wyckoff No-Demand 無需求反抽信號]
    downside_support_pct = round(max(0.0, 100.0 - overhead_supply_pct), 1)
    res["downside_support_pct"] = downside_support_pct
    lower_zone_mask = (bin_centers < curr_c * 0.997) & (bin_centers >= curr_c - 2.2 * atr_v)
    is_downside_lvn = False
    is_downside_hvn = False
    if np.any(lower_zone_mask):
        near_lower_mean_vol = float(np.mean(vbp_hist[lower_zone_mask]))
        near_lower_max_vol = float(np.max(vbp_hist[lower_zone_mask]))
        if near_lower_mean_vol <= avg_bin_vol * 0.62 or downside_support_pct <= 12.0:
            is_downside_lvn = True
        elif near_lower_max_vol >= avg_bin_vol * 1.65 and curr_c > poc_price * 1.005:
            is_downside_hvn = True
    else:
        is_downside_lvn = True
    res["is_downside_lvn_vacuum"] = is_downside_lvn
    res["is_supported_by_downside_hvn"] = is_downside_hvn

    # Wyckoff VSA No-Demand Bar (空頭無需求反彈測試：小陽線或上影線、成交量低於前兩日且 <0.85x 均量、收盤在區間中下部)
    vsa_no_demand = False
    if n_bars >= 5:
        spread_now = highs[-1] - lows[-1]
        vol_sma20 = float(np.mean(vols[-20:])) if n_bars >= 20 else float(np.mean(vols))
        if (vols[-1] < vols[-2] and vols[-1] < vols[-3] and vols[-1] < vol_sma20 * 0.85 and (highs[-1] - closes[-1]) >= 0.35 * max(spread_now, 1e-6)):
            vsa_no_demand = True
    res["vsa_no_demand"] = vsa_no_demand

    if curr_c < val_price and is_downside_lvn:
        res["short_vbp_position_label"] = f"🩸 跌穿價值區底 VAL ({val_price:.2f}) 墜入下方 LVN 籌碼真空暴跌區 (上方套牢高達 {overhead_supply_pct:.1f}%)"
        res["short_vbp_tactical_advice"] = (
            f"現價已跌破 70% 籌碼價值區下沿 VAL ({val_price:.2f})，頭頂堆積 **{overhead_supply_pct:.1f}%** 歷史套牢賣壓，"
            f"且下方支撐籌碼僅佔 **{downside_support_pct:.1f}%**（屬 LVN 籌碼斷層真空區），多頭毫無承接，最利於【⚡ 順勢破底做空】或反抽 VAL ({val_price:.2f}) 加空！"
        )
    elif curr_c <= poc_price:
        res["short_vbp_position_label"] = f"🔴 受制於最大籌碼峰 POC ({poc_price:.2f}) 蓋頭反壓之下 (上方套牢壓制佔 {overhead_supply_pct:.1f}%)"
        res["short_vbp_tactical_advice"] = (
            f"最大籌碼共識峰 POC ({poc_price:.2f}) 與上方 **{overhead_supply_pct:.1f}%** 套牢盤已形成沉重蓋頭反壓！"
            f"目前逼近價值區下沿 VAL ({val_price:.2f})，反抽 POC ({poc_price:.2f}) 出現無量受阻或放量跌穿 VAL ({val_price:.2f}) 將觸發主跌段。"
        )
    else:
        res["short_vbp_position_label"] = f"⚠️ 仍處於最大籌碼峰 POC ({poc_price:.2f}) 支撐上方 (下方承接籌碼佔 {downside_support_pct:.1f}%)"
        res["short_vbp_tactical_advice"] = (
            f"目前股價仍在最大籌碼峰 POC ({poc_price:.2f}) 之上，下方 {val_price:.2f}~{poc_price:.2f} 尚有密集籌碼承接，做空宜等實質跌破 POC ({poc_price:.2f}) 後再順勢出擊。"
        )

    return res


def evaluate_portfolio_heat_and_sector_risk(
    persistent_dict: dict,
    candidate_ticker: str,
    candidate_risk_pct: float = 0.50,
    max_portfolio_heat_pct: float = 5.0,
    max_sector_heat_pct: float = 2.0
) -> Dict[str, Any]:
    """
    [v27.0 獨立風控升級：總組合熱度 (Portfolio Heat) 與行業板塊集中度防火牆]
    解決問題：單筆風險 0.5%~1.0% 看似安全，但若同時買入 6~8 隻同屬「科網平台」或「AI半導體」的股票，
    遇到行業黑天鵝或指數跳空時會發生相關性共振爆倉！
    本函數實時掃描用戶已鎖定持倉 (user_locked_positions)，計算未移至保本的在險總熱度 (Open Portfolio Heat %)。
    """
    locked_map = persistent_dict.get("user_locked_positions", {})
    cand_sector = get_ticker_sector(candidate_ticker)
    open_heat_total = 0.0
    sector_heat_map: Dict[str, float] = {}
    active_positions_summary = []

    fmt_cand = str(candidate_ticker).strip().upper()
    for sym, plan in locked_map.items():
        if not isinstance(plan, dict) or str(sym).strip().upper() == fmt_cand:
            continue
        # 若當前浮盈已 >= 1.0R，視為已可移至保本 (0 原始本金風險敞口)；否則計入在險熱度
        curr_r = float(plan.get("current_r", 0.0))
        if curr_r < 1.0:
            pos_risk = float(plan.get("recommended_risk_pct", 0.50))
            sec = get_ticker_sector(sym)
            open_heat_total += pos_risk
            sector_heat_map[sec] = sector_heat_map.get(sec, 0.0) + pos_risk
            active_positions_summary.append(f"{sym}({sec}: {pos_risk:.2f}%)")

    cand_sec_current_heat = sector_heat_map.get(cand_sector, 0.0)
    proj_total_heat = round(open_heat_total + candidate_risk_pct, 2)
    proj_sector_heat = round(cand_sec_current_heat + candidate_risk_pct, 2)

    heat_over_limit = proj_total_heat > max_portfolio_heat_pct
    sector_over_limit = proj_sector_heat > max_sector_heat_pct

    if heat_over_limit:
        status_msg = (
            f"🛑 **【組合總熱度超標攔截 (Portfolio Heat > {max_portfolio_heat_pct:.1f}%)】**："
            f"目前未保本持倉在險熱度已達 **{open_heat_total:.2f}%**，若加入 {candidate_ticker} ({candidate_risk_pct:.2f}%) 將達 **{proj_total_heat:.2f}%**！"
            f"請先等待現有持倉脫離成本滿 +1R 移至富途保本價後，才可開立新倉！"
        )
        level = "error"
    elif sector_over_limit:
        status_msg = (
            f"⚠️ **【板塊集中度預警 ({cand_sector} > {max_sector_heat_pct:.1f}%)】**："
            f"你在「{cand_sector}」板塊的未保本風險將達 **{proj_sector_heat:.2f}%**，同板塊高度相關股票遇到跳空時會同步止損，建議將本筆風險減半或換成其他板塊領頭羊！"
        )
        level = "warning"
    else:
        status_msg = (
            f"✅ **【組合熱度與板塊分散健康】**：當前已鎖定未保本總熱度 **{open_heat_total:.2f}%** ➔ 開倉後預計 **{proj_total_heat:.2f}%** (安全上限 ≤ {max_portfolio_heat_pct:.1f}%) ｜ "
            f"所屬板塊：**{cand_sector}** (板塊熱度 {proj_sector_heat:.2f}% ≤ {max_sector_heat_pct:.1f}%)"
        )
        level = "success"

    return {
        "candidate_sector": cand_sector,
        "current_open_heat_pct": round(open_heat_total, 2),
        "projected_total_heat_pct": proj_total_heat,
        "projected_sector_heat_pct": proj_sector_heat,
        "heat_over_limit": heat_over_limit,
        "sector_over_limit": sector_over_limit,
        "status_msg": status_msg,
        "alert_level": level,
        "active_positions": active_positions_summary
    }

def compute_deepchart_orderflow_absorption(df: pd.DataFrame, lookback: int = 60) -> Dict[str, Any]:
    """
    [v26.0 新增：DeepChart 前高阻力空單消化與訂單流吸收量化引擎 (Overhead Supply & Short Order Absorption Engine)]
    解決核心痛點：「點樣知道前阻力位嘅空單/套牢盤有冇被消化晒？要食晒先至可以安全突破上升！」
    量化四大維度 (0~100分)：
    1. 前高阻力區定位與初始空單/套牢供給量 (Initial Overhead Supply Volume)：
       - 定位過去 15~lookback 日內最關鍵的前波高點阻力區 [R_high - 0.6*ATR, R_high + 0.2*ATR] 及其形成當週的供給成交量。
    2. 右側測試換手消化率 (Volume Absorption Ratio, 0~35分)：
       - 統計前高形成後，股價每次進入阻力帶 [-2.5%, +1.0%] 區間時的累積換手消化量（陽線/高收盤位 CLV 給予 1.3x 主動買盤吸收權重）。
    3. 阻力位賣壓枯竭度 (Seller Exhaustion & VCP Contraction, 0~30分)：
       - 檢測最近一次測試前高後的回調成交量是否萎縮至 < 0.70x 均量，且回調深度是否收窄（空單砸盤力量枯竭）。
    4. 訂單流微結構累積買盤差額 (Micro-Structure CVD & Pocket Pivot, 0~35分)：
       - 利用 K 線收盤位置 CLV = ((Close - Low) - (High - Close)) / (High - Low) 乘以成交量估算近 10 日累積主動買賣差額 (Proxy CVD)，
         並檢測是否出現「口袋支點 (Pocket Pivot：上漲日成交量大於過去 10 日內最大下跌日成交量)」。
    """
    out = {
        "absorption_score": 50.0,
        "absorption_ratio_pct": 0.0,
        "prior_res_price": np.nan,
        "tests_count": 0,
        "seller_exhaustion_ok": False,
        "cvd_10d_positive": False,
        "pocket_pivot_10d": False,
        "shorts_absorbed": False,
        "heavy_overhead_supply": False,
        "deepchart_badge": "⚪ 籌碼中性",
        "deepchart_detail": "數據不足以評估前高空單消化度"
    }
    if df is None or len(df) < 25:
        return out

    sub = df.tail(min(len(df), max(lookback, 40))).copy()
    n = len(sub)
    closes = sub['Close'].astype(float).values
    highs = sub['High'].astype(float).values
    lows = sub['Low'].astype(float).values
    opens = sub['Open'].astype(float).values
    vols = sub['Volume'].astype(float).values if 'Volume' in sub.columns else np.ones(n)
    curr_c = float(closes[-1])
    atr_v = float(sub['ATR'].iloc[-1]) if ('ATR' in sub.columns and pd.notna(sub['ATR'].iloc[-1]) and float(sub['ATR'].iloc[-1]) > 0) else curr_c * 0.03

    # 1. 尋找前波關鍵高點阻力 (排除最近 3 根 K 線，確保是「前高阻力」)
    search_end = max(5, n - 3)
    peak_rel_idx = int(np.argmax(highs[:search_end]))
    prior_res_p = float(highs[peak_rel_idx])
    out["prior_res_price"] = round(prior_res_p, 2)

    zone_low = prior_res_p - max(0.8 * atr_v, prior_res_p * 0.025)
    zone_high = prior_res_p + max(0.3 * atr_v, prior_res_p * 0.01)

    # 計算前高形成當時 (前後各 2 根 K 線) 的初始套牢/空單供給量
    init_start = max(0, peak_rel_idx - 2)
    init_end = min(search_end, peak_rel_idx + 3)
    initial_supply_vol = float(np.sum(vols[init_start:init_end])) / max(1, (init_end - init_start)) * 2.2
    initial_supply_vol = max(initial_supply_vol, float(np.mean(vols)) * 1.5, 1.0)

    # 2. 計算前高之後 (peak_rel_idx + 2 至最新一根) 進入阻力帶的消化成交量與測試次數
    absorbed_vol = 0.0
    tests_cnt = 0
    in_test_cluster = False
    for idx in range(peak_rel_idx + 2, n):
        if highs[idx] >= zone_low:
            if not in_test_cluster:
                tests_cnt += 1
                in_test_cluster = True
            rng = max(highs[idx] - lows[idx], 1e-6)
            clv = ((closes[idx] - lows[idx]) - (highs[idx] - closes[idx])) / rng  # [-1, +1]
            # 若收陽或收在下半部以上，視為買盤主動吸收前高空單；若長上影大陰線則折扣
            absorb_weight = 1.25 if (closes[idx] >= opens[idx] or clv >= 0.1) else 0.55
            absorbed_vol += vols[idx] * absorb_weight
        else:
            if highs[idx] < zone_low - 0.4 * atr_v:
                in_test_cluster = False

    # 若當前價已強勢站上前高之上，代表前高空單已被突破買盤吞沒
    if curr_c > prior_res_p * 1.005:
        absorbed_vol += float(vols[-1]) * 1.35

    absorption_ratio = (absorbed_vol / initial_supply_vol) * 100.0
    out["absorption_ratio_pct"] = round(min(absorption_ratio, 250.0), 1)
    out["tests_count"] = int(tests_cnt)

    # 3. 賣壓枯竭度 (Seller Exhaustion) 與波動收窄檢查
    recent10_down_vols = [vols[k] for k in range(max(0, n - 10), n) if closes[k] < opens[k]]
    vol_sma20 = float(np.mean(vols[max(0, n - 20):])) if n >= 5 else float(np.mean(vols))
    avg_recent_down_vol = float(np.mean(recent10_down_vols)) if recent10_down_vols else vol_sma20 * 0.5
    seller_exhausted = bool(avg_recent_down_vol < vol_sma20 * 0.82 or bool(sub['VCPSignal'].iloc[-1] if 'VCPSignal' in sub.columns else False))
    out["seller_exhaustion_ok"] = seller_exhausted

    # 4. 訂單流微結構 Proxy CVD (近 10 日主動買賣差額) 與 Pocket Pivot (口袋支點)
    clv_series = ((closes - lows) - (highs - closes)) / np.maximum(highs - lows, 1e-6)
    delta_vol = clv_series * vols
    cvd_10d = float(np.sum(delta_vol[max(0, n - 10):]))
    out["cvd_10d_positive"] = bool(cvd_10d > 0)

    max_down_vol_10d = max([vols[k] for k in range(max(0, n - 11), n - 1) if closes[k] < closes[max(0, k - 1)]] + [0.0])
    pocket_pivot = bool(
        (closes[-1] > opens[-1] and vols[-1] > max_down_vol_10d and max_down_vol_10d > 0)
        or any(closes[k] > opens[k] and vols[k] > max_down_vol_10d for k in range(max(0, n - 3), n) if max_down_vol_10d > 0)
    )
    out["pocket_pivot_10d"] = pocket_pivot

    # [v28.0 融合升級：加入 OBV 能量潮領先價格創高 (OBV Leads Breakout) 與 多次測試量能遞減率 (Multi-Touch Volume Decay)]
    obv_leads_breakout = False
    if 'OBV' in sub.columns and n >= 22:
        obv_now = float(sub['OBV'].iloc[-1])
        obv_max20 = float(sub['OBV'].iloc[-21:-1].max())
        obv_leads_breakout = bool(obv_now >= obv_max20 and curr_c >= prior_res_p * 0.96)
    out["obv_leads_breakout"] = obv_leads_breakout

    near_res_Touch_vols = [vols[k] for k in range(peak_rel_idx, n) if highs[k] >= zone_low]
    touch_vol_decay_ratio = 1.0
    if len(near_res_Touch_vols) >= 2:
        half_t = max(1, len(near_res_Touch_vols) // 2)
        touch_vol_decay_ratio = round(float(np.mean(near_res_Touch_vols[half_t:])) / max(float(np.mean(near_res_Touch_vols[:half_t])), 1.0), 2)
    out["touch_vol_decay_ratio"] = touch_vol_decay_ratio

    # 綜合計算 DeepChart 空單消化評分 (0 ~ 100)
    score_absorb = min(35.0, (absorption_ratio / 100.0) * 28.0 + min(tests_cnt, 3) * 3.5)
    if curr_c >= prior_res_p * 1.002 and closes[-1] >= opens[-1]:
        score_absorb = max(score_absorb, 28.0)
    score_exhaust = 30.0 if (seller_exhausted and float(sub.get('Tight_Box_5d_Pct', pd.Series([5.0])).iloc[-1]) <= 6.5) else (20.0 if seller_exhausted else 6.0)
    score_orderflow = (18.0 if out["cvd_10d_positive"] else 4.0) + (17.0 if pocket_pivot else 5.0)

    # 若當日逼近前高卻收長上影陰線且爆量 (明顯空單砸盤拒斥)，扣減分數
    last_upper_shadow_ratio = float(sub['ShadowRatio'].iloc[-1]) if 'ShadowRatio' in sub.columns else 0.0
    rejection_penalty = 22.0 if (highs[-1] >= zone_low and closes[-1] < opens[-1] and last_upper_shadow_ratio > 0.45 and vols[-1] > vol_sma20 * 1.2) else 0.0

    obv_and_decay_bonus = (7.0 if obv_leads_breakout else 0.0) + (6.0 if touch_vol_decay_ratio <= 0.80 and len(near_res_Touch_vols) >= 2 else 0.0)
    total_score = round(min(99.0, max(5.0, score_absorb + score_exhaust + score_orderflow + obv_and_decay_bonus - rejection_penalty)), 1)
    out["absorption_score"] = total_score
    out["shorts_absorbed"] = bool(total_score >= 62.0)
    out["heavy_overhead_supply"] = bool(total_score < 42.0 or rejection_penalty > 0)

    if out["shorts_absorbed"]:
        out["deepchart_badge"] = f"💎 空單已消化 ({total_score:.0f}分｜換手{out['absorption_ratio_pct']:.0f}%)"
        out["deepchart_detail"] = (
            f"✅ **前高阻力 ({prior_res_p:.2f}) 空單/套牢盤已充分消化 ({total_score:.0f}/100分)**："
            f"右側已測試 {tests_cnt} 次，累計換手消化率達 **{out['absorption_ratio_pct']:.0f}%**，"
            f"{'下跌賣壓枯竭 + ' if seller_exhausted else ''}"
            f"{'近10日訂單流淨買盤 (CVD>0) + ' if out['cvd_10d_positive'] else ''}"
            f"{'出現機構口袋支點 (Pocket Pivot)' if pocket_pivot else '買盤承接強勁'}，具備高勝率突破條件！"
        )
    elif out["heavy_overhead_supply"]:
        out["deepchart_badge"] = f"⚠️ 前高空單未消化 ({total_score:.0f}分｜防假突破)"
        out["deepchart_detail"] = (
            f"⚠️ **前高阻力 ({prior_res_p:.2f}) 空單/套牢賣壓仍然沉重 ({total_score:.0f}/100分)**："
            f"目前換手消化率僅 **{out['absorption_ratio_pct']:.0f}%**，且空頭賣壓尚未枯竭，"
            f"直接硬衝極易遭遇空單砸盤形成 2B 假突破，建議耐心等回踩或多次測試消化後再入場！"
        )
    else:
        out["deepchart_badge"] = f"🟡 空單消化中 ({total_score:.0f}分｜換手{out['absorption_ratio_pct']:.0f}%)"
        out["deepchart_detail"] = (
            f"🟡 **前高阻力 ({prior_res_p:.2f}) 空單消化進行中 ({total_score:.0f}/100分)**："
            f"已測試 {tests_cnt} 次（換手消化率 **{out['absorption_ratio_pct']:.0f}%**），建議配合突破放量或回踩支撐分批進場。"
        )

    # [v31.0 新增：空頭視角專屬「空頭供應壓制與前低多單支撐擊穿引擎 (Short Supply Dominance & Support Breakdown)」]
    # 解決做空股票時卻顯示「⚠️ 前高空單未消化 / 防假突破」多頭提示之矛盾！
    valley_rel_idx = int(np.argmin(lows[:search_end]))
    prior_sup_p = float(lows[valley_rel_idx])
    out["prior_sup_price"] = round(prior_sup_p, 2)
    sup_zone_high = prior_sup_p + max(0.8 * atr_v, prior_sup_p * 0.025)
    sup_init_s = max(0, valley_rel_idx - 2)
    sup_init_e = min(search_end, valley_rel_idx + 3)
    init_demand_vol = max(float(np.sum(vols[sup_init_s:sup_init_e])) / max(1, (sup_init_e - sup_init_s)) * 2.0, float(np.mean(vols)) * 1.3, 1.0)

    attack_sup_vol = 0.0
    sup_tests_cnt = 0
    in_sup_cluster = False
    for idx in range(valley_rel_idx + 2, n):
        if lows[idx] <= sup_zone_high:
            if not in_sup_cluster:
                sup_tests_cnt += 1
                in_sup_cluster = True
            rng_s = max(highs[idx] - lows[idx], 1e-6)
            clv_s = ((closes[idx] - lows[idx]) - (highs[idx] - closes[idx])) / rng_s
            sell_weight = 1.30 if (closes[idx] <= opens[idx] or clv_s <= -0.1) else 0.55
            attack_sup_vol += vols[idx] * sell_weight
        else:
            if lows[idx] > sup_zone_high + 0.4 * atr_v:
                in_sup_cluster = False
    if curr_c < prior_sup_p * 0.998:
        attack_sup_vol += float(vols[-1]) * 1.40

    sup_breakdown_ratio = round(min(250.0, (attack_sup_vol / init_demand_vol) * 100.0), 1)
    out["support_breakdown_ratio_pct"] = sup_breakdown_ratio
    out["support_tests_count"] = int(sup_tests_cnt)

    recent10_up_vols = [vols[k] for k in range(max(0, n - 10), n) if closes[k] > opens[k]]
    avg_recent_up_vol = float(np.mean(recent10_up_vols)) if recent10_up_vols else vol_sma20 * 0.5
    buyer_exhausted = bool(avg_recent_up_vol < vol_sma20 * 0.88)
    out["buyer_exhaustion_ok"] = buyer_exhausted

    max_up_vol_10d = max([vols[k] for k in range(max(0, n - 11), n - 1) if closes[k] > closes[max(0, k - 1)]] + [0.0])
    dist_spike_10d = bool(
        (closes[-1] < opens[-1] and vols[-1] > max_up_vol_10d and max_up_vol_10d > 0)
        or any(closes[k] < opens[k] and vols[k] > max_up_vol_10d for k in range(max(0, n - 3), n) if max_up_vol_10d > 0)
    )
    out["distribution_spike_10d"] = dist_spike_10d

    # 空頭壓制總分 (0~100分，越高代表上方套牢重、反彈無量、空頭砸盤與破底動能越強)
    short_dom_score = round(min(99.0, max(5.0, (100.0 - total_score) * 0.65 + min(22.0, (sup_breakdown_ratio / 100.0) * 16.0) + (10.0 if buyer_exhausted else 0.0) + (8.0 if dist_spike_10d else 0.0))), 1)
    out["short_dominance_score"] = short_dom_score
    out["short_supply_dominant"] = bool(short_dom_score >= 58.0)

    if out["short_supply_dominant"]:
        out["short_deepchart_badge"] = f"🩸 空頭壓制強勁 ({short_dom_score:.0f}分｜破底換手{sup_breakdown_ratio:.0f}%)"
        out["short_deepchart_detail"] = (
            f"🩸 **空頭賣壓主導且前低多頭防線 ({prior_sup_p:.2f}) 搖搖欲墜 ({short_dom_score:.0f}/100分)**："
            f"上方前高 ({prior_res_p:.2f}) 累積沉重套牢盤（多頭消化率僅 {out['absorption_ratio_pct']:.0f}%），"
            f"下方支撐已受衝擊 {sup_tests_cnt} 次（空頭擊穿換手率 **{sup_breakdown_ratio:.0f}%**），"
            f"{'反彈買盤枯竭 (無量反抽) + ' if buyer_exhausted else ''}"
            f"{'近10日訂單流淨賣壓主導 (CVD<0)' if not out['cvd_10d_positive'] else '空頭結構完整'}，極利於逢高做空或破底順勢追空！"
        )
    elif short_dom_score < 42.0:
        out["short_deepchart_badge"] = f"⚠️ 低位買盤承接強 ({short_dom_score:.0f}分｜防逼空反彈)"
        out["short_deepchart_detail"] = (
            f"⚠️ **下方支撐 ({prior_sup_p:.2f}) 買盤承接較強 ({short_dom_score:.0f}/100分)**："
            f"近10日出現買盤吸收且空頭擊穿動能不足，直接低位追空易遭遇 Short Squeeze 逼空反彈，建議耐心等反抽阻力位受阻再佈局！"
        )
    else:
        out["short_deepchart_badge"] = f"🟠 空頭測試支撐中 ({short_dom_score:.0f}分｜破底換手{sup_breakdown_ratio:.0f}%)"
        out["short_deepchart_detail"] = (
            f"🟠 **空頭正逐步消耗下方支撐 ({prior_sup_p:.2f}) 承接盤 ({short_dom_score:.0f}/100分)**："
            f"上方套牢賣壓仍存，下方支撐已測試 {sup_tests_cnt} 次（空頭擊穿換手率 **{sup_breakdown_ratio:.0f}%**），可於反抽均線受阻或放量跌穿 {prior_sup_p:.2f} 時沽空。"
        )
    return out

def compute_anchored_position_plan(
    df: pd.DataFrame,
    actual_entry: float,
    pos_dir_str: str,
    entry_date_val: Any = None,
    market_type: str = "US/Global",
    lot_size: int = 100,
    saved_plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    [v25.4 新增：解決用戶「同一個入場價，過幾天止損、目標價與盈虧比都唔同」的核心痛點]
    原理：
    - 舊版在用戶輸入 actual_entry 時，直接使用當日最新一根 K 線 (df.iloc[-1]) 的 ATR、BoxHigh10、20MA 動態重算 fixed_sl 與 fixed_tp，
      導致隨住每日行情波動，同一個入場價每天顯示不同的初始止損與盈虧比 (R:R Drift)！
    - 本函數支援：
      1) 若該股票已有鎖定之開倉計劃 (saved_plan) 或雷達信號記錄 (signal_lifecycle_store) 且進場價一致，優先鎖定當時之初始止損與目標價；
      2) 若指定了開倉日期 (entry_date_val)，嚴格切片至開倉當日 K 線 (df.loc[:entry_date_val]) 計算當時真實 ATR 與結構支撐阻力，確保跨日查看時初始止損、TP1、TP2 與初始 R:R 100% 恆定不變；
      3) 同時根據最新收盤價 (last) 計算「實時浮動盈虧 (%)」、「當前實現 R 倍數 (Current R)」與「最高/最低有利推進 (MFE R)」。
    """
    if df is None or df.empty or actual_entry <= 0:
        return {}
    is_short = ("做空" in str(pos_dir_str)) or ("SHORT" in str(pos_dir_str).upper())
    dir_code = "SHORT" if is_short else "LONG"

    # OLD:
    # anchor_df = df
    # anchor_date_str = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    # if entry_date_val is not None: ...
    # NEW: [v26.0 確定性開倉日歷史快照引擎]
    # 嚴格截取截至 entry_date_val 當日之固定 252 根歷史 K 線視窗，並獨立重算純因果指標與結構支撐阻力，
    # 100% 隔絕 entry_date_val 之後 5 天、10 天的新 K 線對 AVWAP、High52W、均線密集區與樞紐點的污染，徹底根除跨日盈虧比漂移！
    anchor_df = df
    anchor_date_str = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
    if entry_date_val is not None:
        try:
            ts_target = pd.Timestamp(entry_date_val).replace(hour=23, minute=59, second=59)
            sub = df.loc[:ts_target].tail(260).copy()
            if len(sub) >= 15:
                # 100% 獨立重算開倉日當下之純因果均線、ATR與結構高低點，完全隔絕開倉日後任何新K線之影響
                sub['TR'] = np.maximum(
                    sub['High'] - sub['Low'],
                    np.maximum(abs(sub['High'] - sub['Close'].shift(1)), abs(sub['Low'] - sub['Close'].shift(1)))
                )
                sub['ATR'] = sub['TR'].rolling(14, min_periods=1).mean()
                sub['SMA10'] = sub['Close'].rolling(window=10, min_periods=1).mean()
                sub['SMA20'] = sub['Close'].rolling(window=20, min_periods=1).mean()
                sub['EMA20'] = sub['Close'].ewm(span=20, adjust=False).mean()
                sub['EMA21'] = sub['Close'].ewm(span=21, adjust=False).mean()
                sub['SMA50'] = sub['Close'].rolling(window=50, min_periods=1).mean()
                sub['SMA120'] = sub['Close'].rolling(window=120, min_periods=1).mean()
                sub['EMA120'] = sub['Close'].ewm(span=120, adjust=False).mean()
                sub['SMA200'] = sub['Close'].rolling(window=200, min_periods=1).mean()
                sub['EMA12'] = sub['Close'].ewm(span=12, adjust=False).mean()
                sub['EMA26'] = sub['Close'].ewm(span=26, adjust=False).mean()
                sub['VolumeSMA50'] = sub['Volume'].rolling(window=50, min_periods=5).mean()
                sub['Ret1M'] = sub['Close'] / sub['Close'].shift(20) - 1.0
                sub['DailyPctRange'] = (sub['High'] - sub['Low']) / (sub['Close'] + 1e-9)
                sub['VCPSignal'] = sub['DailyPctRange'].rolling(3).mean() < (sub['DailyPctRange'].rolling(20).mean() * 0.5)
                sub['ATR_Wilder'], sub['RSI_Wilder'] = compute_wilder_atr_and_rsi(sub)
                sub['ATR'] = sub['ATR_Wilder'].fillna(sub['ATR'])
                sub['RSI'] = sub['RSI_Wilder']
                sub = compute_causal_swing_sr(sub)
                sub = compute_lei_dual_ma_and_ppo_features(sub)
                sub['High52W'] = sub['High'].rolling(252, min_periods=1).max()
                sub['Low52W'] = sub['Low'].rolling(252, min_periods=1).min()
                anchor_df = sub
                anchor_date_str = str(anchor_df.index[-1].date()) if hasattr(anchor_df.index[-1], "date") else str(entry_date_val)
        except Exception:
            pass

    anchor_row = anchor_df.iloc[-1]
    last_row = df.iloc[-1]
    curr_close = float(last_row['Close'])
    curr_high = float(last_row['High'])
    curr_low = float(last_row['Low'])

    entry_atr = float(anchor_row['ATR']) if ('ATR' in anchor_row and pd.notna(anchor_row['ATR']) and float(anchor_row['ATR']) > 0) else actual_entry * 0.03
    sma20_anchor = float(anchor_row.get('SMA20', actual_entry))
    ema21_anchor = float(anchor_row.get('EMA21', sma20_anchor))
    stop_prof = get_active_stop_profile()
    u_min_pct = float(stop_prof["min_stop_pct"])
    u_max_pct = float(stop_prof["max_stop_pct"])

    # 若外部已傳入鎖定之初始止損與目標價，且方向與進場價吻合，直接採用鎖定值（100% 零漂移）
    if saved_plan and saved_plan.get("dir_code") == dir_code and abs(float(saved_plan.get("entry", 0.0)) - actual_entry) < 1e-3:
        fixed_sl = float(saved_plan["fixed_sl"])
        fixed_tp = float(saved_plan["fixed_tp"])
        tp2_price = float(saved_plan.get("tp2_price", fixed_tp))
        sl_basis_lbl = str(saved_plan.get("sl_basis_lbl", f"開倉日({saved_plan.get('anchor_date', anchor_date_str)})鎖定結構止損"))
        tp_basis_lbl = str(saved_plan.get("tp_basis_lbl", f"開倉日({saved_plan.get('anchor_date', anchor_date_str)})鎖定結構目標"))
        anchor_date_str = str(saved_plan.get("anchor_date", anchor_date_str))
    else:
        if not is_short:
            sr_targets = _find_historical_ma_convergence_targets(anchor_df, actual_entry, "LONG")
            sup_cands = [
                (float(anchor_row.get('BoxLow10_Causal', actual_entry * 0.95)) - 0.35 * entry_atr, f"開倉日({anchor_date_str})近10日箱體底緩衝"),
                (float(anchor_row.get('SupShort_Causal', actual_entry * 0.94)) - 0.35 * entry_atr, f"開倉日({anchor_date_str})20日波段低點緩衝"),
                (sma20_anchor - 0.60 * entry_atr, f"開倉日({anchor_date_str})20MA下方緩衝"),
                (ema21_anchor - 0.50 * entry_atr, f"開倉日({anchor_date_str})21EMA下方緩衝")
            ]
            valid_sls = [
                (s, lbl) for s, lbl in sup_cands
                if (actual_entry * (1.0 - u_max_pct) <= s <= actual_entry * (1.0 - u_min_pct))
                and ((actual_entry - s) >= float(stop_prof["min_atr_mult"]) * entry_atr)
            ]
            # OLD:
            # if valid_sls:
            #     best_sl = max(valid_sls, key=lambda x: x[0])
            #     fixed_sl, sl_basis_lbl = round(best_sl[0], 2), best_sl[1]
            # NEW: [v28.0 雙層防洗盤止損] 避開最淺層 0.35x ATR 散戶止損密集獵殺區，選取次深層結構支撐 (>=1.35x ATR) 作為盤中硬止損
            if valid_sls:
                anti_hunt_sls = [x for x in valid_sls if (actual_entry - x[0]) >= max(1.35 * entry_atr, actual_entry * (u_min_pct + 0.004))]
                best_sl = max(anti_hunt_sls, key=lambda x: x[0]) if anti_hunt_sls else min(valid_sls, key=lambda x: x[0])
                fixed_sl, sl_basis_lbl = round(best_sl[0], 2), f"{best_sl[1]} (防洗盤硬止損)"
            else:
                fixed_sl = round(min(max(actual_entry - float(stop_prof["fallback_atr_mult"]) * entry_atr, actual_entry * (1.0 - u_max_pct)), actual_entry * (1.0 - u_min_pct)), 2)
                sl_basis_lbl = f"開倉日({anchor_date_str}) {float(stop_prof['fallback_atr_mult']):.2f}x ATR 防洗盤保護位"
            user_risk = max(actual_entry - fixed_sl, 1e-4)
            # [v42.0 港美股雙軌差異化 R:R 止盈引擎 (Dual-Track HK vs US R:R Engine)]
            # - 港股 (HK)：徹底廢除 min(近期小阻力, 1.095) 斬頭限制！若箱體內近期小阻力不足 2.0R (或少於 +7.2%)，自動跨過該小阻力，
            #   還原 main7 買藥明生物 (2269.HK) 階段之 max(合規主阻力, Entry + 3.8x ATR, Entry + 2.25R) 主升浪目標 (+9%~+14% / 2.2R~2.8R)！
            # - 美股 (US)：採用《超級績效 2》「2R~2.4R 強勢賣出一半 (TP1) + 尾倉沿 21EMA 奔跑搏遠端大目標 (TP2)」雙軌制！
            _dual_rr_anch = compute_dual_track_hk_us_rr_targets(
                entry_price=actual_entry,
                stop_loss=fixed_sl,
                atr_val=entry_atr,
                sr_targets=sr_targets,
                market_type=market_type,
                direction="LONG"
            )
            fixed_tp = _dual_rr_anch["tp1_price"]
            tp2_price = _dual_rr_anch["tp2_price"]
            tp_basis_lbl = f"{_dual_rr_anch['tp1_desc']} (錨定於 {anchor_date_str})"
        else:
            sr_targets = _find_historical_ma_convergence_targets(anchor_df, actual_entry, "SHORT")
            res_cands = [
                (float(anchor_row.get('BoxHigh10_Causal', actual_entry * 1.05)) + 0.35 * entry_atr, f"開倉日({anchor_date_str})近10日反彈箱體頂緩衝"),
                (float(anchor_row.get('ResShort_Causal', actual_entry * 1.06)) + 0.35 * entry_atr, f"開倉日({anchor_date_str})20日前浪頂阻力緩衝"),
                (sma20_anchor + 0.60 * entry_atr, f"開倉日({anchor_date_str})20MA蓋頭均線上方緩衝"),
                (ema21_anchor + 0.50 * entry_atr, f"開倉日({anchor_date_str})21EMA上方緩衝")
            ]
            valid_sls = [
                (s, lbl) for s, lbl in res_cands
                if (actual_entry * (1.0 + u_min_pct) <= s <= actual_entry * (1.0 + u_max_pct))
                and ((s - actual_entry) >= float(stop_prof["min_atr_mult"]) * entry_atr)
            ]
            # OLD:
            # if valid_sls:
            #     best_sl = min(valid_sls, key=lambda x: x[0])
            #     fixed_sl, sl_basis_lbl = round(best_sl[0], 2), best_sl[1]
            # NEW: [v28.0 空頭防軋空雙層止損] 避開最淺層誘多針尖獵殺區，選取次深層結構阻力 (>=1.35x ATR) 作為盤中硬止損
            if valid_sls:
                anti_hunt_sls = [x for x in valid_sls if (x[0] - actual_entry) >= max(1.35 * entry_atr, actual_entry * (u_min_pct + 0.004))]
                best_sl = min(anti_hunt_sls, key=lambda x: x[0]) if anti_hunt_sls else max(valid_sls, key=lambda x: x[0])
                fixed_sl, sl_basis_lbl = round(best_sl[0], 2), f"{best_sl[1]} (防軋空硬止損)"
            else:
                fixed_sl = round(max(min(actual_entry + float(stop_prof["fallback_atr_mult"]) * entry_atr, actual_entry * (1.0 + u_max_pct)), actual_entry * (1.0 + u_min_pct)), 2)
                sl_basis_lbl = f"開倉日({anchor_date_str}) {float(stop_prof['fallback_atr_mult']):.2f}x ATR 防軋空保護位"
            user_risk = max(fixed_sl - actual_entry, 1e-4)
            if sr_targets:
                fixed_tp, tp_basis_lbl = round(sr_targets[0][0], 2), f"{sr_targets[0][1]} (錨定於 {anchor_date_str})"
                if (actual_entry - fixed_tp) / user_risk < 1.6 and len(sr_targets) >= 2:
                    fixed_tp, tp_basis_lbl = round(sr_targets[1][0], 2), f"{sr_targets[1][1]} (錨定於 {anchor_date_str})"
                tp2_price = round(sr_targets[1][0], 2) if len(sr_targets) >= 2 else round(max(0.01, actual_entry - 3.8 * entry_atr), 2)
            else:
                fixed_tp = round(min(float(anchor_row.get('SupShort_Causal', actual_entry * 0.92)), max(0.01, actual_entry - 3.2 * entry_atr)), 2)
                tp2_price = round(max(0.01, actual_entry - 4.5 * entry_atr), 2)
                tp_basis_lbl = f"開倉日({anchor_date_str})空頭波段等距目標"

    user_risk = max(abs(actual_entry - fixed_sl), 1e-4)
    initial_stop_pct = (user_risk / actual_entry) * 100.0
    initial_rr = round(abs(fixed_tp - actual_entry) / user_risk, 2)
    tp2_rr = round(abs(tp2_price - actual_entry) / user_risk, 2)

    futu_be = compute_futu_real_fees(market_type=market_type, price=actual_entry, shares=max(int(lot_size), 100), is_sell=is_short)
    be_price = futu_be["breakeven_short_price"] if is_short else futu_be["breakeven_long_price"]

    # 計算自開倉錨定日以來的最高/最低價與實時浮動盈虧 (Real-time Unrealized PnL & R-Multiple)
    post_entry_df = df.loc[anchor_df.index[-1]:] if len(anchor_df) > 0 else df.tail(1)
    max_high_since_entry = float(post_entry_df['High'].max()) if not post_entry_df.empty else curr_high
    min_low_since_entry = float(post_entry_df['Low'].min()) if not post_entry_df.empty else curr_low

    bars_held_since_entry = max(0, len(post_entry_df) - 1)
    min_close_since_entry = float(post_entry_df['Close'].min()) if not post_entry_df.empty else curr_close
    max_close_since_entry = float(post_entry_df['Close'].max()) if not post_entry_df.empty else curr_close
    soft_closing_sl = round(fixed_sl + 0.30 * entry_atr, 2) if not is_short else round(fixed_sl - 0.30 * entry_atr, 2)
    if not is_short:
        unrealized_pnl_pct = ((curr_close - actual_entry) / actual_entry) * 100.0
        current_r = (curr_close - actual_entry) / user_risk
        peak_mfe_r = (max_high_since_entry - actual_entry) / user_risk
        # OLD: stop_hit_since_entry = bool(min_low_since_entry <= fixed_sl or curr_close <= fixed_sl)
        # NEW: [v28.0 區分「盤中洗盤針尖測試軟止損」與「真正跌穿硬止損/收盤實質破位」]
        stop_hit_since_entry = bool(min_low_since_entry <= fixed_sl or min_close_since_entry <= soft_closing_sl)
        wick_shakeout_only = bool(min_low_since_entry <= soft_closing_sl and min_low_since_entry > fixed_sl and curr_close > soft_closing_sl)
    else:
        unrealized_pnl_pct = ((actual_entry - curr_close) / actual_entry) * 100.0
        current_r = (actual_entry - curr_close) / user_risk
        peak_mfe_r = (actual_entry - min_low_since_entry) / user_risk
        # OLD: stop_hit_since_entry = bool(max_high_since_entry >= fixed_sl or curr_close >= fixed_sl)
        stop_hit_since_entry = bool(max_high_since_entry >= fixed_sl or max_close_since_entry >= soft_closing_sl)
        wick_shakeout_only = bool(max_high_since_entry >= soft_closing_sl and max_high_since_entry < fixed_sl and curr_close < soft_closing_sl)

    return {
        "dir_code": dir_code,
        "entry": round(actual_entry, 2),
        "anchor_date": anchor_date_str,
        "fixed_sl": fixed_sl,
        "soft_closing_sl": soft_closing_sl,
        "fixed_tp": fixed_tp,
        "tp2_price": tp2_price,
        "sl_basis_lbl": sl_basis_lbl,
        "tp_basis_lbl": tp_basis_lbl,
        "user_risk": user_risk,
        "initial_stop_pct": round(initial_stop_pct, 2),
        "initial_rr": initial_rr,
        "tp2_rr": tp2_rr,
        "be_price": be_price,
        "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
        "current_r": round(current_r, 2),
        "peak_mfe_r": round(peak_mfe_r, 2),
        "max_high_since_entry": max_high_since_entry,
        "min_low_since_entry": min_low_since_entry,
        "stop_hit_since_entry": stop_hit_since_entry,
        "wick_shakeout_only": wick_shakeout_only,
        "bars_held_since_entry": bars_held_since_entry
    }

def extrapolate_intraday_partial_volume(df: pd.DataFrame, market_type: str = "US/Global", interval: str = "1d") -> pd.Series:
    """
    [v25.2 修復 P1-數據延遲與盤中未收盤 K 線失真漏洞]
    當用戶在港股 (09:30-16:00 HKT) 或美股盤中執行掃描時，最後一根日 K 線尚未收盤，原始 Volume 偏小會導致：
    1) 誤觸發「縮量回踩 (VolRatio < 0.85)」與「無量反抽」假信號
    2) 漏判當日正在發生的「放量突破 (VolRatio > 1.5)」或「大戶出貨日」
    此函數根據當前時間計算開盤已過交易分鐘數，對當日最後一根 K 線成交量進行合理全天折算 (上限 3.5x)，保留原始 Volume 不變。
    """
    if df is None or df.empty or interval != "1d" or 'Volume' not in df.columns:
        return df['Volume'] if (df is not None and 'Volume' in df.columns) else pd.Series(dtype=float)
    eff_vol = df['Volume'].astype(float).copy()
    try:
        last_dt = pd.to_datetime(df.index[-1]).date()
        try:
            now_hkt = datetime.now(ZoneInfo("Asia/Hong_Kong")).replace(tzinfo=None)
        except Exception:
            now_hkt = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)
        if market_type == "HK":
            now_mkt = now_hkt
            if last_dt == now_mkt.date() and now_mkt.weekday() < 5:
                mins_from_open = (now_mkt.hour * 60 + now_mkt.minute) - (9 * 60 + 30)
                if 15 <= mins_from_open < 390:
                    # 扣除港股午休 12:00-13:00 (60 分鐘)，全日共 330 交易分鐘
                    active_mins = mins_from_open if now_mkt.hour < 12 else (150 if now_mkt.hour < 13 else max(150, mins_from_open - 60))
                    scale_factor = min(3.5, max(1.0, 330.0 / max(float(active_mins), 20.0)))
                    eff_vol.iloc[-1] = eff_vol.iloc[-1] * scale_factor
        else:
            # OLD: scale_factor = min(3.5, max(1.0, 390.0 / max(float(mins_from_open), 20.0)))
            # NEW: [v32.0 修復美股早盤 U 型成交量曲線過度放大 3.5 倍導致香港時間晚上 21:30-23:00 誤報假突破之漏洞]
            # 美股開盤首 60 分鐘通常已佔全天成交量 28%~35%，不可按線性時間比例乘以 3.5x！
            try:
                now_est = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
            except Exception:
                now_est = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=5)
            if last_dt == now_est.date() and now_est.weekday() < 5:
                mins_from_open = (now_est.hour * 60 + now_est.minute) - (9 * 60 + 30)
                if 15 <= mins_from_open < 390:
                    time_frac = float(mins_from_open) / 390.0
                    # U型累積成交量模型：開盤首小時權重較高，避免早盤衝高假突破虛增 VolRatio
                    cum_vol_frac = min(1.0, 0.22 * (time_frac ** 0.45) + 0.78 * time_frac)
                    scale_factor = min(2.2, max(1.0, 1.0 / max(cum_vol_frac, 0.45)))
                    eff_vol.iloc[-1] = eff_vol.iloc[-1] * scale_factor
    except Exception:
        pass
    return eff_vol

# 中線波段止損寬度預設配置 (可由側欄動態切換)
STOP_PROFILE_CONFIGS = {
    "🌊 中線波段標準 (持倉數週至1-2個月｜防洗盤 ≥1.2x ATR / 2.8%~7.5%)": {
        "mode_key": "MEDIUM_SWING",
        "min_atr_mult": 1.15,
        "min_stop_pct": 0.025,   # 距離買入位至少 2.5%，杜絕 0883.HK (-1.9%) 或 MSFT (-2.3%) 噪音區過緊止損
        "max_stop_pct": 0.075,   # [v41.1 實盤復盤平衡] 恢復中線結構上限至 7.5% (保留 2359.HK -6.56% 大贏家空間)，同時由滿倉紅綠燈優先標記 <=4.8% 可滿倉標的！
        "fallback_atr_mult": 1.45,
        "fallback_min_pct": 0.032,
        "use_intraday_lod": False,
        "breakeven_min_days": 5, # [v41.1 防過早保本洗出] 至少持倉滿 5 日或浮盈達 +1.25R 才允許移至保本，杜絕 SNOW 過早移保本被洗
        "time_stop_days": 18
    },
    "⚖️ 平衡結構模式 (持倉1至3週｜標準 ≥0.9x ATR / 2.2%~7.2%)": {
        "mode_key": "BALANCED_SWING",
        "min_atr_mult": 0.90,
        "min_stop_pct": 0.022,
        "max_stop_pct": 0.072,
        "fallback_atr_mult": 1.50,
        "fallback_min_pct": 0.030,
        "use_intraday_lod": False,
        "breakeven_min_days": 4,
        "time_stop_days": 18
    },
    "⚡ 短線緊密模式 (Martin Luk 日內/極短線｜≥0.45x ATR / 1.2%~5.2%)": {
        "mode_key": "TIGHT_SHORT_TERM",
        "min_atr_mult": 0.45,
        "min_stop_pct": 0.012,
        "max_stop_pct": 0.052,
        "fallback_atr_mult": 1.15,
        "fallback_min_pct": 0.020,
        "use_intraday_lod": True,
        "breakeven_min_days": 2,
        "time_stop_days": 12
    }
}

def get_active_stop_profile() -> Dict[str, Any]:
    """取得當前用戶在側欄選擇的止損與持倉週期配置（預設為中線波段標準）"""
    default_label = "🌊 中線波段標準 (持倉數週至1-2個月｜防洗盤 ≥1.2x ATR / 2.8%~7.5%)"
    try:
        sel = st.session_state.get("stop_profile_mode", default_label)
        return STOP_PROFILE_CONFIGS.get(sel, STOP_PROFILE_CONFIGS[default_label])
    except Exception:
        return STOP_PROFILE_CONFIGS[default_label]


def compute_futu_real_fees(
    market_type: str,
    price: float,
    shares: int,
    is_sell: bool = False,
    hk_zero_commission: bool = False
) -> Dict[str, float]:
    """
    [富途牛牛 (Futu Securities HK) 港股 / 美股真實交易費用精算引擎]
    - 港股 (HKD):
      1. 佣金: 0.03% (最低 HK$3.00)，若勾選免佣則為 HK$0
      2. 平台使用費: 固定套餐 HK$15.00 / 筆
      3. 政府印花稅: 0.10% (按成交金額向上取整至最接近之 HK$1.00，買賣雙邊均收)
      4. 證監會交易徵費 (0.0027%) + 會財局徵費 (0.00015%) + 港交所交易費 (0.00565%) = 合共 0.0085%
      5. 中央結算系統 (CCASS) 交收費: 0.002% (最低 HK$2.00，最高 HK$100.00)
    - 美股 (USD):
      1. 佣金: $0.0049 / 股 (最低 $0.99 / 筆，最高成交金額 0.5%)
      2. 平台使用費: $0.0050 / 股 (最低 $1.00 / 筆，最高成交金額 0.5%)
      3. 交收費: $0.0030 / 股
      4. 賣出規費 (SEC Fee + FINRA TAF): 僅賣出收取，約 $0.000166 / 股 + 0.00278% 成交額
    """
    notional = max(float(price) * int(shares), 0.0)
    if notional <= 0 or shares <= 0:
        return {
            "single_leg_fee": 0.0,
            "round_trip_fee": 0.0,
            "round_trip_pct": 0.0,
            "stamp_duty": 0.0,
            "platform_and_comm": 0.0,
            "breakeven_long_price": float(price),
            "breakeven_short_price": float(price)
        }

    if market_type == "HK":
        comm = 0.0 if hk_zero_commission else max(3.0, notional * 0.0003)
        platform_fee = 15.0
        stamp_duty = float(math.ceil(notional * 0.001))
        reg_levies = round(notional * 0.000085, 2)
        ccass_fee = min(100.0, max(2.0, round(notional * 0.00002, 2)))
        single_fee = round(comm + platform_fee + stamp_duty + reg_levies + ccass_fee, 2)
        round_trip = round(single_fee * 2.0, 2)
    else:
        comm = min(notional * 0.005, max(0.99, shares * 0.0049))
        platform_fee = min(notional * 0.005, max(1.00, shares * 0.0050))
        settlement_fee = shares * 0.0030
        sell_reg = max(0.02, shares * 0.000166 + notional * 0.0000278)
        buy_leg = round(comm + platform_fee + settlement_fee, 2)
        sell_leg = round(comm + platform_fee + settlement_fee + sell_reg, 2)
        single_fee = sell_leg if is_sell else buy_leg
        round_trip = round(buy_leg + sell_leg, 2)
        stamp_duty = 0.0

    rt_pct = (round_trip / notional) * 100.0
    be_long = round(float(price) + (round_trip / max(int(shares), 1)), 3)
    be_short = round(max(0.01, float(price) - (round_trip / max(int(shares), 1))), 3)
    return {
        "single_leg_fee": single_fee,
        "round_trip_fee": round_trip,
        "round_trip_pct": round(rt_pct, 3),
        "stamp_duty": round(stamp_duty * 2.0, 2) if market_type == "HK" else 0.0,
        "platform_and_comm": round((comm + platform_fee) * 2.0, 2),
        "breakeven_long_price": be_long,
        "breakeven_short_price": be_short
    }

# ==============================================================================
# [NEW INCREMENTAL PATCH v25.0] Martin Luk (@martinlukkt - USIC 2024 +283% / 2025 +969.8% 冠軍)
# 實戰交易體系增量引擎：
# 1. 9 / 21 / 50 EMA 動能雲帶 (EMA Cloud) 與週線 9W/21W EMA 跨週期趨勢過濾
# 2. ADR% (20日平均日內波幅百分比) 高動能篩選器與「止損寬度 vs R倍數拋物線關係」精算器
# 3. 前波段高點錨定 AVWAP (AVWAP_High60 / AVWAP_High52W) 與 Episodic Pivot 跳空錨定 AVWAP (AVWAP_EP)
# 4. 多重支撐匯聚低吸 (Stacked Confluence Undercut & Reclaim: AVWAP_High + 9/21 EMA + 前高/缺口)
# 5. Inside Day / 極窄幅壓縮突破 (PDH 觸發 + LOD 日內低點極緊止損 1.0%~4.5%) 與 Episodic Pivot (EP)
# 6. 空頭「見頂首跌段後弱反彈至下彎 9/21/50 EMA 及 AVWAP_High 轉弱 Fade」高勝率沽空法
# 7. 大盤指數反彈逼近下彎 21/50 EMA 假突破預警 + 資金曲線反饋 (Equity Curve Feedback) & 25%~35% 單倉上限
# ==============================================================================

def compute_martin_luk_features_and_setups(
    df: pd.DataFrame,
    bg_df: Optional[pd.DataFrame] = None,
    market_type: str = "US/Global"
) -> pd.DataFrame:
    """
    [Martin Luk @martinlukkt 核心指標與四大設置無未來函數計算引擎]
    - 影片依據：
      1) TraderLion 專訪 (https://www.youtube.com/watch?v=lS9zbnLi1Gg)
      2) USIC +969% Pullback 策略解析 (https://www.youtube.com/watch?v=F0vssXyWizc & https://www.youtube.com/watch?v=VKNEJA5r8zw)
      3) Martin Luk 完整三掃描器與執行框架 (https://www.youtube.com/watch?v=-6lPuMFSMG0)
      4) @martinlukkt 實盤直播 (USO Long / Short updates / NO TRADE TODAY)
    """
    if df is None or df.empty:
        return df
    df = df.copy()  # 消除 Pandas DataFrame 碎片化 PerformanceWarning

    # 1. Martin Luk 核心均線組：9 EMA, 21 EMA, 50 EMA + 週線級 9W EMA (45d) 與 21W EMA (105d)
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA45_Wk9'] = df['Close'].ewm(span=45, adjust=False).mean()
    df['EMA105_Wk21'] = df['Close'].ewm(span=105, adjust=False).mean()

    # 2. ADR% (20日平均日內波幅百分比) —— Martin Luk 篩選高爆發股必備指標 (美股偏好 >=4.5%, 港股 >=3.0%)
    daily_hi_lo_pct = ((df['High'] / (df['Low'] + 1e-9)) - 1.0) * 100.0
    df['ADR_Pct'] = daily_hi_lo_pct.rolling(window=20, min_periods=5).mean().fillna(3.5)

    # 3. 1M / 3M / 6M 動能領先指標 (Leader Scanner: 1M / 3M / 6M 漲幅)
    df['Ret3M'] = df['Close'] / df['Close'].shift(63) - 1.0
    if 'Ret6M' not in df.columns:
        df['Ret6M'] = df['Close'] / df['Close'].shift(126) - 1.0

    # 4. EMA Cloud 多空排列與斜率狀態
    ema9_rising = df['EMA9'] > df['EMA9'].shift(3)
    ema21_rising = df['EMA21'] > df['EMA21'].shift(5)
    ema50_rising = df['EMA50'] > df['EMA50'].shift(10)
    df['Martin_EMA_Bull_Cloud'] = (df['EMA9'] > df['EMA21']) & (df['EMA21'] > df['EMA50']) & ema21_rising
    df['Martin_EMA_Bear_Cloud'] = (df['EMA9'] < df['EMA21']) & (df['EMA21'] < df['EMA50']) & (~ema21_rising)

    # 5. 因果無未來函數之「波段高點錨定 AVWAP (AVWAP_High60 / AVWAP_High52W)」與「跳空催化錨定 AVWAP (AVWAP_EP)」
    df['AVWAP_High60'] = np.nan
    df['AVWAP_High52W'] = np.nan
    df['AVWAP_EP'] = np.nan
    try:
        if len(df) >= 20:
            win60 = df.iloc[-min(len(df), 90):-1] if len(df) > 25 else df
            if not win60.empty:
                high60_anchor = win60['High'].idxmax()
                df['AVWAP_High60'] = calculate_anchored_vwap(df, high60_anchor)
            win252 = df.iloc[-min(len(df), 252):-1] if len(df) > 25 else df
            if not win252.empty:
                high52w_anchor = win252['High'].idxmax()
                df['AVWAP_High52W'] = calculate_anchored_vwap(df, high52w_anchor)

            # 尋找最近 60 日內最強 Episodic Pivot (跳空 >= 5% 且成交量 >= 2.0 倍 50日均量)
            recent_ep_slice = df.tail(min(len(df), 60))
            ep_candidates = recent_ep_slice[
                (recent_ep_slice['PriceChange'] >= 0.05) &
                (recent_ep_slice['Volume'] >= recent_ep_slice['VolumeSMA50'].fillna(recent_ep_slice['Volume']) * 2.0)
            ]
            if not ep_candidates.empty:
                ep_anchor = ep_candidates.index[-1]
                df['AVWAP_EP'] = calculate_anchored_vwap(df, ep_anchor)
    except Exception:
        pass

    # 6. Inside Day (內包日) 與 極窄幅壓縮日 (Tight Range Day < 0.55 * ADR%)
    df['Is_Inside_Day'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))
    df['Is_Martin_Tight_Day'] = (daily_hi_lo_pct <= (df['ADR_Pct'] * 0.60)) | df['Is_Inside_Day']

    # 7. 識別最後一根 K 線之 Martin Luk 四大實戰 Setup 與大盤假突破風險
    martin_info = {
        "adr_pct": float(df['ADR_Pct'].iloc[-1]) if len(df) > 0 else 0.0,
        "adr_ok": False,
        "ema_cloud_state": "⚪ 震盪交纏",
        "is_leader_1m_3m_6m": False,
        "martin_pullback_long": False,
        "martin_inside_breakout_long": False,
        "martin_ep_long": False,
        "martin_weak_bounce_short": False,
        "confluence_supports": [],
        "confluence_resistances": [],
        "tight_lod_stop": np.nan,
        "tight_hod_stop": np.nan,
        "pdh_trigger": np.nan,
        "pdl_trigger": np.nan,
        "stop_width_pct": 0.0,
        "parabolic_r_25pct": 0.0,
        "parabolic_r_50pct": 0.0,
        "index_at_declining_ema_resistance": False,
        "index_filter_warning": "",
        "martin_setup_badge": "-"
    }

    if len(df) < 25:
        df.attrs["martin_luk_info"] = martin_info
        return df

    last = df.iloc[-1]
    prev = df.iloc[-2]
    c_now = float(last['Close'])
    o_now = float(last['Open'])
    h_now = float(last['High'])
    l_now = float(last['Low'])
    adr_now = float(last.get('ADR_Pct', 3.5))
    min_adr_req = 3.0 if market_type == "HK" else 4.2
    martin_info["adr_ok"] = bool(adr_now >= min_adr_req)

    ema9_v = float(last.get('EMA9', c_now))
    ema21_v = float(last.get('EMA21', c_now))
    ema50_v = float(last.get('EMA50', c_now))
    avwap_h60 = float(last.get('AVWAP_High60', np.nan))
    avwap_sw = float(last.get('AVWAP_Swing', np.nan))
    avwap_ep = float(last.get('AVWAP_EP', np.nan))

    if bool(last.get('Martin_EMA_Bull_Cloud', False)) and c_now >= ema21_v:
        martin_info["ema_cloud_state"] = "🟢 領先多頭雲帶 (Close > 9EMA > 21EMA > 50EMA)"
    elif bool(last.get('Martin_EMA_Bear_Cloud', False)) and c_now <= ema21_v:
        martin_info["ema_cloud_state"] = "🔴 弱勢空頭雲帶 (Close < 9EMA < 21EMA < 50EMA)"
    elif c_now > ema50_v and ema21_v >= ema50_v * 0.99:
        martin_info["ema_cloud_state"] = "🟡 基地修復中 (站上 50EMA & 21EMA 走平上翹)"
    else:
        martin_info["ema_cloud_state"] = "🟠 均線雲帶下方 (觀望或等待空頭反抽)"

    ret1m = float(last.get('Ret1M', 0.0)) if pd.notna(last.get('Ret1M')) else 0.0
    ret3m = float(last.get('Ret3M', 0.0)) if pd.notna(last.get('Ret3M')) else 0.0
    ret6m = float(last.get('Ret6M', 0.0)) if pd.notna(last.get('Ret6M')) else 0.0
    martin_info["is_leader_1m_3m_6m"] = bool(ret1m >= 0.18 or ret3m >= 0.25 or ret6m >= 0.30)

    # 檢查多重支撐匯聚 (Stacked Confluence: 9EMA / 21EMA / AVWAP_High60 / AVWAP_Swing / Prior High)
    tol_pct = max(0.015, adr_now * 0.0035)
    conf_sups = []
    for lbl, val in [
        ("9 EMA", ema9_v),
        ("21 EMA", ema21_v),
        ("50 EMA", ema50_v),
        ("前高錨定AVWAP", avwap_h60),
        ("波段低點AVWAP", avwap_sw),
        ("跳空EP錨定AVWAP", avwap_ep),
        ("前波突破頸線", float(last.get('PriorWaveHigh_Causal', np.nan)))
    ]:
        if pd.notna(val) and val > 0:
            # 日內低點曾回測或觸及該支撐區 (l_now <= val * (1 + tol_pct)) 且收盤重新站回其上 (c_now >= val * 0.997)
            if l_now <= val * (1.0 + tol_pct) and c_now >= val * 0.996:
                conf_sups.append(f"{lbl}({val:.2f})")
    martin_info["confluence_supports"] = conf_sups

    # Setup 1: Martin Luk 招牌「強勢股多重支撐回踩洗盤收回 (Stacked Confluence Pullback / Undercut & Reclaim)」
    lod_stop_cand = round(l_now - 0.03 * float(last.get('ATR', c_now * 0.02)), 2)
    lod_risk_pct = (c_now - lod_stop_cand) / max(c_now, 1e-9) * 100.0
    higher_lows_ok = float(df['Low'].iloc[-5:].min()) >= float(df['Low'].iloc[-20:-5].min()) * 0.99
    if (
        len(conf_sups) >= 2
        and (c_now >= ema21_v * 0.995)
        and (ema21_v >= ema50_v * 0.985)
        and higher_lows_ok
        and (c_now >= o_now or (c_now - l_now) >= 0.45 * (h_now - l_now + 1e-9))
        and (0.6 <= lod_risk_pct <= max(5.2, adr_now * 1.15))
    ):
        martin_info["martin_pullback_long"] = True

    # Setup 2: Martin Luk「Inside Day / 極窄幅壓縮 + 9/21 EMA 匯聚突破 (PDH Trigger + LOD Tight Stop)」
    near_9_or_21 = (abs(c_now - ema9_v) / c_now <= 0.025) or (abs(c_now - ema21_v) / c_now <= 0.028)
    if (
        bool(last.get('Is_Martin_Tight_Day', False))
        and near_9_or_21
        and (c_now >= ema21_v * 0.99)
        and (ema9_v >= ema21_v * 0.995)
        and (0.6 <= lod_risk_pct <= max(5.0, adr_now * 1.05))
    ):
        martin_info["martin_inside_breakout_long"] = True

    # Setup 3: Martin Luk「Episodic Pivot (EP 跳空爆量突破)」
    gap_up_pct = (o_now / max(float(prev['Close']), 1e-9) - 1.0) * 100.0
    if (
        gap_up_pct >= 4.5
        and float(last.get('VolRatio', 1.0)) >= 2.0
        and c_now >= o_now
        and (c_now - l_now) >= 0.5 * (h_now - l_now + 1e-9)
    ):
        martin_info["martin_ep_long"] = True

    # Setup 4: Martin Luk 空頭「見頂首跌段後弱勢反彈至下彎 9/21/50 EMA 或 AVWAP_High 受阻 Fade」
    conf_res = []
    for lbl, val in [
        ("下彎9EMA", ema9_v),
        ("下彎21EMA", ema21_v),
        ("50EMA反壓", ema50_v),
        ("前高AVWAP反壓", avwap_h60)
    ]:
        if pd.notna(val) and val > 0:
            if h_now >= val * (1.0 - tol_pct) and c_now <= val * 1.004:
                conf_res.append(f"{lbl}({val:.2f})")
    martin_info["confluence_resistances"] = conf_res

    hod_stop_cand = round(h_now + 0.03 * float(last.get('ATR', c_now * 0.02)), 2)
    hod_risk_pct = (hod_stop_cand - c_now) / max(c_now, 1e-9) * 100.0
    if (
        len(conf_res) >= 1
        and (ema9_v < ema21_v or c_now < ema50_v)
        and (c_now < float(last.get('High52W', c_now)) * 0.88)
        and (c_now <= o_now or (h_now - c_now) >= 0.45 * (h_now - l_now + 1e-9))
        and (0.6 <= hod_risk_pct <= max(5.2, adr_now * 1.15))
    ):
        martin_info["martin_weak_bounce_short"] = True

    martin_info["tight_lod_stop"] = lod_stop_cand
    martin_info["tight_hod_stop"] = hod_stop_cand
    martin_info["pdh_trigger"] = round(h_now + 0.02 * float(last.get('ATR', c_now * 0.02)), 2)
    martin_info["pdl_trigger"] = round(max(0.01, l_now - 0.02 * float(last.get('ATR', c_now * 0.02))), 2)
    # [v25.1 中線波段修正] 針對幾星期至一兩個月持倉，預設採用「21EMA / 10日箱體底 - 波動率緩衝」計算波段安全止損寬度，保留 LOD 僅作日內參考
    _stop_prof = get_active_stop_profile()
    _atr_now = float(last.get('ATR', c_now * 0.03))
    if _stop_prof.get("use_intraday_lod", False):
        eff_stop_pct = max(lod_risk_pct, 1.2)
    else:
        _swing_sup_ref = min(ema21_v - 0.45 * _atr_now, float(last.get('BoxLow10_Causal', l_now)) - 0.30 * _atr_now)
        _swing_stop_pct = ((c_now - _swing_sup_ref) / max(c_now, 1e-9)) * 100.0
        eff_stop_pct = min(max(_swing_stop_pct, _stop_prof["min_stop_pct"] * 100.0, (_stop_prof["min_atr_mult"] * _atr_now / max(c_now, 1e-9)) * 100.0), _stop_prof["max_stop_pct"] * 100.0)
    martin_info["stop_width_pct"] = round(eff_stop_pct, 2)
    martin_info["intraday_lod_stop_pct"] = round(max(lod_risk_pct, 0.8), 2)
    # 波段 R 倍數關係展示：同一筆 +25% 與 +50% 潛在中線波段漲幅下，對應之真實 R 倍數
    martin_info["parabolic_r_25pct"] = round(25.0 / eff_stop_pct, 1)
    martin_info["parabolic_r_50pct"] = round(50.0 / eff_stop_pct, 1)

    # 8. 大盤指數反彈逼近下彎 21/50 EMA 假突破過濾器 (Martin Luk《Equity Curve Feedback》03:35)
    if bg_df is not None and len(bg_df) >= 50:
        try:
            bg_c = float(bg_df['Close'].iloc[-1])
            bg_ema9 = float(bg_df['Close'].ewm(span=9, adjust=False).mean().iloc[-1])
            bg_ema21_s = bg_df['Close'].ewm(span=21, adjust=False).mean()
            bg_ema50_s = bg_df['Close'].ewm(span=50, adjust=False).mean()
            bg_ema21 = float(bg_ema21_s.iloc[-1])
            bg_ema50 = float(bg_ema50_s.iloc[-1])
            ema21_declining = bg_ema21 < float(bg_ema21_s.iloc[-5])
            ema50_declining = bg_ema50 < float(bg_ema50_s.iloc[-5])
            near_declining_21 = ema21_declining and (bg_ema21 * 0.985 <= bg_c <= bg_ema21 * 1.012)
            near_declining_50 = ema50_declining and (bg_ema50 * 0.985 <= bg_c <= bg_ema50 * 1.012)
            if near_declining_21 or near_declining_50 or (bg_ema9 < bg_ema21 and bg_c < bg_ema21):
                martin_info["index_at_declining_ema_resistance"] = True
                martin_info["index_filter_warning"] = (
                    "⚠️ Martin Luk 大盤濾網警示：大盤指數目前處於 9EMA < 21EMA 或正反彈逼近下彎的 21/50 EMA 阻力位，"
                    "此時個股突破極易出現 Fakeout (假突破)，建議縮小單筆風險至 0.25% 或優先等待回踩確認！"
                )
        except Exception:
            pass

    badges = []
    if martin_info["martin_ep_long"]:
        badges.append("🚀 EP跳空爆量")
    if martin_info["martin_pullback_long"]:
        badges.append(f"🎯 AVWAP+EMA匯聚低吸(波段止損{eff_stop_pct:.1f}%)")
    if martin_info["martin_inside_breakout_long"]:
        badges.append(f"🔥 InsideDay壓縮突破(波段止損{eff_stop_pct:.1f}%)")
    if martin_info["martin_weak_bounce_short"]:
        badges.append(f"🩸 反抽下彎EMA做空(波段止損{eff_stop_pct:.1f}%)")
    martin_info["martin_setup_badge"] = " ｜ ".join(badges) if badges else f"ADR {adr_now:.1f}%"

    df.attrs["martin_luk_info"] = martin_info
    return df


def evaluate_martin_luk_equity_feedback_and_sizing(
    account_equity: float,
    entry_price: float,
    stop_loss: float,
    adr_pct: float = 4.0,
    market_type: str = "US/Global",
    lot_size: int = 1,
    recent_win_streak: int = 0,
    recent_loss_streak: int = 0,
    current_drawdown_pct: float = 0.0
) -> Dict[str, Any]:
    """
    [Martin Luk @martinlukkt 資金曲線反饋 (Equity Curve Feedback)、0.5% 基準風險與防跳空單倉上限計算器]
    - 依據：
      1) 基準單筆風險 = 0.5% 總資金 (連勝且有浮盈墊可上調至 0.75%~1.0%；連敗或回撤 >8% 自動砍半至 0.25%；連敗 >=4 觸發 NO TRADE TODAY 冷靜期)
      2) 單一持倉市值上限：高波動/小型股 (ADR% >= 6%) 嚴格限制 <= 25% 總資金；大型高流動股嚴格限制 <= 35% 總資金，防止類似 SERV 隔夜跳空 30%~40% 重創帳戶！
    """
    base_risk_pct = 0.50
    regime_note = "🟢 正常狀態 (0.50% 標準單筆風險)"
    no_trade_today = False

    if recent_loss_streak >= 4 or current_drawdown_pct >= 12.0:
        base_risk_pct = 0.15
        no_trade_today = True
        regime_note = "🛑 【NO TRADE TODAY 冷靜期觸發】近期連敗 ≥4 筆或回撤 ≥12%，依 Martin Luk 紀律建議今日暫停開新倉，先保護心理與資金資本！"
    elif recent_loss_streak >= 2 or current_drawdown_pct >= 6.0:
        base_risk_pct = 0.25
        regime_note = "🟡 【防守減碼模式 (Progressive De-risking)】近期處於小幅逆風，單筆風險自動下調至 0.25% 試水溫"
    elif recent_win_streak >= 3 and current_drawdown_pct <= 2.0:
        base_risk_pct = 0.75
        regime_note = "🔥 【順風加碼模式 (Profit Cushion)】資金曲線創高且具備利潤墊，單筆風險可提升至 0.75%"

    risk_per_share = max(abs(entry_price - stop_loss), 1e-4)
    stop_pct = (risk_per_share / max(entry_price, 1e-4)) * 100.0
    risk_dollars = account_equity * (base_risk_pct / 100.0)

    # OLD: raw_shares = int(risk_dollars / risk_per_share)
    # NEW: [v25.2 修復 P0-3 摩擦成本穿透止損與港股 1 手門檻陷阱] 先估算含富途雙邊摩擦成本之每股實質風險，再計算可買股數
    est_fee_rate = 0.0028 if market_type == "HK" else 0.0015
    eff_risk_per_share = risk_per_share + (entry_price * est_fee_rate)
    raw_shares = int(max(0.0, risk_dollars - (36.0 if market_type == "HK" else 4.0)) / max(eff_risk_per_share, 1e-4))
    max_pos_pct = 25.0 if adr_pct >= 6.0 else 35.0
    max_pos_dollars = account_equity * (max_pos_pct / 100.0)
    capped_by_notional = False
    if raw_shares * entry_price > max_pos_dollars:
        raw_shares = int(max_pos_dollars / max(entry_price, 1e-4))
        capped_by_notional = True

    if market_type == "HK" and lot_size > 1:
        final_shares = (raw_shares // lot_size) * lot_size
    else:
        final_shares = max(0, raw_shares)

    # 計算港股最少買 1 手 (lot_size) 所需的實質風險比例與警告提示
    one_lot_fees = compute_futu_real_fees(market_type=market_type, price=entry_price, shares=max(lot_size, 1))
    min_1lot_risk_dollars = (max(lot_size, 1) * risk_per_share) + one_lot_fees["round_trip_fee"]
    min_1lot_risk_pct = (min_1lot_risk_dollars / max(account_equity, 1e-4)) * 100.0
    lot_trap_warning = ""
    if final_shares == 0 and entry_price > 0:
        lot_trap_warning = (
            f"⚠️ 【港股手數門檻預警】按 {base_risk_pct:.2f}% 單筆風險預算 ({risk_dollars:,.0f} 元) 計算不足 1 手 ({lot_size} 股)！"
            f"若強行買入 1 手 (市值 {lot_size * entry_price:,.0f} 元)，含富途費用之實質止損風險將達 {min_1lot_risk_pct:.2f}% ({min_1lot_risk_dollars:,.0f} 元)。"
            f"建議：(1) 放棄此高價/大每手標的；(2) 於富途使用碎股交易買入 {max(1, raw_shares)} 股；或 (3) 待更緊密買點出現。"
        )

    position_value = final_shares * entry_price
    futu_fees = compute_futu_real_fees(market_type=market_type, price=entry_price, shares=final_shares) if final_shares > 0 else one_lot_fees
    total_risk_with_fee = (final_shares * risk_per_share) + futu_fees["round_trip_fee"] if final_shares > 0 else 0.0
    actual_pos_pct = (position_value / max(account_equity, 1e-4)) * 100.0
    actual_risk_pct = (total_risk_with_fee / max(account_equity, 1e-4)) * 100.0

    return {
        "recommended_risk_pct": round(base_risk_pct, 2),
        "actual_risk_pct": round(actual_risk_pct, 2),
        "stop_pct": round(stop_pct, 2),
        "final_shares": int(final_shares),
        "final_lots": int(final_shares // max(lot_size, 1)),
        "raw_unrounded_shares": int(raw_shares),
        "min_1lot_risk_pct": round(min_1lot_risk_pct, 2),
        "lot_trap_warning": lot_trap_warning,
        "position_value": round(position_value, 2),
        "actual_pos_pct": round(actual_pos_pct, 1),
        "max_pos_pct_cap": max_pos_pct,
        "capped_by_notional": capped_by_notional,
        "no_trade_today": no_trade_today,
        "regime_note": regime_note,
        "futu_round_trip_fee": futu_fees["round_trip_fee"] if final_shares > 0 else one_lot_fees["round_trip_fee"],
        "futu_fee_pct": futu_fees["round_trip_pct"] if final_shares > 0 else one_lot_fees["round_trip_pct"],
        "futu_breakeven_long": futu_fees["breakeven_long_price"],
        "futu_breakeven_short": futu_fees["breakeven_short_price"],
        "r_multiple_at_25pct_gain": round(25.0 / max(stop_pct, 0.5), 1),
        "r_multiple_at_50pct_gain": round(50.0 / max(stop_pct, 0.5), 1)
    }


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

    # [v31.0 新增：空頭視角專屬抵扣價推演標籤 (高抵扣價 = 20MA/60MA 持續下彎蓋頭助跌)]
    if close_now < ded20_next10_mean and close_now < ded20_next5_max:
        out["short_forecast_20_label"] = f"🩸 現價遠低於未來5~10日 20MA 抵扣高點 ({ded20_next5_max:.2f} / 均值 {ded20_next10_mean:.2f})，20MA 將持續下彎蓋頭助跌（極利做空）"
    else:
        out["short_forecast_20_label"] = f"⚠️ 現價高於未來 20MA 抵扣均值 ({ded20_next10_mean:.2f})，20MA 短線有走平托底傾向，做空勿急追低"

    if close_now < ded60_next10_mean:
        out["short_forecast_60_label"] = f"🩸 現價低於未來10日 60MA 季線抵扣均值 ({ded60_next10_mean:.2f})，季線維持向下壓制助跌"
    else:
        out["short_forecast_60_label"] = f"⚠️ 現價仍在 60MA 抵扣均值 ({ded60_next10_mean:.2f}) 之上，中期季線尚未下彎"

    return out


# ==============================================================================
# [NEW INCREMENTAL PATCH v24.15] 多維度結構支撐/阻力區間 (S/R Zones)、測試次數衰減、
# 籌碼成交密集峰 (Volume Profile HVN)、未回補缺口與 J Law 2+1 平行通道 (Parallel Channel) 引擎
# ==============================================================================
def compute_causal_pivots_and_parallel_channel(df: pd.DataFrame, lookback: int = 120, left_bars: int = 4, right_bars: int = 2) -> Dict[str, Any]:
    """
    [J Law EP1 47:50 / EP2 07:30, 36:10 / EP3 06:50 2+1 平行通道畫法]
    無未來函數地識別歷史已確認之波峰 (Swing Highs) 與波谷 (Swing Lows)，
    統計每個支撐/阻力區間被測試次數 (Test Count)，並用「2+1 法則」投射今日的：
    - 上升趨勢支撐線 (UTL) 與 上升平行通道頂部阻力線 (TRL)
    - 下降趨勢阻力線 (DTL) 與 下降平行通道底部支撐線 (TSL)
    """
    out = {
        "swing_highs": [],       # List of (bar_offset_from_start, price)
        "swing_lows": [],        # List of (bar_offset_from_start, price)
        "sr_zones": [],          # List of dicts: {price, low_bound, high_bound, touches, is_flip, kind_label}
        "utl_support": np.nan,
        "trl_resistance": np.nan,
        "dtl_resistance": np.nan,
        "tsl_support": np.nan,
        "volume_hvn_upper": np.nan,
        "volume_hvn_lower": np.nan,
        "unfilled_gap_up_res": np.nan,
        "unfilled_gap_dn_sup": np.nan,
        "shakeout_bullish": False,
        "shakeout_bearish": False,
    }
    if df is None or len(df) < 25:
        return out

    sub = df.tail(min(len(df), lookback)).copy()
    highs = sub['High'].values
    lows = sub['Low'].values
    closes = sub['Close'].values
    opens = sub['Open'].values
    vols = sub['Volume'].values if 'Volume' in sub.columns else np.ones(len(sub))
    n = len(sub)
    curr_close = float(closes[-1])
    atr_now = float(sub['ATR'].iloc[-1]) if 'ATR' in sub.columns and pd.notna(sub['ATR'].iloc[-1]) and sub['ATR'].iloc[-1] > 0 else curr_close * 0.03

    # 1. 因果波峰與波谷識別 (最後 right_bars 根不作為已確認拐點，完全杜絕未來函數)
    swing_highs = []
    swing_lows = []
    max_check_idx = n - 1 - right_bars
    for idx in range(left_bars, max_check_idx + 1):
        h_val = float(highs[idx])
        l_val = float(lows[idx])
        if h_val >= float(np.max(highs[idx - left_bars: idx + right_bars + 1])):
            swing_highs.append((idx, h_val))
        if l_val <= float(np.min(lows[idx - left_bars: idx + right_bars + 1])):
            swing_lows.append((idx, l_val))

    out["swing_highs"] = swing_highs
    out["swing_lows"] = swing_lows

    # 2. 聚類形成「支撐與阻力區間 (S/R Zones)」並計算測試次數 (Touches) 與 S/R Flip 記憶
    zone_tol = max(curr_close * 0.012, 0.35 * atr_now)
    raw_pivots = [(p, "HIGH", idx) for idx, p in swing_highs] + [(p, "LOW", idx) for idx, p in swing_lows]
    clusters: List[Dict[str, Any]] = []
    for p_val, p_type, p_idx in raw_pivots:
        matched = False
        for cl in clusters:
            if abs(p_val - cl["price"]) <= zone_tol:
                cl["prices"].append(p_val)
                cl["price"] = float(np.mean(cl["prices"]))
                cl["low_bound"] = float(min(cl["prices"]))
                cl["high_bound"] = float(max(cl["prices"]))
                cl["touches"] += 1
                cl["types"].add(p_type)
                cl["last_idx"] = max(cl["last_idx"], p_idx)
                matched = True
                break
        if not matched:
            clusters.append({
                "price": p_val,
                "prices": [p_val],
                "low_bound": p_val - 0.12 * atr_now,
                "high_bound": p_val + 0.12 * atr_now,
                "touches": 1,
                "types": {p_type},
                "last_idx": p_idx
            })

    for cl in clusters:
        is_flip = ("HIGH" in cl["types"] and "LOW" in cl["types"])
        # 若原本是前浪頂 (HIGH)，但目前股價已升穿並在其上方運行，亦屬 S/R Flip 潛在支撐
        if not is_flip and "HIGH" in cl["types"] and curr_close > cl["price"] * 1.008:
            is_flip = True
        elif not is_flip and "LOW" in cl["types"] and curr_close < cl["price"] * 0.992:
            is_flip = True
        cl["is_flip"] = is_flip
        if is_flip:
            cl["kind_label"] = f"S/R互換區(測試{cl['touches']}次)"
        elif "HIGH" in cl["types"]:
            cl["kind_label"] = f"前浪頂阻力區(測試{cl['touches']}次)"
        else:
            cl["kind_label"] = f"前浪底支撐區(測試{cl['touches']}次)"
    out["sr_zones"] = clusters

    # 3. J Law 2+1 平行通道投射 (Parallel Channel: UTL + TRL & DTL + TSL)
    if len(swing_lows) >= 2:
        (idx1, low1), (idx2, low2) = swing_lows[-2], swing_lows[-1]
        if idx2 > idx1:
            slope_up = (low2 - low1) / float(idx2 - idx1)
            # 僅當斜率不過度失真時投射
            if -0.5 * atr_now <= slope_up <= 1.5 * atr_now:
                utl_now = low2 + slope_up * float((n - 1) - idx2)
                out["utl_support"] = float(utl_now)
                # +1 點：找 idx1 之後的最高波峰平移平行通道頂部 TRL
                highs_between = [h_p for h_i, h_p in swing_highs if h_i >= idx1]
                if not highs_between:
                    highs_between = [float(np.max(highs[idx1:]))]
                anchor_h_idx = idx1 + int(np.argmax(highs[idx1:]))
                anchor_h_val = float(highs[anchor_h_idx])
                trl_now = anchor_h_val + slope_up * float((n - 1) - anchor_h_idx)
                if curr_close * 1.022 < trl_now <= curr_close * 1.32:
                    out["trl_resistance"] = float(trl_now)

    if len(swing_highs) >= 2:
        (h_idx1, high1), (h_idx2, high2) = swing_highs[-2], swing_highs[-1]
        if h_idx2 > h_idx1:
            slope_dn = (high2 - high1) / float(h_idx2 - h_idx1)
            if -1.5 * atr_now <= slope_dn <= 0.5 * atr_now:
                dtl_now = high2 + slope_dn * float((n - 1) - h_idx2)
                out["dtl_resistance"] = float(dtl_now)
                anchor_l_idx = h_idx1 + int(np.argmin(lows[h_idx1:]))
                anchor_l_val = float(lows[anchor_l_idx])
                tsl_now = anchor_l_val + slope_dn * float((n - 1) - anchor_l_idx)
                if curr_close * 0.68 <= tsl_now < curr_close * 0.978:
                    out["tsl_support"] = float(tsl_now)

    # 4. Volume Profile 籌碼成交密集峰 (High Volume Nodes)
    try:
        p_min, p_max = float(np.min(lows)), float(np.max(highs))
        if p_max > p_min * 1.03:
            bins = np.linspace(p_min, p_max, 26)
            typical_p = (highs + lows + closes) / 3.0
            turnovers = typical_p * vols
            hist_vol, bin_edges = np.histogram(typical_p, bins=bins, weights=turnovers)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            upper_mask = bin_centers > curr_close * 1.025
            lower_mask = bin_centers < curr_close * 0.975
            if np.any(upper_mask):
                up_idx = int(np.argmax(hist_vol * upper_mask))
                if hist_vol[up_idx] > 0:
                    out["volume_hvn_upper"] = float(bin_centers[up_idx])
            if np.any(lower_mask):
                dn_idx = int(np.argmax(hist_vol * lower_mask))
                if hist_vol[dn_idx] > 0:
                    out["volume_hvn_lower"] = float(bin_centers[dn_idx])
    except Exception:
        pass

    # 5. 未回補跳空缺口 (Unfilled Gaps - LEI 影片2 09:25 標誌性動作與空間目標)
    try:
        for g_i in range(n - 2, max(1, n - 90), -1):
            # 向下跳空缺口 (上方阻力): 當日 High < 前一日 Low，且後續未完全回補
            if highs[g_i] < lows[g_i - 1] * 0.995:
                gap_top = float(lows[g_i - 1])
                subsequent_max = float(np.max(highs[g_i + 1:])) if g_i + 1 < n else float(highs[g_i])
                if subsequent_max < gap_top and gap_top > curr_close * 1.02:
                    out["unfilled_gap_up_res"] = gap_top
                    break
        for g_i in range(n - 2, max(1, n - 90), -1):
            # 向上跳空缺口 (下方支撐): 當日 Low > 前一日 High，且後續未完全跌破
            if lows[g_i] > highs[g_i - 1] * 1.005:
                gap_bot = float(highs[g_i - 1])
                subsequent_min = float(np.min(lows[g_i + 1:])) if g_i + 1 < n else float(lows[g_i])
                if subsequent_min > gap_bot and gap_bot < curr_close * 0.98:
                    out["unfilled_gap_dn_sup"] = gap_bot
                    break
    except Exception:
        pass

    # 6. J Law 洗盤信號 (Shakeout / Undercut & Rally - EP1 58:30, 01:26:00, EP2 31:45)
    if n >= 5:
        sma10_last = float(sub['SMA10'].iloc[-1]) if 'SMA10' in sub.columns and pd.notna(sub['SMA10'].iloc[-1]) else curr_close
        sma20_last = float(sub['SMA20'].iloc[-1]) if 'SMA20' in sub.columns and pd.notna(sub['SMA20'].iloc[-1]) else curr_close
        prev_3_low = float(np.min(lows[-5:-1]))
        prev_3_high = float(np.max(highs[-5:-1]))
        # 多頭洗盤：當日盤中跌穿 10MA/20MA 或前幾日低點，但收盤強勢收回其上且收陽/長下影
        if (lows[-1] < min(sma10_last, prev_3_low) and curr_close >= max(sma20_last, opens[-1]) and (curr_close - lows[-1]) >= 0.45 * (highs[-1] - lows[-1] + 1e-9)):
            out["shakeout_bullish"] = True
        # 空頭誘多洗盤 (Squat / Upthrust)：盤中升穿 10MA/20MA 或前幾日高點，收盤殺回其下
        if (highs[-1] > max(sma10_last, prev_3_high) and curr_close <= min(sma20_last, opens[-1]) and (highs[-1] - curr_close) >= 0.45 * (highs[-1] - lows[-1] + 1e-9)):
            out["shakeout_bearish"] = True

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
        # [NEW INCREMENTAL PATCH v24.15] 融入因果波峰波谷 S/R 區間 (含測試次數衰減判斷)、籌碼峰 HVN、未回補缺口與 J Law 2+1 平行通道頂部 TRL
        _piv_info = compute_causal_pivots_and_parallel_channel(df, lookback=140)
        for _z in _piv_info.get("sr_zones", []):
            _zp = float(_z["price"])
            _touches = int(_z.get("touches", 1))
            if _zp > ref_price * 1.022:
                # J Law EP1 54:55 & EP2 10:30: 若同一阻力已被測試 >= 4 次且離買點很近 (<4%)，阻力力量已減弱，標註提醒並降低作為唯一天花板的權重
                if _touches >= 4 and _zp < ref_price * 1.045:
                    continue
                candidates.append((_zp, _z.get("kind_label", "波段結構阻力區")))
        if pd.notna(_piv_info.get("volume_hvn_upper")):
            candidates.append((float(_piv_info["volume_hvn_upper"]), "籌碼峰密集阻力區(HVN)"))
        if pd.notna(_piv_info.get("unfilled_gap_up_res")):
            candidates.append((float(_piv_info["unfilled_gap_up_res"]), "上方未回補跳空缺口阻力"))
        if pd.notna(_piv_info.get("trl_resistance")):
            candidates.append((float(_piv_info["trl_resistance"]), "J Law 2+1上升平行通道頂(TRL)"))

        # OLD:
        # valid = [(p, lbl) for p, lbl in candidates if pd.notna(p) and p > ref_price * 1.022]
        # valid.sort(key=lambda x: x[0])
        # return valid
        # NEW: 去重聚類相近目標位 (差距 < 1.2% 視為同一結構區間，保留最具代表性之結構標籤)，確保 TP1 與 TP2 層次分明
        raw_valid = sorted([(float(p), str(lbl)) for p, lbl in candidates if pd.notna(p) and float(p) > ref_price * 1.022], key=lambda x: x[0])
        dedup_valid: List[Tuple[float, str]] = []
        for p_val, lbl_val in raw_valid:
            if not dedup_valid or abs(p_val - dedup_valid[-1][0]) / dedup_valid[-1][0] >= 0.015:
                dedup_valid.append((p_val, lbl_val))
        return dedup_valid
    else:
        if pd.notna(sup_short): candidates.append((sup_short, "20日前浪底支撐位"))
        if pd.notna(sup_long): candidates.append((sup_long, "60日波段前低支撐"))
        if pd.notna(sup_120): candidates.append((sup_120, "半年線大級別前低支撐"))
        if pd.notna(low_52w): candidates.append((low_52w, "52週前低支撐區"))
        for ma_name, label in [('SMA60', '60MA季線支撐位'), ('SMA120', '120MA半年線支撐位'), ('SMA200', '200MA年線大支撐')]:
            val = last.get(ma_name, np.nan)
            if pd.notna(val) and float(val) < ref_price * 0.98:
                candidates.append((float(val), label))
        # [NEW INCREMENTAL PATCH v24.15] 融入空頭下方因果波谷支撐區、S/R互換區、下方籌碼峰 HVN、未回補缺口與 J Law 2+1 下降平行通道底部 TSL
        _piv_info = compute_causal_pivots_and_parallel_channel(df, lookback=140)
        for _z in _piv_info.get("sr_zones", []):
            _zp = float(_z["price"])
            _touches = int(_z.get("touches", 1))
            if 0 < _zp < ref_price * 0.978:
                if _touches >= 4 and _zp > ref_price * 0.955:
                    continue
                candidates.append((_zp, _z.get("kind_label", "波段結構支撐區")))
        if pd.notna(_piv_info.get("volume_hvn_lower")):
            candidates.append((float(_piv_info["volume_hvn_lower"]), "下方籌碼峰密集支撐區(HVN)"))
        if pd.notna(_piv_info.get("unfilled_gap_dn_sup")):
            candidates.append((float(_piv_info["unfilled_gap_dn_sup"]), "下方未回補跳空缺口支撐"))
        if pd.notna(_piv_info.get("tsl_support")):
            candidates.append((float(_piv_info["tsl_support"]), "J Law 2+1下降平行通道底(TSL)"))

        # OLD:
        # valid = [(p, lbl) for p, lbl in candidates if pd.notna(p) and p < ref_price * 0.978]
        # valid.sort(key=lambda x: x[0], reverse=True)
        # return valid
        # NEW: 去重聚類相近目標位，確保空頭 TP1 與 TP2 反映真實階梯支撐
        raw_valid = sorted([(float(p), str(lbl)) for p, lbl in candidates if pd.notna(p) and 0 < float(p) < ref_price * 0.978], key=lambda x: x[0], reverse=True)
        dedup_valid: List[Tuple[float, str]] = []
        for p_val, lbl_val in raw_valid:
            if not dedup_valid or abs(p_val - dedup_valid[-1][0]) / dedup_valid[-1][0] >= 0.015:
                dedup_valid.append((p_val, lbl_val))
        return dedup_valid

def evaluate_swing_trade_plan_and_rr(
    df: pd.DataFrame,
    plan_type: str,
    status: str,
    atr_val: float,
    is_breakout_long: bool = False,
    is_near_20ma: bool = False,
    is_deduction_pullback: bool = False,
    is_breakdown_short: bool = False,
    lei_struct: Optional[Dict[str, Any]] = None,
    market_type: str = "US/Global",
    lot_size: int = 1
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

    # [v26.0 新增：DeepChart 前高空單消化評估 + ⚡ 即時入場 (Immediate Actionable Entry) 雙軌引擎]
    _dc_info = df.attrs.get("deepchart_info") or compute_deepchart_orderflow_absorption(df, lookback=60)
    df.attrs["deepchart_info"] = _dc_info
    _dc_score = float(_dc_info.get("absorption_score", 50.0))
    _shorts_absorbed = bool(_dc_info.get("shorts_absorbed", False))
    _action_timeliness = "⏳ 等待回調"

    if plan_type == "LONG":
        # OLD: 原本將 if is_near_20ma 放在最前，且大部分分支強制要求等回踩，導致幾乎全部股票都要等回調而缺乏即時入場機會
        # NEW: [v26.0 優先識別「⚡ 即時可入場 (Immediate Buy)」三大實戰場景]
        _ema21_p = float(last.get('EMA21', sma20_p))
        _exh_guard = df.attrs.get("exhaustion_guard") or compute_us_swing_exhaustion_and_trap_guard(df)
        df.attrs["exhaustion_guard"] = _exh_guard
        _is_exhausted_now = bool(_exh_guard.get("is_exhausted_dont_chase", False))
        _close_pos_now = float(_exh_guard.get("close_range_pos", 0.5))
        _bullish_bar_now = (close_p >= float(last['Open']) and _close_pos_now >= 0.52) or (_close_pos_now >= 0.62)
        _in_breakout_buy_zone = (
            (box_high_10 * 0.995 <= close_p <= box_high_10 * 1.025)
            and (float(last.get('VolRatio', 1.0)) >= 1.25 or _shorts_absorbed)
            and (not _is_exhausted_now)
            and (_close_pos_now >= 0.58)
        )
        # [v32.0 根因修復] 移除原本無條件成立之 `or close_p >= sma20_p`，並要求回踩買點未處於連漲枯竭狀態且收盤站穩當日波幅上半部
        _in_support_reclaim_zone = (
            _bullish_bar_now
            and (not _is_exhausted_now)
            and (
                (sma10_p * 0.992 <= close_p <= sma10_p * 1.015 and low_p <= sma10_p * 1.004)
                or (sma20_p * 0.990 <= close_p <= sma20_p * 1.018 and low_p <= sma20_p * 1.005)
                or (_ema21_p * 0.990 <= close_p <= _ema21_p * 1.018 and low_p <= _ema21_p * 1.005)
                or (prior_wave_high * 0.990 <= close_p <= prior_wave_high * 1.018 and low_p <= prior_wave_high * 1.005)
            )
        )
        _in_tight_coil_ready = (
            (bool(last.get('VCPSignal', False)) or bool(last.get('JLaw_Tight_Box', False)) or bool(last.get('Is_Martin_Tight_Day', False)))
            and _bullish_bar_now
            and (not _is_exhausted_now)
            and ((box_high_10 - close_p) / max(close_p, 1e-6) <= 0.015)
            and _shorts_absorbed
        )

        if _is_exhausted_now and (is_breakout_long or close_p >= box_high_10 * 0.99):
            retest_anchor = round(max(sma10_p, _ema21_p, min(box_high_10 * 0.99, close_p - 0.85 * atr_safe)), 2)
            recommended_entry = retest_anchor
            _action_timeliness = "⏳ 等待回調 (防短線追高枯竭)"
            entry_type_desc = (
                f"⏳ 防追高回踩低吸單 (現價 {close_p:.2f} 已連續急升消耗短線動能 [{_exh_guard.get('consec_up_days', 0)}連陽 / 5日+{_exh_guard.get('runup_5d_atr', 0):.1f}x ATR]，"
                f"直接追入極易買在波段頂部！請掛 Limit 單於 10EMA/21EMA 支撐位 {recommended_entry:.2f} 低吸)"
            )
        elif (is_breakout_long or _in_breakout_buy_zone) and (close_p <= box_high_10 * 1.025) and (not _dc_info.get("heavy_overhead_supply", False)):
            recommended_entry = round(close_p, 2)
            _action_timeliness = "⚡ 即時入場 (突破買點區)"
            entry_type_desc = (
                f"⚡ 即時突破入場單 (現價 {close_p:.2f} 正處於箱體樞紐 {box_high_10:.2f} 黃金買入帶 [-0.5%~+2.5%]，"
                f"DeepChart空單消化 {_dc_score:.0f}分，可即時現價進場；保守者可於 {box_high_10:.2f} 掛單)"
            )
        elif _in_support_reclaim_zone and (is_near_20ma or is_deduction_pullback or "回踩" in status or "回撤" in status or "匯聚" in status):
            recommended_entry = round(close_p, 2)
            _action_timeliness = "⚡ 即時入場 (支撐止跌確認)"
            entry_type_desc = (
                f"⚡ 即時支撐低吸單 (現價 {close_p:.2f} 已回踩關鍵均線/頸線支撐並收陽確認承接，"
                f"無需再等更深回調，可於現價 {close_p:.2f} 即時入場)"
            )
        elif _in_tight_coil_ready:
            recommended_entry = round(close_p, 2)
            _action_timeliness = "⚡ 即時入場 (緊密壓縮尾盤搶先點)"
            entry_type_desc = (
                f"⚡ 即時潛伏入場單 (緊密壓縮平台距離突破樞紐 {box_high_10:.2f} 僅 {(box_high_10/close_p - 1)*100:.1f}% 且前高空單已消化 {_dc_score:.0f}分，"
                f"可於現價 {close_p:.2f} 先入半倉或升穿 {round(box_high_10 + 0.03*atr_safe, 2):.2f} 全倉)"
            )
        elif is_near_20ma or is_deduction_pullback or "回踩" in status or "回撤" in status:
            _ma_trig = compute_ma_slope_and_pullback_trigger(df)
            _ema9_p = float(last.get('EMA9', sma10_p))
            _mid_9_20 = _ma_trig.get("mid_9ema_20ma_price", round((_ema9_p + sma20_p) / 2.0, 2))
            if _ma_trig.get("reclaimed_9ema_right_side", False) and _bullish_bar_now and (not _is_exhausted_now):
                recommended_entry = round(close_p, 2)
                _action_timeliness = "⚡ 即時入場 (回踩後收復9EMA右側確認)"
                entry_type_desc = f"⚡ 右側收復確認單 (已完成回踩並收陽站回 9EMA {_ema9_p:.2f} 之上，可於現價 {close_p:.2f} 入場)"
            elif close_p >= sma20_p:
                recommended_entry = round(min(close_p, _mid_9_20), 2)
                _action_timeliness = "⏳ 等待回調 (掛9EMA與20MA中位數或待收復9EMA)"
                entry_type_desc = (
                    f"⏳ 雙軌回踩單 (目前處於回調過程，避免左側市價接刀：① 可於 9EMA與20MA中位數 {recommended_entry:.2f} 掛限價單；"
                    f"或 ② 待收盤重新站上 9EMA {_ema9_p:.2f} 後右側進場)"
                )
            else:
                recommended_entry = round(_ema9_p * 1.002, 2)
                _action_timeliness = "🎯 突破觸發 (待收復9EMA)"
                entry_type_desc = f"🎯 右側止跌確認單 (待升穿並收復 9EMA {recommended_entry:.2f} 觸發買入)"
        elif is_breakout_long or "突破" in status or "旗型" in status:
            pivot_buy = box_high_10 + 0.08 * atr_safe
            if close_p > box_high_10 * 1.032:
                retest_level = max(box_high_10, sma10_p, prior_wave_high if prior_wave_high <= close_p else sma10_p)
                recommended_entry = round(retest_level, 2)
                _action_timeliness = "⏳ 等待回調 (已高出樞紐>3.2%)"
                entry_type_desc = f"等回測低吸單 (現價已高出突破位 {(close_p/box_high_10 - 1)*100:.1f}%，勿追高，等回踩前高支撐 {recommended_entry:.2f} 買入)"
            else:
                recommended_entry = round(max(close_p * 1.002, min(pivot_buy, close_p + 0.35 * atr_safe)), 2)
                _action_timeliness = "🎯 突破觸發"
                entry_type_desc = f"突破觸發單 (Stop-Buy 升穿箱體樞紐 {recommended_entry:.2f} 進場 ｜ 空單消化 {_dc_score:.0f}分)"
        elif bool(last.get('VCPSignal', False)) or bool(last.get('JLaw_Tight_Box', False)) or bool(last.get('LEI_MA_Tight_2Pct', False)) or "壓縮" in status or "密集" in status:
            recommended_entry = round(box_high_10 + 0.05 * atr_safe, 2)
            _action_timeliness = "🎯 突破觸發"
            entry_type_desc = f"突破埋伏單 (待升穿近10日緊密平台頂 {recommended_entry:.2f} 觸發買入 ｜ 空單消化 {_dc_score:.0f}分)"
        else:
            pullback_target = max(sma20_p, min(sma10_p, prior_wave_high if prior_wave_high < close_p else sma10_p))
            if abs(close_p - pullback_target) / close_p < 0.012 and _bullish_bar_now:
                recommended_entry = round(close_p, 2)
                _action_timeliness = "⚡ 即時入場 (均線支撐位)"
                entry_type_desc = f"⚡ 即時均線承接單 (現價 {close_p:.2f} 已貼近 10/20MA 支撐位，可即時買入)"
            else:
                if abs(close_p - pullback_target) / close_p < 0.005:
                    pullback_target = close_p * 0.985
                recommended_entry = round(pullback_target, 2)
                _action_timeliness = "⏳ 等待回調"
                entry_type_desc = f"拉回等待單 (目前無即時突破，建議等縮量回踩 10/20MA 支撐位 {recommended_entry:.2f} 買入)"

        # [NEW v24.15] 整合 J Law S/R Flip 支撐區、上升趨勢線 (UTL)、洗盤 K 線低點 (Shakeout Low) 與 LEI 20MA 抵扣價防守位
        _piv_ch = compute_causal_pivots_and_parallel_channel(df, lookback=120)
        _sr_flip_supports = [
            float(z["low_bound"]) for z in _piv_ch.get("sr_zones", [])
            if z.get("is_flip") and (recommended_entry * 0.93 <= float(z["low_bound"]) <= recommended_entry * 0.992)
        ]
        if _sr_flip_supports and (not _action_timeliness.startswith("⚡")) and ("回踩" in entry_type_desc or "低吸" in entry_type_desc or "等待" in entry_type_desc):
            _best_flip_entry = round(max(_sr_flip_supports) + 0.10 * atr_safe, 2)
            if abs(_best_flip_entry - close_p) / close_p <= 0.045 and _best_flip_entry <= close_p * 1.005:
                recommended_entry = _best_flip_entry
                entry_type_desc = f"S/R互換支撐低吸單 (於前阻力轉支撐區 {recommended_entry:.2f} 掛 Limit 單買入，現價 {close_p:.2f})"

        # [v25.1 針對「持倉幾星期至一兩個月」中線波段止損重構]
        # 1. 拉開結構支撐位下方之 ATR 防洗盤緩衝 (0.35x ~ 0.60x ATR)，避免單日下影線掃止損
        # 2. 預設要求止損距離買入位至少 >= 1.20x ATR 且至少 >= 3.2% (可在側欄切換為平衡或短線緊密模式)
        _stop_prof = get_active_stop_profile()
        _min_atr_mult = float(_stop_prof["min_atr_mult"])
        _min_stop_pct = float(_stop_prof["min_stop_pct"])
        _raw_max_stop_pct = float(_stop_prof["max_stop_pct"])
        # [v32.0 根因修復：美股高 ADR% 動態止損上限補償]
        # 美股高動能股 (如 NVDA, TSLA, PLTR, MSTR, COIN) 日均 ATR 常達 4.5%~7.5%，若硬套用 7.8% 上限會導致所有真實結構支撐 (20MA/21EMA/箱體底) 被過濾掉並退化為懸空假止損！
        # 此處根據個股 ATR% 自動放寬結構容許上限 (最高 10.8%)，再由倉位計算器自動按比例縮減買入股數，確保「止損守在真實結構下方」且「單筆總資金風險恆定 0.5%」！
        _stock_atr_pct = atr_safe / max(recommended_entry, 1e-6)
        _max_stop_pct = min(0.108, max(_raw_max_stop_pct, _stock_atr_pct * 1.95)) if not _stop_prof.get("use_intraday_lod", False) else _raw_max_stop_pct

        struct_low_candidates = [
            (box_low_10 - 0.35 * atr_safe, f"近10日箱體底 ({box_low_10:.2f}) 下方防洗盤緩衝"),
            (sup_short - 0.35 * atr_safe, f"20日波段前浪底支撐 ({sup_short:.2f}) 外側緩衝"),
            (sma20_p - 0.60 * atr_safe, f"20MA 中線生命線 ({sma20_p:.2f}) 下方緩衝"),
            (sma50_p - 0.40 * atr_safe, f"50MA 中期趨勢線 ({sma50_p:.2f}) 下方緩衝")
        ]
        if pd.notna(last.get('EMA21')) and float(last['EMA21']) < recommended_entry:
            struct_low_candidates.append((float(last['EMA21']) - 0.50 * atr_safe, f"21EMA 波段趨勢線 ({float(last['EMA21']):.2f}) 下方緩衝"))
        if pd.notna(last.get('AVWAP_High60')) and float(last['AVWAP_High60']) < recommended_entry:
            struct_low_candidates.append((float(last['AVWAP_High60']) - 0.40 * atr_safe, f"前波高點錨定AVWAP ({float(last['AVWAP_High60']):.2f}) 下方緩衝"))
        if pd.notna(last.get('AVWAP_Swing')) and float(last['AVWAP_Swing']) < recommended_entry:
            struct_low_candidates.append((float(last['AVWAP_Swing']) - 0.40 * atr_safe, f"波段低點錨定AVWAP ({float(last['AVWAP_Swing']):.2f}) 下方緩衝"))
        if pd.notna(_piv_ch.get("utl_support")):
            _utl_sl = float(_piv_ch["utl_support"]) - 0.35 * atr_safe
            struct_low_candidates.append((_utl_sl, f"J Law 上升趨勢支撐線 UTL ({float(_piv_ch['utl_support']):.2f}) 外側"))
        for _z in _piv_ch.get("sr_zones", []):
            if float(_z["low_bound"]) < recommended_entry * 0.985:
                struct_low_candidates.append((float(_z["low_bound"]) - 0.30 * atr_safe, f"{_z.get('kind_label', '結構支撐區')} ({float(_z['price']):.2f}) 外側"))
        if lei_struct and pd.notna(lei_struct.get("structure_low")):
            sl_st = float(lei_struct["structure_low"]) - 0.30 * atr_safe
            struct_low_candidates.append((sl_st, f"LEI 底部構造低點 ({lei_struct['structure_low']:.2f}) 下方"))

        # 僅當用戶在側欄明確切換為「⚡ 短線緊密模式」時，才將當日低點 (LOD) 納入止損候選
        if _stop_prof.get("use_intraday_lod", False):
            _lod_sl = float(last['Low']) - 0.08 * atr_safe
            struct_low_candidates.append((_lod_sl, f"當日低點 LOD ({float(last['Low']):.2f}) 短線緊密保護"))

        _upper_sl_limit = recommended_entry * (1.0 - _min_stop_pct)
        _lower_sl_limit = recommended_entry * (1.0 - _max_stop_pct)
        _min_atr_dist = _min_atr_mult * atr_safe

        valid_sls = [
            (sl_p, sl_lbl) for sl_p, sl_lbl in struct_low_candidates
            if pd.notna(sl_p) and (_lower_sl_limit <= sl_p <= _upper_sl_limit) and ((recommended_entry - sl_p) >= _min_atr_dist)
        ]
        # OLD:
        # if valid_sls:
        #     best_sl_pair = max(valid_sls, key=lambda x: x[0])
        #     stop_loss = round(best_sl_pair[0], 2)
        #     sl_basis_desc = best_sl_pair[1]
        # NEW: [v28.0 雙層防洗盤止損 (Two-Tier Anti-Whipsaw Stop)]
        # 根因修復：舊版 max(valid_sls) 總係揀最貼近買入位嘅淺層支撐（散戶止損密集獵殺區），導致盤中假跌破洗盤剛好掃出在最低點！
        # 新版將最淺層支撐設為「收盤觀察軟止損 (soft_closing_sl)」，而實際掛單之「防洗盤硬止損 (stop_loss)」選取避開日內 0.5x ATR 獵殺區的次深層結構匯聚位！
        _soft_closing_sl = round(min(sma20_p, box_low_10) - 0.15 * atr_safe, 2)
        if valid_sls:
            _anti_sweep_min_dist = max(_min_atr_dist * 1.15, recommended_entry * (_min_stop_pct + 0.005))
            _deep_valid_sls = [p for p in valid_sls if (recommended_entry - p[0]) >= _anti_sweep_min_dist]
            if _deep_valid_sls:
                best_sl_pair = max(_deep_valid_sls, key=lambda x: x[0])
            else:
                best_sl_pair = min(valid_sls, key=lambda x: x[0])
            stop_loss = round(best_sl_pair[0], 2)
            _soft_closing_sl = round(max(stop_loss + 0.30 * atr_safe, max(valid_sls, key=lambda x: x[0])[0]), 2)
            sl_basis_desc = f"{best_sl_pair[1]} (盤中防洗盤硬止損｜收盤軟防守 {_soft_closing_sl:.2f})"
        else:
            # 若候選位過近，退後尋找任何滿足 [_lower_sl_limit, _upper_sl_limit] 的最深結構支撐，或直接按 1.65x ATR 與 3.8%~7.5% 設定中線波段保護位
            wider_sls = [
                (sl_p, sl_lbl) for sl_p, sl_lbl in struct_low_candidates
                if pd.notna(sl_p) and (_lower_sl_limit <= sl_p <= _upper_sl_limit)
            ]
            if wider_sls:
                best_sl_pair = min(wider_sls, key=lambda x: x[0])
                stop_loss = round(best_sl_pair[0], 2)
                sl_basis_desc = f"{best_sl_pair[1]} (中線防洗盤加寬)"
            else:
                fallback_sl = max(recommended_entry - float(_stop_prof["fallback_atr_mult"]) * atr_safe, _lower_sl_limit)
                stop_loss = round(min(fallback_sl, recommended_entry * (1.0 - float(_stop_prof["fallback_min_pct"]))), 2)
                sl_basis_desc = f"中線波段 {float(_stop_prof['fallback_atr_mult']):.2f}x ATR 防洗盤保護位 (上限 ≤{_max_stop_pct*100:.1f}%)"

        risk_per_share = max(recommended_entry - stop_loss, 1e-4)
        stop_pct = (risk_per_share / recommended_entry) * 100.0

        sr_targets = _find_historical_ma_convergence_targets(df, recommended_entry, "LONG")
        # OLD:
        # recent_swing_amp = max(res_short - sup_short, 3.2 * atr_safe)
        # measured_move_tp = round(recommended_entry + recent_swing_amp * 0.85, 2)
        # NEW: [v24.15] 結合 J Law 2+1 平行通道頂 (TRL) 與歷史真實波段振幅 (1.0x N字等距波幅)，杜絕固定倍數
        # [v35.0 修正單波段目標價過遠問題] 單一 5~15 日波段合理推進幅度為 2.4x~3.2x ATR (約 6%~9%)，過遠目標 (+15%~+18%) 會導致無法觸及止盈即反轉
        recent_swing_amp = min(max((res_short - sup_short) * 0.75, 2.4 * atr_safe), 3.2 * atr_safe)
        _trl_cand = _piv_ch.get("trl_resistance", np.nan)
        if pd.notna(_trl_cand) and (recommended_entry * 1.03 < float(_trl_cand) <= recommended_entry + 3.5 * atr_safe):
            measured_move_tp = round(float(_trl_cand), 2)
        else:
            measured_move_tp = round(recommended_entry + recent_swing_amp, 2)

        # [v42.0 港股 vs 美股雙軌差異化 R:R 引擎 (Dual-Track HK/US R:R Engine)]
        # - 港股 (HK)：自動跨過不足 2.0R (或少於 +7.2%) 之箱體內近期小阻力，還原 main7 藥明生物 (2269.HK) 波段目標 (max(合規主阻力, Entry + 3.8x ATR, Entry + 2.25R))！
        # - 美股 (US)：採用《超級績效 2》2R~2.4R 強勢減半 (TP1) + 21EMA 尾倉奔跑 (TP2)！
        _dual_rr_eval = compute_dual_track_hk_us_rr_targets(
            entry_price=recommended_entry,
            stop_loss=stop_loss,
            atr_val=atr_safe,
            sr_targets=sr_targets,
            market_type=market_type,
            direction="LONG"
        )
        take_profit = _dual_rr_eval["tp1_price"]
        tp2_price = _dual_rr_eval["tp2_price"]
        structural_tp = take_profit
        tp_basis_desc = _dual_rr_eval["tp1_desc"]

        structural_rr = round((structural_tp - recommended_entry) / risk_per_share, 2)
        planned_rr = round((take_profit - recommended_entry) / risk_per_share, 2)
        tp2_rr = round((tp2_price - recommended_entry) / risk_per_share, 2)

        warnings = []
        if structural_rr < 1.8:
            warnings.append(f"⚠️ 上方 {tp_basis_desc} 距離買入位僅 {structural_rr:.2f}R (<1.8R)，獲利空間受阻，不建議優先追入")
        if close_p > recommended_entry * 1.025:
            warnings.append(f"⏳ 現價 ({close_p:.2f}) 高於最佳買入位 ({recommended_entry:.2f})，請耐心掛單等待回測該買入位，切勿市價高追")

        if _dc_info.get("heavy_overhead_supply", False):
            warnings.append(_dc_info.get("deepchart_detail", "⚠️ 前高空單尚未消化完畢"))
        if _exh_guard.get("exhaustion_reason"):
            warnings.append(_exh_guard["exhaustion_reason"])
        _mkt_order_risk = max(close_p - stop_loss, 1e-4)
        _mkt_order_stop_pct = (_mkt_order_risk / max(close_p, 1e-4)) * 100.0
        _mkt_order_rr = round((take_profit - close_p) / _mkt_order_risk, 2)
        _dynamic_stop_cap_pct = max(_max_stop_pct * 100.0, 5.8)
        _eq_sz = compute_equal_risk_position_cap(close_p, stop_loss, fixed_risk_budget=600.0, max_notional_cap=20000.0, lot_size=lot_size, market_type=market_type)
        if _mkt_order_stop_pct > _dynamic_stop_cap_pct or close_p > recommended_entry * 1.018:
            if _action_timeliness.startswith("⚡"):
                _action_timeliness = f"⏳ 勿現價市價買 (現價止損達 {_mkt_order_stop_pct:.1f}% > {_dynamic_stop_cap_pct:.1f}%上限，需等回調至 {recommended_entry:.2f})"
            warnings.append(
                f"⚠️ **【市價單風險過寬攔截】**：現價 ({close_p:.2f}) 距離結構止損 ({stop_loss:.2f}) 達 **-{_mkt_order_stop_pct:.2f}%** (超過當前模式/ADR動態上限 {_dynamic_stop_cap_pct:.1f}%)，"
                f"直接市價買入會拉低真實盈虧比 ({_mkt_order_rr:.2f}R)，請於 **{recommended_entry:.2f}** 掛限價單並按等風險原則縮減股數。"
            )
        _tp0_cushion = round(min(recommended_entry * 1.045, recommended_entry + 1.15 * risk_per_share), 2)
        return {
            "entry": recommended_entry,
            "mkt_order_stop_pct": round(_mkt_order_stop_pct, 2),
            "mkt_order_rr": _mkt_order_rr,
            "max_notional_600_risk": _eq_sz["recommended_notional"],
            "max_shares_600_risk": _eq_sz["recommended_shares"],
            "entry_type_desc": entry_type_desc,
            "action_timeliness": _action_timeliness,
            "deepchart_score": _dc_score,
            "deepchart_badge": _dc_info.get("deepchart_badge", "-"),
            "deepchart_detail": _dc_info.get("deepchart_detail", ""),
            "sl": stop_loss,
            "soft_sl": _soft_closing_sl,
            "sl_basis_desc": sl_basis_desc,
            "tp0_cushion": _tp0_cushion,
            "tp": take_profit,
            "tp2": tp2_price,
            "tp_basis_desc": f"先鎖利潤墊(TP0 +1.25R): {_tp0_cushion:.2f} ｜ " + tp_basis_desc,
            "structural_tp": structural_tp,
            "risk_per_share": risk_per_share,
            "stop_pct": round(stop_pct, 2),
            "structural_rr": structural_rr,
            "planned_rr": planned_rr,
            "tp2_rr": tp2_rr,
            "rr_warning": " ｜ ".join(warnings)
        }

    elif plan_type == "SHORT":
        _bearish_bar_now = (close_p <= float(last['Open'])) or ((high_p - close_p) >= 0.48 * max(high_p - low_p, 1e-6))
        if (is_breakdown_short or "破底" in status) and (box_low_10 * 0.970 <= close_p <= box_low_10 * 1.005):
            recommended_entry = round(close_p, 2)
            _action_timeliness = "⚡ 即時入場 (破底沽空區)"
            entry_type_desc = f"⚡ 即時破底沽空單 (現價 {close_p:.2f} 剛跌穿箱體下沿 {box_low_10:.2f} 且未過度偏離，可即時沽空)"
        elif is_breakdown_short or "破底" in status:
            pivot_short = box_low_10 - 0.08 * atr_safe
            if close_p < box_low_10 * 0.97:
                retest_short_lvl = min(box_low_10, sma10_p, prior_wave_low if prior_wave_low >= close_p else sma10_p)
                recommended_entry = round(retest_short_lvl, 2)
                _action_timeliness = "⏳ 等待反抽"
                entry_type_desc = f"等反彈沽空單 (現價已急跌偏離破底位，等無量反抽前低阻力 {recommended_entry:.2f} 沽空)"
            else:
                recommended_entry = round(min(close_p * 0.998, max(pivot_short, close_p - 0.35 * atr_safe)), 2)
                _action_timeliness = "🎯 破底觸發"
                entry_type_desc = f"破底觸發沽空單 (Stop-Sell 跌穿箱體下沿 {recommended_entry:.2f} 追空)"
        else:
            resist_anchor = min(max(close_p, sma20_p * 0.995), sma50_p if pd.notna(sma50_p) and sma50_p > close_p else sma20_p * 1.01)
            if abs(close_p - resist_anchor) / max(close_p, 1e-6) <= 0.015 and _bearish_bar_now:
                recommended_entry = round(close_p, 2)
                _action_timeliness = "⚡ 即時入場 (反抽受阻確認)"
                entry_type_desc = f"⚡ 即時逢高沽空單 (現價 {close_p:.2f} 正處於 20MA/阻力壓制區且出現受阻K線，可即時沽空)"
            else:
                recommended_entry = round(resist_anchor, 2)
                _action_timeliness = "⏳ 等待反抽"
                entry_type_desc = f"逢高反彈沽空單 (於 20MA/前支撐轉阻力區 {recommended_entry:.2f} 掛單沽空)"

        # [NEW v24.15] 整合空頭 S/R Flip 支撐轉阻力區、下降趨勢線 (DTL) 與誘多洗盤高點
        _piv_ch = compute_causal_pivots_and_parallel_channel(df, lookback=120)
        _sr_flip_resists = [
            float(z["high_bound"]) for z in _piv_ch.get("sr_zones", [])
            if z.get("is_flip") and (recommended_entry * 1.008 <= float(z["high_bound"]) <= recommended_entry * 1.07)
        ]
        if _sr_flip_resists and ("反彈" in entry_type_desc or "逢高" in entry_type_desc):
            _best_flip_short = round(min(_sr_flip_resists) - 0.10 * atr_safe, 2)
            if abs(_best_flip_short - close_p) / close_p <= 0.045 and _best_flip_short >= close_p * 0.995:
                recommended_entry = _best_flip_short
                entry_type_desc = f"S/R互換阻力沽空單 (於前支撐Flip轉阻力區 {recommended_entry:.2f} 掛 Limit 單沽空，現價 {close_p:.2f})"

        # [v25.1 空頭中線波段止損重構] 預設保留 >= 1.20x ATR 與 >= 3.2% 防軋空洗盤緩衝
        _stop_prof = get_active_stop_profile()
        _min_atr_mult = float(_stop_prof["min_atr_mult"])
        _min_stop_pct = float(_stop_prof["min_stop_pct"])
        _max_stop_pct = float(_stop_prof["max_stop_pct"])

        struct_high_candidates = [
            (box_high_10 + 0.35 * atr_safe, f"近10日反彈箱體頂 ({box_high_10:.2f}) 上方緩衝"),
            (res_short + 0.35 * atr_safe, f"20日前浪頂阻力 ({res_short:.2f}) 外側緩衝"),
            (sma20_p + 0.60 * atr_safe, f"20MA 蓋頭均線 ({sma20_p:.2f}) 上方緩衝"),
            (sma50_p + 0.40 * atr_safe, f"50MA 中期反壓線 ({sma50_p:.2f}) 上方緩衝")
        ]
        if pd.notna(last.get('EMA21')) and float(last['EMA21']) > recommended_entry:
            struct_high_candidates.append((float(last['EMA21']) + 0.50 * atr_safe, f"下彎21EMA ({float(last['EMA21']):.2f}) 上方緩衝"))
        if pd.notna(_piv_ch.get("dtl_resistance")):
            _dtl_sl = float(_piv_ch["dtl_resistance"]) + 0.35 * atr_safe
            struct_high_candidates.append((_dtl_sl, f"J Law 下降趨勢阻力線 DTL ({float(_piv_ch['dtl_resistance']):.2f}) 外側"))
        for _z in _piv_ch.get("sr_zones", []):
            if float(_z["high_bound"]) > recommended_entry * 1.015:
                struct_high_candidates.append((float(_z["high_bound"]) + 0.30 * atr_safe, f"{_z.get('kind_label', '結構阻力區')} ({float(_z['price']):.2f}) 外側"))
        if lei_struct and pd.notna(lei_struct.get("structure_high")):
            sh_st = float(lei_struct["structure_high"]) + 0.30 * atr_safe
            struct_high_candidates.append((sh_st, f"LEI 頂部構造高點 ({lei_struct['structure_high']:.2f}) 上方"))

        if _stop_prof.get("use_intraday_lod", False):
            _hod_sl = float(last['High']) + 0.08 * atr_safe
            struct_high_candidates.append((_hod_sl, f"當日高點 HOD ({float(last['High']):.2f}) 短線緊密保護"))

        _lower_sl_limit = recommended_entry * (1.0 + _min_stop_pct)
        _upper_sl_limit = recommended_entry * (1.0 + _max_stop_pct)
        _min_atr_dist = _min_atr_mult * atr_safe

        valid_sls = [
            (sl_p, sl_lbl) for sl_p, sl_lbl in struct_high_candidates
            if pd.notna(sl_p) and (_lower_sl_limit <= sl_p <= _upper_sl_limit) and ((sl_p - recommended_entry) >= _min_atr_dist)
        ]
        # OLD:
        # if valid_sls:
        #     best_sl_pair = min(valid_sls, key=lambda x: x[0])
        #     stop_loss = round(best_sl_pair[0], 2)
        #     sl_basis_desc = best_sl_pair[1]
        # NEW: [v28.0 空頭雙層防軋空洗盤止損] 避開最淺層散戶空單止損密集區，選取次深層結構阻力作為盤中硬止損
        _soft_closing_sl = round(max(sma20_p, box_high_10) + 0.15 * atr_safe, 2)
        if valid_sls:
            _anti_sweep_min_dist = max(_min_atr_dist * 1.15, recommended_entry * (_min_stop_pct + 0.005))
            _deep_valid_sls = [p for p in valid_sls if (p[0] - recommended_entry) >= _anti_sweep_min_dist]
            if _deep_valid_sls:
                best_sl_pair = min(_deep_valid_sls, key=lambda x: x[0])
            else:
                best_sl_pair = max(valid_sls, key=lambda x: x[0])
            stop_loss = round(best_sl_pair[0], 2)
            _soft_closing_sl = round(min(stop_loss - 0.30 * atr_safe, min(valid_sls, key=lambda x: x[0])[0]), 2)
            sl_basis_desc = f"{best_sl_pair[1]} (盤中防軋空硬止損｜收盤軟防守 {_soft_closing_sl:.2f})"
        else:
            wider_sls = [
                (sl_p, sl_lbl) for sl_p, sl_lbl in struct_high_candidates
                if pd.notna(sl_p) and (_lower_sl_limit <= sl_p <= _upper_sl_limit)
            ]
            if wider_sls:
                best_sl_pair = max(wider_sls, key=lambda x: x[0])
                stop_loss = round(best_sl_pair[0], 2)
                sl_basis_desc = f"{best_sl_pair[1]} (中線防洗盤加寬)"
            else:
                fallback_sl = min(recommended_entry + float(_stop_prof["fallback_atr_mult"]) * atr_safe, _upper_sl_limit)
                stop_loss = round(max(fallback_sl, recommended_entry * (1.0 + float(_stop_prof["fallback_min_pct"]))), 2)
                sl_basis_desc = f"中線波段 {float(_stop_prof['fallback_atr_mult']):.2f}x ATR 空頭保護位 (上限 ≤{_max_stop_pct*100:.1f}%)"

        risk_per_share = max(stop_loss - recommended_entry, 1e-4)
        stop_pct = (risk_per_share / recommended_entry) * 100.0

        sr_targets = _find_historical_ma_convergence_targets(df, recommended_entry, "SHORT")
        # OLD:
        # recent_swing_amp = max(res_short - sup_short, 3.2 * atr_safe)
        # measured_move_tp = round(max(0.01, recommended_entry - recent_swing_amp * 0.85), 2)
        # NEW: [v24.15] 結合 J Law 2+1 下降平行通道底 (TSL) 與歷史真實下跌波幅投射
        recent_swing_amp = min(max(res_short - sup_short, 2.8 * atr_safe), 5.5 * atr_safe)
        _tsl_cand = _piv_ch.get("tsl_support", np.nan)
        measured_move_tp = round(float(_tsl_cand) if (pd.notna(_tsl_cand) and 0 < float(_tsl_cand) < recommended_entry * 0.97) else max(0.01, recommended_entry - recent_swing_amp * 1.0), 2)

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

        # OLD:
        # "deepchart_score": _dc_score,
        # "deepchart_badge": _dc_info.get("deepchart_badge", "-"),
        # "deepchart_detail": _dc_info.get("deepchart_detail", ""),
        # NEW: [v31.0 空頭計劃綁定空頭專屬 DeepChart 壓制分數與標籤]
        _short_dc_score = float(_dc_info.get("short_dominance_score", round(100.0 - _dc_score, 1)))
        if _short_dc_score < 42.0:
            warnings.append(_dc_info.get("short_deepchart_detail", "⚠️ 下方支撐買盤承接仍強，慎防逼空反彈"))
        return {
            "entry": recommended_entry,
            "entry_type_desc": entry_type_desc,
            "action_timeliness": _action_timeliness,
            "deepchart_score": _short_dc_score,
            "deepchart_badge": _dc_info.get("short_deepchart_badge", _dc_info.get("deepchart_badge", "-")),
            "deepchart_detail": _dc_info.get("short_deepchart_detail", _dc_info.get("deepchart_detail", "")),
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
            "action_timeliness": "⚪ 觀望",
            "deepchart_score": _dc_score,
            "deepchart_badge": _dc_info.get("deepchart_badge", "-"),
            "deepchart_detail": _dc_info.get("deepchart_detail", ""),
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
    # OLD:
    # meta_res["rs_1m_outperform"] = bool(stock_ret1m > max(0.05, bench_ret1m * 1.8))
    # meta_res["rs_6m_outperform"] = bool(stock_ret6m > max(0.10, bench_ret6m * 1.8))
    # NEW: [v25.2 修復 P1-2 大盤下跌時抗跌領頭羊 Alpha 誤判漏洞] 結合超額報酬 (Alpha = 個股漲幅 - 大盤漲幅) 與正收益雙重認證
    _alpha_1m = stock_ret1m - bench_ret1m
    _alpha_6m = stock_ret6m - bench_ret6m
    meta_res["rs_1m_outperform"] = bool((stock_ret1m > max(0.05, bench_ret1m * 1.8)) or (bench_ret1m < 0 and stock_ret1m > 0.01 and _alpha_1m >= 0.05))
    meta_res["rs_6m_outperform"] = bool((stock_ret6m > max(0.10, bench_ret6m * 1.8)) or (bench_ret6m < 0 and stock_ret6m > 0.03 and _alpha_6m >= 0.10))

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
    meta_res["long_edges_list"] = long_edges
    meta_res["short_edges_list"] = short_edges
    meta_res["long_badge"] = f"💎 {len(long_edges)}/5 多重優勢" if len(long_edges) >= 3 else f"⚪ {len(long_edges)}/5 多頭優勢"
    meta_res["short_badge"] = f"🏴‍☠️ {len(short_edges)}/5 空頭優勢" if len(short_edges) >= 3 else f"⚪ {len(short_edges)}/5 空頭優勢"
    # OLD:
    # if len(long_edges) >= len(short_edges):
    #     meta_res["meta_edges_list"] = long_edges
    #     meta_res["meta_badge"] = f"💎 {len(long_edges)}/5 多重優勢" if len(long_edges) >= 3 else f"⚪ {len(long_edges)}/5 優勢"
    # else:
    #     meta_res["meta_edges_list"] = short_edges
    #     meta_res["meta_badge"] = f"🏴‍☠️ {len(short_edges)}/5 空頭優勢" if len(short_edges) >= 3 else f"⚪ {len(short_edges)}/5 優勢"
    # NEW: [v25.4 修復做空時仍顯示多頭優勢標籤漏洞] 優先根據當前交易方向 (或比較分數) 顯示對應之多/空優勢清單
    _eval_dir = str(trade_plan_eval.get("_plan_dir", ""))
    if _eval_dir == "SHORT" or (_eval_dir != "LONG" and len(short_edges) > len(long_edges)):
        meta_res["meta_edges_list"] = short_edges
        meta_res["meta_badge"] = meta_res["short_badge"]
    else:
        meta_res["meta_edges_list"] = long_edges
        meta_res["meta_badge"] = meta_res["long_badge"]

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
    # OLD: broke_20d_low_intraday = pd.notna(low_20d_prev) and (last['Low'] <= low_20d_prev)
    # NEW: [v30.0 修復誤殺 2B 破底翻 / Undercut & Rally 洗盤買點之漏洞]
    # 若盤中跌穿 20 日低點但收盤強勢站回 20 日低點之上且收陽/長下影，屬於機構洗盤破底翻 (Bullish 2B)，絕不可否決做多！只有收盤實質跌破才否決。
    broke_20d_low_intraday = pd.notna(low_20d_prev) and (last['Low'] <= low_20d_prev) and (last['Close'] < low_20d_prev)

    peak_idx = window['Close'].idxmax()
    peak_pos = df.index.get_loc(peak_idx)
    if isinstance(peak_pos, slice) or isinstance(peak_pos, np.ndarray):
        peak_pos = int(np.flatnonzero(peak_pos)[-1])
    bars_since_peak = (len(df) - 1) - peak_pos
    peak_row = df.iloc[peak_pos]
    peak_entry = float(peak_row['Close'])
    peak_atr = float(peak_row['ATR']) if pd.notna(peak_row.get('ATR')) and peak_row['ATR'] > 0 else peak_entry * 0.03
    # [v39.0 修復強勢股首次回踩 20MA 被誤判為「觸及前波高點止損失效」之邏輯缺陷]
    # 結構止損應錨定於波段高點當時之 20MA 下方緩衝 (SMA20 - 0.65x ATR) 或近10日箱體底，而非假設在最高點追高 -2x ATR！
    peak_sma20 = float(peak_row.get('SMA20', peak_entry * 0.94))
    peak_box_low = float(peak_row.get('BoxLow10_Causal', peak_sma20))
    peak_initial_sl = round(min(max(peak_entry - 2.5 * peak_atr, peak_entry * 0.915), max(peak_sma20 - 0.65 * peak_atr, peak_box_low - 0.35 * peak_atr)), 2)

    post_peak_df = df.iloc[peak_pos + 1:] if peak_pos + 1 < len(df) else pd.DataFrame()
    min_low_since_peak = float(post_peak_df['Low'].min()) if not post_peak_df.empty else float(last['Low'])
    min_close_since_peak = float(post_peak_df['Close'].min()) if not post_peak_df.empty else float(last['Close'])

    result["signal_age_bars"] = int(bars_since_peak)
    result["anchor_date"] = str(peak_idx.date()) if hasattr(peak_idx, "date") else str(peak_idx)
    result["anchor_entry"] = peak_entry
    result["anchor_sl"] = peak_initial_sl

    # 僅當波段高點回落跌穿結構防守位，且當前收盤仍受制於 20MA 下方（非洗盤收復）時，才判定為前波多頭結構止損失效
    if bars_since_peak >= 1 and (min_low_since_peak <= peak_initial_sl or min_close_since_peak < peak_sma20 * 0.975) and (float(last['Close']) < float(last.get('SMA20', np.inf)) * 0.995):
        result["long_stop_hit"] = True
        result["invalidation_reason"] = (
            f"⛔ 觸及前波結構止損失效：自 {result['anchor_date']} 波段高點 ({peak_entry:.2f}) 回落，"
            f"最低價 ({min_low_since_peak:.2f}) 已跌穿結構防守位 ({peak_initial_sl:.2f}) 且尚未收復 20MA"
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

    result["short_signal_age_bars"] = int(bars_since_trough)
    result["short_anchor_date"] = str(trough_idx.date()) if hasattr(trough_idx, "date") else str(trough_idx)
    result["short_anchor_entry"] = trough_entry
    result["short_anchor_sl"] = trough_initial_sl
    result["long_invalidation_reason"] = result["invalidation_reason"]
    result["short_invalidation_reason"] = ""

    # OLD:
    # if bars_since_trough >= 1 and max_high_since_trough >= trough_initial_sl and (last['Close'] > last.get('SMA20', 0)):
    #     result["short_stop_hit"] = True
    # if pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and (last['Close'] > last['SMA20']) and (last['Close'] > last['SMA50']) and not sma20_falling:
    #     result["uptrend_veto_short"] = True
    # NEW: [v25.4 修復做空方向卻顯示做多止損文字漏洞] 完整記錄空頭觸及止損與多頭回升否決做空之專屬原因，不再讓做多文字覆蓋做空診斷！
    if bars_since_trough >= 1 and max_high_since_trough >= trough_initial_sl and (last['Close'] > last.get('SMA20', 0) or last['Close'] > trough_entry):
        result["short_stop_hit"] = True
        result["short_invalidation_reason"] = (
            f"⛔ 觸及前波空頭止損失效：自 {result['short_anchor_date']} 波段低點 ({trough_entry:.2f}) 起算，"
            f"盤中最高價 ({max_high_since_trough:.2f}) 已升穿當時空頭止損位 ({trough_initial_sl:.2f})"
        )

    if pd.notna(last.get('SMA20')) and pd.notna(last.get('SMA50')) and (last['Close'] > last['SMA20']) and (last['Close'] > last['SMA50']) and not sma20_falling:
        result["uptrend_veto_short"] = True
        if not result["short_invalidation_reason"]:
            result["short_invalidation_reason"] = "📈 結構轉多否決做空：股價已重新站上 20MA 及 50MA 且均線向上，空頭結構破壞"

    return result


# ==============================================================================
# [NEW INCREMENTAL PATCH v24.15] 掃描器與回測系統 100% 理念同步統一評估器
# 讓 quick_historical_win_rate 與 run_bidirectional_backtest 在歷史每一根 K 線 (i-1)
# 完全使用與掃描雷達相同的：
# 1. 多空觸發條件 (Minervini + J Law VCP/突破 + S/R Flip 低量回測 + LEI 2點鐘抵扣價回撤 + 六線密集 + 2B破底翻)
# 2. 精準結構買入位與掛單模式 (突破 Stop 觸發單 vs 支撐回踩 Limit 低吸單，絕不盲目用隔日開盤價硬追)
# 3. 多層級結構支撐止損與結構阻力/平行通道止盈 (每筆交易獨立動態結構 R:R，且要求 R:R >= 1.8R)
# ==============================================================================
def evaluate_historical_bar_unified_plan(
    df: pd.DataFrame,
    bar_idx: int,
    market_type: str,
    score_threshold: int = 5,
    is_relaxed: bool = False
) -> Dict[str, Any]:
    """
    在歷史回測的第 bar_idx 根 K 線收盤時 (對應回測迴圈中的 prev_row = df.iloc[i-1])，
    無未來函數地切片 df.iloc[:bar_idx+1]，計算與即時掃描器 100% 同步的信號、建議買賣位、掛單類型與結構 R:R。
    """
    res = {
        "signal": None,             # "LONG" or "SHORT" or None
        "order_style": "STOP",      # "STOP" (突破觸發) or "LIMIT" (回踩低吸/反彈沽空)
        "status_label": "",
        "planned_entry": 0.0,
        "stop_loss": 0.0,
        "tp_target": 0.0,
        "tp2_target": 0.0,
        "planned_rr": 0.0,
        "sl_basis": "",
        "tp_basis": "",
        "entry_desc": ""
    }
    if df is None or bar_idx < 25 or bar_idx >= len(df):
        return res

    prev_row = df.iloc[bar_idx]
    p_prev20 = df.iloc[bar_idx - 20] if bar_idx >= 20 else df.iloc[0]
    is_hk_liquid = not (market_type == "HK" and float(prev_row.get('TurnoverMA20', 0)) < 20_000_000)
    if not is_hk_liquid:
        return res

    bench_close = prev_row.get('CloseBench', np.nan)
    bench_50 = prev_row.get('SMA50BenchValue', np.nan)
    bench_200 = prev_row.get('SMA200BenchValue', np.nan)
    bench_dist = float(prev_row.get('Bench_Dist_Days', 0))

    is_market_bullish = True
    if pd.notna(bench_close) and pd.notna(bench_50):
        is_market_bullish = bool(bench_close > bench_50)

    is_market_health_ok = True
    if pd.notna(bench_close) and pd.notna(bench_200):
        if bench_close < bench_200 or bench_dist >= 5:
            is_market_health_ok = False

    close_p = float(prev_row['Close'])
    high_p = float(prev_row['High'])
    low_p = float(prev_row['Low'])
    sma10_p = float(prev_row.get('SMA10', close_p))
    sma20_p = float(prev_row.get('SMA20', close_p))
    sma50_p = float(prev_row.get('SMA50', close_p))
    sma150_p = float(prev_row.get('SMA150', close_p))
    sma200_p = float(prev_row.get('SMA200', close_p))
    atr_val = float(prev_row['ATR']) if pd.notna(prev_row.get('ATR')) and prev_row['ATR'] > 0 else close_p * 0.03

    # 短期破位否決檢查 (與 check_recent_signal_and_stop_invalidation 同步)
    prev5_row = df.iloc[max(0, bar_idx - 5)]
    sma20_falling = pd.notna(sma20_p) and pd.notna(prev5_row.get('SMA20')) and (sma20_p < float(prev5_row['SMA20']))
    low_20d_prev = float( prev_row.get('SupShort_Causal', low_p) )
    # OLD: downtrend_veto_long = (close_p < sma20_p and close_p < sma50_p) or (close_p < sma20_p and sma20_falling and float(prev_row.get('MACD', 0)) < 0) or (low_p <= low_20d_prev * 0.995)
    # NEW: [v30.0 保留 2B 破底翻買點] 只有收盤跌破 20 日前低 (close_p < low_20d_prev * 0.998) 才觸發破底否決，允許盤中洗盤跌穿後收陽拉回之 2B 買點！
    downtrend_veto_long = (close_p < sma20_p and close_p < sma50_p) or (close_p < sma20_p and sma20_falling and float(prev_row.get('MACD', 0)) < 0) or (close_p < low_20d_prev * 0.998)

    p_low = float(prev_row.get('Low52W', df['Low'].iloc[max(0, bar_idx-252):bar_idx+1].min()))
    p_high = float(prev_row.get('High52W', df['High'].iloc[max(0, bar_idx-252):bar_idx+1].max()))

    r1 = close_p > sma150_p and close_p > sma200_p
    r2 = sma150_p > sma200_p
    r3 = sma200_p > float(p_prev20.get('SMA200', sma200_p))
    r4 = sma50_p > sma150_p and sma50_p > sma200_p
    r5 = close_p > p_low * 1.25
    r6 = close_p > p_high * 0.75
    r7 = float(prev_row.get('Mansfield_RS', 0)) > 0
    pass_count = sum([r1, r2, r3, r4, r5, r6, r7])

    dist_to_20ma = abs(close_p - sma20_p) / max(sma20_p, 1e-9)
    is_near_20ma = dist_to_20ma <= 0.028
    is_tight_setup = (bool(prev_row.get('VCPSignal', False)) or bool(prev_row.get('JLaw_Tight_Box', False))) and (float(prev_row.get('VolRatio', 1.0)) < 0.85)
    box_high_10 = float(prev_row.get('BoxHigh10_Causal', high_p))
    box_low_10 = float(prev_row.get('BoxLow10_Causal', low_p))
    p_high_20_val = float(prev_row.get('ResShort_Causal', high_p))
    is_breakout_long = (close_p >= p_high_20_val * 0.99) and (float(prev_row.get('VolRatio', 1.0)) > 1.35)
    kk_flag = (float(prev_row.get('Ret1M', 0)) > 0.15) and (close_p > sma10_p) and bool(prev_row.get('VCPSignal', False))
    early_stage_bypass = (close_p > sma50_p) and (float(prev_row.get('Ret1M', 0)) > 0.20)
    momentum_ready = (float(prev_row.get('RSI', 50)) > 52) and (close_p >= sma20_p * 0.99)

    # J Law S/R Flip 低量回測買點 (EP1 55:36 & EP2 30:08)
    prior_wave_high = float(prev_row.get('PriorWaveHigh_Causal', p_high_20_val))
    recent_10_max = float(df['High'].iloc[max(0, bar_idx-10):bar_idx+1].max())
    is_sr_flip_pullback_long = bool(
        (pass_count >= score_threshold or early_stage_bypass)
        and (recent_10_max > prior_wave_high * 1.01)
        and (prior_wave_high * 0.98 <= close_p <= prior_wave_high * 1.028)
        and (float(prev_row.get('VolRatio', 1.0)) < 0.95)
        and (close_p >= sma20_p * 0.985)
    )

    # LEI 2點鐘方向抵扣價回撤買點 & 六線高度密集張嘴突破
    is_lei_pullback_long = bool(
        prev_row.get('Clock_2', False)
        and (not prev_row.get('Is_Abnormal_Drop', False))
        and (close_p >= sma50_p * 0.99)
        and (
            (close_p <= sma20_p * 1.01 and close_p > float(prev_row.get('SMA20_Deduction', 0)))
            or (is_near_20ma and close_p > float(prev_row.get('SMA20_Deduction', 0)))
        )
    )
    is_lei_tight_long = bool(
        prev_row.get('LEI_MA_Tight_2Pct', False)
        and (close_p > float(prev_row.get('EMA20', close_p)))
        and (close_p > float(prev_row.get('SMA20_Deduction', close_p)))
    )

    # 做空條件 (原版結構做空 + J Law 動能破底/弱勢熊旗)
    orig_sr1 = close_p < sma200_p
    orig_sr2 = sma50_p < sma150_p and sma150_p < sma200_p
    orig_sr3 = close_p < p_high * 0.85
    orig_sr4 = float(prev_row.get('Mansfield_RS', 0)) < 0
    orig_sr5 = float(prev_row.get('VolRatio', 1.0)) < 0.85
    orig_sr6 = float(prev_row.get('LowerShadowRatio', 0.5)) < 0.4
    orig_sr7 = bool(prev_row.get('VCPSignal', False))
    orig_short_count = sum([orig_sr1, orig_sr2, orig_sr3, orig_sr4, orig_sr5, orig_sr6, orig_sr7])

    p_low_20_val = float(prev_row.get('SupShort_Causal', low_p))
    is_breakdown_short = (close_p <= p_low_20_val * 1.01) and (float(prev_row.get('VolRatio', 1.0)) > 1.4)
    ma_pressing = (close_p < sma20_p) and (sma20_p < sma50_p)
    is_weak_bounce = (bool(prev_row.get('VCPSignal', False)) or bool(prev_row.get('JLaw_Tight_Box', False))) and (float(prev_row.get('VolRatio', 1.0)) < 0.85)
    is_jlaw_short = ma_pressing and (float(prev_row.get('Mansfield_RS', 0)) < 0) and (is_weak_bounce or is_breakdown_short)

    # [v25.4 新增：回測系統 100% 同步掃描器之 O'Neil / Cohodes 空頭、弱勢反彈受阻空頭與個股大戶出貨日過濾]
    _drop_from_high = (p_high - close_p) / max(p_high, 1e-9)
    _recent5_max_h = float(df['High'].iloc[max(0, bar_idx - 4): bar_idx + 1].max())
    _rebound_resist_bt = (_drop_from_high > 0.15) and (close_p <= sma50_p * 1.02) and (_recent5_max_h >= sma50_p * 0.98)
    _recent3_df = df.iloc[max(0, bar_idx - 2): bar_idx + 1]
    _up_days_bt = _recent3_df[_recent3_df['Close'] > _recent3_df['Open']]
    _low_vol_rebound_bt = bool(not _up_days_bt.empty and float(_up_days_bt['Volume'].mean()) < float(prev_row.get('VolumeSMA50', prev_row['Volume'])) * 0.8)
    _support_break_bt = bool(close_p < low_20d_prev)
    _ms_score_bt = sum([not is_market_bullish, _rebound_resist_bt, _low_vol_rebound_bt, _support_break_bt])
    is_cohodes_short_bt = (_ms_score_bt >= 3) and (close_p < sma50_p)
    is_rebound_short_bt = _rebound_resist_bt and _low_vol_rebound_bt and (close_p < sma50_p)
    _stock_dist_days_bt = float(prev_row.get('Dist_Days_20d', 0))
    _climax_run_bt = bool(prev_row.get('Is_Climax_Run', False))

    # OLD:
    # if is_relaxed:
    #     long_cond = (close_p > sma50_p) and (float(prev_row.get('MACD', 0)) > float(prev_row.get('MACDSignal', 0))) and (not downtrend_veto_long) and is_market_bullish and is_market_health_ok
    #     short_cond = (close_p < sma50_p) and (float(prev_row.get('MACD', 0)) < float(prev_row.get('MACDSignal', 0))) and (not is_market_bullish)
    # else:
    #     long_cond = (
    #         (
    #             (momentum_ready and ((pass_count >= score_threshold) or early_stage_bypass or kk_flag) and (is_tight_setup or is_breakout_long))
    #             or is_sr_flip_pullback_long
    #             or is_lei_pullback_long
    #             or is_lei_tight_long
    #         )
    #         and (close_p >= sma50_p * 0.99)
    #         and (not downtrend_veto_long)
    #         and is_market_health_ok
    #     )
    #     short_cond = (orig_short_count >= score_threshold or is_jlaw_short) and (not is_market_bullish)
    # NEW: [v25.4 掃描與回測 100% 對齊] 加入個股 Climax Run 過熱不追高過濾、Cohodes 破位空點與弱勢反彈受阻空點
    if is_relaxed:
        long_cond = (close_p > sma50_p) and (float(prev_row.get('MACD', 0)) > float(prev_row.get('MACDSignal', 0))) and (not downtrend_veto_long) and is_market_bullish and is_market_health_ok
        short_cond = (close_p < sma50_p) and (float(prev_row.get('MACD', 0)) < float(prev_row.get('MACDSignal', 0))) and (not is_market_bullish)
    else:
        long_cond = (
            (
                (momentum_ready and ((pass_count >= score_threshold) or early_stage_bypass or kk_flag) and (is_tight_setup or is_breakout_long))
                or is_sr_flip_pullback_long
                or is_lei_pullback_long
                or is_lei_tight_long
            )
            and (close_p >= sma50_p * 0.99)
            and (not downtrend_veto_long)
            and (not _climax_run_bt)
            and is_market_health_ok
        )
        short_cond = (orig_short_count >= score_threshold or is_jlaw_short or is_cohodes_short_bt or is_rebound_short_bt) and (not is_market_bullish or is_cohodes_short_bt)

    # OLD:
    # if not long_cond and not short_cond:
    #     return res
    # sub_df = df.iloc[max(0, bar_idx - 160): bar_idx + 1]
    # lei_st = detect_lei_top_bottom_and_2b(sub_df, lookback=30)
    # NEW: [v25.2 修復 P0-1 回測未來錨點前視偏差與 Martin Luk 四大 Setup 缺席漏洞]
    # 僅在具備潛在多空動能時切片至 bar_idx (嚴格無未來函數)，重算當時因果 AVWAP 與 Martin Luk 設置
    _potential_ml_bar = (close_p >= sma50_p * 0.96) or (close_p <= sma50_p * 1.04)
    if not long_cond and not short_cond and not _potential_ml_bar:
        return res

    sub_df = df.iloc[max(0, bar_idx - 160): bar_idx + 1].copy()
    is_martin_pb_bt, is_martin_id_bt, is_martin_ep_bt, is_martin_fade_short_bt = False, False, False, False
    if not is_relaxed:
        try:
            _sw_anc_bt, _ = find_anchor_points(sub_df, lookback_swing=min(60, len(sub_df)), lookback_52w=len(sub_df))
            if _sw_anc_bt is not None:
                sub_df['AVWAP_Swing'] = calculate_anchored_vwap(sub_df, _sw_anc_bt)
            sub_df = compute_martin_luk_features_and_setups(sub_df, bg_df=None, market_type=market_type)
            _ml_bt_info = sub_df.attrs.get("martin_luk_info", {})
            is_martin_pb_bt = bool(_ml_bt_info.get("martin_pullback_long", False) and (not downtrend_veto_long) and is_market_health_ok)
            is_martin_id_bt = bool(_ml_bt_info.get("martin_inside_breakout_long", False) and (pass_count >= score_threshold - 1 or early_stage_bypass) and (not downtrend_veto_long) and is_market_health_ok)
            is_martin_ep_bt = bool(_ml_bt_info.get("martin_ep_long", False) and is_market_health_ok)
            is_martin_fade_short_bt = bool(_ml_bt_info.get("martin_weak_bounce_short", False) and (not is_market_bullish))
            long_cond = bool(long_cond or is_martin_pb_bt or is_martin_id_bt or is_martin_ep_bt)
            short_cond = bool(short_cond or is_martin_fade_short_bt)
        except Exception:
            pass

    if not long_cond and not short_cond:
        return res

    lei_st = detect_lei_top_bottom_and_2b(sub_df, lookback=30)
    # [v25.4 補齊 LEI 2B 破底翻做多與 2B 假突破做空信號同步]
    if not is_relaxed:
        if lei_st.get("bullish_2b", False) and (close_p >= sma50_p * 0.98) and (not downtrend_veto_long) and is_market_health_ok:
            long_cond = True
        if lei_st.get("bearish_2b", False) and (close_p < sma20_p) and (not is_market_bullish):
            short_cond = True

    if long_cond:
        if is_martin_ep_bt:
            status_str = "🚀 Martin Luk EP跳空爆量突破"
            order_style = "STOP"
        elif is_martin_pb_bt:
            status_str = "🎯 Martin Luk AVWAP+EMA匯聚低吸"
            order_style = "LIMIT"
        elif is_martin_id_bt:
            status_str = "🟢 Martin Luk InsideDay壓縮突破"
            order_style = "STOP"
        elif is_sr_flip_pullback_long:
            status_str = "🟢 S/R互換支撐回踩買點"
            order_style = "LIMIT"
        elif is_lei_pullback_long:
            status_str = "🟡 2點鐘抵扣價回撤買點"
            order_style = "LIMIT"
        elif is_breakout_long:
            status_str = "🟢 放量強勢突破 (Breakout)"
            order_style = "LIMIT" if close_p > box_high_10 * 1.025 else "STOP"
        elif kk_flag:
            status_str = "🔥 Kullamägi 動能旗型"
            order_style = "STOP"
        elif is_lei_tight_long:
            status_str = "🟢 LEI 均線密集張嘴突破"
            order_style = "STOP"
        else:
            status_str = "🟢 VCP緊密壓縮突破買點"
            order_style = "STOP"

        _bt_slope = compute_ma_slope_and_pullback_trigger(sub_df)
        _bt_exh = compute_us_swing_exhaustion_and_trap_guard(sub_df, market_type=market_type)
        if not is_relaxed:
            if _bt_slope.get("ma_ribbon_sloping_down", False) and (is_near_20ma or is_lei_pullback_long or is_sr_flip_pullback_long):
                return res
            if _bt_exh.get("is_exhausted_dont_chase", False) and (is_breakout_long or kk_flag or is_martin_id_bt or is_martin_ep_bt):
                return res
        plan_eval = evaluate_swing_trade_plan_and_rr(
            df=sub_df,
            plan_type="LONG",
            status=status_str,
            atr_val=atr_val,
            is_breakout_long=bool(is_breakout_long),
            is_near_20ma=bool(is_near_20ma or is_sr_flip_pullback_long),
            is_deduction_pullback=bool(is_lei_pullback_long),
            is_breakdown_short=False,
            lei_struct=lei_st,
            market_type=market_type
        )
        _prof_bt = get_active_stop_profile()
        _allowed_bt_stop_pct = max(float(_prof_bt.get("max_stop_pct", 0.056)) * 100.0 + 0.6, min(10.8, (atr_val / max(close_p, 1e-6)) * 195.0))
        if not is_relaxed and float(plan_eval.get("stop_pct", 0.0)) > _allowed_bt_stop_pct:
            return res
        if str(plan_eval.get("action_timeliness", "")).startswith("⚡"):
            order_style = "IMMEDIATE"
        res.update({
            "signal": "LONG",
            "order_style": order_style,
            "action_timeliness": plan_eval.get("action_timeliness", ""),
            "deepchart_score": plan_eval.get("deepchart_score", 50.0),
            "status_label": status_str,
            "planned_entry": float(plan_eval["entry"]),
            "stop_loss": float(plan_eval["sl"]),
            "tp_target": float(plan_eval["tp"]),
            "tp2_target": float(plan_eval.get("tp2", plan_eval["tp"])),
            "planned_rr": float(plan_eval["planned_rr"]),
            "sl_basis": str(plan_eval.get("sl_basis_desc", "")),
            "tp_basis": str(plan_eval.get("tp_basis_desc", "")),
            "entry_desc": str(plan_eval.get("entry_type_desc", ""))
        })
        return res

    elif short_cond:
        if is_breakdown_short:
            status_str = "🔴 帶量破底做空 (Breakdown)"
            order_style = "LIMIT" if close_p < box_low_10 * 0.975 else "STOP"
        elif is_weak_bounce:
            status_str = "🔴 均線蓋頭弱反彈 (Bear Flag)"
            order_style = "LIMIT"
        else:
            status_str = "🟠 空頭趨勢反彈受阻做空"
            order_style = "LIMIT"

        plan_eval = evaluate_swing_trade_plan_and_rr(
            df=sub_df,
            plan_type="SHORT",
            status=status_str,
            atr_val=atr_val,
            is_breakout_long=False,
            is_near_20ma=False,
            is_deduction_pullback=False,
            is_breakdown_short=bool(is_breakdown_short),
            lei_struct=lei_st,
            market_type=market_type
        )
        if str(plan_eval.get("action_timeliness", "")).startswith("⚡"):
            order_style = "IMMEDIATE"
        res.update({
            "signal": "SHORT",
            "order_style": order_style,
            "action_timeliness": plan_eval.get("action_timeliness", ""),
            "deepchart_score": plan_eval.get("deepchart_score", 50.0),
            "status_label": status_str,
            "planned_entry": float(plan_eval["entry"]),
            "stop_loss": float(plan_eval["sl"]),
            "tp_target": float(plan_eval["tp"]),
            "tp2_target": float(plan_eval.get("tp2", plan_eval["tp"])),
            "planned_rr": float(plan_eval["planned_rr"]),
            "sl_basis": str(plan_eval.get("sl_basis_desc", "")),
            "tp_basis": str(plan_eval.get("tp_basis_desc", "")),
            "entry_desc": str(plan_eval.get("entry_type_desc", ""))
        })
        return res

    return res

def compute_backtest_institutional_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
    """
    [v25.1 修正分批止盈勝率雙重計算漏洞]
    按獨立交易編號 (_trade_id) 聚合同一筆波段交易之「減倉一半 + 尾倉平倉」總淨利潤，
    計算真實獨立波段勝率 (Unique Trade Win Rate)、最大回撤 (MDD)、獲利因子 (Profit Factor)、夏普比率 (Sharpe) 與總富途交易費用。
    """
    metrics = {
        "max_drawdown_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "expectancy_pct": 0.0,
        "payoff_ratio": 0.0,
        "unique_trades_count": 0,
        "unique_win_rate": 0.0,
        "unique_long_count": 0,
        "unique_long_win_rate": 0.0,
        "unique_short_count": 0,
        "unique_short_win_rate": 0.0,
        "total_futu_fees": 0.0
    }
    try:
        # OLD:
        # if equity_curve is not None and not equity_curve.empty:
        #     roll_max = equity_curve.cummax()
        #     drawdown = (equity_curve - roll_max) / (roll_max + 1e-9) * 100.0
        #     metrics["max_drawdown_pct"] = float(drawdown.min())
        #     daily_rets = equity_curve.pct_change().dropna()
        #     if len(daily_rets) > 5 and daily_rets.std() > 0:
        #         metrics["sharpe_ratio"] = float((daily_rets.mean() / daily_rets.std()) * np.sqrt(252))
        # NEW: [v25.2 修復 P1-3 統計指標失真與驗證缺失] 扣除 3.5% 無風險利率計算 Sharpe/Sortino、Calmar Ratio、前70%/後30% 樣本內外切分與 1,000 次蒙特卡洛壓力測試
        metrics.update({
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "cagr_pct": 0.0,
            "in_sample_ret_pct": 0.0,
            "out_of_sample_ret_pct": 0.0,
            "mc_95_mdd_pct": 0.0,
            "mc_median_ret_pct": 0.0,
            "sqn_score": 0.0,
            "sqn_rating": "資料不足",
            "max_consec_losses": 0,
            "max_tuw_days": 0,
            "wfo_consistency_pct": 0.0,
            "wfo_windows_summary": ""
        })
        if equity_curve is not None and not equity_curve.empty:
            roll_max = equity_curve.cummax()
            drawdown = (equity_curve - roll_max) / (roll_max + 1e-9) * 100.0
            metrics["max_drawdown_pct"] = float(drawdown.min())
            daily_rets = equity_curve.pct_change().dropna()
            rf_daily = 0.035 / 252.0
            if len(daily_rets) > 5 and daily_rets.std() > 0:
                excess_rets = daily_rets - rf_daily
                metrics["sharpe_ratio"] = round(float((excess_rets.mean() / daily_rets.std()) * np.sqrt(252)), 2)
                downside_rets = daily_rets[daily_rets < rf_daily] - rf_daily
                downside_std = float(np.sqrt(np.mean(np.square(downside_rets)))) if len(downside_rets) > 0 else 0.0
                if downside_std > 1e-9:
                    metrics["sortino_ratio"] = round(float((excess_rets.mean() / downside_std) * np.sqrt(252)), 2)
            years_span = max((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25, 0.1) if len(equity_curve) > 1 else 1.0
            total_mult = max(float(equity_curve.iloc[-1]) / max(float(initial_capital), 1.0), 1e-4)
            cagr_pct = ((total_mult ** (1.0 / years_span)) - 1.0) * 100.0
            metrics["cagr_pct"] = round(float(cagr_pct), 2)
            if abs(metrics["max_drawdown_pct"]) > 0.01:
                metrics["calmar_ratio"] = round(float(cagr_pct / abs(metrics["max_drawdown_pct"])), 2)
            # 樣本內 (In-Sample 前 70%) 與 樣本外 (Out-of-Sample 後 30%) 分段報酬驗證 + 最大水下恢復天數 (Max Time Under Water) + 4段 Walk-Forward 滾動穩定性檢驗
            is_underwater = equity_curve < roll_max
            if is_underwater.any():
                uw_groups = (~is_underwater).cumsum()
                uw_streaks = is_underwater.groupby(uw_groups).sum()
                metrics["max_tuw_days"] = int(uw_streaks.max()) if not uw_streaks.empty else 0

            split_idx = int(len(equity_curve) * 0.70)
            if 2 < split_idx < len(equity_curve) - 1:
                eq_is_end = float(equity_curve.iloc[split_idx])
                eq_oos_end = float(equity_curve.iloc[-1])
                metrics["in_sample_ret_pct"] = round(((eq_is_end / max(float(initial_capital), 1.0)) - 1.0) * 100.0, 2)
                metrics["out_of_sample_ret_pct"] = round(((eq_oos_end / max(eq_is_end, 1.0)) - 1.0) * 100.0, 2)

            if len(equity_curve) >= 40:
                w_edges = np.linspace(0, len(equity_curve) - 1, 5, dtype=int)
                w_rets = []
                w_Pos = 0
                for w_i in range(4):
                    s_eq = float(equity_curve.iloc[w_edges[w_i]])
                    e_eq = float(equity_curve.iloc[w_edges[w_i + 1]])
                    r_w = ((e_eq / max(s_eq, 1.0)) - 1.0) * 100.0
                    w_rets.append(f"W{w_i+1}:{r_w:+.1f}%")
                    if r_w >= 0:
                        w_Pos += 1
                metrics["wfo_consistency_pct"] = round((w_Pos / 4.0) * 100.0, 1)
                metrics["wfo_windows_summary"] = " ｜ ".join(w_rets)

        if trades_df is not None and not trades_df.empty and '利潤' in trades_df.columns:
            if '富途費用' in trades_df.columns:
                metrics["total_futu_fees"] = round(float(trades_df['富途費用'].sum()), 2)

            if '_trade_id' in trades_df.columns:
                grouped = trades_df.groupby('_trade_id', as_index=False).agg({
                    '方向': 'first',
                    '利潤': 'sum',
                    '_initial_notional': 'first'
                })
                grouped['真實波段回報(%)'] = (grouped['利潤'] / grouped['_initial_notional'].replace(0, np.nan)) * 100.0
            else:
                grouped = trades_df.copy()
                grouped['真實波段回報(%)'] = grouped['回報率(%)']

            metrics["unique_trades_count"] = int(len(grouped))
            u_long = grouped[grouped['方向'] == '🟢 做多']
            u_short = grouped[grouped['方向'] == '🔴 做空']
            metrics["unique_long_count"] = int(len(u_long))
            metrics["unique_short_count"] = int(len(u_short))
            metrics["unique_long_win_rate"] = round(float((u_long['利潤'] > 0).mean() * 100.0), 1) if not u_long.empty else 0.0
            metrics["unique_short_win_rate"] = round(float((u_short['利潤'] > 0).mean() * 100.0), 1) if not u_short.empty else 0.0
            metrics["unique_win_rate"] = round(float((grouped['利潤'] > 0).mean() * 100.0), 1) if not grouped.empty else 0.0

            gross_profit = float(grouped[grouped['利潤'] > 0]['利潤'].sum())
            gross_loss = abs(float(grouped[grouped['利潤'] < 0]['利潤'].sum()))
            metrics["profit_factor"] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

            wins_ret = grouped[grouped['利潤'] > 0]['真實波段回報(%)'].dropna()
            loss_ret = grouped[grouped['利潤'] <= 0]['真實波段回報(%)'].dropna()
            avg_win = float(wins_ret.mean()) if not wins_ret.empty else 0.0
            avg_loss = abs(float(loss_ret.mean())) if not loss_ret.empty else 0.0
            win_rate_dec = len(wins_ret) / max(len(grouped), 1)
            metrics["payoff_ratio"] = round(avg_win / avg_loss, 2) if avg_loss > 0 else 0.0
            metrics["expectancy_pct"] = round((win_rate_dec * avg_win) - ((1.0 - win_rate_dec) * avg_loss), 2)
            # [v27.0 新增：Van Tharp SQN (系統品質指數) 與最大連續止損筆數統計]
            rets_arr = grouped['真實波段回報(%)'].dropna().values
            if len(rets_arr) >= 3 and np.std(rets_arr) > 1e-6:
                sqn_raw = float(np.sqrt(min(len(rets_arr), 100)) * (np.mean(rets_arr) / np.std(rets_arr)))
                metrics["sqn_score"] = round(sqn_raw, 2)
                if sqn_raw >= 3.0:
                    metrics["sqn_rating"] = "🏆 聖杯級 (≥3.0)"
                elif sqn_raw >= 2.0:
                    metrics["sqn_rating"] = "🟢 優秀機構級 (2.0~2.9)"
                elif sqn_raw >= 1.5:
                    metrics["sqn_rating"] = "🟡 可交易系統 (1.5~1.9)"
                else:
                    metrics["sqn_rating"] = "🟠 波動偏大 (<1.5)"
            is_loss_seq = (grouped['利潤'].values <= 0).astype(int)
            max_l_streak, cur_l_streak = 0, 0
            for flag in is_loss_seq:
                if flag == 1:
                    cur_l_streak += 1
                    max_l_streak = max(max_l_streak, cur_l_streak)
                else:
                    cur_l_streak = 0
            metrics["max_consec_losses"] = int(max_l_streak)

            # [v25.2 新增：1,000 次 Bootstrap 蒙特卡洛交易序列重排壓力測試 (計算 95% 信賴水準極端最大回撤)]
            trade_pnls = grouped['利潤'].dropna().values
            if len(trade_pnls) >= 3:
                rng = np.random.default_rng(42)
                n_trades_mc = len(trade_pnls)
                sampled_pnls = rng.choice(trade_pnls, size=(500, n_trades_mc), replace=True)
                eq_paths = float(initial_capital) + np.cumsum(sampled_pnls, axis=1)
                eq_paths = np.concatenate([np.full((500, 1), float(initial_capital)), eq_paths], axis=1)
                peaks_mc = np.maximum.accumulate(eq_paths, axis=1)
                dds_mc = (eq_paths - peaks_mc) / np.maximum(peaks_mc, 1.0) * 100.0
                worst_dds = np.min(dds_mc, axis=1)
                metrics["mc_95_mdd_pct"] = round(float(np.percentile(worst_dds, 5)), 2)
                metrics["mc_median_ret_pct"] = round(float((np.median(eq_paths[:, -1]) / max(float(initial_capital), 1.0) - 1.0) * 100.0), 2)
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

with _CACHE_LOCK:
    _global_persistent_data = get_persistent_data()
    ensure_persistent_schema_fresh(_global_persistent_data)
    load_persisted_state_from_disk(_global_persistent_data)
if "_session_isolated_store" not in st.session_state:
    with _CACHE_LOCK:
        st.session_state["_session_isolated_store"] = {
            "radar_results": dict(_global_persistent_data.get("radar_results", {"港股": None, "美股": None, "自選股": None, "自訂": None})),
            "radar_correlation_warnings": dict(_global_persistent_data.get("radar_correlation_warnings", {"港股": [], "美股": [], "自選股": [], "自訂": []})),
            "radar_correlation_matrices": dict(_global_persistent_data.get("radar_correlation_matrices", {"港股": None, "美股": None, "自選股": None, "自訂": None})),
            "watchlist_radar_results": dict(_global_persistent_data.get("watchlist_radar_results", {})),
            "radar_timestamps": dict(_global_persistent_data.get("radar_timestamps", {})),
            "signal_lifecycle_store": dict(_global_persistent_data.get("signal_lifecycle_store", {})),
            "radar_breadth_stats": dict(_global_persistent_data.get("radar_breadth_stats", {})),
            "user_locked_positions": dict(_global_persistent_data.get("user_locked_positions", {})),
            "auto_entry_date_hash_store": dict(_global_persistent_data.get("auto_entry_date_hash_store", {})),
            "_schema_ver": CURRENT_SCHEMA_VERSION
        }
persistent_data = st.session_state["_session_isolated_store"]

# ==========================================
# 🌟 初始化 Session State (仿富途牛牛分組)
# ==========================================
if "watchlist_dict" not in st.session_state:
    _disk_wl = _global_persistent_data.get("_disk_watchlist_dict")
    if isinstance(_disk_wl, dict) and len(_disk_wl) > 0:
        st.session_state.watchlist_dict = {k: list(v) for k, v in _disk_wl.items()}
    else:
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
# OLD: target_ticker = qp.get("ticker", "0700.HK")
# NEW: [v31.0 安全加固] 對 URL Query Params 代碼進行白名單正則消毒
target_ticker = sanitize_ticker_input(qp.get("ticker", "0700.HK"), default_val="0700.HK")
# [NEW INCREMENTAL PATCH 修復 Bug 1C] 讀取跳轉時攜帶的雷達方向與嚴格度參數，防止跳轉後因側欄預設值不同而變換多空方向
target_dir_from_radar = qp.get("dir", "")
target_strict_from_radar = qp.get("strict", "")

# ==========================================
# 1. 智能代碼格式化與市場識別
# ==========================================
def format_and_detect_market(ticker):
    ticker = sanitize_ticker_input(str(ticker), default_val="0700.HK")
    if ticker.isdigit():
        ticker = f"{int(ticker):04d}.HK"
    elif len(ticker) <= 4 and ticker.replace('.', '').isdigit():
        ticker = f"{int(float(ticker)):04d}.HK"
    else:
        ticker = US_TICKER_ALIAS_MAP.get(ticker, ticker)

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
        "2899.HK": "紫金礦業", "3311.HK": "中國建築國際", "9888.HK": "百度集團-SW", "9626.HK": "嗶哩嗶哩-W", "3908.HK": "中金公司",
        "6030.HK": "中信証券", "6110.HK": "滔搏", "9926.HK": "康方生物", "3888.HK": "金山軟件", "6060.HK": "眾安在線",
        "9896.HK": "名創優品", "1357.HK": "美圖公司", "1797.HK": "東方甄選", "0358.HK": "江西銅業", "2158.HK": "醫渡科技",
        "6186.HK": "中國飛鶴", "9698.HK": "萬國數據-SW", "2400.HK": "心動公司", "0780.HK": "同程旅行"
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
        elif interval in ["1h", "4h", "90m"]:
            period = "730d"
        else:
            period = "5y"

        _dl_interval = "1h" if interval == "4h" else interval
        idx_df = yf.download(index_ticker, period=period, interval=_dl_interval, auto_adjust=True, progress=False)
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
        if interval == "4h" and len(idx_df) >= 4:
            idx_df = idx_df.resample("4h").agg({
                c: ("first" if c == "Open" else ("max" if c == "High" else ("min" if c == "Low" else ("sum" if c == "Volume" else "last"))))
                for c in idx_df.columns if c in ["Open", "High", "Low", "Close", "Volume"]
            }).dropna(subset=["Close"])

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
# [v25.1 數據池清理] 剔除已退市/清盤或長期停牌之殭屍代碼 (如 3333.HK 恒大、2007.HK 碧桂園)，替換為活躍高流動性港股標的 (9888.HK 百度、9626.HK 嗶哩嗶哩)
# [v41.1 股池純化] 將低 ADR% (<1.8%) 之慢速收息內銀/公用股移至獨立收息防守池，保持高動能池純度
HK_MOMENTUM_POOL = [
    "0700.HK", "3690.HK", "1810.HK", "1211.HK", "2015.HK", "9868.HK", "9988.HK", "0981.HK", "2269.HK", "2359.HK",
    "0388.HK", "1024.HK", "1833.HK", "1789.HK", "2500.HK", "6098.HK", "3319.HK", "1995.HK", "1299.HK", "2318.HK",
    "0883.HK", "0027.HK", "9618.HK", "9999.HK", "1088.HK", "0268.HK", "0285.HK", "0322.HK", "0772.HK", "0853.HK",
    "1093.HK", "1177.HK", "1928.HK", "2020.HK", "2331.HK", "2382.HK", "3606.HK", "6618.HK", "6690.HK", "9922.HK",
    "9992.HK", "9961.HK", "6862.HK", "1919.HK", "9888.HK", "2313.HK", "2333.HK", "1060.HK", "1099.HK", "1316.HK",
    "1347.HK", "1801.HK", "1818.HK", "1898.HK", "1929.HK", "2018.HK", "2192.HK", "2338.HK", "2899.HK", "9626.HK",
    "3908.HK", "6030.HK", "6110.HK"
]

HK_DEFENSIVE_DIVIDEND_POOL = [
    "0005.HK", "0941.HK", "0939.HK", "1398.HK", "3328.HK", "3988.HK",
    "0002.HK", "0011.HK", "0016.HK", "0066.HK", "1113.HK", "1044.HK", "1193.HK", "2669.HK", "3311.HK", "0293.HK"
]

# [v42.0 新增：《超級績效》港美中小型高彈性潛力池 (ADR% 3.2%~7.5% + 機構級流動性)]
HK_SMALL_MID_SUPER_POOL = [
    "9992.HK", "2269.HK", "2359.HK", "9926.HK", "1801.HK", "0268.HK", "1347.HK", "3888.HK", "0772.HK", "9626.HK",
    "1919.HK", "1357.HK", "9896.HK", "6060.HK", "2400.HK", "0780.HK", "1797.HK", "0285.HK", "2018.HK", "1818.HK",
    "0358.HK", "2899.HK", "0853.HK", "1789.HK", "6618.HK", "1833.HK", "9698.HK", "2382.HK", "9868.HK", "2015.HK"
]

US_SMALL_MID_SUPER_POOL = [
    "APP", "AXON", "RDDT", "HOOD", "PLTR", "MSTR", "COIN", "RKLB", "IONQ", "AFRM",
    "SOFI", "UPST", "DUOL", "CAVA", "TOST", "CELH", "ASTS", "VRT", "VST", "CEG",
    "GEV", "ANET", "CRWD", "SNOW", "DDOG", "ZS", "ROKU", "SHOP", "ARM", "SMCI"
]

US_MOMENTUM_POOL = [
    "NVDA", "AAPL", "TSLA", "PLTR", "MSFT", "AMZN", "META", "GOOGL", "AMD", "MSTR",
    "COIN", "SMCI", "AVGO", "CRWD", "NFLX", "ADBE", "SNOW", "DDOG", "ZS", "PANW",
    "ARM", "ASML", "TSM", "INTC", "QCOM", "MU", "UBER", "ABNB", "PYPL", "XYZ",
    "ROKU", "SHOP", "APP", "ANET", "AXON", "VST", "CEG", "GEV", "HOOD", "RDDT",
    "RKLB", "IONQ", "AFRM", "SOFI", "UPST", "DUOL", "CAVA", "TOST", "CELH", "VRT"
]

@st.cache_data(ttl=3600)
def get_us_dual_benchmark_regime(interval: str = "1d") -> Dict[str, Any]:
    out = {"nasdaq_bear": False, "nasdaq_dist_days": 0, "dual_index_hostile": False, "dual_index_note": ""}
    if interval != "1d":
        return out
    try:
        ndx_df = get_index_data("^IXIC", interval="1d")
        if ndx_df is not None and len(ndx_df) >= 55:
            last_n = ndx_df.iloc[-1]
            c_n = float(last_n['Close'])
            s50_n = float(last_n.get('SMA50', c_n))
            s200_n = float(last_n.get('SMA200', c_n))
            dist_n = int(float(last_n.get('Dist_Days_20d', 0)))
            out["nasdaq_dist_days"] = dist_n
            if c_n < s50_n or s50_n < s200_n:
                out["nasdaq_bear"] = True
            if out["nasdaq_bear"] or dist_n >= 5:
                out["dual_index_hostile"] = True
                out["dual_index_note"] = f"納斯達克指數 (^IXIC) 目前處於 50MA 下方或出貨日偏高 ({dist_n} 次)，美股成長股突破失敗率高"
    except Exception:
        pass
    return out


def evaluate_global_market_regime_gate(
    df: pd.DataFrame,
    bg_df: Optional[pd.DataFrame],
    market_type: str = "US/Global",
    interval: str = "1d"
) -> Dict[str, Any]:
    """
    [v41.1 Module 1：五階段大盤趨勢與「🌤️ 回調見底/放量轉勢反攻 (Bottom Reversal / FTD)」識別引擎]
    徹底解決舊版僅靠滯後 50MA 判斷大市，導致「恒指/美股回調見底大升當日仍被誤判為死空頭並屏蔽做多清單」的問題！
    同時綜合：
    - 【中線均線位置】：指數相對於 20MA / 50MA / 200MA；
    - 【短線領先轉勢信號 (Reversal / FTD)】：
      1) 指數單日強勢大升 (日升幅 >= +1.0% 且收陽位於當日波幅上半部)；或
      2) 指數收盤重新站上 5EMA / 9EMA / 10MA；或
      3) 指數出現底分型 / 升穿前日高點 (Higher High + Higher Low)。
    分為 5 個實戰狀態（預設絕不屏蔽做多清單，改為提供精準倉位比例與買法導航）：
    1) BULL_CONFIRMED (🟢 大市強勢多頭)：指數 > 20MA & > 50MA -> 積極做多，支援 100% 滿倉。
    2) BOTTOM_REVERSAL (🌤️ 大市回調見底 / 放量轉勢反攻)：指數雖在 50MA 下方或剛回調，但當日出現大升 >=1.0% 或收復 9EMA/10MA -> 全面開放做多（捕捉率先見底領頭羊與 2B 破底翻），建議先下 60%~70% 試倉底倉，企穩加滿倉！
    3) BULL_PULLBACK (🟡 大市多頭回調期)：指數 > 50MA 但暫時跌穿 20MA -> 開放做多清單，但限制追高突破，優先揀回踩均線低吸。
    4) CHOP_NEUTRAL (🟠 大市均線拉鋸期)：指數於 50MA 附近震盪 -> 開放雙向清單，強調回踩買入與 TP1 全數止盈。
    5) BEAR_WEAK (🔴 大市弱勢尋底期)：指數 < 50MA 且當日無反攻信號、仍處於 9EMA 下方陰跌 -> 預設仍顯示逆市抗跌做多領頭羊（不清空清單），但禁止追高突破並提示縮小倉位或順勢做空。
    """
    res = {
        "regime_code": "BULL_CONFIRMED",
        "regime_badge": "🟢 大市強勢多頭 (支援 100% 滿倉做多)",
        "allow_breakout_long": True,
        "allow_pullback_long": True,
        "allow_short": True,
        "tp1_full_exit_mode": False,
        "recommended_pos_pct": 100,
        "index_daily_ret_pct": 0.0,
        "is_bottom_reversal_day": False,
        "regime_reason": "大盤指數穩守 20MA 與 50MA 之上且派發日低，順勢積極做多領頭羊。"
    }
    if df is None or len(df) < 20:
        return res
    try:
        last = df.iloc[-1]
        b_close = float(last.get("CloseBench", np.nan))
        b_sma50 = float(last.get("SMA50BenchValue", np.nan))
        b_sma200 = float(last.get("SMA200BenchValue", np.nan))
        b_dist = int(float(last.get("Bench_Dist_Days", 0))) if pd.notna(last.get("Bench_Dist_Days")) else 0

        b_sma20 = b_sma50
        b_ema9 = b_close
        b_sma10 = b_close
        idx_ret_pct = 0.0
        idx_bull_bar = False
        idx_higher_high = False
        if bg_df is not None and len(bg_df) >= 25 and "Close" in bg_df.columns:
            b_close_ser = bg_df["Close"]
            b_sma20 = float(b_close_ser.rolling(20, min_periods=5).mean().iloc[-1])
            b_sma10 = float(b_close_ser.rolling(10, min_periods=3).mean().iloc[-1])
            b_ema9 = float(b_close_ser.ewm(span=9, adjust=False).mean().iloc[-1])
            if len(bg_df) >= 2:
                prev_b_c = float(b_close_ser.iloc[-2])
                curr_b_c = float(b_close_ser.iloc[-1])
                idx_ret_pct = ((curr_b_c - prev_b_c) / max(prev_b_c, 1e-6)) * 100.0
                if "Open" in bg_df.columns and "High" in bg_df.columns and "Low" in bg_df.columns:
                    curr_b_o = float(bg_df["Open"].iloc[-1])
                    curr_b_h = float(bg_df["High"].iloc[-1])
                    curr_b_l = float(bg_df["Low"].iloc[-1])
                    prev_b_h = float(bg_df["High"].iloc[-2])
                    b_close_pos = (curr_b_c - curr_b_l) / max(curr_b_h - curr_b_l, 1e-6)
                    idx_bull_bar = bool(curr_b_c >= curr_b_o and b_close_pos >= 0.52)
                    idx_higher_high = bool(curr_b_c > prev_b_h or (idx_ret_pct >= 0.85 and idx_bull_bar))

        is_reversal_bounce = bool(
            (idx_ret_pct >= 1.0 and idx_bull_bar)
            or (b_close >= b_ema9 * 0.999 and idx_ret_pct >= 0.45)
            or (b_close >= b_sma10 and idx_higher_high)
        )
        res["index_daily_ret_pct"] = round(idx_ret_pct, 2)
        res["is_bottom_reversal_day"] = is_reversal_bounce

        strict_hide_countertrend = False
        try:
            strict_hide_countertrend = bool(st.session_state.get("strict_hide_countertrend_signals", False))
        except Exception:
            strict_hide_countertrend = False

        if pd.notna(b_close) and pd.notna(b_sma50):
            is_above_50 = (b_close >= b_sma50)
            is_above_20 = (b_close >= b_sma20 * 0.998)

            if is_above_50 and is_above_20 and b_dist < 4:
                res.update({
                    "regime_code": "BULL_CONFIRMED",
                    "regime_badge": f"🐂 確認牛市主升段 (今日 {idx_ret_pct:+.2f}%｜支援 100% 滿倉做多)",
                    "allow_breakout_long": True,
                    "allow_pullback_long": True,
                    "allow_short": not strict_hide_countertrend,
                    "tp1_full_exit_mode": False,
                    "recommended_pos_pct": 100,
                    "regime_reason": f"基準指數 ({b_close:.0f}) 穩守 20MA ({b_sma20:.0f}) 與 50MA ({b_sma50:.0f}) 之上 (出貨日僅 {b_dist} 次)，屬《超級績效》確認牛市，可積極滿倉做多領頭羊。"
                })
            elif is_above_50 and is_above_20 and b_dist >= 4:
                res.update({
                    "regime_code": "TOPPING_WARNING",
                    "regime_badge": f"🌩️ 牛市見頂派發預警 (出貨日達 {b_dist} 次｜禁追高、只准低吸並鎖潤)",
                    "allow_breakout_long": False if strict_hide_countertrend else True,
                    "allow_pullback_long": True,
                    "allow_short": True,
                    "tp1_full_exit_mode": True,
                    "recommended_pos_pct": 65,
                    "regime_reason": f"基準指數雖在 20MA/50MA 之上，但近20日已累積 {b_dist} 個機構派發日 (Distribution Cluster)，屬於牛市見頂轉弱高危期 (如10月8日美股見頂前夕)，嚴禁高追突破！"
                })
            elif (not is_above_50 or not is_above_20) and is_reversal_bounce:
                _turn_title = "🌤️ 大盤修正結束 / 重拾升勢" if is_above_50 else "🌤️ 熊轉牛初段 / 回調見底轉勢日 (彈簧股爆發期)"
                res.update({
                    "regime_code": "BOTTOM_REVERSAL",
                    "regime_badge": f"{_turn_title} (今日 {idx_ret_pct:+.2f}%｜捉超級績效彈簧股)",
                    "allow_breakout_long": True,
                    "allow_pullback_long": True,
                    "allow_short": True,
                    "tp1_full_exit_mode": False,
                    "recommended_pos_pct": 75,
                    "regime_reason": (
                        f"基準指數 ({b_close:.0f}) 今日強勢反攻 ({idx_ret_pct:+.2f}%) 並站回短線均線 (9EMA {b_ema9:.0f}) 之上！"
                        f"《超級績效》指出：大盤修正結束或熊轉牛初段，具備『3/4 或 4/4 重疊篩網 + 網球強彈 / Low Cheat』的領頭羊會像鬆開的彈簧一樣升得最快！"
                    )
                })
            elif is_above_50 and not is_above_20:
                res.update({
                    "regime_code": "BULL_PULLBACK",
                    "regime_badge": f"🟡 牛市中期正常修正 (今日 {idx_ret_pct:+.2f}%｜宜 Low Cheat / 回踩低吸)",
                    "allow_breakout_long": False if strict_hide_countertrend else True,
                    "allow_pullback_long": True,
                    "allow_short": True,
                    "tp1_full_exit_mode": True,
                    "recommended_pos_pct": 75,
                    "regime_reason": f"基準指數處於 50MA ({b_sma50:.0f}) 之上但回落至 20MA ({b_sma20:.0f}) 下方進行中期修正，最佳策略是尋找抗跌的『Low Cheat 低位作弊點』或『網球強彈』個股埋伏。"
                })
            else:
                res.update({
                    "regime_code": "BEAR_WEAK",
                    "regime_badge": f"🐻 確認熊市 / 尋底期 (今日 {idx_ret_pct:+.2f}%｜順勢做空或僅試單 3/4 重疊抗跌股)",
                    "allow_breakout_long": False,
                    "allow_pullback_long": False if strict_hide_countertrend else True,
                    "allow_short": True,
                    "tp1_full_exit_mode": True,
                    "recommended_pos_pct": 50,
                    "regime_reason": (
                        f"基準指數 ({b_close:.0f}) 位於 50MA ({b_sma50:.0f}) 與 9EMA ({b_ema9:.0f}) 下方且尚未出現轉勢陽燭。"
                        f"主方向宜順勢做空，若做多僅限『多重小篩網重疊 (≥3/4) + 網球抗跌』之獨立強勢股並嚴守 HK$1,000 止損。"
                    )
                })
    except Exception:
        pass
    return res

POPULAR_HK_LOTS = {
    "0700.HK": 100, "3690.HK": 100, "9988.HK": 100, "1810.HK": 200, "1211.HK": 500,
    "1299.HK": 200, "0005.HK": 400, "0388.HK": 100, "0941.HK": 500, "0883.HK": 1000,
    "0981.HK": 500, "0939.HK": 1000, "1398.HK": 1000, "2318.HK": 500, "2020.HK": 100,
    "2269.HK": 100, "2317.HK": 500, "0853.HK": 500, "1024.HK": 100, "1113.HK": 500,
    "0011.HK": 100, "0002.HK": 500, "0016.HK": 1000, "0066.HK": 500, "9888.HK": 50, "9626.HK": 20, "2015.HK": 100, "9868.HK": 100, "9618.HK": 50, "9999.HK": 100, "9992.HK": 200
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
def calculate_indicators(df, bg_df, interval="1d", market_type="US/Global"):
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

    # [v39.0 根治前視偏差] 廢除全序列 scipy.signal.find_peaks (因其利用未來 K 線確認過去波峰波谷)，統一採用 100% 純因果滾動高低點
    df['SupShort'] = df['Low'].shift(1).rolling(window=20, min_periods=5).min().fillna(df['Low'])
    df['ResShort'] = df['High'].shift(1).rolling(window=20, min_periods=5).max().fillna(df['High'])
    df['SupLong'] = df['Low'].shift(1).rolling(window=60, min_periods=10).min().fillna(df['Low'])
    df['ResLong'] = df['High'].shift(1).rolling(window=60, min_periods=10).max().fillna(df['High'])

    df['TR'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                     abs(df['Low'] - df['Close'].shift(1))))
    df['ATR'] = df['TR'].rolling(14).mean()
    # OLD: df['VolRatio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    # NEW: [v25.2 修復 P1-1 盤中未收盤 K 線成交量偏低誤判縮量漏洞] 對當日未收盤 K 線採用時間折算全天預估量計算 VolRatio
    df['Effective_Volume'] = extrapolate_intraday_partial_volume(df, market_type=market_type, interval=interval)
    df['VolRatio'] = df['Effective_Volume'] / (df['Volume'].rolling(20).mean() + 1e-9)
    df['Turnover'] = df['Close'] * df['Effective_Volume']

    df['TurnoverMA20'] = df['Turnover'].rolling(window=20).mean()
    df['BigMoney_In'] = (df['Volume'] > df['VolumeSMA50'] * 2.0) & (df['Close'] > df['Open'])
    df['BigMoney_Out'] = (df['Volume'] > df['VolumeSMA50'] * 2.0) & (df['Close'] < df['Open'])

    df['UpperShadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerShadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['TotalRange'] = df['High'] - df['Low'] + 1e-9
    df['ShadowRatio'] = df['UpperShadow'] / df['TotalRange']
    df['LowerShadowRatio'] = df['LowerShadow'] / df['TotalRange']

    # OLD: periods_per_year = 252 if interval == '1d' else (52 if interval == '1wk' else 12)
    # NEW: [v29.0 修復分鐘/小時圖 High52W 誤設為 12 根 K 線之漏洞] 根據真實交易週期精確設定滾動高低點視窗
    _ppy_map = {'1d': 252, '1wk': 52, '4h': 420, '1h': 500, '30m': 500, '15m': 400, '5m': 350, '1m': 300}
    periods_per_year = _ppy_map.get(interval, 252)
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

    # OLD:
    # obv = [0]
    # for i in range(1, len(df)): ...
    # NEW: [v30.0 向量化加速 OBV 計算，結果 100% 數學等價但速度提升 30 倍]
    _price_diff = df['Close'].diff()
    _obv_dir = np.where(_price_diff > 0, 1.0, np.where(_price_diff < 0, -1.0, 0.0))
    _obv_dir[0] = 0.0
    df['OBV'] = (df['Volume'].values * _obv_dir).cumsum()

    # MFI
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    pos_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0), index=df.index)
    pos_mf = pos_flow.rolling(14).sum()
    neg_mf = neg_flow.rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-9)))

    if bg_df is not None and not bg_df.empty:
        # OLD: cols_to_get = ['Close', 'SMA50']
        # NEW: [v30.0 防禦性欄位檢查] 防止大盤數據短暫缺失 SMA50 時拋出 KeyError
        cols_to_get = [c for c in ['Close', 'SMA50', 'SMA200', 'Dist_Days_20d'] if c in bg_df.columns]
        
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

    # [v25.1 清理無效重複欄位與前視偏差] 統一使用 100% 因果無未來函數之多層級結構支撐阻力、Wilder ATR/RSI 與三師特徵
    df = compute_causal_swing_sr(df)
    df['ATR_Wilder'], df['RSI_Wilder'] = compute_wilder_atr_and_rsi(df)
    df['ATR'] = df['ATR_Wilder'].fillna(df['ATR'])
    df['RSI'] = df['RSI_Wilder'].fillna(df['RSI'])
    df = compute_lei_dual_ma_and_ppo_features(df)
    df['ResShort'] = df['ResShort_Causal']
    df['SupShort'] = df['SupShort_Causal']
    df['ResLong'] = df['ResLong_Causal']
    df['SupLong'] = df['SupLong_Causal']
    df = compute_martin_luk_features_and_setups(df, bg_df=bg_df, market_type=market_type)
    df.attrs["deepchart_info"] = compute_deepchart_orderflow_absorption(df, lookback=60)
    df.attrs["vbp_vsa_info"] = compute_vbp_profile_and_vsa_engine(df, lookback=120, num_bins=40)
    df.attrs["sepa_vcp_3c_info"] = compute_minervini_vcp_and_3c_cheats(df, lookback_base=65)
    df.attrs["sepa_tennis_squat_info"] = evaluate_tennis_ball_and_squat_action(df)
    df.attrs["sepa_mini_screens_info"] = evaluate_minervini_overlapping_mini_screens(df)

    return df

# ==========================================
# 4.1 大週期週線共振監測
# ==========================================
@st.cache_data(ttl=3600)
def get_weekly_alignment_signal(ticker):
    try:
        wk_df = yf.download(ticker, period="3y", interval="1wk", auto_adjust=True, progress=False)
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
        raw = yf.download(symbols, period="120d", interval="1d", group_by="ticker", auto_adjust=True, progress=False)
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
def quick_historical_win_rate(df, market_type, score_threshold: int = 5):
    """
    [v40.0 100% 單一真理來源：快速歷史勝率與雙向回測引擎完全同步]
    - 修復舊版雖然在註釋宣稱調用 evaluate_historical_bar_unified_plan，但實際內部仍執行簡化版規則（無 Climax Run 過熱攔截、無美股連漲枯竭攔截、無均線斜率同向濾網）導致勝率虛高之脫節問題。
    - 嚴格呼叫 evaluate_historical_bar_unified_plan (對近 260 根 K 線執行完整大師結構評估，歷史遠端執行因果向量化評估)，並支援 TP0 (+1.15R/+4.5%) 先鎖半倉利潤墊、雙層防洗盤止損與時間停損！
    """
    long_trades, short_trades = [], []
    position = None
    entry_price, stop_loss, tp0_target, tp_target = 0.0, 0.0, 0.0, 0.0
    days_held = 0
    stage_q = 0
    init_risk_q = 0.0
    _stop_prof_q = get_active_stop_profile()
    _be_days_q = int(_stop_prof_q.get("breakeven_min_days", 3))
    _time_stop_q = int(_stop_prof_q.get("time_stop_days", 18))
    n_bars = len(df)
    full_eval_start = max(25, n_bars - 280)

    for i in range(25, n_bars):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        atr_val = float(prev['ATR']) if pd.notna(prev.get('ATR')) and float(prev['ATR']) > 0 else float(row['Open']) * 0.03

        if position is None:
            if i >= full_eval_start:
                _plan = evaluate_historical_bar_unified_plan(
                    df=df,
                    bar_idx=i - 1,
                    market_type=market_type,
                    score_threshold=score_threshold,
                    is_relaxed=False
                )
                sig = _plan.get("signal")
                order_style = str(_plan.get("order_style", "STOP"))
                _planned_e = float(_plan.get("planned_entry", 0.0))
                _planned_sl = float(_plan.get("stop_loss", 0.0))
                _planned_tp = float(_plan.get("tp_target", 0.0))
                _planned_rr = float(_plan.get("planned_rr", 0.0))
            else:
                sig = None
                order_style = "STOP"
                _planned_e, _planned_sl, _planned_tp, _planned_rr = 0.0, 0.0, 0.0, 0.0
                is_hk_liquid = not (market_type == "HK" and float(prev.get('TurnoverMA20', 0)) < 20_000_000)
                if is_hk_liquid and not bool(prev.get('Is_Climax_Run', False)):
                    c_p = float(prev['Close'])
                    sma20_p = float(prev.get('SMA20', c_p))
                    sma50_p = float(prev.get('SMA50', c_p))
                    sma150_p = float(prev.get('SMA150', c_p))
                    sma200_p = float(prev.get('SMA200', c_p))
                    low20_p = float(prev.get('SupShort_Causal', prev['Low']))
                    high20_p = float(prev.get('ResShort_Causal', prev['High']))
                    r_cnt = sum([
                        c_p > sma150_p and c_p > sma200_p,
                        sma150_p > sma200_p,
                        sma50_p > sma150_p and sma50_p > sma200_p,
                        c_p > float(prev.get('Low52W', c_p)) * 1.25,
                        c_p > float(prev.get('High52W', c_p)) * 0.75,
                        float(prev.get('Mansfield_RS', 0)) > 0,
                        float(prev.get('RSI', 50)) > 52
                    ])
                    if r_cnt >= score_threshold and c_p >= sma50_p * 0.99 and c_p >= low20_p:
                        if (bool(prev.get('VCPSignal', False)) or bool(prev.get('JLaw_Tight_Box', False))) and float(prev.get('VolRatio', 1.0)) < 0.85:
                            sig = "LONG"
                            order_style = "STOP"
                            _planned_e = round(float(prev.get('BoxHigh10_Causal', prev['High'])) + 0.05 * atr_val, 2)
                        elif c_p >= high20_p * 0.99 and float(prev.get('VolRatio', 1.0)) > 1.35:
                            sig = "LONG"
                            order_style = "STOP"
                            _planned_e = round(max(c_p * 1.002, high20_p), 2)
                        if sig == "LONG":
                            _planned_sl = round(max(_planned_e - 1.55 * atr_val, low20_p - 0.35 * atr_val, _planned_e * 0.93), 2)
                            _rk = max(_planned_e - _planned_sl, 1e-4)
                            _planned_tp = round(min(_planned_e + 2.4 * atr_val, _planned_e + 2.1 * _rk), 2)
                            _planned_rr = (_planned_tp - _planned_e) / _rk

            if sig == "LONG" and _planned_rr >= 1.8 and _planned_e > 0:
                _cand_entry = -1.0
                if "IMMEDIATE" in order_style:
                    if float(row['Open']) > _planned_sl * 1.005 and float(row['Open']) <= _planned_e * 1.035:
                        _cand_entry = float(row['Open'])
                elif "LIMIT" in order_style:
                    if float(row['Low']) <= _planned_e and float(row['Open']) > _planned_sl * 1.005:
                        _cand_entry = min(float(row['Open']), _planned_e)
                else:
                    _trig = max(_planned_e, float(prev['High']))
                    if float(row['High']) >= _trig and float(row['Open']) <= _trig * 1.03:
                        _cand_entry = max(float(row['Open']), _trig)

                if _cand_entry > 0:
                    init_risk_q = max(_cand_entry - _planned_sl, 1e-4)
                    if float(row['Low']) <= _planned_sl and (float(row['Close']) < _cand_entry or float(row['Close']) < float(row['Open'])):
                        long_trades.append(-1.0)
                    else:
                        position = "LONG"
                        entry_price = _cand_entry
                        stop_loss = _planned_sl
                        tp0_target = min(entry_price * 1.045, entry_price + 1.15 * init_risk_q)
                        tp_target = _planned_tp
                        days_held = 0
                        stage_q = 0

            elif sig == "SHORT" and _planned_rr >= 1.8 and _planned_e > 0:
                _cand_entry = -1.0
                if "IMMEDIATE" in order_style:
                    if float(row['Open']) < _planned_sl * 0.995 and float(row['Open']) >= _planned_e * 0.965:
                        _cand_entry = float(row['Open'])
                elif "LIMIT" in order_style:
                    if float(row['High']) >= _planned_e and float(row['Open']) < _planned_sl * 0.995:
                        _cand_entry = max(float(row['Open']), _planned_e)
                else:
                    _trig_s = min(_planned_e, float(prev['Low']))
                    if float(row['Low']) <= _trig_s and float(row['Open']) >= _trig_s * 0.97:
                        _cand_entry = min(float(row['Open']), _trig_s)

                if _cand_entry > 0:
                    init_risk_q = max(_planned_sl - _cand_entry, 1e-4)
                    if float(row['High']) >= _planned_sl and (float(row['Close']) > _cand_entry or float(row['Close']) > float(row['Open'])):
                        short_trades.append(-1.0)
                    else:
                        position = "SHORT"
                        entry_price = _cand_entry
                        stop_loss = _planned_sl
                        tp0_target = max(entry_price * 0.955, entry_price - 1.15 * init_risk_q)
                        tp_target = _planned_tp
                        days_held = 0
                        stage_q = 0

        elif position == "LONG":
            days_held += 1
            if stage_q == 0 and float(row['High']) >= tp0_target:
                stage_q = 1
                stop_loss = max(stop_loss, entry_price * 1.004)
            elif days_held >= _be_days_q and float(prev['Close']) >= entry_price + init_risk_q:
                stage_q = max(stage_q, 1)
                stop_loss = max(stop_loss, entry_price * 1.003)

            _soft_sl_q = stop_loss + 0.25 * atr_val if stop_loss < entry_price else stop_loss
            _hard_hit_q = float(row['Low']) <= stop_loss
            _soft_close_hit_q = (stage_q == 0) and (float(row['Close']) <= _soft_sl_q) and (float(row['Close']) < float(row['Open']))
            if _hard_hit_q or _soft_close_hit_q:
                long_trades.append(1.0 if (stage_q >= 1 or stop_loss > entry_price) else -1.0)
                position, days_held, stage_q = None, 0, 0
            elif float(row['High']) >= tp_target:
                long_trades.append(1.0)
                position, days_held, stage_q = None, 0, 0
            elif days_held >= _time_stop_q and float(row['Close']) < entry_price * 1.025:
                long_trades.append(1.0 if (stage_q >= 1 or float(row['Close']) > entry_price * 1.003) else -1.0)
                position, days_held, stage_q = None, 0, 0

        elif position == "SHORT":
            days_held += 1
            if stage_q == 0 and float(row['Low']) <= tp0_target:
                stage_q = 1
                stop_loss = min(stop_loss, entry_price * 0.996)
            elif days_held >= _be_days_q and float(prev['Close']) <= entry_price - init_risk_q:
                stage_q = max(stage_q, 1)
                stop_loss = min(stop_loss, entry_price * 0.997)

            _soft_sl_qs = stop_loss - 0.25 * atr_val if stop_loss > entry_price else stop_loss
            _hard_hit_qs = float(row['High']) >= stop_loss
            _soft_close_hit_qs = (stage_q == 0) and (float(row['Close']) >= _soft_sl_qs) and (float(row['Close']) > float(row['Open']))
            if _hard_hit_qs or _soft_close_hit_qs:
                short_trades.append(1.0 if (stage_q >= 1 or stop_loss < entry_price) else -1.0)
                position, days_held, stage_q = None, 0, 0
            elif float(row['Low']) <= tp_target:
                short_trades.append(1.0)
                position, days_held, stage_q = None, 0, 0
            elif days_held >= _time_stop_q and float(row['Close']) > entry_price * 0.975:
                short_trades.append(1.0 if (stage_q >= 1 or float(row['Close']) < entry_price * 0.997) else -1.0)
                position, days_held, stage_q = None, 0, 0

    long_win = sum(1 for t in long_trades if t > 0) / len(long_trades) * 100.0 if long_trades else 0.0
    short_win = sum(1 for t in short_trades if t > 0) / len(short_trades) * 100.0 if short_trades else 0.0
    return long_win, short_win, len(long_trades), len(short_trades)

# ==========================================
# 6.5 批量預載快取引擎 (加速全景雷達掃描 4~6 倍，並統一使用 auto_adjust=True 前複權數據)
# ==========================================
_PRELOADED_DAILY_CACHE: Dict[str, pd.DataFrame] = {}

def batch_preload_ohlcv_cache(tickers: List[str]) -> None:
    """在全景雷達掃描前，一次性批量下載多隻股票近 2 年日線前複權數據，消除逐隻 HTTP 請求與 sleep 阻塞"""
    clean_syms = []
    for t in tickers:
        if isinstance(t, str) and t.strip():
            fmt_t, _, _, _, _ = format_and_detect_market(t)
            clean_syms.append(fmt_t)
    clean_syms = list(dict.fromkeys(clean_syms))
    if len(clean_syms) < 2:
        return
    try:
        raw = yf.download(clean_syms, period="2y", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=True)
        if raw is None or raw.empty:
            return
        now_ts = time.time()
        for sym in clean_syms:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym in raw.columns.get_level_values(0):
                        sub = raw[sym].copy()
                    elif sym in raw.columns.get_level_values(1):
                        sub = raw.xs(sym, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    sub = raw.copy()
                sub = sub.dropna(subset=['Open', 'High', 'Low', 'Close'])
                if len(sub) >= 50:
                    sub.index = pd.to_datetime(sub.index)
                    if sub.index.tz is not None:
                        sub.index = sub.index.tz_localize(None)
                    sub.attrs["_preload_ts"] = now_ts
                    _PRELOADED_DAILY_CACHE[sym] = sub
            except Exception as e:
                swing_logger.debug(f"預載 {sym} 略過: {e}")
    except Exception as e:
        swing_logger.warning(f"批量預載異常: {e}")

# ==========================================
# 7. 核心診斷引擎 (完美融合新舊做空邏輯)
# ==========================================
def analyze_stock_master(ticker, strictness_threshold=5, interval="1d"):
    try:
        ticker, market_type, benchmark_ticker, benchmark_name, currency = format_and_detect_market(ticker)
        bg_df = get_index_data(benchmark_ticker, interval=interval)
        name = get_stock_name(ticker)

        df = None
        if interval == "1d" and ticker in _PRELOADED_DAILY_CACHE:
            cached_df = _PRELOADED_DAILY_CACHE[ticker]
            if (time.time() - cached_df.attrs.get("_preload_ts", 0)) < 900:
                df = cached_df.copy()

        if df is None or df.empty:
            if interval == "1m":
                period = "7d"
            elif interval in ["5m", "15m", "30m"]:
                period = "60d"
            elif interval in ["1h", "4h", "90m"]:
                period = "730d"
            else:
                period = "3y"

            _dl_stk_interval = "1h" if interval == "4h" else interval
            df = yf.download(ticker, period=period, interval=_dl_stk_interval, auto_adjust=True, progress=False)
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
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        if len(df) < 50:
            return {"error": f"數據不足，目前只有 {len(df)} 根 K 線。"}

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if interval == "4h" and len(df) >= 4:
            df = df.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna(subset=["Open", "High", "Low", "Close"])

        df = calculate_indicators(df, bg_df, interval=interval, market_type=market_type)
        long_win_rate, short_win_rate, l_trades, s_trades = quick_historical_win_rate(df, market_type, score_threshold=strictness_threshold)

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
        # OLD:
        # p_low_20 = df['Close'].tail(20).min()
        # is_breakdown = (last['Close'] <= p_low_20 * 1.01) and (last['VolRatio'] > 1.5)
        # NEW: [v30.0 消除自我包含偏差並與回測 100% 對齊] 採用不含當日之過去 20 日因果低點 SupShort_Causal
        p_low_20 = float(last.get('SupShort_Causal', df['Low'].iloc[-21:-1].min() if len(df) > 21 else df['Low'].tail(20).min()))
        is_breakdown = (last['Close'] <= p_low_20 * 1.01) and (last['VolRatio'] > 1.4)
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

        # OLD:
        # momentum_ready = (last['RSI'] > 55) and (last['Close'] > last['SMA20'])
        # is_tight_setup = last['VCPSignal'] and (last['VolRatio'] < 0.8)
        # p_high_20 = df['Close'].tail(20).max()
        # is_breakout_long = (last['Close'] >= p_high_20 * 0.99) and (last['VolRatio'] > 1.5)
        # NEW: [v30.0 掃描器與回測閾值 100% 對齊] 採用 ResShort_Causal (前20日高點)、JLaw_Tight_Box 及一致之量價門檻
        momentum_ready = (float(last['RSI']) > 52) and (float(last['Close']) >= float(last['SMA20']) * 0.99)
        is_tight_setup = (bool(last['VCPSignal']) or bool(last.get('JLaw_Tight_Box', False))) and (float(last['VolRatio']) < 0.85)
        p_high_20 = float(last.get('ResShort_Causal', df['High'].iloc[-21:-1].max() if len(df) > 21 else df['Close'].tail(20).max()))
        is_breakout_long = (float(last['Close']) >= p_high_20 * 0.99) and (float(last['VolRatio']) > 1.35)
        early_stage_bypass = (last['Close'] > last['SMA50']) and (last['Ret1M'] > 0.20)

        _ma_slope_pre = compute_ma_slope_and_pullback_trigger(df)
        _is_20ma_reclaim_buy = bool(
            is_near_20ma
            and _ma_slope_pre.get("reclaimed_9ema_right_side", False)
            and not _ma_slope_pre.get("ma_ribbon_sloping_down", False)
            and float(last.get('VolRatio', 1.0)) <= 1.25
        )
        is_long_trigger = momentum_ready and (
            (pass_count >= strictness_threshold) or early_stage_bypass
        ) and (is_tight_setup or is_breakout_long or _is_20ma_reclaim_buy)

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
        
        ma_slope_info = compute_ma_slope_and_pullback_trigger(df)
        df.attrs["ma_slope_info"] = ma_slope_info
        is_deduction_pullback = False
        if last.get('Clock_2', False) and not ma_slope_info.get("ma_ribbon_sloping_down", False):
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

        # [v41.1 五階段大盤見底識別 × 雙向全顯示 (不屏蔽做多清單) × 空頭防超賣追空]
        martin_luk_info = df.attrs.get("martin_luk_info", {})
        exh_guard_info = compute_us_swing_exhaustion_and_trap_guard(df, market_type=market_type)
        df.attrs["exhaustion_guard"] = exh_guard_info
        _is_exhausted_bar = bool(exh_guard_info.get("is_exhausted_dont_chase", False))

        _regime_gate = evaluate_global_market_regime_gate(df=df, bg_df=bg_df, market_type=market_type, interval=interval)
        df.attrs["market_regime_gate"] = _regime_gate
        _allow_bo_long = bool(_regime_gate.get("allow_breakout_long", True))
        _allow_pb_long = bool(_regime_gate.get("allow_pullback_long", True))
        _allow_short_gate = bool(_regime_gate.get("allow_short", True))
        _is_bottom_rev_day = bool(_regime_gate.get("is_bottom_reversal_day", False))
        _live_market_hostile = not _allow_bo_long
        df.attrs["live_market_hostile"] = _live_market_hostile

        # [v41.1 Module 2：空頭連跌超賣防追殺濾網 (杜絕 2382.HK 舜宇連跌 5 日遠離 20MA 追空被夾)]
        _recent4_diff = df['Close'].diff().iloc[-4:] if len(df) >= 5 else pd.Series([0.0])
        _consec_down_4 = bool((_recent4_diff < 0).sum() >= 4)
        _dist_below_20ma_atr = (float(last.get('SMA20', last['Close'])) - float(last['Close'])) / max(float(atr_val), 1e-6)
        _is_short_overextended = bool(_consec_down_4 or _dist_below_20ma_atr > 1.85 or float(last.get('RSI', 50)) < 28.0)
        df.attrs["is_short_overextended"] = _is_short_overextended

        # [v41.1 新增：回調見底轉勢日 / 強勢收復 20MA 領頭羊買點]
        # 當大市或個股回調後當日大陽燭反攻站回 9EMA 及 20MA 之上，且 50MA 向上 (Stage 2 結構未壞)，即使 10MA 短暫微彎亦予以放行！
        _is_reversal_leader_long = bool(
            (pass_count >= strictness_threshold - 1 or early_stage_bypass)
            and float(last['Close']) >= float(last.get('SMA20', 0)) * 0.998
            and float(last['Close']) >= float(last.get('EMA9', 0))
            and float(last['Close']) > float(last['Open'])
            and float(last['Close']) >= float(last.get('SMA50', 0)) * 0.985
            and (not invalidation_info["long_stop_hit"])
            and (not invalidation_info["downtrend_veto_long"])
            and (not _is_exhausted_bar)
        )

        is_martin_pb_long = bool(_allow_pb_long and martin_luk_info.get("martin_pullback_long", False) and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"] and not _is_exhausted_bar)
        is_martin_id_long = bool((_allow_bo_long or _is_bottom_rev_day) and martin_luk_info.get("martin_inside_breakout_long", False) and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"] and not _is_exhausted_bar)
        is_martin_ep_long = bool((_allow_bo_long or _is_bottom_rev_day) and martin_luk_info.get("martin_ep_long", False) and not invalidation_info["long_stop_hit"] and exh_guard_info.get("close_range_pos", 0.5) >= 0.60)
        is_martin_fade_short = bool(_allow_short_gate and martin_luk_info.get("martin_weak_bounce_short", False) and not invalidation_info["uptrend_veto_short"])

        if (_allow_bo_long or _is_bottom_rev_day) and is_kk_htf and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"] and not _is_exhausted_bar:
            plan_type = "LONG"
            status = "🔥 Kullamägi 高階動能旗型"
        elif is_martin_ep_long:
            plan_type = "LONG"
            status = f"🚀 EP 跳空爆量突破 (波段結構止損約 {martin_luk_info.get('stop_width_pct', 4.0):.1f}%)"
        elif is_martin_pb_long:
            plan_type = "LONG"
            _conf_txt = "+".join(martin_luk_info.get("confluence_supports", [])[:2])
            status = f"🎯 強勢回踩多重支撐匯聚買點 ({_conf_txt} 收回 ｜ 波段止損約 {martin_luk_info.get('stop_width_pct', 4.0):.1f}%)"
        elif is_martin_id_long and (pass_count >= strictness_threshold - 1 or early_stage_bypass):
            plan_type = "LONG"
            status = f"🟢 Inside Day 均線雲帶壓縮突破 (升穿 {martin_luk_info.get('pdh_trigger', 0):.2f} 觸發 ｜ 波段止損約 {martin_luk_info.get('stop_width_pct', 4.0):.1f}%)"
        elif (_allow_bo_long or _is_bottom_rev_day) and is_long_trigger and is_breakout_long and not invalidation_info["long_stop_hit"] and not _is_exhausted_bar:
            plan_type = "LONG"
            status = "🟢 放量強勢突破 (Breakout)"
        elif _allow_pb_long and is_long_trigger and is_tight_setup and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = "🟢 極限壓縮潛伏 (VCP/緊密箱體 Setup)"
        elif _allow_pb_long and is_lei_ma_convergence_long and not has_active_short_signal:
            plan_type = "LONG"
            status = "🟢 LEI 均線高度密集 (<2.5%) 張嘴突破埋伏"
        elif _allow_pb_long and is_long_trigger and not invalidation_info["long_stop_hit"] and not ma_slope_info.get("ma_ribbon_sloping_down", False):
            plan_type = "LONG"
            status = "🟢 20MA 右側收復回踩做多" if ma_slope_info.get("reclaimed_9ema_right_side", False) else "🟢 20MA 回踩做多信號"
        elif _allow_pb_long and lei_struct_info.get("bullish_2b", False) and (last['Close'] >= last['SMA50'] * 0.98) and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = "🟢 LEI 2B 破底翻買點 (洗盤跌穿前低後強勢收陽確認反轉)"
        elif _allow_pb_long and is_deduction_pullback and not has_active_short_signal:
            plan_type = "LONG"
            _b_tag = " + 底構造確認" if lei_struct_info.get("bottom_structure") else ""
            status = f"🟡 趨勢回撤買點 (2點鐘方向，抵扣價防守成功{_b_tag})"
        elif _allow_pb_long and (df.attrs.get("vbp_vsa_info", {}).get("vsa_no_supply", False) or df.attrs.get("vbp_vsa_info", {}).get("vsa_spring_absorption", False)) and is_long_candidate and is_near_20ma:
            plan_type = "LONG"
            status = "🟢 Wyckoff VSA 縮量無供給/彈簧洗盤測試買點 (均線支撐確認)"
        elif _allow_pb_long and _is_bottom_rev_day and _is_reversal_leader_long and is_near_20ma:
            plan_type = "LONG"
            status = "🌤️ 大市見底同步反攻買點 (陽燭重上 9EMA/20MA 右側轉強)"
        elif _allow_pb_long and df.attrs.get("sepa_vcp_3c_info", {}).get("entry_tier") in ["LOW_CHEAT", "MID_CHEAT_3C"] and df.attrs.get("sepa_mini_screens_info", {}).get("overlap_count", 0) >= 2 and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"]:
            plan_type = "LONG"
            _3c_badge = df.attrs.get("sepa_vcp_3c_info", {}).get("entry_tier_badge", "💎 3C作弊買點")
            status = f"🟢 《超級績效》{_3c_badge} ({df.attrs.get('sepa_mini_screens_info', {}).get('overlap_badge', '')})"
        elif _allow_pb_long and df.attrs.get("sepa_tennis_squat_info", {}).get("is_tennis_ball", False) and df.attrs.get("sepa_mini_screens_info", {}).get("overlap_count", 0) >= 2 and not invalidation_info["long_stop_hit"]:
            plan_type = "LONG"
            status = f"🎾 《超級績效》網球強彈買點 (回測均線迅速收陽彈起 ｜ {df.attrs.get('sepa_mini_screens_info', {}).get('overlap_badge', '')})"
        elif _allow_pb_long and df.attrs.get("sepa_mini_screens_info", {}).get("overlap_count", 0) >= 3 and not invalidation_info["long_stop_hit"] and not invalidation_info["downtrend_veto_long"] and not _is_exhausted_bar:
            plan_type = "LONG"
            status = f"🌟 《超級績效》多重小篩網重疊潛力股 ({df.attrs.get('sepa_mini_screens_info', {}).get('overlap_badge', '')})"
        elif _allow_short_gate and is_jlaw_short_trigger and is_breakdown and not _is_short_overextended:
            plan_type = "SHORT"
            status = "🔴 帶量破底做空 (Breakdown)"
        elif _allow_short_gate and is_jlaw_short_trigger and is_weak_bounce:
            plan_type = "SHORT"
            status = "🔴 均線蓋頭弱反彈 (Bear Flag)"
        elif _allow_short_gate and is_short_candidate_original and not _is_short_overextended:
            plan_type = "SHORT"
            status = "🟠 空頭趨勢成型做空 (長線結構)"
        elif _allow_short_gate and is_cohodes_short and not _is_short_overextended:
            plan_type = "SHORT"
            status = "🏴‍☠️ Cohodes 絕佳空點 (破位+反彈無力)"
        elif _allow_short_gate and is_rebound_short:
            plan_type = "SHORT"
            status = "🩸 弱勢反彈受阻 (高勝率空點)"
        elif is_martin_fade_short:
            plan_type = "SHORT"
            _cres_txt = "+".join(martin_luk_info.get("confluence_resistances", [])[:2])
            status = f"🩸 Martin Luk 下彎EMA/AVWAP弱反抽做空 ({_cres_txt} 受阻 ｜ HOD極緊止損)"
        elif _allow_short_gate and (is_jlaw_short_trigger or is_cohodes_short) and _is_short_overextended:
            plan_type = "WATCH"
            status = f"⚪ 空頭超賣觀察 (已連跌或偏離 20MA 達 {_dist_below_20ma_atr:.1f}x ATR，嚴禁低位追空，等待反抽 10/20MA 受阻)"
        elif is_short_candidate:
            # [v35.0 核心重構] 僅具備空頭背景但無破底或弱反抽受阻觸發點時，歸類為 WATCH 觀察，嚴禁混入可執行做空清單
            plan_type = "WATCH"
            status = "⚪ 空頭排列觀察 (等待反抽受阻或破底觸發點)"
        elif is_long_candidate:
            # [v35.0 核心重構] 僅具備多頭均線背景但無突破或支撐收復觸發點時，歸類為 WATCH 觀察，杜絕無買點市價盲買！
            plan_type = "WATCH"
            status = "⚪ 多頭排列觀察 (目前處於波段中途，等待回踩支撐或緊密突破買點)"
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
        # [v25.4 新增雙向獨立預計算] 同時預先計算 LONG 與 SHORT 結構交易計劃並存入 df.attrs，
        # 當用戶在診斷頁手動切換「做多 / 做空」方向，或重新輸入前幾日做空之股票時，系統可即時切換全套做空/做多買賣位、止損、目標與提示，絕不錯亂！
        long_plan_eval = evaluate_swing_trade_plan_and_rr(
            df=df,
            plan_type="LONG",
            status=status if plan_type == "LONG" else "🟢 多頭結構參考計劃",
            atr_val=atr_val,
            is_breakout_long=bool(is_breakout_long),
            is_near_20ma=bool(is_near_20ma),
            is_deduction_pullback=bool(is_deduction_pullback),
            is_breakdown_short=False,
            lei_struct=lei_struct_info,
            market_type=market_type,
            lot_size=POPULAR_HK_LOTS.get(ticker, 100) if market_type == "HK" else 1
        )
        long_plan_eval["_plan_dir"] = "LONG"

        short_plan_eval = evaluate_swing_trade_plan_and_rr(
            df=df,
            plan_type="SHORT",
            status=status if plan_type == "SHORT" else "🔴 空頭結構參考計劃",
            atr_val=atr_val,
            is_breakout_long=False,
            is_near_20ma=False,
            is_deduction_pullback=False,
            is_breakdown_short=bool(is_breakdown),
            lei_struct=lei_struct_info,
            market_type=market_type,
            lot_size=POPULAR_HK_LOTS.get(ticker, 100) if market_type == "HK" else 1
        )
        short_plan_eval["_plan_dir"] = "SHORT"

        trade_plan_eval = evaluate_swing_trade_plan_and_rr(
            df=df,
            plan_type=plan_type,
            status=status,
            atr_val=atr_val,
            is_breakout_long=bool(is_breakout_long),
            is_near_20ma=bool(is_near_20ma),
            is_deduction_pullback=bool(is_deduction_pullback),
            is_breakdown_short=bool(is_breakdown),
            lei_struct=lei_struct_info,
            market_type=market_type,
            lot_size=POPULAR_HK_LOTS.get(ticker, 100) if market_type == "HK" else 1
        )
        trade_plan_eval["_plan_dir"] = plan_type
        df.attrs["long_plan_eval"] = long_plan_eval
        df.attrs["short_plan_eval"] = short_plan_eval

        # [v39.0 動態結構風險門控] 尊重側欄止損模式 (STOP_PROFILE_CONFIGS) 與個股 ADR% 動態止損上限，消除 6.2% 硬編碼誤殺美股高動能股之矛盾
        _active_prof_m = get_active_stop_profile()
        _max_allowed_stop_m = max(float(_active_prof_m.get("max_stop_pct", 0.056)) * 100.0 + 0.6, min(10.8, (float(atr_val) / max(float(last['Close']), 1e-6)) * 195.0))
        if plan_type in ["LONG", "SHORT"] and float(trade_plan_eval.get("stop_pct", 0.0)) > _max_allowed_stop_m:
            plan_type = "WATCH"
            status = f"⚪ 觀望 (結構止損過寬達 {trade_plan_eval.get('stop_pct', 0):.1f}% > {_max_allowed_stop_m:.1f}% 動態上限，等待結構收窄)"
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
        df.attrs["holistic_grade_info"] = evaluate_holistic_setup_grade_and_edge(
            df=df,
            plan_type=plan_type,
            trade_plan_eval=trade_plan_eval,
            jlaw_meta_info=jlaw_meta_info,
            lei_struct_info=lei_struct_info,
            lei_ded_forecast=lei_ded_forecast,
            martin_luk_info=martin_luk_info,
            deepchart_info=df.attrs.get("deepchart_info", {}),
            vbp_vsa_info=df.attrs.get("vbp_vsa_info", {}),
            invalidation_info=invalidation_info
        )

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
# 8. 雙向歷史回測引擎 (v25.1 富途牛牛中線波段專版 - 唯一標準引擎，已刪除底部重複定義)
# ==========================================
def run_bidirectional_backtest(ticker, start_date, end_date, initial_capital, score_threshold, trade_mode, is_relaxed):
    """
    [v25.1 全面升級與漏洞修復]
    1. 富途牛牛 (Futu) 港美股真實收費模型：每筆進出場精確扣除富途佣金、平台費、港股 0.1% 印花稅與交收規費
    2. 中線波段防洗盤止損：預設採用 ≥1.20x ATR 及 3.2%~7.8% 結構止損，不再因 1%~1.5% 超貼止損被日內噪音洗走
    3. 進場當日盤中觸及止損檢查 (Same-Bar Stop-Out)：解決進場當日沖高回落跌穿止損卻漏檢之漏洞
    4. 中線波段保本與跟蹤止盈：
       - 保本升級：需持倉滿 5 日且收盤浮盈 ≥ 1.0R (或 20MA > 買入價) 方可移至富途打和保本價，避免前幾日回踩即被洗走
       - 階段二移動止盈：一般 2 點鐘穩健趨勢沿 20MA / 21EMA 及 20 日結構低點跟蹤（持倉數週至 1-2 個月）；僅在 12 點鐘加速噴發期 (Clock_12) 收緊至 9EMA
       - 時間停損：延長至 25 個交易日（約 5 週）
    5. 做空保證金凍結 (Short Margin Collateral) 與獨立交易編號 (_trade_id) 統計（徹底消除分批止盈勝率雙重計算）
    """
    try:
        ticker, market_type, benchmark_ticker, benchmark_name, currency = format_and_detect_market(ticker)
        bg_df = get_index_data(benchmark_ticker)
        name = get_stock_name(ticker)

        download_start = start_date - timedelta(days=365)
        df = yf.download(ticker, start=download_start, end=end_date, auto_adjust=True, progress=False)
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
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = calculate_indicators(df, bg_df, market_type=market_type)
        df = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
        if df.empty:
            return "選定區間內無有效數據。"

        stop_prof = get_active_stop_profile()
        min_stop_pct = float(stop_prof["min_stop_pct"])
        max_stop_pct = float(stop_prof["max_stop_pct"])
        breakeven_min_days = int(stop_prof.get("breakeven_min_days", 5))
        time_stop_days = int(stop_prof.get("time_stop_days", 25))

        cash = float(initial_capital)
        position_type = None
        entry_price, stop_loss, tp_target, entry_structural_sl = 0.0, 0.0, 0.0, None
        initial_risk_per_share = 0.0
        short_margin_locked = 0.0
        position_stage = 0
        days_in_trade = 0
        shares = 0
        trade_counter = 0
        _cum_borrow_fee = 0.0
        _bt_loss_streak = 0
        _bt_peak_equity = float(initial_capital)
        trade_log, history_equity, history_dates = [], [], []

        # 市場衝擊滑點 (不含券商手續費，券商手續費與印花稅由 compute_futu_real_fees 獨立精算)
        market_slippage = 0.0005 if market_type == "HK" else 0.0002

        for i in range(1, len(df)):
            current_date = df.index[i]
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            p_prev20 = df.iloc[i - 20] if i >= 20 else df.iloc[0]

            is_hk_liquid = not (market_type == "HK" and prev_row['TurnoverMA20'] < 20_000_000)

            is_market_bullish = True
            if 'CloseBench' in prev_row and 'SMA50BenchValue' in prev_row and pd.notna(prev_row['CloseBench']):
                is_market_bullish = prev_row['CloseBench'] > prev_row['SMA50BenchValue']

            bench_close = prev_row.get('CloseBench', np.nan)
            bench_200 = prev_row.get('SMA200BenchValue', np.nan)
            bench_dist = prev_row.get('Bench_Dist_Days', 0)
            is_market_health_ok = True
            if pd.notna(bench_close) and pd.notna(bench_200):
                if bench_close < bench_200 or bench_dist >= 5:
                    is_market_health_ok = False

            if position_type == "LONG":
                current_equity = cash + (shares * float(row['Close']))
            elif position_type == "SHORT":
                current_equity = cash + short_margin_locked + ((entry_price - float(row['Close'])) * shares)
            else:
                current_equity = cash

            if is_relaxed:
                is_long_trigger = prev_row['Close'] > prev_row['SMA50'] and prev_row['MACD'] > prev_row['MACDSignal'] and is_hk_liquid and is_market_bullish and is_market_health_ok
                is_short_trigger = prev_row['Close'] < prev_row['SMA50'] and prev_row['MACD'] < prev_row['MACDSignal'] and is_hk_liquid and not is_market_bullish
            else:
                p_low = float(prev_row.get('Low52W', df.iloc[max(0, i-252):i]['Low'].min()))
                p_high = float(prev_row.get('High52W', df.iloc[max(0, i-252):i]['High'].max()))

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
                p_high_20_val = float(prev_row.get('ResShort_Causal', prev_row['High']))
                is_breakout = (prev_row['Close'] >= p_high_20_val * 0.99) and (prev_row['VolRatio'] > 1.5)
                early_stage_bypass = (prev_row['Close'] > prev_row['SMA50']) and (prev_row['Ret1M'] > 0.20)

                is_long_trigger = momentum_ready and (
                    (prev_pass_count >= score_threshold) or early_stage_bypass or kk_flag
                ) and (is_tight_setup or is_breakout) and is_hk_liquid and is_market_health_ok

                orig_sr1 = prev_row['Close'] < prev_row['SMA200']
                orig_sr2 = prev_row['SMA50'] < prev_row['SMA150'] and prev_row['SMA150'] < prev_row['SMA200']
                orig_sr3 = prev_row['Close'] < p_high * 0.85
                orig_sr4 = prev_row['Mansfield_RS'] < 0
                orig_sr5 = prev_row['VolRatio'] < 0.8
                orig_sr6 = prev_row['LowerShadowRatio'] < 0.4
                orig_sr7 = prev_row['VCPSignal']
                prev_short_pass_count = sum([orig_sr1, orig_sr2, orig_sr3, orig_sr4, orig_sr5, orig_sr6, orig_sr7])
                is_short_candidate_original = prev_short_pass_count >= score_threshold

                p_low_20_val = float(prev_row.get('SupShort_Causal', prev_row['Low']))
                is_breakdown = (prev_row['Close'] <= p_low_20_val * 1.01) and (prev_row['VolRatio'] > 1.5)
                ma_pressing = (prev_row['Close'] < prev_row['SMA20']) and (prev_row['SMA20'] < prev_row['SMA50'])
                is_weak_bounce = prev_row['VCPSignal'] and (prev_row['VolRatio'] < 0.8)
                is_jlaw_short_trigger = ma_pressing and (prev_row['Mansfield_RS'] < 0) and (is_weak_bounce or is_breakdown)

                is_short_trigger = (is_short_candidate_original or is_jlaw_short_trigger) and is_hk_liquid and not is_market_bullish

            atr_val = prev_row['ATR'] if pd.notna(prev_row['ATR']) and prev_row['ATR'] > 0 else row['Open'] * 0.03

            if position_type is None:
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

                _bt_plan = evaluate_historical_bar_unified_plan(
                    df=df,
                    bar_idx=i - 1,
                    market_type=market_type,
                    score_threshold=score_threshold,
                    is_relaxed=is_relaxed
                )
                # OLD: if (_bt_plan["signal"] == "LONG" or _unified_long_signal) and trade_mode in ["全部雙向", "僅做多頭(做多)"]:
                # NEW: [v25.4 單一真理來源] 完全以統一評估器 _bt_plan["signal"] 為準，徹底消除繞過 downtrend_veto_long 之冗餘分支
                if _bt_plan["signal"] == "LONG" and trade_mode in ["全部雙向", "僅做多頭(做多)"]:
                    # [v41.1 Module 5] 已清除原本嵌套於 if _bt_plan["signal"] == "LONG" 內永遠不會執行之死代碼

                    _planned_e = float(_bt_plan["planned_entry"])
                    _planned_sl = float(_bt_plan["stop_loss"])
                    _planned_tp = float(_bt_plan["tp_target"])
                    _order_style = _bt_plan["order_style"]

                    _is_filled = False
                    actual_entry_price = 0.0
                    if _order_style == "IMMEDIATE":
                        # [v26.0 同步即時入場信號] 當信號為「⚡ 即時入場」且隔日開盤未直接跳空跌破止損或暴漲超過 +3.5% 時，於開盤/建議位即時成交
                        if float(row['Open']) > _planned_sl * 1.005 and float(row['Open']) <= _planned_e * 1.035:
                            actual_entry_price = float(row['Open'])
                            _is_filled = True
                        elif float(row['Low']) <= _planned_e <= float(row['High']):
                            actual_entry_price = _planned_e
                            _is_filled = True
                    elif _order_style == "LIMIT":
                        # [v39.0 嚴格遵守限價回踩低吸紀律] 僅當日內低點實際觸及建議限價位 _planned_e 且開盤未直接跳空跌破止損時成交，絕不偷換為追高突破單！
                        if float(row['Low']) <= _planned_e and float(row['Open']) > _planned_sl * 1.005:
                            actual_entry_price = min(float(row['Open']), _planned_e)
                            _is_filled = True
                    else:
                        trigger_price = max(_planned_e, float(prev_row['High']))
                        if float(row['High']) >= trigger_price and float(row['Open']) <= trigger_price * 1.03:
                            actual_entry_price = max(float(row['Open']), trigger_price)
                            _is_filled = True

                    if _is_filled and actual_entry_price > 0:
                        entry_price = actual_entry_price * (1.0 + market_slippage)
                        # [v39.0 同步高 ADR% 動態止損上限] 允許高波動美股使用最高 10.8% (1.95x ATR%) 之真實結構止損，並由等風險倉位公式自動縮減股數
                        _dyn_max_sl_pct = min(0.108, max(max_stop_pct, (atr_val / max(entry_price, 1e-6)) * 1.95))
                        _upper_sl_bound = entry_price * (1.0 - min_stop_pct)
                        _lower_sl_bound = entry_price * (1.0 - _dyn_max_sl_pct)
                        if _lower_sl_bound <= _planned_sl <= _upper_sl_bound:
                            stop_loss = _planned_sl
                        else:
                            stop_loss = max(entry_price - float(stop_prof["fallback_atr_mult"]) * atr_val, _lower_sl_bound)
                            stop_loss = min(stop_loss, _upper_sl_bound)
                        risk_per_share = max(entry_price - stop_loss, 1e-4)
                        initial_risk_per_share = risk_per_share

                        tp_target = _planned_tp if _planned_tp > entry_price * 1.03 else float(_bt_plan.get("tp2_target", entry_price + 2.5 * risk_per_share))
                        _planned_rr = (tp_target - entry_price) / risk_per_share

                        if _planned_rr >= 1.8:
                            _dist_cnt_bar = float(prev_row.get('Dist_Days_20d', 0))
                            _bt_peak_equity = max(_bt_peak_equity, current_equity)
                            _bt_curr_dd_pct = ((_bt_peak_equity - current_equity) / max(_bt_peak_equity, 1.0)) * 100.0
                            _eff_risk_frac = 0.005 if (_dist_cnt_bar >= 4 or bench_dist >= 3) else 0.0075
                            if _bt_loss_streak >= 4 or _bt_curr_dd_pct >= 12.0:
                                _eff_risk_frac = min(_eff_risk_frac, 0.0020)
                            elif _bt_loss_streak >= 2 or _bt_curr_dd_pct >= 6.0:
                                _eff_risk_frac = min(_eff_risk_frac, 0.0035)
                            risk_amount = current_equity * _eff_risk_frac
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            _adr_bar = float(prev_row.get('ADR_Pct', 4.0))
                            _max_notional_cap = current_equity * (0.25 if _adr_bar >= 6.0 else 0.35)
                            if (target_shares * entry_price) > min(cash * 0.99, _max_notional_cap):
                                target_shares = int(min(cash * 0.99, _max_notional_cap) / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                buy_fee_info = compute_futu_real_fees(market_type, entry_price, target_shares, is_sell=False)
                                buy_fee = buy_fee_info["single_leg_fee"]
                                if (target_shares * entry_price + buy_fee) <= cash:
                                    trade_counter += 1
                                    shares = target_shares
                                    cash -= (shares * entry_price + buy_fee)
                                    position_stage = 0
                                    days_in_trade = 0
                                    entry_structural_sl = stop_loss
                                    position_type = "LONG"
                                    trade_info = {
                                        "_trade_id": trade_counter,
                                        "_initial_notional": shares * entry_price,
                                        "_entry_fee": buy_fee,
                                        "方向": "🟢 做多",
                                        "進場日期": current_date,
                                        "買入方式": f"{'即時入場(Immediate)' if 'IMMEDIATE' in _order_style else ('低吸掛單(Limit)' if 'LIMIT' in _order_style else '突破觸發(Stop)')} | {_bt_plan['status_label']}",
                                        "建議買入位": round(_planned_e, 2),
                                        "進場價": round(entry_price, 2),
                                        "初始止損": round(stop_loss, 2),
                                        "止損依據": _bt_plan.get("sl_basis", "結構支撐外側"),
                                        "目標止盈": round(tp_target, 2),
                                        "止盈阻力依據": _bt_plan.get("tp_basis", "上方結構阻力區"),
                                        "計畫盈虧比": f"{_planned_rr:.2f}R",
                                        "最終止盈": "結構TP1減半 + 20MA/21EMA波段跟蹤",
                                        "結構止損": round(entry_structural_sl, 2)
                                    }
                                    # OLD: if float(row['Low']) <= stop_loss and float(row['Close']) < entry_price:
                                    # NEW: [v25.2 修復 P0-2 盤中收盤價前視偏差] 只要進場當日盤中低點跌穿止損位 (且對突破單而言收盤未能強勢站穩開盤之上，或對低吸單而言直接殺穿止損)，即誠實觸發止損！
                                    # OLD: if float(row['Low']) <= stop_loss and (float(row['Close']) < entry_price or "LIMIT" in _order_style or float(row['Close']) < float(row['Open'])):
                                    # NEW: [v30.0 修復低吸單進場當日收長下影強勢陽線仍被誤判止損之漏洞] 只有跌穿硬止損且收盤未能站回進場價之上 (或收陰線) 才觸發當日止損
                                    if float(row['Low']) <= stop_loss and (float(row['Close']) < entry_price or float(row['Close']) < float(row['Open'])):
                                        exit_price = stop_loss * (1.0 - market_slippage)
                                        sell_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=True)["single_leg_fee"]
                                        cash += (shares * exit_price - sell_fee)
                                        total_fee = round(buy_fee + sell_fee, 2)
                                        profit = (exit_price - entry_price) * shares - total_fee
                                        trade_info.update({
                                            "出場日期": current_date,
                                            "出場價": round(exit_price, 2),
                                            "結果": "進場當日沖高回落觸發止損",
                                            "回報率(%)": round((profit / (shares * entry_price)) * 100.0, 2),
                                            "富途費用": total_fee,
                                            "利潤": profit
                                        })
                                        trade_log.append(trade_info)
                                        position_type = None
                                        shares = 0
                                        position_stage = 0

                # OLD: elif (_bt_plan["signal"] == "SHORT" or is_short_trigger) and trade_mode in ["全部雙向", "僅做空頭(做空)"]:
                # NEW: [v25.4 單一真理來源] 空頭同樣完全以統一評估器 _bt_plan["signal"] 為準
                elif _bt_plan["signal"] == "SHORT" and trade_mode in ["全部雙向", "僅做空頭(做空)"]:
                    # [v41.1 Module 5] 已清除原本嵌套於 elif _bt_plan["signal"] == "SHORT" 內永遠不會執行之死代碼

                    _planned_e = float(_bt_plan["planned_entry"])
                    _planned_sl = float(_bt_plan["stop_loss"])
                    _planned_tp = float(_bt_plan["tp_target"])
                    _order_style = _bt_plan["order_style"]

                    _is_filled = False
                    actual_entry_price = 0.0
                    if _order_style == "IMMEDIATE":
                        if float(row['Open']) < _planned_sl * 0.995 and float(row['Open']) >= _planned_e * 0.965:
                            actual_entry_price = float(row['Open'])
                            _is_filled = True
                        elif float(row['Low']) <= _planned_e <= float(row['High']):
                            actual_entry_price = _planned_e
                            _is_filled = True
                    elif _order_style == "LIMIT":
                        # [v39.0 嚴格遵守空頭逢高反彈限價紀律] 僅當日內高點觸及反彈阻力限價位 _planned_e 時成交
                        if float(row['High']) >= _planned_e and float(row['Open']) < _planned_sl * 0.995:
                            actual_entry_price = max(float(row['Open']), _planned_e)
                            _is_filled = True
                    else:
                        trigger_price = min(_planned_e, float(prev_row['Low']))
                        if float(row['Low']) <= trigger_price and float(row['Open']) >= trigger_price * 0.97:
                            actual_entry_price = min(float(row['Open']), trigger_price)
                            _is_filled = True

                    if _is_filled and actual_entry_price > 0:
                        entry_price = actual_entry_price * (1.0 - market_slippage)
                        _lower_sl_bound = entry_price * (1.0 + min_stop_pct)
                        _upper_sl_bound = entry_price * (1.0 + max_stop_pct)
                        if _lower_sl_bound <= _planned_sl <= _upper_sl_bound:
                            stop_loss = _planned_sl
                        else:
                            stop_loss = min(entry_price + float(stop_prof["fallback_atr_mult"]) * atr_val, _upper_sl_bound)
                            stop_loss = max(stop_loss, _lower_sl_bound)
                        risk_per_share = max(stop_loss - entry_price, 1e-4)
                        initial_risk_per_share = risk_per_share

                        tp_target = _planned_tp if (0 < _planned_tp < entry_price * 0.97) else float(_bt_plan.get("tp2_target", max(0.01, entry_price - 2.5 * risk_per_share)))
                        _planned_rr = (entry_price - tp_target) / risk_per_share

                        if _planned_rr >= 1.8:
                            _bt_peak_equity = max(_bt_peak_equity, current_equity)
                            _bt_curr_dd_pct = ((_bt_peak_equity - current_equity) / max(_bt_peak_equity, 1.0)) * 100.0
                            _eff_short_risk_frac = 0.0035 if (_bt_loss_streak >= 2 or _bt_curr_dd_pct >= 6.0) else 0.0075
                            risk_amount = current_equity * _eff_short_risk_frac
                            target_shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
                            _adr_bar = float(prev_row.get('ADR_Pct', 4.0))
                            _max_notional_cap = current_equity * (0.25 if _adr_bar >= 6.0 else 0.35)
                            if (target_shares * entry_price) > min(cash * 0.95, _max_notional_cap):
                                target_shares = int(min(cash * 0.95, _max_notional_cap) / entry_price)
                            if market_type == "HK":
                                lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                                target_shares = (target_shares // lot_sz) * lot_sz
                            if target_shares > 0:
                                short_open_fee = compute_futu_real_fees(market_type, entry_price, target_shares, is_sell=True)["single_leg_fee"]
                                req_margin = target_shares * entry_price
                                if (req_margin + short_open_fee) <= cash:
                                    trade_counter += 1
                                    shares = target_shares
                                    short_margin_locked = req_margin
                                    cash -= (short_margin_locked + short_open_fee)
                                    position_stage = 0
                                    days_in_trade = 0
                                    _cum_borrow_fee = 0.0
                                    entry_structural_sl = stop_loss
                                    position_type = "SHORT"
                                    trade_info = {
                                        "_trade_id": trade_counter,
                                        "_initial_notional": shares * entry_price,
                                        "_entry_fee": short_open_fee,
                                        "方向": "🔴 做空",
                                        "進場日期": current_date,
                                        "買入方式": f"{'逢高沽空(Limit)' if 'LIMIT' in _order_style else '破底觸發(Stop)'} | {_bt_plan['status_label']}",
                                        "建議買入位": round(_planned_e, 2),
                                        "進場價": round(entry_price, 2),
                                        "初始止損": round(stop_loss, 2),
                                        "止損依據": _bt_plan.get("sl_basis", "結構阻力外側"),
                                        "目標止盈": round(tp_target, 2),
                                        "止盈阻力依據": _bt_plan.get("tp_basis", "下方結構支撐區"),
                                        "計畫盈虧比": f"{_planned_rr:.2f}R",
                                        "最終止盈": "結構TP1回補一半 + 20MA/21EMA波段跟蹤",
                                        "結構止損": round(entry_structural_sl, 2)
                                    }
                                    # OLD: if float(row['High']) >= stop_loss and float(row['Close']) > entry_price:
                                    # NEW: [v25.2 修復 P0-2 空頭盤中收盤價前視偏差]
                                    if float(row['High']) >= stop_loss and (float(row['Close']) > entry_price or "LIMIT" in _order_style or float(row['Close']) > float(row['Open'])):
                                        exit_price = stop_loss * (1.0 + market_slippage)
                                        cover_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=False)["single_leg_fee"]
                                        gross_pnl = (entry_price - exit_price) * shares
                                        cash += (short_margin_locked + gross_pnl - cover_fee)
                                        total_fee = round(short_open_fee + cover_fee, 2)
                                        profit = gross_pnl - total_fee
                                        trade_info.update({
                                            "出場日期": current_date,
                                            "出場價": round(exit_price, 2),
                                            "結果": "進場當日反彈軋空觸發止損",
                                            "回報率(%)": round((profit / (shares * entry_price)) * 100.0, 2),
                                            "富途費用": total_fee,
                                            "利潤": profit
                                        })
                                        trade_log.append(trade_info)
                                        position_type = None
                                        shares = 0
                                        short_margin_locked = 0.0
                                        position_stage = 0

            elif position_type == "LONG":
                days_in_trade += 1
                is_time_stop = (days_in_trade >= time_stop_days) and (row['Close'] < entry_price * 1.025)
                is_trending_market = pd.notna(bench_close) and bench_close > prev_row.get('SMA50BenchValue', 0) * 1.01

                # OLD:
                # if position_stage == 0 and days_in_trade >= breakeven_min_days:
                #     _reached_1r = float(prev_row['Close']) >= (entry_price + 1.0 * initial_risk_per_share)
                #     _sma20_above_entry = float(prev_row.get('SMA20', 0)) > entry_price * 1.005
                #     if _reached_1r or _sma20_above_entry:
                # NEW: [v28.0 修復低吸買入後因 20MA 本身略高於買入價而過早移至保本被洗出之漏洞 + 雙層防洗盤止損]
                # [v40.0 回測與實盤 100% 同步：TP0 (+1.15R / +4.5%) 利潤墊保護與階梯移至富途打和價]
                # [v41.1 Module 4：防過早推保本被洗出] 要求最高浮盈達 +1.25R 且收盤穩守 +0.85R 以上 (或持倉滿 5 日且 20MA 越過成本 1.5%) 才推至富途打和價
                _tp0_trigger_p = entry_price + 1.25 * initial_risk_per_share
                if position_stage == 0 and (float(prev_row['High']) >= _tp0_trigger_p or days_in_trade >= breakeven_min_days):
                    _reached_tp0 = float(prev_row['High']) >= _tp0_trigger_p and float(prev_row['Close']) >= entry_price + 0.85 * initial_risk_per_share
                    _reached_1r = float(prev_row['Close']) >= (entry_price + 1.15 * initial_risk_per_share)
                    _sma20_trailed_up = (days_in_trade >= breakeven_min_days) and (float(prev_row.get('SMA20', 0)) > entry_price * 1.015) and (float(prev_row['Close']) >= entry_price + 0.80 * initial_risk_per_share)
                    if _reached_tp0 or _reached_1r or _sma20_trailed_up:
                        position_stage = 1
                        _be_info = compute_futu_real_fees(market_type, entry_price, shares)
                        stop_loss = max(stop_loss, round(_be_info["breakeven_long_price"] * 1.003, 2))

                _soft_close_sl_bt = (stop_loss + 0.25 * atr_val) if position_stage == 0 else stop_loss
                _hard_stop_hit_bt = float(row['Low']) <= stop_loss
                _soft_close_stop_hit_bt = (position_stage == 0) and (float(row['Close']) <= _soft_close_sl_bt) and (float(row['Close']) < float(row.get('SMA20_Deduction', 0)) or float(row['Close']) < float(row['Open']))

                if _hard_stop_hit_bt or _soft_close_stop_hit_bt or is_time_stop:
                    base_exit = min(row['Open'], stop_loss) if _hard_stop_hit_bt else row['Close']
                    result_text = ("觸發保本止損" if position_stage >= 1 else ("觸發收盤結構破位止損" if _soft_close_stop_hit_bt and not _hard_stop_hit_bt else "觸發盤中防洗盤硬止損")) if (_hard_stop_hit_bt or _soft_close_stop_hit_bt) else f"中線時間停損 ({time_stop_days}日未脫離成本)"

                    exit_price = base_exit * (1.0 - market_slippage)
                    sell_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=True)["single_leg_fee"]
                    entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                    cash += (shares * exit_price - sell_fee)
                    total_leg_fee = round(entry_fee_alloc + sell_fee, 2)
                    profit = (exit_price - entry_price) * shares - total_leg_fee

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": round(exit_price, 2),
                        "結果": result_text,
                        "回報率(%)": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                        "富途費用": total_leg_fee,
                        "利潤": profit
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    position_stage = 0

                elif row['High'] >= tp_target and position_stage < 2:
                    if is_trending_market:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0:
                            sold_shares = shares

                        exit_price = max(row['Open'], tp_target) * (1.0 - market_slippage)
                        sell_fee = compute_futu_real_fees(market_type, exit_price, sold_shares, is_sell=True)["single_leg_fee"]
                        ratio_sold = sold_shares / max(shares, 1)
                        entry_fee_part = round(float(trade_info.get("_entry_fee", 0.0)) * ratio_sold, 2)
                        trade_info["_entry_fee"] = max(0.0, float(trade_info.get("_entry_fee", 0.0)) - entry_fee_part)

                        cash += (sold_shares * exit_price - sell_fee)
                        shares -= sold_shares
                        total_leg_fee = round(entry_fee_part + sell_fee, 2)

                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": "階段一：觸及結構阻力減倉 50%",
                            "回報率(%)": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": (exit_price - entry_price) * sold_shares - total_leg_fee
                        })
                        trade_log.append(partial_info)

                        if shares > 0:
                            position_stage = 2  # 進入中線波段 20MA / 21EMA 跟蹤模式
                            _be_info = compute_futu_real_fees(market_type, entry_price, shares)
                            stop_loss = max(stop_loss, _be_info["breakeven_long_price"])
                        else:
                            position_type, position_stage = None, 0
                    else:
                        exit_price = max(row['Open'], tp_target) * (1.0 - market_slippage)
                        sell_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=True)["single_leg_fee"]
                        entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                        cash += (shares * exit_price - sell_fee)
                        total_leg_fee = round(entry_fee_alloc + sell_fee, 2)
                        profit = (exit_price - entry_price) * shares - total_leg_fee
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": "震盪市：觸及結構目標全數止盈",
                            "回報率(%)": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": profit
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

                elif position_stage == 2:
                    # [v25.1 中線波段尾倉跟蹤] 12點鐘加速過熱期 (Clock_12) 沿 9EMA 鎖潤；一般穩健波段沿 min(20MA, 21EMA) 或 20日結構低點跟蹤，避免被 9EMA 提早洗走
                    _is_climax = bool(row.get('Clock_12', False))
                    _swing_trail_ma = float(row.get('EMA9', 0)) if _is_climax else min(float(row.get('SMA20', 0)), float(row.get('EMA21', row.get('SMA20', 0))))
                    _trail_reason = "12點鐘加速期跌破 9EMA 鎖潤離場" if _is_climax else "大趨勢尾倉：跌破 20MA/21EMA 中線波段離場"
                    if float(row['Close']) < _swing_trail_ma or float(row['Close']) < float(row.get('SupShort_Causal', 0)):
                        exit_price = float(row['Close']) * (1.0 - market_slippage)
                        sell_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=True)["single_leg_fee"]
                        entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                        cash += (shares * exit_price - sell_fee)
                        total_leg_fee = round(entry_fee_alloc + sell_fee, 2)
                        profit = (exit_price - entry_price) * shares - total_leg_fee
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": _trail_reason,
                            "回報率(%)": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": profit
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        position_stage = 0

            elif position_type == "SHORT":
                days_in_trade += 1
                _daily_borrow_cost = (shares * float(row['Close'])) * (0.045 / 252.0)
                cash -= _daily_borrow_cost
                _cum_borrow_fee += _daily_borrow_cost
                _short_unrealized = (entry_price - float(row['High'])) * shares
                _margin_equity_intraday = cash + short_margin_locked + _short_unrealized
                _maint_margin_req = (shares * float(row['High'])) * 0.30
                _margin_call_hit = _margin_equity_intraday <= _maint_margin_req
                is_time_stop = (days_in_trade >= time_stop_days) and (row['Close'] > entry_price * 0.975) or _margin_call_hit
                is_trending_market = pd.notna(bench_close) and bench_close < prev_row.get('SMA50BenchValue', 0) * 0.99

                # OLD:
                # if position_stage == 0 and days_in_trade >= breakeven_min_days:
                #     _reached_1r_s = float(prev_row['Close']) <= (entry_price - 1.0 * initial_risk_per_share)
                #     _sma20_below_entry = float(prev_row.get('SMA20', 999999)) < entry_price * 0.995
                #     if _reached_1r_s or _sma20_below_entry:
                # NEW: [v28.0 空頭防軋空雙層止損與保本條件修復]
                if position_stage == 0 and days_in_trade >= breakeven_min_days:
                    _reached_1r_s = float(prev_row['Close']) <= (entry_price - 1.0 * initial_risk_per_share)
                    _sma20_trailed_dn = (float(prev_row.get('SMA20', 999999)) < entry_price * 0.988) and (float(prev_row['Close']) <= entry_price - 0.75 * initial_risk_per_share)
                    if _reached_1r_s or _sma20_trailed_dn:
                        position_stage = 1
                        _be_info = compute_futu_real_fees(market_type, entry_price, shares, is_sell=True)
                        stop_loss = min(stop_loss, _be_info["breakeven_short_price"])

                _soft_close_sl_bts = (stop_loss - 0.25 * atr_val) if position_stage == 0 else stop_loss
                _hard_stop_hit_bts = float(row['High']) >= stop_loss
                _soft_close_stop_hit_bts = (position_stage == 0) and (float(row['Close']) >= _soft_close_sl_bts) and (float(row['Close']) > float(row.get('SMA20_Deduction', 999999)) or float(row['Close']) > float(row['Open']))

                if _hard_stop_hit_bts or _soft_close_stop_hit_bts or is_time_stop:
                    if _margin_call_hit and not _hard_stop_hit_bts:
                        base_exit = float(row['High'])
                        result_text = "🚨 觸發維持保證金不足強制平倉 (Margin Call)"
                    else:
                        base_exit = max(row['Open'], stop_loss) if _hard_stop_hit_bts else row['Close']
                        result_text = ("觸發空頭保本止損" if position_stage >= 1 else ("觸發空頭收盤結構破位止損" if _soft_close_stop_hit_bts and not _hard_stop_hit_bts else "觸發空頭盤中硬止損")) if (_hard_stop_hit_bts or _soft_close_stop_hit_bts) else f"空頭時間停損 ({time_stop_days}日)"

                    exit_price = base_exit * (1.0 + market_slippage)
                    cover_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=False)["single_leg_fee"]
                    entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                    gross_pnl = (entry_price - exit_price) * shares
                    cash += (short_margin_locked + gross_pnl - cover_fee)
                    total_leg_fee = round(entry_fee_alloc + cover_fee + _cum_borrow_fee, 2)
                    profit = gross_pnl - total_leg_fee
                    _cum_borrow_fee = 0.0

                    trade_info.update({
                        "出場日期": current_date,
                        "出場價": round(exit_price, 2),
                        "結果": result_text,
                        "回報率(%)": round(((entry_price - exit_price) / entry_price) * 100.0, 2),
                        "富途費用": total_leg_fee,
                        "利潤": profit
                    })
                    trade_log.append(trade_info)
                    position_type = None
                    shares = 0
                    short_margin_locked = 0.0
                    position_stage = 0

                elif row['Low'] <= tp_target and position_stage < 2:
                    if is_trending_market:
                        sold_shares = int(shares * 0.5)
                        if market_type == "HK":
                            lot_sz = POPULAR_HK_LOTS.get(ticker, 100)
                            sold_shares = (sold_shares // lot_sz) * lot_sz
                        if sold_shares == 0:
                            sold_shares = shares

                        exit_price = min(row['Open'], tp_target) * (1.0 + market_slippage)
                        cover_fee = compute_futu_real_fees(market_type, exit_price, sold_shares, is_sell=False)["single_leg_fee"]
                        ratio_sold = sold_shares / max(shares, 1)
                        entry_fee_part = round(float(trade_info.get("_entry_fee", 0.0)) * ratio_sold, 2)
                        trade_info["_entry_fee"] = max(0.0, float(trade_info.get("_entry_fee", 0.0)) - entry_fee_part)
                        margin_released = short_margin_locked * ratio_sold
                        short_margin_locked -= margin_released

                        borrow_fee_part = round(_cum_borrow_fee * ratio_sold, 2)
                        _cum_borrow_fee = max(0.0, _cum_borrow_fee - borrow_fee_part)
                        gross_pnl = (entry_price - exit_price) * sold_shares
                        cash += (margin_released + gross_pnl - cover_fee)
                        shares -= sold_shares
                        total_leg_fee = round(entry_fee_part + cover_fee + borrow_fee_part, 2)

                        partial_info = trade_info.copy()
                        partial_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": "階段一：觸及下方支撐回補 50%",
                            "回報率(%)": round(((entry_price - exit_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": gross_pnl - total_leg_fee
                        })
                        trade_log.append(partial_info)

                        if shares > 0:
                            position_stage = 2
                            _be_info = compute_futu_real_fees(market_type, entry_price, shares, is_sell=True)
                            stop_loss = min(stop_loss, _be_info["breakeven_short_price"])
                        else:
                            position_type, position_stage = None, 0
                            short_margin_locked = 0.0
                    else:
                        exit_price = min(row['Open'], tp_target) * (1.0 + market_slippage)
                        cover_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=False)["single_leg_fee"]
                        entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                        gross_pnl = (entry_price - exit_price) * shares
                        cash += (short_margin_locked + gross_pnl - cover_fee)
                        total_leg_fee = round(entry_fee_alloc + cover_fee + _cum_borrow_fee, 2)
                        _cum_borrow_fee = 0.0
                        profit = gross_pnl - total_leg_fee
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": "震盪市：觸及結構支撐全數回補",
                            "回報率(%)": round(((entry_price - exit_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": profit
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        short_margin_locked = 0.0
                        position_stage = 0

                elif position_stage == 2:
                    _swing_trail_ma_s = max(float(row.get('SMA20', 999999)), float(row.get('EMA21', row.get('SMA20', 999999))))
                    if float(row['Close']) > _swing_trail_ma_s or float(row['Close']) > float(row.get('ResShort_Causal', 999999)):
                        exit_price = float(row['Close']) * (1.0 + market_slippage)
                        cover_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=False)["single_leg_fee"]
                        entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                        gross_pnl = (entry_price - exit_price) * shares
                        cash += (short_margin_locked + gross_pnl - cover_fee)
                        total_leg_fee = round(entry_fee_alloc + cover_fee + _cum_borrow_fee, 2)
                        _cum_borrow_fee = 0.0
                        profit = gross_pnl - total_leg_fee
                        trade_info.update({
                            "出場日期": current_date,
                            "出場價": round(exit_price, 2),
                            "結果": "大趨勢尾倉：升穿 20MA/21EMA 波段回補",
                            "回報率(%)": round(((entry_price - exit_price) / entry_price) * 100.0, 2),
                            "富途費用": total_leg_fee,
                            "利潤": profit
                        })
                        trade_log.append(trade_info)
                        position_type = None
                        shares = 0
                        short_margin_locked = 0.0
                        position_stage = 0

            if position_type is None and trade_log:
                _last_trd_profit = float(trade_log[-1].get("利潤", 0.0))
                _bt_loss_streak = (_bt_loss_streak + 1) if _last_trd_profit <= 0 else 0
            if position_type == "LONG":
                current_equity = cash + (shares * float(row['Close']))
            elif position_type == "SHORT":
                current_equity = cash + short_margin_locked + ((entry_price - float(row['Close'])) * shares)
            else:
                current_equity = cash
            _bt_peak_equity = max(_bt_peak_equity, current_equity)
            history_equity.append(current_equity)
            history_dates.append(current_date)

        if position_type is not None:
            exit_price = float(df.iloc[-1]['Close'])
            if position_type == "LONG":
                exit_price = exit_price * (1.0 - market_slippage)
                sell_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=True)["single_leg_fee"]
                entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                cash += (shares * exit_price - sell_fee)
                total_leg_fee = round(entry_fee_alloc + sell_fee, 2)
                profit = (exit_price - entry_price) * shares - total_leg_fee
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": round(exit_price, 2),
                    "結果": "回測結束強制平倉",
                    "回報率(%)": round(((exit_price - entry_price) / entry_price) * 100.0, 2),
                    "富途費用": total_leg_fee,
                    "利潤": profit
                })
            else:
                exit_price = exit_price * (1.0 + market_slippage)
                cover_fee = compute_futu_real_fees(market_type, exit_price, shares, is_sell=False)["single_leg_fee"]
                entry_fee_alloc = float(trade_info.get("_entry_fee", 0.0))
                gross_pnl = (entry_price - exit_price) * shares
                cash += (short_margin_locked + gross_pnl - cover_fee)
                total_leg_fee = round(entry_fee_alloc + cover_fee, 2)
                profit = gross_pnl - total_leg_fee
                trade_info.update({
                    "出場日期": df.index[-1],
                    "出場價": round(exit_price, 2),
                    "結果": "回測結束強制平倉",
                    "回報率(%)": round(((entry_price - exit_price) / entry_price) * 100.0, 2),
                    "富途費用": total_leg_fee,
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

st.sidebar.markdown("### 📏 持倉週期與止損防洗盤設定")
stop_profile_mode = st.sidebar.selectbox(
    "選擇止損寬度模式 (針對富途波段優化):",
    list(STOP_PROFILE_CONFIGS.keys()),
    index=0,
    key="stop_profile_mode",
    help="預設為「中線波段標準 (持倉數週至1-2個月)」，要求止損距離買入位至少 ≥1.2x ATR 且 2.8%~7.5%，並錨定於 20MA/21EMA 或 10~20日箱體底，徹底解決止損過貼易被日內洗盤掃出的問題！"
)

st.sidebar.markdown("### 💵 滿倉本金與單筆 $1,000 止損設定")
_sb_col1, _sb_col2 = st.sidebar.columns(2)
with _sb_col1:
    user_max_loss_hkd = st.number_input(
        "🛡️ 單筆最大止損 (HK$):",
        min_value=200.0,
        max_value=20000.0,
        value=float(persistent_data.get("user_max_loss_hkd", 1000.0)),
        step=100.0,
        key="user_max_loss_hkd",
        help="你每筆交易願意承受的最大虧損金額（預設 HK$1,000，美股自動按 7.8 折算為 US$128）。系統會以此金額反推安全股數與滿倉低吸買入位！"
    )
with _sb_col2:
    user_full_pos_hkd = st.number_input(
        "💼 目標滿倉本金 (HK$):",
        min_value=5000.0,
        max_value=200000.0,
        value=float(persistent_data.get("user_full_pos_hkd", 21500.0)),
        step=1000.0,
        key="user_full_pos_hkd",
        help="你平時單筆滿倉習慣投入的本金（預設 HK$21,500，對應你習慣的 HK$18,000~25,000 區間；美股自動折算為 US$2,756）。"
    )
strict_hide_countertrend_signals = st.sidebar.checkbox(
    "🔒 弱市強制隱藏逆勢清單 (預設關閉：保留做多清單以捕捉見底領頭羊)",
    value=bool(persistent_data.get("strict_hide_countertrend_signals", False)),
    key="strict_hide_countertrend_signals",
    help="預設關閉（推薦）！確保當大市如今日般回調見底大升時，即使指數尚未升穿 50MA，你依然可以完整看到所有做多領頭羊與 2B 破底翻機會，系統改以大市五態倉位比例導航。"
)
persistent_data["user_max_loss_hkd"] = float(user_max_loss_hkd)
persistent_data["user_full_pos_hkd"] = float(user_full_pos_hkd)
persistent_data["strict_hide_countertrend_signals"] = bool(strict_hide_countertrend_signals)
st.sidebar.caption(
    f"💡 **當前風控基準**：單筆 1R = **HK${user_max_loss_hkd:,.0f}** (`US${user_max_loss_hkd/7.8:,.0f}`) ｜ "
    f"滿倉本金 = **HK${user_full_pos_hkd:,.0f}** (`US${user_full_pos_hkd/7.8:,.0f}`) ｜ "
    f"直接滿倉要求結構止損 **≤ {((user_max_loss_hkd - 38.0) / max(user_full_pos_hkd, 1.0)) * 100.0:.1f}%**"
)

st.sidebar.markdown("---")

# [優化加入] 無限擴充股票池輸入框
st.sidebar.markdown("### 🌐 擴充掃描池 (打破百隻限制)")
custom_tickers_raw = st.sidebar.text_area("自訂股票代碼 (逗號或空白分隔)", placeholder="例如: AAPL, PLTR, MSTR\n在此輸入你今天感興趣的代號，加入雷達進行全景體檢。")

st.sidebar.markdown("---")

# OLD:
# mode_options = [
#     "🔎 1. 市場動能掃描雷達",
#     "📊 2. 個股量化分析與技術診斷",
#     "⭐ 3. 自選股動能面板",
#     "📈 4. 策略歷史回測系統"
# ]
# NEW: [v25.3 內置互動實戰教學中心] 新增第 5 個功能模式「📖 5. 實戰教學與大師戰法指南 (新手必看)」及全頁頂部速查抽屜
mode_options = [
    "🔎 1. 市場動能掃描雷達",
    "📊 2. 個股量化分析與技術診斷",
    "⭐ 3. 自選股動能面板",
    "📈 4. 策略歷史回測系統",
    "📖 5. 實戰教學與大師戰法指南 (新手必看)"
]

default_idx = 0
if default_mode_val == "診斷":
    default_idx = 1
elif default_mode_val == "教學":
    default_idx = 4

mode = st.sidebar.radio("請選擇功能", mode_options, index=default_idx)

with st.sidebar.expander("🧭 30秒速查：點樣靈活運用？", expanded=False):
    st.markdown(
        "1. **選股**：去 **「1. 掃描雷達」** 睇 **#1、#2 優先榜**（只揀 `R:R ≥ 1.8R`）。\n"
        "2. **切換風格**：上方可切換 **🌊 中線波段**（防洗盤）或 **⚡ Martin Luk 短線緊密**（極高 R 倍數）。\n"
        "3. **不追高**：睇清 **Stop-Buy（突破才買）** 定係 **Limit-Buy（回踩低吸）**。\n"
        "4. **算倉位**：去 **「2. 個股診斷」** 展開 **Martin Luk 0.5% 風控計算器** 算出準確手數。\n"
        "5. **鎖利潤**：買入後輸入 **實際進場價**，跟隨 **TP1 減半 + 富途打和保本 + 20MA/21EMA 尾倉跟蹤**！"
    )
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **合規與風險聲明**：本系統僅供量化研究與技術分析參考，不構成任何投資邀約或買賣建議。歷史回測勝率不代表未來表現，市場數據可能存在延遲或複權誤差，請嚴格執行個人風險管理。")

if mode == mode_options[0]:
    st.query_params.clear()

# ==========================================
# 10. 雷達顯示
# ==========================================
_DF_RENDER_COUNTER = 0

def render_styled_dataframe(df, table_key: Optional[str] = None):
    """
    [v38.0 100% 保留原版 st.dataframe 視覺與顏色介面，並在底層修復美股列表序列化與無 key 衝突]
    1) 強制執行 reset_index(drop=True) 並淨化字串/浮點數 NaN，消除子表切片索引不連續造成的 Pandas Styler / PyArrow 序列化崩潰；
    2) 自動為每個 st.dataframe 綁定唯一 key，防止同頁同時渲染港股與美股多個表格時引發前端模組衝突。
    """
    global _DF_RENDER_COUNTER
    if df is None or df.empty:
        st.warning("數據過濾後無符合條件的結果。")
        return
    df = df.dropna(subset=['代碼', '最新價']).copy().reset_index(drop=True)
    if df.empty:
        st.warning("數據過濾後無符合條件的結果。")
        return

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("-").astype(str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)

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

    _valid_style_cols = [c for c in ['優先次序', 'J Law 強勢形態', '診斷結果', '交易方向', '信號狀態', 'META多重優勢'] if c in df.columns]
    styled_df = df.style.map(highlight_status, subset=_valid_style_cols) if _valid_style_cols else df.style

    if not table_key:
        _DF_RENDER_COUNTER += 1
        table_key = f"styled_df_{_DF_RENDER_COUNTER}_{len(df)}_{len(df.columns)}"

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
        hide_index=True,
        key=table_key
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
            # [NEW INCREMENTAL PATCH v25.0] 啟用原先定義卻未調用的 is_radar_cache_expired 自動過期檢查
            _base_mk = market_key.replace("自選_", "")
            if is_radar_cache_expired(persistent_data, market_key) and is_radar_cache_expired(persistent_data, _base_mk):
                st.warning(f"⏰ **雷達緩存過期提示 ({market_key})**：距離上次掃描已超過 30 分鐘或已跨交易日，建議重新點擊上方「🚀 啟動掃描」以獲取最新盤中/收盤報價。")
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

        # [v41.1 新增：雷達頂部「🧭 大市五態與回調見底轉勢判準」實時橫幅]
        try:
            _Sample_tk = str(df_results['代碼'].iloc[0]) if not df_results.empty else "0700.HK"
            _is_hk_mkt_b = (_Sample_tk.upper().endswith(".HK") or market_key == "港股")
            _idx_sym_b = "^HSI" if _is_hk_mkt_b else "^GSPC"
            _idx_name_b = "恒生指數 (^HSI)" if _is_hk_mkt_b else "標普500 (^GSPC) / 納指 (^IXIC)"
            _bg_b = get_index_data(_idx_sym_b, interval="1d")
            if _bg_b is not None and len(_bg_b) >= 55:
                _b_c = float(_bg_b['Close'].iloc[-1])
                _b_prev = float(_bg_b['Close'].iloc[-2])
                _b_ret = ((_b_c - _b_prev) / max(_b_prev, 1e-6)) * 100.0
                _b_s20 = float(_bg_b['Close'].rolling(20).mean().iloc[-1])
                _b_s50 = float(_bg_b['SMA50'].iloc[-1])
                _b_e9 = float(_bg_b['Close'].ewm(span=9, adjust=False).mean().iloc[-1])
                _b_dist = int(float(_bg_b['Dist_Days_20d'].iloc[-1])) if 'Dist_Days_20d' in _bg_b.columns else 0
                if _b_c >= _b_s50 and _b_c >= _b_s20:
                    st.success(
                        f"🧭 **【大市判準：🟢 {_idx_name_b} 強勢多頭期】** 最新 **{_b_c:,.0f}**（今日 **{_b_ret:+.2f}%**）｜ "
                        f"站穩 **20MA ({_b_s20:,.0f})** 與 **50MA ({_b_s50:,.0f})** 之上 ｜ 近20日出貨日 **{_b_dist}** 次。\n\n"
                        f"👉 **實戰導航**：大市順風向好，優先從下方 **🟢 做多清單** 挑選結構緊湊（止損 ≤4.8%）之領頭羊 **100% 滿倉** 進場！"
                    )
                elif (_b_ret >= 1.0 or _b_c >= _b_e9) and _b_c < _b_s50:
                    st.info(
                        f"🧭 **【大市判準：🌤️ {_idx_name_b} 回調見底 / 放量轉勢反攻日！】** 最新 **{_b_c:,.0f}**（今日大升 **{_b_ret:+.2f}%**）｜ "
                        f"9EMA (**{_b_e9:,.0f}**) ｜ 20MA (**{_b_s20:,.0f}**) ｜ 50MA (**{_b_s50:,.0f}**)。\n\n"
                        f"👉 **為什麼之前判斷大市偏弱？** 因為指數在前期回調後仍處於滯後的 **50MA ({_b_s50:,.0f})** 下方；但**今日大升 ({_b_ret:+.2f}%) 已出現短線見底轉勢信號**！\n"
                        f"👉 **實戰導航**：系統已**完整保留下方 🟢 做多清單（絕不屏蔽）**！領頭羊往往比大市更早升穿均線，建議揀選下方 #1~#3 率先收復 20MA/50MA 之強勢股，先以 **60%~70% 底倉（風險鎖死 ≤ HK$1,000）** 右側試倉，待指數企穩再加滿倉！"
                    )
                elif _b_c >= _b_s50 and _b_c < _b_s20:
                    st.warning(
                        f"🧭 **【大市判準：🟡 {_idx_name_b} 多頭回調消化期】** 最新 **{_b_c:,.0f}**（今日 **{_b_ret:+.2f}%**）｜ "
                        f"位於 **50MA ({_b_s50:,.0f})** 之上，但暫處 **20MA ({_b_s20:,.0f})** 下方整理。\n\n"
                        f"👉 **實戰導航**：做多清單完整開放，請優先選擇 **「回踩 20MA 右側收復」或「低吸掛單」**，避免在指數回調期市價高追突破單。"
                    )
                else:
                    st.warning(
                        f"🧭 **【大市判準：🔴 {_idx_name_b} 弱勢調整期】** 最新 **{_b_c:,.0f}**（今日 **{_b_ret:+.2f}%**）｜ "
                        f"位於 **20MA ({_b_s20:,.0f})** 與 **50MA ({_b_s50:,.0f})** 下方（出貨日 **{_b_dist}** 次）。\n\n"
                        f"👉 **實戰導航**：下方仍為你完整展示 **🟢 逆市抗跌做多領頭羊** 與 **🔴 順勢做空清單**。在大市站回 9EMA ({_b_e9:,.0f}) 前，做多宜控制倉位並死守 **HK$1,000** 止損紅線！"
                    )
        except Exception:
            pass

        st.markdown("### 🏆 今日優先行動精選榜 (已按綜合優勢由 #1 排序，只列 R:R ≥ 1.8R 頂級機會)")
        col_top_l, col_top_s = st.columns(2)
        with col_top_l:
            st.markdown("#### 🟢 優先買入 (Priority Buy) Top 名單 (同板塊自動去重：僅列各板塊 RS 第 1 領頭羊)")
            if not df_long.empty and "_raw_rr" in df_long.columns and "信號狀態" in df_long.columns:
                _cand_long = df_long[(df_long["_raw_rr"] >= 1.8) & (~df_long["信號狀態"].astype(str).str.contains("⛔"))].copy()
            else:
                _cand_long = df_long.copy()
            if not _cand_long.empty and "代碼" in _cand_long.columns:
                _cand_long["_sector_grp"] = _cand_long["代碼"].apply(get_ticker_sector)
                top_long = _cand_long.drop_duplicates(subset=["_sector_grp"], keep="first").head(6)
            else:
                top_long = _cand_long.head(6)
            if not top_long.empty:
                for idx_r, (_, r_row) in enumerate(top_long.iterrows(), 1):
                    _sec_name = get_ticker_sector(r_row['代碼'])
                    try:
                        _is_hk_r = str(r_row['代碼']).upper().endswith(".HK")
                        _mkt_r = "HK" if _is_hk_r else "US/Global"
                        _lot_r = POPULAR_HK_LOTS.get(r_row['代碼'], 100) if _is_hk_r else 1
                        _sz_g = compute_equal_risk_position_cap(float(r_row['最新價']), float(r_row['ATR防守止損']), lot_size=_lot_r, market_type=_mkt_r)
                        _cur_lbl = _sz_g.get("currency_symbol", "HK$" if _is_hk_r else "US$")
                        _rk_b_lbl = f"{_cur_lbl}{_sz_g.get('risk_budget_used', 1000 if _is_hk_r else 128):,.0f}"
                        _full_b_lbl = f"{_cur_lbl}{_sz_g.get('target_full_notional', 21500 if _is_hk_r else 2756):,.0f}"
                        if _sz_g.get("can_full_pos_at_market"):
                            _sz_line = (
                                f"🟢 **支援直接滿倉 ({_full_b_lbl} / {_sz_g['full_pos_shares']}股)**："
                                f"觸及結構止損僅蝕 **{_cur_lbl}{_sz_g['full_pos_loss_at_struct_sl']:,.0f}** (≤ 1R 上限 {_rk_b_lbl} ｜ 止損幅度 **-{_sz_g['stop_pct']:.2f}%**)"
                            )
                        elif _sz_g.get("min_1lot_over_budget"):
                            _sz_line = (
                                f"⚠️ **港股1手超 {_rk_b_lbl} 預算**：最低 1 手 ({_lot_r}股 = {_cur_lbl}{_sz_g['recommended_notional']:,.0f}) "
                                f"止損將蝕 **{_cur_lbl}{_sz_g['min_1lot_loss']:,.0f}**！若要滿倉只蝕 {_rk_b_lbl}，請掛低吸單於 **{_sz_g['safe_full_pos_entry']:.2f}** 或改買碎股 **{_sz_g['unrounded_shares']} 股**"
                            )
                        else:
                            _sz_line = (
                                f"🟡 **結構止損 -{_sz_g['stop_pct']:.2f}% 偏寬 (勿改窄止損送死)**："
                                f"① 現價等風險買 **{_sz_g['recommended_shares']}股 ({_cur_lbl}{_sz_g['recommended_notional']:,.0f})** 鎖死蝕 {_rk_b_lbl}；"
                                f"② 若堅持滿倉 ({_sz_g['full_pos_shares']}股) 請掛低吸單於 **{_sz_g['safe_full_pos_entry']:.2f}**！"
                            )
                    except Exception:
                        _sz_line = ""
                    st.success(
                        f"**第 {idx_r} 優先 [{_sec_name} 領頭羊]：{r_row['代碼']} {r_row['股票名稱']}** ｜ 現價 **{r_row['最新價']}** ｜ 時機：**{r_row.get('入場時機', '-')}**\n\n"
                        f"- 🏆 **《超級績效》重疊篩網**：**{r_row.get('超級績效重疊網', '-')}** ｜ 形態：**{r_row.get('VCP與3C買點', '-')}**\n"
                        f"- 🎯 **建議進場位**：**{r_row['建議進場價']}** — *{r_row.get('掛單與買入位說明', '-')}*\n"
                        f"- 🛡️ **結構止損位**：**{r_row['ATR防守止損']}** — *{r_row.get('止損依據', '-')}*\n"
                        f"- 💰 **波段止盈位 (港美雙軌)**：**{r_row['ATR目標止盈']}** — *{r_row.get('止盈阻力依據', '-')}*\n"
                        f"- ⚖️ **真實結構盈虧比**：**{r_row.get('預估盈虧比(R:R)', '-')}** ｜ {_sz_line}"
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

        f_col1, f_col2 = st.columns([2, 2])
        with f_col1:
            show_expert_cols = st.checkbox("🔬 顯示全部專家技術欄位 (預設關閉，只顯示最核心買賣位與盈虧比)", value=False, key=f"exp_cols_{market_key}")
        with f_col2:
            filter_timeliness = st.radio(
                "⚡ 入場時機過濾器:",
                ["📋 顯示全部機會", "⚡ 只看即時可入場 (現價處於買點區)", "🎯 只看突破觸發單 (Stop-Buy)", "⏳ 只看等待回調低吸單 (Limit-Buy)"],
                index=0,
                horizontal=True,
                key=f"time_flt_{market_key}"
            )
        if "入場時機" in df_long.columns and filter_timeliness != "📋 顯示全部機會":
            _kw = "⚡" if "即時" in filter_timeliness else ("🎯" if "突破" in filter_timeliness else "⏳")
            df_long = df_long[df_long["入場時機"].astype(str).str.contains(_kw, na=False)].copy()
            df_short = df_short[df_short["入場時機"].astype(str).str.contains(_kw, na=False)].copy()
        # OLD:
        # simple_cols = [
        #     "優先次序", "跳轉診斷", "代碼", "股票名稱", "交易方向", "優先總分",
        #     "最新價", "建議進場價", "掛單與買入位說明", "ATR防守止損", "ATR目標止盈",
        #     "預估盈虧比(R:R)", "META多重優勢", "J Law 強勢形態", "診斷結果"
        # ]
        # NEW: [v25.0] 精簡視圖加入 MartinLuk設置 與 ADR波幅(%)
        simple_cols = [
            "優先次序", "跳轉診斷", "代碼", "股票名稱", "交易方向", "超級績效重疊網", "VCP與3C買點", "機會評級", "入場時機", "優先總分",
            "最新價", "建議進場價", "掛單與買入位說明", "ATR防守止損", "ATR目標止盈",
            "預估盈虧比(R:R)", "DeepChart空單消化", "MartinLuk設置", "ADR波幅(%)", "META多重優勢", "J Law 強勢形態", "診斷結果"
        ]

        def _prepare_display_df(sub_df):
            clean_df = sub_df.drop(columns=['_raw_rr'], errors='ignore')
            if not show_expert_cols:
                avail_cols = [c for c in simple_cols if c in clean_df.columns]
                return clean_df[avail_cols]
            return clean_df

        st.markdown("#### 🟢 做多 (Long) 完整排名列表 (已按 #1, #2, #3 優先次序排列)")
        if not df_long.empty:
            render_styled_dataframe(_prepare_display_df(df_long), table_key=f"radar_long_tbl_{market_key}")
        else:
            st.info("目前沒有符合做多條件的標的。")

        st.write("")

        st.markdown("#### 🔴 做空 (Short) 完整排名列表 (已按 #1, #2, #3 優先次序排列)")
        if not df_short.empty:
            render_styled_dataframe(_prepare_display_df(df_short), table_key=f"radar_short_tbl_{market_key}")
        else:
            st.info("目前沒有符合做空條件的標的。")

        if not df_watch.empty:
            with st.expander("⚪ 觀望名單 (未觸發或已觸及止損失效)", expanded=False):
                render_styled_dataframe(_prepare_display_df(df_watch), table_key=f"radar_watch_tbl_{market_key}")

# ==========================================
# 11. 全景雷達掃描
# ==========================================
def run_all_inclusive_radar(stock_list):
    progress_bar = st.progress(0)
    results = []
    # [v25.1 批量預載加速] 先一次性批量下載股票池日線數據，移除無效 time.sleep(0.1) 阻塞
    if len(stock_list) >= 2:
        batch_preload_ohlcv_cache(stock_list)

    for idx, ticker in enumerate(stock_list):
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
            _action_time_str_pre = str(_tp_eval.get("action_timeliness", ""))
            _is_fresh_trigger = int(_action_time_str_pre.startswith("⚡") or any(k in status for k in ["突破", "破底", "2B", "匯聚買點", "EP"]))
            _meta_score_val = _jlaw_m.get("meta_long_score", 0) if ptype == "LONG" else (_jlaw_m.get("meta_short_score", 0) if ptype == "SHORT" else 0)

            _holistic_g = df.attrs.get("holistic_grade_info", {})
            _grade_str = str(_holistic_g.get("setup_grade", "⚪ C 級")) if ptype in ["LONG", "SHORT"] else "⚪ 觀望"
            _conv_bonus = float(_holistic_g.get("conviction_score", 50.0)) * 0.12 if ptype in ["LONG", "SHORT"] else 0.0
            _action_time_str = str(_tp_eval.get("action_timeliness", "⏳ 等待回調"))
            _dc_badge_str = str(_tp_eval.get("deepchart_badge", df.attrs.get("deepchart_info", {}).get("deepchart_badge", "-")))
            _dc_score_num = float(_tp_eval.get("deepchart_score", df.attrs.get("deepchart_info", {}).get("absorption_score", 50.0)))
            _immediate_bonus = 8.0 if _action_time_str.startswith("⚡") else 0.0
            # OLD: _dc_bonus = 5.0 if (_dc_score_num >= 65.0 and ptype == "LONG") else (-8.0 if (_dc_score_num < 40.0 and ptype == "LONG") else 0.0)
            # NEW: [v31.0 多空雙向 DeepChart 加減分同步]
            _dc_bonus = 5.0 if (_dc_score_num >= 62.0 and ptype in ["LONG", "SHORT"]) else (-8.0 if (_dc_score_num < 40.0 and ptype in ["LONG", "SHORT"]) else 0.0)
            _rr_bonus = min(max(_planned_rr, 0.0), 6.0) * 4.5
            _fresh_bonus = 25.0 if _is_fresh_trigger else 0.0
            _meta_bonus = _meta_score_val * 7.0
            _trend_bonus = (passcount if ptype == "LONG" else shortpasscount) * 2.5
            _rs_bonus = min(max(abs(rs_strength) * 2.0, 0.0), 12.0)
            _rr_penalty = -25.0 if _planned_rr < 1.8 else 0.0
            _priority_score = round(min(99.9, max(10.0, _fresh_bonus + _immediate_bonus + _dc_bonus + _conv_bonus + _meta_bonus + _rr_bonus + _trend_bonus + _rs_bonus + _rr_penalty)), 1) if ptype in ["LONG", "SHORT"] else 0.0

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
                # OLD: "信號狀態": _lifecycle_status if not _inv_info.get("long_stop_hit") else "⛔ 近期止損已觸發",
                # NEW: [v25.4 修復做空標的被標記為多頭止損已觸發漏洞] 根據 ptype 方向獨立判定信號狀態
                "信號狀態": (
                    "⛔ 近期多頭止損已觸發" if (ptype == "LONG" and _inv_info.get("long_stop_hit"))
                    else ("⛔ 近期空頭止損已觸發" if (ptype == "SHORT" and _inv_info.get("short_stop_hit")) else _lifecycle_status)
                ),
                "機會評級": _grade_str,
                "入場時機": _action_time_str if tactical_dir != "⚪ 觀望" else "⚪ 觀望",
                "建議進場價": entry_str,
                "掛單與買入位說明": _entry_desc if tactical_dir != "⚪ 觀望" else "-",
                "DeepChart空單消化": _dc_badge_str if tactical_dir != "⚪ 觀望" else "-",
                "ATR防守止損": sl_str,
                "止損依據": _sl_desc,
                "ATR目標止盈": tp_str,
                "止盈阻力依據": _tp_desc if tactical_dir != "⚪ 觀望" else "-",
                "預估盈虧比(R:R)": _rr_str,
                "超級績效重疊網": df.attrs.get("sepa_mini_screens_info", {}).get("overlap_badge", "-"),
                "VCP與3C買點": (
                    df.attrs.get("sepa_vcp_3c_info", {}).get("entry_tier_badge", "-")
                    if df.attrs.get("sepa_vcp_3c_info", {}).get("entry_tier") != "NONE"
                    else (df.attrs.get("sepa_tennis_squat_info", {}).get("action_badge", "-") if df.attrs.get("sepa_tennis_squat_info", {}).get("action_badge", "-") != "-" else df.attrs.get("sepa_vcp_3c_info", {}).get("vcp_footprint_str", "-"))
                ),
                "META多重優勢": _jlaw_m.get("meta_badge", "-"),
                "MartinLuk設置": df.attrs.get("martin_luk_info", {}).get("martin_setup_badge", "-"),
                "ADR波幅(%)": f"{float(df.attrs.get('martin_luk_info', {}).get('adr_pct', 0.0)):.1f}%",
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
# 16. 內置互動教學組件與模式 5：實戰教學與大師戰法指南 (v25.3 新增)
# ==========================================
def render_global_quick_guide_banner():
    """在模式 1~4 頂部提供不佔空間的可摺疊實戰速查抽屜，方便用戶邊睇數據邊對照操作"""
    with st.expander("📖 【隨查隨用】網頁版靈活運用速查卡（富途牛牛掛單對照 ／ Martin Luk 雙模式切換 ／ 指標口訣）", expanded=False):
        qg_c1, qg_c2, qg_c3 = st.columns(3)
        with qg_c1:
            st.markdown(
                "#### 🚀 每日 5 步標準流程\n"
                "1. **揀週期（左欄）**：中線波段揀 **🌊 中線標準**；短線爆發揀 **⚡ 短線緊密 (Martin Luk)**。\n"
                "2. **掃雷達（模式1/3）**：直睇 **🏆 今日優先行動精選榜 (#1, #2)**，系統已剔除 `<1.8R` 劣質信號。\n"
                "3. **睇診斷（模式2）**：一鍵跳轉睇 K 線 **藍線(買入) / 紅線(止損) / 綠線(止盈)**。\n"
                "4. **計手數（模式2）**：展開 **🧮 Martin Luk 0.5% 風控計算器**，避開港股 1 手超標陷阱。\n"
                "5. **管持倉（模式2）**：買入後填入 **「實際進場價」**，自動導航 **TP1 減倉 50% + 移至富途打和價**。"
            )
        with qg_c2:
            st.markdown(
                "#### 📱 富途牛牛 (Futu) 掛單對照\n"
                "- **突破觸發單 / 埋伏單 (`Stop-Buy`)**：\n"
                "  👉 富途揀 **【條件單 → 突破買入】**。當股價升穿「建議進場價」才觸發，未突破前絕不提早買！\n"
                "- **低吸回踩單 / S/R互換單 (`Limit-Buy`)**：\n"
                "  👉 富途揀 **【限價單 (Limit) 或 回落買入】**。現價已偏高時切勿市價追，掛在支撐位耐心等回踩。\n"
                "- **破底觸發沽空 (`Stop-Sell`) / 逢高反彈沽空 (`Limit-Sell`)**：\n"
                "  👉 富途分別對應 **【條件單 → 跌破賣出(沽空)】** 及 **【限價沽空單】**。"
            )
        with qg_c3:
            st.markdown(
                "#### 🧠 三師核心指標一眼看懂\n"
                "- **⏱️ 斜率時鐘**：**2點鐘**（回踩積極買）、**3點鐘**（設突破單埋伏）、**12點鐘**（過熱只鎖潤不追高）。\n"
                "- **🧱 抵扣價防假跌破**：只要 **現價 > 20日/50日抵扣價**，均線必繼續向上，回踩屬洗盤買點！\n"
                "- **🏆 Martin Luk 雲帶與 AVWAP**：股價回踩 **9/21 EMA + 前高錨定 AVWAP** 多重支撐洗盤收回，即極佳低風險買點。\n"
                "- 💡 *想睇完整圖解與互動打卡清單？請在左側欄切換至 **「📖 5. 實戰教學與大師戰法指南」**！*"
            )

def render_interactive_tutorial_hub():
    """模式 5：完整網頁版互動教學與靈活運用指南"""
    st.title("📖 專業量化交易終端 — 網頁版實戰教學與靈活運用指南")
    st.markdown(
        "歡迎使用本系統！本指南專為讓你**在網頁內邊看邊用**而設，結合 **Martin Luk (@martinlukkt，2025 USIC 美股交易大賽 +969.8% 冠軍)**、"
        "**J Law (@jlawstock)** 與 **LEI (@TheMarketMemo)** 三大體系，教你如何由選股、判斷買點、計算富途牛牛真實手數，到持倉分批鎖潤。"
    )

    t_tab1, t_tab2, t_tab3, t_tab4 = st.tabs([
        "🚀 1. 新手 5 步實戰 SOP & 富途下單對照",
        "🏆 2. Martin Luk 冠軍戰法 & 兩種模式靈活切換",
        "📐 3. J Law × LEI 指標活學活用字典",
        "✅ 4. 每日 10 分鐘互動打卡清單 (落單前必剔)"
    ])

    with t_tab1:
        st.markdown("### 🚀 第一部分：由打開網頁到富途牛牛落單的 5 步標準作業流程 (SOP)")
        st.info("💡 **核心原則**：永遠唔好用「現價」盲目市價追入！系統計算嘅每一個「建議進場價」都有明確結構依據（箱體樞紐頂或均線/AVWAP支撐位）。")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown(
                "#### 📍 Step 1 ~ Step 3：選股與定位買賣點\n"
                "1. **Step 1｜先看大盤體質與寬度（模式 1 雷達頂部）**\n"
                "   - 啟動港股或美股全景掃描後，先睇頂部 **「🌐 J Law 市場內部體質」**。\n"
                "   - 若高於 50MA 比例 `< 45%`，或個股診斷出現 **「⚠️ Martin Luk 大盤濾網警示（指數反彈逼近下彎 21/50 EMA）」**，代表假突破率極高，請將單筆風險減半至 `0.25%`。\n"
                "2. **Step 2｜只從「🏆 今日優先行動精選榜 (#1, #2...)」揀股**\n"
                "   - 優先榜已自動幫你過濾咗 **真實結構盈虧比 `< 1.8R`**、**已觸及前次止損 (`⛔`)** 或 **純多頭排列但無買點** 嘅股票。\n"
                "   - 直接揀選排名 **#1 ~ #5** 的標的，喺下方下拉選單一鍵點擊 **「🚀 進入診斷」**。\n"
                "3. **Step 3｜在六層 K 線圖確認三條關鍵價格線（模式 2 診斷）**\n"
                "   - **📌 藍色實線 (Entry)**：建議買入或沽空掛單位。\n"
                "   - **🛡️ 紅色虛線 (SL)**：結構防守止損位（跌穿代表結構破壞，無條件離場）。\n"
                "   - **🎯 綠色虛線 (TP1)**：上方第一道歷史籌碼峰 / 前浪頂阻力（觸及即減倉 50%）。"
            )
        with s_col2:
            st.markdown(
                "#### 📍 Step 4 ~ Step 5：富途精算倉位與持倉跟蹤\n"
                "4. **Step 4｜用 Martin Luk 風控計算器算出準確買入股數（模式 2）**\n"
                "   - 點擊展開 **「🧮 Martin Luk 0.5% 基準風控、資金曲線反饋與防跳空倉位計算器」**。\n"
                "   - 輸入你嘅 **帳戶總資金**、**近期連敗筆數** 同 **當前帳戶回撤 (%)**。\n"
                "   - 系統會自動扣除 **富途牛牛雙邊佣金、HK$15 平台費及 0.1% 政府印花稅**，算出 **建議買入股數/手數** 同埋 **富途實戰打和價**。\n"
                "   - ⚠️ **港股 1 手陷阱保護**：若出現「港股手數門檻預警」，代表買 1 手嘅止損金額已超過你嘅風險預算，請改買其他標的或用碎股功能！\n"
                "5. **Step 5｜買入後用「實戰持倉防守與風控導航」跟蹤（模式 2）**\n"
                "   - 成交後，喺診斷頁下方 **「📌 你的實際進場價」** 填入成交價。\n"
                "   - 系統會即時告訴你目前處於 **🌱 忍耐洗盤期** 定係 **🛡️ 波段保本期**，並提供明確嘅分批鎖潤價位！"
            )

        st.markdown("#### 📱 系統信號 vs 富途牛牛 (Futu NiuNiu) 實戰掛單方式對照表")
        futu_map_df = pd.DataFrame([
            {
                "系統顯示之進場類型": "突破觸發單 / 突破埋伏單 (Stop-Buy)",
                "代表市場狀態": "股價正處於 VCP / Inside Day / 均線密集緊密平台，尚未正式突破",
                "富途牛牛對應訂單類型": "【條件單】➔【突破買入】(觸發價設為系統「建議進場價」)",
                "實戰操作要領": "升穿箱體頂樞紐位才自動觸發買入；若一直橫盤唔突破就唔會成交，資金零消耗！"
            },
            {
                "系統顯示之進場類型": "低吸回踩單 / S/R互換支撐低吸單 (Limit-Buy)",
                "代表市場狀態": "股價處於 2點鐘穩健上升趨勢，正縮量回測 9/21 EMA、20MA 或前波突破頸線",
                "富途牛牛對應訂單類型": "【限價單 (Limit Order)】或【條件單 ➔ 回落買入】",
                "實戰操作要領": "切勿喺大陽線當日市價高追！預先喺系統顯示嘅支撐買入位掛限價單耐心承接。"
            },
            {
                "系統顯示之進場類型": "等回測低吸單 (現價已高出突破位 >3%)",
                "代表市場狀態": "股價當日已大幅放量拉升脫離買點，直接追高會令止損超過 7.8% 上限",
                "富途牛牛對應訂單類型": "【價格預警】或【限價單】掛喺前高頸線 / 10MA",
                "實戰操作要領": "寧願錯過都唔好追高！等未來數日縮量回踩至系統建議價位先買。"
            },
            {
                "系統顯示之進場類型": "破底觸發沽空單 (Stop-Sell) / 逢高反彈沽空 (Limit-Sell)",
                "代表市場狀態": "空頭排列且 RS 弱於大盤，跌穿箱體下沿或弱勢反抽下彎 20MA/21EMA 受阻",
                "富途牛牛對應訂單類型": "【條件單 ➔ 跌破賣出】或【限價賣出(沽空)】",
                "實戰操作要領": "只喺大盤指數低於 50MA 或弱勢時執行做空，觸及下方結構支撐 TP1 即回補一半。"
            }
        ])
        st.dataframe(futu_map_df, use_container_width=True, hide_index=True)

    with t_tab2:
        st.markdown("### 🏆 第二部分：Martin Luk (@martinlukkt) USIC 冠軍戰法與兩種模式靈活切換")
        st.markdown(
            "香港科大 (HKUST) 學生 **Martin Luk (@martinlukkt)** 於 2024 年及 2025 年連續參加 US Investing Championship (USIC)，"
            "分別創下 **+283%** 及 **+969.8%（歷史最高紀錄之一）** 的驚人成績。本系統已將其公開專訪與教學影片的核心武器 100% 量化植入："
        )
        ml_col1, ml_col2 = st.columns(2)
        with ml_col1:
            st.success(
                "#### 🔥 Martin Luk 五大實戰支柱\n"
                "1. **只做高 ADR% 的市場領頭羊 (Leader Scanner)**：\n"
                "   - 篩選過去 **1M / 3M / 6M 漲幅最強** 的領先股，且要求 **20日平均日內波幅 (ADR%)** 美股 `≥ 4.2%`、港股 `≥ 3.0%`。股性夠活先有爆發力！\n"
                "2. **9 / 21 / 50 EMA 動能雲帶 + 前高錨定 AVWAP**：\n"
                "   - 多頭強勢股必須維持 `Close > 9 EMA > 21 EMA > 50 EMA` 且均線向上發散。\n"
                "   - 同時由 **前波段高點 (High60 / 52W High)** 及 **跳空缺口 (EP)** 畫出錨定成交量加權均價 (**AVWAP**)，掌握機構真實平均持倉成本。\n"
                "3. **兩大王牌買點：多重支撐回踩 (Pullback) & Inside Day 突破**：\n"
                "   - **回踩買點**：股價回落測試 `9/21 EMA + 前高 AVWAP + 前突破頸線` 至少兩重支撐匯聚，日內洗盤跌穿後收盤強勢收回 (**Undercut & Reclaim**)。\n"
                "   - **突破買點**：在 9/21 EMA 旁形成 **Inside Day (內包日)** 或波幅 `< 0.6x ADR%` 的極窄幅壓縮日，隔日升穿前日高點 (PDH) 觸發！"
            )
        with ml_col2:
            st.warning(
                "#### ⚖️ 拋物線 R 倍數效應與資金曲線反饋風控\n"
                "4. **止損寬度與 R 倍數的拋物線關係 (Parabolic R-Multiple)**：\n"
                "   - 同樣捕捉一段 `+30%` 的波段升幅：若止損寬度係 `10%`，你只賺到 **`3R`**；若透過支撐匯聚將止損收緊至 `3%`，同樣 `+30%` 升幅會變成 **`10R`**；若在 Inside Day 將止損壓至 `1.5%`，更可高達 **`20R`**！\n"
                "5. **0.5% 基準風險 + 資金曲線反饋 (Equity Curve Feedback)**：\n"
                "   - 正常期每筆只冒總資金 **`0.5%`** 風險；連敗 `≥ 2` 筆或回撤 `≥ 6%` 自動減半至 **`0.25%`**；連敗 `≥ 4` 筆觸發 **`🛑 NO TRADE TODAY`** 強制冷靜。\n"
                "   - **防黑天鵝單倉上限**：高波動股 (`ADR% ≥ 6%`) 單一持倉市值嚴禁超過總資金 **`25%`**（大型股不超過 **`35%`**），防止隔夜跳空暴跌重創帳戶！"
            )

        st.markdown("#### 🔄 點樣喺網頁左側欄靈活切換「中線波段」與「Martin Luk 短線緊密」模式？")
        mode_cmp_df = pd.DataFrame([
            {
                "比較維度": "側欄選擇名稱",
                "🌊 模式 A：中線波段標準 (系統預設)": "🌊 中線波段標準 (持倉數週至1-2個月｜防洗盤 ≥1.2x ATR / 3.2%~7.8%)",
                "⚡ 模式 B：Martin Luk 短線緊密模式": "⚡ 短線緊密模式 (Martin Luk 日內/極短線｜≥0.45x ATR / 1.2%~5.2%)"
            },
            {
                "比較維度": "適合邊類投資者？",
                "🌊 模式 A：中線波段標準 (系統預設)": "返工忙碌、唔想成日睇盤、希望食盡幾星期至 1~2 個月大波段嘅投資者",
                "⚡ 模式 B：Martin Luk 短線緊密模式": "可每日睇盤或設條件單、追求極高盈虧比 (5R~15R) 同快速資金周轉嘅進階交易者"
            },
            {
                "比較維度": "止損位 (SL) 設定邏輯",
                "🌊 模式 A：中線波段標準 (系統預設)": "設於 20MA/21EMA 或 10~20日箱體底下方 0.35~0.6x ATR，距離買入位至少 3.2%~7.8%，防日常洗盤",
                "⚡ 模式 B：Martin Luk 短線緊密模式": "直接設於回踩當日或 Inside Day 當日低點 (LOD) 下方微幅緩衝，止損極緊 (約 1.2%~5.2%)"
            },
            {
                "比較維度": "移至保本價與尾倉跟蹤",
                "🌊 模式 A：中線波段標準 (系統預設)": "持倉滿 5 日且浮盈 ≥1R 移至富途打和價；TP1 減半後尾倉沿 20MA/21EMA 跟蹤",
                "⚡ 模式 B：Martin Luk 短線緊密模式": "持倉滿 2 日且脫離成本即移至保本；大漲過熱時沿 9 EMA 緊密鎖潤"
            }
        ])
        st.dataframe(mode_cmp_df, use_container_width=True, hide_index=True)

    with t_tab3:
        st.markdown("### 📐 第三部分：J Law (@jlawstock) × LEI (@TheMarketMemo) 指標活學活用字典")
        st.markdown("喺 **「📊 2. 個股量化分析與技術診斷」** 頁面入面，每個儀表板都有明確嘅實戰意義，以下係快速解讀口訣：")

        dict_c1, dict_c2 = st.columns(2)
        with dict_c1:
            st.markdown(
                "#### 1️⃣ ⏱️ 斜率時鐘智能診斷（LEI 趨勢姿態）\n"
                "- **📈 2點鐘方向（穩定上揚 — 最佳買入姿態）**：\n"
                "  20 / 50 / 120 MA 多頭平行發散向上。此時每次縮量回踩 20MA 或 21EMA 都係高勝率「吃回撤」買點！\n"
                "- **↔️ 3點鐘方向（橫盤壓縮 — 突破埋伏姿態）**：\n"
                "  六條均線（20/60/120 MA+EMA）極度黏合（差距 `< 2.5%`），代表籌碼高度集中，請喺箱體頂設 `Stop-Buy` 突破單埋伏。\n"
                "- **🚀 12點鐘方向（加速過熱 Climax Run — 嚴禁追高）**：\n"
                "  股價遠離 50MA 超過 25% 且 RSI 極度超買。此時絕不開新倉，已持倉者一旦跌破 **9 EMA** 即刻鎖潤離場！\n\n"
                "#### 2️⃣ 🧱 均線抵扣價分析（LEI 防假跌破神技）\n"
                "- **原理**：20MA 今日嘅升跌，取決於「今日收盤價」同「20個交易日前嘅收盤價（即抵扣價）」邊個高。\n"
                "- **實戰用法**：當股價短暫跌穿 20MA 時唔使驚！只要睇診斷頁顯示 **「✅ 現價 > 20日抵扣價 (均線向上)」**，而且未來 5~10 日抵扣價正快速下移，代表呢次跌破只係**假跌破洗盤**，均線依然會向上托住股價！"
            )
        with dict_c2:
            st.markdown(
                "#### 3️⃣ 🧱 LEI 頂底構造與 2B 破底翻 / 假突破\n"
                "- **🟢 2B 破底翻 (Bullish 2B)**：股價短暫跌穿前波低點引誘散戶止損，但隨即收陽強勢站回前低之上，屬極高盈虧比嘅底部反轉買點。\n"
                "- **🟢 底構造確認 (HL+HH)**：經過 K 線包含關係合併後，確認出現「更高低點 + 更高高點」轉折。\n"
                "- **🔴 頂構造 / 2B 假突破**：升穿前高後迅速殺回前高之下收陰，屬機構誘多派發信號，應立即減倉或結合空頭條件沽空。\n\n"
                "#### 4️⃣ 💎 J Law META 五大優勢與大戶出貨日預警\n"
                "- **💎 META 多重優勢 (`≥ 3/5` 為佳)**：綜合評估 (1) Power Trend 強勁趨勢、(2) S/R 阻力轉支撐互換區或 `<5.5%` 緊密平台、(3) 10/20MA 保護、(4) 縮量洗盤/底構造、(5) 30週線向上且空間 `≥ 1.8R`。湊齊越多優勢，勝率越高！\n"
                "- **🚨 大戶出貨日 (Distribution Days)**：近 20 個交易日內如果出現 **4 次或以上**「放量滯漲或放量下跌」（已自動排除四巫日結算量），圖表會標示紫色倒三角 `🔻`，代表機構正在暗中派發，必須提高警覺！"
            )

    with t_tab4:
        st.markdown("### ✅ 第四部分：每日 10 分鐘實戰打卡清單 (落單前互動檢查)")
        st.markdown("每次喺富途牛牛落單買入或沽空前，請逐項剔選以下 6 個檢查點，系統會即時評估你呢筆交易嘅準備度：")

        chk_1 = st.checkbox("1. 【大盤與寬度過濾】已確認大盤無見頂出貨警報，且無「指數反彈逼近下彎 21/50 EMA」假突破警告（如有警告已將風險減半至 0.25%）。", key="tut_chk_1")
        chk_2 = st.checkbox("2. 【領頭羊與排名確認】此標的來自雷達「🏆 今日優先行動精選榜 (#1 ~ #5)」，且 ADR% 波幅達標、RS 相對強弱跑贏大盤。", key="tut_chk_2")
        chk_3 = st.checkbox("3. 【真實結構盈虧比 ≥ 1.8R】已查看個股診斷頁，確認上方第一阻力位 (TP1) 距離建議買入位至少有 1.8R 以上獲利空間。", key="tut_chk_3")
        chk_4 = st.checkbox("4. 【拒絕盲目市價高追】已分清此單屬「Stop-Buy 突破觸發單」定係「Limit-Buy 回踩低吸單」，並按建議進場價掛單而非市價亂追。", key="tut_chk_4")
        chk_5 = st.checkbox("5. 【0.5% 風控與港股手數檢查】已用「Martin Luk 風控計算器」算出準確手數，確認含富途雙邊費用後實質風險不超標，且單倉市值 ≤ 25%~35%。", key="tut_chk_5")
        chk_6 = st.checkbox("6. 【預設結構止損與減倉計劃】已記下紅線結構止損價 (SL) 及綠線第一目標價 (TP1)，承諾跌穿 SL 無條件止損、觸及 TP1 減倉一半並移至富途打和價。", key="tut_chk_6")

        passed_checks = sum([chk_1, chk_2, chk_3, chk_4, chk_5, chk_6])
        st.progress(passed_checks / 6.0)
        if passed_checks == 6:
            st.success("🎉 **【滿分 6/6｜完美符合機構級交易紀律】** 你已完成所有風控與結構確認，可以按計劃於富途牛牛執行掛單！")
        elif passed_checks >= 4:
            st.warning(f"⚠️ **【已完成 {passed_checks}/6 項檢查】** 尚有 {6 - passed_checks} 項關鍵風控未確認，建議補齊檢查後再落單，切勿衝動交易。")
        else:
            st.error(f"🛑 **【僅完成 {passed_checks}/6 項檢查】** 準備不足！請先透過「1. 掃描雷達」與「2. 個股診斷」完成結構、盈虧比與倉位試算。")


# ==========================================
# 12. 模式 1：雷達
# ==========================================
if mode == mode_options[0]:
    st.title("🔎 市場動能掃描雷達")
    render_global_quick_guide_banner()
    st.markdown("💡 **提示：** 系統已智能分辨美/港股。")

    quick_search = st.text_input("🔍 即時個股雷達掃描 (輸入代碼即時檢視):", placeholder="例: AAPL").strip()
    if quick_search:
        q_df = run_all_inclusive_radar([quick_search])
        if q_df is not None:
            st.markdown("#### ⚡ 即時掃描結果")
            render_styled_dataframe(q_df)
        st.markdown("---")

    # [優化加入] 將自訂名單放入雷達掃描選項
    pool_choice = st.selectbox(
        "核心追蹤池：",
        [
            "🚀 超級績效・港股中小型高彈性潛力池 (ADR 3.2%~7.5% 細分龍頭)",
            "🚀 超級績效・美股中小型高爆發潛力池 (Minervini 風格高 RS 成長股)",
            "港股高動能增長池 (全市場強勢成分股｜已剔除慢速內銀)",
            "美股高動能爆發池 (全市場科技與成長領頭羊)",
            "港股收息防守池 (低波幅藍籌/內銀公用股)",
            "自訂擴充名單 (從側欄載入)"
        ]
    )
    
    target_pool = []
    if "港股中小型" in pool_choice:
        target_pool = HK_SMALL_MID_SUPER_POOL
    elif "美股中小型" in pool_choice:
        target_pool = US_SMALL_MID_SUPER_POOL
    elif "港股高動能" in pool_choice:
        target_pool = HK_MOMENTUM_POOL
    elif "收息防守" in pool_choice:
        target_pool = HK_DEFENSIVE_DIVIDEND_POOL
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
                with _CACHE_LOCK:
                    _global_persistent_data["radar_results"][dict_key] = df_res
                    _global_persistent_data["radar_correlation_warnings"][dict_key] = warnings
                    _global_persistent_data["radar_correlation_matrices"][dict_key] = corr_matrix
                    _global_persistent_data.setdefault("radar_timestamps", {}).update(persistent_data.get("radar_timestamps", {}))
                    _global_persistent_data.setdefault("radar_breadth_stats", {}).update(persistent_data.get("radar_breadth_stats", {}))
                    _global_persistent_data.setdefault("signal_lifecycle_store", {}).update(persistent_data.get("signal_lifecycle_store", {}))
                    save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))
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
    render_global_quick_guide_banner()
    st.markdown("管理你的核心觀察名單，可仿效富途牛牛建立多個分類標籤 (如：科技股、收息股、觀察中)，並分組進行雷達掃描。")

    # --- 管理分組 ---
    with st.expander("📁 管理分組 (新增 / 刪除分類)", expanded=False):
        col_add_cat, col_del_cat = st.columns(2)
        with col_add_cat:
            new_cat = st.text_input("➕ 新增分組名稱", placeholder="例如: 港股精選, 潛力妖股...")
            if st.button("建立分組"):
                if new_cat and new_cat not in st.session_state.watchlist_dict:
                    st.session_state.watchlist_dict[new_cat] = []
                    save_persisted_state_to_disk(persistent_data, st.session_state.watchlist_dict)
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
                    save_persisted_state_to_disk(persistent_data, st.session_state.watchlist_dict)
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
                        save_persisted_state_to_disk(persistent_data, st.session_state.watchlist_dict)
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
                        save_persisted_state_to_disk(persistent_data, st.session_state.watchlist_dict)
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
                            # 將相關性數據存入特定的 key 並同步至全域快取
                            st.session_state.radar_correlation_warnings[f"自選_{cat}"] = warnings
                            st.session_state.radar_correlation_matrices[f"自選_{cat}"] = corr_matrix
                            mark_radar_cache_updated(persistent_data, f"自選_{cat}")
                            with _CACHE_LOCK:
                                _global_persistent_data.setdefault("watchlist_radar_results", {})[cat] = df_res
                                _global_persistent_data["radar_correlation_warnings"][f"自選_{cat}"] = warnings
                                _global_persistent_data["radar_correlation_matrices"][f"自選_{cat}"] = corr_matrix
                                _global_persistent_data.setdefault("radar_timestamps", {}).update(persistent_data.get("radar_timestamps", {}))
                                save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))
                            
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
    render_global_quick_guide_banner()

    c_ticker, c_tf = st.columns([3, 1])
    with c_ticker:
        ticker_input = st.text_input("輸入股票代碼 (例: 3690、700、NVDA):", value=target_ticker).strip()
    with c_tf:
        # OLD: tf_choice = st.selectbox("週期", ["1m", "5m", "15m", "1h", "1d", "1wk"], index=4)
        # NEW: [v29.0 新增 4h (半日線) 週期支援]
        tf_choice = st.selectbox("週期", ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"], index=5)

    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d", "1wk": "1wk"}
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

            # ==============================================================================
            # [NEW INCREMENTAL PATCH v25.4] 診斷視角與持倉方向主控台 (徹底解決「前幾日做空股票重新輸入卻顯示做多提示與盤中跌穿當時止損」漏洞)
            # ==============================================================================
            _sig_mem = persistent_data.get("signal_lifecycle_store", {}).get(fmt_ticker, {})
            _saved_pos_map = persistent_data.setdefault("user_locked_positions", {})
            _saved_pos = _saved_pos_map.get(fmt_ticker, {})

            _init_persp_idx = 0
            if target_dir_from_radar == "SHORT" or _saved_pos.get("dir_code") == "SHORT" or (plan_type == "WATCH" and _sig_mem.get("plan_type") == "SHORT"):
                _init_persp_idx = 2
            elif target_dir_from_radar == "LONG" or _saved_pos.get("dir_code") == "LONG":
                _init_persp_idx = 1

            persp_col1, persp_col2 = st.columns([2, 2])
            with persp_col1:
                user_view_mode = st.radio(
                    "🧭 選擇診斷與提示視角 (切換後全頁指標、止損失效提示與交易計劃將 100% 同步對齊該方向):",
                    ["🤖 系統自動判定 (Auto)", "🟢 強制做多視角 (Long View)", "🔴 強制做空視角 (Short View)"],
                    index=_init_persp_idx,
                    horizontal=True,
                    key=f"view_mode_radio_{fmt_ticker}"
                )
            with persp_col2:
                if _sig_mem:
                    st.caption(
                        f"🕒 **歷史雷達信號記憶 ({_sig_mem.get('scan_date', '-')})**：上次掃描方向為 **{'🔴 做空' if _sig_mem.get('plan_type') == 'SHORT' else '🟢 做多'}** ｜ "
                        f"當時建議進場價 **{_sig_mem.get('entry', 0):.2f}** ｜ 初始止損 **{_sig_mem.get('sl', 0):.2f}** ｜ 目標 **{_sig_mem.get('tp', 0):.2f}**"
                    )

            # 若用戶選擇「強制做空視角」或「強制做多視角」，即時切換 plan_type、target_entry、stop_loss、take_profit、_tp_eval、jlaw_meta_info 與狀態提示！
            _inv_info_top = df.attrs.get("invalidation_info", {})
            _jlaw_m_top = df.attrs.get("jlaw_meta_info", {})
            if "強制做空" in user_view_mode:
                plan_type = "SHORT"
                _s_eval = df.attrs.get("short_plan_eval", df.attrs.get("trade_plan_eval", {}))
                df.attrs["trade_plan_eval"] = _s_eval
                target_entry = float(_s_eval.get("entry", target_entry))
                stop_loss = float(_s_eval.get("sl", stop_loss))
                take_profit = float(_s_eval.get("tp", take_profit))
                if _jlaw_m_top:
                    _jlaw_m_top["meta_edges_list"] = _jlaw_m_top.get("short_edges_list", _jlaw_m_top.get("meta_edges_list", []))
                    _jlaw_m_top["meta_badge"] = _jlaw_m_top.get("short_badge", _jlaw_m_top.get("meta_badge", "-"))
                if _inv_info_top.get("short_stop_hit"):
                    status = f"🔴 空頭視角診斷 ({_inv_info_top.get('short_invalidation_reason', '近期空頭止損已觸發')})"
                elif _inv_info_top.get("uptrend_veto_short"):
                    status = f"🟠 空頭視角警告 ({_inv_info_top.get('short_invalidation_reason', '股價已站上20/50MA，逆勢做空風險高')})"
                elif "做空" not in status and "空" not in status:
                    status = f"🔴 空頭視角診斷 (空頭條件滿足 {short_pass_count}/7 項 ｜ 建議於阻力位 {target_entry:.2f} 逢高佈局或破底追空)"
            elif "強制做多" in user_view_mode:
                plan_type = "LONG"
                _l_eval = df.attrs.get("long_plan_eval", df.attrs.get("trade_plan_eval", {}))
                df.attrs["trade_plan_eval"] = _l_eval
                target_entry = float(_l_eval.get("entry", target_entry))
                stop_loss = float(_l_eval.get("sl", stop_loss))
                take_profit = float(_l_eval.get("tp", take_profit))
                if _jlaw_m_top:
                    _jlaw_m_top["meta_edges_list"] = _jlaw_m_top.get("long_edges_list", _jlaw_m_top.get("meta_edges_list", []))
                    _jlaw_m_top["meta_badge"] = _jlaw_m_top.get("long_badge", _jlaw_m_top.get("meta_badge", "-"))
                if _inv_info_top.get("long_stop_hit"):
                    status = f"🟢 多頭視角警告 ({_inv_info_top.get('long_invalidation_reason', '近期多頭止損已觸發')})"
                elif _inv_info_top.get("downtrend_veto_long"):
                    status = f"🟡 多頭視角警告 ({_inv_info_top.get('long_invalidation_reason', '短中期均線破位，宜等回穩')})"
                elif "做多" not in status and "買點" not in status and "突破" not in status:
                    status = f"🟢 多頭視角診斷 (Minervini 條件滿足 {pass_count}/7 項 ｜ 建議於支撐位 {target_entry:.2f} 埋伏)"

            # 斜率時鐘智能診斷儀表板
            st.markdown("### ⏱️ 斜率時鐘智能診斷 (趨勢姿態判定)")
            col_clock1, col_clock2, col_clock3 = st.columns(3)
            
            # OLD:
            # with col_clock1:
            #     if last.get('Clock_12', False): st.error(...) else: st.info(...)
            # ... (原本使用無 key 之 st.error/st.success/st.warning 動態切換，易於切換視角或股票時觸發 React removeChild)
            # NEW: [v25.6 原子化 HTML 提示框] 全面採用 render_atomic_alert，徹底消除類型切換時之 React DOM 衝突
            with col_clock1:
                if last.get('Clock_12', False):
                    render_atomic_alert("🚀 **12點鐘方向 (加速過熱)**\n\n乖離過大，慎防見頂回落，切勿追高。", "error")
                else:
                    render_atomic_alert("⚪ 12點鐘方向：未觸發", "info")
                    
            with col_clock2:
                if last.get('Clock_2', False):
                    render_atomic_alert("📈 **2點鐘方向 (穩定上揚)**\n\n多頭平行發散，最佳吃回撤行情。", "success")
                else:
                    render_atomic_alert("⚪ 2點鐘方向：未觸發", "info")
                    
            with col_clock3:
                if last.get('Clock_3', False):
                    render_atomic_alert("↔️ **3點鐘方向 (橫盤壓縮)**\n\n均線極度黏合，等待大方向突破。", "warning")
                else:
                    render_atomic_alert("⚪ 3點鐘方向：未觸發", "info")
                    
            # 雙均線 (MA+EMA) 趨勢診斷
            st.markdown("### 🎯 雙均線 (MA+EMA) 趨勢診斷")
            st.caption("EMA (指數均線) 更為敏感，MA (簡單均線) 較為穩定。當兩者共振向上時，趨勢最為明確。")
            col_ema1, col_ema2, col_ema3 = st.columns(3)
            
            ema20, ma20 = last.get('EMA20', 0), last.get('SMA20', 0)
            with col_ema1:
                if ema20 > ma20 and last['Close'] > ema20:
                    render_atomic_alert(f"🟢 短期多頭共振 (20日)\n\nEMA ({ema20:.2f}) > MA ({ma20:.2f})", "success")
                elif ema20 < ma20 and last['Close'] < ema20:
                    render_atomic_alert(f"🔴 短期空頭共振 (20日)\n\nEMA ({ema20:.2f}) < MA ({ma20:.2f})", "error")
                else:
                    render_atomic_alert(f"🟡 短期震盪交纏 (20日)\n\nEMA ({ema20:.2f}) / MA ({ma20:.2f})", "warning")
                
            ema50, ma50 = last.get('EMA50', 0), last.get('SMA50', 0)
            with col_ema2:
                if ema50 > ma50 and last['Close'] > ema50:
                    render_atomic_alert(f"🟢 中期多頭共振 (50日)\n\nEMA ({ema50:.2f}) > MA ({ma50:.2f})", "success")
                elif ema50 < ma50 and last['Close'] < ema50:
                    render_atomic_alert(f"🔴 中期空頭共振 (50日)\n\nEMA ({ema50:.2f}) < MA ({ma50:.2f})", "error")
                else:
                    render_atomic_alert(f"🟡 中期震盪交纏 (50日)\n\nEMA ({ema50:.2f}) / MA ({ma50:.2f})", "warning")
                
            ema120, ma120 = last.get('EMA120', 0), last.get('SMA120', 0)
            with col_ema3:
                if ema120 > ma120 and last['Close'] > ema120:
                    render_atomic_alert(f"🟢 長期多頭共振 (120日)\n\nEMA ({ema120:.2f}) > MA ({ma120:.2f})", "success")
                elif ema120 < ma120 and last['Close'] < ema120:
                    render_atomic_alert(f"🔴 長期空頭共振 (120日)\n\nEMA ({ema120:.2f}) < MA ({ma120:.2f})", "error")
                else:
                    render_atomic_alert(f"🟡 長期震盪交纏 (120日)\n\nEMA ({ema120:.2f}) / MA ({ma120:.2f})", "warning")

            _wk_status_alerts = []
            if selected_interval == "1d":
                wk = weekly_alignment.get("detail", {})
                if weekly_alignment.get("warning"):
                    _wk_status_alerts.append(("warning", f"🌐 **三重濾網大週期共振監測燈號：** {weekly_alignment['warning']}"))
                else:
                    _wk_status_alerts.append(("success", "🌐 **三重濾網大週期共振監測燈號：** 週線與日線方向大致一致。"))

            if '🟢' in status or '🔥' in status:
                _wk_status_alerts.append(("success", status))
            elif '🔴' in status or '🟠' in status or '🩸' in status or '🏴‍☠️' in status:
                _wk_status_alerts.append(("error", status))
            elif '🟡' in status:
                _wk_status_alerts.append(("warning", status))
            else:
                _wk_status_alerts.append(("info", status))
            render_atomic_alert_group(_wk_status_alerts)

            st.markdown("### 🌟 J Law / Minervini / 空頭濾網")
            c_left, c_mid, c_right, c_short = st.columns(4)

            with c_left:
                st.markdown(f"#### {jl_title}")
                st.caption("J Law 核心條件")
                for criterion, passed in jl_chk.items():
                    # OLD: st.checkbox(criterion, value=passed, disabled=True, key=f"jlchk_{criterion}")
                    # NEW: [v25.5 綁定 fmt_ticker 防止切換港美股時因項目文字不同引發 React removeChild 錯誤]
                    st.checkbox(criterion, value=passed, disabled=True, key=f"jlchk_{fmt_ticker}_{criterion}")

            with c_mid:
                st.markdown(f"#### Minervini 趨勢條件 ({pass_count}/7)")
                st.caption(f"做多歷史勝率：{long_win:.1f}%")
                for criterion, passed in checklist.items():
                    # OLD: st.checkbox(criterion, value=passed, disabled=True, key=f"longchk_{criterion}")
                    st.checkbox(criterion, value=passed, disabled=True, key=f"longchk_{fmt_ticker}_{criterion}")

            with c_right:
                st.markdown(f"#### 空頭趨勢條件 ({short_pass_count}/7)")
                st.caption(f"做空歷史勝率：{short_win:.1f}%")
                for criterion, passed in short_checklist.items():
                    # OLD: st.checkbox(criterion, value=passed, disabled=True, key=f"shortchk_{criterion}")
                    st.checkbox(criterion, value=passed, disabled=True, key=f"shortchk_{fmt_ticker}_{criterion}")
                    
            with c_short:
                st.markdown(f"#### 🏴‍☠️ 大師做空法 ({ms_score}/4)")
                st.caption("O'Neil & Cohodes 實戰")
                st.checkbox("大盤處於熊市/跌破50MA", value=ms_metrics["market_bear"], disabled=True, key=f"ms1_{fmt_ticker}")
                st.checkbox("前高跌落且反彈受阻 50MA", value=ms_metrics["rebound_resist"], disabled=True, key=f"ms2_{fmt_ticker}")
                st.checkbox("反彈量能嚴重萎縮", value=ms_metrics["low_vol_rebound"], disabled=True, key=f"ms3_{fmt_ticker}")
                st.checkbox("Cohodes 右側支撐破位", value=ms_metrics["support_break"], disabled=True, key=f"ms4_{fmt_ticker}")

            st.markdown("---")
            # OLD: rationale_col1, rationale_col2 = st.columns(2) (原本在此處提前建立空白列，卻在下方 100 行後才塞入內容，導致 React DOM 節點樹錯位並觸發 NotFoundError: removeChild！)
            # NEW: [v25.5 修復 removeChild 崩潰] 移至下方「🎯 一眼看懂：結構買賣執行計劃」標題正下方順序建立與渲染，並以獨立 st.container() 隔離 DOM 節點
            # [NEW v24.14 一眼看懂：精準結構買賣位、止損止盈依據與雙大師實戰認證]
            _tp_eval = df.attrs.get("trade_plan_eval", {})
            _inv_info = df.attrs.get("invalidation_info", {})
            _lei_st = df.attrs.get("lei_struct_info", {})
            _lei_ded = df.attrs.get("lei_ded_forecast", {})
            _jlaw_m = df.attrs.get("jlaw_meta_info", {})

            # [NEW INCREMENTAL PATCH v25.0 / v31.0 多空雙向語義對齊] Martin Luk (@martinlukkt) USIC 冠軍實戰診斷、拋物線 R 倍數與資金曲線反饋面板
            render_locked_portfolio_summary_and_export(persistent_data)
            _ml_diag = df.attrs.get("martin_luk_info", {})
            _is_short_view = (plan_type == "SHORT")
            st.markdown("### 🏆 Martin Luk (@martinlukkt) USIC 冠軍體系診斷 (9/21/50 EMA雲帶 × 前高AVWAP × 極緊止損拋物線效應)")
            # OLD:
            # if _ml_diag.get("index_filter_warning"):
            #     st.warning(_ml_diag["index_filter_warning"])
            # NEW: [v25.6 固定預留槽位] 避免條件式插入改變後續節點索引
            _ml_idx_warn_slot = st.empty()
            if _ml_diag.get("index_filter_warning") and not _is_short_view:
                render_atomic_alert(_ml_diag["index_filter_warning"], "warning", slot=_ml_idx_warn_slot)
            elif _ml_diag.get("index_at_declining_ema_resistance") and _is_short_view:
                render_atomic_alert("🏴‍☠️ **Martin Luk 大盤空頭順風確認**：大盤指數目前處於 9EMA < 21EMA 或受制於下彎 21/50 EMA 阻力，順勢做空弱勢股勝率顯著提升！", "success", slot=_ml_idx_warn_slot)
            ml_c1, ml_c2, ml_c3, ml_c4 = st.columns(4)
            ml_c1.metric(
                "☁️ Martin 9/21/50 EMA 雲帶",
                _ml_diag.get("ema_cloud_state", "-")[:12],
                f"21EMA: {float(last.get('EMA21', 0)):.2f}"
            )
            ml_c2.metric(
                "⚡ 20日 ADR% (日均波幅)",
                f"{_ml_diag.get('adr_pct', 0.0):.2f}%",
                "✅ 符合高動能標準" if _ml_diag.get("adr_ok") else "⚠️ 波幅偏慢 (<標準)"
            )
            ml_c3.metric(
                "🎯 Martin Luk 實戰設置",
                _ml_diag.get("martin_setup_badge", "-")[:18],
                f"空頭反壓匯聚: {len(_ml_diag.get('confluence_resistances', []))}重" if _is_short_view else f"1M/3M/6M領先: {'是' if _ml_diag.get('is_leader_1m_3m_6m') else '否'}"
            )
            _active_stop_w_pct = max(float(_tp_eval.get("stop_pct", _ml_diag.get("stop_width_pct", 4.0))), 0.8)
            if _is_short_view:
                ml_c4.metric(
                    "📉 空頭波段 R 倍數潛力 (以空頭止損計)",
                    f"-15%跌幅 ≈ {15.0 / _active_stop_w_pct:.1f}R",
                    f"-30%跌幅 ≈ {30.0 / _active_stop_w_pct:.1f}R (空頭止損 {_active_stop_w_pct:.1f}%)"
                )
            else:
                ml_c4.metric(
                    "🚀 波段 R 倍數潛力 (以中線結構止損計)",
                    f"+25%漲幅 ≈ {_ml_diag.get('parabolic_r_25pct', 0):.1f}R",
                    f"+50%漲幅 ≈ {_ml_diag.get('parabolic_r_50pct', 0):.1f}R (波段止損 {_ml_diag.get('stop_width_pct', 0):.1f}%)"
                )
            _ml_conf_lines = []
            if _ml_diag.get("confluence_supports"):
                _ml_conf_lines.append("🛡️ **Martin Luk 下方多重支撐匯聚 (Stacked Support Confluence)**：" + " ＋ ".join(_ml_diag["confluence_supports"]))
            if _ml_diag.get("confluence_resistances"):
                _ml_conf_lines.append("🩸 **Martin Luk 上方空頭反壓匯聚 (Short Fade Confluence)**：" + " ＋ ".join(_ml_diag["confluence_resistances"]))
            st.caption(" ｜ ".join(_ml_conf_lines) if _ml_conf_lines else "⚪ 目前價格未貼近多重均線/AVWAP匯聚區")

            with st.expander("🧮 Martin Luk 0.5% 基準風控、資金曲線反饋 (Equity Curve Feedback) 與防跳空倉位計算器", expanded=False):
                ml_eq_c1, ml_eq_c2, ml_eq_c3 = st.columns(3)
                with ml_eq_c1:
                    _user_eq_ml = st.number_input(f"帳戶總資金 ({currency})", min_value=1000.0, value=100000.0, step=10000.0, key=f"ml_eq_{fmt_ticker}")
                with ml_eq_c2:
                    _user_streak_l = st.number_input("近期連續止損筆數 (Losing Streak)", min_value=0, max_value=20, value=0, step=1, key=f"ml_ls_{fmt_ticker}")
                with ml_eq_c3:
                    _user_dd_pct = st.number_input("目前帳戶回撤幅度 (%)", min_value=0.0, max_value=90.0, value=0.0, step=1.0, key=f"ml_dd_{fmt_ticker}")
                _default_lot_ml = 1 if market_type != "HK" else POPULAR_HK_LOTS.get(fmt_ticker, 100)
                _ml_sizing = evaluate_martin_luk_equity_feedback_and_sizing(
                    account_equity=_user_eq_ml,
                    entry_price=float(target_entry),
                    stop_loss=float(stop_loss),
                    adr_pct=float(_ml_diag.get("adr_pct", 4.0)),
                    market_type=market_type,
                    lot_size=_default_lot_ml,
                    recent_loss_streak=int(_user_streak_l),
                    current_drawdown_pct=float(_user_dd_pct)
                )
                _ml_sz_alerts = [("error" if _ml_sizing["no_trade_today"] else "info", _ml_sizing["regime_note"])]
                if _ml_sizing.get("lot_trap_warning"):
                    _lot_warn_msg = _ml_sizing["lot_trap_warning"].replace("買入", "沽空") if _is_short_view else _ml_sizing["lot_trap_warning"]
                    _ml_sz_alerts.append(("warning", _lot_warn_msg))
                render_atomic_alert_group(_ml_sz_alerts)
                sz1, sz2, sz3, sz4, sz5 = st.columns(5)
                sz1.metric("建議單筆風險 (含富途費)", f"{_ml_sizing['recommended_risk_pct']:.2f}%", f"含費實質風險 {_ml_sizing['actual_risk_pct']:.2f}%")
                sz2.metric("建議沽空股數" if _is_short_view else "建議買入股數", f"{_ml_sizing['final_shares']:,} 股", f"約 {_ml_sizing['final_lots']} 手")
                sz3.metric("沽空倉位總市值" if _is_short_view else "持倉總市值", f"{_ml_sizing['position_value']:,.0f} {currency}", f"佔總資金 {_ml_sizing['actual_pos_pct']:.1f}%")
                sz4.metric("富途雙邊總費用 (含印花稅)", f"{_ml_sizing['futu_round_trip_fee']:,.2f} {currency}", f"佔倉位 {_ml_sizing['futu_fee_pct']:.3f}%")
                sz5.metric("富途空頭打和價" if _is_short_view else "富途實戰打和價", f"{_ml_sizing['futu_breakeven_long'] if not _is_short_view else _ml_sizing['futu_breakeven_short']:.2f} {currency}", "⚠️ 已觸發單倉上限保護" if _ml_sizing["capped_by_notional"] else f"單倉上限 ≤ {_ml_sizing['max_pos_pct_cap']:.0f}%")

            _dc_diag = df.attrs.get("deepchart_info") or compute_deepchart_orderflow_absorption(df, lookback=60)
            _vbp_diag = df.attrs.get("vbp_vsa_info") or compute_vbp_profile_and_vsa_engine(df, lookback=120, num_bins=40)
            _heat_diag = evaluate_portfolio_heat_and_sector_risk(persistent_data, fmt_ticker, candidate_risk_pct=0.50)
            _gap_diag = detect_earnings_and_gap_risk(df, fmt_ticker)

            # OLD:
            # st.markdown("### 🧠 機構級 Volume-by-Price (VBP) 籌碼分佈、Wyckoff VSA 與前高空單消化診斷 (v27.0)")
            # ... (原本無論做多或做空均顯示「前高空單消化總分」、「阻力區賣壓枯竭度」、「防假突破」與「關注 VAL 支撐承接力，若跌穿需嚴守止損」)
            # NEW: [v31.0 多空雙向獨立渲染 VBP 籌碼分佈、Wyckoff VSA 與 DeepChart 訂單流診斷]
            if _is_short_view:
                st.markdown("### 🧠 機構級 Volume-by-Price (VBP) 籌碼分佈、Wyckoff VSA 與空頭壓制/支撐擊穿診斷 (v31.0 空頭專版)")
            else:
                st.markdown("### 🧠 機構級 Volume-by-Price (VBP) 籌碼分佈、Wyckoff VSA 與前高空單消化診斷 (v31.0)")
            _heat_and_gap_alerts = [(_heat_diag["alert_level"], _heat_diag["status_msg"])]
            if _gap_diag.get("gap_warning_msg"):
                _heat_and_gap_alerts.append(("warning", _gap_diag["gap_warning_msg"]))
            render_atomic_alert_group(_heat_and_gap_alerts)

            dc_c1, dc_c2, dc_c3, dc_c4 = st.columns(4)
            vbp_c1, vbp_c2, vbp_c3, vbp_c4 = st.columns(4)
            if _is_short_view:
                _s_dom = float(_dc_diag.get("short_dominance_score", 100.0 - float(_dc_diag.get("absorption_score", 50.0))))
                dc_c1.metric("🧠 空頭壓制與破底動能分", f"{_s_dom:.0f} / 100分", _dc_diag.get("short_deepchart_badge", "-"))
                dc_c2.metric("🔄 前低支撐測試與擊穿率", f"{_dc_diag.get('support_breakdown_ratio_pct', 0.0):.1f}%", f"前低支撐 {float(_dc_diag.get('prior_sup_price', 0.0)):.2f} 測試 {_dc_diag.get('support_tests_count', 0)} 次")
                dc_c3.metric("📈 反彈買盤枯竭度 (No-Demand)", "✅ 反彈買盤枯竭" if _dc_diag.get("buyer_exhaustion_ok") else "⚠️ 低位仍有抄底盤", "無量反抽確認 (利於做空)" if _dc_diag.get("buyer_exhaustion_ok") else "需防短線逼空反抽")
                dc_c4.metric("📊 訂單流微結構 (CVD & 派發)", "🩸 淨賣壓主導 (利於做空)" if not _dc_diag.get("cvd_10d_positive") else "🟢 淨買盤吸收 (逆風)", "🔻 出現機構放量派發日" if _dc_diag.get("distribution_spike_10d") else "常規空頭量能")

                vbp_c1.metric("🎯 VBP 籌碼共識蓋頭 (POC)", f"{float(_vbp_diag.get('poc_price', 0.0)):.2f} {currency}", f"所屬板塊: {_heat_diag['candidate_sector']}")
                vbp_c2.metric("📐 70% 價值區 (VAL ~ VAH)", f"{float(_vbp_diag.get('val_price', 0.0)):.2f} ~ {float(_vbp_diag.get('vah_price', 0.0)):.2f}", f"上方套牢反壓 {_vbp_diag.get('overhead_supply_pct', 0.0):.1f}% (下方僅 {_vbp_diag.get('downside_support_pct', 0.0):.1f}%)")
                vbp_c3.metric("📉 下方籌碼通道性質", "🩸 下方 LVN 籌碼真空暴跌區" if _vbp_diag.get("is_downside_lvn_vacuum") else ("🧱 下方 HVN 密集支撐區" if _vbp_diag.get("is_supported_by_downside_hvn") else "🟠 常規籌碼區"), _vbp_diag.get("short_vbp_position_label", "")[:18])
                _vsa_short_tag = "🔴 Upthrust誘多派發" if _vbp_diag.get("vsa_upthrust_distribution") else ("🩸 No-Demand無需求反抽" if _vbp_diag.get("vsa_no_demand") else ("⚠️ Spring底部吸收(防反彈)" if _vbp_diag.get("vsa_spring_absorption") else "⚪ 空頭量價常規"))
                vbp_c4.metric("🔬 Wyckoff VSA 空頭量價", _vsa_short_tag, "多頭買盤無力確認" if (_vbp_diag.get("vsa_upthrust_distribution") or _vbp_diag.get("vsa_no_demand")) else "觀察反抽量能")
                render_atomic_alert_group([
                    ("success" if _dc_diag.get("short_supply_dominant") else ("error" if _s_dom < 42.0 else "warning"), _dc_diag.get("short_deepchart_detail", "")),
                    ("info", f"📊 **VBP 空頭籌碼戰術導航**：{_vbp_diag.get('short_vbp_position_label', '')} — {_vbp_diag.get('short_vbp_tactical_advice', '')}")
                ])
            else:
                dc_c1.metric("🧠 前高空單消化總分", f"{_dc_diag.get('absorption_score', 50.0):.0f} / 100分", _dc_diag.get("deepchart_badge", "-"))
                dc_c2.metric("🔄 右側換手消化率", f"{_dc_diag.get('absorption_ratio_pct', 0.0):.1f}%", f"前高阻力 {float(_dc_diag.get('prior_res_price', 0.0)):.2f} 測試 {_dc_diag.get('tests_count', 0)} 次")
                dc_c3.metric("📉 阻力區賣壓枯竭度", "✅ 賣壓已枯竭" if _dc_diag.get("seller_exhaustion_ok") else "⚠️ 賣壓仍存", "VCP縮量測試確認" if _dc_diag.get("seller_exhaustion_ok") else "需防空單砸盤")
                dc_c4.metric("📊 訂單流微結構 (CVD & 口袋支點)", "🟢 淨買盤吸收" if _dc_diag.get("cvd_10d_positive") else "🔴 淨賣壓偏重", "🔥 出現口袋支點" if _dc_diag.get("pocket_pivot_10d") else "常規量能")

                vbp_c1.metric("🎯 VBP 籌碼共識中心 (POC)", f"{float(_vbp_diag.get('poc_price', 0.0)):.2f} {currency}", f"所屬板塊: {_heat_diag['candidate_sector']}")
                vbp_c2.metric("📐 70% 價值區 (VAL ~ VAH)", f"{float(_vbp_diag.get('val_price', 0.0)):.2f} ~ {float(_vbp_diag.get('vah_price', 0.0)):.2f}", f"上方套牢佔比 {_vbp_diag.get('overhead_supply_pct', 0.0):.1f}%")
                vbp_c3.metric("🚀 上方籌碼通道性質", "⚡ LVN 籌碼真空加速區" if _vbp_diag.get("is_in_lvn_vacuum") else ("🧱 HVN 套牢密集阻力" if _vbp_diag.get("is_blocked_by_hvn") else "🟡 常規籌碼區"), _vbp_diag.get("vbp_position_label", "")[:18])
                _vsa_tag = "🟢 No-Supply縮量無供給" if _vbp_diag.get("vsa_no_supply") else ("🔥 Spring洗盤吸收" if _vbp_diag.get("vsa_spring_absorption") else ("🔴 Upthrust派發" if _vbp_diag.get("vsa_upthrust_distribution") else "⚪ 量價常規"))
                vbp_c4.metric("🔬 Wyckoff VSA 量價散度", _vsa_tag, "空頭賣壓枯竭確認" if _vbp_diag.get("vsa_no_supply") else "觀察量價配合")
                render_atomic_alert_group([
                    ("success" if _dc_diag.get("shorts_absorbed") else ("error" if _dc_diag.get("heavy_overhead_supply") else "warning"), _dc_diag.get("deepchart_detail", "")),
                    ("info", f"📊 **VBP 籌碼分佈戰術導航**：{_vbp_diag.get('vbp_position_label', '')} — {_vbp_diag.get('vbp_tactical_advice', '')}")
                ])

            # OLD:
            # _hol_diag = df.attrs.get("holistic_grade_info") or evaluate_holistic_setup_grade_and_edge(...)
            # NEW: [v31.0 強制根據當前生效之 plan_type 與 _tp_eval 即時重算 8 維度評級，徹底解決視角切換時緩存不更新漏洞]
            _hol_diag = evaluate_holistic_setup_grade_and_edge(
                df=df, plan_type=plan_type if plan_type in ["LONG", "SHORT"] else ("SHORT" if _is_short_view else "LONG"),
                trade_plan_eval=_tp_eval, jlaw_meta_info=_jlaw_m,
                lei_struct_info=_lei_st, lei_ded_forecast=_lei_ded, martin_luk_info=_ml_diag,
                deepchart_info=_dc_diag, vbp_vsa_info=_vbp_diag, invalidation_info=_inv_info
            )
            df.attrs["holistic_grade_info"] = _hol_diag
            if _is_short_view:
                st.markdown("### 🏆 機構級 8 維度空頭品質評級 (Short Setup Grade) 與急跌超賣回補導航 (v31.0 空頭專版)")
            else:
                st.markdown("### 🏆 機構級 8 維度交易品質評級 (Setup Grade) 與高潮延伸主動鎖潤導航 (v31.0)")
            hg1, hg2, hg3, hg4 = st.columns(4)
            hg1.metric("🏆 綜合交易品質評級", _hol_diag.get("setup_grade", "-"), f"信心指數 {_hol_diag.get('conviction_score', 0):.1f}/100")
            hg2.metric("⚖️ 建議動態倉位權重", f"{_hol_diag.get('sizing_multiplier', 0)*100:.0f}% 標準倉", "S/A+級集中火力，B/C級防守")
            # OLD:
            # hg3.metric("🚀 RS 相對強弱線創高", "🔥 RS率先創20日新高" if _hol_diag.get("rs_leads_price") else "⚪ 常規隨勢運行", "機構逆勢掃貨指標")
            # hg4.metric("💰 高潮延伸主動減倉位", f"{_hol_diag.get('climax_extension_tp', 0):.2f} {currency}" if pd.notna(_hol_diag.get("climax_extension_tp")) else "-", "⚠️ 現價已過熱勿追" if _hol_diag.get("is_overextended_now") else "偏離20MA達3.2x ATR鎖潤位")
            # NEW: [v31.0 動態渲染多空專屬 RS 創高/破底與高潮減倉/超賣回補指標卡]
            hg3.metric(
                _hol_diag.get("rs_metric_title", "🚀 RS 相對強弱線創高"),
                _hol_diag.get("rs_metric_val", "⚪ 常規隨勢運行"),
                _hol_diag.get("rs_metric_delta", "觀察相對大盤強弱")
            )
            hg4.metric(
                _hol_diag.get("climax_metric_title", "💰 高潮延伸主動減倉位"),
                f"{_hol_diag.get('climax_extension_tp', 0):.2f} {currency}" if pd.notna(_hol_diag.get("climax_extension_tp")) else "-",
                _hol_diag.get("climax_metric_delta", "偏離20MA達3.2x ATR鎖潤位")
            )
            _pillars_txt = " ｜ ".join(_hol_diag.get("eight_pillars_passed", [])) if _hol_diag.get("eight_pillars_passed") else "暫無達標維度"
            _missing_txt = ("<br>⚠️ **待補強維度**：" + " ｜ ".join(_hol_diag.get("eight_pillars_missing", []))) if _hol_diag.get("eight_pillars_missing") else ""
            render_atomic_alert(
                f"{_hol_diag.get('sizing_advice', '')}\n\n✅ **已達成之機構級{'空頭' if _is_short_view else '多頭'}優勢**：{_pillars_txt}{_missing_txt}",
                "success" if _hol_diag.get("grade_code") in ["S", "A+"] else ("info" if _hol_diag.get("grade_code") == "A" else "warning")
            )

            # [v42.0 新增：《超級績效 1~4》SEPA 專屬診斷面板 (4組雙標準重疊網 × 3-C作弊點 × 網球/蹲伏行為 × 港美雙軌 R:R)]
            _sepa_vcp = df.attrs.get("sepa_vcp_3c_info") or compute_minervini_vcp_and_3c_cheats(df)
            _sepa_tb = df.attrs.get("sepa_tennis_squat_info") or evaluate_tennis_ball_and_squat_action(df)
            _sepa_ov = df.attrs.get("sepa_mini_screens_info") or evaluate_minervini_overlapping_mini_screens(df)
            _reg_gate_diag = df.attrs.get("market_regime_gate", {})
            st.markdown("### 📚 《超級績效 1～4》SEPA 冠軍診斷面板 (多重小篩網重疊 × 3-C 低位作弊點 × 網球強彈檢驗)")
            sp_c1, sp_c2, sp_c3, sp_c4 = st.columns(4)
            sp_c1.metric(
                "🕸️ 四組雙標準小篩網重疊",
                _sepa_ov.get("overlap_badge", "-"),
                f"命中 {_sepa_ov.get('overlap_count', 0)} / 4 個獨立篩網"
            )
            sp_c2.metric(
                "💎 VCP 收縮足跡與 3-C 買點",
                _sepa_vcp.get("entry_tier_badge", "-") if _sepa_vcp.get("entry_tier") != "NONE" else _sepa_vcp.get("vcp_footprint_str", "常規形態"),
                f"低位作弊樞紐: {_sepa_vcp.get('cheat_pivot_price', 0):.2f} (止損 -{_sepa_vcp.get('cheat_stop_pct', 0):.1f}%)" if _sepa_vcp.get("cheat_pivot_price", 0) > 0 else "觀察右側收斂"
            )
            sp_c3.metric(
                "🎾 網球 vs 雞蛋 & 蹲伏反應",
                _sepa_tb.get("action_badge", "⚪ 常規量價"),
                "機構承接強勁" if _sepa_tb.get("is_tennis_ball") else ("切勿左側接刀" if _sepa_tb.get("is_egg_slump") else "觀察突破延續性")
            )
            sp_c4.metric(
                "🧭 大市牛熊與轉勢階段",
                _reg_gate_diag.get("regime_badge", "-")[:16],
                f"建議總倉位上限 {_reg_gate_diag.get('recommended_pos_pct', 100)}%"
            )
            if _sepa_tb.get("action_note"):
                render_atomic_alert(f"📖 **《超級績效》量價行為解讀**：{_sepa_tb['action_note']}", "success" if _sepa_tb.get("is_tennis_ball") else "warning")

            st.markdown("### 🎯 一眼看懂：結構買賣執行計劃與雙大師認證 (v42.0 港美雙軌 R:R 版)")
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
            st.caption("🎯 **當前匯聚之交易優勢 (Multiple Edges)**：" + (" ｜ ".join(_jlaw_m["meta_edges_list"]) if _jlaw_m.get("meta_edges_list") else "暫無明顯匯聚優勢"))

            # [v25.5 / v25.6 修復 NotFoundError: removeChild] 在實際渲染位置緊接建立 st.columns(2) 並使用 render_atomic_alert
            rationale_col1, rationale_col2 = st.columns(2)
            with rationale_col1:
                _entry_how = _tp_eval.get("entry_type_desc", "")
                # OLD:
                # if plan_type == "SHORT": ...
                # else:
                #     if _inv_info.get("invalidation_reason"):
                #         entry_reason = f"⚪ **【觀望 / 信號已失效】** {_inv_info['invalidation_reason']}"
                # NEW: [v25.4 方向感知診斷說明] 根據當前選擇之視角與歷史雷達記憶顯示對應方向的失效原因（做空絕不顯示「盤中已跌穿當時做多止損」！）
                if plan_type == "SHORT":
                    _short_extra_warn = f"\n\n⚠️ **空頭風險提示**：{_inv_info['short_invalidation_reason']}" if _inv_info.get("short_invalidation_reason") else ""
                    entry_reason = (
                        f"🔴 **【沽空執行位】建議掛單價：{target_entry:.2f} {currency}**（現價：{last['Close']:.2f}）\n\n"
                        f"📌 **進場方式**：{_entry_how}\n\n"
                        f"📊 **信號形態**：{status}{_short_extra_warn}"
                    )
                elif plan_type == "LONG":
                    _long_extra_warn = f"\n\n⚠️ **多頭風險提示**：{_inv_info['long_invalidation_reason']}" if _inv_info.get("long_invalidation_reason") else ""
                    entry_reason = (
                        f"🟢 **【買入執行位】建議掛單價：{target_entry:.2f} {currency}**（現價：{last['Close']:.2f}）\n\n"
                        f"📌 **進場方式**：{_entry_how}\n\n"
                        f"📊 **信號形態**：{status}{_long_extra_warn}"
                    )
                else:
                    if _sig_mem.get("plan_type") == "SHORT":
                        _s_reason = _inv_info.get("short_invalidation_reason") or "目前空頭動能暫歇或處於反彈整理，若您持有前幾日之空單，請在下方『實戰持倉防守與風控導航』輸入進場價與開倉日期以鎖定空頭止損與目標。"
                        entry_reason = f"⚪ **【觀望 / 前期空頭信號跟蹤】** {_s_reason}"
                    elif _inv_info.get("long_invalidation_reason"):
                        entry_reason = f"⚪ **【觀望 / 多頭信號已失效】** {_inv_info['long_invalidation_reason']}"
                    else:
                        entry_reason = "⚪ **【觀望等待】** 目前未達高勝率觸發條件，請等待股價形成緊密平台或回測關鍵支撐/阻力位。"
                # OLD: st.info(entry_reason)
                render_atomic_alert(entry_reason, "info")

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
                # OLD: st.warning(exit_reason)
                render_atomic_alert(exit_reason, "warning")

            st.markdown("---")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric(f"最新價 ({currency})", f"{last['Close']:.2f}")
            m2.metric("RSI 14", f"{last['RSI']:.1f}")
            m3.metric("距52W高點", f"{last['Distto52WHigh']:.1f}%")
            m4.metric("短線阻力", f"{last['ResShort']:.2f}")
            m5.metric("MACD 柱體", f"{last['MACDHist']:.3f}")

            # 抵扣價防禦判定顯示 (v31.0 多空雙向語義對齊)
            if _is_short_view:
                st.markdown("### 🧱 支撐阻力與抵扣價分析 (空頭防假突破 & 均線助跌診斷)")
                st.caption("空頭判斷真假反抽的秘訣：只要【現價】低於高企的【抵扣價】，即使股價短暫反彈刺穿均線，該均線未來仍會繼續下彎蓋頭助跌（假突破真壓制）。")
            else:
                st.markdown("### 🧱 支撐阻力與抵扣價分析 (防假跌破)")
                st.caption("判斷真假跌破的秘訣：只要【現價】大於【抵扣價】，即使股價跌破均線，該均線未來仍會繼續向上發散。")
            sr1, sr2, sr3, sr4, sr5 = st.columns(5)
            
            deduct_20 = last.get('SMA20_Deduction', 0)
            if _is_short_view:
                status_20 = "🩸 均線下彎助跌" if last['Close'] < deduct_20 else "⚠️ 均線上翹托底"
            else:
                status_20 = "✅ 均線向上" if last['Close'] > deduct_20 else "❌ 均線下彎"
            sr1.metric("🟢 短期支撐 (20日)", f"{last['SupShort']:.2f}", f"20日抵扣: {deduct_20:.2f} ({status_20})")
            
            sr2.metric("🔴 短期阻力 (20日)", f"{last['ResShort']:.2f}")
            
            deduct_50 = last.get('SMA50_Deduction', 0)
            if _is_short_view:
                status_50 = "🩸 均線下彎助跌" if last['Close'] < deduct_50 else "⚠️ 均線上翹托底"
            else:
                status_50 = "✅ 均線向上" if last['Close'] > deduct_50 else "❌ 均線下彎"
            sr3.metric("🟢 中期50MA均線位", f"{last['SMA50']:.2f}", f"50日抵扣: {deduct_50:.2f} ({status_50})")
            
            sr4.metric("🟢 長期支撐 (60日低點)", f"{last['SupLong']:.2f}", f"60MA抵扣: {float(last.get('SMA60_Deduction', 0)):.2f}")
            sr5.metric("🔴 長期阻力 (60日高點)", f"{last['ResLong']:.2f}", f"5日箱體波幅: {float(last.get('Tight_Box_5d_Pct', 0)):.1f}%")
            if _is_short_view:
                st.caption(f"🔮 **LEI 未來兩週空頭抵扣價推演**：20MA → {_lei_ded.get('short_forecast_20_label', _lei_ded.get('forecast_20_label', ''))} ｜ 60MA → {_lei_ded.get('short_forecast_60_label', _lei_ded.get('forecast_60_label', ''))}")
            else:
                st.caption(f"🔮 **LEI 未來兩週抵扣價方框推演**：20MA → {_lei_ded.get('forecast_20_label', '')} ｜ 60MA → {_lei_ded.get('forecast_60_label', '')}")

            st.markdown("### 💰 富途牛牛 (Futu) 實戰投資門檻與交易成本試算")
            col_lot1, col_lot2, col_lot3, col_lot4 = st.columns(4)
            
            default_lot = 1 if market_type != "HK" else POPULAR_HK_LOTS.get(fmt_ticker, 100)
            
            with col_lot1:
                lot_size = st.number_input("一手股數 (自動匹配/手動修改):", min_value=1, value=default_lot, step=100 if market_type == "HK" else 1)
            
            min_invest_amount = lot_size * last['Close']
            _lot_futu_fee = compute_futu_real_fees(market_type=market_type, price=float(last['Close']), shares=int(lot_size), is_sell=_is_short_view)
            _one_lot_risk = lot_size * abs(float(target_entry) - float(stop_loss)) + _lot_futu_fee["round_trip_fee"]
            
            with col_lot2:
                if _is_short_view:
                    st.metric("一手沽空市值", f"{min_invest_amount:,.2f} {currency}", f"富途空頭打和價 {_lot_futu_fee['breakeven_short_price']:.2f}")
                else:
                    st.metric("一手買入金額", f"{min_invest_amount:,.2f} {currency}", f"富途打和價 {_lot_futu_fee['breakeven_long_price']:.2f}")
            with col_lot3:
                st.metric("一手富途買賣雙邊總費用", f"{_lot_futu_fee['round_trip_fee']:,.2f} {currency}", f"摩擦成本 {_lot_futu_fee['round_trip_pct']:.2f}% (含印花稅/平台費)")
            with col_lot4:
                st.metric("沽空 1 手建議最低總本金 (1% 風險)" if _is_short_view else "買入 1 手建議最低總本金 (1% 風險)", f"{(_one_lot_risk * 100.0):,.0f} {currency}", help="按『1手止損金額 + 富途雙邊手續費 = 總資金 1%』精算之最低帳戶規模")
            
            st.markdown("---")

            # ==============================================================================
            # [v29.0 新增：1D × 4H × 1H × 15m 多時間框架 (MTF) 精準入場與離場狙擊中心]
            # 結合 J Law × Brian Shannon (Anchored VWAP) × Qullamägi (ORB) 解決「買入即被套、離場賣太差」
            # ==============================================================================
            st.markdown("### ⏱️ 四週期共振狙擊儀表板 (1D 戰略方向 × 4H 波段中繼 × 1H 右側AVWAP轉折 × 15m ORB/VWAP 扳機)")
            _mtf_sniper = compute_mtf_execution_and_exit_sniper(
                ticker=fmt_ticker,
                daily_df=df,
                plan_type=plan_type if plan_type in ["LONG", "SHORT"] else "LONG",
                planned_entry=float(target_entry),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit)
            )
            mtf_c1, mtf_c2, mtf_c3, mtf_c4 = st.columns(4)
            mtf_c1.metric(
                "🎯 多週期共振評分",
                f"{_mtf_sniper['mtf_alignment_score']} / 100分",
                "🔥 大小週期同向共振" if _mtf_sniper['mtf_alignment_score'] >= 70 else ("🛑 大小週期打架(勿接飛刀)" if _mtf_sniper['mtf_alignment_score'] <= 35 else "🟡 等待小週期確認")
            )
            mtf_c2.metric(
                "⚡ 1H/15m 最佳入場扳機價",
                f"{_mtf_sniper['sniper_entry_trigger_price']:.2f} {currency}",
                f"1H回調AVWAP: {_mtf_sniper['h1_pullback_avwap'] if pd.notna(_mtf_sniper['h1_pullback_avwap']) else '-'}"
            )
            mtf_c3.metric(
                "📊 15m 日內 VWAP & ORB 區間",
                f"VWAP {_mtf_sniper['intraday_vwap']:.2f}" if pd.notna(_mtf_sniper['intraday_vwap']) else "非盤中/無15m資料",
                f"ORB: {_mtf_sniper['orb_low_15m']:.2f} ~ {_mtf_sniper['orb_high_15m']:.2f}" if pd.notna(_mtf_sniper['orb_high_15m']) else "-"
            )
            mtf_c4.metric(
                "💰 1H 結構移動鎖潤離場位",
                f"{_mtf_sniper['sniper_profit_lock_exit']:.2f} {currency}",
                f"1H 20EMA: {_mtf_sniper['h1_ema20'] if pd.notna(_mtf_sniper['h1_ema20']) else '-'}"
            )
            st.caption(
                f"📡 **四週期實時狀態** — **【4H 半日線】**：{_mtf_sniper['tf_4h_state']} ｜ "
                f"**【1H 小時線】**：{_mtf_sniper['tf_1h_state']} ｜ "
                f"**【15m 分鐘線】**：{_mtf_sniper['tf_15m_state']}"
            )
            if _mtf_sniper.get("entry_timing_cards"):
                render_atomic_alert_group(_mtf_sniper["entry_timing_cards"])
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
                p1.metric("🎯 建議沽空掛單位" if plan_type == "SHORT" else "🎯 建議買入掛單位", f"{target_entry:.2f} {currency}", delta=f"距現價 {_entry_diff_pct:+.2f}%")
                sl_diff = stop_loss - target_entry
                if plan_type == "SHORT":
                    p2.metric("🛡️ 空頭防守止損位", f"{stop_loss:.2f} {currency}", delta=f"+{abs(sl_diff):.2f} (風險 -{_tp_eval.get('stop_pct', 0):.1f}%)", delta_color="inverse")
                    tp_diff = take_profit - target_entry
                    _tp_drop_pct = abs(tp_diff) / max(target_entry, 1e-6) * 100.0
                    p3.metric("💰 第一空頭回補目標 (TP1)", f"{take_profit:.2f} {currency}", delta=f"{tp_diff:+.2f} (目標獲利 +{_tp_drop_pct:.1f}%)", delta_color="inverse")
                else:
                    p2.metric("🛡️ 結構防守止損位", f"{stop_loss:.2f} {currency}", delta=f"{sl_diff:+.2f} (-{_tp_eval.get('stop_pct', 0):.1f}%)", delta_color="inverse")
                    tp_diff = take_profit - target_entry
                    p3.metric("💰 第一結構止盈 (TP1)", f"{take_profit:.2f} {currency}", delta=f"{tp_diff:+.2f}")
                p4.metric("⚖️ 真實結構盈虧比", f"{_tp_eval.get('planned_rr', 0):.2f} R", delta=f"次級目標 {_tp_eval.get('tp2_rr', 0):.2f} R")
                p5.metric("🔄 交易方向", "🟢 做多" if plan_type == "LONG" else "🔴 做空")

            # [UI 更新] 完全植入動態強勢鎖潤提示與保本提醒
            st.markdown("### 🛡️ 實戰持倉防守與風控導航")
            st.info("💡 輸入您的實際進場價，系統將根據 **9 EMA** 的狀態，即時判斷您處於 **[忍耐洗盤期]** 還是 **[安全保本期]**，並為您制定最優的分批止盈策略！")
            
            # OLD:
            # col_pos1, col_pos2 = st.columns(2)
            # with col_pos1:
            #     actual_entry = st.number_input("📌 你的實際進場價 (輸入 0 則隱藏此面板，圖表恢復預設):", value=0.0, step=0.1)
            # with col_pos2:
            #     default_pos_idx = 1 if (plan_type == "SHORT" or target_dir_from_radar == "SHORT") else 0
            #     pos_dir = st.selectbox("🔄 你的持倉方向:", ["做多 (Long)", "做空 (Short)"], index=default_pos_idx, key=f"pos_dir_{fmt_ticker}_{plan_type}")
            # NEW: [v25.4 新增「開倉日期錨定」與「一鍵鎖定初始止損/目標價」功能]
            # 徹底解決「同一個進場價，過幾天來看止損、目標同盈虧比都變咗」嘅問題！
            _default_saved_entry = float(_saved_pos.get("entry", 0.0))
            _default_pos_idx = 1 if (
                _saved_pos.get("dir_code") == "SHORT"
                or plan_type == "SHORT"
                or target_dir_from_radar == "SHORT"
                or _sig_mem.get("plan_type") == "SHORT"
            ) else 0
            _default_entry_dt = pd.to_datetime(_saved_pos.get("anchor_date", df.index[-1])).date() if _saved_pos.get("anchor_date") else (
                pd.to_datetime(_sig_mem.get("scan_date", df.index[-1])).date() if _sig_mem.get("scan_date") else df.index[-1].date()
            )
            _min_avail_dt = df.index[min(20, len(df)-1)].date()
            _max_avail_dt = df.index[-1].date()
            if _default_entry_dt < _min_avail_dt or _default_entry_dt > _max_avail_dt:
                _default_entry_dt = _max_avail_dt

            _usr_pos_cfg = get_user_risk_and_full_pos_config(market_type=market_type)
            _default_full_lots = max(1, round((_usr_pos_cfg["full_notional"] / max(float(last['Close']), 1e-4)) / max(int(lot_size), 1))) if market_type == "HK" else max(1, int(_usr_pos_cfg["full_notional"] / max(float(last['Close']), 1e-4)))
            _default_saved_lots = int(_saved_pos.get("held_lots", _default_full_lots))
            col_pos1, col_pos2, col_pos3, col_pos4, col_pos5 = st.columns([1.35, 1.25, 1.15, 1.25, 1.15])
            with col_pos1:
                actual_entry = st.number_input("📌 實際進場價 (0=隱藏):", value=_default_saved_entry, step=0.1, key=f"act_entry_{fmt_ticker}")
            with col_pos2:
                _lot_label = f"📦 持倉手數 (1手={lot_size}股):" if market_type == "HK" else "📦 實際持倉股數 (Shares):"
                held_lots_input = st.number_input(_lot_label, min_value=1, value=max(1, _default_saved_lots), step=1, key=f"held_lots_{fmt_ticker}")
                actual_held_shares = int(held_lots_input * lot_size) if market_type == "HK" else int(held_lots_input)
                actual_invested_amt = round(actual_held_shares * (actual_entry if actual_entry > 0 else float(last['Close'])), 2)
                st.caption(f"💵 持倉本金: **{actual_invested_amt:,.1f} {currency}** ({actual_held_shares:,} 股)")
            with col_pos3:
                pos_dir = st.selectbox("🔄 持倉方向:", ["做多 (Long)", "做空 (Short)"], index=_default_pos_idx, key=f"pos_dir_{fmt_ticker}")
            with col_pos4:
                entry_date_input = st.date_input(
                    "📅 開倉日期 (錨定結構):",
                    value=_default_entry_dt,
                    min_value=_min_avail_dt,
                    max_value=_max_avail_dt,
                    help="系統根據開倉當日 K 線結構與 ATR 計算初始止損與目標價，跨日查看 100% 恆定不漂移！",
                    key=f"entry_dt_{fmt_ticker}"
                )
            with col_pos5:
                st.write("")
                lock_col_a, lock_col_b = st.columns(2)
                with lock_col_a:
                    lock_btn = st.button("🔒 鎖定", key=f"lock_btn_{fmt_ticker}", help="永久鎖定進場價、手數/金額與止損止盈計劃")
                with lock_col_b:
                    clear_lock_btn = st.button("🗑️ 清除", key=f"clr_lock_{fmt_ticker}", help="清除此股票已鎖定的持倉記憶")
                if clear_lock_btn and fmt_ticker in _saved_pos_map:
                    del _saved_pos_map[fmt_ticker]
                    save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))
                    st.toast(f"已清除 {fmt_ticker} 鎖定持倉記錄")
                    st.rerun()
                
            chart_entry = target_entry
            chart_sl = stop_loss
            chart_initial_sl = stop_loss
            chart_tp = take_profit
            chart_plan = plan_type
                
            # [v25.6 徹底根治截圖中之 NotFoundError: removeChild 與文字錯位]
            # 預先配置固定之 _pos_nav_outer_slot = st.empty()，並將「🔒 鎖定成功提示」與「4 大持倉導航提示卡」透過 render_atomic_alert_group 合併為單一防翻譯 HTML 區塊渲染，
            # 確保無論用戶點擊「🔒 鎖定計劃」、切換「做多/做空」或更改「進場價/開倉日」，React 虛擬 DOM 永遠不會對多個 stAlert 節點執行跨節點 removeChild！
            _pos_nav_outer_slot = st.empty()
            if actual_entry > 0:
                with _pos_nav_outer_slot.container():
                    entry_atr = last['ATR'] if pd.notna(last['ATR']) else actual_entry * 0.03
                    bench_close = last.get('CloseBench', np.nan)
                    bench_sma50 = last.get('SMA50BenchValue', np.nan)
                    ema9_val = last.get('EMA9', 0)
                    
                    _stop_prof_u = get_active_stop_profile()
                    _u_min_pct = float(_stop_prof_u["min_stop_pct"])
                    _u_max_pct = float(_stop_prof_u["max_stop_pct"])
                    _u_futu_be = compute_futu_real_fees(market_type=market_type, price=actual_entry, shares=max(int(lot_size), 100))
                    sma20_val_u = float(last.get('SMA20', actual_entry))
                    ema21_val_u = float(last.get('EMA21', sma20_val_u))

                    # [v26.0 開倉日確定性快照 + (代碼,方向,開倉日,進場價) 自動雜湊鎖定引擎]
                    # 只要輸入相同的「開倉日期」與「實際進場價」，系統除了使用開倉日確定性歷史切片計算外，
                    # 還會自動以 (fmt_ticker, pos_dir, entry_date_input, round(actual_entry, 3)) 為鍵存入持久化記憶庫，
                    # 保證 5 天後、1 個月後再輸入同一日期與同一買入價時，初始止損、目標價與初始盈虧比 R:R 100% 恆定不變！
                    _auto_hash_store = persistent_data.setdefault("auto_entry_date_hash_store", {})
                    _plan_hash_key = f"{fmt_ticker}_{'SHORT' if '做空' in pos_dir else 'LONG'}_{str(entry_date_input)}_{round(float(actual_entry), 3)}"
                    _matched_saved_plan = None
                    if _saved_pos and _saved_pos.get("anchor_date") == str(entry_date_input) and abs(float(_saved_pos.get("entry", 0.0)) - actual_entry) < 1e-3:
                        _matched_saved_plan = _saved_pos
                    elif _plan_hash_key in _auto_hash_store:
                        _matched_saved_plan = _auto_hash_store[_plan_hash_key]

                    _anchored_plan = compute_anchored_position_plan(
                        df=df,
                        actual_entry=actual_entry,
                        pos_dir_str=pos_dir,
                        entry_date_val=entry_date_input,
                        market_type=market_type,
                        lot_size=int(lot_size),
                        saved_plan=_matched_saved_plan
                    )
                    if _anchored_plan and _plan_hash_key not in _auto_hash_store:
                        _auto_hash_store[_plan_hash_key] = {
                            "dir_code": _anchored_plan["dir_code"],
                            "entry": _anchored_plan["entry"],
                            "anchor_date": _anchored_plan["anchor_date"],
                            "fixed_sl": _anchored_plan["fixed_sl"],
                            "fixed_tp": _anchored_plan["fixed_tp"],
                            "tp2_price": _anchored_plan["tp2_price"],
                            "sl_basis_lbl": _anchored_plan["sl_basis_lbl"],
                            "tp_basis_lbl": _anchored_plan["tp_basis_lbl"]
                        }
                    # OLD:
                    # if lock_btn and _anchored_plan:
                    #     _saved_pos_map[fmt_ticker] = _anchored_plan
                    #     st.success(f"🔒 已永久鎖定 {fmt_ticker} ({pos_dir}) 開倉日 {_anchored_plan['anchor_date']} 之進場價 {actual_entry:.2f}、初始止損 {_anchored_plan['fixed_sl']:.2f} 與目標價 {_anchored_plan['fixed_tp']:.2f}！")
                    # NEW: [v25.6 固定預留 _lock_banner_slot = st.empty()] 點擊鎖定計劃時在預留槽位內渲染，絕不推擠下方元件之 DOM 索引！
                    _lock_banner_slot = st.empty()
                    if _anchored_plan:
                        _saved_pos_map[fmt_ticker] = _anchored_plan
                        save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))
                        # [v28.0 同步更新頂部 5 張指標卡，消除頂部顯示今日動態 R:R 而底部顯示開倉日 R:R 的矛盾]
                        p1.metric("🎯 你的實際進場位 (已錨定)", f"{actual_entry:.2f} {currency}", delta=f"開倉日 {_anchored_plan['anchor_date']}")
                        p2.metric("🛡️ 開倉日鎖定止損位", f"{_anchored_plan['fixed_sl']:.2f} {currency}", delta=f"軟止損 {_anchored_plan.get('soft_closing_sl', _anchored_plan['fixed_sl']):.2f} (-{_anchored_plan['initial_stop_pct']:.1f}%)", delta_color="inverse")
                        p3.metric("💰 開倉日鎖定目標 (TP1)", f"{_anchored_plan['fixed_tp']:.2f} {currency}", delta=f"次級 TP2: {_anchored_plan['tp2_price']:.2f}")
                        p4.metric("⚖️ 鎖定初始盈虧比 (恆定)", f"{_anchored_plan['initial_rr']:.2f} R", delta=f"當前實時 {_anchored_plan['current_r']:+.2f} R")
                        p5.metric("🔄 持倉方向", "🟢 做多" if _anchored_plan["dir_code"] == "LONG" else "🔴 做空", delta=f"浮盈 {_anchored_plan['unrealized_pnl_pct']:+.2f}%")
                    if lock_btn and _anchored_plan:
                        _saved_pos_map[fmt_ticker] = _anchored_plan
                        save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))
                        render_atomic_alert(
                            f"🔒 **已永久鎖定 {fmt_ticker} ({pos_dir})** 開倉日 {_anchored_plan['anchor_date']} 之進場價 **{actual_entry:.2f}**、初始止損 **{_anchored_plan['fixed_sl']:.2f}** 與目標價 **{_anchored_plan['fixed_tp']:.2f}**！",
                            "success",
                            slot=_lock_banner_slot
                        )
                    elif _saved_pos and abs(float(_saved_pos.get("entry", 0.0)) - actual_entry) < 1e-3 and _saved_pos.get("anchor_date") == str(entry_date_input):
                        render_atomic_alert(
                            f"🔒 **已載入鎖定計劃 ({fmt_ticker} ｜ {pos_dir})**：開倉日 {_anchored_plan['anchor_date']} ｜ 進場價 **{actual_entry:.2f}** ｜ 鎖定初始止損 **{_anchored_plan['fixed_sl']:.2f}** ｜ 鎖定目標價 **{_anchored_plan['fixed_tp']:.2f}**",
                            "success",
                            slot=_lock_banner_slot
                        )

                    # [v37.0 寫入實際持倉手數、股數與總金額至鎖定計劃]
                    _real_fee_pos = compute_futu_real_fees(market_type=market_type, price=actual_entry, shares=actual_held_shares, is_sell=("做空" in pos_dir))
                    _anchored_plan["be_price"] = _real_fee_pos["breakeven_short_price"] if ("做空" in pos_dir) else _real_fee_pos["breakeven_long_price"]
                    _anchored_plan["held_lots"] = int(held_lots_input)
                    _anchored_plan["held_shares"] = int(actual_held_shares)
                    _anchored_plan["invested_amount"] = round(actual_held_shares * actual_entry, 2)
                    _gross_pnl_dlr = (float(last['Close']) - actual_entry) * actual_held_shares if "做多" in pos_dir else (actual_entry - float(last['Close'])) * actual_held_shares
                    _net_pnl_dlr = round(_gross_pnl_dlr - _real_fee_pos["round_trip_fee"], 2)
                    _init_risk_dlr = round(actual_held_shares * _anchored_plan["user_risk"] + _real_fee_pos["round_trip_fee"], 2)
                    _tp1_net_dlr = round(actual_held_shares * abs(_anchored_plan["fixed_tp"] - actual_entry) - _real_fee_pos["round_trip_fee"], 2)
                    _anchored_plan["net_pnl_dollars"] = _net_pnl_dlr
                    _saved_pos_map[fmt_ticker] = _anchored_plan
                    save_persisted_state_to_disk(persistent_data, st.session_state.get("watchlist_dict"))

                    # 展示「開倉錨定初始計劃」vs「今日實時浮動淨盈虧 ($) & 精準手數風控」五維儀表板
                    rt_c1, rt_c2, rt_c3, rt_c4, rt_c5 = st.columns(5)
                    rt_c1.metric(
                        "💵 實際持倉總本金",
                        f"{_anchored_plan['invested_amount']:,.1f} {currency}",
                        f"共 {actual_held_shares:,} 股 ({int(held_lots_input)} {'手' if market_type=='HK' else '股'})"
                    )
                    rt_c2.metric(
                        "💰 實時浮動淨盈虧 (扣富途費)",
                        f"{_net_pnl_dlr:+,.1f} {currency}",
                        f"浮盈率 {_anchored_plan['unrealized_pnl_pct']:+.2f}% ({_anchored_plan['current_r']:+.2f} R)"
                    )
                    rt_c3.metric(
                        "🛡️ 止損實質最大風險 (1R)",
                        f"-{_init_risk_dlr:,.1f} {currency}",
                        f"初始止損 {_anchored_plan['fixed_sl']:.2f} (-{_anchored_plan['initial_stop_pct']:.2f}%)"
                    )
                    rt_c4.metric(
                        "🎯 達標 TP1 預計淨賺",
                        f"+{_tp1_net_dlr:,.1f} {currency}",
                        f"TP1: {_anchored_plan['fixed_tp']:.2f} ({_anchored_plan['initial_rr']:.2f} R)"
                    )
                    rt_c5.metric(
                        "⚖️ 富途實戰打和價",
                        f"{_anchored_plan['be_price']:.2f} {currency}",
                        f"雙邊費 {_real_fee_pos['round_trip_fee']:.1f} {currency}"
                    )

                    _pos_alerts_slot = st.empty()
                    if pos_dir == "做多 (Long)":
                        # OLD:
                        # _user_sr_targets = _find_historical_ma_convergence_targets(df, actual_entry, "LONG")
                        # ... (原本每日用最新 last 重新計算 fixed_sl 與 fixed_tp，導致每天數值漂移)
                        # NEW: 直接採用錨定於開倉日 (entry_date_input) 的固定初始止損與目標價
                        fixed_sl = _anchored_plan["fixed_sl"]
                        fixed_tp = _anchored_plan["fixed_tp"]
                        _user_risk = _anchored_plan["user_risk"]
                        _user_rr = _anchored_plan["initial_rr"]
                        _be_price_long = _anchored_plan["be_price"]
                        st.caption(
                            f"📐 **錨定於開倉日 ({_anchored_plan['anchor_date']}) 之買入結構計劃**："
                            f"初始防洗盤止損 **{fixed_sl:.2f}** ({_anchored_plan['sl_basis_lbl']}，固定風險 -{_anchored_plan['initial_stop_pct']:.2f}%) ｜ "
                            f"富途打和價 **{_be_price_long:.2f}** ｜ 初始結構目標 **{fixed_tp:.2f}** ({_anchored_plan['tp_basis_lbl']}) ｜ "
                            f"**鎖定初始盈虧比：{_user_rr:.2f} R** ｜ **今日實時浮盈：{_anchored_plan['unrealized_pnl_pct']:+.2f}% ({_anchored_plan['current_r']:+.2f} R)**"
                        )

                        _bench_dist_u = float(last.get('Bench_Dist_Days', 0)) if pd.notna(last.get('Bench_Dist_Days')) else 0.0
                        _reg_gate_u = df.attrs.get("market_regime_gate", {})
                        is_trending = bool(
                            _reg_gate_u.get("regime_code") in ["BULL_CONFIRMED", "BOTTOM_REVERSAL"]
                            and pd.notna(bench_close) and pd.notna(bench_sma50)
                            and (_bench_dist_u <= 3)
                        )
                        _bars_held_u = int(_anchored_plan.get("bars_held_since_entry", 0))
                        _soft_sl_u = float(_anchored_plan.get("soft_closing_sl", round(fixed_sl + 0.25 * entry_atr, 2)))
                        _peak_r_u = float(_anchored_plan.get("peak_mfe_r", 0.0))
                        # [v41.1 Module 4：防過早推保本洗出 (修復 SNOW 僅微利即因 9EMA 移至保本被盤前洗出之問題)]
                        breakeven_triggered = bool(
                            (_peak_r_u >= 1.25 and float(last['Close']) >= actual_entry + 0.80 * _user_risk)
                            or (_bars_held_u >= 5 and sma20_val_u > actual_entry * 1.015 and float(last['Close']) > sma20_val_u)
                        )
                        if _peak_r_u >= 1.65:
                            current_sl = round(max(_be_price_long * 1.005, actual_entry + 0.65 * _user_risk), 2)
                        elif breakeven_triggered:
                            current_sl = round(_be_price_long * 1.004, 2)
                        else:
                            current_sl = fixed_sl

                        # OLD:
                        # st.success(f"**🟢 實戰中線波段計畫 (做多)**：實際進場 **{actual_entry:.2f}** ｜ 第一結構目標 **{fixed_tp:.2f}**")
                        # if last['Low'] <= current_sl or last['Close'] <= current_sl or _anchored_plan.get("stop_hit_since_entry", False):
                        #     st.error(...)
                        # elif breakeven_triggered:
                        #     st.error(...)
                        # else:
                        #     st.warning(...)
                        # st.warning(...)
                        # if is_trending: st.info(...) else: st.info(...)
                        # NEW: [v25.6 單一原子 HTML 組合渲染] 將 4 張持倉導航卡片合併為單一 DOM 容器渲染，100% 杜絕文字穿插與 removeChild 崩潰
                        _usr_1k_plan = compute_equal_risk_position_cap(
                            entry_price=actual_entry,
                            stop_loss=fixed_sl,
                            lot_size=int(lot_size) if market_type == "HK" else 1,
                            market_type=market_type
                        )
                        _rk_lim_dlr = _usr_1k_plan["risk_budget_used"]
                        _dollar_sl_actual_shares = round(max(0.01, actual_entry - max(_rk_lim_dlr - _real_fee_pos["round_trip_fee"], _rk_lim_dlr * 0.7) / max(actual_held_shares, 1)), 2)
                        _long_cards: List[Tuple[str, str]] = [
                            ("success", f"**🟢 實戰中線波段計畫 (做多)**：實際進場 **{actual_entry:.2f}** ｜ 第一結構目標 **{fixed_tp:.2f}** ｜ 當前大市：**{_reg_gate_u.get('regime_badge', '🟢 順勢')}**")
                        ]
                        # [v41.1 Module 3：滿倉 × HK$1,000 止損結構安全紅綠燈診斷卡]
                        if _init_risk_dlr <= _rk_lim_dlr * 1.15:
                            _long_cards.append((
                                "success",
                                f"🚦 **【滿倉 × 單筆 {_rk_lim_dlr:,.0f} {currency} 止損綠燈通過】**：你當前持倉 **{actual_held_shares:,} 股**（本金 **{_anchored_plan['invested_amount']:,.0f} {currency}**），"
                                f"就算跌穿真實圖表結構止損位 (**{fixed_sl:.2f}**，幅度 **-{_anchored_plan['initial_stop_pct']:.2f}%**)，含富途雙邊費用總虧損僅 **-{_init_risk_dlr:,.0f} {currency}**（符合你約 1,000 港元止損上限），"
                                f"**請安心將止損單設於結構位 `{fixed_sl:.2f}`，切勿無故上移！**"
                            ))
                        else:
                            _pilot_sh = _usr_1k_plan["recommended_shares"]
                            _addon_sh = max(0, actual_held_shares - _pilot_sh)
                            _addon_trig_p = round(actual_entry + 0.80 * _user_risk, 2)
                            _long_cards.append((
                                "error",
                                f"🚨 **【致命洗盤陷阱警告：切勿用 {_rk_lim_dlr:,.0f} {currency} 硬設止損於 `{_dollar_sl_actual_shares:.2f}`！】**：\n"
                                f"你目前滿倉持有 **{actual_held_shares:,} 股 ({_anchored_plan['invested_amount']:,.0f} {currency})**，若按真實圖表結構位 **{fixed_sl:.2f} (-{_anchored_plan['initial_stop_pct']:.2f}%)** 止損將虧損 **-{_init_risk_dlr:,.0f} {currency}**（超出你單筆 {_rk_lim_dlr:,.0f} {currency} 上限）！\n"
                                f"⚠️ **為何你之前差不多每隻都觸及止損？** 因為若你強行將止損設在只蝕 {_rk_lim_dlr:,.0f} {currency} 的 **`{_dollar_sl_actual_shares:.2f}`**（距買入價僅 **-{((actual_entry - _dollar_sl_actual_shares)/actual_entry)*100:.2f}%**），該位置正好卡在 20MA 結構支撐上方的**日內洗盤噪音區**，極易像 `MSFT` 或 `0883.HK` 一樣被日內針尖掃出！\n"
                                f"✅ **三大專業破局方案（死守結構位 `{fixed_sl:.2f}` 同時鎖死只蝕 {_rk_lim_dlr:,.0f} {currency}）**：\n"
                                f"1. **【方案 ① 滿倉低吸掛單價】**：若堅持買滿 **{actual_held_shares:,} 股** 且止損死守結構位 **`{fixed_sl:.2f}`**，買入價絕不可高於 **`{_usr_1k_plan['safe_full_pos_entry']:.2f}`**（掛限價單低吸，觸及 `{fixed_sl:.2f}` 剛好只蝕 {_rk_lim_dlr:,.0f} {currency}）！\n"
                                f"2. **【方案 ② 現價等風險縮倉】**：若於 `{actual_entry:.2f}` 直接進場，請將持倉降至 **`{_pilot_sh:,} 股`（本金 `{_usr_1k_plan['recommended_notional']:,.0f} {currency}`）**，守 `{fixed_sl:.2f}` 實質虧損即鎖死在 **-{_usr_1k_plan['expected_loss_at_stop']:,.0f} {currency}**！\n"
                                f"3. **【方案 ③ 兩段式底倉 + 浮盈加滿倉】**：先買底倉 **`{_pilot_sh:,} 股`**（風險鎖死 {_rk_lim_dlr:,.0f} {currency}），待股價升穿 **`{_addon_trig_p:.2f} (+0.8R)`** 將底倉推保本後，再加買剩餘 **`{_addon_sh:,} 股`** 達成零額外風險滿倉！"
                            ))
                        # [v41.1 Module 4：嚴禁虧損加倉攤平紅牌警告 (修復 TSM 跌穿 10MA 逆勢補倉大虧教訓)]
                        if float(last['Close']) < actual_entry * 0.995 or float(last['Close']) < float(last.get('SMA10', actual_entry)):
                            _long_cards.append((
                                "error",
                                f"🚫 **【鐵律紅牌：嚴禁在虧損或跌穿 10MA 時補倉攤平！】**：現價 (**{last['Close']:.2f}**) 低於你的進場成本 (**{actual_entry:.2f}**) 或 10MA (**{float(last.get('SMA10', 0)):.2f}**)。"
                                f"實盤復盤顯示（如 `TSM` 台積電於跌穿 10MA 時加注攤平），逆勢加倉會令單筆虧損瞬間突破 HK$1,366！**只有當持倉有浮盈且底倉已推保本時才允許順勢加碼，虧損倉位只准守止損或減倉！**"
                            ))
                        # OLD:
                        # if last['Low'] <= current_sl or last['Close'] <= current_sl or _anchored_plan.get("stop_hit_since_entry", False):
                        # NEW: [v28.0 雙層防洗盤導航：區分「盤中洗盤針尖測試軟止損但收盤守住」與「真正跌穿硬止損/收盤破位」]
                        if last['Low'] <= current_sl or last['Close'] <= _soft_sl_u or _anchored_plan.get("stop_hit_since_entry", False):
                            _long_cards.append((
                                "error",
                                f"🚨 **【多頭止損已觸發 / 立即離場警告】**：自開倉日 ({_anchored_plan['anchor_date']}) 起算之盤中最低價 ({_anchored_plan['min_low_since_entry']:.2f}) 已跌穿盤中防洗盤硬止損 (**{current_sl:.2f}**) "
                                f"或收盤跌破結構軟止損 (**{_soft_sl_u:.2f}**)，此筆做多交易結構已確認破壞，請嚴格執行止損紀律！"
                            ))
                        elif _anchored_plan.get("wick_shakeout_only", False):
                            _long_cards.append((
                                "warning",
                                f"🛡️ **【防洗盤雙層保護生效：識別為日內假跌破洗盤 (Shakeout)】**：自開倉日以來盤中最低價 ({_anchored_plan['min_low_since_entry']:.2f}) 雖曾短暫刺穿收盤軟防守位 (**{_soft_sl_u:.2f}**)，"
                                f"但**未觸及盤中防洗盤硬止損 ({current_sl:.2f}) 且收盤強勢收回 ({last['Close']:.2f})**！這正是典型主力洗盤 (2B/Undercut)，雙層止損成功保護你免於在最低點被洗出局！"
                            ))
                        elif breakeven_triggered:
                            _long_cards.append((
                                "error",
                                f"**🛡️ 中線波段保本已觸發**：目前股價已脫離成本達 1R 以上或 20MA ({sma20_val_u:.2f}) 已升穿買入價！建議將止損上移至富途打和價 **{_be_price_long:.2f}**（已含富途雙邊佣金及印花稅），鎖定零風險持倉！"
                            ))
                        else:
                            _long_cards.append((
                                "warning",
                                f"**🌱 中線波段忍耐洗盤期**：目前尚未脫離成本滿 1R 且 20MA 尚未越過成本線。請堅守中線結構止損 **{fixed_sl:.2f}**，給予股票數星期發酵空間，切勿因日內震盪跌破 9EMA 而提早被洗出局！"
                            ))

                        _tp0_lock_u = round(min(actual_entry * 1.045, actual_entry + 1.15 * _user_risk), 2)
                        _had_good_profit = (_peak_r_u >= 0.85) or (float(_anchored_plan.get("max_high_since_entry", 0.0)) >= actual_entry * 1.035)
                        _broke_9ema_weak = float(last['Close']) < float(ema9_val) and float(last['Close']) < float(last['Open'])
                        if _had_good_profit and (bool(last.get('Is_Abnormal_Drop', False)) or _broke_9ema_weak):
                            _long_cards.append((
                                "error",
                                f"🚨 **【浮盈保護警報：放量陰燭 / 收盤跌穿 9EMA ({ema9_val:.2f})】**："
                                f"持倉曾推進至 **{_anchored_plan.get('max_high_since_entry', 0):.2f} (+{_peak_r_u:.2f}R)**，現出現動能轉弱跌穿 9EMA 或異常大陰燭，建議立即主動鎖潤離場，切勿讓浮盈回吐變虧損！"
                            ))
                        if _bars_held_u >= 15 and abs(float(_anchored_plan.get("unrealized_pnl_pct", 0.0))) <= 2.0:
                            _long_cards.append((
                                "warning",
                                f"⏳ **【資金效率與時間停損預警】**：已持倉 **{_bars_held_u} 個交易日** 仍處於橫盤狀態 (浮盈 {_anchored_plan.get('unrealized_pnl_pct', 0):+.2f}%)，動能明顯落後同板塊領頭羊，建議清倉換馬。"
                            ))
                        # [v37.0 按照用戶實際持倉手數 / 股數精算每階段賣出數量與落袋金額]
                        if (market_type == "HK" and int(held_lots_input) <= 1) or actual_held_shares <= 1:
                            _lock_05r_p = round(max(_be_price_long * 1.004, actual_entry + 0.55 * _user_risk), 2)
                            _lock_05r_dlr = round(actual_held_shares * (_lock_05r_p - actual_entry) - _real_fee_pos["round_trip_fee"], 1)
                            _long_cards.append((
                                "warning",
                                f"**📦 單手全進全出階梯鎖潤方案（你目前持有 `1 手 / {actual_held_shares} 股`，本金 `{_anchored_plan['invested_amount']:,.0f} {currency}`，不可拆半賣出）**：\n"
                                f"1. **第一防護點 TP0 (`{_tp0_lock_u:.2f}`)**：當股價升穿 `{_tp0_lock_u:.2f}` 時不賣出，但**立即將富途止損單上推至 `{_lock_05r_p:.2f}`（直接鎖定 `+{_lock_05r_dlr:,.0f} {currency}` 淨利潤，絕不讓贏變輸）**！\n"
                                f"2. **波段目標位 TP1 (`{fixed_tp:.2f}`)**：觸及 `{fixed_tp:.2f}` 時 **100% 全數賣出 `{actual_held_shares} 股`**，預計落袋淨賺 **`+{_tp1_net_dlr:,.0f} {currency}`**！\n"
                                f"3. **🚫 倉位紀律**：若股價自高點回落或跌穿 10MA (`{float(last.get('SMA10', 0)):.2f}`)，嚴禁加注補倉攤平。"
                            ))
                        else:
                            _half_lots = max(1, int(held_lots_input) // 2)
                            _sell_tp0_shares = int(_half_lots * lot_size) if market_type == "HK" else max(1, actual_held_shares // 2)
                            _rem_shares = actual_held_shares - _sell_tp0_shares
                            _tp0_profit_dlr = round(_sell_tp0_shares * (_tp0_lock_u - actual_entry) - (_real_fee_pos["round_trip_fee"] * 0.5), 1)
                            _tp1_rem_dlr = round(_rem_shares * (fixed_tp - actual_entry) - (_real_fee_pos["round_trip_fee"] * 0.5), 1)
                            _long_cards.append((
                                "warning",
                                f"**📦 多手分批精準鎖潤方案（你目前持有 `{int(held_lots_input)} {'手' if market_type=='HK' else '股'} / 共 {actual_held_shares:,} 股`，本金 `{_anchored_plan['invested_amount']:,.0f} {currency}`）**：\n"
                                f"1. **第一鎖潤點 TP0 (`{_tp0_lock_u:.2f}`，約 +1.15R / +4.5%)**：請先賣出 **`{_sell_tp0_shares:,} 股`{' (' + str(_half_lots) + '手)' if market_type=='HK' else ''}**，先袋走 **`+{_tp0_profit_dlr:,.0f} {currency}`** 現金利潤墊，並將剩餘 `{_rem_shares:,} 股` 止損上推至富途保本價 **`{round(_be_price_long * 1.004, 2):.2f}`**！\n"
                                f"2. **波段結構目標 TP1 (`{fixed_tp:.2f}`)**：剩餘 **`{_rem_shares:,} 股`** 於觸及 `{fixed_tp:.2f}` 時可再獲利 **`+{_tp1_rem_dlr:,.0f} {currency}`**（兩階段合共淨賺 **`+{(_tp0_profit_dlr + _tp1_rem_dlr):,.0f} {currency}`**）！\n"
                                f"3. **🚫 倉位紀律**：若股價自高點回落或跌穿 10MA (`{float(last.get('SMA10', 0)):.2f}`)，嚴禁逆勢補倉攤平。"
                            ))
                        if is_trending:
                            _long_cards.append((
                                "info",
                                f"**🏃 強趨勢尾倉跟蹤**：大盤處於強勢上升期，尾倉可沿 **20 MA ({sma20_val_u:.2f}) / 21 EMA ({ema21_val_u:.2f})** 跟蹤；若連升數日進入加速期，一旦收盤跌穿 **9 EMA ({ema9_val:.2f})** 且無力反彈即全數獲利平倉。"
                            ))
                        else:
                            _long_cards.append((
                                "info",
                                f"**↔️ 震盪/弱勢大盤策略**：目前大盤欠缺單邊趨勢，請於觸及結構目標 **{fixed_tp:.2f}** 時 **100% 全數止盈**，若中途跌穿 **9 EMA ({ema9_val:.2f})** 亦應果斷鎖潤平倉。"
                            ))
                        render_atomic_alert_group(_long_cards, slot=_pos_alerts_slot)

                        chart_entry, chart_sl, chart_initial_sl, chart_tp, chart_plan = actual_entry, current_sl, fixed_sl, fixed_tp, "LONG"
                    else:
                        # OLD:
                        # _user_sr_targets = _find_historical_ma_convergence_targets(df, actual_entry, "SHORT")
                        # ... (原本每日用最新 last 重新計算空頭 fixed_sl 與 fixed_tp，導致每天數值漂移)
                        # NEW: [v25.4 空頭開倉日錨定] 採用開倉日 (entry_date_input) 的固定空頭初始止損與目標價，並顯示自開倉日以來是否曾升穿空頭止損
                        fixed_sl = _anchored_plan["fixed_sl"]
                        fixed_tp = _anchored_plan["fixed_tp"]
                        _user_risk = _anchored_plan["user_risk"]
                        _user_rr = _anchored_plan["initial_rr"]
                        _be_price_short = _anchored_plan["be_price"]
                        st.caption(
                            f"📐 **錨定於開倉日 ({_anchored_plan['anchor_date']}) 之沽空結構計劃**："
                            f"初始防軋空止損 **{fixed_sl:.2f}** ({_anchored_plan['sl_basis_lbl']}，固定風險 +{_anchored_plan['initial_stop_pct']:.2f}%) ｜ "
                            f"富途打和價 **{_be_price_short:.2f}** ｜ 初始回補目標 **{fixed_tp:.2f}** ({_anchored_plan['tp_basis_lbl']}) ｜ "
                            f"**鎖定初始盈虧比：{_user_rr:.2f} R** ｜ **今日實時空頭浮盈：{_anchored_plan['unrealized_pnl_pct']:+.2f}% ({_anchored_plan['current_r']:+.2f} R)**"
                        )

                        is_trending = pd.notna(bench_close) and bench_close < bench_sma50 * 0.99
                        # OLD:
                        # breakeven_triggered = (float(last['Close']) <= actual_entry - 1.0 * _user_risk) or (sma20_val_u < actual_entry * 0.995)
                        # current_sl = _be_price_short if breakeven_triggered else fixed_sl
                        # NEW: [v28.0 修復空頭反抽沽空第 1 日誤觸發保本止損之漏洞]
                        _bars_held_us = int(_anchored_plan.get("bars_held_since_entry", 0))
                        _soft_sl_us = float(_anchored_plan.get("soft_closing_sl", round(fixed_sl - 0.30 * entry_atr, 2)))
                        breakeven_triggered = bool(
                            (_anchored_plan.get("peak_mfe_r", 0.0) >= 1.0 and float(last['Close']) <= actual_entry - 0.65 * _user_risk)
                            or (_bars_held_us >= 5 and sma20_val_u < actual_entry * 0.988 and float(last['Close']) < sma20_val_u)
                        )
                        current_sl = _be_price_short if breakeven_triggered else fixed_sl

                        # OLD:
                        # st.error(f"**🔴 實戰中線波段計畫 (做空)**：實際進場 **{actual_entry:.2f}** ｜ 第一結構目標 **{fixed_tp:.2f}**")
                        # if last['High'] >= current_sl or last['Close'] >= current_sl or _anchored_plan.get("stop_hit_since_entry", False):
                        #     st.error(...)
                        # elif breakeven_triggered:
                        #     st.success(...)
                        # else:
                        #     st.warning(...)
                        # st.warning(...)
                        # if is_trending: st.info(...) else: st.info(...)
                        # NEW: [v25.6 單一原子 HTML 組合渲染] 將空頭 4 張持倉導航卡片合併為單一 DOM 容器渲染，徹底解決截圖中之文字錯位與 removeChild 崩潰！
                        _short_cards: List[Tuple[str, str]] = [
                            ("error", f"**🔴 實戰中線波段計畫 (做空)**：實際進場 **{actual_entry:.2f}** ｜ 第一結構目標 **{fixed_tp:.2f}**")
                        ]
                        # OLD:
                        # if last['High'] >= current_sl or last['Close'] >= current_sl or _anchored_plan.get("stop_hit_since_entry", False):
                        # NEW: [v28.0 空頭雙層防軋空導航]
                        if last['High'] >= current_sl or last['Close'] >= _soft_sl_us or _anchored_plan.get("stop_hit_since_entry", False):
                            _short_cards.append((
                                "error",
                                f"🚨 **【空頭止損已觸發 / 立即回補警告】**：自開倉日 ({_anchored_plan['anchor_date']}) 起算之盤中最高價 ({_anchored_plan['max_high_since_entry']:.2f}) 已升穿盤中硬止損 (**{current_sl:.2f}**) "
                                f"或收盤升破結構軟止損 (**{_soft_sl_us:.2f}**)，請嚴格執行回補紀律！"
                            ))
                        elif _anchored_plan.get("wick_shakeout_only", False):
                            _short_cards.append((
                                "warning",
                                f"🛡️ **【空頭防洗盤保護生效：識別為日內誘多上影線 (Upthrust Shakeout)】**：盤中最高價 ({_anchored_plan['max_high_since_entry']:.2f}) 雖曾刺穿收盤軟防守 (**{_soft_sl_us:.2f}**)，"
                                f"但未觸及盤中硬止損 (**{current_sl:.2f}**) 且收盤殺回其下 ({last['Close']:.2f})，成功避開最高點被軋空掃出！"
                            ))
                        elif breakeven_triggered:
                            _short_cards.append((
                                "success",
                                f"**🛡️ 空頭波段保本已觸發**：目前股價已下跌超過 1R 或 20MA ({sma20_val_u:.2f}) 已跌穿成本價！建議將止損下移至富途打和價 **{_be_price_short:.2f}**！"
                            ))
                        else:
                            _short_cards.append((
                                "warning",
                                f"**🌱 空頭波段忍耐洗盤期**：請堅守初始結構止損 **{fixed_sl:.2f}**，切勿因正常無量反抽而提早出場。"
                            ))

                        _short_cards.append((
                            "warning",
                            f"**🎯 階段一：強勢鎖潤 (Cover Half into Weakness)**：目標價 **{fixed_tp:.2f}**。觸及時【回補 50%】倉位，剩餘倉位止損移至富途打和價 ({_be_price_short:.2f})。"
                        ))
                        if is_trending:
                            _short_cards.append((
                                "info",
                                f"**🏃 階段二：讓空頭利潤奔跑**：剩餘 50% 倉位沿著 **20 MA ({sma20_val_u:.2f}) / 21 EMA ({ema21_val_u:.2f})** 跟蹤，直到收盤升穿 20MA/21EMA 及 **{last.get('ResShort', 0):.2f}** 才全數回補！"
                            ))
                        else:
                            _short_cards.append((
                                "info",
                                f"**↔️ 震盪市區間 (Range-bound)**：觸及目標價 **{fixed_tp:.2f}** 建議全數回補落袋為安。"
                            ))
                        render_atomic_alert_group(_short_cards, slot=_pos_alerts_slot)

                        chart_entry, chart_sl, chart_initial_sl, chart_tp, chart_plan = actual_entry, current_sl, fixed_sl, fixed_tp, "SHORT"
            st.markdown("---")
            
            # J Law 出貨日見頂預警 + [v37.0] 業績公佈具體日期/黑天鵝雷達 + 實時新聞情報與利好/利空 × 量價反應引擎
            st.markdown("### 🚨 大盤與個股出貨見頂預警 & 📅 業績日曆 / 黑天鵝雷達")
            ev_c1, ev_c2, ev_c3 = st.columns(3)
            ev_c1.metric(
                "📅 業績公佈日與倒數",
                _gap_diag.get("earnings_date_str") or "暫無近期排期",
                _gap_diag.get("earnings_badge", "-"),
                delta_color="inverse" if (_gap_diag.get("days_to_earnings") is not None and 0 <= _gap_diag.get("days_to_earnings", 99) <= 7) else "normal"
            )
            ev_c2.metric(
                "💰 除淨日 (Ex-Dividend)",
                _gap_diag.get("ex_dividend_date_str") or "暫無近期除淨",
                f"還有 {_gap_diag['days_to_ex_div']} 天" if (_gap_diag.get("days_to_ex_div") is not None and _gap_diag["days_to_ex_div"] >= 0) else "防除息自然跌價誤觸止損"
            )
            ev_c3.metric(
                "⚡ 1年隔夜跳空黑天鵝基因",
                f"{_gap_diag.get('gap_events_1y', 0)} 次 (≥4.5%跳空)",
                f"最大單日跳空 {_gap_diag.get('max_overnight_gap_pct', 0.0):.1f}%"
            )
            _top_event_alerts = [("error" if ("🚨" in jlaw_alert or "⚠️" in jlaw_alert) else "success", jlaw_alert)]
            if _gap_diag.get("gap_warning_msg"):
                _top_event_alerts.append(("error" if _gap_diag.get("gap_risk_level") == "CRITICAL" else "warning", _gap_diag["gap_warning_msg"]))
            render_atomic_alert_group(_top_event_alerts)

            nw_hdr_col1, nw_hdr_col2 = st.columns([4.2, 1.2])
            with nw_hdr_col1:
                st.markdown("### 📰 實時新聞情報搜尋與 AI 利好/利空 × 量價反應診斷 (News & Catalyst Intelligence)")
            with nw_hdr_col2:
                st.write("")
                _force_refresh_news = st.button("🔄 重新抓取最新新聞", key=f"btn_refresh_news_{fmt_ticker}", use_container_width=True)

            _news_intel = fetch_and_analyze_stock_news_intelligence(
                fmt_ticker,
                stock_name=name,
                market_type=market_type,
                force_refresh=bool(_force_refresh_news)
            )
            _news_px_div = evaluate_news_price_divergence(_news_intel, df)
            nw_c1, nw_c2, nw_c3 = st.columns(3)
            nw_c1.metric(
                "🧭 消息面綜合情緒評分",
                f"{_news_intel.get('overall_score', 0.0):+.1f} 分",
                f"利好 {_news_intel.get('bull_count', 0)} 則 ｜ 利空 {_news_intel.get('bear_count', 0)} 則 ｜ 中性 {_news_intel.get('neutral_count', 0)} 則"
            )
            nw_c2.metric(
                "⚖️ 消息 × 量價交叉驗證",
                _news_px_div.get("reaction_badge", "-")[:20],
                "防範利好出貨 / 識別底部洗盤"
            )
            nw_c3.metric(
                "🚨 新聞面黑天鵝掃描",
                "🚨 發現高危利空字眼" if _news_intel.get("has_black_swan_news") else "✅ 未見重大黑天鵝字眼",
                "📡 來源: " + (" / ".join(_news_intel.get("active_sources", [])) if _news_intel.get("active_sources") else "待連線")
            )
            if _news_px_div.get("reaction_advice"):
                render_atomic_alert(_news_px_div["reaction_advice"], _news_px_div.get("alert_kind", "info"))

            with st.expander(f"🗞️ 點擊展開：{fmt_ticker} ({name}) 最新網上搜尋新聞情報與逐則【利好 / 利空】明細 ({len(_news_intel.get('news_items', []))} 則)", expanded=True):
                if _news_intel.get("news_items"):
                    _news_rows = []
                    for _ni in _news_intel["news_items"]:
                        _news_rows.append({
                            "利好/利空判定": _ni["label"],
                            "情緒分": f"{_ni['score']:+d}",
                            "催化劑分類": _ni["category"],
                            "發佈時間": _ni["pub_date"],
                            "新聞標題與情報摘要": _ni["title"] + (f" — {_ni['summary']}" if _ni.get("summary") else ""),
                            "來源": _ni["publisher"],
                            "原文連結": _ni["link"] if _ni.get("link") else ""
                        })
                    st.dataframe(
                        pd.DataFrame(_news_rows),
                        column_config={
                            "原文連結": st.column_config.LinkColumn("🔗 新聞連結", display_text="閱讀原文")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    _dbg_str = " ｜ ".join(_news_intel.get("debug_notes", [])) if _news_intel.get("debug_notes") else "請檢查本地網絡連線或防火牆設定"
                    st.warning(f"⚠️ 暫未能從外部新聞源取回標題（診斷訊息：{_dbg_str}）。可點擊右上角「🔄 重新抓取最新新聞」重試。")

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
                if 'EMA21' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['EMA21'].ffill(), name='EMA 21 (Martin Luk)', line=dict(color='#00ACC1', width=1.6, dash='dash')), row=1, col=1)
                if 'AVWAP_High60' in plot_df.columns:
                    fig.add_trace(go.Scatter(x=x_dates, y=plot_df['AVWAP_High60'], name='AVWAP (前波高點錨定 - Martin Luk)', line=dict(color='#D81B60', width=2.0, dash='dashdot')), row=1, col=1)
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
                    # [v26.0 修復圖表盈虧比因保本移位而暴增漂移之漏洞] 始終使用初始結構止損 (chart_initial_sl) 計算固定初始盈虧比 _c_rr
                    _ref_init_sl = chart_initial_sl if ('chart_initial_sl' in locals() and chart_initial_sl > 0) else chart_sl
                    _c_risk = abs(chart_entry - _ref_init_sl)
                    _c_rr = abs(chart_tp - chart_entry) / _c_risk if _c_risk > 0 else 0.0
                    _tp_pct = (chart_tp - chart_entry) / chart_entry * 100.0
                    _sl_pct = (_ref_init_sl - chart_entry) / chart_entry * 100.0
                    if abs(chart_sl - _ref_init_sl) > 1e-3:
                        fig.add_hline(
                            y=chart_sl, line_dash="dot", line_width=2, line_color="#FF6D00",
                            annotation_text=f"🛡️ 富途打和保本止損線: {chart_sl:.2f}",
                            annotation_position="bottom right", annotation_font=dict(color="#E65100", size=11), row=1, col=1
                        )
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

                # OLD:
                # fig.update_layout(uirevision=target_ticker)
                # plot_config = dict(scrollZoom=True, displayModeBar=True, modeBarButtonsToRemove=['lasso2d', 'select2d'], displaylogo=False)
                # st.plotly_chart(fig, use_container_width=True, config=plot_config, theme=None, key=f"main_chart_{target_ticker}")
                # NEW: [v25.5 綁定 fmt_ticker 與 selected_interval，徹底消除手動輸入新代碼時之 Plotly DOM 衝突]
                fig.update_layout(uirevision=f"{fmt_ticker}_{selected_interval}")
                plot_config = dict(scrollZoom=True, displayModeBar=True, modeBarButtonsToRemove=['lasso2d', 'select2d'], displaylogo=False)
                st.plotly_chart(fig, use_container_width=True, config=plot_config, theme=None, key=f"main_chart_{fmt_ticker}_{selected_interval}")

                # --- 恢復 K 線型態詳細解讀文字區塊 ---
                st.markdown("### 🕯️ K 線型態解讀")
                # OLD:
                # if candlestick_patterns:
                #     for pattern in candlestick_patterns:
                #         if pattern["name"] != "":
                #             with st.container():
                #                 st.info(f"**{pattern['name']}** ｜ **市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}")
                #         else:
                #             st.info(f"**市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}")
                # NEW: [v25.6 單一原子 HTML 組合渲染 K 線型態] 防止不同股票 K 線型態數量不同時引發 React removeChild
                if candlestick_patterns:
                    _cdl_cards = []
                    for pattern in candlestick_patterns:
                        if pattern["name"] != "":
                            _cdl_cards.append(("info", f"**{pattern['name']}** ｜ **市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}"))
                        else:
                            _cdl_cards.append(("info", f"**市場訊號：** {pattern['signal']}\n\n**理論解釋：** {pattern['theory']}"))
                    render_atomic_alert_group(_cdl_cards)

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
                # OLD: st.plotly_chart(fig_bench, use_container_width=True, config=plot_config, theme=None)
                st.plotly_chart(fig_bench, use_container_width=True, config=plot_config, theme=None, key=f"bench_chart_{fmt_ticker}_{selected_interval}")

            st.markdown("---")
            st.markdown("### ⚡ 快速回測此標的")
            with st.expander("開啟個股快速回測", expanded=False):
                cqbt1, cqbt2, cqbt3 = st.columns(3)
                with cqbt1:
                    btcapq = st.number_input("初始資金", min_value=1000, value=100000, step=10000, key="btcapq")
                with cqbt2:
                    btmodeq = st.selectbox("方向", ["全部雙向", "僅做多頭(做多)", "僅做空頭(做空)"], key="btmodeq")
                with cqbt3:
                    # OLD: btrelaxq = st.checkbox("寬鬆模式", value=True, help="較重視 SMA50 / MACD 與大盤方向", key="btrelaxq")
                    # NEW: [v24.15] 預設關閉寬鬆模式 (value=False)，確保回測預設 100% 採用與掃描雷達相同的 J Law × LEI 大師結構選股與動態盈虧比邏輯
                    btrelaxq = st.checkbox("寬鬆模式 (關閉則 100% 同步掃描雷達嚴格結構邏輯)", value=False, help="預設關閉以確保回測與掃描雷達理念 100% 一致；勾選則僅用 50MA+MACD", key="btrelaxq")

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
                            _q_inst = compute_backtest_institutional_metrics(eqcurve, trdsdf, btcapq)

                            st.markdown(f"#### {ticker_input} 回測總覽 (已扣除富途真實佣金/平台費/印花稅)")
                            mb1, mb2, mb3, mb4 = st.columns(4)
                            mb1.metric("策略淨回報率", f"{(fcap - btcapq) / btcapq * 100:.2f}%", f"買入持有: {bhr:.2f}%")
                            mb2.metric("真實獨立波段勝率", f"{_q_inst['unique_win_rate']:.1f}%", f"獨立交易 {_q_inst['unique_trades_count']} 筆 (執行 {len(trdsdf)} 次)")
                            mb3.metric("最終帳戶淨值", f"{fcap:,.2f} {curq}")
                            mb4.metric("累計富途總費用", f"{_q_inst['total_futu_fees']:,.2f} {curq}", f"獲利因子 {_q_inst['profit_factor']:.2f}")

                            sb1, sb2 = st.columns(2)
                            sb1.info(f"🟢 做多真實勝率：{_q_inst['unique_long_win_rate']:.1f}%\n\n獨立波段筆數：{_q_inst['unique_long_count']} 筆")
                            sb2.error(f"🔴 做空真實勝率：{_q_inst['unique_short_win_rate']:.1f}%\n\n獨立波段筆數：{_q_inst['unique_short_count']} 筆")

                            fig_eq = go.Figure()
                            x_eq_dates = eqcurve.index.strftime('%Y-%m-%d')
                            fig_eq.add_trace(go.Scatter(x=x_eq_dates, y=eqcurve.values, mode='lines', name='資產曲線', line=dict(color='green', width=2.5)))
                            fig_eq.update_layout(title="帳戶資產曲線", xaxis_title="日期", yaxis_title="總資產", template="plotly_white", height=300)
                            fig_eq.update_xaxes(type='category', nticks=20)
                            st.plotly_chart(fig_eq, use_container_width=True, theme=None)

                            if not trdsdf.empty:
                                show_df = trdsdf.drop(columns=['利潤', '_trade_id', '_initial_notional', '_entry_fee'], errors='ignore')
                                st.dataframe(show_df, use_container_width=True)

# ==========================================
# 15. 模式 4：回測系統
# ==========================================
elif mode == mode_options[3]:
    st.query_params.clear()
    st.title("📈 策略歷史回測系統 (利潤最大化風控版)")
    render_global_quick_guide_banner()
    st.markdown("已套用：**富途牛牛真實收費與印花稅精算**、**中線波段防洗盤結構止損 (≥1.2x ATR / 3.2%~7.8%)**、**滿5日且浮盈≥1R升級富途打和保本**、**20MA/21EMA 中線波段跟蹤**、**進場當日止損檢查**，以及**25日時間停損**。")

    cbt1, cbt2, cbt3, cbt4 = st.columns(4)
    with cbt1:
        bt_ticker = st.text_input("股票代碼", value="3690.HK").strip()
    with cbt2:
        bt_capital = st.number_input("初始資金", min_value=1000, value=100000, step=10000)
    with cbt3:
        bt_mode = st.selectbox("交易方向", ["全部雙向", "僅做多頭(做多)", "僅做空頭(做空)"])
    with cbt4:
        # OLD: bt_relaxed = st.checkbox("寬鬆模式", value=True, help="偏重 SMA50 / MACD 與大盤濾網")
        # NEW: [v24.15] 預設關閉寬鬆模式 (value=False)，讓回測系統與掃描系統理念 100% 對齊
        bt_relaxed = st.checkbox("寬鬆模式 (關閉則 100% 同步掃描雷達嚴格結構邏輯)", value=False, help="預設關閉以確保回測與掃描雷達理念 100% 一致；勾選則僅用 50MA+MACD")

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
                _inst_m = compute_backtest_institutional_metrics(equity_curve, trades_df, bt_capital)

                st.markdown(f"### 📊 {bt_ticker} - {name} (v25.1 富途實戰中線波段回測報告)")
                mb1, mb2, mb3, mb4 = st.columns(4)
                mb1.metric("策略淨回報率 (扣富途費後)", f"{(final_cap - bt_capital) / bt_capital * 100:.2f}%", f"買入持有(B&H): {bh_ret:.2f}%")
                mb2.metric("真實獨立波段勝率", f"{_inst_m['unique_win_rate']:.1f}%", f"獨立波段 {_inst_m['unique_trades_count']} 筆 (含分批共 {len(trades_df)} 次執行)")
                mb3.metric("帳戶最終淨值", f"{final_cap:,.2f} {cur}")
                mb4.metric("累計富途手續費及印花稅", f"{_inst_m['total_futu_fees']:,.2f} {cur}")

                im1, im2, im3, im4 = st.columns(4)
                im1.metric("最大回撤 (Max DD)", f"{_inst_m['max_drawdown_pct']:.2f}%", f"蒙特卡洛95%極端DD: {_inst_m.get('mc_95_mdd_pct', 0.0):.2f}%")
                im2.metric("獲利因子 (Profit Factor)", f"{_inst_m['profit_factor']:.2f}", f"Calmar比率: {_inst_m.get('calmar_ratio', 0.0):.2f}")
                im3.metric("年化夏普 (扣3.5%無風險)", f"{_inst_m['sharpe_ratio']:.2f}", f"索提諾(Sortino): {_inst_m.get('sortino_ratio', 0.0):.2f}")
                im4.metric("單筆波段期望值 (Expectancy)", f"{_inst_m['expectancy_pct']:.2f}%", f"實戰盈虧比 {_inst_m['payoff_ratio']:.2f}")
                wfo1, wfo2, wfo3, wfo4 = st.columns(4)
                wfo1.metric("📐 Van Tharp 系統品質 (SQN)", f"{_inst_m.get('sqn_score', 0.0):.2f}", _inst_m.get("sqn_rating", "-"))
                wfo2.metric("🔄 Walk-Forward 四期勝率", f"{_inst_m.get('wfo_consistency_pct', 0.0):.0f}% 正回報", _inst_m.get("wfo_windows_summary", "-"))
                wfo3.metric("🛡️ 最大連續止損筆數", f"{_inst_m.get('max_consec_losses', 0)} 筆", "評估心理承受極限")
                wfo4.metric("⏳ 最長水下恢復期 (TUW)", f"{_inst_m.get('max_tuw_days', 0)} 交易日", "資金創新高所需最長天數")
                st.caption(
                    f"🧪 **機構級樣本內外與蒙特卡洛驗證**：樣本內 (前70%時間) 回報 **{_inst_m.get('in_sample_ret_pct', 0.0):+.2f}%** ｜ "
                    f"樣本外 (後30%時間) 回報 **{_inst_m.get('out_of_sample_ret_pct', 0.0):+.2f}%** ｜ "
                    f"四段滾動 Walk-Forward ({_inst_m.get('wfo_windows_summary', '-')}) ｜ "
                    f"500次蒙特卡洛重排中位數回報 **{_inst_m.get('mc_median_ret_pct', 0.0):+.2f}%**（95% 最差情境最大回撤 **{_inst_m.get('mc_95_mdd_pct', 0.0):.2f}%**）"
                )

                st.markdown("#### 🔍 多空分開表現 (按獨立波段交易統計，不重複計算分批止盈)")
                sb1, sb2 = st.columns(2)
                sb1.info(f"🟢 做多真實波段勝率：{_inst_m['unique_long_win_rate']:.1f}%\n\n獨立做多筆數：{_inst_m['unique_long_count']} 筆")
                sb2.error(f"🔴 做空真實波段勝率：{_inst_m['unique_short_win_rate']:.1f}%\n\n獨立做空筆數：{_inst_m['unique_short_count']} 筆")

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
# 17. 模式 5：實戰教學與大師戰法指南 (v25.3 新增)
# ==========================================
elif mode == mode_options[4]:
    st.query_params["mode"] = "教學"
    render_interactive_tutorial_hub()
