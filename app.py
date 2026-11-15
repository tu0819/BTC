import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 設定網頁標題與寬度
st.set_page_config(page_title="專業級智慧股票與資產管理平台", layout="wide")

st.title("專業級智慧股票與量化資產管理平台 (穩定高效版)")

# ==================== 初始化 Session State ====================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["0050.TW", "2330.TW", "2317.TW", "009816.TW"]

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# ==================== 側邊欄設定 ====================
st.sidebar.header("參數與導航面板")

selected_fav = st.sidebar.selectbox("快速載入收藏股票", ["-- 選擇收藏股票 --"] + st.session_state.watchlist)
default_symbol = selected_fav if selected_fav != "-- 選擇收藏股票 --" else "0050.TW"
stock_symbol = st.sidebar.text_input("輸入或修改股票代號 (台股加 .TW)", value=default_symbol)

col_fav1, col_fav2 = st.sidebar.columns(2)
with col_fav1:
    if st.button("加入收藏"):
        if stock_symbol.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(stock_symbol.upper())
            st.success(f"已將 {stock_symbol.upper()} 加入收藏！")
with col_fav2:
    if st.button("移除收藏"):
        if stock_symbol.upper() in st.session_state.watchlist:
            st.session_state.watchlist.remove(stock_symbol.upper())
            st.warning(f"已將 {stock_symbol.upper()} 移除收藏。")

st.sidebar.markdown("---")
benchmark_symbol = st.sidebar.text_input("大盤對標代號", value="^TWII")
period = st.sidebar.selectbox("選擇回測時間範圍", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=1)

st.sidebar.subheader("交易成本設定 (台股)")
include_fees = st.sidebar.checkbox("計入手續費與證交稅", value=True, help="自動扣除買賣手續費及證交稅（ETF享0.1%優惠稅率）")
broker_discount = st.sidebar.slider("電子手續費折扣 (折)", 1.0, 10.0, 2.8, 0.1)

st.sidebar.subheader("停損與停利設定")
enable_sl = st.sidebar.checkbox("啟用停損機制 (Stop-Loss)", value=True)
sl_pct = st.sidebar.slider("停損跌幅 (%)", 1.0, 20.0, 5.0, 0.5)

enable_tp = st.sidebar.checkbox("啟用停利機制 (Take-Profit)", value=True)
tp_pct = st.sidebar.slider("停利漲幅 (%)", 2.0, 50.0, 15.0, 0.5)

# 個股專屬持倉評估與即時現價覆蓋
st.sidebar.subheader("當前個股持倉與現價校正")
user_cost = st.sidebar.number_input("此股票平均買價 (若無持倉填 0)", value=0.0, step=0.1)
user_shares = st.sidebar.number_input("此股票持有股數 (若無持倉填 0)", value=0.0, step=100.0)
user_manual_price = st.sidebar.number_input(
    "手動覆蓋現價 (若填 0 則自動抓取行情)", 
    value=0.0, 
    step=0.01, 
    help="💡 若想與台新 Woojii 完全一致，可在此輸入券商 App 當下的現價！"
)

# ==================== 資料載入函數 (高穩定度) ====================
@st.cache_data
def load_stock_data(symbol, time_period):
    try:
        df = yf.download(symbol, period=time_period)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"下載股票資料發生錯誤: {e}")
        return pd.DataFrame()

@st.cache_data
def load_benchmark(bench_symbol, time_period):
    try:
        bench = yf.download(bench_symbol, period=time_period)
        if isinstance(bench.columns, pd.MultiIndex):
            bench.columns = bench.columns.get_level_values(0)
        return bench
    except:
        return pd.DataFrame()

data = load_stock_data(stock_symbol, period)
bench_data = load_benchmark(benchmark_symbol, period)

# ==================== 主畫面分頁籤設計 ====================
tab1, tab2 = st.tabs(["個股深度分析與持倉建議", "我的多檔持股總覽與資產診斷"])

with tab1:
    if not data.empty and len(data) > 0:
        st.subheader(f"📊 {stock_symbol.upper()} 技術分析與進出場訊號圖")

        window_len = min(5, len(data))
        long_window = min(20, len(data))
        
        data['SMA_5'] = data['Close'].rolling(window=window_len).mean()
        data['SMA_20'] = data['Close'].rolling(window=long_window).mean()
        
        data['BB_Middle'] = data['Close'].rolling(window=long_window).mean()
        bb_std = data['Close'].rolling(window=long_window).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        data['Position'] = np.where(data['SMA_5'] > data['SMA_20'], 1, 0)
        data['Signal'] = data['Position'].diff()

        buy_signals = data[data['Signal'] == 1]
        sell_signals = data[data['Signal'] == -1]

        trades = []
        in_trade = False
        entry_price = 0
        
        for i in range(len(data)):
            row = data.iloc[i]
            price = row['Close']
            if not in_trade:
                if row['Signal'] == 1:
                    entry_price = price
                    in_trade = True
            else:
                current_return = (price - entry_price) / entry_price
                hit_sl = enable_sl and (current_return <= -sl_pct / 100.0)
                hit_tp = enable_tp and (current_return >= tp_pct / 100.0)
                hit_signal_exit = row['Signal'] == -1
                
                if hit_sl or hit_tp or hit_signal_exit:
                    trades.append(current_return)
                    in_trade = False
                    
        if in_trade:
            trades.append((data['Close'].iloc[-1] - entry_price) / entry_price)
            
        trades_df = pd.Series(trades) if len(trades) > 0 else pd.Series()
        
        if not trades_df.empty:
            winning = trades_df[trades_df > 0]
            losing = trades_df[trades_df < 0]
            total_trades = len(trades_df)
            win_rate = len(winning) / total_trades if total_trades > 0 else 0
            loss_rate = 1 - win_rate
            avg_win = winning.mean() if len(winning) > 0 else 0
            avg_loss = abs(losing.mean()) if len(losing) > 0 else 0
            expected_value = (win_rate * avg_win) - (loss_rate * avg_loss)
        else:
            total_trades, win_rate, avg_win, avg_loss, expected_value = 0, 0, 0, 0, 0

        # 繪製圖表
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=("價格、均線、布林通道與進出場點", "成交量 (Volume)", "RSI 強弱指標"))
        
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='black', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_5'], name='短期均線', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='長期均線', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='布林上軌', line=dict(color='rgba(0,0,255,0.2)', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='布林下軌', line=dict(color='rgba(0,0,255,0.2)', dash='dash'), fill='tonexty'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='建議進場 (買)', marker=dict(symbol='triangle-up', size=14, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='建議出場 (賣)', marker=dict(symbol='triangle-down', size=14, color='red')), row=1, col=1)
        
        if not bench_data.empty and len(bench_data) > 0:
            bench_close = bench_data['Close'] / bench_data['Close'].iloc[0] * data['Close'].iloc[0]
            fig.add_trace(go.Scatter(x=bench_data.index, y=bench_close, name=f'大盤基準 ({benchmark_symbol})', line=dict(color='gray', dash='dot')), row=1, col=1)

        colors = ['red' if row['Close'] < row['Open'] else 'green' for index, row in data.iterrows()]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='成交量', marker_color=colors), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple', width=1)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)

        # 策略績效看板
        st.subheader("🎯 量化回測表現與每筆交易期望值 (EV)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("總回測交易次數", f"{total_trades} 次")
        with c2:
            st.metric("勝率 (Win Rate)", f"{win_rate * 100:.2f}%")
        with c3:
            st.metric("平均賺賠比", f"賺 {avg_win*100:.2f}% / 賠 {avg_loss*100:.2f}%")
        with c4:
            st.metric("每筆交易期望值 (EV)", f"{expected_value * 100:.2f}%", 
                      delta="正向優勢" if expected_value > 0 else "負向劣勢")

        # 個人持倉績效與風控評估
        st.markdown("---")
        st.subheader(f"🛡️ {stock_symbol.upper()} 個人持倉分析與進出場建議")

        market_latest_close = float(data['Close'].iloc[-1])
        effective_close = user_manual_price if user_manual_price > 0 else market_latest_close

        if user_cost > 0 and user_shares > 0:
            buy_val = user_cost * user_shares
            sell_val = effective_close * user_shares
            
            is_etf = stock_symbol.strip().upper().startswith("0")
            tax_rate = 0.001 if is_etf else 0.003
            
            if include_fees:
                buy_fee = max(20.0, buy_val * 0.001425 * (broker_discount / 10.0))
                sell_fee = max(20.0, sell_val * 0.001425 * (broker_discount / 10.0))
                sell_tax = sell_val * tax_rate
                total_cost = buy_val + buy_fee
                net_sell_val = sell_val - sell_fee - sell_tax
                pnl_amount = net_sell_val - total_cost
                my_return = (pnl_amount / total_cost) * 100
            else:
                total_cost = buy_val
                pnl_amount = sell_val - buy_val
                my_return = (effective_close - user_cost) / user_cost * 100

            my_sl_price = user_cost * (1 - sl_pct / 100)
            my_tp_price = user_cost * (1 + tp_pct / 100)

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("持倉總市值 / 總成本", f"{sell_val:,.0f} / {total_cost:,.0f}")
            with mc2:
                label_suffix = " (含手續費/稅)" if include_fees else ""
                st.metric(f"未實現損益{label_suffix}", f"{pnl_amount:+,.0f} 元", delta=f"{my_return:.2f}%", delta_color="normal" if pnl_amount >= 0 else "inverse")
            with mc3:
                st.metric(f"依成本停損價 (-{sl_pct}%)", f"{my_sl_price:.2f}")
            with mc4:
                st.metric(f"依成本停利價 (+{tp_pct}%)", f"{my_tp_price:.2f}")
            
            if effective_close <= my_sl_price:
                st.error(f"🚨 【當下出場警示】目前參考價 ({effective_close:.2f}) 已跌破您的停損價 ({my_sl_price:.2f})！建議果斷執行紀律停損出場。")
            elif effective_close >= my_tp_price:
                st.success(f"🎯 【當下出場警示】目前參考價 ({effective_close:.2f}) 已達到您的停利價 ({my_tp_price:.2f})！建議考慮獲利了結或分批減碼。")
            else:
                buffer_amt = effective_close - my_sl_price
                st.info(f"🛡️ 【續抱建議】目前持倉安全運作中。距離停損價還有 {buffer_amt:.2f} 元的緩衝空間，可繼續續抱觀望。")
        else:
            st.info("💡 提示：若您目前持有這檔股票，請在左側側邊欄輸入您的「平均買價」與「持有股數」，這裡將即時為您分析該檔股票的未實現損益！")

        with st.expander("🔍 檢視詳細歷史價格與變動明細表"):
            detail_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(30).copy()
            detail_df['價格變動(元)'] = detail_df['Close'].diff()
            detail_df['漲跌幅(%)'] = detail_df['Close'].pct_change() * 100
            detail_df = detail_df.sort_index(ascending=False)
            st.dataframe(detail_df.style.format({
                'Open': '{:.2f}', 'High': '{:.2f}', 'Low': '{:.2f}', 'Close': '{:.2f}',
                'Volume': '{:,.0f}', '價格變動(元)': '{:+.2f}', '漲跌幅(%)': '{:+.2f}%'
            }), use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 當下技術面綜合操作指引")
        latest_sma5 = float(data['SMA_5'].iloc[-1])
        latest_sma20 = float(data['SMA_20'].iloc[-1])
        latest_rsi = float(data['RSI'].iloc[-1])

        if latest_sma5 > latest_sma20:
            if latest_rsi > 70:
                box_func = st.warning
                action_title, action_desc = "多頭格局但短線過熱（建議觀望 / 逢高分批減碼）", "短期均線大於長期均線，但 RSI 進入超買區，短線有回檔壓力。"
            else:
                box_func = st.success
                action_title, action_desc = "建議偏多操作 / 可考慮進場續抱", "短期均線大於長期均線且 RSI 處於健康區間，符合多頭趨勢。"
        else:
            if latest_rsi < 30:
                box_func = st.info
                action_title, action_desc = "空頭排列但跌至超賣區（醞釀反彈 / 可輕倉試單）", "均線雖屬空頭，但 RSI 已跌破 30 超賣區，可能出現跌深反彈。"
            else:
                box_func = st.error
                action_title, action_desc = "建議空手觀望 / 暫時避險", "短期均線小於長期均線，趨勢向下，建議耐心等待黃金交叉。"

        box_func(f"### {action_title}\n{action_desc}")
    else:
        st.error("找不到該股票代號的歷史資料，請確認代號是否正確（台股請加 .TW，例如 0050.TW）。")

with tab2:
    st.subheader("📋 我的多檔持股總覽與資產配置診斷")
    st.markdown("在此新增您名下的各個持股部位，App 將自動計算含手續費與稅費的總市值與未實現損益！")

    with st.form("add_holding_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            new_sym = st.text_input("股票代號 (例如 2330.TW)", value="2330.TW")
        with col_f2:
            new_shares = st.number_input("持有股數", value=1000.0, step=100.0)
        with col_f3:
            new_cost = st.number_input("平均買價", value=500.0, step=0.5)
        
        submitted = st.form_submit_button("➕ 新增至我的持股清單")
        if submitted:
            st.session_state.portfolio.append({
                "symbol": new_sym.upper(),
                "shares": new_shares,
                "cost": new_cost
            })
            st.success(f"成功新增 {new_sym.upper()}！")

    if len(st.session_state.portfolio) > 0:
        st.markdown("### 目前持股明細")
        portfolio_data = []
        total_market_value = 0
        total_cost_value = 0

        for idx, item in enumerate(st.session_state.portfolio):
            sym = item['symbol']
            shs = item['shares']
            cst = item['cost']
            
            try:
                t_df = load_stock_data(sym, "1mo")
                curr_p = float(t_df['Close'].iloc[-1]) if not t_df.empty else cst
            except:
                curr_p = cst

            b_val = cst * shs
            s_val = curr_p * shs
            is_etf = sym.strip().upper().startswith("0")
            tax_r = 0.001 if is_etf else 0.003

            if include_fees:
                b_fee = max(20.0, b_val * 0.001425 * (broker_discount / 10.0))
                s_fee = max(20.0, s_val * 0.001425 * (broker_discount / 10.0))
                s_tax = s_val * tax_r
                item_cost = b_val + b_fee
                item_net_val = s_val - s_fee - s_tax
                pnl = item_net_val - item_cost
                pnl_pct = (pnl / item_cost * 100) if item_cost > 0 else 0
            else:
                item_cost = b_val
                item_net_val = s_val
                pnl = s_val - b_val
                pnl_pct = ((curr_p - cst) / cst * 100) if cst > 0 else 0

            total_market_value += item_net_val
            total_cost_value += item_cost

            portfolio_data.append({
                "代號": sym,
                "股數": shs,
                "平均成本": cst,
                "現價": round(curr_p, 2),
                "含稅市值(元)": round(item_net_val, 2),
                "未實現損益(元)": round(pnl, 2),
                "報酬率(%)": f"{pnl_pct:.2f}%"
            })

        pf_df = pd.DataFrame(portfolio_data)
        st.dataframe(pf_df, use_container_width=True)

        total_pnl = total_market_value - total_cost_value
        total_pnl_pct = (total_pnl / total_cost_value * 100) if total_cost_value > 0 else 0

        st.markdown("---")
        st.subheader("💰 整體資產總結 (含交易成本)")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.metric("總投資成本 (含買進手續費)", f"{total_cost_value:,.2f} 元")
        with pc2:
            st.metric("目前總淨市值 (扣除賣出費稅)", f"{total_market_value:,.2f} 元")
        with pc3:
            st.metric("總未實現損益", f"{total_pnl:+,.2f} 元", delta=f"{total_pnl_pct:.2f}%", delta_color="normal" if total_pnl >= 0 else "inverse")

        if st.button("🗑️ 清空所有持股紀錄"):
            st.session_state.portfolio = []
            st.rerun()
    else:
        st.info("目前尚無持股紀錄，請透過上方表格新增您的股票部位。")