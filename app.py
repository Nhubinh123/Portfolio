
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import quantstats as qs
import requests
import re
from scipy.stats import norm
from datetime import datetime, timedelta
from vnstock import *
import logging
import warnings
from plotly.subplots import make_subplots
from scipy import stats
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman, BlackLittermanModel
from sklearn.linear_model import LinearRegression
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
import warnings
logging.getLogger('vnstock').setLevel(logging.ERROR)
logging.getLogger('vnstock.common.data.data_explorer').setLevel(logging.CRITICAL)
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore")
# Đặt cấu hình trang ngay đầu tiên
st.set_page_config(page_title="Tối Ưu Danh Mục Đầu Tư", layout="wide")
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 95%;
        padding-left: 1rem;
        padding-right: 1rem;}
    </style>
    """,
    unsafe_allow_html=True)
# Danh sách mã cổ phiếu
symbols = ["MBB", "CTG", "TCB", "FPT", "CMG", "KSB", "VCG", "HHV", 
          "HPG", "HSG", "NKG", "PLX", "PVT", "MSN", "MWG" ] 
symbols1 = symbols + ["VNINDEX"]
# Số lượng cổ phiếu lưu hành
shares_outstanding = {
    "MBB": 6102272659, "CTG": 5369991748, "TCB": 7064851739, "FPT": 1471069183, "CMG": 211300807,
    "KSB": 114443703, "VCG": 598593458, "HHV": 432255528, "HPG": 6396250200, "HSG": 620982309, 
    "NKG": 447570881, "PLX": 1270592235,"PVT": 356012638, "MSN": 1438351617, "MWG": 120218540 }
company_names = {
    "MBB": "Ngân hàng TMCP Quân Đội (MB Bank)",
    "CTG": "Ngân hàng TMCP Công Thương Việt Nam (VietinBank)",
    "TCB": "Ngân hàng TMCP Kỹ Thương Việt Nam (Techcombank)",
    "FPT": "Tập đoàn FPT",
    "CMG": "CTCP Tập đoàn Công nghệ CMC",
    "KSB": "CTCP Khoáng sản và Xây dựng Bình Dương",
    "VCG": "Tổng CTCP Xuất nhập khẩu và Xây dựng Việt Nam",
    "HHV": "CTCP Đầu tư Hạ tầng Giao thông Đèo Cả",
    "HPG": "CTCP Tập đoàn Hòa Phát",
    "HSG": "CTCP Tập đoàn Hoa Sen",
    "NKG": "CTCP Thép Nam Kim",
    "PLX": "Tập đoàn Xăng dầu Việt Nam (Petrolimex)",
    "PVT": "Tổng CTCP Vận tải Dầu khí (PV Trans)",
    "MSN": "CTCP Tập đoàn Masan",
    "MWG": "CTCP Đầu tư Thế Giới Di Động" }
# Hàm lấy dữ liệu
@st.cache_data
def financial_ratios(symbol, file_path="financial_ratios.csv"):
    try:
        df_ratio_all = pd.read_csv(file_path)
        df_ratio = df_ratio_all[df_ratio_all["ticker"] == symbol]
        if not df_ratio.empty:
            return df_ratio, []
    except FileNotFoundError:
        df_ratio_all = pd.DataFrame()
    errors = []
    try:
        finance = Finance(symbol=symbol)
        df_ratio = finance.ratio(period="year", lang="vi")
        if df_ratio is not None and not df_ratio.empty:
            df_ratio.reset_index(inplace=True)
            df_ratio.rename(columns={'period': 'Year'}, inplace=True)
            df_ratio["ticker"] = symbol
            df_ratio_all = pd.concat([df_ratio_all, df_ratio], ignore_index=True)
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            df_ratio_all.to_csv(file_path, index=False, encoding="utf-8-sig")
            return df_ratio, []
        else:
            errors.append(f"Không có dữ liệu tỷ số tài chính cho {symbol}.")
            return pd.DataFrame(), errors
    except Exception as e:
        errors.append(f"Lỗi khi lấy dữ liệu tỷ số tài chính cho {symbol}: {str(e)}")
        return pd.DataFrame(), errors

@st.cache_data
def income_all(symbol, file_path="income_all.csv"):
    try:
        df_income_all = pd.read_csv(file_path)
        df_income = df_income_all[df_income_all["ticker"] == symbol]
        if not df_income.empty:
            return df_income, []
    except FileNotFoundError:
        df_income_all = pd.DataFrame()
    errors = []
    try:
        finance = Finance(symbol=symbol)
        df_income = finance.income_statement(period="year", lang="vi")
        if df_income is not None and not df_income.empty:
            df_income.reset_index(inplace=True)
            df_income.rename(columns={'period': 'Year'}, inplace=True)
            selected_columns = ['Year', 'revenue', 'year_revenue_growth', 'post_tax_profit']
            df_income = df_income[selected_columns]
            df_income["ticker"] = symbol
            df_income_all = pd.concat([df_income_all, df_income], ignore_index=True)
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            df_income_all.to_csv(file_path, index=False, encoding="utf-8-sig")
            return df_income, []
        else:
            errors.append(f"Không có dữ liệu kết quả kinh doanh cho {symbol}.")
            return pd.DataFrame(), errors
    except Exception as e:
        errors.append(f"Lỗi khi lấy dữ liệu kết quả kinh doanh cho {symbol}: {str(e)}")
        return pd.DataFrame(), errors

@st.cache_data
def fetch_stock_data(symbols1, start_date="01/01/2020", save_dir="."):
    if isinstance(symbols1, str):
        symbols1 = [symbols1]
    url = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    dfs = {}
    errors = []
    for symbol in symbols1:
        output_file = os.path.join(save_dir, f"{symbol}.csv")
        try:
            df = pd.read_csv(output_file)
            df["Ngay"] = pd.to_datetime(df["Ngay"], errors="coerce")
            if not df.empty and "GiaDongCua" in df.columns:
                df_close = df[["Ngay", "GiaDongCua"]].copy()
                df_close.set_index("Ngay", inplace=True)
                dfs[symbol] = df_close.rename(columns={"GiaDongCua": symbol})
                continue
        except FileNotFoundError:
            pass
        params = {
            "Symbol": symbol,
            "StartDate": start_date,
            "EndDate": datetime.today().strftime("%d/%m/%Y"),
            "PageIndex": 1,
            "PageSize": 2000 }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Referer": f"https://cafef.vn/du-lieu/lich-su-giao-dich-{symbol.lower()}-1.chn",
            "Accept": "/" }
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data.get("Data") and data["Data"].get("Data"):
                df = pd.DataFrame(data["Data"]["Data"])
                df["Ngay"] = pd.to_datetime(df["Ngay"], format="%d/%m/%Y", errors="coerce")
                os.makedirs(save_dir, exist_ok=True)
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                df_close = df[["Ngay", "GiaDieuChinh"]].copy()
                df_close.set_index("Ngay", inplace=True)
                dfs[symbol] = df_close.rename(columns={"GiaDieuChinh": symbol})
            else:
                errors.append(f"Không có dữ liệu giá cho {symbol}.")
        except requests.exceptions.RequestException as e:
            errors.append(f"Lỗi khi tải dữ liệu giá {symbol}: {str(e)}")
        time.sleep(0.5)
    if dfs:
        df_close_all = pd.concat(dfs.values(), axis=1)
        df_close_all = df_close_all.sort_index(ascending=False)
        return df_close_all, errors
    return pd.DataFrame(), errors
# Hàm tiền xử lý dữ liệu
def process_financial_data(data):
    columns_to_keep = ["Year", "price_to_earning", "price_to_book", "roe", "roa", "earning_per_share", "ticker"]
    filtered_data = data[columns_to_keep]
    filtered_data["Year"] = filtered_data["Year"].astype(int)
    filtered_data = filtered_data[filtered_data["Year"].isin([2020, 2021, 2022, 2023, 2024])]
    industry_map = {
        "MBB": "Ngân hàng", "CTG": "Ngân hàng", "TCB": "Ngân hàng",
        "FPT": "Công nghệ viễn thông", "CMG": "Công nghệ viễn thông",
        "KSB": "Xây dựng & VLXD", "VCG": "Xây dựng & VLXD", "HHV": "Xây dựng & VLXD",
        "HPG": "Thép", "HSG": "Thép", "NKG": "Thép",
        "PLX": "Năng lượng & Dầu khí", "PVT": "Năng lượng & Dầu khí",
        "MSN": "Bán lẻ – Tiêu dùng", "MWG": "Bán lẻ – Tiêu dùng" }
    filtered_data["industry"] = filtered_data["ticker"].map(industry_map)
    return filtered_data
def process_income_data(dt):
    dt["Year"] = dt["Year"].astype(int)
    dt = dt[dt["Year"].isin([2020, 2021, 2022, 2023, 2024])]
    return dt
def fill_missing_values(stock_df):
    if stock_df.empty:
        return stock_df
    stock_df = stock_df.replace(0, np.nan)
    stock_df = stock_df.ffill().bfill()
    stock_df = stock_df.dropna(axis=1, how='all').dropna()
    return stock_df
# Load dữ liệu
@st.cache_data
def load_data(symbols, symbols1):
    errors = []
    df_ratios = pd.DataFrame()
    df_income = pd.DataFrame()
    for symbol in symbols:
        df_r, r_errors = financial_ratios(symbol)
        df_i, i_errors = income_all(symbol)
        if not df_r.empty:
            df_ratios = pd.concat([df_ratios, df_r], ignore_index=True)
        if not df_i.empty:
            df_income = pd.concat([df_income, df_i], ignore_index=True)
        errors.extend(r_errors + i_errors)
    data_finan = process_financial_data(df_ratios)
    df_incomes = process_income_data(df_income)
    data_financial = pd.merge(data_finan, df_incomes, on=["ticker", "Year"], how="inner")
    
    df_stock, stock_errors = fetch_stock_data(symbols1)
    errors.extend(stock_errors)
    data_stock = fill_missing_values(df_stock)
    data_stocks = data_stock.drop(columns=["VNINDEX"], errors="ignore")
    return data_financial, data_stock, data_stocks, errors
# Tải dữ liệu
data_financial, data_stock, data_stocks, errors = load_data(symbols, symbols1)

# Sidebar
st.sidebar.image("logo.png", width=300)
st.sidebar.title("TỐI ƯU VIỆC XÂY DỰNG DANH MỤC ĐẦU TƯ")
tickers = data_financial["ticker"].unique().tolist()
symbol = st.sidebar.selectbox("Chọn mã cổ phiếu:", tickers)
menu = st.sidebar.radio("MENU:", ["🏠Dashboard - Tài chính doanh nghiệp", "💼 Danh mục đầu tư"])
# Dashboard tổng quan
if menu == "🏠Dashboard - Tài chính doanh nghiệp":
    st.markdown(
        f"<h1 style='text-align: center;'> Dashboard - Tổng quan tài chính doanh nghiệp {symbol}</h1>", unsafe_allow_html=True )
    company_name = company_names.get(symbol, symbol)
    st.markdown(f"<h2 style='text-align: center;'>{company_name}</h2>", unsafe_allow_html=True )
    data_financial["Year"] = pd.to_numeric(data_financial["Year"], errors="coerce").astype("Int64")
    ticker_data = data_financial[data_financial["ticker"] == symbol]
    if ticker_data.empty:
        st.warning("Không có dữ liệu để hiển thị. Vui lòng thử lại sau!")
    else:
        st.markdown("""
            <style>
            [data-testid="stMetricLabel"] { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; }
            [data-testid="stMetricLabel"] p { font-size: 1.2rem !important; font-weight: bold !important; text-align: center !important; margin: 0 auto !important; }
            [data-testid="stMetricValue"] { text-align: center !important; font-size: 1.4srem !important; }
            .center-table { display: block; text-align: center; }
            .custom-table {font-size: 24px !important;  max-width: 90% !important;  margin: 0 auto !important; border-collapse: collapse !important; background-color: #f9f9f9 !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;  }
            .custom-table th,
            .custom-table td { padding: 15px 10px !important;  border: 1px solid #ddd !important; }
            .custom-table th { background-color: #AEC6CF !important; font-weight: bold !important; }
            </style> """, unsafe_allow_html=True)
        st.subheader("Chỉ số tài chính nổi bật")
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
        industry = ticker_data["industry"].iloc[0] if not ticker_data["industry"].empty else "N/A"
        avg_roe = ticker_data["roe"].mean() if not ticker_data["roe"].empty else 0
        avg_roa = ticker_data["roa"].mean() if not ticker_data["roa"].empty else 0
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏭 Ngành", industry)
        with col2:
            st.metric("📈 ROE trung bình", f"{avg_roe:.3f}")
        with col3:
            st.metric("🔄 ROA trung bình", f"{avg_roa:.3f}")
        st.subheader("Thông tin tài chính qua các năm")
        selected_columns = ["Year", "roe", "roa", "price_to_book", "year_revenue_growth"]
        df_display = ticker_data[selected_columns].head(5)
        column_widths = { "Year": 100, "roe": 120, "roa": 120, "price_to_book": 130,"year_revenue_growth": 140 }
        html = '<table class="custom-table"><thead><tr>'
        for col in df_display.columns:
            width = column_widths.get(col, 100)
            html += f'<th style="width: {width}px">{col}</th>'
        html += '</tr></thead><tbody>'
        for row in df_display.values:
            html += '<tr>'
            for i, val in enumerate(row):
                col_name = df_display.columns[i]
                width = column_widths.get(col_name, 100)
                html += f'<td style="width: {width}px">{val}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        st.markdown(f'<div class="center-table" style="margin-bottom: 45px">{html}</div>', unsafe_allow_html=True)
        tick = dict(size=14, color="black") 
        font=dict(size=16, color="black")
        text = dict(size=16, color="black")
        # Hàng 1: Biểu đồ doanh thu và lợi nhuận hoạt động
        col1, col2 = st.columns(2)
        ticker_data = ticker_data.sort_values(by="Year", ascending=True)
        with col1:
            fig_revenue = px.line(ticker_data, x="Year", y="revenue", labels={"revenue": "Doanh thu (tỷ đồng)", "Year": "Năm"})
            fig_revenue.update_traces( mode="lines+markers+text",text=ticker_data["revenue"].apply(lambda x: f"{x:,.0f}"), textposition="top center" )
            fig_revenue.update_layout(xaxis_tickformat="d",xaxis_type="category", title="Biểu đồ doanh thu", title_x=0.4, title_font=dict(size=24), font=font,
                xaxis=dict(title_font=text, tickfont=tick), yaxis=dict(title_font=text, tickfont=tick))
            st.plotly_chart(fig_revenue, use_container_width=True)
        with col2:
            fig_profit = px.line(ticker_data, x="Year", y="post_tax_profit",labels={"post_tax_profit": "Lợi nhuận sau thuế (tỷ đồng)", "Year": "Năm"})
            fig_profit.update_traces(
                mode="lines+markers+text",
                text=ticker_data["post_tax_profit"].apply(lambda x: f"{x:,.0f}"),
                textposition="top center")
            fig_profit.update_layout(xaxis_tickformat="d",xaxis_type="category", title="Biểu đồ lợi nhuận", title_x=0.4, title_font=dict(size=24), font=font,
                xaxis=dict(title_font=text, tickfont=tick), yaxis=dict(title_font=text, tickfont=tick))
            st.plotly_chart(fig_profit, use_container_width=True)
        # Hàng 2: Biểu đồ P/E và EPS
        col3, col4 = st.columns(2)
        ticker_data = ticker_data.sort_values(by="Year", ascending=True)
        with col3:
            fig_pe = px.line(ticker_data, x="Year", y="price_to_earning",labels={"price_to_earning": "Giá trên lợi nhuận", "Year": "Năm"})
            fig_pe.update_traces( mode="lines+markers+text", text=ticker_data["price_to_earning"].round(2).astype(str), textposition="top center")
            fig_pe.update_layout(xaxis_tickformat="d",xaxis_type="category", title="Biểu đồ P/E", title_x=0.4,  title_font=dict(size=24),font=font,
                xaxis=dict(title_font=text, tickfont=tick), yaxis=dict(title_font=text, tickfont=tick))
            st.plotly_chart(fig_pe, use_container_width=True)
        with col4:
            fig_eps = px.line(ticker_data, x="Year", y="earning_per_share",labels={"earning_per_share": "Lợi nhuận trên mỗi cổ phiếu (VND/cp)", "Year": "Năm"})
            fig_eps.update_traces( mode="lines+markers+text",text=ticker_data["earning_per_share"].round(0).astype(str), textposition="top center" )
            fig_eps.update_layout(xaxis_tickformat="d",xaxis_type="category", title="Biểu đồ EPS", title_x=0.4,  title_font=dict(size=24),font=font,
                xaxis=dict(title_font=text, tickfont=tick), yaxis=dict(title_font=text, tickfont=tick))
            st.plotly_chart(fig_eps, use_container_width=True)
        # Hàng 3: Biểu đồ giá đóng cửa
        st.subheader("Biểu đồ tỷ suất sinh lợi của các cổ phiếu")
        selected_stock_data = data_stocks[[symbol]]  
        ret = selected_stock_data / selected_stock_data.iloc[0] 
        ret = ret.reset_index().melt(id_vars=["Ngay"], var_name="Mã cổ phiếu", value_name="Tỷ suất sinh lợi")
        ret = ret.rename(columns={"Ngay": "Ngày"})
        fig_returns = px.line( ret, x="Ngày", y="Tỷ suất sinh lợi", color="Mã cổ phiếu",
            labels={"Ngày": "Ngày", "Tỷ suất sinh lợi": "Tỷ suất sinh lợi"})
        st.plotly_chart(fig_returns, use_container_width=True)
        
elif menu == "💼 Danh mục đầu tư":
    with st.sidebar:
        portfolio_submenu = st.radio("Chọn mô hình", ["Mô hình Markowitz", "Mô hình Black-Litterman"], key="portfolio_submenu")
    if portfolio_submenu == "Mô hình Markowitz":
        st.header("Tối ưu danh mục đầu tư theo mô hình Markowitz")
        # Bước 1: Chuẩn bị dữ liệu từ tất cả cổ phiếu trong data_stocks
        try:
            all_tickers = list(data_stocks.columns) 
            if not all_tickers:
                st.error("Không có cổ phiếu nào trong dữ liệu để tối ưu!")
                st.stop()
            data_selected = data_stocks[all_tickers]  
            if data_selected.empty or len(data_selected) < 2:
                st.error("Dữ liệu giá cổ phiếu không đủ để tính toán!")
                st.stop()
            # Bước 2: Tối ưu danh mục bằng mô hình Markowitz
            returns = data_selected / data_selected.shift(1)
            logReturns = np.log(returns)
            ind_er = data_selected.resample('Y').last().pct_change().mean()
            cov_matrix = data_selected.pct_change().cov()
            ann_sd = data_selected.pct_change().std() * np.sqrt(252)
            num_assets = len(data_selected.columns)
            num_portfolios = 10000
            
            np.random.seed(42)
            results = np.zeros((3, num_portfolios))
            weights_record = []
            for portfolio in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights = weights/np.sum(weights)
                weights_record.append(weights)
                portfolio_return = np.sum(ind_er * weights)
                results[0, portfolio] = portfolio_return
                var = np.dot(weights.T, np.dot(cov_matrix, weights)) 
                ann_sd_portfolio = np.sqrt(var) * np.sqrt(252)  # Độ lệch chuẩn hàng năm
                results[1, portfolio] = ann_sd_portfolio 
                sharpe_ratio = (portfolio_return - 0.01) / ann_sd_portfolio   
                results[2, portfolio] = sharpe_ratio
               
            portfolios = pd.DataFrame({'Returns': results[0], 'Volatility': results[1], 'Sharpe': results[2]})
            
            # Danh mục tối ưu theo Sharpe Ratio (Sharpe ratio cao nhất)
            optimal_idx_sharpe = portfolios['Sharpe'].idxmax()
            optimal_risky_port_sharpe = portfolios.iloc[optimal_idx_sharpe]
            optimal_weights_sharpe = weights_record[optimal_idx_sharpe]
            # Danh mục có rủi ro thấp nhất (độ lệch chuẩn thấp nhất)
            min_vol_idx = portfolios['Volatility'].idxmin()
            min_vol_port = portfolios.iloc[min_vol_idx]
            optimal_weights_min_vol = weights_record[min_vol_idx]
            sharpe_ratio_min_vol = (min_vol_port['Returns'] - 0.01) / min_vol_port['Volatility']
            cleaned_weights_sharpe = dict(zip(all_tickers, optimal_weights_sharpe))
            cleaned_weights_sharpe = {k: round(v, 4) for k, v in cleaned_weights_sharpe.items() if v > 0.0001}
            cleaned_weights_min_vol = dict(zip(all_tickers, optimal_weights_min_vol))
            cleaned_weights_min_vol = {k: round(v, 4) for k, v in cleaned_weights_min_vol.items() if v > 0.0001}
        
            # Tính lại tỷ suất sinh lợi kỳ vọng và độ lệch chuẩn
            portfolio_expected_return_sharpe = optimal_risky_port_sharpe['Returns']
            portfolio_volatility_sharpe = optimal_risky_port_sharpe['Volatility']
            sharpe_ratio_sharpe = optimal_risky_port_sharpe['Sharpe']
        
            portfolio_expected_return_min_vol = min_vol_port['Returns']
            portfolio_volatility_min_vol = min_vol_port['Volatility']
        
            # Hiển thị kết quả Sharpe Ratio cao nhất
            st.subheader("Danh mục tối ưu - Sharpe Ratio cao nhất")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='font-size:22px; margin-top:60px; margin-bottom: 24px'><b>Sharpe Ratio:</b> {sharpe_ratio_sharpe * 100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Tỷ suất sinh lợi kỳ vọng:</b> {portfolio_expected_return_sharpe * 100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Độ lệch chuẩn danh mục:</b> {portfolio_volatility_sharpe * 100:.2f}%</div>", unsafe_allow_html=True)
            with col2:
                # Vẽ biểu đồ phân phối trọng số cổ phiếu cho danh mục tối ưu Sharpe Ratio
                weights_df_sharpe = pd.DataFrame({
                    'Stock': [k for k, v in cleaned_weights_sharpe.items()],
                    'Weight': [v * 100 for k, v in cleaned_weights_sharpe.items()] })
                fig_pie_sharpe = px.pie(weights_df_sharpe, values='Weight', names='Stock', color_discrete_sequence=px.colors.qualitative.Plotly)
                fig_pie_sharpe.update_traces(textinfo='percent+label', textposition='inside', showlegend=True)
                fig_pie_sharpe.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin=dict(t=50, b=50, l=50, r=50), height=600)
                st.plotly_chart(fig_pie_sharpe, use_container_width=True)
            # Hiển thị kết quả của danh mục có rủi ro thấp nhất
            st.subheader("Danh mục tối ưu - Rủi ro thấp nhất")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='font-size:22px; margin-top:60px; margin-bottom: 24px'><b>Sharpe Ratio:</b> {sharpe_ratio_min_vol * 100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Tỷ suất sinh lợi kỳ vọng:</b> {portfolio_expected_return_min_vol * 100:.2f}%</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Độ lệch chuẩn danh mục:</b> {portfolio_volatility_min_vol * 100:.2f}%</div>", unsafe_allow_html=True)
            with col2:
                # Vẽ biểu đồ phân phối trọng số cổ phiếu cho danh mục có rủi ro thấp nhất
                weights_df_min_vol = pd.DataFrame({
                    'Stock': [k for k, v in cleaned_weights_min_vol.items()],
                    'Weight': [v * 100 for k, v in cleaned_weights_min_vol.items()]  })
                fig_pie_min_vol = px.pie(weights_df_min_vol, values='Weight', names='Stock', color_discrete_sequence=px.colors.qualitative.Plotly)
                fig_pie_min_vol.update_traces(textinfo='percent+label', textposition='inside', showlegend=True)
                fig_pie_min_vol.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin=dict(t=50, b=50, l=50, r=50), height=600)
                st.plotly_chart(fig_pie_min_vol, use_container_width=True)
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu hoặc tính toán: {e}")

    elif portfolio_submenu == "Mô hình Black-Litterman":
        st.header("Tối ưu danh mục đầu tư theo mô hình Black-Litterman")
        # Bước 1: Nhập quan điểm cá nhân của nhà đầu tư
        st.markdown("<h4>Nhập quan điểm cá nhân của bạn về lợi suất kỳ vọng cùng với mức độ tin cậy dự đoán</h4>", unsafe_allow_html=True)
        all_tickers = list(data_stocks.columns)
        if not all_tickers:
            st.error("Không có cổ phiếu nào trong dữ liệu để nhập quan điểm!")
            st.stop()
        views = {}
        confidences = {}
        if "num_views" not in st.session_state:
            st.session_state.num_views = 5
        col_add, col_remove = st.columns([1, 1])
        with col_add:
            if st.button("Thêm quan điểm"):
                st.session_state.num_views += 1
        with col_remove:
            if st.session_state.num_views > 1 and st.button("Xóa quan điểm"):
                st.session_state.num_views -= 1
        chosen_tickers = []
        for i in range(st.session_state.num_views):
            col1, col2, col3 = st.columns([2, 1, 1])
            available_tickers = [ticker for ticker in all_tickers if ticker not in chosen_tickers]
            with col1:
                if not available_tickers:
                    st.selectbox(f"Chọn cổ phiếu {i+1}", ["Không còn cổ phiếu để chọn"], key=f"asset_view_{i}", disabled=True)
                    asset = None
                else:
                    asset = st.selectbox(f"Chọn cổ phiếu {i+1}", available_tickers, key=f"asset_view_{i}")
            with col2:
                value = st.number_input(f"Lợi suất kỳ vọng(%)", min_value=-100.0, max_value=100.0, value=0.0, step=0.1, key=f"view_value_{i}")
            with col3:
                confidence = st.number_input(f"Mức độ tin cậy(%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key=f"confidence_{i}")
            if asset and asset != "Không còn cổ phiếu để chọn":
                views[asset] = value / 100
                confidences[asset] = confidence / 100
                chosen_tickers.append(asset)
        if not views:
            st.warning("Bạn chưa nhập quan điểm cá nhân nào. Vui lòng nhập ít nhất một quan điểm để tiếp tục!")
            st.stop()
        tickers = list(views.keys())
        if st.button("Tiến hành tối ưu"):
            # Bước 2: Chuẩn bị dữ liệu và tham số cho mô hình Black-Litterman
            try: 
                market_caps_series = pd.Series(
                    {ticker: data_stocks.loc[data_stocks.index[0], ticker] * shares_outstanding[ticker] for ticker in tickers},index=tickers)
                total_market_cap = market_caps_series.sum()
                allocations = (market_caps_series / total_market_cap).to_dict() #Tính tỷ trọng dựa trên vốn hóa thị trường
                selected_tickers = tickers
                data_selected = data_stocks[selected_tickers] 
                if data_selected.empty or len(data_selected) < 2:
                    st.error("Dữ liệu giá cổ phiếu không đủ để tính toán!")
                    st.stop()
                data_selected = data_selected[(np.abs(stats.zscore(data_selected)) < 3).all(axis=1)]
                returns = data_selected.pct_change().dropna()
                mu = expected_returns.mean_historical_return(data_selected, frequency=252) # Lợi suất kỳ vọng lịch sử
                S = returns.cov()*252
                variances = pd.Series(np.diag(S), index=returns.columns)
                returns_variance = returns.var()
                if variances.max() > returns_variance.max() * 10:
                    S = S / 1000  # Điều chỉnh thử (có thể thay đổi hệ số)
                    variances = pd.Series(np.diag(S), index=returns.columns)
                data_vnindex = data_stock["VNINDEX"]
                delta = black_litterman.market_implied_risk_aversion(data_vnindex) 
                if delta < 2 or delta > 4: delta = 2.5
                market_prior = black_litterman.market_implied_prior_returns(market_caps_series, delta, S*1000)# Lợi suất kỳ vọng cân bằng của thị trường
                # Lọc lại quan điểm cá nhân
                views_selected = {ticker: views[ticker] for ticker in views if ticker in selected_tickers}
                confidences_selected = {ticker: confidences[ticker] for ticker in confidences if ticker in selected_tickers}
                if not views_selected:
                    st.error("Không có quan điểm nào áp dụng cho các cổ phiếu đã chọn để tối ưu. Vui lòng chọn các cổ phiếu có trong quan điểm của bạn!")
                    st.stop()
                # Bước 3: Áp dụng mô hình Black-Litterman và hiển thị kết quả
                st.markdown("<h4>Danh mục tối ưu được tính toán bằng mô hình Black-Litterman</h4>", unsafe_allow_html=True)
                Q = pd.Series(views_selected)
                P = pd.DataFrame(0.0, index=Q.index, columns=data_selected.columns)
                for asset in views_selected:
                    P.loc[asset, asset] = 1.0
                Omega = np.diag([1 - confidences_selected.get(asset, 0.65) for asset in Q.index])
                bl = BlackLittermanModel(S, pi=market_prior, Q=Q, P=P, omega=Omega)
                weights = bl.bl_weights()
                adjusted_returns = bl.bl_returns()
                # Hiển thị danh mục tối ưu
                st.markdown("""
                <style>
                [data-testid="stMetricLabel"] { display: flex !important; justify-content: center !important; align-items: center !important; width: 100% !important; }
                [data-testid="stMetricLabel"] p { font-size: 1.1rem !important; font-weight: bold !important; text-align: center !important; margin: 0 auto !important; }
                [data-testid="stMetricValue"] { text-align: center !important; font-size: 1.3srem !important; }
                .center-table { display: block; text-align: center; }
                .custom-table {font-size: 22px !important;  max-width: 90% !important;  margin: 0 auto !important; border-collapse: collapse !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;  }
                .custom-table th,
                .custom-table td { padding: 10px 7px !important;  border: 1px solid #ddd !important; }
                .custom-table th { background-color: #AEC6CF !important; font-weight: bold !important; }
                </style> """, unsafe_allow_html=True)
                optimized_data = []
                for ticker in selected_tickers:
                    adjusted_return = adjusted_returns[ticker] * 100
                    weight = weights[ticker] * 100
                    optimized_data.append({
                        "Ticker": ticker,
                        "Name": company_names.get(ticker, ticker),
                        "Adjusted Return": f"{adjusted_return:.2f}%",
                        "Allocation": f"{weight:.1f}%" })
                optimized_df = pd.DataFrame(optimized_data)
                column_widths_opt = {"Ticker": 100, "Name": 300, "Adjusted Return": 180, "Allocation": 150 }
                html = '<table class="custom-table"><thead><tr>'
                for col in optimized_df.columns:
                    w = column_widths_opt.get(col, 100)
                    html += f'<th style="width:{w}px">{col}</th>'
                html += '</tr></thead><tbody>'
                for _, row in optimized_df.iterrows():
                    html += '<tr>'
                    for col in optimized_df.columns:
                        w = column_widths_opt.get(col, 100)
                        html += f'<td style="width:{w}px">{row[col]}</td>'
                    html += '</tr>'
                html += '</tbody></table>'
                st.markdown( f"<div class='center-table' style='margin-bottom:45px'>{html}</div>",  unsafe_allow_html=True)
                # Phân tích rủi ro của danh mục tối ưu
                portfolio_volatility_daily = np.sqrt(np.dot(pd.Series(weights).T, np.dot(S, pd.Series(weights))))
                portfolio_volatility = portfolio_volatility_daily * np.sqrt(252)
                portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
                mean_return = portfolio_returns.mean()
                std_return = portfolio_returns.std()
                # Thông tin danh mục cổ phiếu
                st.subheader("Thông tin danh mục cổ phiếu")
                risk_free_rate = 0.02
                portfolio_expected_return = np.sum(adjusted_returns * pd.Series(weights))
                market_returns = data_stock["VNINDEX"].pct_change().dropna()
                market_returns = market_returns[market_returns.index.isin(portfolio_returns.index)]
                portfolio_returns = portfolio_returns[portfolio_returns.index.isin(market_returns.index)]
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance
                market_return = market_returns.mean() * 252
                sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<div style='font-size:22px; margin-top:40px; margin-bottom: 24px'><b>Sharpe Ratio:</b> {sharpe_ratio:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Tỷ suất sinh lợi kỳ vọng:</b> {portfolio_expected_return * 100:.2f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Độ lệch chuẩn danh mục:</b> {portfolio_volatility * 100:.2f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:22px; margin-bottom: 24px'><b>Beta:</b> {beta:.2f}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div style='font-size:20px; font-weight:bold; '>"  "Tỷ trọng danh mục tối ưu" "</div>", unsafe_allow_html=True)
                    weights_df = pd.Series(weights)
                    fig_pie = px.pie(weights_df, values=weights_df.values * 100, names=weights_df.index )
                    st.plotly_chart(fig_pie, use_container_width=True)
                st.subheader("Chi tiết danh mục cổ phiếu")
                industry_map = {
                    "MBB": "Ngân hàng", "CTG": "Ngân hàng", "TCB": "Ngân hàng",
                    "FPT": "Công nghệ viễn thông", "CMG": "Công nghệ viễn thông",
                    "KSB": "Xây dựng & VLXD", "VCG": "Xây dựng & VLXD", "HHV": "Xây dựng & VLXD",
                    "HPG": "Thép", "HSG": "Thép", "NKG": "Thép",
                    "PLX": "Năng lượng & Dầu khí", "PVT": "Năng lượng & Dầu khí",
                    "MSN": "Bán lẻ – Tiêu dùng", "MWG": "Bán lẻ – Tiêu dùng" }
                table_data = []
                for idx, (ticker, weight) in enumerate(weights.items(), 1):
                    industry = industry_map.get(ticker, "Không xác định")
                    weight_percent = weight * 100
                    previous_weight = 0
                    weight_change = weight_percent - previous_weight
                    current_price = data_stocks[ticker].iloc[0] 
                    estimated_price = current_price * 1.2
                    table_data.append({
                        "STT": idx,
                        "Mã CP": ticker,
                        "Ngành": industry,
                        "Tỷ trọng danh mục cổ phiếu": f"{weight_percent:.2f}%",
                        "Thay đổi tỷ trọng tối ưu": f"{weight_change:.2f}%",
                        "Giá CP": f"{current_price:,.0f}",
                        "Định giá (VND)": f"{estimated_price:,.0f}" })
                html_columns = [ "STT", "Mã CP", "Ngành", "Tỷ trọng danh mục cổ phiếu", "Thay đổi tỷ trọng tối ưu", "Giá CP", "Định giá (VND)"]
                table_df = pd.DataFrame(table_data, columns=html_columns)
                column_widths = {
                    "STT": 50, "Mã CP": 80, "Ngành": 150, "Tỷ trọng danh mục cổ phiếu": 250,
                    "Thay đổi tỷ trọng tối ưu": 220, "Giá CP": 110, "Định giá (VND)": 160, }
                html = '<table class="custom-table"><thead><tr>'
                for col in table_df.columns:
                    width = column_widths.get(col, 100)
                    html += f'<th style="width: {width}px">{col}</th>'
                html += '</tr></thead><tbody>'
                for row in table_df.values:
                    html += '<tr>'
                    for i, val in enumerate(row):
                        col_name = table_df.columns[i]
                        width = column_widths.get(col_name, 100)
                        html += f'<td style="width: {width}px">{val}</td>'
                    html += '</tr>'
                html += '</tbody></table>'
                st.markdown("""
                <style>
                    .custom-table {
                        font-size: 18px !important; max-width: 95% !important;  margin: 0 auto !important;
                        border-collapse: collapse !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; }
                    .custom-table th,
                    .custom-table td { padding: 10px 8px !important; border: 1px solid #ddd !important; text-align: center; }
                    .custom-table th { background-color: #AEC6CF !important; font-weight: bold !important; }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f'<div class="center-table" style="margin-bottom: 45px">{html}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Lỗi khi xử lý dữ liệu hoặc tính toán: {e}")