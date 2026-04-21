"""
Financial Health Analyzer - ACC102 Mini Assignment-Track 4
Student：Yuping Jiao
ID：2472725

A tool for analyzing company financial health using real-time data.
I chose Alpha Vantage API because it's free, reliable, and accessible worldwide.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import os
import time
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ============================================================================
# CONSTANTS - I put these at the top so they're easy to find and modify
# ============================================================================

# Scoring weights (I chose these based on what seems most important)
# I weighted leverage and profitability higher because they seem more important
# for long-term health. But honestly, this is somewhat subjective.
LIQUIDITY_WEIGHT = 0.30
LEVERAGE_WEIGHT = 0.35
PROFITABILITY_WEIGHT = 0.35

# Scoring thresholds for different methods
# I found these from textbooks and online resources
# They might not be perfect, but they're a reasonable starting point
SCORING_THRESHOLDS = {
    'Standard': {
        'current_ratio': [2.0, 1.5, 1.0, 0.8],
        'debt_to_assets': [0.30, 0.50, 0.70, 0.85],
        'roe': [0.20, 0.15, 0.10, 0.05]
    },
    'Conservative': {
        'current_ratio': [2.5, 2.0, 1.5, 1.0],
        'debt_to_assets': [0.25, 0.40, 0.55, 0.70],
        'roe': [0.25, 0.20, 0.15, 0.10]
    },
    'Aggressive': {
        'current_ratio': [1.5, 1.2, 0.9, 0.7],
        'debt_to_assets': [0.40, 0.55, 0.70, 0.85],
        'roe': [0.15, 0.10, 0.07, 0.05]
    }
}

# Industry benchmarks from Damodaran datasets (NYU Stern)
# I found these really helpful for context - different industries have very
# different "normal" values for financial ratios
INDUSTRY_BENCHMARKS = {
    'Technology': {'current_ratio': 2.5, 'debt_to_assets': 0.25, 'roe': 0.25},
    'Finance': {'current_ratio': 1.0, 'debt_to_assets': 0.85, 'roe': 0.12},
    'Healthcare': {'current_ratio': 1.8, 'debt_to_assets': 0.35, 'roe': 0.18},
    'Retail': {'current_ratio': 1.2, 'debt_to_assets': 0.45, 'roe': 0.20},
    'Energy': {'current_ratio': 1.3, 'debt_to_assets': 0.40, 'roe': 0.15},
    'Consumer': {'current_ratio': 1.5, 'debt_to_assets': 0.40, 'roe': 0.22}
}

# Company list
# I picked these because they're well-known and from different industries
COMPANIES = {
    'AAPL': 'Apple Inc. (Technology)',
    'MSFT': 'Microsoft Corp. (Technology)',
    'GOOGL': 'Alphabet Inc. (Technology)',
    'JPM': 'JPMorgan Chase (Finance)',
    'JNJ': 'Johnson & Johnson (Healthcare)',
    'WMT': 'Walmart Inc. (Retail)',
    'XOM': 'Exxon Mobil (Energy)',
    'PG': 'Procter & Gamble (Consumer)'
}

# API settings
# Alpha Vantage free tier allows 5 calls per minute
# I learned this the hard way - got blocked a few times!
API_RATE_LIMIT_DELAY = 12

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def api_delay() -> None:
    """
    Wait to avoid hitting API rate limits.
    
    Alpha Vantage free tier allows 5 calls per minute, so I need to wait
    12 seconds between requests to be safe.
    
    I learned this the hard way - got blocked a few times before adding this!
    It makes the app slower, but it's necessary.
    
    Returns:
        None
    """
    time.sleep(API_RATE_LIMIT_DELAY)


def get_industry_benchmark(industry: Optional[str]) -> Dict[str, float]:
    """
    Find the appropriate industry benchmark for a company.
    
    Args:
        industry: The industry name from Alpha Vantage (can be None)
        
    Returns:
        Dictionary with benchmark ratios:
        - current_ratio: Industry average current ratio
        - debt_to_assets: Industry average debt-to-assets ratio
        - roe: Industry average ROE
        
    Note:
        If industry doesn't match any known category, I default to Technology.
        This isn't perfect but it's a reasonable fallback.
        
    Example:
        >>> benchmark = get_industry_benchmark("Technology")
        >>> print(benchmark['current_ratio'])
        2.5
    """
    if not industry:
        return INDUSTRY_BENCHMARKS['Technology']
    
    industry_lower = industry.lower()
    for key in INDUSTRY_BENCHMARKS.keys():
        if key.lower() in industry_lower:
            return INDUSTRY_BENCHMARKS[key]
    
    # Default to Technology if no match
    return INDUSTRY_BENCHMARKS['Technology']


def format_percentage(value: float) -> str:
    """
    Format a decimal as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.15 for 15%)
        
    Returns:
        Formatted string (e.g., "15.0%")
        
    Example:
        >>> format_percentage(0.156)
        '15.6%'
    """
    return f"{value * 100:.1f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.
    
    Args:
        value: The number to format
        decimals: Number of decimal places (default: 2)
        
    Returns:
        Formatted string
        
    Example:
        >>> format_number(3.14159, 2)
        '3.14'
    """
    return f"{value:.{decimals}f}"


def normalize_for_radar(value: float, max_value: float, invert: bool = False) -> float:
    """
    Normalize a value to 0-1 range for radar chart.
    
    Args:
        value: The value to normalize
        max_value: The maximum expected value
        invert: Whether to invert (for metrics where lower is better)
        
    Returns:
        Normalized value between 0 and 1
        
    Note:
        I use this for the radar chart so all metrics are on the same scale.
        For metrics like debt-to-assets where lower is better, I invert them.
    """
    if invert:
        return max(0, min(1, 1 - (value / max_value)))
    else:
        return max(0, min(1, value / max_value))

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_company_overview(symbol: str, api_key: str) -> Dict:
    """
    Fetch company overview from Alpha Vantage.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        api_key: Alpha Vantage API key
        
    Returns:
        Dictionary with:
        - name: Company name
        - industry: Industry classification
        
    Raises:
        Exception: If API request fails (caught and handled)
        
    Note:
        I cache this for 1 hour to avoid unnecessary API calls.
    """
    base_url = "https://www.alphavantage.co/query"
    
    params = {
        'function': 'OVERVIEW',
        'symbol': symbol,
        'apikey': api_key
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        data = response.json()
        
        return {
            'name': data.get('Name', symbol),
            'industry': data.get('Industry', 'Unknown')
        }
    except Exception as e:
        st.warning(f"Could not fetch overview for {symbol}: {str(e)}")
        return {'name': symbol, 'industry': 'Unknown'}


@st.cache_data(ttl=3600)
def fetch_financial_statements(symbol: str, api_key: str) -> Optional[Dict]:
    """
    Fetch financial statements from Alpha Vantage.
    
    This function fetches both income statement and balance sheet,
    then matches them by year. I had to add delays between API calls
    to avoid rate limits.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        api_key: Alpha Vantage API key
        
    Returns:
        Dictionary with:
        - symbol: Stock ticker
        - company_name: Company name
        - industry: Industry classification
        - years: List of yearly financial data
        
        Returns None if fetching fails
        
    Note:
        I had to add delays between API calls to avoid rate limits.
        This makes the app slower (about 12 seconds per company),
        but it's necessary for the free tier.
        
    Example:
        >>> data = fetch_financial_statements('AAPL', 'your_api_key')
        >>> print(data['company_name'])
        'Apple Inc'
    """
    base_url = "https://www.alphavantage.co/query"
    
    data = {
        'symbol': symbol,
        'company_name': None,
        'industry': None,
        'years': []
    }
    
    try:
        # Step 1: Get company overview
        overview = fetch_company_overview(symbol, api_key)
        data['company_name'] = overview['name']
        data['industry'] = overview['industry']
        
        # Wait to avoid rate limit
        api_delay()
        
        # Step 2: Get income statement
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol,
            'apikey': api_key
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        income_data = response.json()
        
        # Wait again
        api_delay()
        
        # Step 3: Get balance sheet
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol,
            'apikey': api_key
        }
        
        response = requests.get(base_url, params=params, timeout=10)
        balance_data = response.json()
        
        # Step 4: Parse and match data
        if 'annualReports' in income_data and 'annualReports' in balance_data:
            income_reports = income_data['annualReports']
            balance_reports = balance_data['annualReports']
            
            # Get last 3 years
            for income in income_reports[:3]:
                year = income.get('fiscalDateEnding', '')[:4]
                
                # Find matching balance sheet
                balance = None
                for b in balance_reports:
                    if b.get('fiscalDateEnding', '')[:4] == year:
                        balance = b
                        break
                
                if balance:
                    try:
                        year_data = {
                            'year': int(year),
                            'revenue': float(income.get('totalRevenue', 0)),
                            'net_income': float(income.get('netIncome', 0)),
                            'total_assets': float(balance.get('totalAssets', 0)),
                            'current_assets': float(balance.get('totalCurrentAssets', 0)),
                            'current_liabilities': float(balance.get('totalCurrentLiabilities', 0)),
                            'total_equity': float(balance.get('totalShareholderEquity', 0)),
                        }
                        
                        # Calculate liabilities using accounting equation
                        # Assets = Liabilities + Equity
                        if year_data['total_assets'] and year_data['total_equity']:
                            year_data['total_liabilities'] = (
                                year_data['total_assets'] - year_data['total_equity']
                            )
                        else:
                            year_data['total_liabilities'] = 0
                        
                        data['years'].append(year_data)
                        
                    except (ValueError, TypeError) as e:
                        # Skip this year if data is incomplete
                        continue
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


def calculate_ratios(data: Dict) -> List[Dict]:
    """
    Calculate financial ratios from raw financial data.
    
    I referenced these formulas from my finance textbook.
    Some ratios might be None if data is missing.
    
    Args:
        data: Dictionary with company data and yearly financials
        
    Returns:
        List of dictionaries with calculated ratios for each year
        
    Note:
        I calculate 8 key ratios:
        - Liquidity: Current Ratio
        - Leverage: Debt-to-Assets, Debt-to-Equity
        - Profitability: Net Margin, ROA, ROE
        - Efficiency: Asset Turnover
        
    Example:
        >>> ratios = calculate_ratios(company_data)
        >>> print(ratios[0]['current_ratio'])
        1.52
    """
    if not data or not data.get('years'):
        return []
    
    ratios_list = []
    
    for year_data in data['years']:
        try:
            ratios = {
                'symbol': data['symbol'],
                'company': data['company_name'],
                'industry': data['industry'],
                'year': year_data['year'],
            }
            
            # Liquidity ratios
            # Current Ratio = Current Assets / Current Liabilities
            if year_data['current_liabilities'] > 0:
                ratios['current_ratio'] = (
                    year_data['current_assets'] / year_data['current_liabilities']
                )
            else:
                ratios['current_ratio'] = None
            
            # Leverage ratios
            # Debt-to-Assets = Total Liabilities / Total Assets
            if year_data['total_assets'] > 0:
                ratios['debt_to_assets'] = (
                    year_data['total_liabilities'] / year_data['total_assets']
                )
            else:
                ratios['debt_to_assets'] = None
            
            # Debt-to-Equity = Total Liabilities / Total Equity
            if year_data['total_equity'] > 0:
                ratios['debt_to_equity'] = (
                    year_data['total_liabilities'] / year_data['total_equity']
                )
            else:
                ratios['debt_to_equity'] = None
            
            # Profitability ratios
            # Net Margin = Net Income / Revenue
            if year_data['revenue'] > 0:
                ratios['net_margin'] = year_data['net_income'] / year_data['revenue']
            else:
                ratios['net_margin'] = None
            
            # ROA = Net Income / Total Assets
            if year_data['total_assets'] > 0:
                ratios['roa'] = year_data['net_income'] / year_data['total_assets']
            else:
                ratios['roa'] = None
            
            # ROE = Net Income / Total Equity
            if year_data['total_equity'] > 0:
                ratios['roe'] = year_data['net_income'] / year_data['total_equity']
            else:
                ratios['roe'] = None
            
            # Efficiency ratios
            # Asset Turnover = Revenue / Total Assets
            if year_data['total_assets'] > 0:
                ratios['asset_turnover'] = year_data['revenue'] / year_data['total_assets']
            else:
                ratios['asset_turnover'] = None
            
            ratios_list.append(ratios)
            
        except Exception as e:
            # Skip this year if calculation fails
            continue
    
    return ratios_list


def calculate_health_scores(ratios_list: List[Dict], method: str = 'Standard') -> List[Dict]:
    """
    Calculate health scores for each year of data.
    
    I created this scoring system based on general guidelines from textbooks.
    The thresholds might not be perfect, but they're a starting point.
    
    Args:
        ratios_list: List of dictionaries with financial ratios
        method: Scoring method - 'Standard', 'Conservative', or 'Aggressive'
        
    Returns:
        List of dictionaries with added health scores:
        - liquidity_score: 1-5 score for liquidity
        - liquidity_status: Text description
        - leverage_score: 1-5 score for leverage
        - leverage_status: Text description
        - profitability_score: 1-5 score for profitability
        - profitability_status: Text description
        - overall_score: Weighted average (0-5)
        
    Note:
        Score interpretation:
        - 5 = Excellent
        - 4 = Good
        - 3 = Adequate
        - 2 = Weak
        - 1 = Poor
        - 0 = N/A (missing data)
        
        I use different thresholds for different risk preferences:
        - Conservative: Stricter thresholds (for risk-averse users)
        - Standard: Balanced approach
        - Aggressive: More lenient (for growth-focused users)
    """
    if not ratios_list:
        return []
    
    thresholds = SCORING_THRESHOLDS.get(method, SCORING_THRESHOLDS['Standard'])
    scored_list = []
    
    for ratios in ratios_list:
        scored = ratios.copy()
        
        # Score liquidity (higher is better)
        cr = ratios.get('current_ratio')
        if cr:
            if cr >= thresholds['current_ratio'][0]:
                scored['liquidity_score'] = 5
                scored['liquidity_status'] = 'Excellent'
            elif cr >= thresholds['current_ratio'][1]:
                scored['liquidity_score'] = 4
                scored['liquidity_status'] = 'Good'
            elif cr >= thresholds['current_ratio'][2]:
                scored['liquidity_score'] = 3
                scored['liquidity_status'] = 'Adequate'
            elif cr >= thresholds['current_ratio'][3]:
                scored['liquidity_score'] = 2
                scored['liquidity_status'] = 'Weak'
            else:
                scored['liquidity_score'] = 1
                scored['liquidity_status'] = 'Poor'
        else:
            scored['liquidity_score'] = 0
            scored['liquidity_status'] = 'N/A'
        
        # Score leverage (lower is better, so thresholds are inverted)
        da = ratios.get('debt_to_assets')
        if da:
            if da <= thresholds['debt_to_assets'][0]:
                scored['leverage_score'] = 5
                scored['leverage_status'] = 'Excellent'
            elif da <= thresholds['debt_to_assets'][1]:
                scored['leverage_score'] = 4
                scored['leverage_status'] = 'Good'
            elif da <= thresholds['debt_to_assets'][2]:
                scored['leverage_score'] = 3
                scored['leverage_status'] = 'Adequate'
            elif da <= thresholds['debt_to_assets'][3]:
                scored['leverage_score'] = 2
                scored['leverage_status'] = 'Weak'
            else:
                scored['leverage_score'] = 1
                scored['leverage_status'] = 'Poor'
        else:
            scored['leverage_score'] = 0
            scored['leverage_status'] = 'N/A'
        
        # Score profitability (higher is better)
        roe = ratios.get('roe')
        if roe:
            if roe >= thresholds['roe'][0]:
                scored['profitability_score'] = 5
                scored['profitability_status'] = 'Excellent'
            elif roe >= thresholds['roe'][1]:
                scored['profitability_score'] = 4
                scored['profitability_status'] = 'Good'
            elif roe >= thresholds['roe'][2]:
                scored['profitability_score'] = 3
                scored['profitability_status'] = 'Adequate'
            elif roe >= thresholds['roe'][3]:
                scored['profitability_score'] = 2
                scored['profitability_status'] = 'Weak'
            else:
                scored['profitability_score'] = 1
                scored['profitability_status'] = 'Poor'
        else:
            scored['profitability_score'] = 0
            scored['profitability_status'] = 'N/A'
        
        # Calculate overall score (weighted average)
        # I weighted leverage and profitability higher because they seem more important
        if (scored['liquidity_score'] > 0 and 
            scored['leverage_score'] > 0 and 
            scored['profitability_score'] > 0):
            scored['overall_score'] = (
                scored['liquidity_score'] * LIQUIDITY_WEIGHT +
                scored['leverage_score'] * LEVERAGE_WEIGHT +
                scored['profitability_score'] * PROFITABILITY_WEIGHT
            )
        else:
            scored['overall_score'] = 0
        
        scored_list.append(scored)
    
    return scored_list

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                     title: str, color_col: str = None) -> go.Figure:
    """
    Create an interactive bar chart.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        title: Chart title
        color_col: Column name for color (optional)
        
    Returns:
        Plotly Figure object
        
    Note:
        I use Plotly instead of matplotlib because:
        - Interactive (hover, zoom, pan)
        - Better looking
        - Easier to create complex charts
    """
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        text=y_col
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(showlegend=False)
    
    return fig


def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str,
                      color_col: str, title: str) -> go.Figure:
    """
    Create an interactive line chart.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color grouping
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        markers=True,
        title=title
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                        color_col: str, size_col: str, title: str) -> go.Figure:
    """
    Create an interactive scatter plot.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Column name for color grouping
        size_col: Column name for marker size
        title: Chart title
        
    Returns:
        Plotly Figure object
        
    Note:
        I use this for showing relationships between two metrics,
        like ROE vs Debt-to-Assets.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=title,
        hover_data=[color_col]
    )
    
    return fig


def create_radar_chart(df: pd.DataFrame, categories: List[str],
                       values_cols: List[str], title: str) -> go.Figure:
    """
    Create a radar chart for multi-dimensional comparison.
    
    Args:
        df: DataFrame with data
        categories: List of category names (e.g., ['Liquidity', 'Leverage', ...])
        values_cols: List of column names for values
        title: Chart title
        
    Returns:
        Plotly Figure object
        
    Note:
        I normalize all values to 0-1 range so they're on the same scale.
        For metrics where lower is better (like debt-to-assets), I invert them.
    """
    fig = go.Figure()
    
    for _, row in df.iterrows():
        values = [row[col] for col in values_cols]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['company']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=title
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function.
    
    This is the entry point for the Streamlit app. It handles:
    1. User input (company selection, API key)
    2. Data fetching and processing
    3. Visualization and analysis
    
    Note:
        I use session_state to persist data across tab switches.
        This prevents re-fetching data when users switch between tabs.
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Financial Health Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session state
    # I learned this is necessary to persist data across tab switches
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Title
    st.title("📊 Financial Health Analyzer")
    st.markdown("*An interactive tool for analyzing company financial health*")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Company selection
    selected = st.sidebar.multiselect(
        "Select companies to analyze",
        list(COMPANIES.keys()),
        default=['AAPL', 'MSFT', 'JPM'],
        format_func=lambda x: COMPANIES[x]
    )
    
    # API Key input
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔑 API Configuration")
    
    # Try to get API key from environment first
    env_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    
    if env_api_key:
        st.sidebar.success("✅ API key loaded from .env file")
        api_key = env_api_key
    else:
        api_key = st.sidebar.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Get your free API key at: https://www.alphavantage.co/support/#api-key"
        )
        
        if not api_key:
            st.sidebar.warning("⚠️ Please enter your API key")
            st.sidebar.info("""
            **How to get a free API key:**
            1. Visit: https://www.alphavantage.co/support/#api-key
            2. Enter your email
            3. Get instant free access
            
            **Tip:** You can also create a .env file with:
            ALPHA_VANTAGE_API_KEY=your_key_here
            """)
            st.stop()
    
    # Scoring method selection
    method = st.sidebar.selectbox(
        "Scoring Method",
        ["Standard", "Conservative", "Aggressive"]
    )
    
    # Chart preferences
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Chart Settings")
    
    chart_type = st.sidebar.radio(
        "Preferred Chart Type",
        ["Interactive (Plotly)", "Simple (Static)"]
    )
    
    color_scheme = st.sidebar.selectbox(
        "Color Scheme",
        ["Default", "Viridis", "Plasma", "Inferno"]
    )
    
    # Main analysis button
    if st.button("🔍 Analyze", type="primary"):
        if not selected:
            st.error("Please select at least one company")
        else:
            # Fetch data
            all_data = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(selected):
                status_text.text(
                    f"Fetching data for {symbol}... ({i+1}/{len(selected)})"
                )
                progress_bar.progress((i + 1) / len(selected))
                
                data = fetch_financial_statements(symbol, api_key)
                if data:
                    ratios = calculate_ratios(data)
                    if ratios:
                        scored = calculate_health_scores(ratios, method)
                        if scored:
                            all_data.extend(scored)
                
                # Wait between requests to avoid rate limit
                if i < len(selected) - 1:
                    api_delay()
            
            progress_bar.empty()
            status_text.empty()
            
            if not all_data:
                st.error("Failed to fetch data. Please check your API key and try again.")
                st.info("""
                **Common issues:**
                - Invalid API key
                - Rate limit exceeded (wait 1 minute)
                - Network connection issues
                
                **Get a free API key:** https://www.alphavantage.co/support/#api-key
                """)
            else:
                # Store data in session state
                # This is crucial for persisting data across tab switches
                st.session_state.df = pd.DataFrame(all_data)
                st.session_state.data_loaded = True
                st.success(f"✅ Successfully loaded data for {len(selected)} companies!")
    
    # Only show analysis if data is loaded
    if st.session_state.data_loaded and st.session_state.df is not None:
        df = st.session_state.df
        
        # Show summary
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_score = df['overall_score'].mean()
            st.metric("Avg Health Score", f"{avg_score:.2f}/5.0")
        
        with col2:
            avg_roe = df['roe'].mean() * 100
            st.metric("Avg ROE", f"{avg_roe:.1f}%")
        
        with col3:
            avg_cr = df['current_ratio'].mean()
            st.metric("Avg Current Ratio", f"{avg_cr:.2f}")
        
        with col4:
            avg_da = df['debt_to_assets'].mean() * 100
            st.metric("Avg Debt/Assets", f"{avg_da:.1f}%")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Health Scores",
            "📈 Ratios",
            "📉 Charts",
            "🔍 Industry Comparison",
            "⚠️ Limitations"
        ])
        
        # Tab 1: Health Scores
        with tab1:
            st.subheader("Financial Health Assessment")
            
            st.info("""
            **How to interpret:**
            - Score 5 = Excellent, 4 = Good, 3 = Adequate, 2 = Weak, 1 = Poor
            - Overall score is weighted: Liquidity (30%), Leverage (35%), Profitability (35%)
            - I chose these weights because leverage and profitability seem more important for long-term health
            """)
            
            # Display table
            display_df = df[['company', 'year', 'current_ratio', 'liquidity_status',
                           'debt_to_assets', 'leverage_status', 'roe', 'profitability_status',
                           'overall_score']].copy()
            
            display_df['current_ratio'] = display_df['current_ratio'].round(2)
            display_df['debt_to_assets'] = (
                display_df['debt_to_assets'] * 100
            ).round(1).astype(str) + '%'
            display_df['roe'] = (display_df['roe'] * 100).round(1).astype(str) + '%'
            display_df['overall_score'] = display_df['overall_score'].round(2)
            
            display_df.columns = ['Company', 'Year', 'Current Ratio', 'Liquidity',
                                 'Debt/Assets', 'Leverage', 'ROE', 'Profitability', 'Score']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Chart type selection
            chart_style = st.radio(
                "Select chart style:",
                ["Bar Chart", "Horizontal Bar"],
                horizontal=True,
                key='health_chart_style'
            )
            
            latest_year = df['year'].max()
            latest_data = df[df['year'] == latest_year].sort_values('overall_score')
            
            if chart_style == "Bar Chart":
                fig = create_bar_chart(
                    latest_data,
                    'company',
                    'overall_score',
                    f'Health Scores ({latest_year})'
                )
                fig.update_layout(yaxis_range=[0, 5])
            else:
                fig = px.bar(
                    latest_data,
                    x='overall_score',
                    y='company',
                    orientation='h',
                    title=f'Health Scores ({latest_year})',
                    text='overall_score',
                    color='overall_score',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(xaxis_range=[0, 5], showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Ratios
        with tab2:
            st.subheader("Key Financial Ratios")
            
            ratio_df = df[['company', 'year', 'current_ratio', 'debt_to_assets',
                          'debt_to_equity', 'net_margin', 'roa', 'roe', 'asset_turnover']].copy()
            
            # Format percentages
            for col in ['debt_to_assets', 'net_margin', 'roa', 'roe']:
                ratio_df[col] = (ratio_df[col] * 100).round(2).astype(str) + '%'
            
            ratio_df['current_ratio'] = ratio_df['current_ratio'].round(2)
            ratio_df['debt_to_equity'] = ratio_df['debt_to_equity'].round(2)
            ratio_df['asset_turnover'] = ratio_df['asset_turnover'].round(2)
            
            ratio_df.columns = ['Company', 'Year', 'Current Ratio', 'Debt/Assets',
                               'Debt/Equity', 'Net Margin', 'ROA', 'ROE', 'Asset Turnover']
            
            st.dataframe(ratio_df, use_container_width=True, hide_index=True)
            
            # Ratio explanations
            with st.expander("📚 Ratio Definitions"):
                st.markdown("""
                **Liquidity Ratios:**
                - **Current Ratio** = Current Assets / Current Liabilities
                  - Measures ability to pay short-term debts
                  - Generally > 1.5 is healthy
                
                **Leverage Ratios:**
                - **Debt-to-Assets** = Total Liabilities / Total Assets
                  - Shows what % of assets are financed by debt
                  - Lower is generally safer
                
                - **Debt-to-Equity** = Total Liabilities / Total Equity
                  - Compares debt to equity financing
                
                **Profitability Ratios:**
                - **Net Margin** = Net Income / Revenue
                  - How much profit per dollar of sales
                
                - **ROA** = Net Income / Total Assets
                  - How efficiently assets generate profit
                
                - **ROE** = Net Income / Total Equity
                  - Return to shareholders
                
                **Efficiency Ratios:**
                - **Asset Turnover** = Revenue / Total Assets
                  - How efficiently assets generate sales
                """)
        
        # Tab 3: Charts
        with tab3:
            st.subheader("Visualizations")
            
            # Chart type selector
            viz_type = st.selectbox(
                "Select visualization:",
                ["ROE Trend", "Current Ratio Trend", "Debt-to-Assets Trend", 
                 "Profitability vs Leverage", "Multi-dimensional Radar"],
                key='viz_type_selector'
            )
            
            if viz_type == "ROE Trend":
                fig = create_line_chart(
                    df.sort_values('year'),
                    'year',
                    'roe',
                    'company',
                    'ROE Trend Analysis'
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Upward trend = improving profitability
                - Downward trend = declining profitability
                - Compare trends across companies
                """)
            
            elif viz_type == "Current Ratio Trend":
                fig = create_line_chart(
                    df.sort_values('year'),
                    'year',
                    'current_ratio',
                    'company',
                    'Current Ratio Trend Analysis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Upward trend = improving liquidity
                - Downward trend = declining liquidity
                - Generally > 1.5 is considered healthy
                """)
            
            elif viz_type == "Debt-to-Assets Trend":
                fig = create_line_chart(
                    df.sort_values('year'),
                    'year',
                    'debt_to_assets',
                    'company',
                    'Debt-to-Assets Trend Analysis'
                )
                fig.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Upward trend = increasing leverage (riskier)
                - Downward trend = decreasing leverage (safer)
                - Lower is generally better
                """)
            
            elif viz_type == "Profitability vs Leverage":
                latest = df[df['year'] == df['year'].max()]
                
                fig = create_scatter_plot(
                    latest,
                    'debt_to_assets',
                    'roe',
                    'company',
                    'revenue',
                    'ROE vs Debt-to-Assets (Latest Year)'
                )
                fig.update_layout(xaxis_tickformat='.1%', yaxis_tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Upper-left = High profitability, low debt (ideal)
                - Lower-right = Low profitability, high debt (risky)
                """)
            
            else:  # Radar chart
                latest = df[df['year'] == df['year'].max()]
                
                # Normalize for radar chart
                radar_data = latest[['company', 'current_ratio', 'debt_to_assets', 
                                    'roe', 'roa', 'net_margin']].copy()
                
                # Scale to 0-1 range
                radar_data['current_ratio'] = radar_data['current_ratio'].apply(
                    lambda x: normalize_for_radar(x, 3, invert=False)
                )
                radar_data['debt_to_assets'] = radar_data['debt_to_assets'].apply(
                    lambda x: normalize_for_radar(x, 1, invert=True)
                )
                radar_data['roe'] = radar_data['roe'].apply(
                    lambda x: normalize_for_radar(x, 0.3, invert=False)
                )
                radar_data['roa'] = radar_data['roa'].apply(
                    lambda x: normalize_for_radar(x, 0.15, invert=False)
                )
                radar_data['net_margin'] = radar_data['net_margin'].apply(
                    lambda x: normalize_for_radar(x, 0.25, invert=False)
                )
                
                categories = ['Liquidity', 'Low Leverage', 'ROE', 'ROA', 'Net Margin']
                value_cols = ['current_ratio', 'debt_to_assets', 'roe', 'roa', 'net_margin']
                
                fig = create_radar_chart(radar_data, categories, value_cols, 
                                        'Financial Health Radar')
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Interpretation:**
                - Larger area = better overall health
                - Balanced shape = well-rounded performance
                - Compare shapes across companies
                """)
        
        # Tab 4: Industry Comparison
        with tab4:
            st.subheader("Industry Benchmark Comparison")
            
            st.info("Benchmarks from Damodaran datasets (NYU Stern)")
            
            latest = df[df['year'] == df['year'].max()]
            comparison_data = []
            
            for _, row in latest.iterrows():
                industry = row.get('industry', '')
                benchmark = get_industry_benchmark(industry)
                
                comparison_data.append({
                    'Company': row['company'],
                    'Industry': industry,
                    'Current Ratio': f"{row['current_ratio']:.2f}",
                    'Industry Avg CR': f"{benchmark['current_ratio']:.2f}",
                    'Debt/Assets': f"{row['debt_to_assets']*100:.1f}%",
                    'Industry Avg DA': f"{benchmark['debt_to_assets']*100:.1f}%",
                    'ROE': f"{row['roe']*100:.1f}%",
                    'Industry Avg ROE': f"{benchmark['roe']*100:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(comparison_data), 
                        use_container_width=True, hide_index=True)
            
            # Comparison chart
            st.markdown("#### Company vs Industry Average")
            
            metric = st.radio(
                "Select metric to compare:",
                ["ROE", "Current Ratio", "Debt/Assets"],
                horizontal=True,
                key='metric_selector'
            )
            
            chart_data = []
            for _, row in latest.iterrows():
                industry = row.get('industry', '')
                benchmark = get_industry_benchmark(industry)
                
                if metric == "ROE":
                    company_val = row['roe'] * 100
                    industry_val = benchmark['roe'] * 100
                elif metric == "Current Ratio":
                    company_val = row['current_ratio']
                    industry_val = benchmark['current_ratio']
                else:
                    company_val = row['debt_to_assets'] * 100
                    industry_val = benchmark['debt_to_assets'] * 100
                
                chart_data.append({
                    'Company': row['company'],
                    'Type': 'Company',
                    'Value': company_val
                })
                chart_data.append({
                    'Company': row['company'],
                    'Type': 'Industry',
                    'Value': industry_val
                })
            
            fig = px.bar(
                pd.DataFrame(chart_data),
                x='Company',
                y='Value',
                color='Type',
                barmode='group',
                title=f'{metric}: Company vs Industry Average'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tab 5: Limitations
        with tab5:
            st.subheader("Limitations & Disclaimers")
            
            st.warning("⚠️ **This is NOT investment advice. For educational purposes only.**")
            
            st.markdown("### Data Limitations")
            st.markdown("""
            - **API Rate Limits**: Alpha Vantage free tier allows 5 calls per minute
            - **Data Delays**: Financial data may not be real-time
            - **Missing Values**: Some companies don't report all metrics
            - **Small Sample**: Only analyzing a few companies
            """)
            
            st.markdown("### Methodology Limitations")
            st.markdown("""
            - **Arbitrary Thresholds**: I chose scoring thresholds based on general guidelines
            - **Subjective Weights**: The 30%/35%/35% weights are my personal choice
            - **No Statistical Tests**: I didn't perform significance testing
            - **Industry Matching**: Industry classification might not be accurate
            - **No Qualitative Factors**: Only looks at numbers, not management quality
            """)
            
            st.markdown("### What I Learned")
            st.markdown("""
            This was my first time building a financial analysis tool. Some challenges:
            
            1. **API Integration**: Had to figure out rate limits and error handling
            2. **Data Quality**: Real-world data is messy and incomplete
            3. **Industry Differences**: Different sectors have very different financial structures
            4. **Scoring Design**: Creating a fair scoring system is harder than I thought
            
            If I had more time, I would:
            - Add more companies and industries
            - Include statistical analysis
            - Better industry classification
            - More educational content
            """)
            
            st.markdown("### AI Disclosure")
            st.info("""
            **AI Tools Used:**
            - Claude (Anthropic) for code suggestions and debugging
            - I used it maybe 10-15% of the time, mainly for:
              - Understanding Alpha Vantage API
              - Debugging errors
              - Improving chart layouts
            
            **My Contribution:**
            - I designed the overall structure
            - I chose which companies and ratios to include
            - I created the scoring system
            - I wrote most of the code
            - I tested everything
            - I wrote all explanations
            
            **Understanding:**
            I confirm that I understand how this tool works and can explain:
            - How data is fetched from Alpha Vantage
            - How each ratio is calculated
            - How the scoring system works
            - What the limitations are
            """)
        
        # Export
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Data")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "📥 Download CSV",
            csv,
            f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Financial Health Analyzer | ACC102 Mini Assignment</p>
        <p style='font-size: 0.9em;'>Built with Python, Streamlit, and Alpha Vantage API</p>
        <p style='font-size: 0.9em;'>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
