"""
ATR Box Scalping Trading Tool - Web Version
Mobile-optimized Streamlit application
Educational purpose only - Not financial advice
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="ATR Box Scalping",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for mobile optimization
st.markdown("""
<style>
    .main-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: bold;
        color: #555;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .error-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

def log_message(message):
    """Add message to log with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")
    # Keep only last 50 messages
    if len(st.session_state.logs) > 50:
        st.session_state.logs = st.session_state.logs[-50:]

def calculate_atr(data, period=14):
    """Calculate Average True Range"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr.iloc[-1]

def is_hammer(candle):
    """Detect hammer candlestick pattern"""
    open_price = float(candle['Open'])
    high = float(candle['High'])
    low = float(candle['Low'])
    close = float(candle['Close'])
    
    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    upper_shadow = high - max(open_price, close)
    
    if lower_shadow > 2 * body and upper_shadow < body * 0.3 and body > 0:
        return True
    return False

def is_bullish_engulfing(prev_candle, curr_candle):
    """Detect bullish engulfing pattern"""
    prev_open = float(prev_candle['Open'])
    prev_close = float(prev_candle['Close'])
    curr_open = float(curr_candle['Open'])
    curr_close = float(curr_candle['Close'])
    
    if (prev_close < prev_open and 
        curr_close > curr_open and
        curr_open < prev_close and
        curr_close > prev_open):
        return True
    return False

def is_bearish_engulfing(prev_candle, curr_candle):
    """Detect bearish engulfing pattern"""
    prev_open = float(prev_candle['Open'])
    prev_close = float(prev_candle['Close'])
    curr_open = float(curr_candle['Open'])
    curr_close = float(curr_candle['Close'])
    
    if (prev_close > prev_open and 
        curr_close < curr_open and
        curr_open > prev_close and
        curr_close < prev_open):
        return True
    return False

def fetch_and_analyze(symbol):
    """Fetch data and perform analysis"""
    try:
        log_message(f"Fetching data for {symbol}...")
        
        # Get daily data for ATR
        daily_data = yf.download(symbol, period='1mo', interval='1d', progress=False)
        
        if daily_data.empty:
            return None, "No daily data received"
        
        # Flatten multi-index columns
        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_data.columns = daily_data.columns.get_level_values(0)
        
        # Calculate ATR
        atr_value = calculate_atr(daily_data)
        log_message(f"ATR(14) = ${atr_value:.2f}")
        
        # Get 5-minute data
        data_5min = yf.download(symbol, period='5d', interval='5m', progress=False)
        
        if data_5min.empty:
            return None, "No 5-minute data received"
        
        # Flatten multi-index columns
        if isinstance(data_5min.columns, pd.MultiIndex):
            data_5min.columns = data_5min.columns.get_level_values(0)
        
        # Get today's data or most recent
        today = datetime.now().date()
        today_data = data_5min[data_5min.index.date == today]
        
        is_live_data = True
        if len(today_data) < 4:
            log_message("⚠️ Market appears closed. Using most recent trading day.")
            available_dates = data_5min.index.date
            unique_dates = sorted(set(available_dates), reverse=True)
            
            if len(unique_dates) == 0:
                return None, "No historical data available"
            
            most_recent_date = unique_dates[0]
            today_data = data_5min[data_5min.index.date == most_recent_date]
            is_live_data = False
            log_message(f"📊 Showing data from: {most_recent_date.strftime('%Y-%m-%d')}")
        
        if len(today_data) < 4:
            return None, "Insufficient data available"
        
        # Define box (first 15 minutes = 3 candles)
        box_data = today_data.head(3)
        box_high = float(box_data['High'].max())
        box_low = float(box_data['Low'].min())
        box_range = box_high - box_low
        
        # Check volatility filter
        volatility_threshold = 0.25 * atr_value
        is_valid_volatility = box_range < volatility_threshold
        
        # Get current price
        current_price = float(today_data['Close'].iloc[-1])
        
        # Check for signals
        signal = None
        if is_valid_volatility and len(today_data) >= 4:
            prev_candle = today_data.iloc[-2]
            curr_candle = today_data.iloc[-1]
            
            if current_price > box_high:
                if is_bearish_engulfing(prev_candle, curr_candle):
                    signal = {
                        'direction': 'SHORT',
                        'type': 'Bearish Engulfing',
                        'entry': float(curr_candle['High']),
                        'stop_loss': float(curr_candle['High']),
                        'take_profit': float(box_low),
                        'candle_time': curr_candle.name
                    }
                    log_message("🔴 BEARISH ENGULFING detected - SHORT signal")
            
            elif current_price < box_low:
                if is_hammer(curr_candle):
                    signal = {
                        'direction': 'LONG',
                        'type': 'Hammer',
                        'entry': float(curr_candle['Low']),
                        'stop_loss': float(curr_candle['Low']),
                        'take_profit': float(box_high),
                        'candle_time': curr_candle.name
                    }
                    log_message("🟢 HAMMER detected - LONG signal")
                elif is_bullish_engulfing(prev_candle, curr_candle):
                    signal = {
                        'direction': 'LONG',
                        'type': 'Bullish Engulfing',
                        'entry': float(curr_candle['Low']),
                        'stop_loss': float(curr_candle['Low']),
                        'take_profit': float(box_high),
                        'candle_time': curr_candle.name
                    }
                    log_message("🟢 BULLISH ENGULFING detected - LONG signal")
        
        result = {
            'atr': atr_value,
            'box_high': box_high,
            'box_low': box_low,
            'box_range': box_range,
            'volatility_threshold': volatility_threshold,
            'is_valid_volatility': is_valid_volatility,
            'current_price': current_price,
            'data': today_data,
            'is_live_data': is_live_data,
            'signal': signal
        }
        
        return result, None
        
    except Exception as e:
        log_message(f"Error: {str(e)}")
        return None, str(e)

def create_chart(data, box_high, box_low, signal, is_live_data, symbol):
    """Create interactive Plotly candlestick chart"""
    
    # Create candlestick chart
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Add box boundaries
    fig.add_hline(y=box_high, line_dash="dash", line_color="blue", 
                  annotation_text=f"Box High: ${box_high:.2f}",
                  annotation_position="right")
    fig.add_hline(y=box_low, line_dash="dash", line_color="blue",
                  annotation_text=f"Box Low: ${box_low:.2f}",
                  annotation_position="right")
    
    # Add box shading
    fig.add_hrect(y0=box_low, y1=box_high, fillcolor="blue", opacity=0.1,
                  layer="below", line_width=0)
    
    # Add signal marker if present
    if signal:
        fig.add_trace(go.Scatter(
            x=[signal['candle_time']],
            y=[signal['entry']],
            mode='markers',
            marker=dict(size=20, color='yellow', symbol='star',
                       line=dict(color='black', width=2)),
            name=f"{signal['direction']} Signal"
        ))
    
    # Update layout
    title_text = f"{symbol} - 5 Minute Chart"
    if not is_live_data:
        data_date = data.index[0].strftime('%Y-%m-%d')
        title_text += f" (HISTORICAL: {data_date})"
    else:
        title_text += " (LIVE)"
    
    fig.update_layout(
        title=title_text,
        yaxis_title='Price ($)',
        xaxis_title='Time',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Add warning annotation if historical
    if not is_live_data:
        fig.add_annotation(
            text="⚠️ MARKET CLOSED - HISTORICAL DATA ONLY",
            xref="paper", yref="paper",
            x=0.5, y=0.95,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="yellow",
            opacity=0.8
        )
    
    return fig

# Main App
st.markdown('<div class="main-title">📈 ATR Box Scalping Strategy</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    symbol = st.text_input("Symbol", value="SPY", help="Enter stock symbol (e.g., SPY, AAPL, QQQ)")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.last_update = datetime.now()
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
    
    st.markdown("---")
    
    st.subheader("📚 Strategy Overview")
    st.markdown("""
    1. Calculate ATR(14) on Daily chart
    2. Record High/Low in first 15min
    3. Box must be < 25% of ATR
    4. Wait for breakout
    5. Look for reversal candle
    6. Enter at candle extreme
    7. TP at opposite box boundary
    """)
    
    st.markdown("---")
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        **Mobile Tips:**
        - Pinch to zoom on chart
        - Swipe to pan
        - Tap data points for details
        - Works offline after first load
        
        **Best Symbols:**
        - SPY (S&P 500 ETF)
        - QQQ (Nasdaq ETF)
        - AAPL, MSFT, GOOGL
        
        **Market Hours:**
        - 9:30 AM - 4:00 PM ET
        - Monday - Friday
        """)

# Main content area
if st.session_state.last_update or st.button("▶️ Start Analysis", use_container_width=True):
    
    with st.spinner(f"Analyzing {symbol}..."):
        result, error = fetch_and_analyze(symbol.upper())
    
    if error:
        st.markdown(f'<div class="error-box">❌ Error: {error}</div>', unsafe_allow_html=True)
    
    elif result:
        # Market Status
        if result['is_live_data']:
            st.markdown('<div class="success-box">🟢 <strong>MARKET STATUS: LIVE DATA</strong></div>', 
                       unsafe_allow_html=True)
        else:
            data_date = result['data'].index[0].strftime('%Y-%m-%d')
            st.markdown(f'<div class="warning-box">🔴 <strong>MARKET CLOSED</strong> - Showing historical data from {data_date}</div>', 
                       unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("ATR (Daily)", f"${result['atr']:.2f}")
        with col2:
            st.metric("Box High", f"${result['box_high']:.2f}")
        with col3:
            st.metric("Box Low", f"${result['box_low']:.2f}")
        with col4:
            st.metric("Current Price", f"${result['current_price']:.2f}")
        
        # Volatility Status
        if result['is_valid_volatility']:
            st.markdown('<div class="success-box">✅ <strong>Volatility Valid</strong> - Box range acceptable for trading</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ <strong>Volatility Too Wide</strong> - Box range (${result["box_range"]:.2f}) exceeds 25% of ATR (${result["volatility_threshold"]:.2f})</div>', 
                       unsafe_allow_html=True)
        
        # Signal Status
        if result['signal']:
            signal = result['signal']
            st.markdown(f"""
            <div class="info-box">
                <h3>⭐ {signal['direction']} SIGNAL DETECTED</h3>
                <p><strong>Pattern:</strong> {signal['type']}</p>
                <p><strong>Entry Price:</strong> ${signal['entry']:.2f}</p>
                <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
                <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
                <p><strong>Risk/Reward:</strong> {abs(signal['take_profit'] - signal['entry']) / max(abs(signal['entry'] - signal['stop_loss']), 0.01):.2f}:1</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⏳ No signal detected. Waiting for valid setup...")
        
        # Chart
        st.subheader("📊 5-Minute Chart")
        chart = create_chart(
            result['data'], 
            result['box_high'], 
            result['box_low'],
            result['signal'],
            result['is_live_data'],
            symbol.upper()
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # Additional Stats
        with st.expander("📈 Detailed Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Box Range", f"${result['box_range']:.2f}")
                st.metric("25% of ATR", f"${result['volatility_threshold']:.2f}")
                st.metric("High-Low Ratio", f"{(result['box_range']/result['atr']*100):.1f}%")
            with col2:
                st.metric("Data Points", len(result['data']))
                st.metric("Time Span", f"{len(result['data']) * 5} minutes")
                if result['signal']:
                    risk = abs(result['signal']['entry'] - result['signal']['stop_loss'])
                    reward = abs(result['signal']['take_profit'] - result['signal']['entry'])
                    st.metric("Potential Risk", f"${risk:.2f}")
                    st.metric("Potential Reward", f"${reward:.2f}")

# Trade Log
with st.expander("📝 Trade Log", expanded=False):
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-20:]):  # Show last 20 logs
            st.text(log)
    else:
        st.info("No logs yet. Start analyzing to see activity.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>⚠️ EDUCATIONAL TOOL ONLY - NOT FINANCIAL ADVICE</strong><br>
    Trading involves substantial risk of loss. Always consult with a licensed financial advisor.
</div>
""", unsafe_allow_html=True)

# Auto-refresh logic
if auto_refresh and 'auto_refresh' in locals():
    time.sleep(refresh_interval)
    st.rerun()
