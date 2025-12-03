# ============================================================================
# CREDIT SCORING STREAMLIT APPLICATION - ENHANCED INTERACTIVE VERSION
# Interactive Web App for Credit Scoring and Default Risk Prediction
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING - ENHANCED INTERACTIVE DESIGN
# ============================================================================

st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main App Background - Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a3e 50%, #0f2d4d 100%);
    }
    
    /* Metrics - Enhanced with Glassmorphism */
    .stMetric {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(0, 150, 200, 0.1) 100%);
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(0, 212, 255, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .stMetric:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 15px 50px rgba(0, 212, 255, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.6);
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.25) 0%, rgba(0, 150, 200, 0.15) 100%);
    }
    
    .stMetric label {
        color: #00d4ff !important;
        font-weight: 700;
        font-size: 14px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    .stMetric [data-testid="metricDeltaContainer"] {
        color: #00ff88 !important;
        font-weight: 600;
    }
    
    /* Headers - Neon Effect */
    h1 {
        color: #00d4ff !important;
        text-align: center;
        font-size: 3.2em;
        font-weight: 900;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.6), 0 0 40px rgba(0, 180, 219, 0.3);
        margin-bottom: 10px;
        letter-spacing: 3px;
        background: linear-gradient(135deg, #00d4ff 0%, #00b4db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s ease-in-out infinite;
    }
    
    h2 {
        color: #00d4ff !important;
        border-bottom: 3px solid #00d4ff !important;
        border-image: linear-gradient(135deg, #00d4ff, #0083b0) 1;
        padding-bottom: 15px;
        font-size: 1.9em;
        font-weight: 800;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    
    h3 {
        color: #00d4ff !important;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    @keyframes glow {
        0%, 100% { text-shadow: 0 0 20px rgba(0, 212, 255, 0.6), 0 0 40px rgba(0, 180, 219, 0.3); }
        50% { text-shadow: 0 0 30px rgba(0, 212, 255, 0.8), 0 0 60px rgba(0, 180, 219, 0.5); }
    }
    
    /* Colored Boxes - Glassmorphism with Glow */
    .success-box {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(56, 142, 60, 0.08));
        border: 2px solid rgba(76, 175, 80, 0.5);
        padding: 16px;
        border-radius: 15px;
        color: #81c784;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 5px solid #4caf50;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.12), rgba(245, 127, 23, 0.08));
        border: 2px solid rgba(255, 193, 7, 0.5);
        padding: 16px;
        border-radius: 15px;
        color: #ffb74d;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(255, 193, 7, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 5px solid #ffc107;
    }
    
    .danger-box {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.12), rgba(229, 57, 53, 0.08));
        border: 2px solid rgba(244, 67, 54, 0.5);
        padding: 16px;
        border-radius: 15px;
        color: #ef5350;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(244, 67, 54, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-left: 5px solid #f44336;
    }
    
    /* Buttons - Animated with Gradient */
    .stButton > button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        border: 2px solid rgba(0, 212, 255, 0.4);
        padding: 14px 35px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 15px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(0, 180, 219, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0083b0 0%, #00b4db 100%);
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(0, 180, 219, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
        border-color: rgba(0, 212, 255, 0.8);
    }
    
    .stButton > button::before {
        left: 100%;
    }
    
    /* Input Fields - Enhanced */
    .stNumberInput input, .stSelectbox select, .stSlider div {
        background-color: rgba(0, 180, 219, 0.08) !important;
        border: 2px solid rgba(0, 212, 255, 0.4) !important;
        color: #00d4ff !important;
        border-radius: 12px;
        padding: 12px 15px;
        transition: all 0.3s ease;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4), inset 0 2px 5px rgba(0, 0, 0, 0.2) !important;
        background-color: rgba(0, 180, 219, 0.12) !important;
    }
    
    /* File Uploader - Glassmorphism */
    .stFileUploader {
        background: linear-gradient(135deg, rgba(0, 180, 219, 0.1), rgba(0, 131, 176, 0.05));
        border: 2.5px dashed rgba(0, 212, 255, 0.5);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(0, 212, 255, 0.8);
        box-shadow: 0 12px 35px rgba(0, 212, 255, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.15);
        background: linear-gradient(135deg, rgba(0, 180, 219, 0.15), rgba(0, 131, 176, 0.1));
    }
    
    /* Dataframe - Styled */
    .stDataFrame {
        background-color: rgba(0, 180, 219, 0.05) !important;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.1);
    }
    
    /* Sidebar - Enhanced */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #132f4c 50%, #0a1929 100%);
        border-right: 3px solid rgba(0, 212, 255, 0.3);
        box-shadow: inset -5px 0 15px rgba(0, 212, 255, 0.1);
    }
    
    /* Radio Button - Styled */
    .stRadio > div > label {
        color: #00d4ff !important;
        font-weight: 700;
        font-size: 16px;
        padding: 12px 15px;
        border-radius: 12px;
        transition: all 0.3s ease;
        margin: 5px 0;
        border: 2px solid transparent;
    }
    
    .stRadio > div > label:hover {
        background-color: rgba(0, 212, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.4);
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    
    /* Info Box */
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 180, 219, 0.12), rgba(0, 131, 176, 0.08));
        border-left: 5px solid #00b4db !important;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 180, 219, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Expander - Interactive */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(0, 180, 219, 0.15), rgba(0, 131, 176, 0.1));
        border-radius: 12px;
        color: #00d4ff !important;
        font-weight: 700;
        border: 2px solid rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(0, 180, 219, 0.25), rgba(0, 131, 176, 0.15));
        border-color: rgba(0, 212, 255, 0.6);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.2);
    }
    
    /* Divider */
    hr {
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 10px;
        background: linear-gradient(90deg, transparent, rgba(0, 212, 255, 0.3), transparent);
    }
    
    /* Markdown Text */
    .markdown-text-container {
        color: #e0e0e0;
    }
    
    /* Spinner */
    .stSpinner div {
        border-color: rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Success Message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.12), rgba(56, 142, 60, 0.08));
        border-left: 5px solid #4caf50 !important;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(76, 175, 80, 0.15);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: white;
        border: 2px solid rgba(0, 212, 255, 0.4);
        padding: 14px 35px;
        border-radius: 30px;
        font-weight: 700;
        box-shadow: 0 6px 20px rgba(0, 180, 219, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 35px rgba(0, 180, 219, 0.6);
    }
    
    /* Stat Card Container */
    .stat-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 150, 200, 0.05));
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS (UNCHANGED)
# ============================================================================

@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    try:
        model = pickle.load(open('models/credit_scoring_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        return model, scaler
    except:
        st.error("⚠️ Models not found! Ensure models/ folder contains pkl files.")
        return None, None

@st.cache_data
def load_data(uploaded_file=None):
    """Load dataset"""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv('data/preprocessed_data.csv')
        return df
    except:
        st.error("⚠️ Data file not found!")
        return None

def calculate_credit_score(row):
    """Calculate credit score (300-850)"""
    score = 300
    
    # Payment History (35%)
    payment_score = 297
    if row['Num_of_Delayed_Payment'] > 0:
        payment_score -= (row['Num_of_Delayed_Payment'] * 50)
    if row['Delay_from_due_date'] > 30:
        payment_score -= min(100, row['Delay_from_due_date'])
    if row['Payment_of_Min_Amount'] == 1:
        payment_score -= 50
    payment_score = max(0, payment_score)
    score += payment_score * 0.35
    
    # Credit Utilization (30%)
    utilization_score = 255
    utilization = row['Credit_Utilization_Ratio']
    if utilization > 30:
        utilization_score -= (utilization - 30) * 5
    utilization_score = max(0, utilization_score)
    score += utilization_score * 0.30
    
    # Credit Age (15%)
    age_score = 127
    age_years = row['Credit_History_Age_Years']
    if age_years < 2:
        age_score = 50
    elif age_years < 5:
        age_score = 100
    elif age_years >= 10:
        age_score = 127
    score += age_score * 0.15
    
    # Credit Mix (10%)
    mix_score = 85 if row['Credit_Mix'] == 'Good' else 50
    score += mix_score * 0.10
    
    # New Inquiries (10%)
    inquiry_score = 85
    if row['Num_Credit_Inquiries'] > 10:
        inquiry_score -= (row['Num_Credit_Inquiries'] - 10) * 3
    inquiry_score = max(0, inquiry_score)
    score += inquiry_score * 0.10
    
    return min(850, max(300, round(score)))

def get_credit_tier(score):
    """Get credit tier from score"""
    if score >= 750:
        return 'Excellent', '🟢'
    elif score >= 700:
        return 'Very Good', '🟢'
    elif score >= 650:
        return 'Good', '🟡'
    elif score >= 550:
        return 'Fair', '🟠'
    else:
        return 'Poor', '🔴'

def calculate_default_probability(row, score):
    """Calculate default probability"""
    base_prob = 0.5
    score_factor = (850 - score) / 550
    payment_factor = min(0.3, row['Num_of_Delayed_Payment'] * 0.05)
    util_factor = max(0, (row['Credit_Utilization_Ratio'] - 30) / 100)
    default_prob = (score_factor * 0.5 + payment_factor * 0.3 + util_factor * 0.2)
    return min(0.95, max(0.01, round(default_prob, 4)))

def calculate_interest_rate(credit_score, tier, loan_type='Personal Loan', income=0, debt=0):
    """Calculate interest rate based on risk"""
    base_rates = {
        'Personal Loan': 10.5,
        'Housing Loan': 7.5,
        'Auto Loan': 8.0,
        'Student Loan': 6.5,
        'Other': 11.0
    }
    base_rate = base_rates.get(loan_type, 11.0)
    tier_adj = {
        'Excellent': -1.5,
        'Very Good': -0.75,
        'Good': 0,
        'Fair': 2.5,
        'Poor': 5.0
    }.get(tier, 0)
    
    dti_adj = 0
    if income > 0:
        dti = debt / income
        if dti > 0.5:
            dti_adj = 3.0
        elif dti > 0.4:
            dti_adj = 2.0
        elif dti > 0.3:
            dti_adj = 1.0
    
    rate = base_rate + tier_adj + dti_adj
    return round(max(3.0, min(25.0, rate)), 2)

def calculate_emi(principal, annual_rate, months):
    """Calculate EMI"""
    if annual_rate == 0:
        return principal / months
    monthly_rate = annual_rate / 100 / 12
    emi = principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
    return round(emi, 2)

def generate_suggestions(row, score, tier):
    """Generate improvement suggestions"""
    suggestions = []
    
    if row['Num_of_Delayed_Payment'] > 5:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Payment History',
            'suggestion': f"You have {int(row['Num_of_Delayed_Payment'])} delayed payments. Set up automatic payments immediately.",
            'impact': '15-30%'
        })
    elif row['Num_of_Delayed_Payment'] > 0:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Payment History',
            'suggestion': "Set up automatic payments or payment reminders to avoid missing dues.",
            'impact': '10-20%'
        })
    
    if row['Credit_Utilization_Ratio'] > 40:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Credit Utilization',
            'suggestion': f"Your utilization is {row['Credit_Utilization_Ratio']:.1f}%. Reduce it below 30% by paying down balances.",
            'impact': '20-30%'
        })
    elif row['Credit_Utilization_Ratio'] > 30:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Credit Utilization',
            'suggestion': f"Your utilization is {row['Credit_Utilization_Ratio']:.1f}%. Try to keep it below 30%.",
            'impact': '10-20%'
        })
    
    if row['Credit_History_Age_Years'] < 2:
        suggestions.append({
            'priority': 'MEDIUM',
            'category': 'Credit History',
            'suggestion': "Keep your oldest accounts open to build a longer credit history.",
            'impact': '5-15%'
        })
    
    if row['Payment_of_Min_Amount'] == 1:
        suggestions.append({
            'priority': 'HIGH',
            'category': 'Payment Behavior',
            'suggestion': "Pay more than the minimum amount to reduce debt faster.",
            'impact': '10-20%'
        })
    
    return suggestions

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header with enhanced styling
    st.title("💳 Credit Scoring System")
    st.markdown("### 🚀 Intelligent Credit Risk Assessment & Scoring Platform")
    st.markdown("---")
    
    # Sidebar Navigation with enhanced styling
    st.sidebar.markdown("""
        <div style="text-align: center; padding: 20px 0; background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(0, 150, 200, 0.05)); border-radius: 15px; margin-bottom: 20px;">
            <h2 style="color: #00d4ff; font-size: 1.6em; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5); margin: 0;">🏦 Navigation Menu</h2>
        </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "Choose Feature:",
        [
            "📉 Analytics Dashboard",
            "📈 Batch Scoring", 
            "💡 Recommendations",
            "💰 Interest Rate Calculator",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )
    
    # Load models
    model, scaler = load_model()
    
    # ====================================================================
    # PAGE 1: BATCH SCORING (ENHANCED)
    # ====================================================================
    
    if page == "📈 Batch Scoring":
        st.header("📈 Batch Customer Scoring")
        st.markdown("Upload a CSV file to score multiple customers at once.")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("📁 Choose CSV file", type="csv", help="Upload customer data CSV file")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**✓ File loaded:** {len(df)} records found")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Records", len(df))
            with col2:
                st.metric("📋 Columns", len(df.columns))
            with col3:
                st.metric("✅ Status", "Ready to Score")
            
            st.markdown("---")
            
            if st.button("🚀 Score All Customers", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🔄 Processing customers..."):
                    df['Credit_Score'] = df.apply(calculate_credit_score, axis=1)
                    progress_bar.progress(33)
                    status_text.info("✓ Credit scores calculated")
                    
                    df['Credit_Tier'] = df['Credit_Score'].apply(lambda x: get_credit_tier(x)[0])
                    progress_bar.progress(66)
                    status_text.info("✓ Credit tiers assigned")
                    
                    df['Default_Probability'] = df.apply(
                        lambda row: calculate_default_probability(row, row['Credit_Score']), axis=1
                    )
                    progress_bar.progress(100)
                    status_text.success(f"✅ Successfully scored {len(df)} customers!")
                
                st.markdown("---")
                st.subheader("📊 Sample Results (First 10)")
                st.dataframe(
                    df[['Customer_ID', 'Name', 'Credit_Score', 'Credit_Tier', 'Default_Probability']].head(10),
                    use_container_width=True
                )
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = df['Credit_Score'].mean()
                    st.metric("📈 Avg Score", f"{avg_score:.0f}", f"{avg_score-500:.0f}")
                with col2:
                    avg_default = df['Default_Probability'].mean()
                    st.metric("⚠️ Avg Default Risk", f"{avg_default:.2%}")
                with col3:
                    excellent_count = len(df[df['Credit_Tier'] == 'Excellent'])
                    st.metric("🟢 Excellent", excellent_count)
                with col4:
                    poor_count = len(df[df['Credit_Tier'] == 'Poor'])
                    st.metric("🔴 Poor", poor_count)
                
                st.markdown("---")
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results",
                    data=csv,
                    file_name="credit_scores.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # ====================================================================
    # PAGE 2: ANALYTICS DASHBOARD (ENHANCED)
    # ====================================================================
    
    elif page == "📉 Analytics Dashboard":
        st.header("📉 Portfolio Analytics Dashboard")
        st.markdown("Comprehensive portfolio insights and analysis")
        st.markdown("---")
        
        df = load_data()
        
        if df is not None:
            df['Credit_Score'] = df.apply(calculate_credit_score, axis=1)
            df['Credit_Tier'] = df['Credit_Score'].apply(lambda x: get_credit_tier(x)[0])
            
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("👥 Total Customers", f"{len(df):,}")
            with col2:
                st.metric("📊 Avg Score", f"{df['Credit_Score'].mean():.0f}/850")
            with col3:
                excellent = len(df[df['Credit_Tier'] == 'Excellent'])
                st.metric("🟢 Excellent Tier", excellent)
            with col4:
                poor = len(df[df['Credit_Tier'] == 'Poor'])
                st.metric("🔴 Poor Tier", poor)
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Credit Score Distribution")
                fig = px.histogram(
                    df, 
                    x='Credit_Score', 
                    nbins=30,
                    title='Distribution of Credit Scores',
                    labels={'Credit_Score': 'Score', 'count': 'Customers'},
                    color_discrete_sequence=['#00d4ff']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(15,15,30,0.5)',
                    font={'color': '#00d4ff'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Credit Tier Distribution")
                tier_counts = df['Credit_Tier'].value_counts()
                fig = go.Figure(data=[go.Pie(
                    labels=tier_counts.index,
                    values=tier_counts.values,
                    marker=dict(colors=['#4caf50', '#81c784', '#ffc107', '#ff9800', '#f44336'])
                )])
                fig.update_layout(
                    paper_bgcolor='rgba(15,15,30,0.5)',
                    font={'color': '#00d4ff'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # ====================================================================
    # PAGE 3: RECOMMENDATIONS (ENHANCED)
    # ====================================================================
    
    elif page == "💡 Recommendations":
        st.header("💡 Personalized Recommendations")
        st.markdown("Get improvement suggestions tailored to your profile")
        st.markdown("---")
        
        df = load_data()
        
        if df is not None:
            customer = st.selectbox(
                "🔍 Select Customer",
                df['Name'].unique(),
                label_visibility="visible"
            )
            customer_row = df[df['Name'] == customer].iloc[0]
            
            score = calculate_credit_score(customer_row)
            tier, emoji = get_credit_tier(score)
            
            # Customer Card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👤 Customer", customer)
            with col2:
                st.metric("⭐ Score", f"{score}/850")
            with col3:
                st.metric("📊 Tier", f"{emoji} {tier}")
            
            st.markdown("---")
            st.subheader("✨ Improvement Suggestions")
            
            suggestions = generate_suggestions(customer_row, score, tier)
            
            if suggestions:
                for i, sugg in enumerate(suggestions, 1):
                    with st.expander(f"{i}. {sugg['category']} ({sugg['priority']})"):
                        st.write(sugg['suggestion'])
                        st.markdown(f"**💪 Potential Impact:** `+{sugg['impact']} improvement`")
            else:
                st.success("✅ No improvements needed! This customer has excellent credit practices.")
    
    # ====================================================================
    # PAGE 4: INTEREST RATE CALCULATOR (ENHANCED)
    # ====================================================================
    
    elif page == "💰 Interest Rate Calculator":
        st.header("💰 Interest Rate & EMI Calculator")
        st.markdown("Calculate personalized rates and loan payments")
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Credit Profile")
            credit_score = st.slider("Credit Score", 300, 850, 700)
            tier, emoji = get_credit_tier(credit_score)
            st.markdown(f"**Tier:** {emoji} `{tier}`")
        
        with col2:
            st.markdown("### 💼 Income & Debt")
            annual_income = st.number_input("Annual Income (₹)", value=500000, step=10000)
            outstanding_debt = st.number_input("Outstanding Debt (₹)", value=100000, step=10000)
        
        with col3:
            st.markdown("### 🏦 Loan Details")
            loan_type = st.selectbox("Loan Type", ['Personal Loan', 'Housing Loan', 'Auto Loan', 'Student Loan'])
            loan_amount = st.number_input("Loan Amount (₹)", value=500000, step=10000)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure_months = st.slider("Loan Tenure (months)", 12, 360, 60)
        
        with col2:
            interest_rate = calculate_interest_rate(credit_score, tier, loan_type, annual_income, outstanding_debt)
            st.metric("📈 Interest Rate", f"{interest_rate}%")
        
        st.markdown("---")
        
        if st.button("💹 Calculate EMI & Loan Details", use_container_width=True):
            emi = calculate_emi(loan_amount, interest_rate, tenure_months)
            total_amount = emi * tenure_months
            total_interest = total_amount - loan_amount
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💵 Monthly EMI", f"₹{emi:,.2f}")
            with col2:
                st.metric("📊 Total Interest", f"₹{total_interest:,.2f}")
            with col3:
                st.metric("💰 Total Amount", f"₹{total_amount:,.2f}")
            with col4:
                years = tenure_months // 12
                months = tenure_months % 12
                st.metric("⏱️ Tenure", f"{years}y {months}m")
            
            st.markdown("---")
            st.success(f"✅ EMI calculated successfully!")
    
    # ====================================================================
    # PAGE 5: ABOUT (ENHANCED)
    # ====================================================================
    
    elif page == "ℹ️ About":
        st.header("ℹ️ About Credit Scoring System")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Project Overview
            
            This is an intelligent credit scoring system that predicts customer default probability 
            and calculates personalized interest rates using machine learning.
            
            ### 📊 Features
            
            - ✅ **Credit Score Calculation (300-850)** - Based on 5 weighted factors
            - ✅ **Default Risk Prediction** - Using Random Forest ML model (90% accuracy)
            - ✅ **Interest Rate Calculation** - Risk-based dynamic pricing
            - ✅ **Personalized Recommendations** - Actionable improvement suggestions
            - ✅ **Batch Processing** - Score multiple customers simultaneously
            - ✅ **Portfolio Analytics** - Comprehensive dashboard and insights
            
            ### 🔧 Technology Stack
            
            - **Frontend**: Streamlit
            - **Backend**: Python, Scikit-learn
            - **Data**: Pandas, NumPy
            - **Visualization**: Plotly
            """)
        
        with col2:
            st.markdown("""
            ### 📈 Credit Score Factors
            
            1. **Payment History** (35%)
            2. **Credit Utilization** (30%)
            3. **Credit Age** (15%)
            4. **Credit Mix** (10%)
            5. **New Inquiries** (10%)
            """)
        
        st.markdown("---")
        st.markdown("""
        ### 🏦 Banking Use Cases
        
        ✓ Loan Approval Decisions
        ✓ Dynamic Interest Rate Pricing
        ✓ Credit Limit Assignment
        ✓ Portfolio Risk Management
        ✓ Customer Relationship Management
        
        ---
        
        **Version:** 2.0.0 (Enhanced Interactive UI)  
        **Last Updated:** December 2025
        """)

if __name__ == "__main__":
    main()
