# ============================================================================
# CREDIT SCORING STREAMLIT APPLICATION - ENHANCED INTERACTIVE VERSION
# Interactive Web App for Credit Scoring and Default Risk Prediction
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings


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

        .stApp {
            background: #F5F7FA;
        }

        h1 {
            color: #1F3B57 !important;
            text-align: center;
            font-size: 2.4em;
            font-weight: 800;
            margin-bottom: 0.4rem;
            letter-spacing: 0.5px;
        }

        h2 {
            color: #1F3B57 !important;
            border-bottom: 1px solid #D3D9E4 !important;
            padding-bottom: 0.4rem;
            font-size: 1.5em;
            font-weight: 700;
        }

        h3 {
            color: #274766 !important;
            font-weight: 600;
        }

    /* Metrics - simple cards */
    .stMetric {
        background: #FFFFFF;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        border: 1px solid #E1E5EE;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }

    .stMetric:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(15, 23, 42, 0.12);
    }

    .stMetric label {
        color: #4B5563 !important;
        font-weight: 600;
        font-size: 13px;
    }

    .stMetric [data-testid="metricDeltaContainer"] {
        color: #2F6F3E !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"], 
    [data-testid="stMetricValue"] > div {
        color: #1F3B57 !important;
    }



    /* Sidebar - flat, formal blue */
    [data-testid="stSidebar"] {
        background: #233649;
        border-right: 1px solid #D3D9E4;
    }

    [data-testid="stSidebar"] * {
        color: #F9FAFB !important;
    }

    /* Radio buttons in sidebar */
    .stRadio > div > label {
        color: #F9FAFB !important;
        font-weight: 500;
        font-size: 14px;
        padding: 6px 10px;
        border-radius: 6px;
        margin: 3px 0;
        transition: background-color 0.15s ease;
    }

    .stRadio > div > label:hover {
        background-color: rgba(255, 255, 255, 0.10);
    }

    /* Buttons - toned-down blue */
    /* Buttons - toned-down blue */
    .stButton > button, .stDownloadButton > button {
        background: #265A88;
        color: #FFFFFF !important;
        border: 1px solid #1E476B;
        padding: 9px 22px;
        border-radius: 18px;
        font-weight: 600;
        font-size: 13px;
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.1s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        background: #1E476B;
        box-shadow: 0 3px 8px rgba(30, 71, 107, 0.22);
        transform: translateY(-1px);
    }

    .stButton > button:active, .stDownloadButton > button:active {
        transform: translateY(0);
        box-shadow: none;
    }

    /* Success / warning / danger boxes - softer colors */
    .success-box {
        background: #EDF6EF;
        border: 1px solid #2F6F3E;
        border-left: 4px solid #2F6F3E;
        padding: 10px;
        border-radius: 6px;
        color: #23472D;
        font-weight: 500;
    }

    .warning-box {
        background: #FFF6E5;
        border: 1px solid #D2961A;
        border-left: 4px solid #D2961A;
        padding: 10px;
        border-radius: 6px;
        color: #725535;
        font-weight: 500;
    }

    .danger-box {
        background: #FDEBEC;
        border: 1px solid #B53A3A;
        border-left: 4px solid #B53A3A;
        padding: 10px;
        border-radius: 6px;
        color: #7F2525;
        font-weight: 500;
    }

    /* Inputs */
    .stNumberInput input, .stSelectbox select, .stSlider div {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        color: #111827 !important;
        border-radius: 8px;
        padding: 8px 10px;
    }

    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: #265A88 !important;
        box-shadow: 0 0 0 1px #265A88 !important;
    }

    /* File uploader */
    .stFileUploader {
        background: #FFFFFF;
        border: 1px dashed #C4CCDA;
        border-radius: 10px;
        padding: 14px;
    }

    /* Dataframe container */
    .stDataFrame {
        background-color: #FFFFFF !important;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    }

    /* Info / success messages */
    .stInfo {
        background: #E6EEF9;
        border-left: 4px solid #265A88 !important;
        border-radius: 8px;
    }

    .stSuccess {
        background: #EDF6EF;
        border-left: 4px solid #2F6F3E !important;
        border-radius: 8px;
    }

    /* Dividers */
    hr {
        border: none;
        border-top: 1px solid #D3D9E4 !important;
        margin: 0.8rem 0;
    }

    /* General text */
    .markdown-text-container, .stMarkdown,
        html, body, .stApp,
        h1, h2, h3, h4, h5, h6,
        label, p, span {
            color: #000000 !important;
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
    
    # Payment History (35%) - Target Max: 192.5 points
    # Base 550 * 0.35 = 192.5
    payment_score = 550
    if row['Num_of_Delayed_Payment'] > 0:
        payment_score -= (row['Num_of_Delayed_Payment'] * 50)
    if row['Delay_from_due_date'] > 30:
        payment_score -= min(100, row['Delay_from_due_date'] * 2)
    if row['Payment_of_Min_Amount'] == 1:
        payment_score -= 100
    payment_score = max(0, payment_score)
    score += payment_score * 0.35
    
    # Credit Utilization (30%) - Target Max: 165 points
    # Base 550 * 0.30 = 165
    utilization_score = 550
    utilization = row['Credit_Utilization_Ratio']
    if utilization > 30:
        utilization_score -= (utilization - 30) * 10
    utilization_score = max(0, utilization_score)
    score += utilization_score * 0.30
    
    # Credit Age (15%) - Target Max: 82.5 points
    # Base 550 * 0.15 = 82.5
    age_score = 550
    age_years = row['Credit_History_Age_Years']
    if age_years < 2:
        age_score = 250
    elif age_years < 5:
        age_score = 400
    elif age_years >= 10:
        age_score = 550
    else:
        age_score = 450
    score += age_score * 0.15
    
    # Credit Mix (10%) - Target Max: 55 points
    # Base 550 * 0.10 = 55
    mix_score = 550 if row['Credit_Mix'] == 'Good' else 350
    score += mix_score * 0.10
    
    # New Inquiries (10%) - Target Max: 55 points
    # Base 550 * 0.10 = 55
    inquiry_score = 550
    if row['Num_Credit_Inquiries'] > 5:
        inquiry_score -= (row['Num_Credit_Inquiries'] - 5) * 50
    if row['Num_Credit_Inquiries'] > 12:
        inquiry_score = 0
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
            "📉 Analytics Dashboard 💡 Recommendations",
            "📈 Batch Scoring", 
            "💰 Interest Rate Calculator",
            "ℹ️ About"
        ],
        label_visibility="collapsed"
    )
    
    
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
            
            # Initialize session state for batch results
            if 'batch_df' not in st.session_state:
                st.session_state.batch_df = None
            
            # Clear session state if a new file is uploaded
            if st.session_state.batch_df is not None:
                # Simple check: if rows differ, it might be a new file or just re-run. 
                # Ideally we track file ID, but for now let's rely on the button to overwrite.
                pass

            
            st.markdown("---")
            
            if st.button("🚀 Score All Customers", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("🔄 Processing customers..."):
                    # Process the local df but save to session_state
                    df['Credit_Score'] = df.apply(calculate_credit_score, axis=1)
                    progress_bar.progress(25)
                    status_text.info("✓ Credit scores calculated")
                    
                    df['Credit_Tier'] = df['Credit_Score'].apply(lambda x: get_credit_tier(x)[0])
                    progress_bar.progress(50)
                    status_text.info("✓ Credit tiers assigned")
                    
                    df['Default_Probability'] = df.apply(
                        lambda row: calculate_default_probability(row, row['Credit_Score']), axis=1
                    )
                    progress_bar.progress(75)
                    status_text.info("✓ Default probabilities calculated")

                    # Generate Suggestions
                    def get_formatted_suggestions(row):
                        suggestions = generate_suggestions(row, row['Credit_Score'], row['Credit_Tier'])
                        if not suggestions:
                            return "No improvements needed"
                        return " | ".join([f"{s['category']} ({s['priority']}): {s['suggestion']}" for s in suggestions])

                    df['Recommendations'] = df.apply(get_formatted_suggestions, axis=1)
                    progress_bar.progress(100)
                    status_text.success(f"✅ Successfully scored {len(df)} customers!")
                    
                    # Store in session state
                    st.session_state.batch_df = df
            
            # Display results if they exist in session state
            if st.session_state.batch_df is not None:
                df_results = st.session_state.batch_df
                
                st.markdown("---")
                st.subheader("📊 Sample Results (First 10)")
                st.dataframe(
                    df_results[['Customer_ID', 'Name', 'Credit_Score', 'Credit_Tier', 'Default_Probability', 'Recommendations']].head(10),
                    use_container_width=True
                )
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_score = df_results['Credit_Score'].mean()
                    st.metric("📈 Avg Score", f"{avg_score:.0f}", f"{avg_score-500:.0f}")
                with col2:
                    avg_default = df_results['Default_Probability'].mean()
                    st.metric("⚠️ Avg Default Risk", f"{avg_default:.2%}")
                with col3:
                    excellent_count = len(df_results[df_results['Credit_Tier'] == 'Excellent'])
                    st.metric("🟢 Excellent", excellent_count)
                with col4:
                    poor_count = len(df_results[df_results['Credit_Tier'] == 'Poor'])
                    st.metric("🔴 Poor", poor_count)
                
                st.markdown("---")
                
                # Recommendations Info
                st.markdown("### 💡 AI Recommendations Generated")
                st.info("""
                    **Note:** The system has analyzed each customer's profile and generated specific actions to improve their credit score. 
                    These personalized suggestions are included in the **'Recommendations'** column of the downloadable CSV file below.
                """)

                # Download results
                csv = df_results[['Customer_ID', 'Name', 'Credit_Score', 'Credit_Tier', 'Default_Probability', 'Recommendations']].to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Results (with Recommendations)",
                    data=csv,
                    file_name="credit_scores_with_recommendations.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Interactive Recommendations Section
                st.markdown("---")
                st.subheader("🔍 Interactive Analysis")
                st.markdown("Select a customer below to view detailed insights and specific improvement suggestions.")
                
                selected_customer = st.selectbox(
                    "Select Customer to Analyze",
                    df_results['Name'].unique(),
                    key="batch_customer_select"
                )
                
                if selected_customer:
                    customer_row = df_results[df_results['Name'] == selected_customer].iloc[0]
                    score = customer_row['Credit_Score']
                    tier = customer_row['Credit_Tier']
                    
                    # Logic to get emoji (re-using helper or just mapping if needed, but we have the function)
                    # We can re-call get_credit_tier to get the emoji, as the DataFrame only stored the tier name
                    _, emoji = get_credit_tier(score)
                    
                    # Customer Card
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("👤 Customer", selected_customer)
                    with col2:
                        st.metric("⭐ Score", f"{score}/850")
                    with col3:
                        st.metric("📊 Tier", f"{emoji} {tier}")
                    
                    st.markdown("### ✨ Improvement Suggestions")
                    
                    # We can re-generate suggestions for display or parse the string. 
                    # Re-generating is cleaner for UI (list format) vs parsing the string.
                    suggestions = generate_suggestions(customer_row, score, tier)
                    
                    if suggestions:
                        for i, sugg in enumerate(suggestions, 1):
                            with st.expander(f"{i}. {sugg['category']} ({sugg['priority']})", expanded=True):
                                st.write(sugg['suggestion'])
                                st.markdown(f"**💪 Potential Impact:** `+{sugg['impact']} improvement`")
                    else:
                        st.success("✅ No improvements needed! This customer has excellent credit practices.")

                
    
    # ====================================================================
    # PAGE 2: ANALYTICS DASHBOARD (ENHANCED)
    # ====================================================================
    
    elif page == "📉 Analytics Dashboard 💡 Recommendations":
        st.header("dataset Analytics Dashboard")
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
