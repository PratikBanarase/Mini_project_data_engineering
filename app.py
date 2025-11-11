import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# Set page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-approved {
        background-color: #d4edda;
        color: #155724;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .prediction-rejected {
        background-color: #f8d7da;
        color: #721c24;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    .sidebar-profile {
        text-align: center;
        padding: 20px 0;
    }
    .sidebar-profile img {
        border-radius: 50%;
        width: 100px;
        height: 100px;
        object-fit: cover;
        margin-bottom: 15px;
    }
    .sidebar-link {
        display: flex;
        align-items: center;
        padding: 8px 0;
        text-decoration: none;
        color: inherit;
    }
    .sidebar-link:hover {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model with error handling"""
    try:
        with open('loan_approval_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run 'train_model.py' first to create the model.")
        st.info("""
        **To fix this issue:**
        1. Run `python train_model.py` to create the model file
        2. Make sure 'loan_approval_model.pkl' is in the same directory as this app
        3. Restart the Streamlit app
        """)
        return None

def load_feature_info():
    """Load feature information"""
    try:
        with open('loan_approval_model.pkl', 'rb') as f:
            feature_info = pickle.load(f)
        return feature_info
    except FileNotFoundError:
        return None

def create_sample_input():
    """Create a sample input form"""
    st.markdown('<div class="feature-input">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("👤 Age", min_value=18, max_value=70, value=35, 
                       help="Age of the applicant")
        income = st.number_input("💰 Annual Income ($)", min_value=20000, 
                               max_value=150000, value=50000, step=1000,
                               help="Annual income in dollars")
        credit_score = st.slider("📊 Credit Score", min_value=300, max_value=850, 
                               value=650, help="Credit score (300-850)")
    
    with col2:
        loan_amount = st.number_input("💵 Loan Amount ($)", min_value=5000, 
                                    max_value=100000, value=25000, step=1000,
                                    help="Requested loan amount")
        employment_years = st.slider("💼 Years of Employment", min_value=0, 
                                   max_value=40, value=5, 
                                   help="Number of years employed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return age, income, credit_score, loan_amount, employment_years

def predict_loan_approval(model, input_features):
    """Make prediction and return results"""
    try:
        # Convert to numpy array and reshape for single prediction
        features_array = np.array(input_features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def create_sidebar():
    """Create sidebar with profile information"""
    with st.sidebar:
        st.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
        
        # Profile header
        st.markdown("### 👨‍💻 Python Devloper")
        st.markdown("---")
        
        # Name
        st.markdown("**Pratik Banarase**")
        st.markdown("*Data Scientist*")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Contact Information
        st.markdown("### 📞 Contact Info")
        
        # Email
        st.markdown(
            f'<a href="pratikbanarse8@gmail.com" class="sidebar-link">'
            f'📧 john.doe@email.com</a>', 
            unsafe_allow_html=True
        )
        
        # LinkedIn
        st.markdown(
            f'<a href="https://www.linkedin.com/in/pratikbanarse/" target="_blank" class="sidebar-link">'
            f'🔗 LinkedIn: johndoe</a>', 
            unsafe_allow_html=True
        )
        
        # GitHub
        st.markdown(
            f'<a href="https://github.com/PratikBanarase" target="_blank" class="sidebar-link">'
            f'🐙 GitHub: johndoe</a>', 
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # App Info
        st.markdown("### ℹ️ App Information")
        st.markdown("""
        This loan approval predictor uses machine learning to analyze applicant data and predict loan approval chances.
        
        **Features analyzed:**
        - Age
        - Annual Income
        - Credit Score
        - Loan Amount
        - Employment History
        """)
        
        # Model info
        st.markdown("### 🤖 Model Info")
        st.markdown("""
        **Algorithm:** Random Forest Classifier
        **Accuracy:** ~95% (on test data)
        **Training Samples:** 1,000
        """)

def main():
    # Create sidebar
    create_sidebar()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Header
        st.markdown('<h1 class="main-header">🏦 Loan Approval Predictor</h1>', 
                   unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    feature_info = load_feature_info()
    
    if model is None:
        return
    
    # Display feature descriptions if available
    if feature_info:
        with st.expander("📋 Feature Descriptions"):
            for feature, description in feature_info['feature_descriptions'].items():
                st.write(f"**{feature}**: {description}")
    
    # Input section
    st.markdown("### 📝 Applicant Information")
    age, income, credit_score, loan_amount, employment_years = create_sample_input()
    
    # Create feature array
    input_features = [age, income, credit_score, loan_amount, employment_years]
    
    # Display current input values
    st.markdown("### 📊 Current Input Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Age", age)
    with col2:
        st.metric("Income", f"${income:,}")
    with col3:
        st.metric("Credit Score", credit_score)
    with col4:
        st.metric("Loan Amount", f"${loan_amount:,}")
    with col5:
        st.metric("Employment Years", employment_years)
    
    # Calculate debt-to-income ratio
    debt_to_income = (loan_amount / income) * 100 if income > 0 else 0
    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
    
    # Prediction button
    if st.button("🚀 Predict Loan Approval", type="primary", use_container_width=True):
        with st.spinner("Analyzing application..."):
            prediction, probability = predict_loan_approval(model, input_features)
            
            if prediction is not None:
                st.markdown("### 📈 Prediction Results")
                
                # Display prediction
                if prediction == 1:
                    st.markdown('<div class="prediction-approved">✅ LOAN APPROVED!</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-rejected">❌ LOAN REJECTED</div>', 
                               unsafe_allow_html=True)
                
                # Display probabilities
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Probability of Approval", 
                             f"{probability[1]*100:.2f}%")
                with col2:
                    st.metric("Probability of Rejection", 
                             f"{probability[0]*100:.2f}%")
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### 🔍 Feature Importance")
                    feature_names = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Employment Years']
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature')['Importance'])
    
    # Add some information
    with st.expander("💡 How to improve your chances"):
        st.markdown("""
        - **Maintain a good credit score** (650+)
        - **Stable employment history** (2+ years)
        - **Reasonable debt-to-income ratio** (<50%)
        - **Adequate income** for the requested loan amount
        """)

if __name__ == "__main__":
    main()

