# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model, scaler, and encoders
with open("loan_approval_model.pkl", "rb") as f:
    model, scaler, encoders = pickle.load(f)

# ---- HEADER BAR SECTION ----
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

# Header with custom styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sidebar-link {
        display: flex;
        align-items: center;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .sidebar-link:hover {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)

# ---- SIDEBAR SECTION ----
with st.sidebar:
    st.header("👤 Python Devloper")
    
    # Developer Name
    st.subheader("Pratik Banarase")
    st.write("Data Scientist | ML Engineer")
    
    st.markdown("---")
    
    # Social Links
    st.subheader("🔗 Connect with Me")
    
    # LinkedIn Link
    linkedin_url = "https://www.linkedin.com/in/pratikbanarse/"
    st.markdown(f"""
    <div class="sidebar-link">
        <a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="20" height="20" style="margin-right: 10px;">
            <strong>LinkedIn</strong>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # GitHub Link
    github_url = "https://github.com/PratikBanarase"
    st.markdown(f"""
    <div class="sidebar-link">
        <a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="20" height="20" style="margin-right: 10px;">
            <strong>GitHub</strong>
        </a>
    </div>
    """, unsafe_allow_html=True)
    


    st.markdown("---")
    
    st.header("ℹ️ About This App")
    st.markdown("""
    This application predicts whether a loan application will be **approved** or **rejected** 
    based on the applicant's information.
    
    ### How to use:
    1. Fill in all the applicant details
    2. Click the **'Predict Loan Approval'** button
    3. View the prediction result
    
    ### Features used:
    - Personal details (Gender, Marital status)
    - Financial information (Income, Loan amount)
    - Employment details
    - Credit history
    - Property area
    """)
    
    st.markdown("---")
    st.subheader("📊 Model Information")
    st.write(f"Model loaded: {type(model).__name__}")
    st.write("Scaler and encoders loaded successfully")
    
    st.markdown("---")
    st.subheader("🔍 Data Summary")
    if st.button("Show Feature Summary"):
        feature_info = {
            'Feature': ['Applicant Income', 'Coapplicant Income', 'Loan Amount', 'Credit History'],
            'Description': ['Primary income source', 'Secondary income source', 'Requested loan amount', 'Creditworthiness (0=No, 1=Yes)']
        }
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True)

# ---- MAIN CONTENT AREA ----
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Applicant Information Form")
    st.markdown("Fill in the applicant details below to predict loan approval:")
    
    # Create two columns for better organization
    col1a, col1b = st.columns(2)
    
    with col1a:
        st.markdown("**Personal Details**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    with col1b:
        st.markdown("**Financial Details**")
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100, help="Monthly income of the applicant")
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100, help="Monthly income of co-applicant")
        loan_amount = st.number_input("Loan Amount", min_value=0, step=10, help="Requested loan amount")
        loan_amount_term = st.selectbox("Loan Amount Term (in months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
        credit_history = st.selectbox("Credit History", [0, 1], format_func=lambda x: "Good" if x == 1 else "Poor", help="1 = Good credit history, 0 = Poor credit history")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    st.subheader("📋 Input Summary")
    st.markdown("**Current Input Values:**")
    
    # Display current input values
    input_summary = {
        "Field": ["Gender", "Married", "Dependents", "Education", "Self Employed", 
                 "Applicant Income", "Coapplicant Income", "Loan Amount", 
                 "Loan Term", "Credit History", "Property Area"],
        "Value": [gender, married, dependents, education, self_employed,
                 f"${applicant_income:,}", f"${coapplicant_income:,}", f"${loan_amount:,}",
                 f"{loan_amount_term} months", "Good" if credit_history == 1 else "Poor", property_area]
    }
    
    summary_df = pd.DataFrame(input_summary)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # Quick statistics
    st.markdown("**Quick Stats:**")
    total_income = applicant_income + coapplicant_income
    loan_to_income = (loan_amount / total_income * 100) if total_income > 0 else 0
    
    st.metric("Total Monthly Income", f"${total_income:,}")
    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.1f}%")

# ---- Create Input DataFrame ----
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}

df_input = pd.DataFrame(input_dict)

# Apply label encoding (must match training encoders)
for col in df_input.columns:
    if col in encoders:
        df_input[col] = encoders[col].transform(df_input[col].astype(str))

# Scale numerical features
df_input_scaled = scaler.transform(df_input)

# ---- PREDICTION SECTION ----
st.markdown("---")
st.subheader("🎯 Prediction Result")

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    if st.button("🚀 Predict Loan Approval", use_container_width=True):
        prediction = model.predict(df_input_scaled)[0]
        prediction_proba = model.predict_proba(df_input_scaled)[0]
        
        with pred_col2:
            if prediction == 1:
                st.success(f"## ✅ Loan Approved!")
                st.balloons()
            else:
                st.error(f"## ❌ Loan Rejected")
            
            # Show confidence score
            confidence = prediction_proba[prediction] * 100
            st.metric("Confidence Level", f"{confidence:.1f}%")
            
            # Show probability breakdown
            st.progress(int(confidence))
            st.write(f"Approval Probability: {prediction_proba[1]:.2%}")
            st.write(f"Rejection Probability: {prediction_proba[0]:.2%}")

# ---- FOOTER ----
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Loan Approval Prediction System • Built with Streamlit</div>", 
    unsafe_allow_html=True
)
