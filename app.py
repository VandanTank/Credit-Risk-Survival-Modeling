import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib 
import os 

# --- Inject Custom CSS for Better User Experience ---
st.markdown("""
<style>
/* Targets the dropdown/select box buttons to show a pointer (hand) cursor on hover */
.stSelectbox div[role="button"] {
    cursor: pointer !important;
}

/* Targets the slider handles for better interaction feedback */
.stSlider div[data-testid="stThumbValue"] {
    cursor: pointer !important;
}

/* Make the prediction button more engaging */
.stButton>button {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# --- 1. LOAD MODEL (Ensure Files are Loaded via Absolute Path) ---

# Get the absolute path to the directory containing this script
BASE_DIR = pathlib.Path(__file__).parent.resolve()

# Define the full, absolute paths to the model files
MODEL_PATH = BASE_DIR / 'final_survival_model.joblib'
FEATURES_PATH = BASE_DIR / 'model_feature_names.joblib'

@st.cache_resource
def load_model():
    """Loads the trained model and feature names from the script's absolute directory."""
    
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(FEATURES_PATH):
        st.error(f"ERROR: Model/feature files not found.")
        st.error(f"Please ensure 'final_survival_model.joblib' and 'model_feature_names.joblib' are in the root folder: {BASE_DIR}")
        return None, None
    
    try:
        model = joblib.load(MODEL_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, features
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        return None, None

model, feature_names = load_model()

# --- 2. SET UP THE USER INTERFACE (UI) ---
st.set_page_config(layout="wide")
st.title('🏦 Credit Risk Survival Analysis')
st.subheader('Predicting when a loan is likely to default using a Random Survival Forest')

st.sidebar.header('Borrower Information')

# --- 3. CREATE INPUT WIDGETS ---
def user_inputs():
    """Creates sidebar widgets and returns user inputs as a dictionary."""
    int_rate = st.sidebar.slider('Interest Rate (%)', min_value=5.0, max_value=35.0, value=13.99, step=0.1)
    term = st.sidebar.selectbox('Loan Term (Months)', options=[36, 60], index=0)
    grade = st.sidebar.selectbox('Loan Grade', options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2)
    dti = st.sidebar.slider('Debt-to-Income Ratio (DTI)', min_value=0.0, max_value=50.0, value=15.0, step=0.1)
    loan_amnt = st.sidebar.slider('Loan Amount ($)', min_value=1000, max_value=40000, value=10000, step=100)
    home_ownership = st.sidebar.selectbox('Home Ownership', options=['RENT', 'MORTGAGE', 'OWN', 'ANY', 'OTHER', 'NONE'])

    # Include default values for features not exposed in the UI (based on SHAP analysis)
    data = {
        'int_rate': int_rate,
        'term': term,
        'grade': grade,
        'dti': dti,
        'loan_amnt': loan_amnt,
        'home_ownership': home_ownership,
        'emp_length': 5.0, 
        'annual_inc': 65000.0,
    }
    return data

input_data = user_inputs()

# --- 4. TRANSFORM INPUTS FOR THE MODEL ---
def preprocess_input(input_data, feature_list):
    """Converts user input into the one-hot encoded format the model expects."""
    
    # Initialize the input DataFrame with zeros, matching all 30+ expected features
    input_df = pd.DataFrame(columns=feature_list)
    input_df.loc[0] = 0.0 
    
    # Fill in the numerical features
    simple_features = ['int_rate', 'term', 'dti', 'loan_amnt', 'emp_length', 'annual_inc']
    for feature in simple_features:
        if feature in input_df.columns:
            input_df.loc[0, feature] = input_data[feature]

    # Handle One-Hot-Encoded features
    grade_col = f"grade_{input_data['grade']}"
    if grade_col in input_df.columns:
        input_df.loc[0, grade_col] = 1.0
    
    home_col = f"home_ownership_{input_data['home_ownership']}"
    if home_col in input_df.columns:
        input_df.loc[0, home_col] = 1.0
    
    # Ensure all columns are in the correct order for the model
    input_df = input_df[feature_list]
    return input_df

# --- 5. RUN THE PREDICTION ---
if st.sidebar.button('Calculate Risk Profile'):
    
    if model is not None and feature_names is not None:
        
        input_df = preprocess_input(input_data, feature_names)
        
        # Get the survival prediction functions
        pred_curves = model.predict_survival_function(input_df)
        
        curve = pred_curves[0]
        plot_df = pd.DataFrame({
            'Loan Duration (Months)': curve.x,
            'Survival Probability': curve.y
        })
        
        st.subheader(f"Predicted Survival Curve for this Borrower")
        st.line_chart(plot_df, x='Loan Duration (Months)', y='Survival Probability')
        st.write("This chart shows the model's predicted probability that the borrower will *not* have defaulted by a given month.")
        
        st.subheader("Model Input (After One-Hot Encoding):")
        st.dataframe(input_df)

    else:
        st.error("Model is not loaded. Please check the error message at the top.")

else:
    st.info("Adjust the parameters on the left and click 'Calculate Risk Profile' to see the prediction.")

# --- 6. Add Project Credit to Footer ---
st.markdown(
    """
    <br><br><br>
    <div style='text-align: right; font-size: small; color: grey;'>
    Project by: Vandan Tank
    </div>
    """,
    unsafe_allow_html=True
)