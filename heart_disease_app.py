import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positive {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    .negative {
        background: linear-gradient(135deg, #26de81, #20bf6b);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

def preprocess_input(data, scaler, columns):
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    
    for col in columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    df_encoded = df_encoded[columns]
    numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])
    
    return df_encoded

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#e74c3c"},
            'steps': [
                {'range': [0, 30], 'color': '#d5f5e3'},
                {'range': [30, 70], 'color': '#fef9e7'},
                {'range': [70, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_risk_factors_chart(input_data):
    factors = {
        'Age': input_data['Age'] / 100,
        'Resting BP': input_data['RestingBP'] / 200,
        'Cholesterol': input_data['Cholesterol'] / 400,
        'Max HR (inv)': 1 - (input_data['MaxHR'] / 220),
        'Oldpeak': input_data['Oldpeak'] / 6,
        'Fasting BS': input_data['FastingBS']
    }
    
    df_factors = pd.DataFrame({
        'Factor': list(factors.keys()),
        'Value': list(factors.values())
    })
    
    fig = px.bar(df_factors, x='Value', y='Factor', orientation='h',
                 color='Value', color_continuous_scale='RdYlGn_r',
                 range_x=[0, 1])
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
                      title="Normalized Risk Factors")
    return fig

# Load model
try:
    model, scaler, columns = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Header
st.markdown('<div class="main-header">❤️ Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Machine Learning Model for Cardiovascular Risk Assessment</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.info("""
    **Model**: K-Nearest Neighbors (KNN)
    **Features**: 15 clinical parameters
    **Training Samples**: 734 patients
    """)
    
    st.markdown("### 🏥 About")
    st.markdown("""
    This application uses machine learning to predict heart disease likelihood.
    
    **Note**: For educational purposes only. Consult a healthcare professional.
    """)
    
    st.markdown("### 📖 Feature Descriptions")
    with st.expander("Click to view"):
        st.markdown("""
        - **Age**: Patient age in years
        - **Sex**: Biological sex (M/F)
        - **Chest Pain Type**: ATA, NAP, ASY, TA
        - **Resting BP**: Resting blood pressure (mm Hg)
        - **Cholesterol**: Serum cholesterol (mg/dl)
        - **Fasting BS**: Fasting blood sugar > 120 mg/dl
        - **Resting ECG**: Normal, ST, LVH
        - **Max HR**: Maximum heart rate achieved
        - **Exercise Angina**: Exercise-induced angina (Y/N)
        - **Oldpeak**: ST depression induced by exercise
        - **ST Slope**: Up, Flat, Down
        """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Patient Information")
    
    with st.form("patient_form"):
        st.markdown("**Demographics**")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            age = st.slider("Age", 20, 100, 50)
        with col_d2:
            sex = st.selectbox("Sex", ["M", "F"])
        
        st.markdown("**Clinical Measurements**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            resting_bp = st.slider("Resting BP", 80, 200, 130)
        with col_c2:
            cholesterol = st.slider("Cholesterol", 100, 600, 240)
        with col_c3:
            max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        
        st.markdown("**Symptoms & Conditions**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        with col_s2:
            exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
            oldpeak = st.slider("Oldpeak", -3.0, 6.0, 1.0, 0.1)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
        
        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

with col2:
    st.markdown("### 📈 Analysis Results")
    
    if submitted and model_loaded:
        input_data = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingECG': resting_ecg,
            'ExerciseAngina': exercise_angina,
            'ST_Slope': st_slope
        }
        
        processed_data = preprocess_input(input_data, scaler, columns)
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        risk_probability = probabilities[1]
        
        if prediction == 1:
            st.markdown(
                f'<div class="prediction-box positive">⚠️ HIGH RISK<br>Heart Disease Detected<br>Confidence: {risk_probability*100:.1f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="prediction-box negative">✅ LOW RISK<br>No Heart Disease Detected<br>Confidence: {(1-risk_probability)*100:.1f}%</div>',
                unsafe_allow_html=True
            )
        
        st.plotly_chart(create_gauge_chart(risk_probability), use_container_width=True)
        st.plotly_chart(create_risk_factors_chart(input_data), use_container_width=True)
        
        st.markdown("### 📋 Detailed Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Age", f"{age} years")
        with col_m2:
            st.metric("Resting BP", f"{resting_bp} mmHg")
        with col_m3:
            st.metric("Cholesterol", f"{cholesterol} mg/dl")
        with col_m4:
            st.metric("Max HR", f"{max_hr} bpm")
            
    elif not model_loaded:
        st.error("⚠️ Model not loaded. Ensure .pkl files are present.")
    else:
        st.info("👈 Fill in patient information and click 'Predict' to see results.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #95a5a6;'>"
    "Made with ❤️ | For educational purposes only"
    "</div>",
    unsafe_allow_html=True
)