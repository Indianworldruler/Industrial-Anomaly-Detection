import streamlit as st
import pandas as pd
import joblib

# Set the Streamlit page configuration
st.set_page_config(page_title="Industrial Anomaly Detection", page_icon="🏭", layout="wide")

# Add compact application styling
st.markdown("""
<style>
.main {background: linear-gradient(135deg, #f7fbff 0%, #eef7ff 50%, #f9f5ff 100%);}
.block-container {max-width: 1150px; padding-top: 1.3rem; padding-bottom: 1.5rem;}
.hero {padding: 22px 28px; border-radius: 18px; background: linear-gradient(135deg, #1769aa, #7048b8); color: white; margin-bottom: 14px; box-shadow: 0 6px 18px rgba(40,70,120,0.15);}
.hero h1 {margin: 0; font-size: 30px;}
.hero p {margin: 5px 0 0 0; font-size: 14px;}
.section {background: white; padding: 14px 18px; border-radius: 14px; margin-bottom: 10px; box-shadow: 0 3px 12px rgba(40,70,120,0.06);}
.section-title {font-size: 17px; font-weight: 700; margin-bottom: 1px;}
.section-note {color: #667085; font-size: 12px; margin-bottom: 7px;}
.result-normal {padding: 18px 20px; border-radius: 14px; background: #e9f8ef; border: 1px solid #9ed8b3; color: #176b37; font-size: 20px; font-weight: 700;}
.result-anomaly {padding: 18px 20px; border-radius: 14px; background: #fff0f0; border: 1px solid #efaaaa; color: #a12626; font-size: 20px; font-weight: 700;}
.info-box {padding: 11px 14px; border-radius: 10px; background: #f4f7fb; border-left: 4px solid #1769aa; color: #344054; font-size: 13px;}
.footer {text-align: center; color: #667085; font-size: 12px; padding-top: 8px;}
div[data-testid="stTextInput"] {margin-bottom: -5px;}
</style>
""", unsafe_allow_html=True)

# Load the trained machine-learning model
@st.cache_resource
def load_model():
    return joblib.load("industrial_anomaly_model.pkl")

package = load_model()
model = package["model"]
threshold = package["threshold"]
features = package["features"]

# Store the two example test cases
# Test 1 is the first normal observation from the compiled dataset
test1 = {
    "Accelerometer 1 RMS": "0.250072",
    "Accelerometer 2 RMS": "0.293697",
    "Current": "2.41322",
    "Pressure": "0.382638",
    "Temperature": "86.3671",
    "Thermocouple": "28.7819",
    "Voltage": "227.064",
    "Volume Flow Rate RMS": "126.309"
}

# Test 2 is the first anomaly observation from the compiled dataset
test2 = {
    "Accelerometer 1 RMS": "0.24863",
    "Accelerometer 2 RMS": "0.286879",
    "Current": "2.28049",
    "Pressure": "0.382638",
    "Temperature": "85.9805",
    "Thermocouple": "28.7235",
    "Voltage": "246.388",
    "Volume Flow Rate RMS": "127.389"
}

# Create input values in session state
defaults = {**test1}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Load Test 1 values into the input fields
def load_test1():
    for key, value in test1.items():
        st.session_state[key] = value

# Load Test 2 values into the input fields
def load_test2():
    for key, value in test2.items():
        st.session_state[key] = value

# Display the application header
st.markdown("""
<div class="hero">
    <h1>🏭 Industrial Anomaly Detection</h1>
    <p>Machine-learning based monitoring of industrial operating conditions.</p>
</div>
""", unsafe_allow_html=True)

# Display a compact explanation and test controls
st.markdown("""
<div class="section">
    <div class="section-title">Quick Test</div>
    <div class="section-note">Use a sample case or enter your own sensor readings below.</div>
</div>
""", unsafe_allow_html=True)

test_col1, test_col2, test_col3 = st.columns([1, 1, 2])
with test_col1:
    st.button("🟢 Load Test 1", use_container_width=True, on_click=load_test1)
with test_col2:
    st.button("🔴 Load Test 2", use_container_width=True, on_click=load_test2)
with test_col3:
    st.caption("Test 1 = Normal example  •  Test 2 = Anomaly example")

# Display the vibration inputs
st.markdown('<div class="section"><div class="section-title">🔊 Vibration</div><div class="section-note">Machine vibration measurements.</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    accelerometer1 = st.text_input("Accelerometer 1 RMS", key="Accelerometer 1 RMS", placeholder="Enter a number, e.g. 0.250072")
with col2:
    accelerometer2 = st.text_input("Accelerometer 2 RMS", key="Accelerometer 2 RMS", placeholder="Enter a number, e.g. 0.293697")
st.markdown("</div>", unsafe_allow_html=True)

# Display the electrical inputs
st.markdown('<div class="section"><div class="section-title">⚡ Electrical</div><div class="section-note">Current and voltage measurements.</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    current = st.text_input("Current", key="Current", placeholder="Enter a number, e.g. 2.41322")
with col2:
    voltage = st.text_input("Voltage", key="Voltage", placeholder="Enter a number, e.g. 227.064")
st.markdown("</div>", unsafe_allow_html=True)

# Display the thermal inputs
st.markdown('<div class="section"><div class="section-title">🌡️ Thermal</div><div class="section-note">Temperature measurements.</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    temperature = st.text_input("Temperature", key="Temperature", placeholder="Enter a number, e.g. 86.3671")
with col2:
    thermocouple = st.text_input("Thermocouple", key="Thermocouple", placeholder="Enter a number, e.g. 28.7819")
st.markdown("</div>", unsafe_allow_html=True)

# Display the process inputs
st.markdown('<div class="section"><div class="section-title">💧 Process</div><div class="section-note">Pressure and flow measurements.</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    pressure = st.text_input("Pressure", key="Pressure", placeholder="Enter a number, e.g. 0.382638")
with col2:
    flow_rate = st.text_input("Volume Flow Rate RMS", key="Volume Flow Rate RMS", placeholder="Enter a number, e.g. 126.309")
st.markdown("</div>", unsafe_allow_html=True)

# Display the prediction button
if st.button("🔍 Analyse Industrial Condition", use_container_width=True, type="primary"):

    # Convert the manually entered text values into numbers
    try:
        values = {
            "Accelerometer1RMS": float(accelerometer1),
            "Accelerometer2RMS": float(accelerometer2),
            "Current": float(current),
            "Pressure": float(pressure),
            "Temperature": float(temperature),
            "Thermocouple": float(thermocouple),
            "Voltage": float(voltage),
            "Volume Flow RateRMS": float(flow_rate)
        }
    except ValueError:
        st.error("Please enter valid numeric values in all eight sensor fields. You can use positive or negative numbers.")
        st.stop()

    # Prepare the model input using only the eight sensor readings
    input_data = pd.DataFrame([values])

    # Arrange inputs in the exact order used during model training
    input_data = input_data[features]

    # Generate the anomaly probability and prediction
    probability = float(model.predict_proba(input_data)[0][1])
    prediction = int(probability >= threshold)

    # Display the main prediction result
    st.markdown('<div class="section"><div class="section-title">📋 Detection Result</div>', unsafe_allow_html=True)

    if prediction == 1:
        st.markdown('<div class="result-anomaly">🔴 Anomaly Detected — the entered sensor condition differs from the learned normal operating pattern.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-normal">🟢 Normal Operation — the entered sensor condition matches the learned normal operating pattern.</div>', unsafe_allow_html=True)

    # Display the essential prediction values
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("Anomaly Probability", f"{probability * 100:.2f}%")
    with result_col2:
        st.metric("Decision Threshold", f"{threshold * 100:.0f}%")

    st.progress(min(probability, 1.0))
    st.markdown('<div class="info-box">The probability is the model\'s estimated likelihood of the anomaly class. It does not by itself confirm physical equipment failure.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Keep technical details optional
with st.expander("ℹ️ About the Model"):
    st.write("Model: HistGradientBoosting Classifier")
    st.write("Task: Binary industrial anomaly classification")
    st.write("Output: Normal (0) or Anomaly (1)")
    st.write("The model uses eight industrial sensor readings only.")
    st.write("Evaluation includes accuracy, precision, recall, F1-score, ROC-AUC and a confusion matrix.")

# Display the footer
st.markdown('<div class="footer">Industrial Anomaly Detection • Machine Learning Prototype</div>', unsafe_allow_html=True)
