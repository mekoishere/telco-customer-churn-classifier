import streamlit as st
import pandas as pd
import joblib
import os

#config stronki
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

@st.cache_resource
def load_assets():
    if os.path.exists('churn_model.pkl') and os.path.exists('model_columns.pkl'):
        model = joblib.load('churn_model.pkl')
        cols = joblib.load('model_columns.pkl')
        return model, cols
    return None, None

model, model_columns = load_assets()

st.title("Panel Utrzymania Klienta")

if model is None:
    st.error("Blad: Brak plikow churn_model.pkl lub model_columns.pkl. Uruchom main.py.")
    st.stop()

#wprowadzanie danych przez uzytkownika
st.sidebar.header("Dane klienta")

def get_user_input():
    tenure = st.sidebar.slider("Tenure (miesiace)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 18, 120, 50)
    total_charges = st.sidebar.number_input("Total Charges", value=float(tenure * monthly_charges))
    
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])

    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", 
                                   ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])

    data = {
        'tenure': tenure,
        'SeniorCitizen': senior,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'InternetService': internet,
        'TechSupport': tech_support,
        'OnlineSecurity': security
    }
    return pd.DataFrame([data])

input_df = get_user_input()

st.subheader("Dane do analizy")
st.write(input_df)

# Predykcja
if st.button("Analizuj ryzyko odejscia"):
    input_encoded = pd.get_dummies(input_df)
    
    final_df = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_df)
    probability = model.predict_proba(final_df)
    
    st.divider()
    
    if prediction == 1:
        st.error(f"WYNIK: KLIENT PRAWDOPODOBNIE ODEJDZIE")
    else:
        st.success(f"WYNIK: KLIENT PRAWDOPODOBNIE ZOSTANIE")
        
    prob = probability[0][1]    
    st.metric("Prawdopodobienstwo odejscia", f"{prob*100:.2f}%")

    log_file = 'churn_logs.csv'
    new_log = pd.DataFrame([[pd.Timestamp.now(), probability]], columns=['Timestamp', 'Probability'])
    if not os.path.isfile(log_file):
        new_log.to_csv(log_file, index=False)
    else:
        new_log.to_csv(log_file, mode='a', header=False, index=False)

#logi
if os.path.exists('churn_logs.csv'):
    st.divider()
    st.subheader("Historia predykcji (Monitoring)")
    logs = pd.read_csv('churn_logs.csv')
    st.dataframe(logs.tail(10))