import streamlit as st
import requests
import json

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🛡️ Fraud Transaction Prediction System")

# ดึง URL จาก kubectl get isvc fraud-detection
# หมายเหตุ: ในเครื่อง Local อาจต้องใช้ Port-forward หรือแก้ URL ตามจริง
KSERVE_URL = "http://fraud-detection.default.example.com/v2/models/fraud-detection/infer"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Info")
    amount = st.number_input("Transaction Amount", value=100.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", value=1000.0)
    newbalanceOrig = st.number_input("Sender New Balance", value=900.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", value=100.0)
    
with col2:
    st.subheader("Action")
    if st.button("Predict Fraud", use_container_width=True):
        # เตรียม Data ตาม V2 Protocol
        payload = {
            "inputs": [
                {
                    "name": "input-0",
                    "shape": [1, 5], 
                    "datatype": "FP32",
                    "data": [amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]
                }
            ]
        }
        
        try:
            response = requests.post(KSERVE_URL, json=payload)
            result = response.json()
            # ดึงค่าทำนาย (ตัวอย่าง: [0.99] หมายถึงโกงแน่นอน)
            score = result["outputs"][0]["data"][0]
            
            if score > 0.5:
                st.error(f"🚨 FRAUD DETECTED! (Score: {score:.4f})")
            else:
                st.success(f"✅ TRANSACTION LEGIT (Score: {score:.4f})")
        except Exception as e:
            st.warning("Could not connect to KServe. Make sure the URL is accessible.")
            st.error(str(e))
