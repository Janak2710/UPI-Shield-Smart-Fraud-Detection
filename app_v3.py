import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "experimental" / "upi_fraud_model_v3.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "experimental" / "upi_fraud_threshold_v3.joblib"
DATA_PATH = BASE_DIR / "data" / "transactions_cleaned.csv"

st.set_page_config(
    page_title="UPI Shield | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #08111f 0%, #101c30 55%, #0b1424 100%);
    color: #e5edf8;
}

[data-testid="stSidebar"] {
    background: #0c1728;
    border-right: 1px solid #25344b;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(115deg, #142d4b, #123c56, #16465a);
    border: 1px solid #28516c;
    margin-bottom: 25px;
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #c4d6e9;
    font-size: 15px;
}

.metric-card {
    background: #142238;
    border: 1px solid #2a3d57;
    padding: 20px;
    border-radius: 14px;
    min-height: 115px;
}

.metric-label {
    color: #a9bad0;
    font-size: 13px;
}

.metric-value {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    color: #eaf2ff;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

div.stButton > button {
    background: linear-gradient(90deg, #087fce, #13a5c7);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 700;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0969aa, #0c829f);
    color: white;
    border: none;
}

div[data-testid="stForm"] {
    background: #142238;
    border: 1px solid #2a3d57;
    padding: 20px;
    border-radius: 14px;
}

.result-card {
    background: #142238;
    border: 1px solid #2a3d57;
    padding: 22px;
    border-radius: 14px;
    margin-top: 15px;
}

.small-note {
    color: #9eb1c9;
    font-size: 12px;
}

h2, h3 {
    color: #eaf2ff;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. LOAD MODEL AND DATASET
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_threshold():
    return joblib.load(THRESHOLD_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    threshold = load_threshold()
    df = load_dataset()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.info("Please check your models and data folders.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load project files: {error}")
    st.stop()


# --------------------------------------------------
# 4. SESSION HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("# 🛡️ UPI Shield")
st.sidebar.caption("Smart Transaction Security")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Analyze Transaction",
        "Detection History",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Machine Learning System**")
st.sidebar.caption("Educational project using a synthetic dataset.")


# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ UPI Shield</h1>
        <p>Intelligent UPI Transaction Fraud Detection System</p>
        <p>Monitor transaction patterns and identify potentially fraudulent activity using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    total_transactions = len(df)
    total_fraud = int(df["is_fraud"].sum())
    genuine_transactions = total_transactions - total_fraud
    fraud_percentage = (total_fraud / total_transactions * 100) if total_transactions else 0

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        (col1, "TOTAL TRANSACTIONS", f"{total_transactions:,}"),
        (col2, "FRAUD TRANSACTIONS", f"{total_fraud:,}"),
        (col3, "GENUINE TRANSACTIONS", f"{genuine_transactions:,}"),
        (col4, "FRAUD RATE", f"{fraud_percentage:.2f}%")
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Transaction Overview</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Fraud vs Genuine Transactions")
        distribution = pd.DataFrame({
            "Transaction Type": ["Genuine", "Fraud"],
            "Transactions": [genuine_transactions, total_fraud]
            }).set_index("Transaction Type")

        st.bar_chart(distribution)

    with chart_col2:
        st.markdown("#### Dataset Information")
        st.markdown(f"""
        <div class="result-card">
        <p><b>Dataset:</b> UPI Transaction Dataset</p>
        <p><b>Records:</b> {total_transactions:,} transactions</p>
        <p><b>Prediction Target:</b> is_fraud</p>
        <p><b>Model:</b> Logistic Regression</p>
        <p><b>Application:</b> Educational fraud detection demonstration</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("Dataset values are synthetic and do not represent actual bank or UPI transactions.")


# --------------------------------------------------
# 7. TRANSACTION ANALYSIS
# --------------------------------------------------

elif page == "Analyze Transaction":

    st.markdown("""
    <div class="hero">
        <h1>Transaction Analysis</h1>
        <p>Analyze transaction details and assess potential fraud risk using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("This demonstration uses the trained model and the same input features used during model training.")

    with st.form("transaction_form"):

        st.markdown("### 💳 Transaction Details")

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=13.70,
                max_value=32259.72,
                value=1500.0,
                step=100.0,
                help="Enter an amount within the range observed in the training dataset (₹13.70 to ₹32,259.72)."
                )

            receiver_type = st.selectbox(
                "Receiver Type",
                ["Merchant", "User"]
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                ["P2P", "P2M", "Recharge", "Bill Payment", "EMI", "Subscription"]
            )

            payment_app = st.selectbox(
                "Payment Application",
                ["GPay", "PhonePe", "Paytm", "Amazon Pay", "BHIM", "WhatsApp Pay"]
                )

            device_type = st.selectbox(
                "Device Type",
                ["Android", "iOS", "Web"]
            )

            balance = st.number_input(
                "Balance After Transaction (₹)",
                min_value=0.0,
                value=5000.0,
                step=500.0
            )

        with col2:
            hour = st.slider("Transaction Hour", 0, 23, 14)

            day = st.selectbox(
                "Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

            city_tier = st.selectbox(
                "User City Tier",
                ["Tier 1", "Tier 2", "Tier 3"]
            )

            kyc = st.selectbox(
                "KYC Status",
                ["Verified", "Not Verified"]
            )

            time_gap = st.number_input(
                "Minutes Since Last Transaction",
                min_value=0.0,
                max_value=500000.0,
                value=60.0,
                step=1.0,
                help="Time elapsed since the user's previous transaction, measured in minutes."
                )

            failed_attempts = st.number_input(
                "Failed Attempts in Last 24 Hours",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                help="Number of unsuccessful transaction attempts made by the user in the last 24 hours."
                )

        st.markdown("### 👤 User Behaviour")

        col3, col4 = st.columns(2)

        with col3:
            avg_monthly_txn = st.number_input(
                "Average Monthly Transactions",
                min_value=0.0,
                value=20.0
            )

            avg_txn_value = st.number_input(
                "Average Transaction Value (₹)",
                min_value=0.0,
                value=2000.0
            )

            loyalty_score = st.slider(
                "User Loyalty Score",
                0.0, 1.0, 0.7
            )

            transaction_velocity = st.number_input(
                "Transaction Velocity",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Indicates how frequently the user makes transactions. The value should follow the definition used in the training dataset."
                )

        with col4:
            new_device = st.selectbox(
                "New Device",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Select Yes if the user is making this transaction from a device not previously used by them. Select No otherwise."
                )

            location_mismatch = st.selectbox(
                "IP Location Mismatch",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Select Yes if the transaction's detected IP location differs from the user's usual location. Select No otherwise."
                )
            
            if avg_txn_value > 0:
                amount_deviation = abs(amount - avg_txn_value) / avg_txn_value
            else:
                amount_deviation = 0.0

            st.number_input(
                "Amount Deviation Score (Auto-calculated)",
                value=float(amount_deviation),
                disabled=True,
                help="Automatically calculated from the transaction amount and the user's average transaction value."
            )

            recurring = st.selectbox(
                "Recurring Payment",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            frequency_score = st.slider(
                "Transaction Frequency Score",
                0.0, 1.0, 0.5
            )

        submitted = st.form_submit_button("🔍 Analyze Transaction")

    if submitted:

        transaction_input = pd.DataFrame([{
            "receiver_type": receiver_type,
            "amount": amount,
            "hour_of_day": hour,
            "day_of_week": day,
            "is_weekend": int(day in ["Saturday", "Sunday"]),
            "is_night_transaction": int(hour >= 22 or hour <= 5),
            "time_since_last_txn_min": time_gap,
            "transaction_type": transaction_type,
            "payment_app": payment_app,
            "device_type": device_type,
            "user_city_tier": city_tier,
            "user_kyc_status": kyc,
            "user_avg_monthly_txn": avg_monthly_txn,
            "user_avg_txn_value": avg_txn_value,
            "user_loyalty_score": loyalty_score,
            "new_device_flag": new_device,
            "ip_location_mismatch": location_mismatch,
            "failed_attempts_last_24h": failed_attempts,
            "transaction_velocity": transaction_velocity,
            "amount_deviation_score": amount_deviation,
            "recurring_payment_flag": recurring,
            "balance_after_transaction": balance,
            "transaction_frequency_score": frequency_score
        }])

        try:
            probability = model.predict_proba(transaction_input)[0][1]
            prediction = int(probability >= threshold)
            
            if probability > 0.467:
                st.warning(
                    "⚠️ This risk score is higher than the maximum score "
                    "reported in the model's test dataset (46.7%). "
                    "The prediction may be unreliable for unusual inputs."
                    )

            st.markdown('<div class="section-title">Analysis Result</div>', unsafe_allow_html=True)

            if prediction == 1:
                st.error("⚠️ Transaction Flagged for Review")
                st.write(
                    "This transaction has been flagged for review based on the model's "
                    "risk score. A flagged transaction is not necessarily fraudulent. "
                    "The model was trained on synthetic data and may produce false alarms."
                    )
            else:
                st.success("✅ Transaction Not Flagged")
                st.write(
                    "This transaction was not flagged for review based on the model's "
                    "risk score. However, the model may miss fraudulent transactions. "
                    "This result does not guarantee that the transaction is safe."
                    )

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Model Risk Score", f"{probability * 100:.2f}%")

            with result_col2:
                st.metric(
                    "Model Decision",
                    "Flagged for Review" if prediction else "Not Flagged"
                    )

            st.caption(
                f"Decision threshold: {threshold * 100:.2f}% "
                "(selected using validation data). "
                "This is an experimental model trained on synthetic data, "
                "not a real banking security decision."
                )

            st.session_state.history.append({
                "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Amount (₹)": amount,
                "Model Risk Score (%)": round(probability * 100, 2),
                "Prediction": "Flagged for Review" if prediction else "Not Flagged"
            })

        except Exception as error:
            st.error(f"Prediction error: {error}")
            st.warning("The website input categories or features may need to be matched with the training dataset.")


# --------------------------------------------------
# 8. DETECTION HISTORY
# --------------------------------------------------

elif page == "Detection History":

    st.markdown("""
    <div class="hero">
        <h1>Detection History</h1>
        <p>Review transactions analyzed during this browser session.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        flagged_count = int((history_df["Prediction"] == "Flagged for Review").sum())
        not_flagged_count = len(history_df) - flagged_count

        col1, col2, col3 = st.columns(3)
        col1.metric("Analyzed", len(history_df))
        col2.metric("Flagged for Review", flagged_count)
        col3.metric("Not Flagged", not_flagged_count)

        csv_data = history_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="upi_detection_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No transactions have been analyzed in this session yet.")


# --------------------------------------------------
# 9. ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <h1>About UPI Shield</h1>
        <p>A machine learning based academic project for UPI fraud detection.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objective")

    st.write(
        "The objective of this project is to explore how machine learning "
        "can help identify potentially fraudulent UPI transactions by "
        "analyzing transaction details and user behaviour."
    )

    st.markdown("### Machine Learning")

    st.write(
    "The application uses a trained machine learning pipeline "
    "to analyze transaction details and generate a model risk score. "
    "Based on this score, the system classifies transactions as "
    "potentially fraudulent or genuine."
    )

    st.markdown("### Important Dataset Disclaimer")

    st.warning(
        "This project uses a synthetic dataset created for academic "
        "and educational purposes. It does not use actual bank records, "
        "real customer information, or live UPI transaction data. "
        "Predictions are experimental and must not be used for real "
        "financial decisions."
    )

    st.markdown("### Project Technology")

    st.write(
        "- Python\n"
        "- Pandas and NumPy\n"
        "- Scikit-learn\n"
        "- Streamlit\n"
        "- Joblib\n"
        "- Machine Learning"
    )

# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8194ad;font-size:12px;'>"
    "UPI Shield | Academic Major Project | Machine Learning Demonstration"
    "</p>",
    unsafe_allow_html=True
)