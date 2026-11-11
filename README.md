# UPI Shield – Smart Fraud Detection

## Project Overview

UPI Shield is a machine learning-based fraud detection system developed as a BCA major project.

The application analyzes UPI transaction details and flags potentially suspicious transactions for review.

The system is built using Python, Scikit-learn, and Streamlit.

## Dataset

- Dataset: Synthetic UPI Transaction Dataset
- Total Transactions: 20,000
- Fraud Transactions: 763
- Genuine Transactions: 19,237

The dataset is synthetic and does not represent actual banking transactions.

## Machine Learning Model

The project evaluates Logistic Regression and Random Forest models.

The experimental V3 system uses unbalanced Logistic Regression with a validation-selected decision threshold.

### V3 Model Performance

| Metric | Test Result |
|---|---|
| Accuracy | 80.20% |
| Precision | 10.30% |
| Recall | 49.19% |
| F1-Score | 17.04% |
| ROC-AUC | 72.60% |
| PR-AUC | 10.96% |
| Decision Threshold | 4.73% |

These results are based on a synthetic test dataset and should not be interpreted as real-world banking performance.

## Website Features

- Dashboard showing transaction statistics
- Transaction analysis using machine learning
- Model risk score and review decision
- Detection History
- CSV download of session history
- Project information page

## Installation

Open a terminal in the project folder and install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

## Run the Application

To run the V3 website:

```powershell
py -m streamlit run app_v3.py
```

To run the original website:

```powershell
py -m streamlit run app.py
```

## Important Limitations

- The system uses synthetic transaction data.
- The model can generate false alarms and miss fraudulent transactions.
- Model risk scores are not guaranteed real-world fraud probabilities.
- Predictions outside the tested data distribution may be unreliable.
- The application is intended for academic demonstration, not real banking security decisions.

## Project Type

BCA Major Project – Machine Learning and Fraud Detection.

## Dataset Source and Attribution

The synthetic UPI transaction dataset and original risk-signal design were sourced from:

Original GitHub Repository:
https://github.com/iamgokulchowdary/UPI-Fraud-Detection-System
The original project's README explains that the fraud labels were generated using a probabilistic scoring method based on eight risk signals.

UPI Shield builds on this dataset to evaluate machine learning models, select a validation-based decision threshold, and create a Streamlit application that flags suspicious transactions for review.

Credit for the original dataset and risk-signal design belongs to the original project creator.
