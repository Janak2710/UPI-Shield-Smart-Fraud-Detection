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

**Label provenance and potential circularity:** The original GitHub project's README states that the fraud labels were generated from a probabilistic score combining eight risk signals: new device, IP location mismatch, failed attempts, transaction velocity, amount deviation, KYC status, night transaction, and high-risk user. The original label-generator code is not included, so this claim cannot be independently re-checked. Most of these signals are also model inputs, meaning the results largely demonstrate recovery of a synthetic scoring rule rather than proven effectiveness against real-world UPI fraud.

**Test sample size:** The chronological test split contains 3,000 transactions, including **124 fraud cases**. Recall and other fraud-class metrics are estimates from a limited sample and may vary with a different sample.

## Machine Learning Model

The V3 training script compares **Logistic Regression (Balanced)**, **Logistic Regression (Unbalanced)**, and **Random Forest**. The selected model is unbalanced Logistic Regression, based on validation PR-AUC.

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

V3 detects fraud at broadly similar levels to the original model. Its principal improvements are a validation-selected decision threshold and a clearer, more honest presentation of the model risk score.

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

## Run Order (from project folder)

For a fresh run, execute these commands in order:

```powershell
py -m pip install -r requirements.txt
py data_preprocessing.py
py train_model_v3.py
py -m streamlit run app_v3.py
```

Preprocessing generates `data/transactions_cleaned.csv`. Training generates model artifacts in `models/experimental/` and evaluation outputs in `reports/`. The included model artifacts allow launching the app without retraining, but follow the full sequence to reproduce the workflow. The training script reuses the supplied synthetic labels; it does not generate the original labels.

## Run the Application

To run the V3 website:

```powershell
py -m streamlit run app_v3.py
```


## Important Limitations

- The system uses synthetic transaction data; the original project README describes an eight-signal probabilistic labeling rule, but the generator code was not supplied for independent verification.
- Labels may overlap with risk-related input features, potentially limiting the meaning of model evaluation.
- The held-out test set contains only 124 fraud cases, so fraud-class percentages are approximate.
- The average-transaction-value input is restricted to ₹50–₹5,000 for demonstration based on the observed dataset range (approximately ₹52–₹4,539); other inputs may still be outside the training distribution.
- Dependency versions are specified as compatibility ranges, not a fully reproducible lockfile; test the environment before deployment.
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
