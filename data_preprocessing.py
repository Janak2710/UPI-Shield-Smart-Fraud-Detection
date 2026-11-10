
import pandas as pd
from pathlib import Path

# Define project folders
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load the original transaction dataset
input_file = DATA_DIR / "transactions.csv"

print("Loading UPI transaction dataset...")

df = pd.read_csv(input_file)

# Display basic dataset information
print("\nOriginal Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert timestamp column into datetime format
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Sort transactions by user and transaction time
df = df.sort_values(
    by=["user_id", "timestamp"]
).reset_index(drop=True)

# Recalculate time since the user's previous transaction
df["time_since_last_txn_min"] = (
    df.groupby("user_id")["timestamp"]
    .diff()
    .dt.total_seconds() / 60
)

# Display cleaned dataset information
print("\nCleaned Dataset Shape:", df.shape)

print("\nUpdated Time Gap Information:")
print(df["time_since_last_txn_min"].describe())

# Save cleaned dataset
output_file = DATA_DIR / "transactions_cleaned.csv"

df.to_csv(output_file, index=False)

print("\nData preprocessing completed successfully!")
print("Cleaned dataset saved at:", output_file)