import glob
import os

import pandas as pd

# --- Configuration ---
INPUT_PATTERN = "*.csv"  # Change to a folder path like "data/*.csv" if needed
OUTPUT_FILE = "flightData.csv"
CUTOFF_DATE = pd.Timestamp("2025-08-27")

all_frames = []

csv_files = glob.glob(INPUT_PATTERN)
if not csv_files:
    print(f"No CSV files found matching: {INPUT_PATTERN}")
    exit(1)

print(f"Found {len(csv_files)} CSV file(s).")

for filepath in csv_files:
    # Skip the header rows (first 7 lines are metadata, line 8 is the real header)
    try:
        df = pd.read_csv(filepath, skiprows=7)
    except Exception as e:
        print(f"  Skipping {filepath}: {e}")
        continue

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Parse the date column
    df["Date (MM/DD/YYYY)"] = pd.to_datetime(
        df["Date (MM/DD/YYYY)"], format="%m/%d/%Y", errors="coerce"
    )

    # Filter out rows with dates after the cutoff
    before = len(df)
    df = df[df["Date (MM/DD/YYYY)"] <= CUTOFF_DATE]
    after = len(df)

    print(f"  {os.path.basename(filepath)}: {before} rows → {after} kept")
    all_frames.append(df)

if not all_frames:
    print("No data to write.")
    exit(1)

merged = pd.concat(all_frames, ignore_index=True)

# Write output — one header, all data
merged.to_csv(OUTPUT_FILE, index=False)
print(f"\nDone! {len(merged)} total rows written to: {OUTPUT_FILE}")
