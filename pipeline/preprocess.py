import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

OHE_COLS = ["body", "transmission", "state", "color", "interior"]
HASH_COLS = ["make", "model", "trim", "seller"]
NUM_COLS = ["year", "condition", "odometer"]
DROP_COLS = ["vin", "saledate"]
TARGET = "sellingprice"


def hash_encode(frame, cols, n_features=16):
    for col in cols:
        if col not in frame.columns:
            continue
        hasher = FeatureHasher(n_features=n_features, input_type="string")
        hashed = hasher.transform(frame[col].astype(str).apply(lambda x: [x]))
        hashed_df = pd.DataFrame(
            hashed.toarray(),
            columns=[f"{col}_hash_{i}" for i in range(n_features)],
            index=frame.index,
        )
        frame = pd.concat([frame.drop(columns=[col]), hashed_df], axis=1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample_2.csv")
    ap.add_argument("--outdir", default="data/processed")
    args = ap.parse_args()

    df = pd.read_csv(args.input)

    # --- Cleaning (stateless, safe before split) ---
    df = df[~df["transmission"].isin(["Sedan", "sedan"])].copy()
    df["make"] = df["make"].str.capitalize()
    df["body"] = df["body"].str.lower()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df.dropna(subset=[TARGET]).copy()

    # --- Split FIRST (this is what prevents data leakage) ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, test_df = train_df.copy(), test_df.copy()

    # --- Outlier thresholds / rare categories: learned on TRAIN only ---
    p99_price = np.percentile(train_df[TARGET], 99)
    p99_odo = np.percentile(train_df["odometer"].dropna(), 99)
    train_df = train_df[
        (train_df[TARGET] <= p99_price)
        & (train_df["odometer"].fillna(0) <= p99_odo)
    ].copy()

    body_counts = train_df["body"].value_counts()
    rare_body = set(body_counts[body_counts < 50].index)
    for fr in (train_df, test_df):
        fr["body"] = fr["body"].where(~fr["body"].isin(rare_body), "other")

    # --- Imputation: statistics learned on TRAIN only ---
    make_cond_median = train_df.groupby("make")["condition"].median()
    global_cond_median = train_df["condition"].median()
    odo_median = train_df["odometer"].median()
    trans_mode = train_df["transmission"].mode()[0]
    for fr in (train_df, test_df):
        fr["condition"] = fr["condition"].fillna(fr["make"].map(make_cond_median))
        fr["condition"] = fr["condition"].fillna(global_cond_median)
        fr["odometer"] = fr["odometer"].fillna(odo_median)
        fr["transmission"] = fr["transmission"].fillna(trans_mode)
        for col in ["make", "model", "trim", "body", "color", "interior"]:
            fr[col] = fr[col].fillna("unknown")

    # --- One-Hot Encoding (categories from TRAIN, TEST aligned) ---
    ohe = [c for c in OHE_COLS if c in train_df.columns]
    train_df = pd.get_dummies(train_df, columns=ohe)
    test_df = pd.get_dummies(test_df, columns=ohe)

    # --- Hash Encoding (stateless, cannot leak) ---
    train_df = hash_encode(train_df, HASH_COLS)
    test_df = hash_encode(test_df, HASH_COLS)
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # --- Scaling: StandardScaler fit on TRAIN only ---
    scaler = StandardScaler()
    train_df[NUM_COLS] = scaler.fit_transform(train_df[NUM_COLS])
    test_df[NUM_COLS] = scaler.transform(test_df[NUM_COLS])

    # --- Save to data/... (correct location) ---
    os.makedirs(args.outdir, exist_ok=True)
    assert list(train_df.columns) == list(test_df.columns), "column mismatch"
    train_df.to_csv(os.path.join(args.outdir, "train_prepared.csv"), index=False)
    test_df.to_csv(os.path.join(args.outdir, "test_prepared.csv"), index=False)
    print(f"Saved train {train_df.shape} and test {test_df.shape} to {args.outdir}")


if __name__ == "__main__":
    main()
