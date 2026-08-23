"""
cleaning_pipeline.py

Data cleaning and transformation pipeline for the US Accidents (2016-2023)
dataset, implementing the strategy documented in:
Week2_Data_Cleaning_Transformation.docx

Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

Usage:
    python cleaning_pipeline.py --input US_Accidents_March23.csv --output cleaned_accidents.csv
"""

import argparse
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Missingness audit
# ---------------------------------------------------------------------------
def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing count and missing percentage, sorted
    from most to least missing."""
    missing = df.isna().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    }).sort_values("missing_pct", ascending=False)
    return report


# ---------------------------------------------------------------------------
# 2. Tiered missing-value handling
#    <=5%   : impute (median / mode)
#    5-40%  : impute + add a "<col>_was_missing" flag column
#    >40%   : drop the column entirely
#    target : drop the row if Severity itself is missing
# ---------------------------------------------------------------------------
def handle_missing_values(
    df: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    target_col: str = "Severity",
    drop_threshold: float = 40.0,
    flag_threshold: float = 5.0,
) -> pd.DataFrame:
    df = df.copy()
    missing_pct = (df.isna().sum() / len(df)) * 100

    for col in numeric_cols:
        if col not in df.columns:
            continue
        pct = missing_pct.get(col, 0)
        if pct > drop_threshold:
            df.drop(columns=[col], inplace=True)
            continue
        if pct > flag_threshold:
            df[f"{col}_was_missing"] = df[col].isna().astype(int)
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if col not in df.columns:
            continue
        pct = missing_pct.get(col, 0)
        if pct > drop_threshold:
            df.drop(columns=[col], inplace=True)
            continue
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val.iloc[0])

    if target_col in df.columns:
        df = df.dropna(subset=[target_col])

    return df


# ---------------------------------------------------------------------------
# 3. Duplicate detection using a composite key
#    (rounded start coordinates + start time, since the source feeds
#     don't guarantee a single natural key)
# ---------------------------------------------------------------------------
def drop_duplicates_composite(
    df: pd.DataFrame,
    lat_col: str = "Start_Lat",
    lng_col: str = "Start_Lng",
    time_col: str = "Start_Time",
    decimals: int = 4,
) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()  # exact full-row duplicates first

    if {lat_col, lng_col, time_col}.issubset(df.columns):
        key = (
            df[lat_col].round(decimals).astype(str)
            + "_"
            + df[lng_col].round(decimals).astype(str)
            + "_"
            + df[time_col].astype(str)
        )
        before = len(df)
        df = df.loc[~key.duplicated(keep="first")]
        after = len(df)
        dup_rate = 1 - (after / before) if before else 0
        print(f"Composite-key duplicate rate: {dup_rate:.4f} "
              f"({before - after} rows dropped)")

    return df


# ---------------------------------------------------------------------------
# 4. Outlier capping using the IQR method, with an optional hard domain
#    ceiling applied first (e.g. physically implausible sensor readings)
# ---------------------------------------------------------------------------
def cap_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return series.clip(lower=lower, upper=upper)


def handle_outliers(
    df: pd.DataFrame,
    cols: list,
    domain_ceilings: dict | None = None,
) -> pd.DataFrame:
    df = df.copy()
    domain_ceilings = domain_ceilings or {}

    for col in cols:
        if col not in df.columns:
            continue
        if col in domain_ceilings:
            ceiling = domain_ceilings[col]
            df.loc[df[col] > ceiling, col] = np.nan
            df[col] = df[col].fillna(df[col].median())
        df[col] = cap_outliers_iqr(df[col])

    return df


# ---------------------------------------------------------------------------
# 5. Type standardization
# ---------------------------------------------------------------------------
def standardize_types(
    df: pd.DataFrame,
    datetime_cols: list,
    bool_cols: list,
) -> pd.DataFrame:
    df = df.copy()

    for col in datetime_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        fail_rate = parsed.isna().mean() - df[col].isna().mean()
        if fail_rate > 0.001:
            print(f"Warning: {fail_rate:.2%} of '{col}' failed to parse "
                  f"as datetime (threshold 0.1%).")
        df[col] = parsed

    for col in bool_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(bool)

    return df


# ---------------------------------------------------------------------------
# 6. Weather-condition consolidation into a smaller controlled vocabulary
# ---------------------------------------------------------------------------
BASE_CONDITION_MAP = {
    "rain": "Rain", "drizzle": "Rain", "showers": "Rain",
    "snow": "Snow", "sleet": "Snow", "wintry": "Snow",
    "fog": "Fog", "mist": "Fog", "haze": "Fog",
    "clear": "Clear", "fair": "Clear",
    "cloud": "Cloudy", "overcast": "Cloudy",
    "thunder": "Thunderstorm",
}


def consolidate_weather_condition(
    df: pd.DataFrame,
    col: str = "Weather_Condition",
) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df

    lower = df[col].astype(str).str.lower()

    df["is_windy"] = lower.str.contains("windy", na=False).astype(int)
    df["has_thunder"] = lower.str.contains("thunder", na=False).astype(int)

    def map_base(value: str) -> str:
        for keyword, base in BASE_CONDITION_MAP.items():
            if keyword in value:
                return base
        return "Other"

    df[f"{col}_base"] = lower.fillna("").map(map_base)
    return df


# ---------------------------------------------------------------------------
# 7. Scaling / standardization
# ---------------------------------------------------------------------------
def z_score_standardize(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std and not np.isnan(std) and std != 0:
            df[f"{col}_z"] = (df[col] - mean) / std
    return df


# ---------------------------------------------------------------------------
# 8. Feature engineering
# ---------------------------------------------------------------------------
def engineer_time_features(
    df: pd.DataFrame,
    start_col: str = "Start_Time",
    end_col: str = "End_Time",
) -> pd.DataFrame:
    df = df.copy()
    if start_col in df.columns:
        df["Hour"] = df[start_col].dt.hour
        df["DayOfWeek"] = df[start_col].dt.dayofweek
        df["Month"] = df[start_col].dt.month
        df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    if {start_col, end_col}.issubset(df.columns):
        df["Duration_Minutes"] = (
            (df[end_col] - df[start_col]).dt.total_seconds() / 60
        )

    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
]
CATEGORICAL_COLS = ["Weather_Condition", "Wind_Direction", "State", "City"]
BOOL_COLS = [
    "Junction", "Crossing", "Traffic_Signal", "Give_Way", "Bump",
]
DATETIME_COLS = ["Start_Time", "End_Time"]
DOMAIN_CEILINGS = {"Wind_Speed(mph)": 150}


def run_pipeline(input_path: str, output_path: str) -> None:
    print(f"Loading {input_path} ...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    report = missingness_report(df)
    print("\nTop missing columns:")
    print(report.head(10))

    # Drop very-high-missing columns (e.g. End_Lat/End_Lng) up front,
    # per the >40% rule documented in the Week 2 report.
    df = handle_missing_values(df, NUMERIC_COLS, CATEGORICAL_COLS)

    df = drop_duplicates_composite(df)

    df = standardize_types(df, DATETIME_COLS, BOOL_COLS)

    df = handle_outliers(
        df,
        cols=[c for c in NUMERIC_COLS if c in df.columns],
        domain_ceilings=DOMAIN_CEILINGS,
    )

    df = consolidate_weather_condition(df)

    df = z_score_standardize(
        df, cols=[c for c in NUMERIC_COLS if c in df.columns]
    )

    df = engineer_time_features(df)

    print(f"\nFinal shape: {df.shape}")
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to raw US Accidents CSV")
    parser.add_argument("--output", default="cleaned_accidents.csv", help="Path to write cleaned CSV")
    args = parser.parse_args()
    run_pipeline(args.input, args.output)
