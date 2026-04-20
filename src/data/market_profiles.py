"""
market_profiles.py — Cluster listings into Duration Risk Profiles.

Uses real Inside Airbnb listing attributes to create the t-SNE
latent space visualization described in DEMO.md.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# Duration risk profile categories
PROFILE_LABELS = {
    0: "Short-Stay Dominant",
    1: "Medium Flex",
    2: "Long-Term Stable",
}


def assign_duration_profiles(listings: pd.DataFrame) -> pd.DataFrame:
    """Assign each listing a Duration Risk Profile based on real attributes."""
    df = listings.copy()

    # Rule-based initial classification using minimum_nights
    conditions = [
        df["minimum_nights"] <= 3,
        (df["minimum_nights"] > 3) & (df["minimum_nights"] <= 14),
        df["minimum_nights"] > 14,
    ]
    labels = ["Short-Stay Dominant", "Medium Flex", "Long-Term Stable"]
    df["duration_profile"] = np.select(conditions, labels, default="Medium Flex")

    return df


def compute_fragmentation_features(listings: pd.DataFrame) -> pd.DataFrame:
    """Compute fragmentation vulnerability features for each listing.

    Higher fragmentation vulnerability = listing is more susceptible to
    "Swiss cheese" booking gaps from short-stay interruptions.
    """
    df = listings.copy()

    # Feature 1: Duration flexibility range
    df["duration_range"] = df["maximum_nights"] - df["minimum_nights"]
    df["duration_range"] = df["duration_range"].clip(0, 365)

    # Feature 2: Price volatility proxy (deviation from neighbourhood median)
    neighbourhood_median = df.groupby("neighbourhood")["price"].transform("median")
    df["price_deviation"] = (df["price"] - neighbourhood_median) / neighbourhood_median.clip(lower=1)

    # Feature 3: Occupancy gap vulnerability
    # High occupancy + short minimum nights = high fragmentation risk
    occ = df.get("occupancy_rate", pd.Series(0.5, index=df.index))
    df["frag_vulnerability"] = occ * (1.0 / df["minimum_nights"].clip(lower=1))

    # Feature 4: Bedroom-based sensitivity
    # Multi-bedroom units lose more revenue from fragmentation
    df["bedroom_sensitivity"] = df["bedrooms"].clip(1, 5) / 5.0

    # Feature 5: Revenue density (RevPAR / beds)
    revpar = df.get("revpar", df["price"] * 0.5)
    df["revenue_density"] = revpar / df["beds"].clip(lower=1)

    return df


def build_feature_matrix(listings: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix for clustering / t-SNE.

    Returns the scaled feature matrix and feature names.
    """
    feature_cols = [
        "price",
        "minimum_nights",
        "bedrooms",
        "duration_range",
        "price_deviation",
        "frag_vulnerability",
        "bedroom_sensitivity",
        "revenue_density",
    ]

    available_cols = [c for c in feature_cols if c in listings.columns]

    X = listings[available_cols].fillna(0).values.astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Replace any remaining NaN/inf
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0)

    return X_scaled, available_cols


def run_profiling(listings: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Run the full profiling pipeline.

    Returns:
        - listings with profile labels and features added
        - scaled feature matrix for t-SNE
    """
    logger.info("Building duration risk profiles...")

    # Step 1: Assign profiles
    df = assign_duration_profiles(listings)
    profile_counts = df["duration_profile"].value_counts()
    for profile, count in profile_counts.items():
        logger.info(f"  {profile}: {count:,} listings")

    # Step 2: Compute fragmentation features
    df = compute_fragmentation_features(df)

    # Step 3: Build feature matrix
    X_scaled, feature_names = build_feature_matrix(df)
    logger.info(f"  ✓ Feature matrix: {X_scaled.shape}")

    return df, X_scaled
