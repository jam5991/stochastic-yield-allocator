"""
fetch_airbnb.py — Download and process real Inside Airbnb data for NYC.

Data source: Inside Airbnb (CC-BY 4.0)
https://insideairbnb.com/get-the-data
"""

import os
import gzip
import io
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)

# --- Austin Inside Airbnb URLs (latest available snapshot) ---
AUSTIN_BASE = "https://data.insideairbnb.com/united-states/tx/austin"
AUSTIN_DATE = "2025-09-16"

URLS = {
    "listings": f"{AUSTIN_BASE}/{AUSTIN_DATE}/data/listings.csv.gz",
    "calendar": f"{AUSTIN_BASE}/{AUSTIN_DATE}/data/calendar.csv.gz",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def download_file(url: str, dest: Path) -> Path:
    """Download a file with progress logging, skip if already cached."""
    if dest.exists():
        logger.info(f"  ✓ Cached: {dest.name}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  ↓ Downloading {dest.name} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                print(f"\r    {pct:5.1f}%  ({downloaded >> 20} MB / {total >> 20} MB)", end="", flush=True)
    print()
    logger.info(f"  ✓ Saved: {dest.name} ({downloaded >> 20} MB)")
    return dest


def fetch_raw_data() -> dict[str, Path]:
    """Download raw CSVs from Inside Airbnb."""
    logger.info("Fetching Inside Airbnb Austin data...")
    paths = {}
    for key, url in URLS.items():
        dest = RAW_DIR / f"{key}.csv.gz"
        paths[key] = download_file(url, dest)
    return paths


def process_listings(raw_path: Path) -> pd.DataFrame:
    """Parse listings.csv.gz into a clean DataFrame."""
    logger.info("Processing listings...")
    df = pd.read_csv(raw_path, compression="gzip", low_memory=False)

    # Select & rename relevant columns
    cols = {
        "id": "listing_id",
        "name": "name",
        "neighbourhood_cleansed": "neighbourhood",
        "neighbourhood_group_cleansed": "borough",
        "room_type": "room_type",
        "bedrooms": "bedrooms",
        "beds": "beds",
        "price": "price_str",
        "minimum_nights": "minimum_nights",
        "maximum_nights": "maximum_nights",
        "availability_365": "availability_365",
        "number_of_reviews": "num_reviews",
        "review_scores_rating": "review_score",
        "latitude": "latitude",
        "longitude": "longitude",
    }

    available_cols = {k: v for k, v in cols.items() if k in df.columns}
    df = df[list(available_cols.keys())].rename(columns=available_cols)

    # Parse price: "$1,234.00" → 1234.0
    if "price_str" in df.columns:
        df["price"] = (
            df["price_str"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )
        df = df.drop(columns=["price_str"])
    else:
        df["price"] = np.nan

    # Fill missing bedrooms
    df["bedrooms"] = df["bedrooms"].fillna(1).astype(int)
    df["beds"] = df.get("beds", pd.Series(1, index=df.index)).fillna(1).astype(int)

    # Filter out unreasonable prices
    df = df[(df["price"] > 10) & (df["price"] < 10_000)].copy()

    # Cap min/max nights to reasonable bounds
    df["minimum_nights"] = df["minimum_nights"].clip(1, 365)
    df["maximum_nights"] = df["maximum_nights"].clip(1, 1095)

    logger.info(f"  ✓ {len(df):,} listings after cleaning")
    return df


def process_calendar(raw_path: Path, listing_ids: set) -> pd.DataFrame:
    """Parse calendar.csv.gz into a clean DataFrame, filtering to valid listings."""
    logger.info("Processing calendar (this may take a moment for Austin-scale data)...")

    df = pd.read_csv(raw_path, compression="gzip", low_memory=False)

    # Rename
    rename_map = {"listing_id": "listing_id", "date": "date", "available": "available", "price": "price_str"}
    available_renames = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df[list(available_renames.keys())].rename(columns=available_renames)

    # Filter to listings we have
    df = df[df["listing_id"].isin(listing_ids)].copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"])

    # Parse availability
    df["available"] = df["available"].map({"t": True, "f": False}).fillna(False)

    # Parse price
    if "price_str" in df.columns:
        df["price"] = (
            df["price_str"]
            .astype(str)
            .str.replace(r"[$,]", "", regex=True)
            .astype(float)
        )
        df = df.drop(columns=["price_str"])

    logger.info(f"  ✓ {len(df):,} calendar rows")
    return df


def compute_listing_metrics(listings_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-listing occupancy, ADR, and RevPAR from real calendar data."""
    logger.info("Computing per-listing yield metrics...")

    # Occupancy = fraction of days booked (not available)
    occupancy = (
        calendar_df.groupby("listing_id")["available"]
        .apply(lambda x: 1.0 - x.mean())
        .rename("occupancy_rate")
    )

    # ADR = average daily rate when booked
    adr = (
        calendar_df[~calendar_df["available"]]
        .groupby("listing_id")["price"]
        .mean()
        .rename("adr")
    )

    # RevPAR = ADR × Occupancy
    metrics = pd.DataFrame({"occupancy_rate": occupancy, "adr": adr})
    metrics["revpar"] = metrics["adr"] * metrics["occupancy_rate"]

    result = listings_df.merge(metrics, left_on="listing_id", right_index=True, how="left")
    result["adr"] = result["adr"].fillna(result["price"])
    result["occupancy_rate"] = result["occupancy_rate"].fillna(0.5)
    result["revpar"] = result["adr"] * result["occupancy_rate"]

    logger.info(f"  ✓ Median Occupancy: {result['occupancy_rate'].median():.1%}")
    logger.info(f"  ✓ Median ADR: ${result['adr'].median():.0f}")
    logger.info(f"  ✓ Median RevPAR: ${result['revpar'].median():.0f}")

    return result


def compute_booking_distributions(calendar_df: pd.DataFrame) -> dict:
    """Extract real booking duration and price distributions from calendar data."""
    logger.info("Extracting booking duration distributions from real data...")

    # Find contiguous booked blocks per listing
    durations = []
    prices = []

    # Sample a subset of listings for efficiency
    sample_ids = calendar_df["listing_id"].unique()
    if len(sample_ids) > 2000:
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(sample_ids, size=2000, replace=False)

    for lid in sample_ids:
        sub = calendar_df[calendar_df["listing_id"] == lid].sort_values("date")
        booked = (~sub["available"]).values
        price_vals = sub["price"].values

        # Find contiguous booked runs
        i = 0
        while i < len(booked):
            if booked[i]:
                start = i
                while i < len(booked) and booked[i]:
                    i += 1
                duration = i - start
                avg_price = np.nanmean(price_vals[start:i])
                if 1 <= duration <= 90:
                    durations.append(duration)
                    if not np.isnan(avg_price):
                        prices.append(avg_price)
            else:
                i += 1

    durations = np.array(durations)
    prices = np.array(prices)

    distributions = {
        "duration_mean": float(np.mean(durations)) if len(durations) > 0 else 3.0,
        "duration_std": float(np.std(durations)) if len(durations) > 0 else 2.0,
        "duration_median": float(np.median(durations)) if len(durations) > 0 else 2.0,
        "price_mean": float(np.mean(prices)) if len(prices) > 0 else 150.0,
        "price_std": float(np.std(prices)) if len(prices) > 0 else 80.0,
        "duration_histogram": np.histogram(durations, bins=range(1, 32))[0].tolist() if len(durations) > 0 else [],
    }

    logger.info(f"  ✓ {len(durations):,} booking blocks found")
    logger.info(f"  ✓ Mean duration: {distributions['duration_mean']:.1f} days")
    logger.info(f"  ✓ Mean price: ${distributions['price_mean']:.0f}/night")

    return distributions


def run_pipeline() -> dict:
    """Run the full data pipeline. Returns paths and computed data."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    listings_parquet = PROCESSED_DIR / "listings.parquet"
    calendar_parquet = PROCESSED_DIR / "calendar_sample.parquet"
    distributions_path = PROCESSED_DIR / "booking_distributions.npz"

    # Check for cached processed data
    if listings_parquet.exists() and calendar_parquet.exists() and distributions_path.exists():
        logger.info("Loading cached processed data...")
        listings = pd.read_parquet(listings_parquet)
        calendar_sample = pd.read_parquet(calendar_parquet)
        dist_data = np.load(distributions_path, allow_pickle=True)
        distributions = dict(dist_data["distributions"].item())
        return {
            "listings": listings,
            "calendar_sample": calendar_sample,
            "distributions": distributions,
        }

    # Download raw data
    raw_paths = fetch_raw_data()

    # Process listings
    listings = process_listings(raw_paths["listings"])

    # Process calendar (for a sample of listings for memory efficiency)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(listings["listing_id"].unique(), size=min(5000, len(listings)), replace=False)
    sample_ids_set = set(sample_ids)

    calendar = process_calendar(raw_paths["calendar"], sample_ids_set)

    # Compute metrics
    listings_with_metrics = compute_listing_metrics(
        listings[listings["listing_id"].isin(sample_ids_set)], calendar
    )

    # Merge metrics back to full listings (for non-sampled, use defaults)
    listings = listings.merge(
        listings_with_metrics[["listing_id", "occupancy_rate", "adr", "revpar"]],
        on="listing_id",
        how="left",
    )
    listings["adr"] = listings["adr"].fillna(listings["price"])
    listings["occupancy_rate"] = listings["occupancy_rate"].fillna(0.5)
    listings["revpar"] = listings["revpar"].fillna(listings["adr"] * 0.5)

    # Compute booking distributions
    distributions = compute_booking_distributions(calendar)

    # Save processed data
    listings.to_parquet(listings_parquet, index=False)

    # Save a calendar sample for the demo (top 200 listings by reviews)
    top_listings = listings.nlargest(200, "num_reviews")["listing_id"]
    calendar_sample = calendar[calendar["listing_id"].isin(set(top_listings))]
    calendar_sample.to_parquet(calendar_parquet, index=False)

    np.savez(distributions_path, distributions=distributions)

    logger.info(f"\n✓ Pipeline complete. {len(listings):,} listings processed.")

    return {
        "listings": listings,
        "calendar_sample": calendar_sample,
        "distributions": distributions,
    }


if __name__ == "__main__":
    data = run_pipeline()
    print(f"\nListings shape: {data['listings'].shape}")
    print(f"Calendar sample shape: {data['calendar_sample'].shape}")
    print(f"Booking distributions: {data['distributions']}")
