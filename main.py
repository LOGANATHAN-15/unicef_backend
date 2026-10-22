"""
UNICEF immunization coverage API.

Serves the precomputed outputs of the TFT notebook:
  data/TFT_future_forecast_2024_2030.csv
  data/TFT_predictions_UNICEF.csv          (walk-forward backtest 2020-2023)
  data/model_comparison.csv
  data/cleaned_unicef_immunization_dataset.csv   (optional, enables observed history)
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data"))

# Comma-separated list of allowed frontend origins, e.g.
# CORS_ORIGINS="https://my-app.vercel.app,http://localhost:5173"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if o.strip()]

app = FastAPI(title="UNICEF Immunization Coverage API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _read(name: str, required: bool = True) -> Optional[pd.DataFrame]:
    path = DATA_DIR / name
    if not path.exists():
        if required:
            raise RuntimeError(f"Missing required data file: {path}")
        return None
    return pd.read_csv(path)


# Loaded once at startup; the files are small and read-only.
FORECAST = _read("TFT_future_forecast_2024_2030.csv")
BACKTEST = _read("TFT_predictions_UNICEF.csv")
COMPARISON = _read("model_comparison.csv")
HISTORY = _read("cleaned_unicef_immunization_dataset.csv", required=False)
if HISTORY is not None:
    HISTORY = HISTORY.dropna(subset=["Country", "Vaccine", "Year", "Coverage"]).copy()
    HISTORY["Year"] = HISTORY["Year"].astype(int)


def _records(df: pd.DataFrame) -> list:
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def _filter(df: pd.DataFrame, country=None, vaccine=None, year_from=None, year_to=None) -> pd.DataFrame:
    if country:
        df = df[df["Country"].str.lower() == country.lower()]
    if vaccine:
        df = df[df["Vaccine"].str.lower() == vaccine.lower()]
    if year_from is not None:
        df = df[df["Year"] >= year_from]
    if year_to is not None:
        df = df[df["Year"] <= year_to]
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta():
    """Dropdown options for the frontend."""
    countries = sorted(FORECAST["Country"].unique().tolist())
    vaccines = sorted(FORECAST["Vaccine"].unique().tolist())
    regions = sorted(FORECAST["UNICEF_Region"].dropna().unique().tolist()) if "UNICEF_Region" in FORECAST else []
    return {
        "countries": countries,
        "vaccines": vaccines,
        "regions": regions,
        "forecast_years": [int(FORECAST["Year"].min()), int(FORECAST["Year"].max())],
        "backtest_years": sorted(BACKTEST["Year"].unique().astype(int).tolist()),
        "has_history": HISTORY is not None,
    }


@app.get("/metrics")
def metrics():
    """Model comparison table (MAE, RMSE, MAPE, SMAPE, R2, Accuracy, Within_5pp)."""
    return _records(COMPARISON)


@app.get("/forecast")
def forecast(
    country: Optional[str] = None,
    vaccine: Optional[str] = None,
    region: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = Query(5000, le=50000),
):
    df = _filter(FORECAST, country, vaccine, year_from, year_to)
    if region and "UNICEF_Region" in df:
        df = df[df["UNICEF_Region"].str.lower() == region.lower()]
    return {"count": len(df), "data": _records(df.head(limit))}


@app.get("/backtest")
def backtest(
    country: Optional[str] = None,
    vaccine: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = Query(5000, le=50000),
):
    """Walk-forward predictions vs actuals for 2020-2023."""
    df = _filter(BACKTEST, country, vaccine, year_from, year_to)
    return {"count": len(df), "data": _records(df.head(limit))}


@app.get("/series")
def series(country: str, vaccine: str):
    """Everything a chart needs for one country + vaccine in a single call."""
    fc = _filter(FORECAST, country, vaccine)
    bt = _filter(BACKTEST, country, vaccine)
    if fc.empty and bt.empty:
        raise HTTPException(404, f"No data for {country} / {vaccine}")
    hist = _filter(HISTORY, country, vaccine)[["Year", "Coverage"]] if HISTORY is not None else pd.DataFrame()
    return {
        "country": country,
        "vaccine": vaccine,
        "history": _records(hist.sort_values("Year")) if not hist.empty else [],
        "backtest": _records(bt.sort_values("Year")),
        "forecast": _records(fc.sort_values("Year")),
    }


@app.get("/summary")
def summary(year: int = 2030, vaccine: Optional[str] = None):
    """Mean predicted coverage per region for a year. Handy for overview cards."""
    df = FORECAST[FORECAST["Year"] == year]
    if vaccine:
        df = df[df["Vaccine"].str.lower() == vaccine.lower()]
    if df.empty:
        raise HTTPException(404, "No forecast rows for that selection")
    out = (df.groupby("UNICEF_Region")["Predicted_Coverage"].agg(["mean", "count"]).round(2)
             .rename(columns={"mean": "mean_coverage", "count": "n_series"}).reset_index())
    return {"year": year, "vaccine": vaccine, "regions": _records(out)}
