"""
Parse the Gilgit-Baltistan climate workbook (GB Data 1984-2019.xlsx) into tidy CSVs.

The workbook has one sheet per weather station. Each sheet stacks three blocks
vertically: Maximum Temperature, Minimum Temperature and Precipitation. Each block
has a "Years" header row followed by yearly rows (Jan..Dec + an annual column).

Outputs (written next to this script, in ../data):
  gb_climate_monthly.csv  -> station, variable, year, month, value
  gb_climate_annual.csv   -> station, variable, year, annual
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Source workbook (edit if you move the file)
SRC = Path(r"C:\Users\HP Z BOOK\Downloads\GB Data 1984-2019.xlsx")
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# Map the messy block titles to clean variable names
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# Match block titles by keyword (handles both the clean sheets and the raw
# PMD "Hunza" export, e.g. "MONTHLY MEAN MAXIMUM TEMPERATURE (oC)").
def _clean_variable(title) -> str | None:
    if not isinstance(title, str):
        return None
    key = title.strip().lower()
    if not key:
        return None
    if "maximum temperature" in key or key == "maximum temperature":
        return "max_temp"
    if "minimum temperature" in key:
        return "min_temp"
    if ("precipitation" in key or "rainfall" in key
            or "amount of" in key):
        return "precipitation"
    return None


def _to_float(val) -> float | None:
    """Convert a cell to float. 'TRACE' -> 0.0; -100 sentinel / other text -> None."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"trace", "t"}:
            return 0.0
        try:
            val = float(s)
        except ValueError:
            return None
    val = float(val)
    if val <= -99:  # PMD "-100.0 means data not available" sentinel
        return None
    return val


def _parse_year(val) -> int | None:
    if isinstance(val, str):
        val = val.strip()
        if not val.isdigit():
            return None
    try:
        year = int(float(val))
    except (ValueError, TypeError):
        return None
    return year if 1900 <= year <= 2100 else None


def parse_sheet(raw: pd.DataFrame, station: str) -> tuple[list[dict], list[dict]]:
    """Return (monthly_rows, annual_rows) parsed from one station sheet."""
    monthly: list[dict] = []
    annual: list[dict] = []

    current_var: str | None = None
    for i in range(len(raw)):
        first = raw.iloc[i, 0]

        # A block title row (e.g. "Maximum Temperature")
        var = _clean_variable(first)
        if var is not None:
            current_var = var
            continue

        # A data row: first cell should be a 4-digit year
        if current_var is None:
            continue
        year = _parse_year(first)
        if year is None:
            continue

        # Months are columns 1..12, annual is column 13
        for m_idx, month in enumerate(MONTHS, start=1):
            val = _to_float(raw.iloc[i, m_idx])
            if val is not None:
                monthly.append(
                    {"station": station, "variable": current_var,
                     "year": year, "month": month, "month_num": m_idx,
                     "value": val}
                )
        ann = _to_float(raw.iloc[i, 13]) if raw.shape[1] > 13 else None
        if ann is not None:
            annual.append(
                {"station": station, "variable": current_var,
                 "year": year, "annual": ann}
            )

    return monthly, annual


def main() -> None:
    xl = pd.ExcelFile(SRC)
    all_monthly: list[dict] = []
    all_annual: list[dict] = []

    for station in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name=station, header=None)
        monthly, annual = parse_sheet(raw, station)
        all_monthly.extend(monthly)
        all_annual.extend(annual)
        print(f"{station:10s} -> {len(monthly):4d} monthly, {len(annual):3d} annual rows")

    monthly_df = pd.DataFrame(all_monthly)
    annual_df = pd.DataFrame(all_annual)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(OUT_DIR / "gb_climate_monthly.csv", index=False)
    annual_df.to_csv(OUT_DIR / "gb_climate_annual.csv", index=False)

    print("\nSaved:")
    print(" ", OUT_DIR / "gb_climate_monthly.csv", monthly_df.shape)
    print(" ", OUT_DIR / "gb_climate_annual.csv", annual_df.shape)
    print("\nStations:", sorted(monthly_df["station"].unique()))
    print("Variables:", sorted(monthly_df["variable"].unique()))
    print("Years:", int(monthly_df["year"].min()), "-", int(monthly_df["year"].max()))


if __name__ == "__main__":
    main()
