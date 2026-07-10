"""
Gilgit-Baltistan Climate Trends & Forecast
==========================================
An interactive dashboard analysing 1984-2019 temperature and precipitation
records from seven weather stations in Gilgit-Baltistan, Pakistan -- the gateway
to the country's glaciers. It quantifies warming trends and uses machine learning
to forecast future annual values.

Run with:
    python -m streamlit run climate/gb_climate_app.py

Data source: PMD station records (GB Data 1984-2019.xlsx), parsed by
climate/parse_gb_data.py into data/gb_climate_monthly.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

DATA = Path(__file__).resolve().parent.parent / "data" / "gb_climate_monthly.csv"

VAR_LABELS = {
    "max_temp": "Maximum Temperature (\u00b0C)",
    "min_temp": "Minimum Temperature (\u00b0C)",
    "precipitation": "Precipitation (mm)",
}
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Approximate coordinates (lat, lon) and elevation of the PMD stations across
# Gilgit-Baltistan, Pakistan. Used to place them on the map.
STATION_META = {
    "Astore": {"lat": 35.3667, "lon": 74.8667, "elev_m": 2168},
    "Bunji":  {"lat": 35.6667, "lon": 74.6333, "elev_m": 1372},
    "Chilas": {"lat": 35.4200, "lon": 74.0956, "elev_m": 1250},
    "Gilgit": {"lat": 35.9200, "lon": 74.3080, "elev_m": 1500},
    "Gupis":  {"lat": 36.2340, "lon": 73.4400, "elev_m": 2156},
    "Hunza":  {"lat": 36.3167, "lon": 74.6500, "elev_m": 2438},
    "Skardu": {"lat": 35.2971, "lon": 75.6333, "elev_m": 2228},
}

st.set_page_config(page_title="Gilgit-Baltistan Climate Trends",
                   page_icon="\U0001f3d4\ufe0f", layout="wide")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)
    return df


def annual_series(df: pd.DataFrame, station: str, variable: str) -> pd.DataFrame:
    """Annual value per year. Temperatures -> yearly mean; precipitation -> yearly total.

    Only keep years with all 12 months reported, so trends aren't distorted by
    partial years.
    """
    sub = df[(df["station"] == station) & (df["variable"] == variable)]
    counts = sub.groupby("year")["value"].count()
    full_years = counts[counts == 12].index
    sub = sub[sub["year"].isin(full_years)]
    agg = "sum" if variable == "precipitation" else "mean"
    out = sub.groupby("year", as_index=False)["value"].agg(agg)
    return out.sort_values("year").reset_index(drop=True)


def trend_per_decade(years: np.ndarray, values: np.ndarray) -> float:
    if len(years) < 2:
        return float("nan")
    slope = np.polyfit(years, values, 1)[0]
    return slope * 10.0


@st.cache_data
def all_station_trends(variable: str) -> pd.DataFrame:
    """ML trend summary for every station for one variable.

    For each station, fit a Linear Regression on annual values vs year and
    report the warming/precipitation rate per decade, the R^2 (how linear the
    change is) and a 5-year-ahead projection. Merged with map coordinates.
    """
    rows = []
    for station, meta in STATION_META.items():
        ann = annual_series(df, station, variable)
        if len(ann) < 3:
            continue
        X = ann["year"].values.reshape(-1, 1)
        y = ann["value"].values
        model = LinearRegression().fit(X, y)
        slope_decade = float(model.coef_[0] * 10.0)
        r2 = float(model.score(X, y))
        last_year = int(ann["year"].max())
        proj = float(model.predict([[last_year + 5]])[0])
        rows.append({
            "station": station,
            "lat": meta["lat"], "lon": meta["lon"], "elev_m": meta["elev_m"],
            "trend_decade": slope_decade,
            "r2": r2,
            "mean_value": float(y.mean()),
            "last_value": float(y[-1]),
            "proj_5yr": proj,
            "first_year": int(ann["year"].min()),
            "last_year": last_year,
            "n_years": len(ann),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
df = load_data()
stations = sorted(df["station"].unique())

st.sidebar.title("\U0001f3d4\ufe0f  Controls")
station = st.sidebar.selectbox("Weather station", stations,
                               index=stations.index("Gilgit") if "Gilgit" in stations else 0)
variable = st.sidebar.radio("Climate variable", list(VAR_LABELS.keys()),
                            format_func=lambda v: VAR_LABELS[v])
st.sidebar.caption(
    "Data: 7 PMD stations across Gilgit-Baltistan, Pakistan (1984\u20132019). "
    "Temperatures shown as annual means; precipitation as annual totals."
)

var_label = VAR_LABELS[variable]
ann = annual_series(df, station, variable)


# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("Gilgit-Baltistan Climate Trends & Forecast")
st.markdown(
    f"**{station}** \u00b7 {var_label} \u00b7 "
    f"{int(ann['year'].min())}\u2013{int(ann['year'].max())}"
    if len(ann) else f"**{station}** \u00b7 {var_label}"
)

if len(ann) < 3:
    st.warning("Not enough complete years for this station/variable to analyse.")
    st.stop()

decade_trend = trend_per_decade(ann["year"].values, ann["value"].values)
total_change = decade_trend * (ann["year"].max() - ann["year"].min()) / 10.0
hottest = ann.loc[ann["value"].idxmax()]
coldest = ann.loc[ann["value"].idxmin()]
unit = "mm" if variable == "precipitation" else "\u00b0C"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Trend", f"{decade_trend:+.2f} {unit}/decade",
          help="Linear rate of change per decade")
k2.metric("Total change (period)", f"{total_change:+.2f} {unit}")
k3.metric(f"Highest year", f"{hottest['value']:.1f} {unit}", f"{int(hottest['year'])}")
k4.metric(f"Lowest year", f"{coldest['value']:.1f} {unit}", f"{int(coldest['year'])}")

tab_map, tab_trend, tab_season, tab_ml, tab_data = st.tabs(
    ["\U0001f5fa\ufe0f Map", "\U0001f4c8 Trend", "\U0001f5d3\ufe0f Seasonal",
     "\U0001f52e ML Forecast", "\U0001f5c2\ufe0f Data"]
)


# ----------------------------------------------------------------------------
# Map tab -- all stations across Gilgit-Baltistan, coloured by ML trend
# ----------------------------------------------------------------------------
with tab_map:
    st.subheader(f"Gilgit-Baltistan stations \u2014 {var_label}")
    st.caption(
        "Each point is a weather station. A Linear Regression is fitted to every "
        "station's annual record; **circle colour and size show the ML-estimated "
        f"trend per decade** for {var_label.lower()}. Bigger, redder circles warm faster."
    )

    summary = all_station_trends(variable).copy()

    if summary.empty:
        st.info("No station trends available for this variable.")
    else:
        warming = variable in ("max_temp", "min_temp")
        # Colour: red = rising, blue = falling (for temperature); for precip
        # green = wetter, brown = drier.
        max_abs = max(summary["trend_decade"].abs().max(), 1e-6)

        def _color(t: float):
            frac = max(-1.0, min(1.0, t / max_abs))
            if warming:
                if frac >= 0:  # warming -> red
                    return [200, int(60 * (1 - frac)) + 30, 40, 200]
                return [40, 90, 200, 200]  # cooling -> blue
            else:
                if frac >= 0:  # wetter -> teal/green
                    return [15, 140, 130, 200]
                return [150, 100, 40, 200]  # drier -> brown

        summary["color"] = summary["trend_decade"].apply(_color)
        summary["radius"] = 6000 + summary["trend_decade"].abs() / max_abs * 22000
        summary["trend_txt"] = summary["trend_decade"].round(2).astype(str)
        summary["proj_txt"] = summary["proj_5yr"].round(1).astype(str)

        unit = "mm" if variable == "precipitation" else "\u00b0C"
        try:
            import pydeck as pdk

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=summary,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.8,
                stroked=True,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1,
            )
            text_layer = pdk.Layer(
                "TextLayer",
                data=summary,
                get_position="[lon, lat]",
                get_text="station",
                get_size=13,
                get_color=[20, 20, 20],
                get_alignment_baseline="'top'",
            )
            view = pdk.ViewState(latitude=35.8, longitude=74.5, zoom=6.6, pitch=0)
            tooltip = {
                "html": "<b>{station}</b><br/>Trend: {trend_txt} " + unit + "/decade"
                        "<br/>Elevation: {elev_m} m<br/>2024 projection: {proj_txt} " + unit,
                "style": {"backgroundColor": "#10233B", "color": "white"},
            }
            st.pydeck_chart(pdk.Deck(
                map_style="road",
                initial_view_state=view,
                layers=[layer, text_layer],
                tooltip=tooltip,
            ))
        except Exception:
            # Fallback to a simple point map if pydeck/tiles are unavailable
            st.map(summary[["lat", "lon"]], zoom=6)

        # Ranked ML summary under the map
        st.markdown("**Machine-learning trend ranking (Linear Regression per station)**")
        show = summary.sort_values("trend_decade", ascending=False)[
            ["station", "elev_m", "first_year", "last_year", "n_years",
             "trend_decade", "r2", "last_value", "proj_5yr"]
        ].rename(columns={
            "station": "Station", "elev_m": "Elevation (m)",
            "first_year": "From", "last_year": "To", "n_years": "Years",
            "trend_decade": f"Trend ({unit}/decade)", "r2": "R\u00b2 (fit)",
            "last_value": f"Latest ({unit})", "proj_5yr": f"+5yr proj ({unit})",
        })
        st.dataframe(
            show.style.format({
                f"Trend ({unit}/decade)": "{:+.2f}", "R\u00b2 (fit)": "{:.2f}",
                f"Latest ({unit})": "{:.1f}", f"+5yr proj ({unit})": "{:.1f}",
            }),
            width="stretch",
        )

        if warming:
            fastest = summary.loc[summary["trend_decade"].idxmax()]
            st.info(
                f"**{fastest['station']}** is warming fastest at "
                f"**{fastest['trend_decade']:+.2f} \u00b0C/decade**. Faster warming at "
                "high-elevation stations accelerates glacial melt and raises "
                "glacial-lake-flood (GLOF) risk downstream."
            )

        st.markdown("**Trend vs elevation** \u2014 are higher stations warming faster?")
        elev_chart = summary.set_index("station")[["elev_m", "trend_decade"]]
        st.scatter_chart(summary, x="elev_m", y="trend_decade", color="station",
                         size="mean_value")


# ----------------------------------------------------------------------------
# Trend tab
# ----------------------------------------------------------------------------
with tab_trend:
    st.subheader(f"Annual {var_label} at {station}")

    plot = ann.set_index("year").rename(columns={"value": var_label})
    # Linear trend line
    coeffs = np.polyfit(ann["year"], ann["value"], 1)
    plot["Trend"] = np.polyval(coeffs, ann["year"].values)
    # 5-year rolling mean
    plot["5-yr average"] = ann["value"].rolling(5, min_periods=1).mean().values
    st.line_chart(plot)

    if variable in ("max_temp", "min_temp"):
        msg = (f"{station} is **warming by {decade_trend:+.2f} \u00b0C per decade**"
               if decade_trend > 0 else
               f"{station} shows a cooling trend of {decade_trend:+.2f} \u00b0C per decade")
        st.info(msg + " \u2014 warmer air accelerates glacial melt and swells glacial lakes.")
    else:
        st.info(f"Precipitation trend: **{decade_trend:+.1f} mm per decade**. "
                "Shifts in rain/snow balance change how fast glaciers gain or lose mass.")


# ----------------------------------------------------------------------------
# Seasonal tab
# ----------------------------------------------------------------------------
with tab_season:
    st.subheader(f"Monthly pattern \u2014 {var_label} at {station}")
    sub = df[(df["station"] == station) & (df["variable"] == variable)]
    monthly_avg = (sub.groupby("month", observed=True)["value"].mean()
                   .reindex(MONTH_ORDER))
    st.bar_chart(monthly_avg)

    # Decade comparison: has the seasonal cycle shifted?
    early = sub[sub["year"] <= sub["year"].min() + 9]
    late = sub[sub["year"] >= sub["year"].max() - 9]
    comp = pd.DataFrame({
        f"First decade ({int(early['year'].min())}\u2013{int(early['year'].max())})":
            early.groupby("month", observed=True)["value"].mean().reindex(MONTH_ORDER),
        f"Last decade ({int(late['year'].min())}\u2013{int(late['year'].max())})":
            late.groupby("month", observed=True)["value"].mean().reindex(MONTH_ORDER),
    })
    st.markdown("**Have the seasons shifted? First vs last decade**")
    st.line_chart(comp)


# ----------------------------------------------------------------------------
# ML Forecast tab
# ----------------------------------------------------------------------------
with tab_ml:
    st.subheader("Machine-learning forecast")
    st.caption(
        "Models are trained on the earlier years and tested on the most recent "
        "years (a time-based split), then used to project future annual values."
    )

    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", ["Linear Regression", "Polynomial (degree 2)"])
    horizon = c2.slider("Forecast horizon (years ahead)", 1, 15, 10)

    X = ann["year"].values.reshape(-1, 1)
    y = ann["value"].values

    # Time-based split: last 20% of years as test
    split = max(3, int(len(ann) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())

    model.fit(X_train, y_train)

    if len(X_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else float("nan")
    else:
        mae, r2 = float("nan"), float("nan")

    # Refit on all data for the forward projection
    model.fit(X, y)
    last_year = int(ann["year"].max())
    future_years = np.arange(ann["year"].min(), last_year + horizon + 1).reshape(-1, 1)
    future_pred = model.predict(future_years)

    m1, m2, m3 = st.columns(3)
    m1.metric("Test MAE", f"{mae:.2f} {unit}" if not np.isnan(mae) else "n/a")
    m2.metric("Test R\u00b2", f"{r2:.2f}" if not np.isnan(r2) else "n/a")
    projected = future_pred[-1]
    m3.metric(f"Projected {last_year + horizon}", f"{projected:.1f} {unit}",
              f"{projected - y[-1]:+.1f} vs {last_year}")

    chart = pd.DataFrame({"year": future_years.ravel(), "Forecast": future_pred})
    chart = chart.merge(ann.rename(columns={"value": "Observed"}), on="year", how="left")
    st.line_chart(chart.set_index("year")[["Observed", "Forecast"]])

    st.caption(
        "Simple, interpretable models on a short record \u2014 a demonstration of method, "
        "not an operational climate projection. Real forecasting would add more stations, "
        "satellite data, and physical climate models."
    )


# ----------------------------------------------------------------------------
# Data tab
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("Station coverage")
    cover = (df.groupby(["station", "variable"])["year"]
             .agg(["min", "max", "count"]).reset_index())
    cover.columns = ["Station", "Variable", "First year", "Last year", "Records"]
    st.dataframe(cover, width="stretch")

    st.subheader(f"Annual values \u2014 {station} \u00b7 {var_label}")
    show = ann.rename(columns={"year": "Year", "value": var_label})
    st.dataframe(show, width="stretch")
    st.download_button("Download this series (CSV)", show.to_csv(index=False),
                       file_name=f"{station}_{variable}.csv", mime="text/csv")
