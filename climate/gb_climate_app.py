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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "gb_climate_monthly.csv"

# background_gradient styling needs matplotlib; degrade gracefully without it.
try:
    import matplotlib  # noqa: F401
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

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


def detect_anomalies(ann: pd.DataFrame, z_thresh: float = 1.5) -> pd.DataFrame:
    """Flag years whose value deviates from the linear trend by > z_thresh sigma."""
    if len(ann) < 4:
        return ann.assign(expected=np.nan, residual=np.nan, anomaly=False)
    coeffs = np.polyfit(ann["year"], ann["value"], 1)
    expected = np.polyval(coeffs, ann["year"].values)
    resid = ann["value"].values - expected
    sd = resid.std() or 1.0
    z = resid / sd
    return ann.assign(expected=expected, residual=resid,
                      z=z, anomaly=np.abs(z) > z_thresh)


@st.cache_data
def station_feature_matrix(variable: str) -> pd.DataFrame:
    """Build a per-station feature table for clustering: mean, trend, seasonality."""
    rows = []
    for st_name in STATION_META:
        ann = annual_series(df, st_name, variable)
        if len(ann) < 3:
            continue
        sub = df[(df["station"] == st_name) & (df["variable"] == variable)]
        monthly = sub.groupby("month", observed=True)["value"].mean()
        rows.append({
            "station": st_name,
            "elev_m": STATION_META[st_name]["elev_m"],
            "mean": float(ann["value"].mean()),
            "trend_decade": trend_per_decade(ann["year"].values, ann["value"].values),
            "seasonality": float(monthly.max() - monthly.min()),
            "variability": float(ann["value"].std()),
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

# Year-range filter -- applies to the single-station analysis tabs.
yr_lo, yr_hi = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year range", yr_lo, yr_hi, (yr_lo, yr_hi))

smooth = st.sidebar.slider("Rolling-average window (years)", 1, 10, 5,
                           help="Smooths the trend line to reveal longer-term change.")

st.sidebar.caption(
    "Data: 7 PMD stations across Gilgit-Baltistan, Pakistan (1984\u20132019). "
    "Temperatures shown as annual means; precipitation as annual totals."
)

var_label = VAR_LABELS[variable]
ann_full = annual_series(df, station, variable)
ann = ann_full[(ann_full["year"] >= year_range[0])
               & (ann_full["year"] <= year_range[1])].reset_index(drop=True)


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

tab_map, tab_trend, tab_season, tab_compare, tab_ml, tab_cluster, tab_report, tab_data = st.tabs(
    ["\U0001f5fa\ufe0f Map", "\U0001f4c8 Trend", "\U0001f5d3\ufe0f Seasonal",
     "\u2696\ufe0f Compare", "\U0001f52e ML Forecast", "\U0001f9e9 Clusters",
     "\U0001f4dd Report", "\U0001f5c2\ufe0f Data"]
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
    # Rolling mean (window from sidebar)
    plot[f"{smooth}-yr average"] = ann["value"].rolling(smooth, min_periods=1).mean().values
    st.line_chart(plot)

    if variable in ("max_temp", "min_temp"):
        msg = (f"{station} is **warming by {decade_trend:+.2f} \u00b0C per decade**"
               if decade_trend > 0 else
               f"{station} shows a cooling trend of {decade_trend:+.2f} \u00b0C per decade")
        st.info(msg + " \u2014 warmer air accelerates glacial melt and swells glacial lakes.")
    else:
        st.info(f"Precipitation trend: **{decade_trend:+.1f} mm per decade**. "
                "Shifts in rain/snow balance change how fast glaciers gain or lose mass.")

    # --- Anomaly detection --------------------------------------------------
    st.markdown("#### \u26a0\ufe0f Anomaly detection")
    st.caption(
        "Years that deviate strongly from the station's own trend line "
        "(residual > threshold \u00d7 standard deviation) are flagged as anomalies."
    )
    z_thresh = st.slider("Sensitivity (lower = more anomalies)", 1.0, 3.0, 1.5, 0.1,
                         key="anom_z")
    anom = detect_anomalies(ann, z_thresh)
    n_anom = int(anom["anomaly"].sum())
    if n_anom:
        flagged = anom[anom["anomaly"]].copy()
        flagged["type"] = np.where(flagged["residual"] > 0, "\U0001f525 Unusually high",
                                   "\u2744\ufe0f Unusually low")
        st.warning(f"**{n_anom} anomalous year(s)** detected at {station}.")
        st.dataframe(
            flagged[["year", "value", "expected", "z", "type"]].rename(columns={
                "year": "Year", "value": f"Actual ({unit})",
                "expected": f"Expected ({unit})", "z": "Std. devs from trend",
                "type": "Type"}).style.format({
                    f"Actual ({unit})": "{:.1f}", f"Expected ({unit})": "{:.1f}",
                    "Std. devs from trend": "{:+.2f}"}),
            width="stretch", hide_index=True)
    else:
        st.success("No strong anomalies at this sensitivity \u2014 the record is fairly steady.")


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
# Compare tab -- overlay multiple stations
# ----------------------------------------------------------------------------
with tab_compare:
    st.subheader(f"Compare stations \u2014 {var_label}")
    st.caption("Overlay the annual series of several stations to see who is warming "
               "(or drying) fastest and how they differ.")

    picks = st.multiselect("Stations to compare", stations,
                           default=stations[: min(4, len(stations))])
    if not picks:
        st.info("Select at least one station.")
    else:
        frames = {}
        rank_rows = []
        for st_name in picks:
            a = annual_series(df, st_name, variable)
            a = a[(a["year"] >= year_range[0]) & (a["year"] <= year_range[1])]
            if len(a) < 2:
                continue
            frames[st_name] = a.set_index("year")["value"]
            rank_rows.append({
                "Station": st_name,
                f"Mean ({unit})": a["value"].mean(),
                f"Trend ({unit}/decade)":
                    trend_per_decade(a["year"].values, a["value"].values),
            })
        if frames:
            wide = pd.DataFrame(frames)
            st.line_chart(wide)

            st.markdown("**Anomaly heat-strip** \u2014 how each year compares to each "
                        "station's average (red = above, blue = below).")
            z = (wide - wide.mean()) / wide.std()
            st.dataframe(
                z.T.style.format("{:+.1f}").background_gradient(cmap="coolwarm", axis=None)
                if _HAS_MPL else z.T.style.format("{:+.1f}"),
                width="stretch")

            rank = pd.DataFrame(rank_rows).sort_values(
                f"Trend ({unit}/decade)", ascending=False)
            st.dataframe(rank.style.format({
                f"Mean ({unit})": "{:.1f}", f"Trend ({unit}/decade)": "{:+.2f}"}),
                width="stretch", hide_index=True)


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
# Clusters tab -- K-Means grouping of stations by climate behaviour
# ----------------------------------------------------------------------------
with tab_cluster:
    st.subheader("Station clustering (K-Means)")
    st.caption(
        "K-Means groups stations with similar climate behaviour using four features: "
        "average value, warming/precipitation trend, seasonal swing, and year-to-year "
        "variability. Useful for spotting which stations behave alike."
    )

    feats = station_feature_matrix(variable)
    if len(feats) < 3:
        st.info("Not enough stations to cluster for this variable.")
    else:
        k = st.slider("Number of clusters (k)", 2, min(5, len(feats) - 1), 3)
        feat_cols = ["mean", "trend_decade", "seasonality", "variability"]
        X = StandardScaler().fit_transform(feats[feat_cols].values)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        feats["Cluster"] = km.fit_predict(X).astype(str)

        st.scatter_chart(feats, x="mean", y="trend_decade", color="Cluster",
                         size="seasonality")
        st.caption(f"X = average {var_label.lower()}, Y = trend per decade, "
                   "point size = seasonal swing.")

        show = feats[["station", "Cluster", "elev_m", "mean", "trend_decade",
                      "seasonality", "variability"]].sort_values("Cluster").rename(columns={
            "station": "Station", "elev_m": "Elevation (m)", "mean": f"Mean ({unit})",
            "trend_decade": f"Trend ({unit}/decade)", "seasonality": "Seasonal swing",
            "variability": "Variability"})
        st.dataframe(show.style.format({
            f"Mean ({unit})": "{:.1f}", f"Trend ({unit}/decade)": "{:+.2f}",
            "Seasonal swing": "{:.1f}", "Variability": "{:.2f}"}),
            width="stretch", hide_index=True)

        # Plain-language description of each cluster
        st.markdown("**What each cluster means**")
        for c in sorted(feats["Cluster"].unique()):
            grp = feats[feats["Cluster"] == c]
            names = ", ".join(grp["station"])
            st.markdown(
                f"- **Cluster {c}** ({names}): avg trend "
                f"{grp['trend_decade'].mean():+.2f} {unit}/decade, "
                f"mean elevation {grp['elev_m'].mean():.0f} m."
            )


# ----------------------------------------------------------------------------
# Report tab -- auto-generated narrative + download
# ----------------------------------------------------------------------------
with tab_report:
    st.subheader("\U0001f4dd Auto-generated climate report")
    st.caption("A written summary built live from the data and ML results. "
               "Download it as a text file for your fellowship or research notes.")

    all_tr = all_station_trends(variable)
    warming = variable in ("max_temp", "min_temp")
    verb = "warming" if warming else "wetting/drying"

    if not all_tr.empty:
        fastest = all_tr.loc[all_tr["trend_decade"].idxmax()]
        slowest = all_tr.loc[all_tr["trend_decade"].idxmin()]
        region_trend = all_tr["trend_decade"].mean()
        # correlation between elevation and trend
        if all_tr["elev_m"].nunique() > 2:
            elev_corr = float(np.corrcoef(all_tr["elev_m"], all_tr["trend_decade"])[0, 1])
        else:
            elev_corr = float("nan")
    else:
        fastest = slowest = None
        region_trend = float("nan")
        elev_corr = float("nan")

    lines = []
    lines.append("GILGIT-BALTISTAN CLIMATE ANALYSIS REPORT")
    lines.append("=" * 42)
    lines.append(f"Variable analysed : {var_label}")
    lines.append(f"Period            : {year_range[0]}-{year_range[1]}")
    lines.append(f"Stations          : {len(all_tr)} PMD stations")
    lines.append("")
    lines.append("1. REGIONAL SUMMARY")
    lines.append("-" * 42)
    lines.append(f"Average regional trend: {region_trend:+.2f} {unit}/decade.")
    if fastest is not None:
        lines.append(f"Fastest {verb}: {fastest['station']} "
                     f"({fastest['trend_decade']:+.2f} {unit}/decade, "
                     f"{fastest['elev_m']:.0f} m).")
        lines.append(f"Slowest change : {slowest['station']} "
                     f"({slowest['trend_decade']:+.2f} {unit}/decade).")
    if not np.isnan(elev_corr):
        rel = ("higher stations changing faster" if elev_corr > 0.2
               else "lower stations changing faster" if elev_corr < -0.2
               else "no strong elevation link")
        lines.append(f"Elevation vs trend correlation: {elev_corr:+.2f} ({rel}).")
    lines.append("")
    lines.append(f"2. FOCUS STATION: {station.upper()}")
    lines.append("-" * 42)
    lines.append(f"Trend         : {decade_trend:+.2f} {unit}/decade")
    lines.append(f"Total change  : {total_change:+.2f} {unit} over the period")
    lines.append(f"Highest year  : {int(hottest['year'])} ({hottest['value']:.1f} {unit})")
    lines.append(f"Lowest year   : {int(coldest['year'])} ({coldest['value']:.1f} {unit})")
    anom = detect_anomalies(ann, 1.5)
    n_anom = int(anom["anomaly"].sum())
    lines.append(f"Anomalous yrs : {n_anom} (>1.5 std devs from trend)")
    lines.append("")
    lines.append("3. WHY IT MATTERS")
    lines.append("-" * 42)
    if warming:
        lines.append("Rising temperatures across Gilgit-Baltistan accelerate glacial")
        lines.append("melt, swell unstable glacial lakes, and raise the risk of Glacial")
        lines.append("Lake Outburst Floods (GLOFs) for downstream communities. High-")
        lines.append("elevation stations near the Karakoram glaciers deserve priority")
        lines.append("monitoring and early-warning investment.")
    else:
        lines.append("Shifts in precipitation change how fast glaciers gain or lose mass")
        lines.append("and affect water security for millions who depend on glacier-fed")
        lines.append("rivers. Changing rain/snow balance also alters flood and drought risk.")
    lines.append("")
    lines.append("Method: annual values per station (complete years only); Linear")
    lines.append("Regression for trend and projection; K-Means for station grouping;")
    lines.append("residual z-scores for anomaly detection. Demonstration analysis on")
    lines.append("station records - not an operational forecast.")

    report_txt = "\n".join(lines)
    st.code(report_txt, language="text")
    st.download_button("\u2b07\ufe0f Download report (.txt)", report_txt,
                       file_name=f"GB_climate_report_{variable}_{year_range[0]}-{year_range[1]}.txt",
                       mime="text/plain")


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
