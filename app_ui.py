import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import date

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Offshore Wind Resource Mapping",
    page_icon="🛰️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    h1, h2, h3 { color: white !important; }
    div[data-testid="stMetricValue"] { color: #00ffd2 !important; }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛰️ Offshore Wind Resource Mapping Using SAR Imagery")
st.caption("Sentinel-1 SAR | ResNet M64RN4 Wind Direction | CMOD5.N Wind Speed | Gujarat Coast, India")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("📅 Date Selection")
target_date = st.sidebar.date_input(
    "Target Date",
    value=date(2024, 6, 15),
    min_value=date(2024, 1, 1),
    max_value=date(2024, 12, 31),
)
target_date_str = target_date.strftime("%Y-%m-%d")

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Bounding Box (Gujarat Coast)")
min_lon = st.sidebar.number_input("Min Longitude (°E)", value=68.0, min_value=60.0, max_value=80.0, step=0.5)
min_lat = st.sidebar.number_input("Min Latitude (°N)",  value=20.0, min_value=10.0, max_value=30.0, step=0.5)
max_lon = st.sidebar.number_input("Max Longitude (°E)", value=74.0, min_value=60.0, max_value=80.0, step=0.5)
max_lat = st.sidebar.number_input("Max Latitude (°N)",  value=24.0, min_value=10.0, max_value=30.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Grid Settings")
grid_spacing = st.sidebar.slider(
    "Grid spacing (km)",
    min_value=10,
    max_value=50,
    value=25,
    step=5,
)

run_pipeline = st.sidebar.button("⚡ Run Wind Analysis", use_container_width=True)

# ── API call ──────────────────────────────────────────────────────────────────
def fetch_wind_vectors(date_str, min_lon, min_lat, max_lon, max_lat, grid_spacing_km):
    payload = {
        "date":            date_str,
        "min_lon":         min_lon,
        "min_lat":         min_lat,
        "max_lon":         max_lon,
        "max_lat":         max_lat,
        "grid_spacing_km": grid_spacing_km,
    }
    try:
        r = requests.post(
            "http://localhost:8000/wind-vectors",
            json=payload,
            timeout=300
        )
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, r.json().get("detail", "Unknown error")
    except requests.exceptions.ConnectionError:
        return None, "API not running. Start it with: py -m uvicorn api_service:app --reload --port 8000"
    except Exception as e:
        return None, str(e)

# ── Wind field plot ───────────────────────────────────────────────────────────
def plot_wind_field(data, min_lon, min_lat, max_lon, max_lat):
    vectors = data["wind_vectors"]
    date    = data["date"]

    lats   = np.array([v["lat"]           for v in vectors])
    lons   = np.array([v["lon"]           for v in vectors])
    u      = np.array([v["u_ms"]          for v in vectors])
    v_comp = np.array([v["v_ms"]          for v in vectors])
    speeds = np.array([v["speed_ms"]      for v in vectors])
    dirs   = np.array([v["direction_deg"] for v in vectors])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0e1117")

    # Quiver plot
    norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
    q = ax.quiver(
        lons, lats, u, v_comp,
        speeds,
        cmap="jet",
        norm=norm,
        scale=150,
        width=0.004,
        headwidth=4,
        headlength=5,
        alpha=0.9,
    )

    # Colorbar
    cbar = plt.colorbar(q, ax=ax, orientation="horizontal",
                        pad=0.05, fraction=0.046)
    cbar.set_label("Wind speed (m/s)", color="white", fontsize=11)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")

    # Grid and labels
    ax.grid(True, linewidth=0.4, alpha=0.3, color="white")
    ax.set_xlim(min_lon - 0.2, max_lon + 0.2)
    ax.set_ylim(min_lat - 0.2, max_lat + 0.2)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=11)
    ax.set_ylabel("Latitude (°N)",  color="white", fontsize=11)
    ax.tick_params(colors="white")

    # Annotations
    ax.text(68.1, 23.8, "Gulf of Kutch",     color="white", fontsize=9, alpha=0.7)
    ax.text(72.0, 20.3, "Gulf of Khambhat",  color="white", fontsize=9, alpha=0.7)

    ax.set_title(
        f"SAR-derived Wind Field | Gujarat Coast | {date}",
        color="white", fontsize=13, pad=15
    )

    # Stats box
    stats = (
        f"Points:      {len(vectors)}\n"
        f"Mean speed:  {speeds.mean():.1f} m/s\n"
        f"Max speed:   {speeds.max():.1f} m/s\n"
        f"Min speed:   {speeds.min():.1f} m/s\n"
        f"Mean dir:    {dirs.mean():.1f}°"
    )
    ax.text(
        0.02, 0.97, stats,
        transform=ax.transAxes,
        fontsize=9, color="white",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6)
    )

    plt.tight_layout()
    return fig

# ── Main display ──────────────────────────────────────────────────────────────
if run_pipeline:
    with st.spinner("Fetching Sentinel-1 SAR data and computing wind vectors..."):
        data, error = fetch_wind_vectors(
            target_date_str, min_lon, min_lat,
            max_lon, max_lat, grid_spacing
        )

    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure the API is running: `py -m uvicorn api_service:app --reload --port 8000`")

    else:
        vectors = data["wind_vectors"]
        speeds  = [v["speed_ms"] for v in vectors]
        dirs    = [v["direction_deg"] for v in vectors]

        # ── KPI metrics row ───────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Grid Points",      f"{data['num_points']}")
        with col2:
            st.metric("Mean Wind Speed",  f"{np.mean(speeds):.1f} m/s")
        with col3:
            st.metric("Max Wind Speed",   f"{np.max(speeds):.1f} m/s")
        with col4:
            st.metric("Mean Direction",   f"{np.mean(dirs):.1f}°")

        st.markdown("---")

        # ── Wind map ──────────────────────────────────────────────────────────
        st.subheader("🗺️ Wind Field Map")
        fig = plot_wind_field(data, min_lon, min_lat, max_lon, max_lat)
        st.pyplot(fig)

        st.markdown("---")

        # ── Data table ────────────────────────────────────────────────────────
        st.subheader("📊 Wind Vector Data Table")
        import pandas as pd
        df = pd.DataFrame(vectors)
        df.columns = ["Latitude", "Longitude", "Speed (m/s)",
                      "Direction (°)", "U (m/s)", "V (m/s)"]
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

        # ── Download button ───────────────────────────────────────────────────
        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download wind vectors as CSV",
            data=csv,
            file_name=f"wind_vectors_{target_date_str}.csv",
            mime="text/csv",
        )

        # ── Source info ───────────────────────────────────────────────────────
        st.markdown("---")
        st.caption(f"Source: {data['source']}")

else:
    # Default landing page
    st.info("👈 Configure the date and bounding box in the sidebar, then click **Run Wind Analysis**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🛰️ Data Source")
        st.write("Sentinel-1 IW GRD (VV polarisation) via Google Earth Engine")
    with col2:
        st.markdown("### 🧠 Wind Direction")
        st.write("ResNet M64RN4 trained on SAR patches (Zanchetta & Zecchetto, 2020)")
    with col3:
        st.markdown("### 💨 Wind Speed")
        st.write("CMOD5.N Geophysical Model Function (Hersbach et al., 2007)")