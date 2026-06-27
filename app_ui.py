import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.interpolate import griddata
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
from shapely.geometry import box
from datetime import date, datetime
import urllib.request
import zipfile
import os
import tempfile

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Offshore Wind Resource Mapping",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
    <style>
    .reportview-container { background: #0e1117; }
    h1, h2, h3 { color: white !important; }
    div[data-testid="stMetricValue"] { color: #00ffd2 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Offshore Wind Resource Mapping Using SAR Imagery")
st.caption("Sentinel-1 SAR | ResNet M64RN4 | CMOD5.N | Gujarat & Tamil Nadu Coast, India")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("📅 Date Selection")
target_date = st.sidebar.date_input(
    "Target Date",
    value=date(2024, 6, 15),
    min_value=date(2024, 1, 1),
    max_value=date(2024, 12, 31),
    format="DD/MM/YYYY",
)
target_date_str = target_date.strftime("%Y-%m-%d")

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Study Area")
region_choice = st.sidebar.selectbox(
    "Select Region",
    ["Gujarat Coast", "Tamil Nadu Coast", "Custom"]
)

if region_choice == "Gujarat Coast":
    min_lon, min_lat, max_lon, max_lat = 68.0, 20.0, 74.0, 24.0
elif region_choice == "Tamil Nadu Coast":
    min_lon, min_lat, max_lon, max_lat = 79.8, 11.5, 81.5, 14.5
else:
    min_lon = st.sidebar.number_input("Min Longitude (°E)", value=68.0, min_value=60.0, max_value=100.0, step=0.5)
    min_lat = st.sidebar.number_input("Min Latitude (°N)",  value=20.0, min_value=5.0,  max_value=35.0,  step=0.5)
    max_lon = st.sidebar.number_input("Max Longitude (°E)", value=74.0, min_value=60.0, max_value=100.0, step=0.5)
    max_lat = st.sidebar.number_input("Max Latitude (°N)",  value=24.0, min_value=5.0,  max_value=35.0,  step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Grid Settings")
grid_spacing = st.sidebar.slider(
    "Grid spacing (km)",
    min_value=10,
    max_value=50,
    value=10,
    step=5,
)

run_pipeline = st.sidebar.button("Run Wind Analysis", use_container_width=True)


# ── Load land shapefile ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_land_mask(min_lon, min_lat, max_lon, max_lat):
    urls = [
        "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
        "https://github.com/nvkelso/natural-earth-vector/raw/master/zips/ne_10m_land.zip",
    ]
    cache_path = os.path.join(tempfile.gettempdir(), "ne_10m_land")
    shp_path   = os.path.join(cache_path, "ne_10m_land.shp")

    if not os.path.exists(shp_path):
        os.makedirs(cache_path, exist_ok=True)
        zip_path   = os.path.join(cache_path, "ne_10m_land.zip")
        downloaded = False
        for url in urls:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp, \
                     open(zip_path, "wb") as f:
                    f.write(resp.read())
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall(cache_path)
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            return None

    try:
        world        = gpd.read_file(shp_path)
        bbox_geom    = box(min_lon, min_lat, max_lon, max_lat)
        land_clipped = gpd.clip(world, bbox_geom)
        return land_clipped
    except Exception:
        return None


def draw_land(ax, land_gdf):
    if land_gdf is None or land_gdf.empty:
        return
    for geom in land_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for poly in polys:
            x, y   = poly.exterior.xy
            coords = np.column_stack([x, y])
            codes  = ([Path.MOVETO] +
                      [Path.LINETO] * (len(coords) - 2) +
                      [Path.CLOSEPOLY])
            path  = Path(coords, codes)
            patch = PathPatch(
                path,
                facecolor="#3a3a3a",
                edgecolor="#555555",
                linewidth=0.5,
                zorder=3,
                transform=ax.transData
            )
            ax.add_patch(patch)


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
        return None, "API not running. Start it with: py -3.11 -m uvicorn api_service:app --reload --port 8000"
    except Exception as e:
        return None, str(e)


# ── Wind field plot ───────────────────────────────────────────────────────────
def plot_wind_field(data):
    vectors = data["wind_vectors"]
    date_   = data["date"]

    lats   = np.array([v["lat"]           for v in vectors])
    lons   = np.array([v["lon"]           for v in vectors])
    u      = np.array([v["u_ms"]          for v in vectors])
    v_comp = np.array([v["v_ms"]          for v in vectors])
    speeds = np.array([v["speed_ms"]      for v in vectors])
    dirs   = np.array([v["direction_deg"] for v in vectors])

    # ── Load land mask ────────────────────────────────────────────────────────
    land_gdf = load_land_mask(
    float(lons.min() - 1.0),
    float(lats.min() - 1.0),
    float(lons.max() + 1.0),
    float(lats.max() + 1.0)
    )

    # ── Filter out land points ────────────────────────────────────────────────
    if land_gdf is not None and not land_gdf.empty:
        try:
            land_union  = unary_union(land_gdf.geometry).buffer(0.05)
            ocean_mask  = np.array([
                not land_union.contains(Point(float(lon), float(lat)))
                for lat, lon in zip(lats, lons)
            ])
            lats   = lats[ocean_mask]
            lons   = lons[ocean_mask]
            u      = u[ocean_mask]
            v_comp = v_comp[ocean_mask]
            speeds = speeds[ocean_mask]
            dirs   = dirs[ocean_mask]
        except Exception:
            pass

    if len(lats) == 0:
        st.warning("No ocean points found in this region.")
        return None

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor("#0a1628")
    fig.patch.set_facecolor("#0e1117")

    # ── Interpolated colour background ────────────────────────────────────────
    if len(lats) >= 4:
        grid_lon = np.linspace(lons.min(), lons.max(), 300)
        grid_lat = np.linspace(lats.min(), lats.max(), 300)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        grid_speed = griddata(
            (lons, lats), speeds,
            (grid_x, grid_y),
            method='linear'
        )

        norm = plt.Normalize(vmin=speeds.min(), vmax=speeds.max())
        cf   = ax.contourf(
            grid_x, grid_y, grid_speed,
            levels=100,
            cmap="jet",
            norm=norm,
            zorder=1,
            alpha=0.85,
        )

        cbar = plt.colorbar(cf, ax=ax, orientation="horizontal",
                            pad=0.05, fraction=0.046)
        cbar.set_label("Wind speed (m/s)", color="white", fontsize=11)
        cbar.ax.xaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.xaxis.get_ticklabels(), color="white")

    # ── Land mask on top ──────────────────────────────────────────────────────
    draw_land(ax, land_gdf)

    # ── Uniform black arrows ──────────────────────────────────────────────────
    magnitude = np.sqrt(u**2 + v_comp**2)
    magnitude = np.where(magnitude == 0, 1, magnitude)
    u_norm    = u / magnitude
    v_norm    = v_comp / magnitude

    ax.quiver(
        lons, lats,
        u_norm, v_norm,
        color="black",
        scale=25,
        width=0.003,
        headwidth=4,
        headlength=5,
        alpha=0.9,
        zorder=5,
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(lons.min() - 0.3, lons.max() + 0.3)
    ax.set_ylim(lats.min() - 0.3, lats.max() + 0.3)
    ax.grid(True, linewidth=0.4, alpha=0.3, color="white", zorder=2)
    ax.set_xlabel("Longitude (°E)", color="white", fontsize=11)
    ax.set_ylabel("Latitude (°N)",  color="white", fontsize=11)
    ax.tick_params(colors="white")

    # ── Title ─────────────────────────────────────────────────────────────────
    date_formatted = datetime.strptime(date_, "%Y-%m-%d").strftime("%d %B %Y")
    ax.set_title(
        f"SAR-derived Wind Field | {date_formatted}",
        color="white", fontsize=13, pad=15
    )

    # ── Stats box ─────────────────────────────────────────────────────────────
    stats = (
        f"Points:      {len(lats)}\n"
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
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
        zorder=6,
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
        st.info("💡 Make sure the API is running: `py -3.11 -m uvicorn api_service:app --reload --port 8000`")

    else:
        vectors = data["wind_vectors"]
        speeds  = [v["speed_ms"]      for v in vectors]
        dirs    = [v["direction_deg"] for v in vectors]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Grid Points",     f"{data['num_points']}")
        with col2:
            st.metric("Mean Wind Speed", f"{np.mean(speeds):.1f} m/s")
        with col3:
            st.metric("Max Wind Speed",  f"{np.max(speeds):.1f} m/s")
        with col4:
            st.metric("Mean Direction",  f"{np.mean(dirs):.1f}°")

        st.markdown("---")

        st.subheader(" Wind Field Map")
        fig = plot_wind_field(data)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)

        st.markdown("---")

        st.subheader(" Wind Vector Data Table")
        import pandas as pd
        df = pd.DataFrame(vectors)
        df.columns = ["Latitude", "Longitude", "Speed (m/s)",
                      "Direction (°)", "U (m/s)", "V (m/s)"]
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

        st.subheader("⬇️ Download Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download wind vectors as CSV",
            data=csv,
            file_name=f"wind_vectors_{target_date_str}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.caption(f"Source: {data['source']}")

else:
    st.info("👈 Select a region and date in the sidebar, then click **Run Wind Analysis**.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Data Source")
        st.write("Sentinel-1 IW GRD (VV polarisation) via Google Earth Engine")
    with col2:
        st.markdown("### Wind Direction")
        st.write("ResNet M64RN4 trained on SAR patches (Zanchetta & Zecchetto, 2020)")
    with col3:
        st.markdown("### Wind Speed")
        st.write("CMOD5.N Geophysical Model Function (Hersbach et al., 2007)")
