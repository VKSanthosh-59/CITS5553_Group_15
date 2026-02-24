from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Dict, Literal, Tuple
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from app.services.io_service import dataframe_from_upload, dataframe_from_upload_cols  # NEW
from pydantic import BaseModel
from app.services.comparisons import COMPARISON_METHODS
from pyproj import Transformer 
from fastapi.responses import StreamingResponse
import io

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

def _clean_and_stats(df: pd.DataFrame, assay_col: str) -> Dict[str, float]:
    if assay_col not in df.columns:
        raise ValueError(f"Column '{assay_col}' not found")
    # coerce to numeric, drop NaNs, drop <= 0
    s = pd.to_numeric(df[assay_col], errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return {"count": 0, "mean": None, "median": None, "max": None, "std": None}
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "max": float(s.max()),
        "std": float(s.std(ddof=1)),  # sample std
    }

@router.post("/summary")
async def summary(
    original: UploadFile = File(..., description="Original Data .csv, .dbf or .zip"),
    dl: UploadFile       = File(..., description="DL Data .csv, .dbf or .zip"),
    original_assay: str  = Form(...),
    dl_assay: str        = Form(...),
):
    try:
        df_o = dataframe_from_upload(original)
        df_d = dataframe_from_upload(dl)
        stats_o = _clean_and_stats(df_o, original_assay)
        stats_d = _clean_and_stats(df_d, dl_assay)
        return {"original": stats_o, "dl": stats_d}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class PlotsResponse(BaseModel):
    original_png: str  # base64 (no data: prefix)
    dl_png: str
    qq_png: str

class PlotsDataResponse(BaseModel):
    original_data: Dict
    dl_data: Dict
    qq_data: Dict

def _clean_series(df: pd.DataFrame, assay_col: str) -> pd.Series:
    if assay_col not in df.columns:
        raise ValueError(f"Column '{assay_col}' not found")
    s = pd.to_numeric(df[assay_col], errors="coerce").dropna()
    s = s[s > 0]
    return s

def _fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

@router.post("/plots", response_model=PlotsResponse)
async def plots(
    original: UploadFile = File(..., description="Original Data .csv, .dbf or .zip"),
    dl: UploadFile       = File(..., description="DL Data .csv, .dbf or .zip"),
    original_assay: str  = Form(...),
    dl_assay: str        = Form(...),
):
    try:
        # read only the assay columns (fast)
        df_o = dataframe_from_upload_cols(original, [original_assay])
        df_d = dataframe_from_upload_cols(dl, [dl_assay])

        s_o = _clean_series(df_o, original_assay)
        s_d = _clean_series(df_d, dl_assay)

        # Histogram (Original) with log-spaced bins
        fig1 = plt.figure(figsize=(7,4))
        ax1 = fig1.add_subplot(111)
        bins_o = np.logspace(np.log10(s_o.min()), np.log10(s_o.max()), 50)
        ax1.hist(s_o, bins=bins_o, color="#7C3AED", edgecolor="black")
        ax1.set_xscale("log")
        ax1.set_title(f"Original {original_assay} Distribution")
        ax1.set_xlabel(original_assay)
        ax1.set_ylabel("Count")
        original_png = _fig_to_b64(fig1)

        # Histogram (DL) with log-spaced bins
        fig2 = plt.figure(figsize=(7,4))
        ax2 = fig2.add_subplot(111)
        bins_d = np.logspace(np.log10(s_d.min()), np.log10(s_d.max()), 50)
        ax2.hist(s_d, bins=bins_d, color="#7C3AED", edgecolor="black")
        ax2.set_xscale("log")
        ax2.set_title(f"DL {dl_assay} Distribution")
        ax2.set_xlabel(dl_assay)
        ax2.set_ylabel("Count")
        dl_png = _fig_to_b64(fig2)

        # QQ plot (log–log)
        q = np.linspace(0.01, 0.99, 50)
        qo = np.quantile(s_o, q)
        qd = np.quantile(s_d, q)
        fig3 = plt.figure(figsize=(6,6))
        ax3 = fig3.add_subplot(111)
        ax3.scatter(qo, qd, s=20, color="#7C3AED")
        line = np.linspace(min(qo.min(), qd.min()), max(qo.max(), qd.max()), 100)
        ax3.plot(line, line, "--", linewidth=1)
        ax3.set_xscale("log"); ax3.set_yscale("log")
        ax3.set_title("QQ Plot (log–log): Original vs DL")
        ax3.set_xlabel(f"Original {original_assay} quantiles")
        ax3.set_ylabel(f"DL {dl_assay} quantiles")
        qq_png = _fig_to_b64(fig3)

        return {"original_png": original_png, "dl_png": dl_png, "qq_png": qq_png}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/plots-data", response_model=PlotsDataResponse)
async def plots_data(
    original: UploadFile = File(..., description="Original Data .csv, .dbf or .zip"),
    dl: UploadFile       = File(..., description="DL Data .csv, .dbf or .zip"),
    original_assay: str  = Form(...),
    dl_assay: str        = Form(...),
):
    try:
        # read only the assay columns (fast)
        df_o = dataframe_from_upload_cols(original, [original_assay])
        df_d = dataframe_from_upload_cols(dl, [dl_assay])

        s_o = _clean_series(df_o, original_assay)
        s_d = _clean_series(df_d, dl_assay)

        # Original histogram data - match matplotlib exactly
        bins_o = np.logspace(np.log10(s_o.min()), np.log10(s_o.max()), 50)
        hist_o, bin_edges_o = np.histogram(s_o, bins=bins_o)
        bin_centers_o = (bin_edges_o[:-1] + bin_edges_o[1:]) / 2
        
        original_data = {
            "x": bin_centers_o.tolist(),
            "y": hist_o.tolist(),
            "bin_edges": bin_edges_o.tolist(),
            "title": f"Original {original_assay} Distribution",
            "xlabel": original_assay,
            "ylabel": "Count",
            "log_x": True
        }

        # DL histogram data - match matplotlib exactly
        bins_d = np.logspace(np.log10(s_d.min()), np.log10(s_d.max()), 50)
        hist_d, bin_edges_d = np.histogram(s_d, bins=bins_d)
        bin_centers_d = (bin_edges_d[:-1] + bin_edges_d[1:]) / 2
        
        dl_data = {
            "x": bin_centers_d.tolist(),
            "y": hist_d.tolist(),
            "bin_edges": bin_edges_d.tolist(),
            "title": f"DL {dl_assay} Distribution",
            "xlabel": dl_assay,
            "ylabel": "Count",
            "log_x": True
        }

        # QQ plot data
        q = np.linspace(0.01, 0.99, 50)
        qo = np.quantile(s_o, q)
        qd = np.quantile(s_d, q)
        
        # Create reference line data
        min_val = min(qo.min(), qd.min())
        max_val = max(qo.max(), qd.max())
        line_x = np.logspace(np.log10(min_val), np.log10(max_val), 100)
        line_y = line_x
        
        qq_data = {
            "x": qo.tolist(),
            "y": qd.tolist(),
            "line_x": line_x.tolist(),
            "line_y": line_y.tolist(),
            "title": "QQ Plot (log–log): Original vs DL",
            "xlabel": f"Original {original_assay} quantiles",
            "ylabel": f"DL {dl_assay} quantiles",
            "log_x": True,
            "log_y": True
        }

        return {
            "original_data": original_data,
            "dl_data": dl_data,
            "qq_data": qq_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Helpers for grid comparison (replace old single-cell helpers) ---
def _grid_meta_xy(east: pd.Series, north: pd.Series, cell_x: float, cell_y: float):
    xmin = float(np.floor(east.min()  / cell_x) * cell_x)
    ymin = float(np.floor(north.min() / cell_y) * cell_y)
    nx   = int(((east.max()  - xmin) // cell_x) + 1)
    ny   = int(((north.max() - ymin) // cell_y) + 1)
    return xmin, ymin, nx, ny

def _index_cols_xy(df: pd.DataFrame, easting: str, northing: str,
                   xmin: float, ymin: float, cell_x: float, cell_y: float) -> pd.DataFrame:
    df = df.copy()
    df["grid_ix"] = ((df[easting]  - xmin) // cell_x).astype(int)
    df["grid_iy"] = ((df[northing] - ymin) // cell_y).astype(int)
    return df

def _looks_like_degrees(east: pd.Series, north: pd.Series) -> bool:
    return bool(east.between(-180, 180).all() and north.between(-90, 90).all())

def _to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

@router.post("/comparison")
async def comparison(
    original: UploadFile = File(...),
    dl: UploadFile       = File(...),
    original_northing: str = Form(...),
    original_easting: str  = Form(...),
    original_assay: str    = Form(...),
    dl_northing: str       = Form(...),
    dl_easting: str        = Form(...),
    dl_assay: str          = Form(...),
    method: Literal["mean","median","max"] = Form(...),
    grid_size: float       = Form(...),
    treat_as: Literal["auto","meters","degrees"] = Form("auto"),
):
    try:
        # 1) Read & clean
        df_o = dataframe_from_upload_cols(original, [original_easting, original_northing, original_assay])
        df_d = dataframe_from_upload_cols(dl,       [dl_easting,       dl_northing,       dl_assay])

        df_o = _to_float(df_o, [original_easting, original_northing, original_assay])
        df_d = _to_float(df_d, [dl_easting, dl_northing, dl_assay])
        df_o = df_o[df_o[original_assay] > 0]
        df_d = df_d[df_d[dl_assay] > 0]
        if df_o.empty or df_d.empty:
            raise ValueError("No valid rows after cleaning (assay <= 0 removed).")

        # 2) Decide units, and project to meters if inputs are degrees
        looks_deg = _looks_like_degrees(
            pd.concat([df_o[original_easting], df_d[dl_easting]], ignore_index=True),
            pd.concat([df_o[original_northing], df_d[dl_northing]], ignore_index=True),
        )
        use_degrees = (treat_as == "degrees") or (treat_as == "auto" and looks_deg)

        if use_degrees:
            # Project lon/lat (EPSG:4326) → meters (EPSG:3577)
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)
            ex_o, ny_o = transformer.transform(df_o[original_easting].values, df_o[original_northing].values)
            ex_d, ny_d = transformer.transform(df_d[dl_easting].values,       df_d[dl_northing].values)
            df_o["__E_m"], df_o["__N_m"] = ex_o, ny_o
            df_d["__E_m"], df_d["__N_m"] = ex_d, ny_d
            e_col_o, n_col_o = "__E_m", "__N_m"
            e_col_d, n_col_d = "__E_m", "__N_m"
            coord_units = "meters"
        else:
            # Already meters
            e_col_o, n_col_o = original_easting, original_northing
            e_col_d, n_col_d = dl_easting,       dl_northing
            coord_units = "meters"

        # 3) Grid meta (meters)
        cell_x = cell_y = float(grid_size)
        e_all = pd.concat([df_o[e_col_o], df_d[e_col_d]], ignore_index=True)
        n_all = pd.concat([df_o[n_col_o], df_d[n_col_d]], ignore_index=True)
        xmin, ymin, nx, ny = _grid_meta_xy(e_all, n_all, cell_x, cell_y)

        # 4) Index + rename assay → Te_ppm
        o_idx = _index_cols_xy(df_o, e_col_o, n_col_o, xmin, ymin, cell_x, cell_y)\
                  .rename(columns={original_assay: "Te_ppm"})
        d_idx = _index_cols_xy(df_d, e_col_d, n_col_d, xmin, ymin, cell_x, cell_y)\
                  .rename(columns={dl_assay: "Te_ppm"})

        # 5) Compute arrays via registry
        fn = COMPARISON_METHODS[method]
        arr_orig, arr_dl, arr_cmp = fn(d_idx, o_idx, nx, ny)

        # 6) Downsample overlay points (in meters)
        def _downsample(df, n=5000):
            return df.sample(n=min(n, len(df)), random_state=42)
        o_pts = _downsample(o_idx)[[e_col_o, n_col_o]].values.tolist()
        d_pts = _downsample(d_idx)[[e_col_d, n_col_d]].values.tolist()

        # 7) Grid centers (meters)
        x = (xmin + (np.arange(nx) + 0.5) * cell_x).tolist()
        y = (ymin + (np.arange(ny) + 0.5) * cell_y).tolist()

        # 8) JSON-safe arrays
        def _to_jsonable(a: np.ndarray):
            return [[(float(v) if np.isfinite(v) else None) for v in row] for row in a]

        return {
            "nx": nx, "ny": ny,
            "xmin": float(xmin), "ymin": float(ymin),
            "cell": float(grid_size),
            "cell_x": float(cell_x),
            "cell_y": float(cell_y),
            "x": x, "y": y,
            "coord_units": coord_units,
            "mean_lat": None,
            "orig": _to_jsonable(arr_orig),
            "dl":   _to_jsonable(arr_dl),
            "cmp":  _to_jsonable(arr_cmp),
            "original_points": o_pts,
            "dl_points": d_pts,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/export/plots")
async def export_plots(
    original_file: UploadFile = File(...),
    dl_file: UploadFile = File(...),
    original_assay: str = Form(...),
    dl_assay: str = Form(...),
    selected_plots: str = Form(...),  # JSON string from the UI
    # optional heatmap params (sent only when user selects heatmaps)
    original_northing: str | None = Form(None),
    original_easting: str | None = Form(None),
    dl_northing: str | None = Form(None),
    dl_easting: str | None = Form(None),
    method: Literal["mean", "median", "max"] | None = Form(None),
    grid_size: float | None = Form(None),
    # optional legend configuration
    legend_config: str | None = Form(None),
):
    """
    Regenerates the selected plots and returns them in a ZIP.
    Histogram + QQ use assay columns only.
    Heatmaps reuse the same gridding logic as /comparison.
    """
    import io, json, zipfile

    try:
        flags = json.loads(selected_plots)  # { originalHistogram: bool, ... }
        
        # Parse legend configuration if provided
        legend_config = json.loads(legend_config) if legend_config else None
        if legend_config:
            # Set defaults for missing values
            default_config = {
                "original": {"min": None, "max": None, "auto": True},
                "dl": {"min": None, "max": None, "auto": True},
                "comparison": {"min": None, "max": None, "auto": True}
            }
            for plot_type in ["original", "dl", "comparison"]:
                if plot_type not in legend_config:
                    legend_config[plot_type] = default_config[plot_type]
                else:
                    for key in ["min", "max", "auto"]:
                        if key not in legend_config[plot_type]:
                            legend_config[plot_type][key] = default_config[plot_type][key]

        images: list[tuple[str, bytes]] = []

        # ---- 1) Histograms + QQ (if requested) ----
        if flags.get("originalHistogram") or flags.get("dlHistogram") or flags.get("qqPlot"):
            df_o = dataframe_from_upload_cols(original_file, [original_assay])
            df_d = dataframe_from_upload_cols(dl_file, [dl_assay])

            s_o = pd.to_numeric(df_o[original_assay], errors="coerce").dropna()
            s_o = s_o[s_o > 0]
            s_d = pd.to_numeric(df_d[dl_assay], errors="coerce").dropna()
            s_d = s_d[s_d > 0]

            if flags.get("originalHistogram"):
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                bins = np.logspace(np.log10(s_o.min()), np.log10(s_o.max()), 50)
                ax.hist(s_o, bins=bins, color="#7C3AED", edgecolor="black", linewidth=0.5)
                ax.set_xscale("log")
                ax.set_title(f"Original {original_assay} Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel(original_assay, fontsize=12); ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=150)
                plt.close(fig); buf.seek(0)
                images.append(("original_histogram.png", buf.read()))

            if flags.get("dlHistogram"):
                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                bins = np.logspace(np.log10(s_d.min()), np.log10(s_d.max()), 50)
                ax.hist(s_d, bins=bins, color="#7C3AED", edgecolor="black", linewidth=0.5)
                ax.set_xscale("log")
                ax.set_title(f"DL {dl_assay} Distribution", fontsize=14, fontweight='bold')
                ax.set_xlabel(dl_assay, fontsize=12); ax.set_ylabel("Count", fontsize=12)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=150)
                plt.close(fig); buf.seek(0)
                images.append(("dl_histogram.png", buf.read()))

            if flags.get("qqPlot"):
                q = np.linspace(0.01, 0.99, 50)
                qo = np.quantile(s_o, q); qd = np.quantile(s_d, q)
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.scatter(qo, qd, s=20, color="#7C3AED")
                line = np.linspace(min(qo.min(), qd.min()), max(qo.max(), qd.max()), 100)
                ax.plot(line, line, "--", linewidth=1, color="black")
                ax.set_xscale("log"); ax.set_yscale("log")
                ax.set_title("QQ Plot (log–log): Original vs DL", fontsize=14, fontweight='bold')
                ax.set_xlabel(f"Original {original_assay} quantiles", fontsize=12)
                ax.set_ylabel(f"DL {dl_assay} quantiles", fontsize=12)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=150)
                plt.close(fig); buf.seek(0)
                images.append(("qq_plot.png", buf.read()))

        # ---- 2) Heatmaps (if any heatmap was requested) ----
        if flags.get("originalHeatmap") or flags.get("dlHeatmap") or flags.get("comparisonHeatmap"):
            # Require the mapping + grid params
            required = [original_northing, original_easting, dl_northing, dl_easting, method, grid_size]
            if any(v in (None, "") for v in required):
                raise HTTPException(status_code=400, detail="Heatmaps selected but missing mapping/method/grid parameters.")

            # Reuse the same data cleaning + projection + gridding as /comparison
            # (these helpers already exist above in this file)
            df_o = dataframe_from_upload_cols(original_file, [original_easting, original_northing, original_assay])
            df_d = dataframe_from_upload_cols(dl_file, [dl_easting, dl_northing, dl_assay])

            # clean (reuse module-level helper)
            df_o = _to_float(df_o, [original_easting, original_northing, original_assay])
            df_d = _to_float(df_d, [dl_easting, dl_northing, dl_assay])
            df_o = df_o[df_o[original_assay] > 0]
            df_d = df_d[df_d[dl_assay] > 0]
            if df_o.empty or df_d.empty:
                raise HTTPException(status_code=400, detail="No valid rows after cleaning for heatmaps.")

            # decide units and possibly project (reuse module-level helper)

            looks_deg = _looks_like_degrees(
                pd.concat([df_o[original_easting], df_d[dl_easting]], ignore_index=True),
                pd.concat([df_o[original_northing], df_d[dl_northing]], ignore_index=True),
            )
            use_degrees = looks_deg  # treat_as="auto"

            if use_degrees:
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)
                ex_o, ny_o = transformer.transform(df_o[original_easting].values, df_o[original_northing].values)
                ex_d, ny_d = transformer.transform(df_d[dl_easting].values, df_d[dl_northing].values)
                df_o["__E_m"], df_o["__N_m"] = ex_o, ny_o
                df_d["__E_m"], df_d["__N_m"] = ex_d, ny_d
                e_col_o, n_col_o = "__E_m", "__N_m"
                e_col_d, n_col_d = "__E_m", "__N_m"
            else:
                e_col_o, n_col_o = original_easting, original_northing
                e_col_d, n_col_d = dl_easting, dl_northing

            # grid meta (reuse module-level helper)
            cell_x = cell_y = float(grid_size)
            xmin, ymin, nx, ny = _grid_meta_xy(
                pd.concat([df_o[e_col_o], df_d[e_col_d]], ignore_index=True),
                pd.concat([df_o[n_col_o], df_d[n_col_d]], ignore_index=True),
                cell_x, cell_y
            )

            # index into grid and rename assay to Te_ppm (reuse module-level helper)

            o_idx = _index_cols_xy(df_o, e_col_o, n_col_o, xmin, ymin, cell_x, cell_y).rename(columns={original_assay: "Te_ppm"})
            d_idx = _index_cols_xy(df_d, e_col_d, n_col_d, xmin, ymin, cell_x, cell_y).rename(columns={dl_assay: "Te_ppm"})

            # aggregate via registry (same as /comparison)
            fn = COMPARISON_METHODS[method]  # type: ignore[arg-type]
            arr_orig, arr_dl, arr_cmp = fn(d_idx, o_idx, nx, ny)

            # draw heatmaps
            x = xmin + (np.arange(nx) + 0.5) * cell_x
            y = ymin + (np.arange(ny) + 0.5) * cell_y

            # Downsample overlay points (same as interactive plots)
            def _downsample(df, n=5000):
                return df.sample(n=min(n, len(df)), random_state=42)
            o_pts = _downsample(o_idx)[[e_col_o, n_col_o]].values
            d_pts = _downsample(d_idx)[[e_col_d, n_col_d]].values

            def _save_fig(fig, name):
                b = io.BytesIO(); fig.tight_layout(pad=2.0); fig.savefig(b, format="png", dpi=150); plt.close(fig); b.seek(0)
                images.append((name, b.read()))
            
            # Helper functions for dynamic legend scaling
            def calculate_data_range(data):
                """Calculate min/max range for linear data"""
                finite_data = data[np.isfinite(data)]
                if len(finite_data) == 0:
                    return 0, 1
                return float(np.min(finite_data)), float(np.max(finite_data))
            
            def calculate_log_range(data):
                """Calculate min/max range for log-scale data"""
                finite_positive = data[np.isfinite(data) & (data > 0)]
                if len(finite_positive) == 0:
                    return -2, 3
                log_data = np.log10(finite_positive)
                return float(np.floor(np.min(log_data))), float(np.ceil(np.max(log_data)))
            
            def get_legend_range(plot_type, data, method):
                """Get legend range based on configuration and method"""
                if not legend_config or plot_type not in legend_config:
                    # Default behavior
                    if plot_type == "comparison":
                        if method == "max":
                            return -100, 100, None, None
                        else:
                            min_val, max_val = calculate_data_range(data)
                            padding = max(0.1 * (max_val - min_val), 0.1)
                            return min_val - padding, max_val + padding, None, None
                    else:
                        min_log, max_log = calculate_log_range(data)
                        ticks = list(range(int(min_log), int(max_log) + 1))
                        tick_labels = [f'10{superscript(i)}' for i in ticks]
                        return min_log, max_log, ticks, tick_labels
                
                config = legend_config[plot_type]
                
                if plot_type == "comparison":
                    if method == "max" and config["auto"]:
                        return -100, 100, None, None
                    elif not config["auto"] and config["min"] is not None and config["max"] is not None:
                        return config["min"], config["max"], None, None
                    else:
                        min_val, max_val = calculate_data_range(data)
                        padding = max(0.1 * (max_val - min_val), 0.1)
                        return min_val - padding, max_val + padding, None, None
                else:
                    if not config["auto"] and config["min"] is not None and config["max"] is not None:
                        ticks = list(range(int(config["min"]), int(config["max"]) + 1))
                        tick_labels = [f'10{superscript(i)}' for i in ticks]
                        return config["min"], config["max"], ticks, tick_labels
                    else:
                        min_log, max_log = calculate_log_range(data)
                        ticks = list(range(int(min_log), int(max_log) + 1))
                        tick_labels = [f'10{superscript(i)}' for i in ticks]
                        return min_log, max_log, ticks, tick_labels
            
            def superscript(n):
                """Convert number to superscript text"""
                superscript_map = {
                    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
                    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"
                }
                return ''.join(superscript_map.get(c, c) for c in str(n))

            if flags.get("originalHeatmap"):
                fig = plt.figure(figsize=(12, 7)); ax = fig.add_subplot(111)
                z = np.where(np.isfinite(arr_orig) & (arr_orig > 0), np.log10(arr_orig), np.nan)

                
                # Get dynamic legend range
                vmin, vmax, ticks, tick_labels = get_legend_range("original", arr_orig, method)
                
                im = ax.imshow(z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], 
                              aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
                # Add black dots for data points (downsampled for consistency)
                ax.scatter(o_pts[:, 0], o_pts[:, 1], c='black', s=4, alpha=0.7, marker='o')
                ax.set_title("Original (log10)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Easting (m)", fontsize=12)
                ax.set_ylabel("Northing (m)", fontsize=12)
                
                # Format axes with thousands separators
                ax.ticklabel_format(style='plain', axis='both')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                
                # Rotate x-axis labels to prevent overlap
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # Create colorbar with proper formatting
                cbar = fig.colorbar(im, ax=ax, label=f"Max {original_assay} (log scale)")
                cbar.set_label(f"Max {original_assay} (log scale)", fontsize=12)
                # Set colorbar ticks dynamically
                if ticks and tick_labels:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(tick_labels)
                
                _save_fig(fig, "original_heatmap.png")

            if flags.get("dlHeatmap"):
                fig = plt.figure(figsize=(12, 7)); ax = fig.add_subplot(111)
                z = np.where(np.isfinite(arr_dl) & (arr_dl > 0), np.log10(arr_dl), np.nan)
                
                # Get dynamic legend range
                vmin, vmax, ticks, tick_labels = get_legend_range("dl", arr_dl, method)
                
                im = ax.imshow(z, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], 
                              aspect="equal", cmap="viridis", vmin=vmin, vmax=vmax)
                # Add black dots for data points
                ax.scatter(d_pts[:, 0], d_pts[:, 1], c='black', s=4, alpha=0.7, marker='o')
                ax.set_title("DL (log10)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Easting (m)", fontsize=12)
                ax.set_ylabel("Northing (m)", fontsize=12)
                
                # Format axes with thousands separators
                ax.ticklabel_format(style='plain', axis='both')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                
                # Rotate x-axis labels to prevent overlap
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # Create colorbar with proper formatting
                cbar = fig.colorbar(im, ax=ax, label=f"Max {dl_assay} (log scale)")
                cbar.set_label(f"Max {dl_assay} (log scale)", fontsize=12)
                # Set colorbar ticks dynamically
                if ticks and tick_labels:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(tick_labels)
                
                _save_fig(fig, "dl_heatmap.png")

            if flags.get("comparisonHeatmap"):
                fig = plt.figure(figsize=(12, 7)); ax = fig.add_subplot(111)
                
                # Get dynamic legend range
                vmin, vmax, ticks, tick_labels = get_legend_range("comparison", arr_cmp, method)
                
                # Use the same custom colorscale as the interactive plot
                from matplotlib.colors import LinearSegmentedColormap
                colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', 
                         '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
                n_bins = len(colors)
                custom_cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
                
                im = ax.imshow(arr_cmp, origin="lower", vmin=vmin, vmax=vmax, cmap=custom_cmap,
                               extent=[x.min(), x.max(), y.min(), y.max()], aspect="equal")
                # Add black dots for both original and DL data points
                ax.scatter(o_pts[:, 0], o_pts[:, 1], c='black', s=4, alpha=0.7, marker='o', label='Original')
                ax.scatter(d_pts[:, 0], d_pts[:, 1], c='black', s=4, alpha=0.7, marker='o', label='DL')
                ax.set_title("Comparison (DL − Original)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Easting (m)", fontsize=12)
                ax.set_ylabel("Northing (m)", fontsize=12)
                
                # Format axes with thousands separators
                ax.ticklabel_format(style='plain', axis='both')
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
                
                # Rotate x-axis labels to prevent overlap
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add grid
                ax.grid(True, alpha=0.3, linewidth=0.5)
                
                # Create colorbar with proper formatting
                cbar = fig.colorbar(im, ax=ax, label="Δ Te_ppm")
                cbar.set_label("Δ Te_ppm", fontsize=12)
                # Set colorbar ticks dynamically
                if ticks and tick_labels:
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(tick_labels)
                else:
                    # Default ticks for comparison plot
                    tick_range = max(abs(vmin), abs(vmax))
                    if tick_range <= 100:
                        cbar.set_ticks([-100, -75, -50, -25, 0, 25, 50, 75, 100])
                    else:
                        # Generate ticks based on range
                        step = tick_range / 4
                        ticks = [vmin + i * step for i in range(5)]
                        cbar.set_ticks(ticks)
                
                _save_fig(fig, "comparison_heatmap.png")

        # ---- 3) Package everything requested into a ZIP ----
        if not images:
            raise HTTPException(status_code=400, detail="No plots were selected.")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            for fname, data in images:
                zf.writestr(fname, data)
        zbuf.seek(0)
        return StreamingResponse(
            zbuf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=plots.zip"},
        )

    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))


