#!/usr/bin/env python

# ==============================
# Imports & working directory
# ==============================

import os
from datetime import datetime
import glob
import mimetypes
import time

import pandas as pd
import streamlit as st

import re
from pathlib import Path

# Base dir for saving uploaded / fixed files (your ms2_project)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_mzml"
UPLOAD_DIR.mkdir(exist_ok=True)

os.chdir(BASE_DIR)

import massql_utils as mu
from rt_histograms import plot_rt_histograms, load_latest_hits
from cyanopeptide_counts_plots import plot_indiv_counts
from rt_mz_plot import plot_precursor_rt, plot_per_file_legend
from plotting_ind_heatmap import plot_heatmaps
from filter_precursor_range_script import filter_precursor_range
from summary_builder import make_summary_ind
from indiv_combo_dot_plot import plot_indiv_scatter
import adduct_pipeline as ap
import adduct_finder as af
from cyanometdb_match import (
    load_library,
    read_any_table,
    match_ms1_to_lib,
    select_ms1_columns,
    _latest_indiv_summary,
    write_outputs,
    plot_matched_tiles,
)
from sum_intensity_from_scans import sum_intensities
from ms2_tilemap_intensities import plot_has_tilemap_from_latest
from group_run_files import group_files_by_timestamp


def save_and_fix_uploaded_mzml(uploaded_file) -> str:
    """
    Save an uploaded mzML under ms2_project/uploaded_mzml,
    and fix spectrum IDs like 'merged=2709 row=0' -> 'scan=2709'
    so massql's loader doesn't crash. Returns the saved file path.
    """
    import streamlit as st  # local import to avoid circulars

    name = uploaded_file.name
    out_path = UPLOAD_DIR / name

    raw_bytes = uploaded_file.getvalue()

    # Try to treat as text (mzML is XML text)
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Weird encoding? just save as-is and hope it already has good IDs
        with open(out_path, "wb") as f:
            f.write(raw_bytes)
        st.warning(f"Could not decode {name} as UTF-8; saved without ID fix.")
        return str(out_path)

    # Replace id="merged=2711 row=0" -> id="scan=2711"
    pattern = re.compile(r'id="merged=(\d+)\s+row=\d+"')
    fixed_text, n = pattern.subn(r'id="scan=\1"', text)

    if n > 0:
        st.info(f"Fixed {n} merged spectrum IDs in uploaded file: {name}")
    else:
        st.info(f"No 'merged=... row=...' IDs found in uploaded file: {name}")

    with open(out_path, "wb") as f:
        f.write(fixed_text.encode("utf-8"))

    return str(out_path)


def find_latest_by_prefix(
    prefix: str,
    exts,
    min_mtime: float | None = None,
    root_dir: str | Path = BASE_DIR,
):
    """
    Search root_dir recursively for the newest file whose name starts with `prefix`
    and ends with one of the extensions in `exts`.
    If min_mtime is given, only consider files modified at or after that time.
    Returns the full path or None.
    """
    root_dir = str(root_dir)
    best_path = None
    best_mtime = -1

    for root, dirs, files in os.walk(root_dir):
        for name in files:
            if not name.startswith(prefix):
                continue
            if not any(name.lower().endswith(e) for e in exts):
                continue

            full_path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue

            if min_mtime is not None and mtime < min_mtime:
                continue

            if mtime > best_mtime:
                best_mtime = mtime
                best_path = full_path

    return best_path


# ==============================
# Config / constants
# ==============================

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
LIB_XLSX = BASE_DIR / "CyanoMetDB_Version03.xlsx"
#LIB_XLSX = os.path.join(BASE_DIR, "CyanoMetDB_Version03.xlsx")

ION_TO_LABEL = {
    184.06:   "MP NMeTyrCl 184.06",
    215.1192: "MP AhpPhe-CO-H2O 215.12",
    243.1127: "MP AhpPhe-H2O 243.11",
    134.0961: "MP NMePhe 134.10",
    181.1331: "MP AhpLxx-CO-H2O 181.13",
    169.0967: "MP AhpThr-CO-H2O 169.10",
    150.0912: "MP NMeTyr 150.09",
    167.1178: "MP AhpVal-CO-H2O 167.12",
    454.15:   "MP Phe-Ahp-complete 454.15",
    135.0804: "MC adda fragment 135.0804",
    163.1113: "MC adda fragment 163.1113",
    213.0870: "MC Mdha 213.0870",
    112.0964: "AR Choi-h2o 112.0964",
    140.1066: "AR Choi 140.1066",
    334.0838: "AR Cl-Hpla-Tyr-CO 334.0838",
    300.1232: "AR Hpla-Tyr-CO 300.1232",
    284.1268: "AR Hlpa-Phe-CO 284.1268",
    250.1440: "AR Hlpa-Leu-CO 250.1440",
    221.1646: "AR Hlpa-Leu 221.1646",
    281.1914: "AR Agma 281.1914",
    314.2199: "AR OH-Choi-Agma 314.2199",
    114.0550: "AB CO + Arg- CH5H3 -CO 114.0550",
    164.1069: "AB NMe-HTyr 164.1069",
    192.1019: "AB CO-NMe-Hty 192.1019",
    233.1285: "AB Phe + MeAla +H 233.1285",
    263.1391: "AB HTyr +MeAla +H 263.1391",
    362.2075: "AB HTyr+ MeAla +Val +H 362.2075",
    136.0759: "MV Tyr 136.0759",
    136.0729: "MV Tyr 136.0729",
    159.0908: "MV Try 159.0908",
    159.0935: "MV Tyr 159.0935",
    120.0807: "AG Phe 120.0807",
    144.0109: "AG Tzc 144.0109",
    188.1428: "AG 188.1428",
    213.0685: "AG Pro-Tzc 213.0685",
    199.0503: "AG Tzl-OH 199.0503",
    267.1483: "AG 267.1483",
    100.1122: "MG Ahoa 100.1122",
    134.0727: "MG Ahoa (Cl) 134.0727",
    168.0341: "MG Ahoa (Di Cl) 168.0341",
    201.9955: "MG Ahoa (Tri Cl) 201.9955",
    114.1278: "MG NMe Ahoa 114.1278",
    148.0888: "MG NMe Ahoa (Cl) 148.0888", 
    182.0494: "MG NMe Ahoa (Di Cl) 182.0494",
    126.1423: "MG Ahda 128.1423",
    162.1039: "MG Ahda (Cl) 162.1039",
    196.0639: "MG Ahda (Di Cl) 196.0639",
    142.1590: "MG NMe Ahda 142.1590",
    176.1195: "MG NMe Ahda (Cl) 176.1196",
    210.0795: "MG NMe Ahda (Di Cl) 201.0795",
}

MCIONS = [135.0804, 163.1113, 213.0870] 
MPIONS = [184.06, 215.1192, 243.1127, 134.0961, 181.1331, 169.0967, 150.0912, 167.1178, 454.15]
ARIONS = [122.0966, 140.1066, 334.0838, 300.1232, 284.1268, 250.1440, 221.1646, 281.1914, 314.2199]
ABIONS = [114.0550, 164.1069, 192.1019, 233.1285, 263.1391, 362.2075]
MVIONS = [136.0757, 136.0729, 159.0908, 159.0935]
AGIONS = [120.0807, 144.0109, 188.1428, 213.0685, 199.0503, 267.1483]
MGIONS = [100.1122, 134.0727, 168.0341, 201.9952, 114.1278, 148.0888, 182.0494, 128.1423, 162.1039, 196.0639, 142.1590, 176.1195, 210.0795]

BASE_DIR = Path(__file__).resolve().parent

# Default files for the UI (edit or empty out if you prefer uploads only)
DEFAULT_FILES_MC = []

DEFAULT_FILES_MP = DEFAULT_FILES_MC
DEFAULT_FILES_AR = DEFAULT_FILES_MC
DEFAULT_FILES_AB = DEFAULT_FILES_MC
DEFAULT_FILES_MV = DEFAULT_FILES_MC
DEFAULT_FILES_AG = DEFAULT_FILES_MC
DEFAULT_FILES_MG = DEFAULT_FILES_MC


CLASS_CONFIGS = {
    "MC": {
        "label": "Microcystin (MC)",
        "ions": MCIONS,
        "default_files": DEFAULT_FILES_MC,
        "label_col": "CyanopeptideClass_MC",
        "class_filter": "Microcystin",
        "class_tag": "MC",
    },
    "MP": {
        "label": "Micropeptin (MP)",
        "ions": MPIONS,
        "default_files": DEFAULT_FILES_MP,
        "label_col": "CyanopeptideClass_MP",
        "class_filter": "Micropeptin",
        "class_tag": "MP",
    },
    "AR": {
        "label": "Aeruginosin (AR)",
        "ions": ARIONS,
        "default_files": DEFAULT_FILES_AR,
        "label_col": "CyanopeptideClass_AR",
        "class_filter": "Aeruginosin",
        "class_tag": "AR",
    },
    "AB": {
        "label": "Anabaenopeptin (AB)",
        "ions": ABIONS,
        "default_files": DEFAULT_FILES_AB,
        "label_col": "CyanopeptideClass_AB",
        "class_filter": "Anabaenopeptin",
        "class_tag": "AB",
    },
    "MV": {
        "label": "Microviridin (MV)",
        "ions": MVIONS,
        "default_files": DEFAULT_FILES_MV,
        "label_col": "CyanopeptideClass_MV",
        "class_filter": "Microviridin",
        "class_tag": "MV",
    },
    "AG": {
        "label": "Aeruginosamide (AG) using 'Other Cyclic Peptide' as label from CyanoMetDB",
        "ions": AGIONS,
        "default_files": DEFAULT_FILES_AG,
        "label_col": "CyanopeptideClass_AG",
        "class_filter": "Other Cyclic Peptide",
        "class_tag": "AG",
    },
    "MG": {
        "label": "Microginin (MG)",
        "ions": MGIONS,
        "default_files": DEFAULT_FILES_MG,
        "label_col": "CyanopeptideClass_MG",
        "class_filter": "Microginin",
        "class_tag": "MG",
    },
}

ALLOWED_EXTS = {
    ".csv", ".tsv", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".svg",
    ".pdf", ".json", ".txt",
}


# ==============================
# Small helpers
# ==============================

def latest_file_in_dir(directory: str, patterns):
    """
    Return the most recently modified file in `directory` matching any pattern
    from `patterns` (a list of glob patterns), or None if nothing matches.
    """
    if not directory or not os.path.isdir(directory):
        return None

    all_paths = []
    for pat in patterns:
        all_paths.extend(glob.glob(os.path.join(directory, pat)))

    if not all_paths:
        return None

    all_paths.sort(key=os.path.getmtime, reverse=True)
    return all_paths[0]


# ==============================
# Core pipeline
# ==============================

def run_class_pipeline(
    files,
    ions,
    ion_to_label,
    label_col: str,
    class_filter: str,
    class_tag: str,
    tol_mz: float = 0.01,
    polarity: str = "POSITIVE",
    rt_window=(2.0, 25),
    min_intensity: float | None = None,
    precursor_mz_min: float = 650.0,
    precursor_mz_max: float = 1400.0,
    run_adduct_pipeline: bool = True,
    lib_tol_mode: str = "Da",   # "Da" or "ppm"
    lib_tol_value: float = 0.1, # default 0.1 Da
):
    """
    Run the repeated analysis block for a given cyanopeptide class.
    Returns key outputs useful for Streamlit (tables, figures, paths).
    """

    # record start time for "latest" file filtering if needed
    run_start_time = time.time()

    # ---- 1) Run searches ----
    massql_query = mu.build_massql_query(
        ions_mz=ions,
        tol_mz=tol_mz,
        polarity=polarity,
        rt_window=None,
    )

    ind_hits = mu.run_across_files_individual(
        files, ions, tol_mz=tol_mz, polarity=polarity, rt_window=rt_window
    )

    # ---- 2) Add labels ----
    ind_hits_l = mu.add_MC_labels(ind_hits, ion_to_label, label_col=label_col)

    # ---- 3) Save CSV of individual hits (in BASE_DIR)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ind_csv = f"individual_hits_{class_tag}_{ts}.csv"
    ind_hits_l.to_csv(ind_csv, index=False)
    ind_csv_path = os.path.abspath(ind_csv)

    # ---- 4) RT histograms ----
    latest_ind, _ = load_latest_hits()
    plot_rt_histograms(latest_ind, ion_to_label, out_dir_root="RT_histogram_plots")

    # ---- 5) Counts ----
    plot_indiv_counts(latest_ind, out_dir="plots_diagnostic_ion_counts")

    # ---- 6) RT vs mz plots ----
    fig_rt, ax_rt = plot_precursor_rt(
        latest_ind,
        ion_to_label=ion_to_label,
        save=True,
    )
    plot_per_file_legend(latest_ind, save=True)

    # ---- 7) Heatmaps ----
    plot_heatmaps(
        latest_ind,
        ion_to_label=ion_to_label,
        save=True,
        out_dir="Heatmap_Individual_ion_plots",
        fmt="png",
    )

    # ---- 8) Summary + dot plot ----
    indiv_summary = make_summary_ind(latest_ind, ion_to_label=ion_to_label)
    fig_dot, ax_dot, dot_path = plot_indiv_scatter(
        indiv_summary,
        out_dir="DotPlot_individual_ion_plots",
        fmt="png",
    )

    # ---- 9) Adduct pipeline ----
    def make_summary_ind_with_labels(df, *args, **kwargs):
        return make_summary_ind(df, *args, ion_to_label=ion_to_label, **kwargs)

    filtered_ind = latest_ind[
        (latest_ind["precmz"] >= precursor_mz_min)
        & (latest_ind["precmz"] <= precursor_mz_max)
    ].copy()

    if filtered_ind is None or filtered_ind.empty:
        print(
            "run_class_pipeline: no features in precursor m/z "
            f"range {precursor_mz_min}–{precursor_mz_max}; skipping adduct pipeline."
        )
        merged_summary = pd.DataFrame()
        merged_edges = pd.DataFrame()
        G = None
    else:
        merged_summary, merged_edges, G = ap.run_merged(
            filtered_ind,
            make_summary_ind=make_summary_ind_with_labels,
            af_module=af,
            mz_col="merged_precmz",
            charge_col="charge",
            rt_col="rt_median",
            out_dir="adduct_outputs",
            save_graph=True,
        )

    #----10) CyanoMetDB matching
    lib_df = load_library(LIB_XLSX, class_filter=class_filter, sheet_index=1)

    try:
        ms1_csv = _latest_indiv_summary()
        ms1_df = read_any_table(ms1_csv)
    except FileNotFoundError:
        # fallback: use the summary we just computed
        ms1_df = indiv_summary.copy()

    ms1_sel = select_ms1_columns(ms1_df)


    # Library matching tolerance
    if lib_tol_mode == "Da":
        matches = match_ms1_to_lib(ms1_sel, lib_df, tol_da=lib_tol_value)
    elif lib_tol_mode.lower() == "ppm":
        matches = match_ms1_to_lib(ms1_sel, lib_df, tol_ppm=lib_tol_value)
    else:
        # Fallback: use Da if mode is weird
        matches = match_ms1_to_lib(ms1_sel, lib_df, tol_da=lib_tol_value)

    matched_only = matches[matches["Compound identifier"].notna()].copy()

    ts2 = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cyanomet_out_dir, cyanomet_paths = write_outputs(
        matches,
        matched_only,
        out_dir="CyanoMetDB_matches_out",
        ts=ts2,
    )

    cyanomet_plot_path = plot_matched_tiles(
        matched_only,
        out_dir_ts=cyanomet_out_dir,
        ts=ts2,
    )

    # ---- 11) Intensities + tilemaps ----
    df_int, intensity_outfile = sum_intensities()
    df_tiles, run_dir, tilemap_summary_file = plot_has_tilemap_from_latest(
        ion_to_label=ion_to_label
    )

    # ---- 12) Group files ----
    group_files_by_timestamp(custom_name=class_tag, minutes=3)

    # ---- 13) Find your specifically-named outputs (THIS RUN ONLY) ----

    # 1) Diagnostic_ion_distribution_individual_(date).png
    diagnostic_individual_png = find_latest_by_prefix(
        "Diagnostic_ion_distribution_individual_",
        [".png", ".jpg", ".jpeg", ".svg"],
        min_mtime=run_start_time,
    )

    # 2) matched_compound_tiles_(date).png
    matched_tiles_png = find_latest_by_prefix(
        "matched_compound_tiles_",
        [".png", ".jpg", ".jpeg", ".svg"],
        min_mtime=run_start_time,
    )

    # 3) Cyanopeptide_detection_intensity_heatmap_adduct_outputs_(date).png
    cyano_heatmap_png = find_latest_by_prefix(
        "Cyanopeptide_detection_intensity_heatmap_adduct_outputs_",
        [".png", ".jpg", ".jpeg", ".svg"],
        min_mtime=run_start_time,
    )
    # 3b) adduct_graph_merged_(date).png  (most recent for this run)
    adduct_graph_png = find_latest_by_prefix(
        "adduct_graph_merged_",
        [".png", ".jpg", ".jpeg", ".svg"],
        min_mtime=run_start_time,
    )

    # 4) indiv_merged_summary_(date).csv / .tsv / .xlsx
    indiv_merged_csv = find_latest_by_prefix(
        "indiv_merged_summary_",
        [".csv", ".tsv", ".xlsx", ".xls"],
        min_mtime=run_start_time,
    )

    indiv_merged_df = None
    if indiv_merged_csv:
        try:
            lower = indiv_merged_csv.lower()
            if lower.endswith(".csv"):
                indiv_merged_df = pd.read_csv(indiv_merged_csv)
            elif lower.endswith(".tsv"):
                indiv_merged_df = pd.read_csv(indiv_merged_csv, sep="\t")
            elif lower.endswith((".xlsx", ".xls")):
                indiv_merged_df = pd.read_excel(indiv_merged_csv)
        except Exception:
            indiv_merged_df = None

    # 5) unknown_features_with_scans_(date).csv / .tsv / .xlsx + PNG
    unknown_features_png = find_latest_by_prefix(
        "unknown_features_with_scans_",
        [".png", ".jpg", ".jpeg"],
        min_mtime=run_start_time,
    )

    unknown_features_csv = find_latest_by_prefix(
        "unknown_features_with_scans_",
        [".csv", ".tsv", ".xlsx", ".xls"],
        min_mtime=run_start_time,
    )

    unknown_features_df = None
    if unknown_features_csv:
        try:
            lower = unknown_features_csv.lower()
            if lower.endswith(".csv"):
                unknown_features_df = pd.read_csv(unknown_features_csv)
            elif lower.endswith(".tsv"):
                unknown_features_df = pd.read_csv(unknown_features_csv, sep="\t")
            elif lower.endswith((".xlsx", ".xls")):
                unknown_features_df = pd.read_excel(unknown_features_csv)
        except Exception:
            unknown_features_df = None

    # ---- 14) Collect all relevant output files (latest run, but broad) ----
    output_dirs = [
        BASE_DIR,
        os.path.join(BASE_DIR, "RT_histogram_plots"),
        os.path.join(BASE_DIR, "plots_diagnostic_ion_counts"),
        os.path.join(BASE_DIR, "Heatmap_Individual_ion_plots"),
        os.path.join(BASE_DIR, "DotPlot_individual_ion_plots"),
        os.path.join(BASE_DIR, "adduct_outputs"),
        os.path.join(BASE_DIR, "CyanoMetDB_matches_out"),
        run_dir,
    ]

    seen = set()
    all_files = []

    for od in output_dirs:
        if not od or not os.path.isdir(od):
            continue
        for root, dirs, filenames in os.walk(od):
            for name in filenames:
                full_path = os.path.join(root, name)
                ext = os.path.splitext(name)[1].lower()
                if ext not in ALLOWED_EXTS:
                    continue
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    continue
                # only include files created/modified during this run
                if mtime < run_start_time:
                    continue
                if full_path not in seen:
                    seen.add(full_path)
                    all_files.append(full_path)

    # ---- 15) Return outputs ----
    return {
        # Directories / paths
        "run_dir": run_dir,
        "ind_csv": ind_csv_path,
        "cyanomet_out_dir": cyanomet_out_dir,
        "cyanomet_paths": cyanomet_paths,
        "cyanomet_plot_path": cyanomet_plot_path,
        "intensity_outfile": intensity_outfile,
        "tilemap_summary_file": tilemap_summary_file,

        # Tables / graph objects
        "ind_hits_l": ind_hits_l,
        "indiv_summary": indiv_summary,
        "merged_summary": merged_summary,
        "merged_edges": merged_edges,
        "adduct_graph": G,
        "library_df": lib_df,
        "ms1_csv": ms1_csv,
        "ms1_df": ms1_df,
        "ms1_selected": ms1_sel,
        "matches": matches,
        "matched_only": matched_only,
        "df_int": df_int,
        "df_tiles": df_tiles,
        "indiv_merged_csv": indiv_merged_csv,
        "indiv_merged_df": indiv_merged_df,
        "unknown_features_csv": unknown_features_csv,
        "unknown_features_df": unknown_features_df,

        # Figures / images
        "fig_rt": fig_rt,
        "fig_dot": fig_dot,
        "dot_path": dot_path,
        "diagnostic_individual_png": diagnostic_individual_png,
        "matched_tiles_png": matched_tiles_png,
        "cyano_heatmap_png": cyano_heatmap_png,
        "unknown_features_png": unknown_features_png,
        "adduct_graph_png": adduct_graph_png,


        # Misc
        "massql_query": massql_query,
        "all_files": all_files,
    }


# ==============================
# Streamlit UI
# ==============================

st.title("MS1/MS2 Detection – Cyanopeptide Pipeline")
# st.sidebar.write("CWD:", os.getcwd())  # uncomment to debug paths

# Initialize session storage for pipeline results
if "results" not in st.session_state:
    st.session_state["results"] = None


class_key = st.selectbox(
    "Cyanopeptide class",
    ["MC", "MP", "AR", "AB", "MV", "AG", "MG"],
    format_func=lambda k: CLASS_CONFIGS[k]["label"],
)

cfg = CLASS_CONFIGS[class_key]

st.subheader("Input mzML files")

input_mode = st.radio(
    "Select input method:",
    ["Upload mzML files", "Use server file paths /nfs/turbo/...mzML"],
    horizontal=True,
)


uploaded_files = []
files_to_use = []

if input_mode == "Upload mzML files":
    st.write("Upload mzML files from your computer. They’ll be fixed and saved under 'uploaded_mzml/'.")
    uploaded_files = st.file_uploader(
        "Upload mzML files",
        type=["mzml", "mzML"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uf in uploaded_files:
            saved_path = save_and_fix_uploaded_mzml(uf)
            files_to_use.append(saved_path)

        st.success(f"{len(files_to_use)} file(s) will be analyzed.")
        st.write("Files that will be analyzed:")
        for p in files_to_use:
            st.code(p)

else:
    st.write(
        f"Running analysis for **{cfg['label']}**.\n\n"
        "Provide mzML file paths on **this server** (one per line), "
        "and/or upload mzML files from your computer."
    )

    # --- server-side paths ---
    files_text = st.text_area(
        "mzML file paths on the server",
        value="",
        height=100,
    )
    server_paths = [line.strip() for line in files_text.splitlines() if line.strip()]

    # --- uploaded files ---
    uploaded_files = st.file_uploader(
        "Or upload mzML files (from your computer)",
        type=["mzML", "mzml"],
        accept_multiple_files=True,
    )

    uploaded_paths = []
    if uploaded_files:
        for uf in uploaded_files:
            saved_path = save_and_fix_uploaded_mzml(uf)
            uploaded_paths.append(saved_path)

    files_to_use = server_paths + uploaded_paths

    st.write("Files that will be analyzed:")
    for p in files_to_use:
        st.code(p)


st.markdown("### Filtering and advanced options")

# RT window
rt_min, rt_max = st.slider(
    "RT window (minutes)",
    min_value=0.0,
    max_value=100.0,
    value=(2.0, 25.0),
    step=0.1,
)

# Precursor m/z filter
col1, col2 = st.columns(2)
with col1:
    precursor_mz_min = st.number_input(
        "Precursor m/z min",
        min_value=0.0,
        max_value=5000.0,
        value=650.0,
        step=1.0,
    )
with col2:
    precursor_mz_max = st.number_input(
        "Precursor m/z max",
        min_value=0.0,
        max_value=5000.0,
        value=1400.0,
        step=1.0,
    )

if precursor_mz_max < precursor_mz_min:
    st.error("Precursor m/z max must be ≥ min!")

# Library tolerance settings
lib_tol_mode = st.radio(
    "Library tolerance type for CyanoMetDB matching",
    ["Da", "ppm"],
    horizontal=True,
)

default_lib_tol = 0.1 if lib_tol_mode == "Da" else 5.0  # e.g. 5 ppm default
lib_tol_value = st.number_input(
    f"Library tolerance ({lib_tol_mode})",
    min_value=0.0001,
    max_value=1000.0,
    value=default_lib_tol,
    step=0.0001 if lib_tol_mode == "Da" else 0.1,
)

# 1) Run the pipeline only when button is clicked
run_clicked = st.button("Run analysis")

if run_clicked:
    actual_files = files_to_use

    if not actual_files:
        st.error("No files provided. Please upload or enter at least one.")
        st.session_state["results"] = None
    else:
        with st.spinner(f"Running {cfg['label']} pipeline..."):
            st.session_state["results"] = run_class_pipeline(
                files=actual_files,
                ions=cfg["ions"],
                ion_to_label=ION_TO_LABEL,
                label_col=cfg["label_col"],
                class_filter=cfg["class_filter"],
                class_tag=cfg["class_tag"],
                # pass through UI options too (see next section)
                rt_window=(rt_min, rt_max),
                precursor_mz_min=precursor_mz_min,
                precursor_mz_max=precursor_mz_max,
                lib_tol_mode=lib_tol_mode,
                lib_tol_value=lib_tol_value,
            )

        st.success("Pipeline finished!")


# 2) Always try to display results *from session_state*
results = st.session_state["results"]

if results is not None:

    # ---------- Individual hits preview ----------
    if "ind_hits_l" in results:
        st.subheader("Individual hits (labeled) – preview")
        st.dataframe(results["ind_hits_l"].head())
    else:
        st.info("No individual hits table returned.")

    # ---------- RT plot ----------
    if results.get("fig_rt") is not None:
        st.subheader("Precursor RT plot")
        st.pyplot(results["fig_rt"])

    # ---------- Dot plot ----------
    if results.get("fig_dot") is not None:
        st.subheader("Individual ion dot plot")
        st.pyplot(results["fig_dot"])

    # ---------- Diagnostic_ion_distribution_individual_(date) ----------
    diag_png = results.get("diagnostic_individual_png")
    if diag_png and os.path.exists(diag_png):
        st.subheader("Diagnostic ion distribution – individual")
        st.image(diag_png, caption=os.path.basename(diag_png))

    # ---------- matched_compound_tiles_(date) ----------
    matched_tiles_png = results.get("matched_tiles_png")
    if matched_tiles_png and os.path.exists(matched_tiles_png):
        st.subheader("Matched compound tiles")
        st.image(matched_tiles_png, caption=os.path.basename(matched_tiles_png))
    else:
        st.info("No 'matches_with_scans_to_cyanometDB' PNG found for this run.")

    # ---------- Unknown features with scans (PNG) ----------
    unknown_png = results.get("unknown_features_png")
    if unknown_png and os.path.exists(unknown_png):
        st.subheader("Unknown features with scans")
        st.image(unknown_png, width='content')
    else:
        st.info("No 'unknown_features_with_scans' PNG found for this run.")
    
    # ---------- adduct_graph_merged_(date).png ----------
    adduct_graph_png = results.get("adduct_graph_png")
    if adduct_graph_png and os.path.exists(adduct_graph_png):
        st.subheader("Adduct graph (merged) if messy please open Parent Adduct Summary Excel")
        st.image(adduct_graph_png, caption=os.path.basename(adduct_graph_png))
    else:
        st.info("No 'adduct_graph_merged_' PNG found for this run.")

    # ---------- Cyanopeptide_detection_intensity_heatmap_adduct_outputs_(date) ----------
    heatmap_png = results.get("cyano_heatmap_png")
    if heatmap_png and os.path.exists(heatmap_png):
        st.subheader("Cyanopeptide detection intensity heatmap")
        st.image(heatmap_png, caption=os.path.basename(heatmap_png))

    # ---------- indiv_merged_summary_(date) ----------
    indiv_merged_df = results.get("indiv_merged_df")
    if indiv_merged_df is not None:
        st.subheader("Individual merged summary")
        st.dataframe(indiv_merged_df.head())

    # ---------- unknown_features_with_scans_(date) CSV/TSV/XLSX ----------
    unknown_features_df = results.get("unknown_features_df")
    if unknown_features_df is not None:
        st.subheader("Unknown features with scans (table)")
        st.dataframe(unknown_features_df.head())

    # ---------- Download individual hits CSV ----------
    if "ind_csv" in results:
        csv_path = results["ind_csv"]

        if not os.path.exists(csv_path):
            matches = glob.glob(f"**/{os.path.basename(csv_path)}", recursive=True)
            if matches:
                csv_path = matches[0]

        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="Download individual hits CSV",
                    data=f,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                )
        else:
            st.error(f"CSV not found: {results['ind_csv']} (even after searching)")

    # ---------- Download any output file ----------
    all_files = results.get("all_files", [])
    if all_files:
        st.subheader("All output files")
        st.write(
            "All files generated for this run (CSVs, PNGs, Excel, etc.) "
            "under the ms2_project directory."
        )

        for path in sorted(all_files):
            if not os.path.exists(path):
                continue

            rel_name = os.path.relpath(path, BASE_DIR)
            mime, _ = mimetypes.guess_type(path)
            if mime is None:
                mime = "application/octet-stream"

            safe_key = f"dl-{rel_name.replace(os.sep, '_')}"
            with open(path, "rb") as f:
                st.download_button(
                    label=f"Download {rel_name}",
                    data=f,
                    file_name=os.path.basename(path),
                    mime=mime,
                    key=safe_key,
                )
    else:
        st.info("No output files recorded in this run.")
else:
    st.info("Run the analysis to see results.")
