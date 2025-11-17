# summary_builder.py
import math
import pandas as pd

def compress_scan_ids(ids, compress=True, force_text=True):
    """Return comma-separated scan ids; optionally compress consecutive ranges."""
    ids = sorted({int(x) for x in ids if pd.notna(x)})
    if not ids:
        out = ""
    elif not compress:
        out = ",".join(str(x) for x in ids)
    else:
        ranges = []
        start = prev = ids[0]
        for s in ids[1:]:
            if s == prev + 1:
                prev = s
            else:
                ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
                start = prev = s
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        out = ",".join(ranges)
    if force_text:
        # Leading apostrophe keeps it as text in Excel
        return "'" + out if out and not out.startswith("'") else out
    return out

def _decimals_from_tol(tol: float) -> int:
    """Infer number of decimal places to round to (e.g., 0.01 -> 2)."""
    if tol <= 0:
        return 2
    return max(0, int(round(-math.log10(tol))))

def make_summary_ind(
    df: pd.DataFrame,
    merge_tol_mz: float = 0.01,
    *,
    compress_scans: bool = False,
    force_text: bool = True,
    ion_to_label: dict | None = None,   # <-- stays here
) -> pd.DataFrame:
    """
    Summarize hits by precursor m/z within tolerance AND charge.
    Keeps track of files and scans that contributed to each precursor.
    """
    if df.empty:
        print("Input dataframe is empty, returning unchanged.")
        return df

    df = df.copy()

    # If no mapping provided, use an empty dict so .get(...) still works
    if ion_to_label is None:
        ion_to_label = {}

    # ---------- IMPORTANT PART: ALWAYS (RE)BUILD ion_label IF WE HAVE ion ----------
    if "ion" in df.columns:
        def _map_label(x):
            if pd.isna(x):
                return ""
            try:
                # try mapping; fall back to formatted numeric
                return ion_to_label.get(float(x), f"{float(x):.4f}")
            except Exception:
                return str(x)

        df["ion_label"] = df["ion"].map(_map_label)

    # ---------- FALLBACK: if no 'ion' column at all ----------
    elif "ion_label" not in df.columns:
        # Ultimate fallback: use precursor m/z as label
        df["ion_label"] = df["precmz"].map(
            lambda x: f"{float(x):.4f}" if pd.notna(x) else ""
        )

    # Round precursors to a bin so "close enough" precursors merge (based on tolerance)
    ndp = _decimals_from_tol(float(merge_tol_mz))
    df["_mz_bin"] = df["precmz"].round(ndp)

    # Group by m/z bin + charge
    grouped = df.groupby(["_mz_bin", "charge"], as_index=False)

    def collect_info(sub):
        return pd.Series({
            "merged_precmz": sub["precmz"].mean(),
            "rt_min": sub["rt"].min(),
            "rt_median": sub["rt"].median(),
            "rt_max": sub["rt"].max(),
            "n_scans": sub["scan"].nunique(),
            "scan_ids": compress_scan_ids(
                sub["scan"].unique(), compress=compress_scans, force_text=force_text
            ),
            "ms1_scan_ids": compress_scan_ids(
                sub["ms1scan"].unique(), compress=compress_scans, force_text=force_text
            ),
            "files": ",".join(sorted(sub["source_file"].unique())),
        })

    summary = grouped.apply(collect_info).reset_index(drop=True)

    # Presence/absence flags per ion_label
    presence = (
        df.groupby(["_mz_bin", "ion_label"])["scan"]
          .size().unstack(fill_value=0).astype(bool).reset_index()
    )
    presence.columns = ["_mz_bin"] + [f"has_{c}" for c in presence.columns if c != "_mz_bin"]

    # Merge presence flags back and drop helper col
    summary = summary.merge(presence, on="_mz_bin", how="left").drop(columns=["_mz_bin"])

    print(f"make_summary_ind: {len(summary)} merged precursors from {len(df)} rows")
    return summary



def make_summary_combo(df: pd.DataFrame, merge_tol_mz: float = 0.01) -> pd.DataFrame:
    """
    Summarize hits by precursor m/z within tolerance AND ion_label identity.
    Keeps track of files and scans that contributed to each precursor.
    """

    if df.empty:
        print("Input dataframe is empty, returning unchanged.")
        return df

    # Round precursors to a bin so "close enough" precursors merge
    df = df.copy()
    df["_mz_bin"] = df["precmz"].round(2)  # adjust precision if needed

    # Group by m/z bin + ion identity + charge
    grouped = df.groupby(["_mz_bin", "combo_label", "charge"], as_index=False)

    def collect_info(sub):
        return pd.Series({
            "merged_precmz": sub["precmz"].mean(),
            "rt_min": sub["rt"].min(),
            "rt_median": sub["rt"].median(),
            "rt_max": sub["rt"].max(),
            "n_scans": sub["scan"].nunique(),
            "scan_ids": ",".join(map(str, sorted(sub["scan"].unique()))),
            "files": ",".join(sorted(sub["source_file"].unique())),
        })

    summary_combo = grouped.apply(collect_info).reset_index(drop=True)

    # Add presence/absence flags per combo_label
    presence = (
        df.groupby(["_mz_bin", "combo_label"])["scan"]
          .size().unstack(fill_value=0).astype(bool).reset_index()
    )
    presence.columns = ["_mz_bin"] + [f"has_{c}" for c in presence.columns if c != "_mz_bin"]

    # Merge presence flags back (safe, no cluster_id needed)
    summary_combo_merge = summary_combo.merge(presence, on="_mz_bin", how="left")

    # Drop helper col
    summary_combo_merge = summary_combo_merge.drop(columns=["_mz_bin"])

    print(f"make_summary_combo: {len(summary_combo_merge)} merged precursors from {len(df)} rows")
    print(summary_combo_merge.head())

    return summary_combo_merge