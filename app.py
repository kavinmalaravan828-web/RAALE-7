"""
AI-Based Warehouse Space & Material Optimization  |  Hackathon Prototype
=========================================================================
A lightweight Streamlit app that re-slots products into warehouse racks so that
fast-moving items sit closest to the dispatch point, while respecting rack
capacity. Pure Python (pandas + numpy): no paid APIs, no GPU.

HOW TO INSTALL & RUN (Python 3.9+)
----------------------------------
1) (Optional but recommended) create a virtual environment:
       python -m venv venv
       source venv/bin/activate          # macOS / Linux
       venv\\Scripts\\activate             # Windows (PowerShell / cmd)

2) Install the libraries:
       pip install streamlit pandas numpy

3) Run the app from the folder that contains app.py:
       streamlit run app.py

   The browser opens at http://localhost:8501. Click "Load Sample Data" in the
   sidebar (or upload your own two CSVs) and the dashboard appears instantly.

CSV FORMATS
-----------
products.csv  : product_id, category, length, width, height, volume,
                daily_demand, picking_frequency (High/Medium/Low), current_location
locations.csv : location_id, capacity_volume, picking_distance, accessibility_level

Units used by the sample data: dimensions in cm, volume / capacity in m3,
picking_distance in metres from the dispatch point.

ALGORITHM (greedy slotting)
---------------------------
  priority_score = daily_demand * frequency_weight   (High=3, Medium=2, Low=1)
  1. Sort products by priority_score, highest first.
  2. Sort racks by picking_distance, nearest first (accessibility breaks ties).
  3. For each product, pick the nearest rack whose free capacity >= product volume.
  4. No rack fits -> NO_SLOT_AVAILABLE.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NO_SLOT = "NO_SLOT_AVAILABLE"
FREQ_WEIGHTS = {"high": 3, "medium": 2, "low": 1}
ACCESS_RANK = {"high": 0, "medium": 1, "low": 2}  # lower = easier to access
EPS = 1e-9  # float tolerance for capacity comparisons

PRODUCT_COLUMNS = [
    "product_id", "category", "length", "width", "height", "volume",
    "daily_demand", "picking_frequency", "current_location",
]
LOCATION_COLUMNS = [
    "location_id", "capacity_volume", "picking_distance", "accessibility_level",
]

MODE_SINGLE = "One product per rack (strict)"
MODE_SHARED = "Shared racks (use remaining capacity)"


# --------------------------------------------------------------------------- #
# 1. Synthetic data generation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def generate_sample_data(
    n_products: int = 40, n_locations: int = 50, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a realistic random warehouse: racks, SKUs and a naive 'current' layout."""
    rng = np.random.default_rng(seed)

    # ---- Locations: mostly standard racks + a few large bulk racks ----------
    n_bulk = max(2, n_locations // 15)
    n_std = n_locations - n_bulk
    std_caps = rng.choice([0.4, 0.8, 1.5, 2.5], size=n_std, p=[0.30, 0.35, 0.25, 0.10])
    std_dist = rng.uniform(5, 100, size=n_std).round(1)
    std_acc = rng.choice(["High", "Medium", "Low"], size=n_std, p=[0.4, 0.4, 0.2])

    locations = pd.DataFrame(
        {
            "location_id": [f"R-{i + 1:02d}" for i in range(n_std)]
            + [f"BULK-{i + 1:02d}" for i in range(n_bulk)],
            "capacity_volume": np.concatenate([std_caps, np.full(n_bulk, 8.0)]),
            "picking_distance": np.concatenate(
                [std_dist, rng.uniform(60, 120, size=n_bulk).round(1)]
            ),
            "accessibility_level": list(std_acc) + ["Low"] * n_bulk,
        }
    )

    # ---- Products: dimension ranges (cm) per category ----------------------
    dims = {
        "Electronics": ((10, 40), (10, 30), (5, 20)),
        "Grocery": ((10, 30), (10, 30), (10, 30)),
        "Apparel": ((20, 50), (15, 40), (5, 20)),
        "Hardware": ((10, 60), (10, 40), (10, 40)),
        "Home": ((20, 80), (20, 60), (15, 60)),
        "Toys": ((20, 50), (20, 40), (10, 40)),
    }
    furniture_dims = ((150, 200), (120, 150), (100, 150))  # large -> need big racks
    categories = list(dims)
    n_bulk_items = max(2, n_products // 15)

    rows = []
    for i in range(n_products):
        if i < n_bulk_items:
            cat, ((l0, l1), (w0, w1), (h0, h1)) = "Furniture", furniture_dims
        else:
            cat = categories[int(rng.integers(len(categories)))]
            (l0, l1), (w0, w1), (h0, h1) = dims[cat]
        length = int(rng.integers(l0, l1 + 1))
        width = int(rng.integers(w0, w1 + 1))
        height = int(rng.integers(h0, h1 + 1))
        rows.append(
            {
                "product_id": f"SKU-{i + 1:03d}",
                "category": cat,
                "length": length,
                "width": width,
                "height": height,
                "volume": round(length * width * height / 1e6, 3),  # cm3 -> m3
            }
        )
    products = pd.DataFrame(rows)

    # ---- Demand (skewed: few fast movers, many slow) + frequency class -----
    demand = np.maximum(1, rng.gamma(1.6, 12, size=n_products)).round().astype(int)
    demand[:n_bulk_items] = rng.integers(1, 6, size=n_bulk_items)  # furniture is slow
    noisy = demand * rng.lognormal(0, 0.35, size=n_products)  # imperfect correlation
    hi, lo = np.quantile(noisy, 0.75), np.quantile(noisy, 0.35)
    products["daily_demand"] = demand
    products["picking_frequency"] = np.where(
        noisy >= hi, "High", np.where(noisy >= lo, "Medium", "Low")
    )

    # ---- "Current" layout: random, capacity-aware (what an unoptimised
    #      warehouse often looks like) -----------------------------------------
    remaining = dict(zip(locations["location_id"], locations["capacity_volume"]))
    loc_ids = list(remaining)
    current = [""] * n_products
    for i in rng.permutation(n_products):
        vol = products.loc[i, "volume"]
        fits = [l for l in loc_ids if remaining[l] >= vol]
        choice = str(rng.choice(fits if fits else loc_ids))
        remaining[choice] -= vol
        current[i] = choice
    products["current_location"] = current

    return products[PRODUCT_COLUMNS], locations[LOCATION_COLUMNS]


# --------------------------------------------------------------------------- #
# 2. Input validation / cleaning
# --------------------------------------------------------------------------- #
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def clean_products(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and normalise the products table. Returns (clean_df, warnings)."""
    warnings: List[str] = []
    df = _normalise_columns(raw)

    missing = [
        c for c in ["product_id", "daily_demand", "picking_frequency", "current_location"]
        if c not in df.columns
    ]
    if "volume" not in df.columns and not {"length", "width", "height"} <= set(df.columns):
        missing.append("volume (or length + width + height)")
    if missing:
        raise ValueError(f"Products CSV is missing required column(s): {', '.join(missing)}")

    # Numeric coercion; absent optional numeric columns become NaN.
    for col in ["length", "width", "height", "volume", "daily_demand"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan

    # Derive volume (cm3 -> m3) from dimensions when the volume cell is empty.
    df["volume"] = df["volume"].fillna(df["length"] * df["width"] * df["height"] / 1e6)

    df = df.dropna(subset=["product_id"])
    df["product_id"] = df["product_id"].astype(str).str.strip()
    if "category" not in df.columns:
        df["category"] = "Uncategorized"
    df["category"] = df["category"].fillna("Uncategorized").astype(str)
    df["current_location"] = df["current_location"].fillna("").astype(str).str.strip()

    # Frequency: case-insensitive; unknown values fall back to Low.
    freq = df["picking_frequency"].astype(str).str.strip().str.title()
    bad_freq = ~freq.str.lower().isin(FREQ_WEIGHTS)
    if bad_freq.any():
        warnings.append(
            f"{int(bad_freq.sum())} product(s) had an unrecognised picking_frequency "
            "and were treated as 'Low'."
        )
    df["picking_frequency"] = freq.where(~bad_freq, "Low")

    # Drop unusable rows.
    invalid = (
        df["volume"].isna() | (df["volume"] <= 0)
        | df["daily_demand"].isna() | (df["daily_demand"] < 0)
        | df["product_id"].isin(["", "nan"])
    )
    if invalid.any():
        warnings.append(f"Dropped {int(invalid.sum())} product row(s) with missing/invalid values.")
        df = df[~invalid]

    dups = df["product_id"].duplicated()
    if dups.any():
        warnings.append(f"Dropped {int(dups.sum())} duplicate product_id row(s).")
        df = df[~dups]

    return df[PRODUCT_COLUMNS].reset_index(drop=True), warnings


def clean_locations(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and normalise the locations table. Returns (clean_df, warnings)."""
    warnings: List[str] = []
    df = _normalise_columns(raw)

    missing = [c for c in ["location_id", "capacity_volume", "picking_distance"] if c not in df.columns]
    if missing:
        raise ValueError(f"Locations CSV is missing required column(s): {', '.join(missing)}")

    df["capacity_volume"] = pd.to_numeric(df["capacity_volume"], errors="coerce")
    df["picking_distance"] = pd.to_numeric(df["picking_distance"], errors="coerce")
    df = df.dropna(subset=["location_id"])
    df["location_id"] = df["location_id"].astype(str).str.strip()

    if "accessibility_level" not in df.columns:
        df["accessibility_level"] = "Medium"
    acc = df["accessibility_level"].fillna("Medium").astype(str).str.strip().str.title()
    df["accessibility_level"] = acc.where(acc.str.lower().isin(ACCESS_RANK), "Medium")

    invalid = (
        df["capacity_volume"].isna() | (df["capacity_volume"] <= 0)
        | df["picking_distance"].isna() | (df["picking_distance"] < 0)
    )
    if invalid.any():
        warnings.append(f"Dropped {int(invalid.sum())} location row(s) with missing/invalid values.")
        df = df[~invalid]

    dups = df["location_id"].duplicated()
    if dups.any():
        warnings.append(f"Dropped {int(dups.sum())} duplicate location_id row(s).")
        df = df[~dups]

    return df[LOCATION_COLUMNS].reset_index(drop=True), warnings


# --------------------------------------------------------------------------- #
# 3. Optimization engine
# --------------------------------------------------------------------------- #
def _benefit_text(action: str, demand: float, cur: float, new: float) -> str:
    """Human-readable benefit of a recommendation for one product."""
    if action == "No slot":
        return "Needs a larger rack or manual review; stays at its current location."
    if action == "Keep":
        return "Already in the best available rack - no move needed."
    saved = demand * (cur - new)
    if saved > 0:
        pct = (cur - new) / cur * 100 if cur > 0 else 0
        return f"Saves {saved:,.0f} m/day ({pct:.0f}% shorter trip per pick)."
    if saved < 0:
        return (f"Adds {-saved:,.0f} m/day, freeing nearer racks for higher-priority items "
                "(net warehouse effect is positive).")
    return "No change in picking distance."


def optimize_placement(
    products: pd.DataFrame, locations: pd.DataFrame, shared_racks: bool
) -> pd.DataFrame:
    """
    Greedy slotting: highest-priority product -> nearest rack with enough free capacity.
    The whole warehouse is re-slotted from empty racks (a full re-organisation).
    """
    prods = products.copy()

    # --- Priority score ------------------------------------------------------
    prods["frequency_weight"] = (
        prods["picking_frequency"].str.lower().map(FREQ_WEIGHTS).fillna(1).astype(int)
    )
    prods["priority_score"] = prods["daily_demand"] * prods["frequency_weight"]
    # Stable sort: ties go to higher demand, then to bigger items (harder to place).
    prods = prods.sort_values(
        ["priority_score", "daily_demand", "volume"],
        ascending=[False, False, False], kind="mergesort",
    ).reset_index(drop=True)
    prods["priority_rank"] = np.arange(1, len(prods) + 1)

    # --- Racks sorted by distance (nearest first), accessibility as tie-break
    locs = locations.copy()
    locs["_acc_rank"] = locs["accessibility_level"].str.lower().map(ACCESS_RANK).fillna(1)
    locs = locs.sort_values(["picking_distance", "_acc_rank"], kind="mergesort").reset_index(drop=True)

    loc_ids = locs["location_id"].to_numpy()
    loc_dist = locs["picking_distance"].to_numpy(dtype=float)
    loc_acc = locs["accessibility_level"].to_numpy()
    free = locs["capacity_volume"].to_numpy(dtype=float).copy()
    used = np.zeros(len(locs), dtype=bool)
    max_capacity = float(free.max()) if len(free) else 0.0

    rec_loc: List[str] = []
    rec_dist: List[float] = []
    reasons: List[str] = []
    n = len(prods)

    for row in prods.itertuples(index=False):
        vol = float(row.volume)
        fits = free >= vol - EPS
        if not shared_racks:
            fits &= ~used                      # one product per rack
        candidates = np.flatnonzero(fits)      # racks are distance-sorted -> first = nearest

        if len(candidates) == 0:
            rec_loc.append(NO_SLOT)
            rec_dist.append(np.nan)
            if vol > max_capacity + EPS:
                why = (f"Product volume {vol:.2f} m3 exceeds the largest rack "
                       f"({max_capacity:.2f} m3).")
            else:
                why = "All racks large enough are already full or occupied."
            reasons.append(f"Rank #{row.priority_rank}/{n}. {why}")
            continue

        j = int(candidates[0])
        reasons.append(
            f"Rank #{row.priority_rank}/{n}: priority {row.priority_score:,.0f} "
            f"({row.daily_demand:,.0f} picks/day x weight {row.frequency_weight} for "
            f"{row.picking_frequency} frequency). Placed in {loc_ids[j]}, the nearest rack "
            f"({loc_dist[j]:.1f} m, {loc_acc[j]} accessibility) with {free[j]:.2f} m3 free "
            f">= {vol:.2f} m3 needed."
        )
        rec_loc.append(str(loc_ids[j]))
        rec_dist.append(float(loc_dist[j]))
        free[j] -= vol
        used[j] = True

    prods["recommended_location"] = rec_loc
    prods["recommended_distance"] = rec_dist
    prods["allocation_reason"] = reasons

    # --- Current distances (unknown current racks use the average distance) --
    dist_map = locations.set_index("location_id")["picking_distance"]
    prods["current_known"] = prods["current_location"].isin(dist_map.index)
    prods["current_distance"] = (
        prods["current_location"].map(dist_map).fillna(dist_map.mean())
    )

    prods["action"] = np.select(
        [
            prods["recommended_location"] == NO_SLOT,
            prods["recommended_location"] == prods["current_location"],
        ],
        ["No slot", "Keep"],
        default="Re-slot",
    )
    prods["benefit"] = [
        _benefit_text(a, d, c, r)
        for a, d, c, r in zip(
            prods["action"], prods["daily_demand"],
            prods["current_distance"], prods["recommended_distance"],
        )
    ]
    return prods


# --------------------------------------------------------------------------- #
# 4. Evaluation & metrics
# --------------------------------------------------------------------------- #
def compute_metrics(result: pd.DataFrame, locations: pd.DataFrame) -> Dict[str, float]:
    """Baseline vs optimized picking distance, space utilization and feasibility."""
    placed = result["recommended_location"] != NO_SLOT

    # Baseline = Sum(daily_demand * current_location_distance)
    baseline = float((result["daily_demand"] * result["current_distance"]).sum())

    # Optimized = Sum(daily_demand * recommended_location_distance).
    # Products with no slot simply stay where they are today.
    effective_dist = np.where(placed, result["recommended_distance"], result["current_distance"])
    optimized = float((result["daily_demand"] * effective_dist).sum())
    reduction = (baseline - optimized) / baseline * 100 if baseline > 0 else 0.0

    cap = locations.set_index("location_id")["capacity_volume"]

    # Space utilization = stored volume / capacity of the racks actually occupied.
    known = result[result["current_known"]]
    occ_before = known["current_location"].unique()
    cap_before = float(cap.loc[occ_before].sum()) if len(occ_before) else 0.0
    util_before = known["volume"].sum() / cap_before * 100 if cap_before > 0 else 0.0

    assigned = result[placed]
    occ_after = assigned["recommended_location"].unique()
    cap_after = float(cap.loc[occ_after].sum()) if len(occ_after) else 0.0
    util_after = assigned["volume"].sum() / cap_after * 100 if cap_after > 0 else 0.0

    return {
        "baseline_distance": baseline,
        "optimized_distance": optimized,
        "distance_saved": baseline - optimized,
        "reduction_pct": reduction,
        "util_before": util_before,
        "util_after": util_after,
        "racks_before": len(occ_before),
        "racks_after": len(occ_after),
        "feasibility_pct": placed.mean() * 100 if len(result) else 0.0,
        "n_products": len(result),
        "n_unplaced": int((~placed).sum()),
    }


# --------------------------------------------------------------------------- #
# 5. Streamlit UI
# --------------------------------------------------------------------------- #
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_sidebar():
    """Sidebar: uploads, sample-data button, strategy. Returns (raw_products, raw_locations, shared)."""
    with st.sidebar:
        st.header("1 · Data")
        up_prod = st.file_uploader("Products CSV", type="csv", key="up_prod")
        up_loc = st.file_uploader("Locations CSV", type="csv", key="up_loc")

        st.markdown("**No files? Generate a demo warehouse:**")
        n_products = st.slider("Number of products", 20, 200, 40, step=5)
        n_locations = st.slider("Number of racks", 20, 150, 50, step=5)
        if st.button("Load Sample Data", type="primary", use_container_width=True):
            st.session_state["sample"] = generate_sample_data(n_products, n_locations)
        if "sample" in st.session_state and st.button("Clear sample data", use_container_width=True):
            del st.session_state["sample"]
            st.rerun()

        # Downloadable CSV templates so users can see the expected format.
        s_prod, s_loc = st.session_state.get("sample", generate_sample_data())
        with st.expander("Download sample CSV templates"):
            st.download_button("products.csv", to_csv_bytes(s_prod), "products.csv", "text/csv")
            st.download_button("locations.csv", to_csv_bytes(s_loc), "locations.csv", "text/csv")

        st.header("2 · Strategy")
        mode = st.radio(
            "Rack usage",
            [MODE_SHARED, MODE_SINGLE],
            help="Shared: several products can share a rack while free capacity remains. "
                 "Strict: each rack holds at most one product.",
        )

    sample = st.session_state.get("sample")
    try:
        raw_p = pd.read_csv(up_prod) if up_prod is not None else (sample[0] if sample else None)
        raw_l = pd.read_csv(up_loc) if up_loc is not None else (sample[1] if sample else None)
    except Exception as exc:  # malformed CSV
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
    return raw_p, raw_l, mode == MODE_SHARED


def render_landing():
    st.info("👈 Upload both CSV files, or click **Load Sample Data** in the sidebar to start.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("products.csv")
        st.code(", ".join(PRODUCT_COLUMNS), language="text")
        st.caption("picking_frequency: High / Medium / Low. volume is derived from "
                   "length x width x height (cm -> m3) when empty.")
    with c2:
        st.subheader("locations.csv")
        st.code(", ".join(LOCATION_COLUMNS), language="text")
        st.caption("picking_distance = metres from the dispatch point. "
                   "capacity_volume uses the same unit as product volume (m3).")


def render_kpis(m: Dict[str, float]):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Distance Saved", f"{m['reduction_pct']:.1f}%",
        delta=f"{m['distance_saved']:,.0f} m/day saved",
        help="(Baseline - Optimized) / Baseline picking distance.",
    )
    k2.metric(
        "Space Utilization (after)", f"{m['util_after']:.1f}%",
        delta=f"{m['util_after'] - m['util_before']:+.1f} pp vs before ({m['util_before']:.1f}%)",
        help="Stored volume / capacity of the racks in use.",
    )
    k3.metric(
        "Feasibility", f"{m['feasibility_pct']:.1f}%",
        delta=f"{m['n_unplaced']} product(s) without slot" if m["n_unplaced"] else "All products placed",
        delta_color="inverse" if m["n_unplaced"] else "normal",
        help="Share of products that received a valid rack.",
    )
    k4.metric(
        "Racks in Use", f"{m['racks_after']}",
        delta=int(m["racks_after"] - m["racks_before"]),
        delta_color="inverse",  # fewer racks used = good
        help=f"Before: {m['racks_before']} racks.",
    )


def render_table(result: pd.DataFrame):
    st.subheader("Recommended placement")
    actions = st.multiselect(
        "Filter by action", ["Re-slot", "Keep", "No slot"],
        default=["Re-slot", "Keep", "No slot"],
    )
    view = result[result["action"].isin(actions)][[
        "priority_rank", "product_id", "category", "picking_frequency", "daily_demand",
        "priority_score", "volume", "current_location", "current_distance",
        "recommended_location", "recommended_distance", "action",
        "allocation_reason", "benefit",
    ]].rename(columns={
        "priority_rank": "Rank", "product_id": "Product", "category": "Category",
        "picking_frequency": "Frequency", "daily_demand": "Daily demand",
        "priority_score": "Priority score", "volume": "Volume (m3)",
        "current_location": "Current rack", "current_distance": "Current dist (m)",
        "recommended_location": "Recommended rack", "recommended_distance": "New dist (m)",
        "action": "Action", "allocation_reason": "Allocation reason", "benefit": "Benefit",
    })
    st.dataframe(view.round(2), hide_index=True, height=420)
    st.download_button(
        "⬇️ Download recommendations (CSV)", to_csv_bytes(view), "recommended_placement.csv", "text/csv"
    )


def render_charts(result: pd.DataFrame, m: Dict[str, float]):
    st.subheader("Before vs after")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Total daily picking distance (m) - lower is better")
        st.bar_chart(pd.DataFrame(
            {"Before": [m["baseline_distance"]], "After": [m["optimized_distance"]]},
            index=["Picking distance"],
        ))
    with c2:
        st.caption("Space utilization (%) - higher is better")
        st.bar_chart(pd.DataFrame(
            {"Before": [m["util_before"]], "After": [m["util_after"]]},
            index=["Utilization"],
        ))

    st.caption("Daily picking distance (m) by category")
    placed = result["recommended_location"] != NO_SLOT
    eff = np.where(placed, result["recommended_distance"], result["current_distance"])
    by_cat = (
        pd.DataFrame({
            "Category": result["category"],
            "Before": result["daily_demand"] * result["current_distance"],
            "After": result["daily_demand"] * eff,
        })
        .groupby("Category").sum()
    )
    st.bar_chart(by_cat)


def main():
    st.set_page_config(page_title="Warehouse Optimizer", page_icon="🏭", layout="wide")
    st.title("🏭 AI-Based Warehouse Space & Material Optimization")
    st.caption(
        "Re-slots fast-moving products closer to dispatch and packs racks efficiently - "
        "rule-based, runs on CPU in milliseconds."
    )

    raw_p, raw_l, shared = render_sidebar()
    if raw_p is None or raw_l is None:
        if raw_p is not None or raw_l is not None:
            st.warning("Please provide **both** a products file and a locations file.")
        render_landing()
        st.stop()

    # Validate inputs.
    try:
        products, w_p = clean_products(raw_p)
        locations, w_l = clean_locations(raw_l)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    for w in w_p + w_l:
        st.warning(w)
    if products.empty or locations.empty:
        st.error("No valid rows left after cleaning - please check your CSV files.")
        st.stop()

    unknown = ~products["current_location"].isin(locations["location_id"])
    if unknown.any():
        st.warning(
            f"{int(unknown.sum())} product(s) have a current_location not found in the locations "
            "file; the average rack distance is used for their baseline and they are excluded "
            "from 'before' utilization."
        )

    # Run optimisation + evaluation.
    result = optimize_placement(products, locations, shared_racks=shared)
    metrics = compute_metrics(result, locations)

    render_kpis(metrics)
    st.divider()

    tab_rec, tab_chart, tab_data, tab_method = st.tabs(
        ["📋 Recommendations", "📊 Before vs After", "🗂 Input data", "ℹ️ Method"]
    )
    with tab_rec:
        render_table(result)
    with tab_chart:
        render_charts(result, metrics)
    with tab_data:
        c1, c2 = st.columns(2)
        c1.subheader(f"Products ({len(products)})")
        c1.dataframe(products, hide_index=True, height=380)
        c2.subheader(f"Locations ({len(locations)})")
        c2.dataframe(locations, hide_index=True, height=380)
    with tab_method:
        st.markdown(
            """
**Priority score** = `daily_demand x frequency_weight` (High = 3, Medium = 2, Low = 1).

1. Products are sorted by priority score (highest first).
2. Racks are sorted by distance to the dispatch point (nearest first).
3. Each product goes to the **nearest rack with enough free capacity**; otherwise it is flagged `NO_SLOT_AVAILABLE`.

**Metrics**
- *Baseline distance* = Σ(daily_demand x current rack distance); *Optimized* = Σ(daily_demand x recommended rack distance).
- Products with no slot keep their current location in the optimized figure.
- *Space utilization* = stored volume / capacity of racks in use (before: current layout, after: recommended layout).
- *Feasibility* = products successfully placed / all products.
            """
        )


if __name__ == "__main__":
    main()
