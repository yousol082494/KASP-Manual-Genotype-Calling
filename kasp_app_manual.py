import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="KASP Manual Caller v2", layout="wide")
st.title("KASP Manual Genotype Caller v1.0")
st.caption(
    "NTC-based origin + known-control-based angular boundaries + distance- and boundary-based Undetermined classification"
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def clean_colname(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def is_blank_column(series):
    return series.isna().all()


def normalize_name(x):
    return str(x).strip()


def auto_make_primer_pairs(df, sample_col, start_col, primer_names, drop_empty=True):
    cols = list(df.columns)
    start_idx = cols.index(start_col)
    target_cols = cols[start_idx:]

    if drop_empty:
        target_cols = [c for c in target_cols if not is_blank_column(df[c])]

    target_cols = [c for c in target_cols if c != sample_col]

    pairs = []
    n_pair = len(target_cols) // 2

    for i in range(n_pair):
        fam_col = target_cols[i * 2]
        hex_col = target_cols[i * 2 + 1]
        primer = primer_names[i] if i < len(primer_names) else f"Primer_{i+1}"
        pairs.append({
            "Primer": primer,
            "FAM_X_col": fam_col,
            "HEX_Y_col": hex_col,
        })

    return pd.DataFrame(pairs)


def make_long_df(df, sample_col, pair_df):
    long_list = []

    for _, row in pair_df.iterrows():
        primer = row["Primer"]
        x_col = row["FAM_X_col"]
        y_col = row["HEX_Y_col"]

        if x_col not in df.columns or y_col not in df.columns:
            continue

        temp = df[[sample_col, x_col, y_col]].copy()
        temp.columns = ["Sample", "FAM_X", "HEX_Y"]
        temp["Sample"] = temp["Sample"].astype(str).str.strip()
        temp["Primer"] = primer
        temp["FAM_X"] = pd.to_numeric(temp["FAM_X"], errors="coerce")
        temp["HEX_Y"] = pd.to_numeric(temp["HEX_Y"], errors="coerce")
        temp = temp.dropna(subset=["FAM_X", "HEX_Y"]).copy()
        long_list.append(temp)

    if not long_list:
        return pd.DataFrame(columns=["Sample", "Primer", "FAM_X", "HEX_Y"])

    return pd.concat(long_list, ignore_index=True)


def ntc_mask(primer_df, ntc_keywords):
    if not ntc_keywords:
        return pd.Series(False, index=primer_df.index)

    names = primer_df["Sample"].astype(str)
    mask = pd.Series(False, index=primer_df.index)
    for kw in ntc_keywords:
        kw = kw.strip()
        if kw:
            mask |= names.str.contains(kw, case=False, na=False, regex=False)
    return mask


def guess_origin(primer_df, ntc_keywords):
    mask = ntc_mask(primer_df, ntc_keywords)
    ntc_df = primer_df.loc[mask].copy()

    if len(ntc_df) > 0:
        ox = float(ntc_df["FAM_X"].median())
        oy = float(ntc_df["HEX_Y"].median())
        msg = f"Detected {len(ntc_df)} NTC sample(s); median FAM/HEX values were used as the origin."
        source = "NTC median"
    else:
        ox = float(primer_df["FAM_X"].min())
        oy = float(primer_df["HEX_Y"].min())
        msg = "No NTC was detected. The minimum FAM/HEX values were temporarily used as the origin. Specify NTC identifiers whenever possible."
        source = "minimum fallback"

    return ox, oy, msg, source, ntc_df


def add_geometry(df, origin_x, origin_y):
    out = df.copy()
    out["Delta_X"] = out["FAM_X"] - origin_x
    out["Delta_Y"] = out["HEX_Y"] - origin_y
    out["Distance"] = np.sqrt(out["Delta_X"] ** 2 + out["Delta_Y"] ** 2)
    out["Angle_Deg"] = np.degrees(np.arctan2(out["Delta_Y"], out["Delta_X"]))
    out["Angle_Deg"] = out["Angle_Deg"].clip(lower=0, upper=90)
    return out


def control_angle_summary(geom_df, allele1_samples, hetero_samples, allele2_samples):
    rows = []
    groups = [
        ("Allele 1", allele1_samples),
        ("Hetero", hetero_samples),
        ("Allele 2", allele2_samples),
    ]

    for label, samples in groups:
        sub = geom_df[geom_df["Sample"].isin(samples)].copy()
        if len(sub) == 0:
            rows.append({"Control_Group": label, "N": 0, "Median_Angle": np.nan, "Median_Distance": np.nan})
        else:
            rows.append({
                "Control_Group": label,
                "N": int(len(sub)),
                "Median_Angle": float(sub["Angle_Deg"].median()),
                "Median_Distance": float(sub["Distance"].median()),
            })
    return pd.DataFrame(rows)


def suggested_boundaries(control_summary, fallback_low=20.0, fallback_high=75.0):
    d = control_summary.set_index("Control_Group")
    a1 = d.loc["Allele 1", "Median_Angle"]
    het = d.loc["Hetero", "Median_Angle"]
    a2 = d.loc["Allele 2", "Median_Angle"]

    low = fallback_low
    high = fallback_high
    ok = False
    msg = "All three known-control classes were not assigned; default angular boundaries are used."

    if pd.notna(a1) and pd.notna(het) and pd.notna(a2):
        if a1 < het < a2:
            low = float((a1 + het) / 2.0)
            high = float((het + a2) / 2.0)
            ok = True
            msg = (
                "Boundaries were calculated from the midpoints between the median angles of "
                "the Allele 1, heterozygous, and Allele 2 control groups."
            )
        else:
            msg = (
                "The control-angle order is not Allele 1 < Heterozygous < Allele 2. "
                "Check the FAM/HEX orientation and control assignments."
            )

    return low, high, ok, msg


def automatic_distance_threshold(geom_df, ntc_df, origin_x, origin_y):
    if len(ntc_df) > 0:
        ntc_geom = add_geometry(ntc_df, origin_x, origin_y)
        # Use three times the 95th percentile of NTC distances as the default threshold.
        ntc_r95 = float(np.percentile(ntc_geom["Distance"], 95)) if len(ntc_geom) > 1 else float(ntc_geom["Distance"].iloc[0])
        threshold = max(ntc_r95 * 3.0, 1.0)
        msg = f"NTC distance 95th percentile × 3 = {threshold:.2f}"
        return threshold, msg

    distances = geom_df["Distance"].dropna()
    threshold = float(np.percentile(distances, 10)) if len(distances) else 0.0
    return threshold, "No NTC detected; the 10th percentile of all sample distances is used as a temporary threshold."


def call_by_angle(angle, low_boundary, high_boundary, uncertainty_margin):
    if pd.isna(angle):
        return "Undetermined"

    # Samples near genotype boundaries are retained as Undetermined.
    if abs(angle - low_boundary) <= uncertainty_margin:
        return "Undetermined"
    if abs(angle - high_boundary) <= uncertainty_margin:
        return "Undetermined"

    if angle < low_boundary:
        return "Allele 1"
    if angle > high_boundary:
        return "Allele 2"
    return "Hetero"


def calculate_calling(
    primer_df,
    origin_x,
    origin_y,
    low_boundary,
    high_boundary,
    use_distance_filter,
    min_distance,
    uncertainty_margin,
    ntc_keywords,
):
    out = add_geometry(primer_df, origin_x, origin_y)
    out["Auto_Call"] = out["Angle_Deg"].apply(
        lambda x: call_by_angle(x, low_boundary, high_boundary, uncertainty_margin)
    )

    if use_distance_filter:
        out.loc[out["Distance"] < min_distance, "Auto_Call"] = "No call"

    # NTC samples are excluded from genotype assignment.
    mask = ntc_mask(out, ntc_keywords)
    out.loc[mask, "Auto_Call"] = "NTC"

    out["Manual_Override"] = ""
    out["Final_Call"] = out["Auto_Call"]
    return out


def apply_final_call(row):
    override = row.get("Manual_Override", "")
    if pd.notna(override) and str(override).strip() != "":
        return str(override).strip()
    return row["Auto_Call"]


def add_boundary_line(fig, origin_x, origin_y, angle_deg, max_len, name):
    rad = np.deg2rad(angle_deg)
    x_end = origin_x + max_len * np.cos(rad)
    y_end = origin_y + max_len * np.sin(rad)
    fig.add_trace(go.Scatter(
        x=[origin_x, x_end],
        y=[origin_y, y_end],
        mode="lines",
        name=name,
        line=dict(dash="dash", width=2),
    ))


def draw_plot(
    plot_df,
    origin_x,
    origin_y,
    low_boundary,
    high_boundary,
    uncertainty_margin,
    selected_primer,
    allele1_samples,
    hetero_samples,
    allele2_samples,
):
    fig = go.Figure()

    call_order = ["Allele 1", "Hetero", "Allele 2", "Undetermined", "No call", "NTC"]
    color_map = {
        "Allele 1": "blue",
        "Hetero": "green",
        "Allele 2": "red",
        "Undetermined": "orange",
        "No call": "gray",
        "NTC": "black",
    }

    control_set = set(allele1_samples) | set(hetero_samples) | set(allele2_samples)

    for call in call_order:
        temp = plot_df[plot_df["Final_Call"] == call].copy()
        if len(temp) == 0:
            continue

        symbols = ["diamond" if s in control_set else "circle" for s in temp["Sample"]]
        sizes = [13 if s in control_set else 9 for s in temp["Sample"]]

        fig.add_trace(go.Scatter(
            x=temp["FAM_X"],
            y=temp["HEX_Y"],
            mode="markers",
            name=call,
            marker=dict(
                size=sizes,
                color=color_map.get(call, "gray"),
                symbol=symbols,
                opacity=0.85,
                line=dict(width=0.7, color="black"),
            ),
            text=temp["Sample"],
            customdata=np.stack([
                temp["Angle_Deg"],
                temp["Distance"],
                temp["Auto_Call"],
                temp["Final_Call"],
            ], axis=-1),
            hovertemplate=(
                "Sample: %{text}<br>"
                "FAM_X: %{x}<br>"
                "HEX_Y: %{y}<br>"
                "Angle: %{customdata[0]:.2f}°<br>"
                "Distance: %{customdata[1]:.2f}<br>"
                "Auto_Call: %{customdata[2]}<br>"
                "Final_Call: %{customdata[3]}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[origin_x], y=[origin_y], mode="markers", name="Origin / NTC median",
        marker=dict(size=16, color="black", symbol="x"),
    ))

    max_x_len = abs(plot_df["FAM_X"].max() - origin_x)
    max_y_len = abs(plot_df["HEX_Y"].max() - origin_y)
    max_len = float(max(max_x_len, max_y_len) * 1.25)
    if max_len <= 0:
        max_len = 1000

    add_boundary_line(fig, origin_x, origin_y, low_boundary, max_len, f"A1/H {low_boundary:.1f}°")
    add_boundary_line(fig, origin_x, origin_y, high_boundary, max_len, f"H/A2 {high_boundary:.1f}°")

    # Uncertainty margin lines
    if uncertainty_margin > 0:
        for ang, label in [
            (max(0, low_boundary - uncertainty_margin), "low margin"),
            (min(90, low_boundary + uncertainty_margin), "low margin"),
            (max(0, high_boundary - uncertainty_margin), "high margin"),
            (min(90, high_boundary + uncertainty_margin), "high margin"),
        ]:
            rad = np.deg2rad(ang)
            x_end = origin_x + max_len * np.cos(rad)
            y_end = origin_y + max_len * np.sin(rad)
            fig.add_trace(go.Scatter(
                x=[origin_x, x_end], y=[origin_y, y_end], mode="lines",
                name=label, showlegend=False,
                line=dict(dash="dot", width=1),
            ))

    fig.update_layout(
        title=f"{selected_primer} KASP manual calling",
        xaxis_title="FAM / Allele 1 / X-axis",
        yaxis_title="HEX / Allele 2 / Y-axis",
        height=720,
        legend_title="Final call",
    )
    return fig


# --------------------------------------------------
# File input: XLSX + CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a KASP result file (.xlsx or .csv)",
    type=["xlsx", "csv"],
)

if uploaded_file is None:
    st.info("Upload a result file to display the analysis settings.")
    st.stop()

file_name = uploaded_file.name.lower()

if file_name.endswith(".xlsx"):
    try:
        xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(
            "The XLSX file could not be opened. This error may occur if a CSV file was renamed with an .xlsx extension. "
            "Save the file again as a valid Excel workbook (.xlsx) or upload the original CSV file."
        )
        st.exception(e)
        st.stop()

    sheet_name = st.selectbox("Select worksheet", xls.sheet_names)
    header_row = st.number_input(
        "Header row number", min_value=0, max_value=20, value=0, step=1,
        help="Use 0 if the first row contains the column names.",
    )
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=header_row, engine="openpyxl")
else:
    header_row = st.number_input(
        "Header row number", min_value=0, max_value=20, value=0, step=1,
        help="Use 0 if the first row contains the column names.",
    )
    uploaded_file.seek(0)
    try:
        df = pd.read_csv(uploaded_file, header=header_row, sep=None, engine="python")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, header=header_row, sep=None, engine="python", encoding="cp949")


df.columns = [clean_colname(c) for c in df.columns]

# Prevent duplicate column names
new_cols = []
seen = {}
for c in df.columns:
    if c not in seen:
        seen[c] = 0
        new_cols.append(c)
    else:
        seen[c] += 1
        new_cols.append(f"{c}_{seen[c]}")
df.columns = new_cols

st.markdown("---")
st.subheader("0. Data Preview")
st.dataframe(df.head(12), use_container_width=True)

# --------------------------------------------------
# Column / primer mapping
# --------------------------------------------------
st.markdown("---")
st.subheader("1. Sample Column and Primer Pair Settings")

col1, col2, col3 = st.columns(3)
with col1:
    sample_col = st.selectbox("Sample name column", df.columns, index=0)
with col2:
    default_start_idx = 1 if len(df.columns) > 1 else 0
    start_col = st.selectbox("First primer-data column", df.columns, index=default_start_idx)
with col3:
    drop_empty = st.checkbox("Automatically exclude completely empty columns", value=True)

primer_text = st.text_input(
    "Enter primer names in order",
    value="P10,P12,P13,P14,P15,P16",
    help="Enter names in the same order as the repeating FAM/HEX column pairs in the Excel/CSV file.",
)
primer_names = [x.strip() for x in primer_text.split(",") if x.strip()]

pair_df = auto_make_primer_pairs(df, sample_col, start_col, primer_names, drop_empty)
if len(pair_df) == 0:
    st.error("Primer pairs could not be generated. Check the first primer-data column.")
    st.stop()

st.markdown("#### Automatically detected primer pairs")
st.caption("If the FAM_X_col and HEX_Y_col assignments are reversed or incorrect, edit them directly in the table below.")
pair_df = st.data_editor(
    pair_df,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Primer": st.column_config.TextColumn("Primer"),
        "FAM_X_col": st.column_config.SelectboxColumn("FAM / Allele 1 / X-axis column", options=list(df.columns)),
        "HEX_Y_col": st.column_config.SelectboxColumn("HEX / Allele 2 / Y-axis column", options=list(df.columns)),
    },
)

long_df = make_long_df(df, sample_col, pair_df)
if len(long_df) == 0:
    st.error("No numeric FAM/HEX data were detected. Check the selected columns.")
    st.stop()

# --------------------------------------------------
# Primer selection
# --------------------------------------------------
st.markdown("---")
st.subheader("2. Primer Selection")
selected_primer = st.selectbox("Select primer", sorted(long_df["Primer"].unique().tolist()))
primer_df = long_df[long_df["Primer"] == selected_primer].copy()

# --------------------------------------------------
# NTC and origin
# --------------------------------------------------
st.markdown("---")
st.subheader("3. NTC and Origin Settings")

ntc_text = st.text_input(
    "NTC identifiers/keywords (comma-separated)",
    value="NTC,H2O,WATER,BLANK",
    help="Samples containing these terms in their names are treated as NTCs. 'CONTROL' is excluded by default to avoid confusion with positive controls.",
)
ntc_keywords = [x.strip() for x in ntc_text.split(",") if x.strip()]

default_origin_x, default_origin_y, origin_msg, origin_source, ntc_df = guess_origin(primer_df, ntc_keywords)
st.caption(origin_msg)

c1, c2, c3 = st.columns(3)
with c1:
    origin_x = st.number_input("X-axis origin (FAM baseline)", value=default_origin_x, step=10.0)
with c2:
    origin_y = st.number_input("Y-axis origin (HEX baseline)", value=default_origin_y, step=10.0)
with c3:
    st.metric("Detected NTCs", len(ntc_df))
    st.caption(f"Default origin source: {origin_source}")

geom_df = add_geometry(primer_df, origin_x, origin_y)

# --------------------------------------------------
# Known controls
# --------------------------------------------------
st.markdown("---")
st.subheader("4. Known Control Assignment")
st.caption(
    "Assign samples with previously known genotypes for this primer. "
    "Controls are displayed as diamonds in the scatter plot. Because allele orientation may differ among primers, controls are assigned manually."
)

sample_options = sorted([s for s in primer_df["Sample"].dropna().unique().tolist() if s])

k1, k2, k3 = st.columns(3)
with k1:
    allele1_samples = st.multiselect("Allele 1 homozygous control (FAM side)", sample_options)
with k2:
    hetero_samples = st.multiselect("Heterozygous control", sample_options)
with k3:
    allele2_samples = st.multiselect("Allele 2 homozygous control (HEX side)", sample_options)

control_summary = control_angle_summary(geom_df, allele1_samples, hetero_samples, allele2_samples)
st.dataframe(control_summary, use_container_width=True, hide_index=True)

sugg_low, sugg_high, control_ok, boundary_msg = suggested_boundaries(control_summary)
if control_ok:
    st.success(boundary_msg)
else:
    st.warning(boundary_msg)

# --------------------------------------------------
# Calling criteria
# --------------------------------------------------
st.markdown("---")
st.subheader("5. Genotype Calling Criteria")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("##### Angular Boundaries")
    low_boundary = st.slider(
        "Allele 1 / Heterozygous boundary",
        min_value=0.0, max_value=90.0,
        value=float(round(sugg_low, 1)), step=0.5,
    )
    high_boundary = st.slider(
        "Heterozygous / Allele 2 boundary",
        min_value=0.0, max_value=90.0,
        value=float(round(sugg_high, 1)), step=0.5,
    )
    if low_boundary >= high_boundary:
        st.error("The Allele 1/Heterozygous boundary must be smaller than the Heterozygous/Allele 2 boundary.")
        st.stop()

with col_b:
    st.markdown("##### Ambiguous Boundary Handling")
    uncertainty_margin = st.slider(
        "Boundary margin (± degrees) → Undetermined",
        min_value=0.0, max_value=15.0, value=3.0, step=0.5,
        help="For example, with a 3° margin, samples within ±3° of either boundary are retained as Undetermined.",
    )

with col_c:
    st.markdown("##### Weak Signal / No Call")
    auto_min_distance, dist_msg = automatic_distance_threshold(geom_df, ntc_df, origin_x, origin_y)
    st.caption(dist_msg)
    use_distance_filter = st.checkbox("Classify samples too close to the origin as No call", value=True)
    min_distance = st.number_input(
        "Minimum amplification distance",
        value=float(round(auto_min_distance, 2)),
        min_value=0.0,
        step=10.0,
    )

called_df = calculate_calling(
    primer_df=primer_df,
    origin_x=origin_x,
    origin_y=origin_y,
    low_boundary=low_boundary,
    high_boundary=high_boundary,
    use_distance_filter=use_distance_filter,
    min_distance=min_distance,
    uncertainty_margin=uncertainty_margin,
    ntc_keywords=ntc_keywords,
)

# --------------------------------------------------
# Manual override
# --------------------------------------------------
st.markdown("---")
st.subheader("6. Manual Sample Review")
st.caption("If Manual_Override is left blank, Auto_Call is used as Final_Call.")

edit_cols = [
    "Sample", "Primer", "FAM_X", "HEX_Y", "Angle_Deg", "Distance", "Auto_Call", "Manual_Override"
]
edited_df = st.data_editor(
    called_df[edit_cols],
    use_container_width=True,
    num_rows="fixed",
    column_config={
        "Manual_Override": st.column_config.SelectboxColumn(
            "Manual_Override",
            options=["", "Allele 1", "Hetero", "Allele 2", "Undetermined", "No call", "NTC"],
        )
    },
)
edited_df["Final_Call"] = edited_df.apply(apply_final_call, axis=1)

# --------------------------------------------------
# Plot
# --------------------------------------------------
st.markdown("---")
st.subheader("7. Allelic Discrimination Plot")
fig = draw_plot(
    plot_df=edited_df,
    origin_x=origin_x,
    origin_y=origin_y,
    low_boundary=low_boundary,
    high_boundary=high_boundary,
    uncertainty_margin=uncertainty_margin,
    selected_primer=selected_primer,
    allele1_samples=allele1_samples,
    hetero_samples=hetero_samples,
    allele2_samples=allele2_samples,
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Summary / output
# --------------------------------------------------
st.markdown("---")
st.subheader("8. Result Summary")
order = ["Allele 1", "Hetero", "Allele 2", "Undetermined", "No call", "NTC"]
summary = (
    edited_df["Final_Call"]
    .value_counts()
    .reindex(order)
    .fillna(0)
    .astype(int)
    .reset_index()
)
summary.columns = ["Final_Call", "Count"]
st.dataframe(summary, use_container_width=True, hide_index=True)

result_df = edited_df.copy()
result_df["Origin_X"] = origin_x
result_df["Origin_Y"] = origin_y
result_df["Low_Boundary_Deg"] = low_boundary
result_df["High_Boundary_Deg"] = high_boundary
result_df["Uncertainty_Margin_Deg"] = uncertainty_margin
result_df["Minimum_Distance"] = min_distance if use_distance_filter else np.nan

st.markdown("#### Final Results")
st.dataframe(result_df, use_container_width=True)

st.markdown("---")
st.subheader("9. Export Results")
output = io.BytesIO()
settings_df = pd.DataFrame([
    {"Setting": "Primer", "Value": selected_primer},
    {"Setting": "Origin_X", "Value": origin_x},
    {"Setting": "Origin_Y", "Value": origin_y},
    {"Setting": "Origin_Source", "Value": origin_source},
    {"Setting": "Low_Boundary_Deg", "Value": low_boundary},
    {"Setting": "High_Boundary_Deg", "Value": high_boundary},
    {"Setting": "Uncertainty_Margin_Deg", "Value": uncertainty_margin},
    {"Setting": "Minimum_Distance", "Value": min_distance if use_distance_filter else "OFF"},
    {"Setting": "Allele1_controls", "Value": ", ".join(allele1_samples)},
    {"Setting": "Hetero_controls", "Value": ", ".join(hetero_samples)},
    {"Setting": "Allele2_controls", "Value": ", ".join(allele2_samples)},
    {"Setting": "NTC_keywords", "Value": ", ".join(ntc_keywords)},
])

with pd.ExcelWriter(output, engine="openpyxl") as writer:
    result_df.to_excel(writer, index=False, sheet_name="result")
    summary.to_excel(writer, index=False, sheet_name="summary")
    control_summary.to_excel(writer, index=False, sheet_name="control_summary")
    settings_df.to_excel(writer, index=False, sheet_name="settings")
    pair_df.to_excel(writer, index=False, sheet_name="primer_pair_info")

output.seek(0)
st.download_button(
    label=f"Download {selected_primer} calling results",
    data=output,
    file_name=f"KASP_called_{selected_primer}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.info(
    "Recommended workflow: 1) verify NTC detection → 2) assign all three known-control groups → "
    "3) review the automatically suggested boundaries → 4) inspect Undetermined/No call samples in the scatter plot → 5) use Manual Override only when necessary."
)
