import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Panel Frequency Converter - simplified Denton-style (no external td lib)
# -------------------------------
st.set_page_config(page_title="Panel Frequency Converter", layout="wide")
st.title("📊 Panel Data Frequency Converter — Denton-style (no external package)")

# --- Access code ---
password = st.text_input("Enter Access Code:", type="password")
if password != "1992":
    st.warning("Access denied. Enter the correct access code to continue.")
    st.stop()
st.success("Access granted.")

# --- File upload ---
st.header("📂 Upload panel data (CSV or Excel)")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
if uploaded_file is None:
    st.info("Please upload a file. Expected columns include a Year/Period column (e.g., '1992' or '1992Q1') and a Country column.")
    st.stop()

# Read file
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

if df.empty:
    st.error("Uploaded file is empty.")
    st.stop()

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10))

# --- Column selection ---
st.header("🧭 Tell me which columns are which")
cols = list(df.columns)
year_col = st.selectbox("Year / Period column", cols, index=0)
country_col = st.selectbox("Country column", cols, index=min(1, len(cols)-1))
numeric_candidates = [c for c in cols if c not in [year_col, country_col]]
if not numeric_candidates:
    st.error("No numeric columns detected. Please ensure your file has at least one numeric variable column.")
    st.stop()
selected_vars = st.multiselect("Numeric variables to process", numeric_candidates, default=numeric_candidates)

# --- Options ---
st.header("⚙️ Options")
target_freq = st.selectbox("Target frequency", ["Annual", "Quarterly", "Monthly"])
interpolation_method = st.selectbox("Interpolation for missing values (per-country)", ["linear", "spline"])
transform_option = st.selectbox("Transformation to apply after conversion", ["Raw Data", "Log (natural)", "Difference"])
preserve_annual_totals = st.checkbox("Preserve annual totals when upsampling (recommended)", value=True)

# --- Helper functions ---
def parse_period_to_timestamp(x):
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "na", "nan."]:
        return pd.NaT
    # Quarter detection like 1992Q1 or 1992-Q1 or 1992 Q1
    if "q" in s.lower():
        s2 = s.upper().replace(" ", "").replace("-", "")
        try:
            p = pd.Period(s2, freq="Q")
            return p.to_timestamp(how="end")
        except Exception:
            # try replacing Q with 'Q'
            try:
                year = int(s2[:4])
                q = int(''.join(filter(str.isdigit, s2[4:])) or 1)
                # construct Period string
                per = f"{year}Q{q}"
                p = pd.Period(per, freq="Q")
                return p.to_timestamp(how="end")
            except Exception:
                return pd.NaT
    # If looks like YYYY-MM-DD or YYYY/MM/DD or contains '-'
    if "-" in s or "/" in s:
        try:
            return pd.to_datetime(s, errors="coerce")
        except Exception:
            return pd.NaT
    # Otherwise assume year only
    try:
        return pd.to_datetime(s + "-12-31")
    except Exception:
        return pd.NaT

def normalize_year_column(df_local, year_col_local):
    df_local = df_local.copy()
    df_local["_orig_year_str"] = df_local[year_col_local].astype(str)
    df_local["_parsed_time"] = df_local[year_col_local].apply(parse_period_to_timestamp)
    n_before = len(df_local)
    df_local = df_local.dropna(subset=["_parsed_time"])
    n_after = len(df_local)
    if n_before != n_after:
        st.warning(f"Dropped {n_before-n_after} rows where '{year_col_local}' could not be parsed.")
    df_local = df_local.drop(columns=[year_col_local]).rename(columns={"_parsed_time": year_col_local})
    return df_local

def detect_input_frequency(dt_index):
    # dt_index expected as DatetimeIndex or array of timestamps at period ends
    months = pd.DatetimeIndex(dt_index).month.unique()
    # If months include quarter end months 3,6,9,12 -> likely quarterly
    if set(months).issuperset({3,6,9,12}) and len(months) > 1:
        return "Quarterly"
    # If most months are month-ends variety, treat as monthly
    if len(months) > 4:
        return "Monthly"
    return "Annual"

def upsample_preserve_totals(low_freq_series, target_periods_per_low):
    # low_freq_series: pandas Series indexed by timestamp (period end) for low freq (annual)
    # target_periods_per_low: int (e.g., 4 for quarterly, 12 for monthly)
    # Strategy: create preliminary high-freq series with linear interpolation of the levels,
    # then for each low-frequency bin scale the high-frequency observations so their sum equals the low-frequency total.
    low_idx = pd.DatetimeIndex(low_freq_series.index)
    # create high freq index spanning min to max at period ends
    start = low_idx.min()
    end = low_idx.max()
    # determine freq string for periods per low
    if target_periods_per_low == 4:
        freq = "Q"
    elif target_periods_per_low == 12:
        freq = "M"
    else:
        raise ValueError("Unsupported target periods per low.")
    # create PeriodIndex in terms of quarters/months at period end
    pr = pd.period_range(start=start.to_period('A').start_time, end=end.to_period('A').end_time, freq=freq)
    high_timestamps = pr.to_timestamp(how="end")
    # create linear interpolation over numeric index positions
    low_positions = np.arange(len(low_freq_series))
    # preliminary values by linear interpolation of low-level series
    total_high = len(high_timestamps)
    # positions at high freq mapped to low position space
    high_positions_in_low_space = np.linspace(0, len(low_freq_series)-1, total_high)
    low_vals = low_freq_series.values.astype(float)
    if np.all(np.isnan(low_vals)):
        return pd.Series(index=high_timestamps, data=np.nan)
    interp = np.interp(high_positions_in_low_space, low_positions, low_vals)
    prelim = pd.Series(data=interp, index=high_timestamps)
    # now scale within each low-frequency bin so that sums match low_vals
    scaled = prelim.copy()
    for i, low_ts in enumerate(low_freq_series.index):
        year = pd.Timestamp(low_freq_series.index[i]).year
        mask = [ts.year == year for ts in high_timestamps]
        mask = np.array(mask)
        if mask.sum() == 0:
            continue
        target_sum = low_vals[i]
        current_sum = prelim[mask].sum()
        if np.isnan(current_sum) or current_sum == 0:
            if np.isnan(target_sum):
                scaled_vals = np.repeat(np.nan, mask.sum())
            else:
                scaled_vals = np.repeat(target_sum / mask.sum(), mask.sum())
        else:
            scale = target_sum / current_sum if current_sum != 0 else 0
            scaled_vals = prelim[mask] * scale
        scaled.iloc[np.where(mask)[0]] = scaled_vals
    return scaled

def aggregate_to_annual(df_country, vars_to_agg, year_col_local):
    tmp = df_country.copy()
    tmp['_year'] = pd.DatetimeIndex(tmp[year_col_local]).year
    agg = tmp.groupby('_year')[vars_to_agg].sum().reset_index()
    agg[year_col_local] = pd.to_datetime(agg['_year'].astype(str) + "-12-31")
    agg = agg.drop(columns=['_year'])
    return agg[[year_col_local] + vars_to_agg]

# --- Processing trigger ---
st.header("▶ Run conversion")
if st.button("Start Conversion"):
    # normalize year column
    try:
        df_work = normalize_year_column(df, year_col)
    except Exception as e:
        st.error(f"Failed to parse '{year_col}': {e}")
        st.stop()

    countries = df_work[country_col].unique()
    all_converted = []
    original_samples = []
    summary = {"country": [], "rows_input": [], "rows_output": [], "warnings": []}

    for country in countries:
        grp = df_work[df_work[country_col] == country].copy()
        grp = grp.sort_values(by=year_col).reset_index(drop=True)
        summary["country"].append(country)
        summary["rows_input"].append(len(grp))
        original_samples.append(grp[[year_col, country_col] + selected_vars].copy())

        for var in selected_vars:
            if var in grp.columns:
                try:
                    grp[var] = pd.to_numeric(grp[var], errors='coerce')
                    grp[var] = grp[var].interpolate(method=interpolation_method, limit_direction='both')
                except Exception:
                    grp[var] = grp[var].interpolate(method='linear', limit_direction='both')

        in_freq = detect_input_frequency(grp[year_col])
        if target_freq == "Annual":
            if in_freq in ["Quarterly", "Monthly"]:
                try:
                    agg = aggregate_to_annual(grp, selected_vars, year_col)
                    agg[country_col] = country
                    all_converted.append(agg)
                    summary["rows_output"].append(len(agg))
                    summary["warnings"].append("")
                except Exception as e:
                    summary["rows_output"].append(0)
                    summary["warnings"].append(f"Aggregation failed: {e}")
            else:
                out = grp[[year_col, country_col] + selected_vars].copy()
                all_converted.append(out)
                summary["rows_output"].append(len(out))
                summary["warnings"].append("")
        else:
            if target_freq == "Quarterly":
                tpp = 4
            else:
                tpp = 12
            if in_freq == "Annual":
                reconstructed = None
                for var in selected_vars:
                    low_series = grp.set_index(year_col)[var]
                    scaled = upsample_preserve_totals(low_series, tpp) if preserve_annual_totals else low_series.reindex(grp[year_col]).interpolate(method='linear')
                    df_var = pd.DataFrame({year_col: scaled.index, country_col: country, var: scaled.values})
                    if reconstructed is None:
                        reconstructed = df_var
                    else:
                        reconstructed = pd.merge(reconstructed, df_var, on=[year_col, country_col], how='outer')
                if reconstructed is not None:
                    all_converted.append(reconstructed)
                    summary["rows_output"].append(len(reconstructed))
                    summary["warnings"].append("Upsampled from annual using Denton-style scaling")
            elif in_freq == "Quarterly" and target_freq == "Monthly":
                try:
                    start = grp[year_col].min().to_period('Q').start_time
                    end = grp[year_col].max().to_period('Q').end_time
                    pr = pd.period_range(start=start, end=end, freq='M')
                    high_ts = pr.to_timestamp(how='end')
                    reconstructed = pd.DataFrame({year_col: high_ts, country_col: country})
                    for var in selected_vars:
                        low_series = grp.set_index(year_col)[var]
                        low_positions = np.arange(len(low_series))
                        total_high = len(high_ts)
                        high_positions_in_low_space = np.linspace(0, len(low_series)-1, total_high)
                        low_vals = low_series.values.astype(float)
                        interp = np.interp(high_positions_in_low_space, low_positions, low_vals)
                        prelim = pd.Series(data=interp, index=high_ts)
                        scaled = prelim.copy()
                        for i, low_ts in enumerate(low_series.index):
                            qyear = pd.DatetimeIndex([low_ts]).year[0]
                            qquarter = pd.DatetimeIndex([low_ts]).quarter[0]
                            mask = [(ts.year == qyear and (((ts.month-1)//3)+1) == qquarter) for ts in high_ts]
                            mask = np.array(mask)
                            if mask.sum() == 0:
                                continue
                            target_sum = low_vals[i]
                            current_sum = prelim[mask].sum()
                            if np.isnan(current_sum) or current_sum == 0:
                                if np.isnan(target_sum):
                                    scaled_vals = np.repeat(np.nan, mask.sum())
                                else:
                                    scaled_vals = np.repeat(target_sum/mask.sum(), mask.sum())
                            else:
                                scale = target_sum/current_sum
                                scaled_vals = prelim[mask]*scale
                            scaled.iloc[np.where(mask)[0]] = scaled_vals
                        reconstructed[var] = scaled.values
                    all_converted.append(reconstructed)
                    summary["rows_output"].append(len(reconstructed))
                    summary["warnings"].append("Upsampled from quarterly to monthly using Denton-style scaling")
                except Exception as e:
                    summary["rows_output"].append(0)
                    summary["warnings"].append(f"Quarterly->Monthly failed: {e}")
            else:
                try:
                    start = grp[year_col].min().to_period('A').start_time
                    end = grp[year_col].max().to_period('A').end_time
                    if target_freq == "Quarterly":
                        pr = pd.period_range(start=start, end=end, freq='Q')
                    else:
                        pr = pd.period_range(start=start, end=end, freq='M')
                    high_ts = pr.to_timestamp(how='end')
                    reconstructed = pd.DataFrame({year_col: high_ts, country_col: country})
                    for var in selected_vars:
                        src_vals = grp[var].values.astype(float)
                        if len(src_vals) == 1:
                            vals = np.repeat(src_vals[0], len(high_ts))
                        else:
                            src_pos = np.linspace(0, 1, len(src_vals))
                            high_pos = np.linspace(0, 1, len(high_ts))
                            vals = np.interp(high_pos, src_pos, src_vals)
                        reconstructed[var] = vals
                    all_converted.append(reconstructed)
                    summary["rows_output"].append(len(reconstructed))
                    summary["warnings"].append("Interpolated to target frequency without benchmark scaling")
                except Exception as e:
                    summary["rows_output"].append(0)
                    summary["warnings"].append(f"Generic upsampling failed: {e}")

    if not all_converted:
        st.error("No converted data produced. Check input file and options.")
        st.stop()

    final_df = pd.concat(all_converted, ignore_index=True, sort=False)
    final_df[year_col] = pd.to_datetime(final_df[year_col], errors='coerce')
    cols_out = [year_col, country_col] + [c for c in final_df.columns if c not in [year_col, country_col]]
    final_df = final_df[cols_out]

    st.success("Conversion completed. See notes and visualizations below.")

    st.warning("""**Transformation summary (please read):**
- Upsampling (Annual -> Quarterly/Monthly) used a Denton-style approach:\n
  1. A preliminary high-frequency series was created by linear interpolation of the low-frequency levels.\n
  2. For each low-frequency period (e.g., each calendar year), the preliminary high-frequency observations falling in that period were scaled so their sum equals the original low-frequency total. This preserves annual totals when 'Preserve annual totals' is checked.\n
- Downsampling (Quarterly/Monthly -> Annual) aggregates by calendar-year sums.\n
- Missing values were interpolated per country using the selected interpolation method **before** up/down-sampling.\n
- Log transforms (if selected) are applied **after** conversion. Zero or negative entries are set to NaN and warned about.\n
**Caveat:** This is a pragmatic, reproducible approximation of Denton proportional smoothing. It preserves totals and produces smooth intra-period patterns but is not a full econometric implementation (no Chow-Lin, no panel pooling, no AR error modeling). Use for exploratory work and be cautious for inferential claims.
""")

    st.header("📈 Compare Before vs After (per country & variable)")
    sel_country = st.selectbox("Country to compare", countries)
    sel_var = st.selectbox("Variable to compare", selected_vars)

    orig_list = [g for g in original_samples if g[country_col].iloc[0] == sel_country]
    orig_df = orig_list[0].copy() if orig_list else pd.DataFrame(columns=[year_col, country_col] + selected_vars)
    conv_df = final_df[final_df[country_col] == sel_country].sort_values(by=year_col)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Show Original (Before)"):
            if orig_df.empty or sel_var not in orig_df.columns:
                st.error("No original data available for this selection.")
            else:
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(orig_df[year_col], orig_df[sel_var], marker='o', linestyle='-')
                ax.set_title(f"Original - {sel_var} ({sel_country})")
                ax.set_xlabel("Time")
                ax.set_ylabel(sel_var)
                n = len(orig_df)
                if n > 20:
                    step = max(1, int(np.ceil(n/20)))
                    ticks = orig_df[year_col].iloc[::step]
                else:
                    ticks = orig_df[year_col]
                plt.xticks(ticks, rotation=45, ha='right')
                st.pyplot(fig)

    with col2:
        if st.button("Show Converted (After)"):
            if conv_df.empty or sel_var not in conv_df.columns:
                st.error("No converted data available for this selection.")
            else:
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(conv_df[year_col], conv_df[sel_var], marker='o', linestyle='-')
                ax.set_title(f"Converted - {sel_var} ({sel_country})")
                ax.set_xlabel("Time")
                ax.set_ylabel(sel_var)
                n = len(conv_df)
                if n > 20:
                    step = max(1, int(np.ceil(n/20)))
                    ticks = conv_df[year_col].iloc[::step]
                else:
                    ticks = conv_df[year_col]
                plt.xticks(ticks, rotation=45, ha='right')
                st.pyplot(fig)

    st.subheader("Processing summary (per country)")
    summary_df = pd.DataFrame(summary)
    st.dataframe(summary_df)

    if transform_option == "Log (natural)":
        for var in selected_vars:
            if var in final_df.columns:
                bad = (final_df[var] <= 0).sum()
                if bad > 0:
                    st.warning(f"Variable '{var}' contains {bad} zero/negative values in converted data; these will be set to NaN before log transform.")

        for var in selected_vars:
            if var in final_df.columns:
                final_df.loc[final_df[var] <= 0, var] = np.nan
                final_df[var] = np.log(final_df[var])

    elif transform_option == "Difference":
        for var in selected_vars:
            if var in final_df.columns:
                final_df[var] = final_df[var].astype(float).diff()

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Converted")
    buffer.seek(0)
    st.download_button("Download converted dataset (Excel)", buffer, file_name="converted_panel_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Configure options above and click 'Start Conversion' to run.")