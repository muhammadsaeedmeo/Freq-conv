import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Panel Frequency Converter - Econometrically Improved
# -------------------------------
st.set_page_config(page_title="Panel Frequency Converter", layout="wide")
st.title("📊 Panel Data Frequency Converter — Econometric Edition")

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
    st.info("Please upload a file. Expected columns include a Year/Period column and a Country column.")
    st.stop()

# Read file with better error handling
try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Basic data validation
    if df.empty:
        st.error("Uploaded file is empty.")
        st.stop()
        
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate rows. These will be removed.")
        df = df.drop_duplicates()
        
except Exception as e:
    st.error(f"Failed to read uploaded file: {str(e)}")
    st.stop()

st.subheader("Preview (first 10 rows)")
st.dataframe(df.head(10))

# --- Column selection with validation ---
st.header("🧭 Data Configuration")
cols = list(df.columns)

# Improved column selection with fallbacks
default_year_idx = 0
default_country_idx = min(1, len(cols)-1) if len(cols) > 1 else 0

# Try to auto-detect year/country columns
for i, col in enumerate(cols):
    col_lower = col.lower()
    if any(x in col_lower for x in ['year', 'date', 'period', 'time']):
        default_year_idx = i
    elif any(x in col_lower for x in ['country', 'nation', 'state', 'region']):
        default_country_idx = i

year_col = st.selectbox("Year / Period column", cols, index=default_year_idx)
country_col = st.selectbox("Country column", cols, index=default_country_idx)

numeric_candidates = [c for c in cols if c not in [year_col, country_col] and pd.api.types.is_numeric_dtype(df[c])]
if not numeric_candidates:
    # Try to convert potential numeric columns
    potential_numeric = [c for c in cols if c not in [year_col, country_col]]
    for col in potential_numeric:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_candidates.append(col)
        except:
            continue
            
if not numeric_candidates:
    st.error("No numeric columns detected. Please ensure your file has at least one numeric variable column.")
    st.stop()

selected_vars = st.multiselect("Numeric variables to process", numeric_candidates, default=numeric_candidates)

# --- Econometric Options ---
st.header("⚙️ Econometric Options")

col1, col2 = st.columns(2)

with col1:
    target_freq = st.selectbox("Target frequency", ["Annual", "Quarterly", "Monthly"])
    
    # Improved interpolation options
    interpolation_method = st.selectbox(
        "Interpolation for missing values", 
        ["linear", "time", "spline", "nearest", "zero"],
        help="Linear: straight lines between points. Time: accounts for time intervals. Spline: smooth polynomial. Nearest: nearest available value."
    )
    
    preserve_annual_totals = st.checkbox(
        "Preserve annual totals when upsampling", 
        value=True,
        help="For flow variables (like GDP), ensures yearly sums match original data"
    )

with col2:
    transform_option = st.selectbox(
        "Transformation", 
        ["Raw Data", "Log (natural)", "Log (1+x)", "Difference", "Percentage Change"],
        help="Log(1+x) handles zeros. Difference: period-to-period change. Percentage: % change."
    )
    
    # New: Variable type specification
    variable_type = st.radio(
        "Variable type assumption",
        ["Flow (e.g., GDP, Investment)", "Stock (e.g., Capital, Population)"],
        help="Flow variables are summed over periods, stock variables are averaged"
    )

# --- Improved Helper Functions ---

def parse_period_to_timestamp(x):
    """More robust period parsing with better error handling"""
    if pd.isna(x):
        return pd.NaT
        
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "na"]:
        return pd.NaT
    
    # Try common date formats
    date_formats = [
        '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
        '%Y-%m', '%Y/%m', '%Y'
    ]
    
    for fmt in date_formats:
        try:
            return pd.to_datetime(s, format=fmt, errors='raise')
        except:
            continue
    
    # Quarter handling
    if 'q' in s.lower():
        try:
            s_clean = s.upper().replace(' ', '').replace('-', '')
            if len(s_clean) == 5:  # 1992Q1 format
                quarter_map = {'Q1': '-03-31', 'Q2': '-06-30', 'Q3': '-09-30', 'Q4': '-12-31'}
                for q, end_date in quarter_map.items():
                    if s_clean.endswith(q):
                        year = s_clean[:4]
                        return pd.to_datetime(year + end_date)
        except:
            pass
    
    # Final fallback
    try:
        return pd.to_datetime(s, errors='coerce')
    except:
        return pd.NaT

def normalize_year_column(df_local, year_col_local):
    """Improved year column normalization with validation"""
    df_local = df_local.copy()
    
    # Store original for error reporting
    original_values = df_local[year_col_local].astype(str)
    
    df_local["_parsed_time"] = df_local[year_col_local].apply(parse_period_to_timestamp)
    
    # Check parsing success rate
    n_before = len(df_local)
    valid_mask = df_local["_parsed_time"].notna()
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        st.error(f"Could not parse any values in '{year_col_local}'. Please check the date format.")
        st.stop()
    
    if n_valid < n_before:
        invalid_examples = original_values[~valid_mask].head(5).tolist()
        st.warning(f"Failed to parse {n_before - n_valid} rows in '{year_col_local}'. Examples: {invalid_examples}")
    
    df_local = df_local[valid_mask].copy()
    df_local = df_local.drop(columns=[year_col_local]).rename(columns={"_parsed_time": year_col_local})
    
    return df_local

def detect_input_frequency(dt_index):
    """More robust frequency detection"""
    if len(dt_index) < 2:
        return "Annual"  # Can't detect with single observation
    
    dt_index = pd.DatetimeIndex(dt_index).sort_values()
    time_diffs = np.diff(dt_index).astype('timedelta64[D]').astype(int)
    
    if len(time_diffs) == 0:
        return "Annual"
    
    avg_diff = np.mean(time_diffs)
    
    if avg_diff > 300:  # ~10 months
        return "Annual"
    elif avg_diff > 80:  # ~2.5 months
        return "Quarterly"
    else:
        return "Monthly"

def econometric_upsample(low_freq_series, target_freq_str, method="denton", variable_type="flow"):
    """
    Improved econometric upsampling with variable type handling
    """
    if low_freq_series.empty or low_freq_series.isna().all():
        return pd.Series([], dtype=float)
    
    # Create high-frequency index
    start = low_freq_series.index.min()
    end = low_freq_series.index.max()
    
    if target_freq_str == "Quarterly":
        freq = 'Q'
        periods_per_year = 4
    else:  # Monthly
        freq = 'M'
        periods_per_year = 12
    
    high_periods = pd.period_range(start=start, end=end, freq=freq)
    high_timestamps = high_periods.to_timestamp(how='end')
    
    # Linear interpolation as preliminary series
    low_vals = low_freq_series.values.astype(float)
    low_positions = np.arange(len(low_vals))
    
    # Handle single observation case
    if len(low_vals) == 1:
        prelim = pd.Series(np.repeat(low_vals[0], len(high_timestamps)), index=high_timestamps)
    else:
        high_positions_scaled = np.linspace(0, len(low_vals)-1, len(high_timestamps))
        prelim_vals = np.interp(high_positions_scaled, low_positions, low_vals)
        prelim = pd.Series(prelim_vals, index=high_timestamps)
    
    if method == "denton" and preserve_annual_totals and variable_type == "flow":
        # Apply Denton proportional scaling
        result = prelim.copy()
        
        for year in low_freq_series.index.year.unique():
            year_low_value = low_freq_series[low_freq_series.index.year == year].iloc[0]
            year_high_mask = result.index.year == year
            
            if year_high_mask.sum() > 0:
                current_sum = prelim[year_high_mask].sum()
                if current_sum != 0 and not np.isnan(current_sum):
                    scale_factor = year_low_value / current_sum
                    result.loc[year_high_mask] = prelim.loc[year_high_mask] * scale_factor
                else:
                    # Distribute evenly if preliminary sum is zero
                    result.loc[year_high_mask] = year_low_value / year_high_mask.sum()
        
        return result
    else:
        # For stock variables or when not preserving totals, use interpolation
        return prelim

def safe_interpolate(series, method='linear'):
    """Robust interpolation with validation"""
    if series.isna().all() or len(series) < 2:
        return series
    
    try:
        if method == 'spline':
            # Spline requires at least 3 non-NaN points
            non_na = series.notna()
            if non_na.sum() >= 3:
                return series.interpolate(method='spline', order=3)
            else:
                return series.interpolate(method='linear')
        else:
            return series.interpolate(method=method, limit_direction='both')
    except:
        # Fallback to linear interpolation
        return series.interpolate(method='linear', limit_direction='both')

# --- Processing trigger ---
st.header("▶ Run Conversion")
if st.button("Start Conversion"):
    with st.spinner("Processing data... This may take a moment for large datasets."):
        
        # Normalize year column
        try:
            df_work = normalize_year_column(df, year_col)
        except Exception as e:
            st.error(f"Failed to parse '{year_col}': {str(e)}")
            st.stop()

        countries = df_work[country_col].unique()
        all_converted = []
        processing_stats = []
        
        # Determine variable type flag
        is_flow_variable = variable_type.startswith("Flow")
        
        for country in countries:
            country_data = df_work[df_work[country_col] == country].copy()
            country_data = country_data.sort_values(by=year_col).reset_index(drop=True)
            
            stats = {"country": country, "rows_input": len(country_data), "warnings": []}
            
            # Handle missing values with improved interpolation
            for var in selected_vars:
                if var in country_data.columns:
                    try:
                        country_data[var] = pd.to_numeric(country_data[var], errors='coerce')
                        country_data[var] = safe_interpolate(country_data[var], interpolation_method)
                        
                        # Check if still has missing values
                        if country_data[var].isna().any():
                            stats["warnings"].append(f"{var}: {country_data[var].isna().sum()} missing values remain")
                    except Exception as e:
                        stats["warnings"].append(f"{var}: interpolation failed - {str(e)}")
            
            # Detect frequency and convert
            input_freq = detect_input_frequency(country_data[year_col])
            
            if target_freq == "Annual":
                # Downsampling to annual
                if input_freq in ["Quarterly", "Monthly"]:
                    country_data['_year'] = country_data[year_col].dt.year
                    if is_flow_variable:
                        # Sum for flow variables
                        annual_data = country_data.groupby('_year')[selected_vars].sum().reset_index()
                    else:
                        # Average for stock variables (or use last value)
                        annual_data = country_data.groupby('_year')[selected_vars].mean().reset_index()
                    
                    annual_data[year_col] = pd.to_datetime(annual_data['_year'].astype(str) + '-12-31')
                    annual_data[country_col] = country
                    annual_data = annual_data.drop(columns=['_year'])
                    all_converted.append(annual_data)
                    stats["rows_output"] = len(annual_data)
                    
                else:
                    # Already annual
                    all_converted.append(country_data[[year_col, country_col] + selected_vars])
                    stats["rows_output"] = len(country_data)
            
            else:
                # Upsampling to quarterly or monthly
                if input_freq == "Annual":
                    country_data = country_data.set_index(year_col)
                    converted_vars = {}
                    
                    for var in selected_vars:
                        try:
                            upsampled = econometric_upsample(
                                country_data[var], 
                                target_freq, 
                                method="denton",
                                variable_type="flow" if is_flow_variable else "stock"
                            )
                            converted_vars[var] = upsampled
                        except Exception as e:
                            stats["warnings"].append(f"{var}: upsampling failed - {str(e)}")
                            continue
                    
                    if converted_vars:
                        # Combine all variables
                        result_df = pd.DataFrame(converted_vars)
                        result_df = result_df.reset_index().rename(columns={"index": year_col})
                        result_df[country_col] = country
                        all_converted.append(result_df)
                        stats["rows_output"] = len(result_df)
                    else:
                        stats["rows_output"] = 0
                        stats["warnings"].append("No variables successfully converted")
                
                else:
                    # Other conversions (quarterly to monthly, etc.)
                    stats["warnings"].append(f"Conversion from {input_freq} to {target_freq} not fully implemented")
                    stats["rows_output"] = 0
            
            processing_stats.append(stats)
        
        if not all_converted:
            st.error("No data was successfully converted. Check warnings above.")
            st.stop()
        
        # Combine all results
        final_df = pd.concat(all_converted, ignore_index=True, sort=False)
        final_df[year_col] = pd.to_datetime(final_df[year_col], errors='coerce')
        
        # Remove any rows with invalid dates
        final_df = final_df[final_df[year_col].notna()].copy()
        
        # Order columns consistently
        output_cols = [year_col, country_col] + [c for c in selected_vars if c in final_df.columns]
        final_df = final_df[output_cols].sort_values([country_col, year_col])
        
        st.success(f"✅ Conversion completed! Processed {len(final_df)} rows across {len(countries)} countries.")

    # --- Transformation Section ---
    st.header("🔄 Data Transformations")
    
    if transform_option != "Raw Data":
        transformation_warnings = []
        
        for var in selected_vars:
            if var in final_df.columns:
                series = final_df[var].astype(float)
                
                if transform_option == "Log (natural)":
                    if (series <= 0).any():
                        n_negative = (series <= 0).sum()
                        transformation_warnings.append(f"{var}: {n_negative} non-positive values set to NaN")
                        final_df[var] = np.where(series > 0, np.log(series), np.nan)
                    else:
                        final_df[var] = np.log(series)
                
                elif transform_option == "Log (1+x)":
                    min_val = series.min()
                    if min_val < 0:
                        transformation_warnings.append(f"{var}: contains negative values, using log(1+x - min)")
                        final_df[var] = np.log(1 + series - min_val)
                    else:
                        final_df[var] = np.log(1 + series)
                
                elif transform_option == "Difference":
                    final_df[var] = series.diff()
                
                elif transform_option == "Percentage Change":
                    final_df[var] = series.pct_change() * 100
        
        if transformation_warnings:
            st.warning("Transformation warnings:\n- " + "\n- ".join(transformation_warnings))

    # --- Results Display ---
    st.header("📊 Results Summary")
    
    # Processing statistics
    stats_df = pd.DataFrame(processing_stats)
    st.subheader("Processing Statistics by Country")
    st.dataframe(stats_df)
    
    # Data preview
    st.subheader("Converted Data Preview")
    st.dataframe(final_df.head(15))
    
    # Visualization
    st.header("📈 Visualization")
    
    col1, col2 = st.columns(2)
    with col1:
        plot_country = st.selectbox("Select Country", sorted(final_df[country_col].unique()))
    with col2:
        plot_var = st.selectbox("Select Variable", [v for v in selected_vars if v in final_df.columns])
    
    if plot_country and plot_var:
        plot_data = final_df[final_df[country_col] == plot_country].sort_values(year_col)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(plot_data[year_col], plot_data[plot_var], marker='o', linewidth=2, markersize=4)
        ax.set_title(f"{plot_var} - {plot_country}\n({target_freq} Frequency)")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{plot_var} {f'({transform_option})' if transform_option != 'Raw Data' else ''}")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # --- Download Section ---
    st.header("📥 Download Results")
    
    # Add methodology documentation
    methodology = f"""
    Methodology Documentation:
    - Conversion: {input_freq} → {target_freq}
    - Method: Denton-style proportional interpolation
    - Variable Type: {variable_type}
    - Transformation: {transform_option}
    - Missing Value Treatment: {interpolation_method} interpolation
    - Annual Totals Preserved: {preserve_annual_totals and is_flow_variable}
    
    Note: This uses pragmatic frequency conversion suitable for exploratory analysis.
    For formal econometric analysis, consider more sophisticated methods.
    """
    
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, sheet_name="Converted_Data", index=False)
        
        # Add methodology sheet
        meth_df = pd.DataFrame({"Methodology": methodology.split('\n')})
        meth_df.to_excel(writer, sheet_name="Methodology", index=False)
        
        # Add processing stats sheet
        stats_df.to_excel(writer, sheet_name="Processing_Stats", index=False)
    
    buffer.seek(0)
    
    st.download_button(
        "📥 Download Full Results (Excel)",
        buffer,
        file_name=f"converted_data_{target_freq.lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Configure options above and click 'Start Conversion' to run.")
