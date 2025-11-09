import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from io import BytesIO

# -------------------------------
# 1. AUTHENTICATION
# -------------------------------
st.set_page_config(page_title="Panel Data Frequency Converter", layout="wide")
st.title("📊 Panel Data Frequency Converter App")

password = st.text_input("Enter Access Code:", type="password")
if password != "1992":
    st.warning("Access denied. Please enter the correct code.")
    st.stop()

st.success("Access granted. Welcome!")

# -------------------------------
# 2. FILE UPLOAD
# -------------------------------
st.header("📂 Upload Your Panel Data File")
uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("✅ Data Preview:")
    st.dataframe(df.head())
else:
    st.stop()

# -------------------------------
# 3. COLUMN SELECTION
# -------------------------------
st.header("🧩 Identify Columns")
cols = list(df.columns)
year_col = st.selectbox("Select Year Column", cols)
country_col = st.selectbox("Select Country Column", cols)
numeric_candidates = [c for c in cols if c not in [year_col, country_col]]
selected_vars = st.multiselect("Select Numeric Variables to Process", numeric_candidates, default=numeric_candidates)

# -------------------------------
# 4. METHOD SELECTION
# -------------------------------
st.header("⚙️ Configuration Options")
target_freq = st.selectbox("Select Target Frequency", ["Annual", "Quarterly", "Monthly"])
interpolation_method = st.selectbox("Interpolation Method", ["linear", "spline", "polynomial"])
transform_option = st.selectbox("Data Transformation", ["Raw Data", "Log Data", "Difference (Transformed)"])

# -------------------------------
# 5. HELPER FUNCTIONS
# -------------------------------
def normalize_year_column(df, year_col):
    """Convert mixed year/quarter strings into proper datetime."""
    df[year_col] = df[year_col].astype(str)

    def parse_period(x):
        if "Q" in x:
            try:
                return pd.Period(x, freq="Q").to_timestamp(how="end")
            except Exception:
                return pd.NaT
        else:
            try:
                return pd.to_datetime(x + "-12-31")
            except Exception:
                return pd.NaT

    df[year_col] = df[year_col].apply(parse_period)
    df = df.dropna(subset=[year_col])
    return df


def convert_frequency(country_df, target_freq):
    """Convert data frequency per country using interpolation."""
    country_df = country_df.sort_values(year_col).set_index(year_col)
    numeric_vars = [v for v in country_df.columns if v in selected_vars]

    # Determine new time index based on target frequency
    if target_freq == "Quarterly":
        new_index = pd.period_range(start=country_df.index.min(), end=country_df.index.max(), freq="Q")
    elif target_freq == "Monthly":
        new_index = pd.period_range(start=country_df.index.min(), end=country_df.index.max(), freq="M")
    else:
        new_index = pd.period_range(start=country_df.index.min(), end=country_df.index.max(), freq="A")

    result = pd.DataFrame({year_col: new_index.to_timestamp()})
    result[country_col] = country_df[country_col].iloc[0]

    for var in numeric_vars:
        y = country_df[var].values
        x = np.arange(len(y))
        if len(y) > 1:
            f = interp1d(x, y, kind="linear", fill_value="extrapolate")
            new_y = f(np.linspace(0, len(y)-1, len(new_index)))
        else:
            new_y = np.repeat(y[0], len(new_index))
        result[var] = new_y

    return result


def interpolate_missing(country_df, method):
    """Interpolate missing values country-wise."""
    numeric_vars = [v for v in country_df.columns if v in selected_vars]
    for var in numeric_vars:
        country_df[var] = country_df[var].interpolate(method="linear", limit_direction="both")
    return country_df


# -------------------------------
# 6. PROCESSING
# -------------------------------
st.header("🔄 Run Conversion")

if st.button("Start Conversion"):
    results = []
    df = normalize_year_column(df, year_col)

    for c in df[country_col].unique():
        country_data = df[df[country_col] == c].copy()

        # Interpolate missing values
        country_data = interpolate_missing(country_data, interpolation_method)

        # Frequency conversion
        country_converted = convert_frequency(country_data, target_freq)

        # Apply transformation
        numeric_vars = [v for v in country_converted.columns if v in selected_vars]
        if transform_option == "Log Data":
            country_converted[numeric_vars] = np.log(country_converted[numeric_vars].replace(0, np.nan))
        elif transform_option == "Difference (Transformed)":
            country_converted[numeric_vars] = country_converted[numeric_vars].diff()

        results.append(country_converted)

    final_df = pd.concat(results, ignore_index=True)
    st.success("✅ Frequency Conversion Complete!")

    # -------------------------------
    # 7. VISUALIZATION
    # -------------------------------
    st.header("📈 Trend Visualization")
    selected_country = st.selectbox("Select Country", final_df[country_col].unique())
    selected_var = st.selectbox("Select Variable", selected_vars)

    plot_df = final_df[final_df[country_col] == selected_country]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_df[year_col], plot_df[selected_var], marker="o")
    ax.set_title(f"{selected_var} Trend for {selected_country}")
    ax.set_xlabel("Time")
    ax.set_ylabel(selected_var)
    st.pyplot(fig)

    # -------------------------------
    # 8. DOWNLOAD
    # -------------------------------
    st.header("📥 Download File")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Converted_Data")
    buffer.seek(0)

    st.download_button(
        label="Download Converted Excel",
        data=buffer,
        file_name="converted_panel_data.xlsx",
        mime="application/vnd.ms-excel"
    )

else:
    st.info("Upload file, configure settings, then click **Start Conversion**.")

