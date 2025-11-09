import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from io import BytesIO

# -------------------------------
# 1. AUTHENTICATION SECTION
# -------------------------------
st.set_page_config(page_title="Panel Data Frequency Converter", layout="wide")

st.title("🔐 Panel Data Frequency Converter App")

password = st.text_input("Enter Access Code:", type="password")
if password != "1992":
    st.warning("Access denied. Please enter the correct code.")
    st.stop()

st.success("Access granted. Welcome!")

# -------------------------------
# 2. FILE UPLOAD SECTION
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
# 3. METHOD SELECTION SECTION
# -------------------------------
st.header("⚙️ Configuration Options")

# Frequency options
target_freq = st.selectbox(
    "Select Target Frequency",
    ["Annual", "Quarterly", "Monthly"]
)

# Missing value interpolation options
interpolation_method = st.selectbox(
    "Select Interpolation Method",
    ["linear", "spline", "polynomial"]
)

# Transformation options
transform_option = st.selectbox(
    "Data Transformation",
    ["Raw Data", "Log Data", "Difference (Transformed)"]
)

# -------------------------------
# 4. FUNCTION DEFINITIONS
# -------------------------------
def convert_frequency(country_df, target_freq):
    """Convert data frequency per country using interpolation."""
    country_df = country_df.sort_values("year").set_index("year")
    numeric_vars = country_df.select_dtypes(include=[np.number]).columns

    # Determine new time index
    if target_freq == "Quarterly":
        new_index = pd.period_range(start=str(country_df.index.min()),
                                    end=str(country_df.index.max()), freq="Q")
    elif target_freq == "Monthly":
        new_index = pd.period_range(start=str(country_df.index.min()),
                                    end=str(country_df.index.max()), freq="M")
    else:
        new_index = pd.period_range(start=str(country_df.index.min()),
                                    end=str(country_df.index.max()), freq="A")

    result = pd.DataFrame({"year": new_index.to_timestamp()})
    result["country"] = country_df["country"].iloc[0]

    for var in numeric_vars:
        y = country_df[var].values
        x = np.arange(len(y))
        f = interp1d(x, y, kind="linear", fill_value="extrapolate")
        new_y = f(np.linspace(0, len(y)-1, len(new_index)))
        result[var] = new_y

    return result

def interpolate_missing(country_df, method):
    """Interpolate missing values country-wise."""
    numeric_vars = country_df.select_dtypes(include=[np.number]).columns
    for var in numeric_vars:
        country_df[var] = country_df[var].interpolate(method=method, limit_direction='both')
    return country_df

# -------------------------------
# 5. PROCESSING SECTION
# -------------------------------
st.header("🔄 Frequency Conversion and Interpolation")

if st.button("Run Conversion"):
    results = []

    for c in df["country"].unique():
        country_data = df[df["country"] == c].copy()

        # Interpolation per country
        country_data = interpolate_missing(country_data, interpolation_method)

        # Frequency conversion per country
        country_converted = convert_frequency(country_data, target_freq)

        # Transformation
        numeric_vars = country_converted.select_dtypes(include=[np.number]).columns
        if transform_option == "Log Data":
            country_converted[numeric_vars] = np.log(country_converted[numeric_vars].replace(0, np.nan))
        elif transform_option == "Difference (Transformed)":
            country_converted[numeric_vars] = country_converted[numeric_vars].diff()

        results.append(country_converted)

    final_df = pd.concat(results, ignore_index=True)
    st.success("✅ Frequency Conversion Complete!")

    # -------------------------------
    # 6. VISUALIZATION SECTION
    # -------------------------------
    st.header("📊 Trend Visualization")

    selected_country = st.selectbox("Select Country to View Trend", df["country"].unique())
    selected_var = st.selectbox("Select Variable to Plot", 
                                [col for col in final_df.columns if col not in ["country", "year"]])

    plot_df = final_df[final_df["country"] == selected_country]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_df["year"], plot_df[selected_var], marker='o')
    ax.set_title(f"{selected_var} Trend for {selected_country}")
    ax.set_xlabel("Time")
    ax.set_ylabel(selected_var)
    st.pyplot(fig)

    # -------------------------------
    # 7. DOWNLOAD SECTION
    # -------------------------------
    st.header("📥 Download Processed File")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        final_df.to_excel(writer, index=False, sheet_name="Converted_Data")
    buffer.seek(0)

    st.download_button(
        label="Download Converted Data (Excel)",
        data=buffer,
        file_name="converted_panel_data.xlsx",
        mime="application/vnd.ms-excel"
    )

else:
    st.info("Upload your file and click **Run Conversion** to start processing.")
