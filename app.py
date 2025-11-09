# ======================================================
#  PANEL DATA FREQUENCY CONVERSION & INTERPOLATION APP
# ======================================================
# Author: You
# Purpose: Convert panel data (annual/quarterly/monthly)
#          using Denton or Chow-Lin, interpolate missing values,
#          apply transformations, and visualize trends.
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from td import td  # temporal-disaggregation package
from io import BytesIO

# ======================================================
#  SECTION 1: BASIC APP SETTINGS AND AUTHENTICATION
# ======================================================

st.set_page_config(page_title="Panel Frequency Converter", layout="wide")

st.title("📊 Panel Data Frequency Converter")

# --- Simple Login ---
password = st.text_input("Enter Access Code:", type="password")
if password != "1992":
    st.warning("Access Denied. Please enter the correct code.")
    st.stop()

st.success("Access Granted ✅")

# ======================================================
#  SECTION 2: FILE UPLOAD
# ======================================================

st.header("📂 Upload Your Panel Data File")
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Basic checks
    required_cols = ["year", "country"]
    if not all(col in df.columns for col in required_cols):
        st.error("Your file must include at least 'year' and 'country' columns.")
        st.stop()

    # Identify numeric variables
    num_vars = [col for col in df.columns if col not in ["year", "country"]]

    # ======================================================
    #  SECTION 3: USER OPTIONS
    # ======================================================

    st.header("⚙️ Choose Conversion Options")

    freq_options = ["Annual to Quarterly", "Annual to Monthly", "Quarterly to Annual"]
    selected_freq = st.selectbox("Select Frequency Conversion", freq_options)

    method = st.selectbox("Select Disaggregation Method", ["Denton", "Chow-Lin"])
    
    transform_option = st.radio("Select Data Type:", ["Raw Data", "Log Transform"])
    
    interpolate_option = st.checkbox("Interpolate Missing Values (Linear)", value=True)

    # ======================================================
    #  SECTION 4: FREQUENCY CONVERSION FUNCTION
    # ======================================================

    def convert_frequency(country_df, method, target_freq):
        country_df = country_df.set_index("year").sort_index()
        new_df = pd.DataFrame(index=country_df.index)
        result = pd.DataFrame()

        for var in num_vars:
            series = country_df[var].dropna()

            # Convert frequency using Denton or Chow-Lin
            try:
                if target_freq == "Q":
                    new_index = pd.period_range(start=series.index.min(), end=series.index.max(), freq='Q')
                elif target_freq == "M":
                    new_index = pd.period_range(start=series.index.min(), end=series.index.max(), freq='M')
                else:
                    new_index = pd.period_range(start=series.index.min(), end=series.index.max(), freq='A')

                if method == "Denton":
                    result_series = td(series, to_freq=target_freq, method="denton")
                else:
                    result_series = td(series, to_freq=target_freq, method="chow-lin")

                tmp = pd.DataFrame(result_series, columns=[var])
                result = pd.concat([result, tmp], axis=1)

            except Exception as e:
                st.warning(f"Could not convert {var} for {country_df.index[0]}: {e}")

        result.reset_index(inplace=True)
        result.rename(columns={"index": "year"}, inplace=True)
        result["country"] = country_df["country"].iloc[0]
        return result

    # ======================================================
    #  SECTION 5: PROCESSING PANEL DATA
    # ======================================================

    st.header("🧮 Processing Data...")

    converted_data = []
    target_freq = "Q" if "Quarterly" in selected_freq else "M" if "Monthly" in selected_freq else "A"

    for country, group in df.groupby("country"):
        group["year"] = pd.PeriodIndex(group["year"], freq='A')
        
        # Handle missing values within country
        if interpolate_option:
            group[num_vars] = group[num_vars].interpolate(method="linear", limit_direction="both")

        result = convert_frequency(group, method, target_freq)
        converted_data.append(result)

    final_df = pd.concat(converted_data, ignore_index=True)

    # Apply transformation
    if transform_option == "Log Transform":
        for var in num_vars:
            final_df[var] = np.log(final_df[var].replace(0, np.nan))

    st.success("✅ Data Conversion Completed")

    # ======================================================
    #  SECTION 6: VISUALIZATION
    # ======================================================

    st.header("📈 Trend Visualization")

    country_choice = st.selectbox("Select Country", sorted(final_df["country"].unique()))
    variable_choice = st.selectbox("Select Variable", num_vars)

    subset = final_df[final_df["country"] == country_choice]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["year"].astype(str), subset[variable_choice], marker='o')
    ax.set_title(f"{variable_choice} Trend - {country_choice}")
    ax.set_xlabel("Year/Period")
    ax.set_ylabel(variable_choice)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ======================================================
    #  SECTION 7: DOWNLOAD PROCESSED FILE
    # ======================================================

    st.header("📥 Download Processed File")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        final_df.to_excel(writer, index=False, sheet_name="ConvertedData")
    buffer.seek(0)

    st.download_button(
        label="Download Converted Excel File",
        data=buffer,
        file_name="converted_panel_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload a file to begin.")
