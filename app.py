import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Try importing td package safely
try:
    from td import td
except ImportError:
    st.error("Required package 'td' not installed. Install it using: pip install temporal-disaggregation")
    st.stop()

# App title
st.title("Panel Data Frequency Conversion (Denton Method Only)")

# File upload
uploaded_file = st.file_uploader("Upload your panel data file (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Basic data checks
    required_cols = {"Year", "Country"}
    if not required_cols.issubset(df.columns):
        st.error("Your file must contain at least the columns: 'Year' and 'Country'.")
        st.stop()

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename for convenience
    df.rename(columns={"year": "year", "country": "country"}, inplace=True)

    # Convert Year column to period index if possible
    try:
        df["year"] = df["year"].astype(str).str.replace("Q", "Q", regex=False)
    except Exception:
        st.warning("Could not normalize 'year' values properly.")

    # Fill missing values with interpolation
    df = df.groupby("country").apply(lambda g: g.interpolate(method="linear", limit_direction="both")).reset_index(drop=True)

    # Select target variable and frequency
    var = st.selectbox("Select variable to convert", [c for c in df.columns if c not in ["year", "country"]])
    target_freq = st.selectbox("Select target frequency", ["Q", "M"])

    # Log transform option
    log_transform = st.checkbox("Apply log transformation")

    # Prepare storage for converted data
    converted_data = []

    # Processing loop
    for country, group in df.groupby("country"):
        group = group.copy()

        try:
            # Store name before modification
            country_name = group["country"].iloc[0]

            # Ensure index
            group.index = pd.PeriodIndex(group["year"], freq='A')
            group = group.sort_index()

            series = group[var]

            # Check for invalid values before log transform
            if log_transform:
                if (series <= 0).any():
                    st.warning(f"Variable {var} for {country_name} has zero or negative values. These are set to NaN for log transform.")
                series = np.log(series.replace(0, np.nan))

            # Convert using Denton
            converted = td(series, to=target_freq, method="denton")

            converted_df = pd.DataFrame({
                "year": converted.index.astype(str),
                "country": country_name,
                var: converted.values
            })
            converted_data.append(converted_df)

        except Exception as e:
            st.error(f"Could not convert {var} for {country_name}: {e}")

    # Combine results
    if len(converted_data) > 0:
        final_df = pd.concat(converted_data)
        st.success(f"Successfully converted {len(final_df)} observations across {final_df['country'].nunique()} countries.")

        # Display note about transformation
        st.warning("""
        **Important Transformation Note:**
        The data has been temporally disaggregated using the Denton method, ensuring that the 
        quarterly/monthly series sum (or average) to the original annual totals. 
        Missing values were filled by linear interpolation per country. 
        Log transformation (if applied) was performed after disaggregation, 
        and zero/negative values were set to NaN before taking logs. 
        Use with caution — this process smooths short-term volatility.
        """)

        # --- Visualization Buttons ---
        st.subheader("Visual Comparison of Data Before and After Transformation")

        selected_country = st.selectbox("Select Country", sorted(df["country"].unique()))
        selected_var = var

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Show Original Data"):
                orig = df[df["country"] == selected_country]
                plt.figure(figsize=(8, 4))
                plt.plot(orig["year"], orig[selected_var], marker="o", label="Original")
                plt.title(f"Original {selected_var} for {selected_country}")
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                st.pyplot(plt.gcf())

        with col2:
            if st.button("Show Converted Data"):
                conv = final_df[final_df["country"] == selected_country]
                plt.figure(figsize=(8, 4))
                plt.plot(conv["year"], conv[selected_var], marker="o", color='orange', label="Converted")
                plt.title(f"Converted {selected_var} ({target_freq}) for {selected_country}")
                plt.xticks(rotation=45, ha='right')
                plt.legend()
                st.pyplot(plt.gcf())

        # --- Download option ---
        buffer = BytesIO()
        final_df.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download Converted Dataset", buffer, file_name="converted_panel_data.csv", mime="text/csv")

    else:
        st.error("No data could be converted. Please check your input file or method settings.")
