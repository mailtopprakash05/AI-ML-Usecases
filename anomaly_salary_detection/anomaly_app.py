import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("Employee Salary Anomaly Detection")

# Landing instructions and sample CSV
st.markdown(
    "Upload a CSV with two columns: `employee_name` and `salary`.\n\n"
    "Column names are case-insensitive but must be present. You can download a sample CSV to see the expected format."
)

st.subheader("Upload Employee Salary CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"]) 

# Provide a small sample CSV for users to download
sample_csv = "employee_name,salary\nAlice,52000\nBob,48000\nEve,200000\n"
st.download_button("Download sample CSV", data=sample_csv, file_name="employee_salaries_sample.csv", mime="text/csv")

if uploaded_file:
    data_df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(data_df.head())

    # Normalize column names to lower-case for validation but keep original df
    cols_lower = [c.lower() for c in data_df.columns]
    has_salary = "salary" in cols_lower
    has_name = "employee_name" in cols_lower

    if not (has_salary and has_name):
        missing = []
        if not has_name:
            missing.append("employee_name")
        if not has_salary:
            missing.append("salary")
        st.error(f"CSV must include the following column(s): {', '.join(missing)}. (case-insensitive)")
    else:
        # Map to original column names (preserve case if user used different casing)
        col_map = {c.lower(): c for c in data_df.columns}
        salary_col = col_map["salary"]
        name_col = col_map["employee_name"]

        anomaly_fraction = st.slider("Anomaly fraction", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
        model = IsolationForest(contamination=anomaly_fraction, random_state=42)
        data_df["anomaly"] = model.fit_predict(data_df[[salary_col]])
        data_df["anomaly"] = data_df["anomaly"].map({1: "Normal", -1: "Anomaly"})
        st.subheader("Anomaly Detection Results")
        st.dataframe(data_df)
        fig, ax = plt.subplots()
        colors = data_df["anomaly"].map({"Normal": "blue", "Anomaly": "red"})
        ax.scatter(range(len(data_df)), data_df[salary_col], c=colors)
        ax.set_xlabel("Index")
        ax.set_ylabel("Salary")
        ax.set_title("Salary Anomaly Detection")
        st.pyplot(fig)
        n_anomalies = (data_df["anomaly"] == "Anomaly").sum()
        st.success(f"Detected {n_anomalies} anomalies out of {len(data_df)} employees.")
        # Display table of anomalous employees with name and salary
        anomalous_employees = data_df.loc[data_df["anomaly"] == "Anomaly", [name_col, salary_col]]
        if not anomalous_employees.empty:
            st.subheader("Anomalous Employees")
            # rename columns to friendly names for display
            anomalous_employees = anomalous_employees.rename(columns={name_col: "employee_name", salary_col: "salary"})
            st.table(anomalous_employees)
        else:
            st.info("No anomalous employees detected.")
