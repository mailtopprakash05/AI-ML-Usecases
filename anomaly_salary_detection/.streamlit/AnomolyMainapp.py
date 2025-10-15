import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.title("Employee Salary Anomaly Detection")

# Upload CSV file
st.subheader("Upload Employee Salary CSV")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
	data_df = pd.read_csv(uploaded_file)
	st.write("Preview of uploaded data:")
	st.dataframe(data_df.head())
	if "salary" not in data_df.columns:
		st.error("CSV must have a 'salary' column.")
	else:
		anomaly_fraction = st.slider("Anomaly fraction", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
		model = IsolationForest(contamination=anomaly_fraction, random_state=42)
		data_df["anomaly"] = model.fit_predict(data_df[["salary"]])
		data_df["anomaly"] = data_df["anomaly"].map({1: "Normal", -1: "Anomaly"})
		st.subheader("Anomaly Detection Results")
		st.dataframe(data_df)
		fig, ax = plt.subplots()
		colors = data_df["anomaly"].map({"Normal": "blue", "Anomaly": "red"})
		ax.scatter(range(len(data_df)), data_df["salary"], c=colors)
		ax.set_xlabel("Index")
		ax.set_ylabel("Salary")
		ax.set_title("Salary Anomaly Detection")
		st.pyplot(fig)
		n_anomalies = (data_df["anomaly"] == "Anomaly").sum()
		st.success(f"Detected {n_anomalies} anomalies out of {len(data_df)} employees.")
		# Display table of anomalous employees with name and salary
		if "employee_name" in data_df.columns:
			anomalous_employees = data_df.loc[data_df["anomaly"] == "Anomaly", ["employee_name", "salary"]]
			if not anomalous_employees.empty:
				st.subheader("Anomalous Employees")
				st.table(anomalous_employees)
			else:
				st.info("No anomalous employees detected.")
	n_samples = st.slider("Number of samples", min_value=100, max_value=1000, value=300, step=100)
