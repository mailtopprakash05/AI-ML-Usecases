# Employee Salary Anomaly Detection

A user-friendly web application that helps identify unusual salary patterns in your employee data using machine learning. Built with Streamlit and scikit-learn's Isolation Forest algorithm.

## What This App Does

- **Anomaly Detection**: Automatically identifies employees with unusually high or low salaries compared to the overall pattern
- **Visual Analysis**: Shows salary distributions and highlights potential anomalies
- **Easy to Use**: Just upload a CSV file and get instant results
- **Interactive**: Adjust sensitivity of anomaly detection with a simple slider

## CSV File Requirements

Your CSV file must have these two columns (case-insensitive):
1. `employee_name` - The name of the employee
2. `salary` - The salary amount (numeric)

Example CSV format:
```csv
employee_name,salary
John Smith,52000
Jane Doe,48000
Robert Brown,51000
```

💡 **Tip**: A sample CSV file can be downloaded directly from the app's interface.

## Quick Start Guide

1. **Install Required Packages**
   ```sh
   pip install -r requirements.txt
   ```
   Or install individually:
   ```sh
   pip install streamlit scikit-learn matplotlib pandas numpy
   ```

2. **Start the Application**
   ```sh
   streamlit run anomaly_app.py
   ```

3. **Use the App**
   - Open the URL shown in your terminal (typically http://localhost:8501) or access it on cloud : https://anomalyapppy-d8yti2mho3ertvxjuejywf.streamlit.app/
   - Click "Browse files" to upload your CSV
   - Adjust the anomaly sensitivity slider if needed
   - View results in the interactive charts and tables

## Understanding the Results

- **Blue dots** in the chart represent normal salaries
- **Red dots** highlight potential anomalies
- The app shows a table of identified anomalies with employee names
- Adjust the "Anomaly fraction" slider to control sensitivity:
  - Lower values (e.g., 0.01) detect only extreme anomalies
  - Higher values (e.g., 0.1) detect more subtle variations

## Troubleshooting

- Make sure your CSV has both required columns (`employee_name` and `salary`)
- Column names are case-insensitive (e.g., "Employee_Name" or "SALARY" will work)
- Salary values should be numbers without currency symbols or commas
- If nothing happens after upload, check the error messages at the top of the page

## Need Help?

- Download the sample CSV from the app to see the correct format
- Check the preview table after upload to ensure your data loaded correctly
- For issues or improvements, please open an issue on GitLab
