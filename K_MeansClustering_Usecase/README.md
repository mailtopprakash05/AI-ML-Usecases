# Healthcare Patient Clustering Analysis

An interactive web application that uses K-Means clustering to analyze patient health data and identify distinct patient groups based on their health metrics. Built with Streamlit and scikit-learn.

## Features

- 📊 Interactive data visualization
- 🔍 Customizable clustering parameters
- 🎯 Multiple feature selection
- 📈 Real-time cluster analysis
- 💾 Export results as CSV
- 📱 Responsive UI design

## Sample Dataset Format

The application expects a CSV file with the following columns:

| Column           | Description                              | Example Values |
|-----------------|------------------------------------------|---------------|
| patient_id      | Unique identifier for each patient        | P001, P002    |
| age            | Patient's age in years                    | 45, 62        |
| bmi            | Body Mass Index                           | 24.5, 31.2    |
| blood_pressure | Systolic blood pressure                   | 120, 145      |
| glucose        | Blood glucose level                       | 85, 110       |
| cholesterol    | Cholesterol level                         | 190, 245      |
| active_lifestyle| Binary indicator (0=sedentary, 1=active) | 0, 1          |

## Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run clustering_app.py
   ```

3. **Use the App**
   - Open your browser to the URL shown in the terminal
   - Upload your CSV file or use the sample data
   - Select features for clustering
   - Adjust the number of clusters
   - Explore the interactive visualizations

## How to Use

1. **Data Upload**
   - Use the sidebar to upload your CSV file
   - Download and view the sample CSV for the correct format
   - The app will validate your data for required columns

2. **Feature Selection**
   - Choose which health metrics to include in the clustering
   - Select at least 2 features
   - Different combinations may reveal different patterns

3. **Clustering Settings**
   - Adjust the number of clusters (k) using the slider
   - The app will automatically update all visualizations

4. **Analyze Results**
   - View the 2D scatter plot with customizable axes
   - Examine cluster characteristics in the summary table
   - Explore feature distributions across clusters
   - Check cluster sizes in the pie chart

5. **Export Results**
   - Download the analyzed data with cluster labels
   - Use the exported CSV for further analysis

## Visualization Types

1. **Interactive Scatter Plot**
   - Choose any two features for the x and y axes
   - Hover over points to see patient details
   - Clusters are color-coded for easy identification

2. **Cluster Characteristics Table**
   - Mean values of all features for each cluster
   - Helps identify cluster patterns and differences

3. **Feature Distribution Plots**
   - Violin plots showing feature distributions by cluster
   - Select different features to visualize

4. **Cluster Size Distribution**
   - Pie chart showing the proportion of patients in each cluster
   - Helps identify dominant patient groups

## Tips for Best Results

- Start with 3-4 clusters and adjust based on the visualizations
- Include diverse features for more meaningful clusters
- Look for patterns in the cluster characteristics table
- Export results for different feature combinations
- Compare results with domain knowledge

## Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- numpy
- plotly
- seaborn
- matplotlib

## Contributing

Feel free to submit issues and enhancement requests!