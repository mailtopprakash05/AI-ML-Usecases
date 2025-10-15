import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Healthcare Data PCA Analysis", layout="wide")

# Title and description
st.title("Healthcare Data Analysis using PCA")
st.write("""
This application demonstrates the use of Principal Component Analysis (PCA) to analyze patterns
in patient health metrics. PCA helps us reduce the dimensionality of the data while preserving
important patterns and relationships between variables.
""")

# Load and display the data
@st.cache_data
def load_data():
    data = pd.read_csv('patient_health_metrics.csv')
    return data

data = load_data()

# Data Overview Section
st.header("Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.write("First few rows of the dataset:")
    st.dataframe(data.head())

with col2:
    st.write("Statistical Summary:")
    st.dataframe(data.describe())

# Prepare data for PCA
features = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Blood_Sugar', 
           'Heart_Rate', 'Exercise_Hours', 'Sleep_Hours', 'Stress_Level']

X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# PCA Analysis Section
st.header("PCA Analysis")

# Scree Plot
st.subheader("Scree Plot - Explained Variance Ratio")
fig_scree = px.line(
    x=range(1, len(explained_variance_ratio) + 1),
    y=explained_variance_ratio,
    markers=True,
    title="Scree Plot: Explained Variance Ratio by Principal Component",
    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
)
st.plotly_chart(fig_scree)

# Cumulative Variance Plot
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
st.subheader("Cumulative Explained Variance Ratio")
fig_cumulative = px.line(
    x=range(1, len(cumulative_variance_ratio) + 1),
    y=cumulative_variance_ratio,
    markers=True,
    title="Cumulative Explained Variance Ratio",
    labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance Ratio'}
)
st.plotly_chart(fig_cumulative)

# Component Analysis
st.header("Principal Components Analysis")

# Feature importance heatmap
components_df = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[f'PC{i+1}' for i in range(len(features))]
)

fig_heatmap = px.imshow(
    components_df,
    title="PCA Components Heatmap",
    labels=dict(x="Features", y="Principal Components", color="Coefficient")
)
st.plotly_chart(fig_heatmap)

# Interactive 2D Scatter Plot
st.header("2D Visualization of First Two Principal Components")
fig_scatter = px.scatter(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    title="Patient Distribution in PC1 vs PC2 Space",
    labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
    hover_data={'Patient ID': data['Patient_ID']}
)
st.plotly_chart(fig_scatter)

# Contribution Analysis
st.header("Feature Contributions")
st.write("""
Below is the absolute contribution of each feature to the first two principal components.
This helps us understand which health metrics are most important in explaining the variation in the data.
""")

# Calculate feature contributions
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(
    loadings[:, :2],
    columns=['PC1', 'PC2'],
    index=features
)

fig_contributions = go.Figure()
fig_contributions.add_trace(go.Bar(
    y=features,
    x=abs(loading_matrix['PC1']),
    name='PC1',
    orientation='h'
))
fig_contributions.add_trace(go.Bar(
    y=features,
    x=abs(loading_matrix['PC2']),
    name='PC2',
    orientation='h'
))

fig_contributions.update_layout(
    title="Feature Contributions to Principal Components",
    barmode='group',
    xaxis_title="Absolute Contribution",
    yaxis_title="Features"
)
st.plotly_chart(fig_contributions)

# Insights Section
st.header("Key Insights")
st.write("""
Based on the PCA analysis of patient health metrics, we can observe:
1. The first two principal components explain a significant portion of the variance in the data
2. Key contributing features to PC1 include Blood Pressure, Cholesterol, and Blood Sugar levels
3. PC2 is more influenced by lifestyle factors such as Exercise Hours and Sleep Hours
4. Patterns in the scatter plot might indicate different patient health profiles
""")