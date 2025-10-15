import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Set page config
st.set_page_config(page_title="Healthcare Patient Clustering", page_icon="🏥", layout="wide")

# Title and description
st.title("Healthcare Patient Clustering Analysis 🏥")
st.markdown("""
This application uses K-Means clustering to group patients based on their health metrics.
Upload your patient data CSV file to discover patterns and potential risk groups.
""")

# Sidebar for parameters
st.sidebar.header("Clustering Parameters")

# File upload
st.sidebar.subheader("1. Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type="csv",
    help="Upload a CSV file with patient health metrics. Download the sample file below to see the required format."
)

# Download sample data button
try:
    import os
    sample_file_path = os.path.join(os.path.dirname(__file__), 'patient_health_data.csv')
    with open(sample_file_path, 'r') as f:
        sample_data = f.read()
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=sample_data,
        file_name="sample_patient_data.csv",
        mime="text/csv"
    )
except Exception as e:
    st.sidebar.warning("Sample data file not available. Please ensure 'patient_health_data.csv' exists in the same directory as this script.")

# Required columns info
st.sidebar.markdown("""
### Required Columns:
- patient_id
- age
- bmi
- blood_pressure
- glucose
- cholesterol
- active_lifestyle (0 or 1)
""")

if uploaded_file is not None:
    # Load and preview data
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns
    required_columns = ['age', 'bmi', 'blood_pressure', 'glucose', 'cholesterol', 'active_lifestyle']
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
    else:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Feature selection
        st.sidebar.subheader("2. Select Features")
        features = st.sidebar.multiselect(
            "Select features for clustering",
            options=[col for col in df.columns if col != 'patient_id'],
            default=['age', 'bmi', 'blood_pressure', 'glucose', 'cholesterol']
        )
        
        if len(features) >= 2:  # Need at least 2 features for clustering
            try:
                # Number of clusters
                st.sidebar.subheader("3. Clustering Settings")
                n_clusters = st.sidebar.slider("Number of clusters (k)", 2, 8, 3)
                
                # Prepare data for clustering
                X = df[features]
                
                # Check for non-numeric data
                if not X.apply(pd.to_numeric, errors='coerce').notna().all().all():
                    st.error("Selected features contain non-numeric data. Please ensure all selected features contain only numbers.")
                    st.stop()
                
                # Check for missing values
                if X.isnull().any().any():
                    st.warning("Data contains missing values. They will be replaced with mean values.")
                    X = X.fillna(X.mean())
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X_scaled)
            except Exception as e:
                st.error(f"An error occurred during clustering: {str(e)}")
                st.stop()
            
            # Results section
            st.header("Clustering Results")
            
            # 1. Interactive scatter plot
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("2D Visualization")
                x_axis = st.selectbox("X-axis", features, index=0)
                y_axis = st.selectbox("Y-axis", features, index=1)
                
                fig = px.scatter(
                    df,
                    x=x_axis,
                    y=y_axis,
                    color='Cluster',
                    hover_data=['patient_id'] + features,
                    title=f"Patient Clusters: {x_axis} vs {y_axis}"
                )
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Cluster Characteristics")
                cluster_stats = df.groupby('Cluster')[features].mean()
                st.dataframe(cluster_stats.round(2))
            
            # 2. Feature distributions by cluster
            st.subheader("Feature Distributions by Cluster")
            
            # Create violin plots
            fig, ax = plt.subplots(figsize=(12, 6))
            selected_feature = st.selectbox("Select feature to visualize", features)
            sns.violinplot(data=df, x='Cluster', y=selected_feature)
            plt.title(f"{selected_feature} Distribution by Cluster")
            st.pyplot(fig)
            
            # 3. Cluster sizes
            st.subheader("Cluster Sizes")
            cluster_sizes = df['Cluster'].value_counts().sort_index()
            fig = px.pie(
                values=cluster_sizes.values,
                names=cluster_sizes.index,
                title="Distribution of Patients Across Clusters"
            )
            st.plotly_chart(fig)
            
            # 4. Export results
            st.subheader("Export Results")
            
            # Add cluster labels to original data
            output_df = df.copy()
            output_df['Cluster'] = output_df['Cluster'].apply(lambda x: f'Cluster {x}')
            
            # Convert to CSV
            csv = output_df.to_csv(index=False)
            st.download_button(
                label="Download Clustered Data (CSV)",
                data=csv,
                file_name="clustered_patient_data.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("Please select at least 2 features for clustering.")