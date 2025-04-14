import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np

def plot_feature_distributions(df, feature_names):
    """
    Plot the distribution of each feature in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features
    feature_names : list
        List of feature names to plot
    """
    # Create tabs for each type of feature
    tabs = st.tabs(["Soil Nutrients (N, P, K)", "Environmental Factors"])
    
    # Tab 1: Soil Nutrients
    with tabs[0]:
        fig = px.histogram(
            df, 
            x=["N", "P", "K"], 
            facet_col="label",
            facet_col_wrap=3,
            histnorm='percent',
            title="Distribution of Soil Nutrients by Crop Type",
            labels={"value": "Nutrient Value (kg/ha)", "variable": "Nutrient Type"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Environmental Factors
    with tabs[1]:
        fig = px.histogram(
            df, 
            x=["temperature", "humidity", "ph", "rainfall"], 
            facet_col="label",
            facet_col_wrap=2,
            histnorm='percent',
            title="Distribution of Environmental Factors by Crop Type",
            labels={"value": "Value", "variable": "Environmental Factor"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df):
    """
    Plot correlation heatmap for the features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing features
    """
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr, 
        text_auto=True, 
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Find and display the strongest correlations
    corr_values = corr.abs().unstack().sort_values(ascending=False)
    # Remove self-correlations
    corr_values = corr_values[corr_values < 1]
    
    if len(corr_values) > 0:
        st.subheader("Strongest Feature Correlations")
        top_corrs = pd.DataFrame(corr_values[:5], columns=['Correlation Strength'])
        top_corrs.index.names = ['Feature 1', 'Feature 2']
        st.table(top_corrs.reset_index())

def plot_pairplot(df, feature_names, target_col):
    """
    Create a pairplot to visualize relationships between features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_names : list
        List of feature names to include in the pairplot
    target_col : str
        Name of the target column for color coding
    """
    # Create a sample if dataset is large
    if len(df) > 1000:
        df_sample = df.sample(n=1000, random_state=42)
    else:
        df_sample = df
    
    # Plot using Seaborn
    fig, ax = plt.subplots(figsize=(12, 10))
    pairplot_fig = sns.pairplot(
        df_sample, 
        vars=feature_names, 
        hue=target_col, 
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
        diag_kws={'alpha': 0.6}
    )
    plt.tight_layout()
    st.pyplot(pairplot_fig.fig)

def plot_crop_distribution(df):
    """
    Plot the distribution of crop types in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing crop labels
    """
    # Count the occurrences of each crop type
    crop_counts = df['label'].value_counts().reset_index()
    crop_counts.columns = ['Crop', 'Count']
    
    # Create bar chart
    fig = px.bar(
        crop_counts, 
        x='Crop', 
        y='Count',
        color='Crop',
        title='Distribution of Crop Types in Dataset',
        labels={'Count': 'Number of Samples', 'Crop': 'Crop Type'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_feature_importance(importance_df):
    """
    Plot feature importance from a trained model.
    
    Parameters:
    -----------
    importance_df : pandas.DataFrame
        DataFrame containing feature importance scores
    """
    # Sort by importance for better visualization
    importance_df = importance_df.sort_values('Importance', ascending=True)
    
    # Create horizontal bar chart
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance in Crop Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def plot_soil_nutrient_distribution(df):
    """
    Create a 3D scatter plot showing the distribution of soil nutrients (N, P, K)
    by crop type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing soil nutrient data and crop labels
    """
    # Create 3D scatter plot using Plotly
    fig = px.scatter_3d(
        df, 
        x='N', 
        y='P', 
        z='K',
        color='label',
        title='3D Distribution of Soil Nutrients by Crop Type',
        labels={'N': 'Nitrogen (kg/ha)', 'P': 'Phosphorus (kg/ha)', 'K': 'Potassium (kg/ha)'},
        opacity=0.7
    )
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Nitrogen (N)',
            yaxis_title='Phosphorus (P)',
            zaxis_title='Potassium (K)'
        ),
        height=700
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    ### Insights from Soil Nutrient Distribution
    
    This 3D plot shows how different crops cluster based on their nitrogen (N), phosphorus (P), and potassium (K) requirements.
    Each point represents a sample, and colors indicate different crop types.
    
    Key observations:
    - Some crops form distinct clusters, indicating specific nutrient requirements
    - Overlapping clusters suggest crops with similar nutrient needs
    - Distance between clusters indicates dissimilarity in nutrient requirements
    """)
