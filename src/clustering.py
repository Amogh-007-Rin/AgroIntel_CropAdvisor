import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

def perform_kmeans_clustering(X, n_clusters=4):
    """
    Perform K-means clustering on the feature data.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    n_clusters : int, default=4
        Number of clusters to form
        
    Returns:
    --------
    kmeans : KMeans object
        Fitted K-means clustering model
    labels : numpy.ndarray
        Cluster labels for each data point
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    return kmeans, labels

def find_optimal_k(X, max_k=10):
    """
    Find the optimal number of clusters using silhouette score.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    max_k : int, default=10
        Maximum number of clusters to test
        
    Returns:
    --------
    silhouette_scores : list
        Silhouette scores for different k values
    inertia_values : list
        Inertia values for different k values
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate silhouette scores for different k values
    silhouette_scores = []
    inertia_values = []
    
    # Skip k=1 as silhouette score requires at least 2 clusters
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_scaled, labels)
        silhouette_scores.append(silhouette_avg)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia_values.append(kmeans.inertia_)
    
    return silhouette_scores, inertia_values

def perform_dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the feature data.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    eps : float, default=0.5
        Maximum distance between two samples for them to be considered in the same neighborhood
    min_samples : int, default=5
        Number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
    --------
    dbscan : DBSCAN object
        Fitted DBSCAN clustering model
    labels : numpy.ndarray
        Cluster labels for each data point
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    return dbscan, labels

def visualize_clusters(X, labels, feature_names, method='kmeans', centers=None):
    """
    Visualize the clustering results using PCA.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    labels : numpy.ndarray
        Cluster labels
    feature_names : list
        List of feature names
    method : str, default='kmeans'
        Clustering method used ('kmeans' or 'dbscan')
    centers : numpy.ndarray, default=None
        Cluster centers (only for K-means)
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a DataFrame with scaled features and cluster labels
    df_clusters = pd.DataFrame(X_scaled, columns=feature_names)
    df_clusters['Cluster'] = labels.astype(str)
    
    # Check if there are noise points (only in DBSCAN)
    if method == 'dbscan' and -1 in labels:
        df_clusters.loc[df_clusters['Cluster'] == '-1', 'Cluster'] = 'Noise'
    
    # Visualize clusters using 3D scatter plot with N, P, K as axes
    st.write("### 3D Cluster Visualization: Soil Nutrients (N, P, K)")
    
    # Select N, P, K columns from original data for better interpretability
    df_plot = pd.DataFrame({
        'N': X['N'],
        'P': X['P'],
        'K': X['K'],
        'Cluster': df_clusters['Cluster']
    })
    
    fig = px.scatter_3d(
        df_plot,
        x='N',
        y='P',
        z='K',
        color='Cluster',
        title=f'{method.upper()} Clustering: Soil Nutrients (N, P, K)',
        labels={'N': 'Nitrogen (kg/ha)', 'P': 'Phosphorus (kg/ha)', 'K': 'Potassium (kg/ha)'},
        opacity=0.7
    )
    
    # Add cluster centers for K-means
    if method == 'kmeans' and centers is not None:
        centers_unscaled = scaler.inverse_transform(centers)
        for i, center in enumerate(centers_unscaled):
            fig.add_trace(
                go.Scatter3d(
                    x=[center[feature_names.index('N')]],
                    y=[center[feature_names.index('P')]],
                    z=[center[feature_names.index('K')]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        symbol='diamond',
                        color=i,
                        line=dict(color='black', width=2)
                    ),
                    name=f'Center {i}'
                )
            )
    
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Create pairplot for additional insights
    if st.checkbox("Show Feature Pairplot"):
        # Use only a sample if the dataset is large
        if len(df_clusters) > 1000:
            df_sample = df_clusters.sample(n=1000, random_state=42)
        else:
            df_sample = df_clusters
            
        # Select a subset of features for clarity
        selected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'Cluster']
        
        # Create pairplot
        fig, ax = plt.subplots(figsize=(10, 8))
        pairplot_fig = sns.pairplot(
            df_sample[selected_features], 
            hue='Cluster',
            diag_kind='kde',
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'alpha': 0.6}
        )
        plt.tight_layout()
        st.pyplot(pairplot_fig.fig)

def analyze_clusters(X, labels, y=None):
    """
    Analyze the characteristics of each cluster.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    labels : numpy.ndarray
        Cluster labels
    y : pandas.Series, default=None
        Crop labels (if available)
    """
    # Create a DataFrame with features and cluster labels
    df_clusters = X.copy()
    df_clusters['Cluster'] = labels
    
    # Create tabs for different analysis views
    tabs = st.tabs(["Cluster Profiles", "Feature Distributions", "Crop Distribution"])
    
    # Tab 1: Cluster Profiles
    with tabs[0]:
        st.write("### Cluster Profiles: Average Values")
        
        # Calculate mean values for each cluster
        cluster_profiles = df_clusters.groupby('Cluster').mean()
        
        # Display as a table
        st.table(cluster_profiles.style.highlight_max(axis=0))
        
        # Create radar chart for cluster profiles
        st.write("### Cluster Comparison: Radar Chart")
        
        # Normalize the data for radar chart
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.min()) / (cluster_profiles.max() - cluster_profiles.min())
        
        # Create radar chart
        fig = go.Figure()
        
        for cluster in cluster_profiles_norm.index:
            fig.add_trace(go.Scatterpolar(
                r=cluster_profiles_norm.loc[cluster].values.tolist(),
                theta=cluster_profiles_norm.columns.tolist(),
                fill='toself',
                name=f'Cluster {cluster}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Normalized Feature Values by Cluster",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Feature Distributions
    with tabs[1]:
        st.write("### Feature Distributions by Cluster")
        
        # Select feature for visualization
        feature = st.selectbox("Select feature to visualize:", X.columns)
        
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Cluster', y=feature, data=df_clusters, ax=ax)
        plt.title(f'{feature} Distribution by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        st.pyplot(fig)
    
    # Tab 3: Crop Distribution (if labels are available)
    with tabs[2]:
        if y is not None:
            st.write("### Crop Distribution by Cluster")
            
            # Add crop labels to the data
            df_clusters['Crop'] = y.values
            
            # Create crosstab
            crop_cluster = pd.crosstab(
                df_clusters['Cluster'], 
                df_clusters['Crop'],
                normalize='index'
            ) * 100
            
            # Display as a table
            st.write("Percentage of each crop type within clusters:")
            st.table(crop_cluster.style.format("{:.1f}%").highlight_max(axis=1))
            
            # Create stacked bar chart
            fig = px.bar(
                crop_cluster.reset_index().melt(id_vars='Cluster', var_name='Crop', value_name='Percentage'),
                x='Cluster',
                y='Percentage',
                color='Crop',
                title='Crop Distribution by Cluster',
                labels={'Percentage': 'Percentage of Crops (%)'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            st.markdown("""
            ### Key Insights from Cluster Analysis:
            
            - Clusters represent different growing conditions that favor specific crops
            - Clusters with similar profiles may support similar types of crops
            - Cluster analysis can help in identifying optimal growing conditions for specific crops
            """)
        else:
            st.warning("Crop labels not available for analysis.")

def plot_optimal_k(X, max_k=10):
    """
    Plot metrics to find the optimal number of clusters.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    max_k : int, default=10
        Maximum number of clusters to test
    """
    # Calculate metrics for different k values
    silhouette_scores, inertia_values = find_optimal_k(X, max_k)
    k_values = list(range(2, max_k + 1))
    
    # Create tabs for different metrics
    tabs = st.tabs(["Elbow Method", "Silhouette Score"])
    
    # Tab 1: Elbow Method
    with tabs[0]:
        st.write("### Elbow Method for Optimal K")
        st.write("""
        The Elbow Method plots the sum of squared distances from each point to its assigned center (inertia).
        The optimal K is where the curve forms an "elbow" - adding more clusters provides diminishing returns.
        """)
        
        # Plot elbow curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, inertia_values, 'bo-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        ax.grid(True)
        st.pyplot(fig)
    
    # Tab 2: Silhouette Score
    with tabs[1]:
        st.write("### Silhouette Score for Optimal K")
        st.write("""
        The Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
        Higher silhouette scores indicate better-defined clusters. The optimal K maximizes this score.
        """)
        
        # Plot silhouette scores
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(k_values, silhouette_scores, 'ro-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score for Different k Values')
        ax.grid(True)
        st.pyplot(fig)
    
    # Automatically suggest optimal k
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    
    # Find elbow point (using a simple heuristic)
    # Calculate the second derivative to find the point of maximum curvature
    second_derivative = np.diff(np.diff(inertia_values))
    optimal_k_elbow = k_values[np.argmax(np.abs(second_derivative)) + 1]
    
    st.info(f"""
    **Suggested optimal number of clusters:**
    - Based on Silhouette Score: {optimal_k_silhouette}
    - Based on Elbow Method: {optimal_k_elbow}
    """)

def run_clustering_analysis(X, y=None):
    """
    Run the complete clustering analysis workflow.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series, default=None
        Crop labels (if available)
    """
    st.header("Clustering Analysis")
    st.write("""
    Clustering analysis helps identify natural groupings in the data based on soil and environmental factors.
    These clusters can represent different growing conditions that may be suitable for specific crops.
    """)
    
    # Create tabs for different clustering methods
    tabs = st.tabs(["K-means Clustering", "DBSCAN Clustering", "Optimal Clusters"])
    
    # Tab 1: K-means Clustering
    with tabs[0]:
        st.subheader("K-means Clustering")
        
        # Select number of clusters
        n_clusters = st.slider("Number of clusters (k):", min_value=2, max_value=10, value=4)
        
        # Perform K-means clustering
        kmeans, labels = perform_kmeans_clustering(X, n_clusters=n_clusters)
        
        # Visualize clusters
        visualize_clusters(X, labels, X.columns, method='kmeans', centers=kmeans.cluster_centers_)
        
        # Analyze clusters
        analyze_clusters(X, labels, y)
    
    # Tab 2: DBSCAN Clustering
    with tabs[1]:
        st.subheader("DBSCAN Clustering")
        st.write("""
        DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm.
        It groups together points that are closely packed, while marking points in low-density regions as outliers.
        """)
        
        # Select DBSCAN parameters
        col1, col2 = st.columns(2)
        
        with col1:
            eps = st.slider("Maximum distance between points (eps):", 
                          min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        
        with col2:
            min_samples = st.slider("Minimum points in neighborhood (min_samples):", 
                                  min_value=2, max_value=20, value=5)
        
        # Perform DBSCAN clustering
        _, labels = perform_dbscan_clustering(X, eps=eps, min_samples=min_samples)
        
        # Count number of clusters (-1 represents noise points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1) if -1 in labels else 0
        
        st.write(f"Number of clusters found: {n_clusters}")
        st.write(f"Number of noise points: {n_noise} ({n_noise/len(X)*100:.1f}% of data)")
        
        # Visualize clusters
        visualize_clusters(X, labels, X.columns, method='dbscan')
        
        # Analyze clusters (exclude noise points for analysis)
        if -1 in labels and n_clusters > 0:
            filtered_X = X.iloc[labels != -1].copy()
            filtered_labels = labels[labels != -1]
            
            if y is not None:
                filtered_y = y.iloc[labels != -1]
            else:
                filtered_y = None
            
            analyze_clusters(filtered_X, filtered_labels, filtered_y)
        elif n_clusters > 0:
            analyze_clusters(X, labels, y)
        else:
            st.warning("No clusters found with current parameters. Try adjusting eps and min_samples.")
    
    # Tab 3: Optimal Number of Clusters
    with tabs[2]:
        st.subheader("Finding the Optimal Number of Clusters")
        
        # Select maximum k to test
        max_k = st.slider("Maximum number of clusters to test:", min_value=3, max_value=15, value=10)
        
        # Plot metrics for optimal k
        plot_optimal_k(X, max_k=max_k)