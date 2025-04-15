import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def create_polynomial_features(X, degree=2):
    """
    Create polynomial features from the original features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    degree : int, default=2
        Degree of the polynomial features
        
    Returns:
    --------
    X_poly : pandas.DataFrame
        Feature matrix with polynomial features
    poly_features : list
        List of polynomial feature names
    """
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_array = poly.fit_transform(X)
    
    # Generate feature names
    feature_names = X.columns
    poly_features = poly.get_feature_names_out(feature_names)
    
    # Convert to DataFrame
    X_poly = pd.DataFrame(X_poly_array, columns=poly_features, index=X.index)
    
    return X_poly, poly_features

def create_ratio_features(X):
    """
    Create ratio features from the original features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
        
    Returns:
    --------
    X_ratio : pandas.DataFrame
        Feature matrix with ratio features
    ratio_features : list
        List of ratio feature names
    """
    X_ratio = X.copy()
    ratio_features = []
    
    # Create N:P:K ratio feature (common in agriculture)
    X_ratio['N_P_ratio'] = X['N'] / X['P']
    X_ratio['N_K_ratio'] = X['N'] / X['K']
    X_ratio['P_K_ratio'] = X['P'] / X['K']
    
    # Temperature to humidity ratio (relevant for plant growth)
    X_ratio['temp_humidity_ratio'] = X['temperature'] / X['humidity']
    
    # PH to rainfall ratio (affects nutrient availability)
    X_ratio['ph_rainfall_ratio'] = X['ph'] / X['rainfall']
    
    ratio_features = ['N_P_ratio', 'N_K_ratio', 'P_K_ratio', 
                     'temp_humidity_ratio', 'ph_rainfall_ratio']
    
    return X_ratio, ratio_features

def create_interaction_features(X):
    """
    Create interaction features from the original features.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
        
    Returns:
    --------
    X_inter : pandas.DataFrame
        Feature matrix with interaction features
    interaction_features : list
        List of interaction feature names
    """
    X_inter = X.copy()
    interaction_features = []
    
    # N, P, K interactions (relevant for plant nutrition)
    X_inter['N_P_interaction'] = X['N'] * X['P']
    X_inter['N_K_interaction'] = X['N'] * X['K']
    X_inter['P_K_interaction'] = X['P'] * X['K']
    
    # Temperature and humidity interaction (affects plant growth)
    X_inter['temp_humidity_interaction'] = X['temperature'] * X['humidity']
    
    # PH and rainfall interaction (affects nutrient availability)
    X_inter['ph_rainfall_interaction'] = X['ph'] * X['rainfall']
    
    interaction_features = ['N_P_interaction', 'N_K_interaction', 'P_K_interaction',
                           'temp_humidity_interaction', 'ph_rainfall_interaction']
    
    return X_inter, interaction_features

def perform_pca(X, n_components=3):
    """
    Perform PCA dimensionality reduction.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    n_components : int, default=3
        Number of principal components to keep
        
    Returns:
    --------
    X_pca : pandas.DataFrame
        Feature matrix with principal components
    pca : PCA object
        Fitted PCA object
    """
    # Standardize the features
    X_scaled = (X - X.mean()) / X.std()
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca_array = pca.fit_transform(X_scaled)
    
    # Convert to DataFrame
    X_pca = pd.DataFrame(
        X_pca_array,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X.index
    )
    
    return X_pca, pca

def select_best_features(X, y, k=5):
    """
    Select the k best features based on ANOVA F-value.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    k : int, default=5
        Number of top features to select
        
    Returns:
    --------
    X_selected : pandas.DataFrame
        Feature matrix with selected features
    selector : SelectKBest object
        Fitted selector object
    """
    # Select k best features
    selector = SelectKBest(f_classif, k=k)
    X_selected_array = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    # Convert to DataFrame
    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_features,
        index=X.index
    )
    
    return X_selected, selector

def plot_feature_distributions_by_crop(df, feature_names, crop_names, num_crops=5):
    """
    Plot the distribution of features for specific crops.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_names : list
        List of feature names
    crop_names : list
        List of crop names
    num_crops : int, default=5
        Number of crops to show (most frequent)
    """
    if len(crop_names) > num_crops:
        # Select the most frequent crops
        top_crops = df['label'].value_counts()[:num_crops].index.tolist()
    else:
        top_crops = crop_names
    
    # Filter data for selected crops
    df_selected = df[df['label'].isin(top_crops)]
    
    # Create tabs for feature categories
    tabs = st.tabs(["Soil Nutrients", "Environmental Factors", "Feature Ratios"])
    
    # Tab 1: Soil Nutrients
    with tabs[0]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        soil_features = ['N', 'P', 'K']
        
        for i, feature in enumerate(soil_features):
            sns.boxplot(x='label', y=feature, data=df_selected, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution by Crop')
            axes[i].set_xlabel('Crop')
            axes[i].set_ylabel(feature)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tab 2: Environmental Factors
    with tabs[1]:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        env_features = ['temperature', 'humidity', 'ph', 'rainfall']
        
        for i, feature in enumerate(env_features):
            row, col = i // 2, i % 2
            sns.boxplot(x='label', y=feature, data=df_selected, ax=axes[row, col])
            axes[row, col].set_title(f'{feature} Distribution by Crop')
            axes[row, col].set_xlabel('Crop')
            axes[row, col].set_ylabel(feature)
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tab 3: Feature Ratios
    with tabs[2]:
        # Create ratio features for the selected crops
        X_selected = df_selected[feature_names]
        X_ratio, ratio_features = create_ratio_features(X_selected)
        df_ratio = pd.concat([df_selected['label'], X_ratio[ratio_features]], axis=1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, feature in enumerate(ratio_features):
            if i < 6:  # Display up to 6 ratio features
                row, col = i // 3, i % 3
                sns.boxplot(x='label', y=feature, data=df_ratio, ax=axes[row, col])
                axes[row, col].set_title(f'{feature} by Crop')
                axes[row, col].set_xlabel('Crop')
                axes[row, col].set_ylabel(feature)
                axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

def plot_pca_analysis(X, y, feature_names):
    """
    Perform and visualize PCA analysis.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector (crop labels)
    feature_names : list
        List of feature names
    """
    # Perform PCA
    X_pca, pca = perform_pca(X, n_components=3)
    
    # Add crop labels
    pca_df = pd.concat([X_pca, y], axis=1)
    pca_df.columns = ['PC1', 'PC2', 'PC3', 'label']
    
    # Create tabs for different PCA visualizations
    tabs = st.tabs(["PCA Components", "Feature Contributions", "Explained Variance"])
    
    # Tab 1: PCA Components
    with tabs[0]:
        st.write("### Principal Component Analysis (3D)")
        st.write("This 3D plot shows the crop data projected onto the first three principal components.")
        
        # Create 3D scatter plot using plotly
        import plotly.express as px
        
        fig = px.scatter_3d(
            pca_df, 
            x='PC1', 
            y='PC2', 
            z='PC3',
            color='label',
            title='PCA: Crop Data in 3D Space',
            labels={'PC1': 'Principal Component 1', 
                   'PC2': 'Principal Component 2', 
                   'PC3': 'Principal Component 3'},
            opacity=0.7
        )
        
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Feature Contributions
    with tabs[1]:
        st.write("### Feature Contributions to Principal Components")
        st.write("This heatmap shows how much each original feature contributes to each principal component.")
        
        # Get component loadings
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Create DataFrame for visualization
        loadings_df = pd.DataFrame(
            loadings,
            columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
            index=feature_names
        )
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(loadings_df, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
        plt.title('Feature Contributions to Principal Components')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Tab 3: Explained Variance
    with tabs[2]:
        st.write("### Explained Variance by Principal Components")
        st.write("This plot shows how much of the total variance in the data is explained by each principal component.")
        
        # Plot explained variance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        ax.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
        ax.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance by Principal Components')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        st.pyplot(fig)

def plot_feature_selection_results(X, y, feature_names):
    """
    Perform and visualize feature selection.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    feature_names : list
        List of feature names
    """
    # Select best features
    _, selector = select_best_features(X, y, k=len(feature_names))
    
    # Get scores
    scores = selector.scores_
    
    # Create DataFrame for visualization
    scores_df = pd.DataFrame({
        'Feature': feature_names,
        'Score': scores
    }).sort_values('Score', ascending=False)
    
    # Plot feature importance
    st.write("### Feature Importance Based on ANOVA F-value")
    st.write("This chart shows the importance of each feature in predicting crop suitability.")
    
    import plotly.express as px
    
    fig = px.bar(
        scores_df,
        x='Score',
        y='Feature',
        orientation='h',
        title='Feature Importance (ANOVA F-value)',
        color='Score',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top features
    st.write("### Top Features for Crop Prediction")
    st.table(scores_df.head())