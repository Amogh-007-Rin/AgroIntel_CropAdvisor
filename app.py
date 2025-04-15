import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
import os

from data_preprocessing import load_and_preprocess_data
from exploratory_analysis import (
    plot_feature_distributions, 
    plot_correlation_heatmap, 
    plot_pairplot,
    plot_crop_distribution,
    plot_feature_importance,
    plot_soil_nutrient_distribution
)
from modeling import (
    train_and_evaluate_models,
    predict_crop,
    feature_importance
)
from utils import (
    get_model_metrics,
    display_metrics,
    interpret_prediction
)
# Temporarily comment out advanced modules to debug the application
# from feature_engineering import (
#     create_polynomial_features,
#     create_ratio_features,
#     create_interaction_features,
#     plot_feature_distributions_by_crop,
#     plot_pca_analysis,
#     plot_feature_selection_results
# )
# from clustering import (
#     run_clustering_analysis
# )
# from advanced_modeling import (
#     run_advanced_model_evaluation
# )
# from seasonal_analysis import (
#     run_seasonal_analysis
# )

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# Page header
st.title("🌱 Crop Recommendation System")
st.markdown("""
This application analyzes soil and environmental conditions to recommend the optimal crop for your farm.
Use the interactive tools to explore data, evaluate models, and get personalized recommendations.
""")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Analysis", "Advanced Analysis", "Model Performance", "Advanced Modeling", "Prediction Tool", "Seasonal Analysis"]
)

# Load and preprocess data
@st.cache_data
def get_data():
    return load_and_preprocess_data("attached_assets/Crop_recommendation.csv")

# Load data
try:
    df, X, y, X_train, X_test, y_train, y_test, feature_names, target_names = get_data()
    
    # Check if models exist, if not train them
    models_exist = os.path.exists("random_forest_model.joblib") and \
                  os.path.exists("knn_model.joblib") and \
                  os.path.exists("svm_model.joblib")
    
    if not models_exist:
        with st.spinner("Training models for the first time..."):
            models, metrics = train_and_evaluate_models(X_train, y_train, X_test, y_test)
            # Save models
            joblib.dump(models["Random Forest"], "random_forest_model.joblib")
            joblib.dump(models["KNN"], "knn_model.joblib")
            joblib.dump(models["SVM"], "svm_model.joblib")
    else:
        # Load pre-trained models
        models = {
            "Random Forest": joblib.load("random_forest_model.joblib"),
            "KNN": joblib.load("knn_model.joblib"),
            "SVM": joblib.load("svm_model.joblib")
        }
        metrics = get_model_metrics(models, X_test, y_test)
    
    # Home Page
    if page == "Home":
        st.header("Welcome to the Crop Recommendation System")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Number of samples: {df.shape[0]}")
            st.write(f"Number of features: {len(feature_names)}")
            st.write(f"Number of crop types: {len(target_names)}")
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Crop Distribution")
            plot_crop_distribution(df)
            
            st.subheader("Features")
            st.markdown("""
            - **N**: Nitrogen content in soil (kg/ha)
            - **P**: Phosphorus content in soil (kg/ha)
            - **K**: Potassium content in soil (kg/ha)
            - **temperature**: Temperature in degree Celsius
            - **humidity**: Relative humidity in %
            - **ph**: pH value of the soil
            - **rainfall**: Rainfall in mm
            """)
    
    # Data Analysis Page
    elif page == "Data Analysis":
        st.header("Exploratory Data Analysis")
        
        st.subheader("Feature Distributions")
        plot_feature_distributions(df, feature_names)
        
        st.subheader("Correlation Between Features")
        plot_correlation_heatmap(df[feature_names])
        
        st.subheader("Soil Nutrient Analysis by Crop Type")
        plot_soil_nutrient_distribution(df)
        
        show_pairplot = st.checkbox("Show Pairplot (May take time to render)")
        if show_pairplot:
            st.subheader("Feature Pairplot")
            plot_pairplot(df, feature_names, "label")
    
    # Advanced Analysis Page
    elif page == "Advanced Analysis":
        st.header("Advanced Data Analysis")
        st.info("Advanced data analysis features are temporarily unavailable. Please check back later.")
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Performance Metrics")
            display_metrics(metrics)
            
            st.subheader("Feature Importance")
            importance_df = feature_importance(models["Random Forest"], feature_names)
            plot_feature_importance(importance_df)
            
        with col2:
            st.subheader("Model Selection")
            st.write("""
            Based on the performance metrics, the Random Forest model typically performs 
            best for this type of data due to its ability to capture non-linear relationships 
            and handle imbalanced classes.
            """)
            
            best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
            st.success(f"Best performing model: {best_model}")
            
            st.subheader("Best Model Settings")
            if best_model == "Random Forest":
                st.code("""
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    bootstrap=True,
                    random_state=42
                )
                """)
            elif best_model == "KNN":
                st.code("""
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights='uniform',
                    algorithm='auto',
                    leaf_size=30,
                    p=2,
                    metric='minkowski'
                )
                """)
            else:  # SVM
                st.code("""
                SVC(
                    C=1.0,
                    kernel='rbf',
                    degree=3,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
                """)
    
    # Advanced Modeling Page
    elif page == "Advanced Modeling":
        st.header("Advanced Model Evaluation")
        st.info("Advanced modeling features are temporarily unavailable. Please check back later.")
    
    # Prediction Tool Page
    elif page == "Prediction Tool":
        st.header("Crop Recommendation Prediction Tool")
        
        st.write("Enter your soil and environmental parameters to get a crop recommendation:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            N = st.number_input("Nitrogen (N) content in soil (kg/ha)", min_value=0, max_value=150, value=90)
            P = st.number_input("Phosphorus (P) content in soil (kg/ha)", min_value=0, max_value=150, value=40)
            K = st.number_input("Potassium (K) content in soil (kg/ha)", min_value=0, max_value=150, value=40)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        
        with col2:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0)
            ph = st.number_input("pH level", min_value=0.0, max_value=14.0, value=6.5)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=200.0)
        
        model_choice = st.selectbox(
            "Select prediction model",
            ["Random Forest", "KNN", "SVM"],
            index=0
        )
        
        if st.button("Predict Crop"):
            input_data = pd.DataFrame({
                'N': [N], 'P': [P], 'K': [K], 
                'temperature': [temperature], 'humidity': [humidity], 
                'ph': [ph], 'rainfall': [rainfall]
            })
            
            # Get prediction and probabilities
            predicted_crop, probabilities = predict_crop(
                input_data, 
                models[model_choice], 
                target_names
            )
            
            # Show prediction
            st.success(f"Recommended crop: **{predicted_crop}**")
            
            # Interpret prediction
            interpretation = interpret_prediction(
                input_data, 
                predicted_crop, 
                df
            )
            st.info(interpretation)
            
            # Show probabilities for top crops
            st.subheader("Crop Suitability Scores")
            proba_df = pd.DataFrame({
                'Crop': target_names,
                'Probability': probabilities[0]
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(proba_df.head(5), x='Crop', y='Probability', color='Probability',
                        color_continuous_scale='Viridis', title='Top 5 Suitable Crops')
            st.plotly_chart(fig)
            
            # Calculate feature importance for this prediction
            if model_choice == "Random Forest":
                st.subheader("Feature Importance for This Prediction")
                st.write("""
                This chart shows how each factor influenced the prediction.
                Higher values indicate more important features for this specific recommendation.
                """)
                
                # Get feature importance from the model
                model = models[model_choice]
                importances = model.feature_importances_
                
                # Create a DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot feature importance
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Feature Importance for Crop Prediction',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
    
    # Seasonal Analysis Page
    elif page == "Seasonal Analysis":
        st.header("Seasonal Crop Analysis")
        st.info("Seasonal analysis features are temporarily unavailable. Please check back later.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please make sure the data file is correctly uploaded and formatted.")
