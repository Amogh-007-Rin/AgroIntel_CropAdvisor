import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("attached_assets/Crop_recommendation.csv")
    return df

try:
    # Load the data
    df = load_data()
    
    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Analysis", "Model Performance", "Prediction Tool"]
    )
    
    # Home Page
    if page == "Home":
        st.header("Welcome to the Crop Recommendation System")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Number of samples: {df.shape[0]}")
            st.write(f"Number of features: {df.shape[1]-1}")  # Excluding the target column
            st.write(f"Number of crop types: {df['label'].nunique()}")
            
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Crop Distribution")
            crop_counts = df['label'].value_counts().reset_index()
            crop_counts.columns = ['Crop', 'Count']
            
            fig = px.bar(
                crop_counts, 
                x='Crop', 
                y='Count',
                color='Crop',
                title='Distribution of Crop Types in Dataset',
                labels={'Count': 'Number of Samples', 'Crop': 'Crop Type'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
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
                labels={"value": "Nutrient Value (kg/ha)", "variable": "Nutrient Type"}
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
                labels={"value": "Value", "variable": "Environmental Factor"}
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Between Features")
        
        # Calculate correlation matrix for features
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr = df[feature_names].corr()
        
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
        
        st.subheader("Soil Nutrient Analysis by Crop Type")
        
        # Create 3D scatter plot
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
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("Model Performance Analysis")
        
        # Train and evaluate models
        with st.spinner("Training and evaluating models..."):
            # Extract features and target
            X = df.drop('label', axis=1)
            y = df['label']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'SVM': SVC(kernel='rbf', probability=True, random_state=42)
            }
            
            # Fit models
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            metrics = {}
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Performance Metrics")
            
            # Create a DataFrame for comparison
            comparison = pd.DataFrame(index=metrics.keys())
            
            # Add metrics
            comparison['Accuracy'] = [m['accuracy'] for m in metrics.values()]
            comparison['Precision'] = [m['precision'] for m in metrics.values()]
            comparison['Recall'] = [m['recall'] for m in metrics.values()]
            comparison['F1 Score'] = [m['f1_score'] for m in metrics.values()]
            
            # Format as percentages
            for col in comparison.columns:
                comparison[col] = comparison[col].map(lambda x: f"{x:.2%}")
            
            # Display the table
            st.table(comparison)
            
            # Get feature importance from Random Forest
            if 'Random Forest' in models:
                st.subheader("Feature Importance")
                
                # Get importance
                importances = models['Random Forest'].feature_importances_
                
                # Create DataFrame
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                # Plot
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
            
        with col2:
            st.subheader("Model Selection")
            st.write("""
            Based on the performance metrics, the Random Forest model typically performs 
            best for this type of data due to its ability to capture non-linear relationships 
            and handle imbalanced classes.
            """)
            
            # Find best model
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
        
        # Model choice
        model_choice = st.selectbox(
            "Select prediction model",
            ["Random Forest", "KNN", "SVM"],
            index=0
        )
        
        if st.button("Predict Crop"):
            # Train the selected model
            with st.spinner("Training model and generating prediction..."):
                # Extract features and target
                X = df.drop('label', axis=1)
                y = df['label']
                
                # Get unique crop names
                crop_names = sorted(y.unique())
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                scaler.fit(X_train)
                
                # Select model
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_choice == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                else:  # SVM
                    model = SVC(kernel='rbf', probability=True, random_state=42)
                
                # Train model
                model.fit(scaler.transform(X_train), y_train)
                
                # Create input data
                input_data = pd.DataFrame({
                    'N': [N], 'P': [P], 'K': [K], 
                    'temperature': [temperature], 'humidity': [humidity], 
                    'ph': [ph], 'rainfall': [rainfall]
                })
                
                # Scale input
                input_scaled = scaler.transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)[0]
                probabilities = model.predict_proba(input_scaled)[0]
                
                # Show prediction
                st.success(f"Recommended crop: **{prediction}**")
                
                # Provide interpretation
                crop_data = df[df['label'] == prediction]
                avg_values = crop_data.mean()
                
                # Compare input values with typical values for this crop
                comparisons = []
                
                for feature in input_data.columns:
                    input_val = input_data[feature].iloc[0]
                    avg_val = avg_values[feature]
                    
                    # Calculate the percentage difference
                    if avg_val != 0:
                        diff_pct = (input_val - avg_val) / avg_val * 100
                        
                        # Determine if the value is within a typical range
                        if abs(diff_pct) <= 10:
                            status = "optimal"
                        elif abs(diff_pct) <= 25:
                            status = "acceptable" if diff_pct > 0 else "slightly low"
                        else:
                            status = "high" if diff_pct > 0 else "low"
                        
                        comparisons.append(f"{feature.capitalize()}: Your value ({input_val:.2f}) is {status} compared to typical values ({avg_val:.2f})")
                
                # Create interpretation text
                interpretation = f"""
                ### Crop Recommendation Analysis
                
                Based on your soil and environmental parameters, **{prediction}** is the recommended crop.
                
                #### Comparison with Typical Values:
                {" ".join([f"- {comp}" for comp in comparisons])}
                
                #### Key Requirements for {prediction}:
                - Soil fertility with {'high' if avg_values['N'] > 80 else 'moderate'} nitrogen content
                - {'High' if avg_values['P'] > 80 else 'Moderate'} phosphorus levels
                - {'High' if avg_values['K'] > 80 else 'Moderate'} potassium levels
                - Temperature range around {avg_values['temperature']:.1f}°C
                - Humidity levels around {avg_values['humidity']:.1f}%
                - Soil pH around {avg_values['ph']:.1f}
                - Annual rainfall around {avg_values['rainfall']:.1f} mm
                """
                
                st.info(interpretation)
                
                # Show probabilities for top crops
                st.subheader("Crop Suitability Scores")
                proba_df = pd.DataFrame({
                    'Crop': crop_names,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(proba_df.head(5), x='Crop', y='Probability', color='Probability',
                            color_continuous_scale='Viridis', title='Top 5 Suitable Crops')
                st.plotly_chart(fig)
                
                # If Random Forest, show feature importance
                if model_choice == "Random Forest":
                    st.subheader("Feature Importance for This Prediction")
                    st.write("""
                    This chart shows how each factor influenced the prediction.
                    Higher values indicate more important features for this specific recommendation.
                    """)
                    
                    # Get feature importance
                    importances = model.feature_importances_
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot
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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please make sure the data file is correctly uploaded and formatted.")