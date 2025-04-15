import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.inspection import permutation_importance
import joblib
import base64
from datetime import datetime
import time

# Set page configuration with dark theme
st.set_page_config(
    page_title="AGROINTEL: Advanced Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sci-fi theme
st.markdown("""
<style>
    /* Main background with gradient */
    .main {
        background: linear-gradient(to bottom right, #000428, #004e92);
        color: #E0E0E0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(to bottom, #000428, #004e92);
    }
    
    /* Headers with sci-fi font and glow */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(0, 195, 255, 0.7);
    }
    
    /* Gradient text for titles */
    .title-gradient {
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #0072ff, #00c6ff);
        color: white;
        border: none;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 123, 255, 0.8);
        transform: translateY(-2px);
    }
    
    /* Card-like containers */
    .css-1r6slb0 {
        background: rgba(1, 10, 50, 0.7);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 123, 255, 0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(5, 15, 40, 0.85);
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        color: #00ff9d;
    }
    
    /* Success message styling */
    .element-container div[data-testid="stImage"] {
        border: 2px solid rgba(0, 255, 200, 0.3);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 255, 200, 0.2);
    }
    
    /* Success message styling */
    .element-container div[data-testid="stAlert"] {
        background: rgba(13, 25, 47, 0.7);
        border-radius: 10px;
        border-left: 4px solid #00f2fe;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.3);
    }
    
    /* Metric value styling */
    .css-1v774eo {
        font-family: 'Orbitron', sans-serif;
        color: #00f2fe;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
    
    /* Chart container styling */
    div[data-testid="stDecoration"], div[data-testid="stDecoration"] div {
        background: rgba(0, 20, 50, 0.6) !important;
        border-radius: 10px;
        border: 1px solid rgba(0, 123, 255, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background-color: rgba(0, 20, 60, 0.5);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 40, 100, 0.7);
        border-radius: 8px;
        color: #00f2fe;
        padding: 5px 15px;
        margin: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 105, 225, 0.7);
    }
    
    /* Loader animation */
    .stSpinner {
        border-color: #00f2fe !important;
        border-top-color: transparent !important;
    }
</style>

<!-- Google Fonts for Orbitron (sci-fi font) -->
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Page header with sci-fi style
st.markdown('<h1 class="title-gradient">AGROINTEL: Advanced Crop Recommendation System 🌱</h1>', unsafe_allow_html=True)

# Animated introduction text
def animate_text():
    text = "Analyzing soil composition and environmental parameters to determine optimal crop selection..."
    placeholder = st.empty()
    for i in range(len(text) + 1):
        placeholder.markdown(f"<p style='font-family: monospace; color: #00f2fe;'>{text[:i]}_</p>", unsafe_allow_html=True)
        time.sleep(0.02)
    placeholder.markdown(f"<p style='font-family: monospace; color: #00f2fe;'>{text}</p>", unsafe_allow_html=True)
    
    # System status message
    st.markdown("""
    <div style="background-color: rgba(0,20,40,0.7); padding: 10px; border-radius: 5px; 
    border-left: 3px solid #00f2fe; margin-top: 20px; font-family: 'Courier New', monospace;">
    <span style="color: #00f2fe;">●</span> <span style="color: #e0e0e0;">SYSTEM ACTIVE</span> | 
    <span style="color: #00f2fe;">●</span> <span style="color: #e0e0e0;">DATA NODES CONNECTED</span> | 
    <span style="color: #00f2fe;">●</span> <span style="color: #e0e0e0;">PREDICTION ENGINE READY</span>
    </div>
    """, unsafe_allow_html=True)

# Run animation on initial load
animate_text()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("attached_assets/Crop_recommendation.csv")
    return df

try:
    # Load the data
    df = load_data()
    
    # Navigation sidebar with sci-fi styling
    st.sidebar.markdown('<h2 class="title-gradient">COMMAND CENTER</h2>', unsafe_allow_html=True)
    
    # Add a system status indicator
    current_time = datetime.now().strftime("%H:%M:%S")
    st.sidebar.markdown(f"""
    <div style="background-color: rgba(0,20,40,0.7); padding: 10px; border-radius: 5px; margin-bottom: 20px; 
    font-family: 'Courier New', monospace; font-size: 0.8em;">
    <span style="color: #00f2fe;">SYSTEM TIME:</span> <span style="color: #e0e0e0;">{current_time}</span><br>
    <span style="color: #00f2fe;">STATUS:</span> <span style="color: #00ff9d;">OPERATIONAL</span><br>
    <span style="color: #00f2fe;">DATA INTEGRITY:</span> <span style="color: #00ff9d;">100%</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a divider
    st.sidebar.markdown('<hr style="margin: 15px 0; border: 0; border-top: 1px solid rgba(0, 242, 254, 0.3);">', unsafe_allow_html=True)
    
    # Navigation menu with icons and styled as a sci-fi console
    st.sidebar.markdown('<div style="color: #00f2fe; margin-bottom: 10px;">SELECT MODULE:</div>', unsafe_allow_html=True)
    
    page = st.sidebar.radio(
        "",
        ["Home", 
         "Data Analysis", 
         "Model Performance", 
         "Prediction Tool", 
         "Advanced Analysis", 
         "Feature Engineering",
         "Clustering Analysis"]
    )
    
    # Add visual indicators
    st.sidebar.markdown(f"""
    <div style="margin-top: 30px; background-color: rgba(0,25,50,0.7); padding: 10px; border-radius: 5px;">
    <div style="font-family: 'Courier New', monospace; color: #00f2fe; font-size: 0.8em; margin-bottom: 5px;">SELECTED MODULE:</div>
    <div style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-weight: bold;">{page.upper()}</div>
    </div>
    """, unsafe_allow_html=True)

    # Add a futuristic help section
    with st.sidebar.expander("SYSTEM DIAGNOSTICS"):
        st.markdown("""
        <div style="font-family: 'Courier New', monospace; color: #e0e0e0; font-size: 0.9em;">
        <span style="color: #00f2fe;">></span> AI Models: <span style="color: #00ff9d;">OPTIMAL</span><br>
        <span style="color: #00f2fe;">></span> Data Pipeline: <span style="color: #00ff9d;">ACTIVE</span><br>
        <span style="color: #00f2fe;">></span> Neural Net: <span style="color: #00ff9d;">CALIBRATED</span><br>
        <span style="color: #00f2fe;">></span> Memory Usage: <span style="color: #00ff9d;">42%</span>
        </div>
        """, unsafe_allow_html=True)
    
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
                # Improve y-axis spacing for better readability
                fig.update_layout(
                    height=400,
                    margin=dict(l=150, r=40, t=80, b=40),
                    yaxis=dict(
                        tickfont=dict(size=12),
                        tickmode='array',
                        tickvals=list(range(len(importance_df))),
                        ticktext=importance_df['Feature']
                    )
                )
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
                    # Improve y-axis spacing for better readability
                    fig.update_layout(
                        height=400,
                        margin=dict(l=150, r=40, t=80, b=40),
                        yaxis=dict(
                            tickfont=dict(size=12),
                            tickmode='array',
                            tickvals=list(range(len(importance_df))),
                            ticktext=importance_df['Feature']
                        )
                    )
                    st.plotly_chart(fig)

# Add Feature Engineering page
    elif page == "Feature Engineering":
        st.markdown('<h1 class="title-gradient">Advanced Feature Engineering</h1>', unsafe_allow_html=True)
        
        # Create a pulsing animation effect for "processing" status
        st.markdown("""
        <style>
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        .pulse {
            animation: pulse 1.5s infinite ease-in-out;
        }
        </style>
        <div style="background-color: rgba(0,40,80,0.7); padding: 15px; border-radius: 10px; 
        margin-bottom: 20px; border: 1px solid rgba(0, 242, 254, 0.3);">
        <div style="display: flex; align-items: center;">
        <span class="pulse" style="color: #00f2fe; margin-right: 10px;">●</span>
        <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
        QUANTUM FEATURE PROCESSOR ACTIVE
        </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different feature engineering methods
        tabs = st.tabs([
            "Polynomial Features", 
            "Feature Ratios", 
            "PCA Transformation",
            "Feature Selection"
        ])
        
        # Extract features
        X = df.drop('label', axis=1)
        y = df['label']
        feature_names = X.columns.tolist()
        
        # Tab 1: Polynomial Features
        with tabs[0]:
            st.subheader("Polynomial Feature Generation")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Polynomial features create new features by calculating products and powers of the original features,
            capturing non-linear relationships and interactions between variables.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            degree = st.slider("Polynomial Degree", min_value=2, max_value=3, value=2)
            poly_sample_size = st.slider("Sample Size for Visualization", min_value=100, max_value=1000, value=500)
            
            # Process
            with st.spinner("Generating polynomial features..."):
                # Create polynomial features
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                X_poly = poly.fit_transform(X)
                
                # Get feature names
                if hasattr(poly, 'get_feature_names_out'):
                    poly_feature_names = poly.get_feature_names_out(X.columns)
                else:
                    # For older scikit-learn versions
                    poly_feature_names = poly.get_feature_names(X.columns)
                
                # Create DataFrame with polynomial features
                X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
                
                # Display stats
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                    <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 5px; flex: 1; margin-right: 10px;">
                        <h4 style="color: #00f2fe; margin: 0;">Original Features</h4>
                        <p style="font-family: 'Orbitron', sans-serif; font-size: 24px; color: #00ff9d; margin: 5px 0;">
                        {X.shape[1]}
                        </p>
                    </div>
                    <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 5px; flex: 1; margin-left: 10px;">
                        <h4 style="color: #00f2fe; margin: 0;">Polynomial Features</h4>
                        <p style="font-family: 'Orbitron', sans-serif; font-size: 24px; color: #00ff9d; margin: 5px 0;">
                        {X_poly.shape[1]}
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sample of polynomial features
                st.subheader("Sample of Generated Features")
                st.dataframe(X_poly_df.head(10))
                
                # Show correlation heatmap for selected polynomial features
                st.subheader("Correlation Between Original and Polynomial Features")
                
                # Select subset of polynomial features to avoid overwhelming visualization
                selected_poly_features = list(X.columns) + list(X_poly_df.columns[X.shape[1]:X.shape[1]+5])
                
                # Calculate correlation
                corr_poly = X_poly_df[selected_poly_features].corr()
                
                # Create heatmap
                fig = px.imshow(
                    corr_poly,
                    color_continuous_scale='inferno',
                    title="Feature Correlation Heatmap"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance for polynomial features
                st.subheader("Polynomial Feature Importance")
                
                # Sample data for faster processing
                indices = np.random.choice(len(X), min(poly_sample_size, len(X)), replace=False)
                X_poly_sample = X_poly[indices]
                y_sample = y.iloc[indices]
                
                # Train a quick Random Forest to get feature importance
                rf_poly = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                rf_poly.fit(X_poly_sample, y_sample)
                
                # Get importance
                importances_poly = rf_poly.feature_importances_
                
                # Create DataFrame for top 15 features
                importance_poly_df = pd.DataFrame({
                    'Feature': poly_feature_names,
                    'Importance': importances_poly
                }).sort_values('Importance', ascending=False).head(15)
                
                # Plot
                fig = px.bar(
                    importance_poly_df, 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 15 Polynomial Feature Importance',
                    color='Importance',
                    color_continuous_scale='plasma'
                )
                # Improve y-axis spacing for better readability
                fig.update_layout(
                    height=500,
                    margin=dict(l=180, r=40, t=80, b=40),
                    yaxis=dict(
                        tickfont=dict(size=12),
                        tickmode='array',
                        tickvals=list(range(len(importance_poly_df))),
                        ticktext=importance_poly_df['Feature']
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Feature Ratios
        with tabs[1]:
            st.subheader("Feature Ratio Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Ratio features capture relationships between pairs of features, which can be particularly useful
            when the relative proportions between features are more important than their absolute values.
            </div>
            """, unsafe_allow_html=True)
            
            # Feature selection for ratios
            selected_features = st.multiselect(
                "Select features to create ratios from:",
                options=feature_names,
                default=["N", "P", "K", "rainfall"]
            )
            
            if len(selected_features) >= 2:
                with st.spinner("Calculating feature ratios..."):
                    # Create ratio features
                    ratio_features = []
                    X_ratio = X.copy()
                    
                    for i, feat1 in enumerate(selected_features):
                        for feat2 in selected_features[i+1:]:
                            # Create ratio names
                            ratio_name1 = f"{feat1}_to_{feat2}"
                            ratio_name2 = f"{feat2}_to_{feat1}"
                            
                            # Calculate ratios (handling division by zero)
                            X_ratio[ratio_name1] = X[feat1] / (X[feat2] + 1e-6)
                            X_ratio[ratio_name2] = X[feat2] / (X[feat1] + 1e-6)
                            
                            ratio_features.extend([ratio_name1, ratio_name2])
                    
                    # Display stats
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                        <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 5px; flex: 1; margin-right: 10px;">
                            <h4 style="color: #00f2fe; margin: 0;">Original Features</h4>
                            <p style="font-family: 'Orbitron', sans-serif; font-size: 24px; color: #00ff9d; margin: 5px 0;">
                            {X.shape[1]}
                            </p>
                        </div>
                        <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 5px; flex: 1; margin-left: 10px;">
                            <h4 style="color: #00f2fe; margin: 0;">Ratio Features Created</h4>
                            <p style="font-family: 'Orbitron', sans-serif; font-size: 24px; color: #00ff9d; margin: 5px 0;">
                            {len(ratio_features)}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sample of ratio features
                    st.subheader("Sample of Ratio Features")
                    st.dataframe(X_ratio[ratio_features].head(10))
                    
                    # Analyze correlation between ratio features and target
                    st.subheader("Correlation Between Ratio Features and Crop Types")
                    
                    # One-hot encode the target
                    y_dummies = pd.get_dummies(y, prefix='crop')
                    
                    # Combine ratio features and target dummies
                    ratio_with_target = pd.concat([X_ratio[ratio_features], y_dummies], axis=1)
                    
                    # Calculate correlations
                    crop_cols = [col for col in ratio_with_target.columns if col.startswith('crop_')]
                    ratio_crop_corr = ratio_with_target.corr().loc[ratio_features, crop_cols]
                    
                    # Create heatmap
                    fig = px.imshow(
                        ratio_crop_corr,
                        color_continuous_scale='RdBu_r',
                        title="Correlation Between Ratio Features and Crop Types"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualize distributions of top ratio features
                    st.subheader("Distribution of Top Ratio Features by Crop")
                    
                    # Find top ratio features based on correlation with any crop
                    abs_corr = np.abs(ratio_crop_corr.values)
                    top_ratios_idx = np.unravel_index(np.argsort(abs_corr, axis=None)[-5:], abs_corr.shape)
                    top_ratio_features = [ratio_features[i] for i in top_ratios_idx[0]]
                    
                    # Plot distributions
                    fig = px.box(
                        pd.concat([X_ratio[top_ratio_features].reset_index(drop=True), 
                                 y.reset_index(drop=True)], axis=1),
                        x="label",
                        y=top_ratio_features,
                        color="label",
                        facet_col="variable",
                        facet_col_wrap=2,
                        title="Distribution of Top Ratio Features by Crop Type"
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least 2 features to create ratios.")
        
        # Tab 3: PCA Transformation
        with tabs[2]:
            st.subheader("Principal Component Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            PCA reduces dimensionality by transforming the original features into a new set of uncorrelated variables
            (principal components) that capture the maximum amount of variance in the data.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            n_components = st.slider("Number of Principal Components", min_value=2, max_value=min(7, X.shape[1]), value=3)
            
            with st.spinner("Performing PCA analysis..."):
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Perform PCA
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # Create DataFrame with PCA results
                pca_df = pd.DataFrame(
                    X_pca, 
                    columns=[f"PC{i+1}" for i in range(n_components)]
                )
                
                # Add the target for coloring
                pca_df['Crop'] = y.values
                
                # Display variance explained
                explained_variance = pca.explained_variance_ratio_ * 100
                cumulative_variance = np.cumsum(explained_variance)
                
                # Create variance plot
                variance_df = pd.DataFrame({
                    'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                    'Explained Variance (%)': explained_variance,
                    'Cumulative Variance (%)': cumulative_variance
                })
                
                # Create animated dashboard style display
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Component explained variance
                    fig = px.bar(
                        variance_df,
                        x='Principal Component',
                        y='Explained Variance (%)',
                        text=variance_df['Explained Variance (%)'].round(2).astype(str) + '%',
                        title="Variance Explained by Principal Components",
                        color='Explained Variance (%)',
                        color_continuous_scale='plasma'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Cumulative variance
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=variance_df['Principal Component'],
                        y=variance_df['Cumulative Variance (%)'],
                        mode='lines+markers+text',
                        name='Cumulative Variance',
                        line=dict(color='#00f2fe', width=3),
                        marker=dict(size=10, color='#00ff9d'),
                        text=variance_df['Cumulative Variance (%)'].round(2).astype(str) + '%',
                        textposition="top center"
                    ))
                    fig.update_layout(
                        title="Cumulative Variance Explained",
                        height=400,
                        plot_bgcolor='rgba(0,10,30,0.7)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e0e0e0')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature loadings/coefficients
                st.subheader("Feature Contributions to Principal Components")
                
                # Get feature loadings
                loadings = pca.components_
                loadings_df = pd.DataFrame(
                    loadings.T,
                    columns=[f'PC{i+1}' for i in range(n_components)],
                    index=feature_names
                )
                
                # Create heatmap of loadings
                fig = px.imshow(
                    loadings_df,
                    color_continuous_scale='RdBu_r',
                    title="Feature Loadings (Contributions to Principal Components)",
                    labels=dict(x="Principal Component", y="Original Feature", color="Loading Value")
                )
                fig.update_layout(height=550)
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D PCA plot if we have at least 3 components
                if n_components >= 3:
                    st.subheader("3D PCA Visualization")
                    
                    fig = px.scatter_3d(
                        pca_df,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Crop',
                        title="3D PCA Plot",
                        opacity=0.8
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 2D PCA plots for combinations of components
                st.subheader("2D PCA Visualizations")
                
                # For larger numbers of components, show only the most interesting combinations
                if n_components > 3:
                    component_pairs = [(0, 1), (0, 2), (1, 2)]
                else:
                    component_pairs = [(i, j) for i in range(n_components) for j in range(i+1, n_components)]
                
                # Create subplot grid
                subplot_rows = (len(component_pairs) + 1) // 2
                fig = make_subplots(rows=subplot_rows, cols=2, 
                                   subplot_titles=[f"PC{pair[0]+1} vs PC{pair[1]+1}" for pair in component_pairs])
                
                # Add scatter plots for each component pair
                for idx, (i, j) in enumerate(component_pairs):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    
                    for crop in pca_df['Crop'].unique():
                        crop_data = pca_df[pca_df['Crop'] == crop]
                        fig.add_trace(
                            go.Scatter(
                                x=crop_data[f'PC{i+1}'],
                                y=crop_data[f'PC{j+1}'],
                                mode='markers',
                                name=crop,
                                legendgroup=crop,
                                showlegend=idx==0
                            ),
                            row=row, col=col
                        )
                
                fig.update_layout(height=300*subplot_rows, title_text="2D PCA Projections")
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 4: Feature Selection
        with tabs[3]:
            st.subheader("Automated Feature Selection")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Feature selection identifies the most relevant features for predicting the target variable,
            reducing dimensionality and potentially improving model performance.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            selection_method = st.selectbox(
                "Feature Selection Method",
                ["ANOVA F-value", "Random Forest Importance", "Permutation Importance"]
            )
            
            # Number of features to select
            k_features = st.slider("Number of Features to Select", min_value=2, max_value=X.shape[1], value=5)
            
            with st.spinner(f"Selecting top {k_features} features using {selection_method}..."):
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Initialize variables for storing results
                importance_df = None
                X_selected = None
                selected_feature_names = None
                
                # Apply the selected method
                if selection_method == "ANOVA F-value":
                    # Use SelectKBest with f_classif (ANOVA F-value)
                    selector = SelectKBest(f_classif, k=k_features)
                    X_selected = selector.fit_transform(X_scaled, y)
                    
                    # Get feature scores
                    scores = selector.scores_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Score': scores
                    }).sort_values('Score', ascending=False)
                    
                    # Get selected feature names
                    selected_indices = selector.get_support(indices=True)
                    selected_feature_names = [feature_names[i] for i in selected_indices]
                
                elif selection_method == "Random Forest Importance":
                    # Use Random Forest for feature importance
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_scaled, y)
                    
                    # Get feature importance
                    importances = rf.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Score': importances
                    }).sort_values('Score', ascending=False)
                    
                    # Select top k features
                    selected_feature_names = importance_df['Feature'].head(k_features).tolist()
                    X_selected = X[selected_feature_names].values
                
                elif selection_method == "Permutation Importance":
                    # Need a base model for permutation importance
                    base_model = RandomForestClassifier(n_estimators=50, random_state=42)
                    base_model.fit(X_scaled, y)
                    
                    # Calculate permutation importance
                    perm_importance = permutation_importance(
                        base_model, X_scaled, y, n_repeats=10, random_state=42
                    )
                    
                    # Get mean importance
                    importances = perm_importance.importances_mean
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Score': importances
                    }).sort_values('Score', ascending=False)
                    
                    # Select top k features
                    selected_feature_names = importance_df['Feature'].head(k_features).tolist()
                    X_selected = X[selected_feature_names].values
                
                # Display feature importance
                st.subheader("Feature Importance Scores")
                
                # Create bar chart of feature importance
                fig = px.bar(
                    importance_df.head(min(10, X.shape[1])),  # Show top 10 or all if less
                    x='Score',
                    y='Feature',
                    orientation='h',
                    color='Score',
                    color_continuous_scale='viridis',
                    title=f"Feature Importance Using {selection_method}"
                )
                # Improve y-axis spacing for better readability
                fig.update_layout(
                    height=500,
                    margin=dict(l=150, r=40, t=80, b=40),
                    yaxis=dict(
                        tickfont=dict(size=12),
                        tickmode='array',
                        tickvals=list(range(len(importance_df.head(min(10, X.shape[1]))))),
                        ticktext=importance_df.head(min(10, X.shape[1]))['Feature']
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Compare model performance with selected vs all features
                st.subheader("Model Performance Comparison")
                
                # Train models with all features vs selected features
                metrics_comparison = {}
                
                # Create train/test splits
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Get indices of selected features
                selected_indices = [feature_names.index(feat) for feat in selected_feature_names]
                
                # Create selected feature train/test sets
                X_train_selected = X_train[:, selected_indices]
                X_test_selected = X_test[:, selected_indices]
                
                # Train and evaluate with all features
                rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_all.fit(X_train, y_train)
                y_pred_all = rf_all.predict(X_test)
                
                metrics_comparison['All Features'] = {
                    'accuracy': accuracy_score(y_test, y_pred_all),
                    'precision': precision_score(y_test, y_pred_all, average='weighted'),
                    'recall': recall_score(y_test, y_pred_all, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred_all, average='weighted'),
                    'num_features': X.shape[1]
                }
                
                # Train and evaluate with selected features
                rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_selected.fit(X_train_selected, y_train)
                y_pred_selected = rf_selected.predict(X_test_selected)
                
                metrics_comparison['Selected Features'] = {
                    'accuracy': accuracy_score(y_test, y_pred_selected),
                    'precision': precision_score(y_test, y_pred_selected, average='weighted'),
                    'recall': recall_score(y_test, y_pred_selected, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred_selected, average='weighted'),
                    'num_features': len(selected_feature_names)
                }
                
                # Create metrics table
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Number of Features'],
                    'All Features': [
                        f"{metrics_comparison['All Features']['accuracy']:.2%}",
                        f"{metrics_comparison['All Features']['precision']:.2%}",
                        f"{metrics_comparison['All Features']['recall']:.2%}",
                        f"{metrics_comparison['All Features']['f1_score']:.2%}",
                        metrics_comparison['All Features']['num_features']
                    ],
                    'Selected Features': [
                        f"{metrics_comparison['Selected Features']['accuracy']:.2%}",
                        f"{metrics_comparison['Selected Features']['precision']:.2%}",
                        f"{metrics_comparison['Selected Features']['recall']:.2%}",
                        f"{metrics_comparison['Selected Features']['f1_score']:.2%}",
                        metrics_comparison['Selected Features']['num_features']
                    ]
                })
                
                # Display styled metrics table
                st.table(metrics_df)
                
                # Visualize distribution of selected features by crop
                st.subheader("Distribution of Selected Features by Crop")
                
                # Create plot
                fig = px.box(
                    pd.concat([X[selected_feature_names].reset_index(drop=True), 
                             y.reset_index(drop=True)], axis=1),
                    x="label",
                    y=selected_feature_names,
                    color="label",
                    facet_col="variable",
                    facet_col_wrap=2,
                    title="Distribution of Selected Features by Crop Type"
                )
                fig.update_layout(height=max(300 * ((len(selected_feature_names) + 1) // 2), 500))
                st.plotly_chart(fig, use_container_width=True)

    # Add Clustering Analysis page
    elif page == "Clustering Analysis":
        st.markdown('<h1 class="title-gradient">Advanced Clustering Analysis</h1>', unsafe_allow_html=True)
        
        # Create a futuristic loading animation
        st.markdown("""
        <style>
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            width: 24px;
            height: 24px;
            border: 4px solid rgba(0, 242, 254, 0.3);
            border-top: 4px solid #00f2fe;
            border-radius: 50%;
            animation: rotate 1s linear infinite;
            display: inline-block;
            vertical-align: middle;
            margin-right: 10px;
        }
        </style>
        <div style="background-color: rgba(0,40,80,0.7); padding: 15px; border-radius: 10px; 
        margin-bottom: 20px; border: 1px solid rgba(0, 242, 254, 0.3);">
        <div style="display: flex; align-items: center;">
        <div class="spinner"></div>
        <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
        CLUSTER ANALYSIS MODULE INITIALIZED
        </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different clustering methods
        tabs = st.tabs([
            "K-Means Clustering", 
            "Optimal Cluster Analysis", 
            "Cluster Profiling"
        ])
        
        # Extract and scale features
        X = df.drop('label', axis=1)
        y = df['label']
        feature_names = X.columns.tolist()
        
        # Scale data for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Tab 1: K-Means Clustering
        with tabs[0]:
            st.subheader("K-Means Clustering Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            K-Means partitions data into k clusters, where each observation belongs to the cluster with 
            the nearest mean, serving as a prototype of the cluster.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)
            
            with st.spinner("Performing K-Means clustering..."):
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Create a dataframe with the results
                cluster_df = pd.DataFrame({
                    'Cluster': cluster_labels,
                    'Crop': y.values
                })
                
                # Get cluster centers
                centers = kmeans.cluster_centers_
                
                # Reduce to 3D for visualization
                pca = PCA(n_components=3)
                X_pca = pca.fit_transform(X_scaled)
                centers_pca = pca.transform(centers)
                
                # Create dataframe for PCA results
                pca_df = pd.DataFrame(
                    X_pca, 
                    columns=['PC1', 'PC2', 'PC3']
                )
                pca_df['Cluster'] = cluster_labels
                pca_df['Crop'] = y.values
                
                # Display cluster statistics
                st.subheader("Cluster Composition")
                
                # Count crops in each cluster
                cluster_crop_counts = pd.crosstab(
                    cluster_df['Cluster'], 
                    cluster_df['Crop'],
                    normalize='index'
                ) * 100
                
                # Create heatmap of crop distribution in clusters
                fig = px.imshow(
                    cluster_crop_counts,
                    color_continuous_scale='viridis',
                    title="Crop Distribution Across Clusters (%)",
                    labels=dict(x="Crop Type", y="Cluster", color="Percentage")
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 3D visualization of clusters
                st.subheader("3D Cluster Visualization")
                
                fig = px.scatter_3d(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    color='Cluster',
                    symbol='Crop',
                    opacity=0.7,
                    title="3D Visualization of Clusters (PCA)",
                    labels={'PC1': 'Principal Component 1', 
                           'PC2': 'Principal Component 2', 
                           'PC3': 'Principal Component 3'}
                )
                
                # Add cluster centers
                for i, center in enumerate(centers_pca):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[center[0]],
                            y=[center[1]],
                            z=[center[2]],
                            mode='markers',
                            marker=dict(size=15, symbol='diamond', color=i),
                            name=f'Cluster {i} Center'
                        )
                    )
                
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster profiles
                st.subheader("Cluster Profiles")
                
                # Add cluster to original data
                profiling_df = X.copy()
                profiling_df['Cluster'] = cluster_labels
                
                # Calculate mean value of each feature for each cluster
                cluster_profiles = profiling_df.groupby('Cluster').mean()
                
                # Create radar chart for cluster profiles
                fig = go.Figure()
                
                for i in range(n_clusters):
                    fig.add_trace(go.Scatterpolar(
                        r=cluster_profiles.iloc[i].values,
                        theta=feature_names,
                        fill='toself',
                        name=f'Cluster {i}'
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, cluster_profiles.values.max()]
                        )
                    ),
                    title="Feature Profiles by Cluster",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Optimal Cluster Analysis
        with tabs[1]:
            st.subheader("Optimal Number of Clusters Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Determining the optimal number of clusters is a critical step in cluster analysis. 
            This analysis uses multiple methods to help identify the best value for K.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            max_k = st.slider("Maximum Number of Clusters to Test", min_value=2, max_value=15, value=10)
            
            with st.spinner("Calculating metrics for different cluster counts..."):
                # Calculate metrics for different k values
                inertia_values = []
                silhouette_scores = []
                
                for k in range(2, max_k + 1):
                    # Run K-Means
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_scaled)
                    
                    # Calculate metrics
                    inertia_values.append(kmeans.inertia_)
                    if len(np.unique(labels)) > 1:  # Silhouette requires at least 2 clusters
                        from sklearn.metrics import silhouette_score
                        silhouette_scores.append(silhouette_score(X_scaled, labels))
                    else:
                        silhouette_scores.append(0)
                
                # Create dataframe for visualization
                k_values = list(range(2, max_k + 1))
                metrics_df = pd.DataFrame({
                    'K': k_values,
                    'Inertia': inertia_values,
                    'Silhouette Score': silhouette_scores
                })
                
                # Display elbow method
                st.subheader("Elbow Method")
                st.markdown("""
                <div style="font-family: 'Courier New', monospace; color: #e0e0e0; font-size: 0.9em;">
                The elbow method looks for the point where adding more clusters provides diminishing returns.
                Look for the "elbow" point where the curve starts to flatten.
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.line(
                    metrics_df, 
                    x='K', 
                    y='Inertia', 
                    markers=True,
                    title="Elbow Method for Optimal K",
                    labels={'K': 'Number of Clusters', 'Inertia': 'Sum of Squared Distances'}
                )
                fig.update_traces(marker=dict(size=10, color='#00f2fe'))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display silhouette scores
                st.subheader("Silhouette Analysis")
                st.markdown("""
                <div style="font-family: 'Courier New', monospace; color: #e0e0e0; font-size: 0.9em;">
                The silhouette score measures how similar an object is to its own cluster compared to other clusters.
                Higher values indicate better defined clusters. Look for the peak value.
                </div>
                """, unsafe_allow_html=True)
                
                fig = px.line(
                    metrics_df, 
                    x='K', 
                    y='Silhouette Score', 
                    markers=True,
                    title="Silhouette Scores for Different K Values",
                    labels={'K': 'Number of Clusters', 'Silhouette Score': 'Silhouette Score'}
                )
                fig.update_traces(marker=dict(size=10, color='#00ff9d'))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Combined visualization
                st.subheader("Combined Analysis")
                
                # Calculate normalized metrics for comparison
                metrics_df['Normalized Inertia'] = 1 - (metrics_df['Inertia'] - metrics_df['Inertia'].min()) / (metrics_df['Inertia'].max() - metrics_df['Inertia'].min())
                
                # Create combined plot
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add inertia line
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df['K'],
                        y=metrics_df['Normalized Inertia'],
                        mode='lines+markers',
                        name='Normalized Inertia',
                        line=dict(color='#00f2fe', width=3),
                        marker=dict(size=8, color='#00f2fe')
                    ),
                    secondary_y=False
                )
                
                # Add silhouette score line
                fig.add_trace(
                    go.Scatter(
                        x=metrics_df['K'],
                        y=metrics_df['Silhouette Score'],
                        mode='lines+markers',
                        name='Silhouette Score',
                        line=dict(color='#00ff9d', width=3),
                        marker=dict(size=8, color='#00ff9d')
                    ),
                    secondary_y=True
                )
                
                # Layout updates
                fig.update_layout(
                    title="Combined Cluster Evaluation Metrics",
                    height=600,
                    hovermode="x unified"
                )
                
                # Update axes
                fig.update_xaxes(title_text="Number of Clusters (K)")
                fig.update_yaxes(title_text="Normalized Inertia (higher is better)", secondary_y=False)
                fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Optimal K recommendation
                # Find best silhouette score
                best_k_silhouette = metrics_df.loc[metrics_df['Silhouette Score'].idxmax(), 'K']
                
                # Find elbow point (simplified approach)
                from scipy.signal import argrelextrema
                inertia_diff = np.diff(inertia_values, 2)  # Second derivative
                elbow_idx = np.abs(inertia_diff).argmax() + 2  # +2 because of double diff and 0-indexing
                best_k_elbow = k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[0]
                
                st.markdown(f"""
                <div style="background-color: rgba(0,30,60,0.7); padding: 20px; border-radius: 10px; 
                margin-top: 20px; border: 1px solid rgba(0, 242, 254, 0.3);">
                <h3 style="color: #00f2fe; margin-top: 0;">Optimal Cluster Recommendation</h3>
                <p style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                Based on silhouette analysis, the optimal number of clusters is: 
                <span style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.2em;">{best_k_silhouette}</span>
                </p>
                <p style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                Based on the elbow method, the optimal number of clusters is: 
                <span style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.2em;">{best_k_elbow}</span>
                </p>
                </div>
                """, unsafe_allow_html=True)
        
        # Tab 3: Cluster Profiling
        with tabs[2]:
            st.subheader("Advanced Cluster Profiling")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Cluster profiling helps interpret what each cluster represents by analyzing the 
            characteristics of observations within each cluster.
            </div>
            """, unsafe_allow_html=True)
            
            # Controls
            profile_n_clusters = st.slider("Number of Clusters for Profiling", min_value=2, max_value=8, value=4)
            
            with st.spinner("Analyzing cluster profiles..."):
                # Perform K-Means clustering with specified number of clusters
                kmeans = KMeans(n_clusters=profile_n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels to original data
                profile_df = X.copy()
                profile_df['Cluster'] = cluster_labels
                profile_df['Crop'] = y.values
                
                # Calculate statistics for each cluster
                cluster_stats = profile_df.groupby('Cluster').agg(['mean', 'std'])
                
                # Show crop distribution for each cluster
                st.subheader("Crop Distribution by Cluster")
                
                # Calculate and plot crop distribution
                crop_cluster_counts = profile_df.groupby(['Cluster', 'Crop']).size().unstack().fillna(0)
                crop_cluster_pct = crop_cluster_counts.div(crop_cluster_counts.sum(axis=1), axis=0) * 100
                
                # Create stacked bar chart
                fig = px.bar(
                    crop_cluster_pct.reset_index().melt(id_vars='Cluster', var_name='Crop', value_name='Percentage'),
                    x='Cluster',
                    y='Percentage',
                    color='Crop',
                    title="Crop Composition of Each Cluster (%)",
                    labels={'Percentage': 'Percentage of Cluster (%)'}
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature comparison across clusters
                st.subheader("Feature Comparison Across Clusters")
                
                # Feature selection for detailed view
                selected_features_for_profile = st.multiselect(
                    "Select features to compare across clusters:",
                    options=feature_names,
                    default=feature_names[:4]  # Default to first 4 features
                )
                
                if selected_features_for_profile:
                    # Create parallel coordinates plot
                    fig = px.parallel_coordinates(
                        profile_df,
                        dimensions=selected_features_for_profile,
                        color='Cluster',
                        title="Parallel Coordinates Plot of Cluster Features"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create box plots for selected features
                    fig = px.box(
                        profile_df.melt(
                            id_vars=['Cluster', 'Crop'],
                            value_vars=selected_features_for_profile,
                            var_name='Feature',
                            value_name='Value'
                        ),
                        x='Cluster',
                        y='Value',
                        color='Cluster',
                        facet_col='Feature',
                        facet_col_wrap=2,
                        title="Distribution of Features by Cluster"
                    )
                    fig.update_layout(height=max(300 * ((len(selected_features_for_profile) + 1) // 2), 500))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics summary
                st.subheader("Cluster Characteristics Summary")
                
                # Calculate mean values for each feature in each cluster
                cluster_means = profile_df.groupby('Cluster')[feature_names].mean()
                
                # Calculate overall dataset means
                overall_means = X[feature_names].mean()
                
                # Calculate relative differences
                relative_diff = ((cluster_means - overall_means) / overall_means).applymap(lambda x: f"{x:.2%}")
                
                # Style the dataframe for display
                st.write("Relative Difference from Overall Mean (%)")
                st.dataframe(relative_diff)
                
                # Create heatmap visualization of cluster means
                # Scale the means for better visualization
                scaled_means = (cluster_means - overall_means) / overall_means
                
                fig = px.imshow(
                    scaled_means,
                    color_continuous_scale='RdBu_r',
                    title="Cluster Feature Profiles (Relative to Overall Mean)",
                    labels=dict(x="Feature", y="Cluster", color="Relative Difference")
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster summaries in natural language
                st.subheader("Cluster Interpretations")
                
                for cluster in range(profile_n_clusters):
                    # Get dominant crops in this cluster
                    cluster_crop_counts = profile_df[profile_df['Cluster'] == cluster]['Crop'].value_counts()
                    dominant_crops = cluster_crop_counts[cluster_crop_counts >= cluster_crop_counts.max() * 0.5].index.tolist()
                    
                    # Find distinguishing features (highest positive and negative deviations)
                    cluster_profile = scaled_means.loc[cluster]
                    distinguishing_high = cluster_profile.nlargest(3).index.tolist()
                    distinguishing_low = cluster_profile.nsmallest(3).index.tolist()
                    
                    high_features_str = ", ".join([f"{feat} (+{scaled_means.loc[cluster, feat]:.1%})" for feat in distinguishing_high])
                    low_features_str = ", ".join([f"{feat} ({scaled_means.loc[cluster, feat]:.1%})" for feat in distinguishing_low])
                    
                    # Create interpretation text
                    st.markdown(f"""
                    <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 10px; 
                    margin-bottom: 15px; border-left: 5px solid rgba(0, 242, 254, 0.7);">
                    <h4 style="color: #00f2fe; margin-top: 0;">Cluster {cluster} Profile</h4>
                    <p style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                    <span style="color: #00f2fe;">></span> <strong>Dominant Crops:</strong> {', '.join(dominant_crops)}<br>
                    <span style="color: #00f2fe;">></span> <strong>Distinguishing High Features:</strong> {high_features_str}<br>
                    <span style="color: #00f2fe;">></span> <strong>Distinguishing Low Features:</strong> {low_features_str}<br>
                    <span style="color: #00f2fe;">></span> <strong>Sample Size:</strong> {(profile_df['Cluster'] == cluster).sum()} samples 
                    ({(profile_df['Cluster'] == cluster).sum() / len(profile_df):.1%} of dataset)
                    </p>
                    </div>
                    """, unsafe_allow_html=True)

    # Add Advanced Analysis page
    elif page == "Advanced Analysis":
        st.markdown('<h1 class="title-gradient">Advanced Data Analysis Lab</h1>', unsafe_allow_html=True)
        
        # Create a pulsing effect for the status indicator
        st.markdown("""
        <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.5); }
            70% { box-shadow: 0 0 0 10px rgba(0, 242, 254, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            background-color: #00f2fe;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        </style>
        <div style="background-color: rgba(0,40,80,0.7); padding: 15px; border-radius: 10px; 
        margin-bottom: 20px; border: 1px solid rgba(0, 242, 254, 0.3);">
        <div style="display: flex; align-items: center;">
        <span class="status-indicator"></span>
        <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
        ADVANCED ANALYSIS SYSTEM OPERATIONAL
        </span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different types of advanced analysis
        tabs = st.tabs([
            "Model Comparison", 
            "Confusion Matrix Analysis", 
            "ROC & Precision-Recall Curves",
            "Learning Curves"
        ])
        
        # Extract and preprocess data
        X = df.drop('label', axis=1)
        y = df['label']
        feature_names = X.columns.tolist()
        class_names = sorted(y.unique())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Tab 1: Model Comparison
        with tabs[0]:
            st.subheader("Comprehensive Model Comparison")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            This analysis compares multiple machine learning models to identify the best performer for crop recommendation.
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Training and evaluating multiple models..."):
                # Define models to compare
                models_to_compare = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42)
                }
                
                # Train and evaluate each model
                results = {}
                predictions = {}
                
                for name, model in models_to_compare.items():
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    predictions[name] = y_pred
                    
                    # Compute metrics
                    results[name] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    }
                
                # Create DataFrame for comparison
                comparison_df = pd.DataFrame(index=results.keys())
                
                # Add metrics
                comparison_df['Accuracy'] = [results[model]['accuracy'] for model in results]
                comparison_df['Precision'] = [results[model]['precision'] for model in results]
                comparison_df['Recall'] = [results[model]['recall'] for model in results]
                comparison_df['F1 Score'] = [results[model]['f1_score'] for model in results]
                
                # Sort by accuracy
                comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
                
                # Create visual comparison
                comparison_long = comparison_df.reset_index().melt(
                    id_vars='index',
                    value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    var_name='Metric',
                    value_name='Score'
                )
                comparison_long.columns = ['Model', 'Metric', 'Score']
                
                # Format as percentages for display
                comparison_display = comparison_df.copy()
                for col in comparison_display.columns:
                    comparison_display[col] = comparison_display[col].map(lambda x: f"{x:.2%}")
                
                # Display the metrics table
                st.subheader("Model Performance Metrics")
                st.table(comparison_display)
                
                # Create bar chart comparison
                fig = px.bar(
                    comparison_long,
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title="Model Performance Comparison",
                    labels={'Score': 'Score (higher is better)'},
                    color_discrete_sequence=['#00f2fe', '#00ff9d', '#5483ff', '#ff54a0']
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find best model
                best_model_name = comparison_df.index[0]
                
                # Display best model info
                st.markdown(f"""
                <div style="background-color: rgba(0,30,60,0.7); padding: 20px; border-radius: 10px; 
                margin: 20px 0; border: 1px solid rgba(0, 242, 254, 0.3);">
                <h3 style="color: #00f2fe; margin-top: 0;">Optimal Model Detected</h3>
                <p style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.5em; margin: 10px 0;">
                {best_model_name}
                </p>
                <p style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                Accuracy: <span style="color: #00ff9d;">{results[best_model_name]['accuracy']:.2%}</span><br>
                F1 Score: <span style="color: #00ff9d;">{results[best_model_name]['f1_score']:.2%}</span>
                </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Prediction agreement analysis
                st.subheader("Model Agreement Analysis")
                st.markdown("""
                <div style="font-family: 'Courier New', monospace; color: #e0e0e0; font-size: 0.9em;">
                This analysis shows how often different models agree on their predictions. 
                Higher agreement suggests more confident predictions.
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate agreement matrix
                model_names = list(models_to_compare.keys())
                agreement_matrix = np.zeros((len(model_names), len(model_names)))
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i == j:
                            agreement_matrix[i, j] = 1.0
                        else:
                            # Calculate agreement percentage
                            agreement = np.mean(predictions[model1] == predictions[model2])
                            agreement_matrix[i, j] = agreement
                
                # Create heatmap
                fig = px.imshow(
                    agreement_matrix,
                    x=model_names,
                    y=model_names,
                    color_continuous_scale='Blues',
                    text_auto='.2%',
                    title="Model Prediction Agreement",
                    labels=dict(x="Model", y="Model", color="Agreement Rate")
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 2: Confusion Matrix Analysis
        with tabs[1]:
            st.subheader("Confusion Matrix Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Confusion matrices show prediction errors across different classes, helping identify which crops
            are most often confused with each other.
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection
            confusion_model = st.selectbox(
                "Select Model for Confusion Matrix Analysis",
                ["Random Forest", "SVM", "KNN", "Gradient Boosting"],
                index=0
            )
            
            with st.spinner("Generating confusion matrix..."):
                # Train selected model
                if confusion_model == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif confusion_model == "SVM":
                    model = SVC(kernel='rbf', probability=True, random_state=42)
                elif confusion_model == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                else:  # Gradient Boosting
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Normalize confusion matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                # Create heatmap visualization
                fig = px.imshow(
                    cm_norm,
                    x=class_names,
                    y=class_names,
                    color_continuous_scale='Viridis',
                    text_auto='.2%',
                    title=f"Normalized Confusion Matrix - {confusion_model}",
                    labels=dict(x="Predicted Crop", y="True Crop", color="Percentage")
                )
                fig.update_layout(height=700, width=700)
                st.plotly_chart(fig, use_container_width=True)
                
                # Identify most confused pairs
                st.subheader("Most Confused Crop Pairs")
                
                # Get off-diagonal elements
                off_diag = np.where(~np.eye(cm_norm.shape[0], dtype=bool), cm_norm, 0)
                
                # Find top 5 confused pairs
                confused_pairs = []
                for _ in range(min(5, (off_diag > 0).sum())):
                    # Find max value
                    max_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
                    true_crop, pred_crop = class_names[max_idx[0]], class_names[max_idx[1]]
                    confusion_rate = off_diag[max_idx]
                    
                    # Add to list
                    confused_pairs.append({
                        'True Crop': true_crop,
                        'Predicted Crop': pred_crop,
                        'Confusion Rate': confusion_rate
                    })
                    
                    # Set this pair to 0 to find next highest
                    off_diag[max_idx] = 0
                
                # Create dataframe
                confused_df = pd.DataFrame(confused_pairs)
                confused_df['Confusion Rate'] = confused_df['Confusion Rate'].map(lambda x: f"{x:.2%}")
                
                # Display the table
                st.table(confused_df)
                
                # Feature analysis for confused crops
                if confused_pairs:
                    st.subheader("Feature Analysis for Most Confused Pair")
                    
                    # Get first pair
                    true_crop = confused_pairs[0]['True Crop']
                    pred_crop = confused_pairs[0]['Predicted Crop']
                    
                    # Get data for these crops
                    true_crop_data = X[y == true_crop]
                    pred_crop_data = X[y == pred_crop]
                    
                    # Compare features
                    comparison_data = pd.DataFrame({
                        'Feature': feature_names,
                        f'{true_crop} Mean': true_crop_data.mean().values,
                        f'{pred_crop} Mean': pred_crop_data.mean().values,
                        'Difference (%)': ((true_crop_data.mean() - pred_crop_data.mean()) / pred_crop_data.mean() * 100).values
                    }).sort_values('Difference (%)', key=abs, ascending=False)
                    
                    # Format percentages
                    comparison_data['Difference (%)'] = comparison_data['Difference (%)'].map(lambda x: f"{x:.1f}%")
                    
                    # Display comparison
                    st.table(comparison_data)
                    
                    # Create comparison visualization
                    st.markdown(f"### Feature Comparison: {true_crop} vs {pred_crop}")
                    
                    # Prepare data for radar chart
                    true_crop_means = true_crop_data.mean().values
                    pred_crop_means = pred_crop_data.mean().values
                    
                    # Scale values for better visualization
                    max_values = np.maximum(true_crop_means, pred_crop_means)
                    min_values = np.minimum(true_crop_means, pred_crop_means)
                    scale_factor = max_values - min_values
                    scale_factor[scale_factor == 0] = 1  # Avoid division by zero
                    
                    scaled_true = (true_crop_means - min_values) / scale_factor
                    scaled_pred = (pred_crop_means - min_values) / scale_factor
                    
                    # Create radar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=scaled_true,
                        theta=feature_names,
                        fill='toself',
                        name=true_crop
                    ))
                    
                    fig.add_trace(go.Scatterpolar(
                        r=scaled_pred,
                        theta=feature_names,
                        fill='toself',
                        name=pred_crop
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        title=f"Normalized Feature Comparison: {true_crop} vs {pred_crop}",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: ROC & Precision-Recall Curves
        with tabs[2]:
            st.subheader("ROC & Precision-Recall Curves")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            These curves evaluate model performance across different threshold settings, 
            helping understand the tradeoff between different types of errors.
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection
            curve_model = st.selectbox(
                "Select Model for Curve Analysis",
                ["Random Forest", "SVM", "KNN", "Gradient Boosting"],
                index=0
            )
            
            with st.spinner("Generating performance curves..."):
                # Train selected model
                if curve_model == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif curve_model == "SVM":
                    model = SVC(kernel='rbf', probability=True, random_state=42)
                elif curve_model == "KNN":
                    model = KNeighborsClassifier(n_neighbors=5)
                else:  # Gradient Boosting
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Get class probabilities
                try:
                    y_scores = model.predict_proba(X_test_scaled)
                    
                    # Create a subplot with 1 row and 2 columns for ROC and PR curves
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC Curves", "Precision-Recall Curves"))
                    
                    # Calculate ROC curves for each class
                    for i, class_name in enumerate(class_names):
                        # Prepare binary labels for this class (one-vs-rest)
                        y_test_binary = (y_test == class_name).astype(int)
                        y_score = y_scores[:, i]
                        
                        # Calculate ROC
                        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
                        roc_auc = auc(fpr, tpr)
                        
                        # Add ROC curve to subplot
                        fig.add_trace(
                            go.Scatter(
                                x=fpr, 
                                y=tpr,
                                name=f'{class_name} (AUC = {roc_auc:.3f})',
                                mode='lines'
                            ),
                            row=1, col=1
                        )
                        
                        # Calculate Precision-Recall
                        precision, recall, _ = precision_recall_curve(y_test_binary, y_score)
                        pr_auc = auc(recall, precision)
                        
                        # Add PR curve to subplot
                        fig.add_trace(
                            go.Scatter(
                                x=recall, 
                                y=precision,
                                name=f'{class_name} (AUC = {pr_auc:.3f})',
                                mode='lines'
                            ),
                            row=1, col=2
                        )
                    
                    # Add diagonal line to ROC plot
                    fig.add_trace(
                        go.Scatter(
                            x=[0, 1], 
                            y=[0, 1],
                            mode='lines',
                            line=dict(dash='dash', color='gray'),
                            name='Random Guess',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # Update layout
                    fig.update_layout(
                        height=600,
                        title_text=f"ROC and Precision-Recall Curves - {curve_model}",
                        showlegend=True
                    )
                    
                    # Update axes labels
                    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
                    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
                    fig.update_xaxes(title_text="Recall", row=1, col=2)
                    fig.update_yaxes(title_text="Precision", row=1, col=2)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate macro-average ROC AUC
                    y_test_encoded = pd.get_dummies(y_test).values
                    macro_roc_auc = 0
                    
                    for i in range(len(class_names)):
                        fpr, tpr, _ = roc_curve(y_test_encoded[:, i], y_scores[:, i])
                        macro_roc_auc += auc(fpr, tpr)
                    
                    macro_roc_auc /= len(class_names)
                    
                    # Display macro-average AUC
                    st.markdown(f"""
                    <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 10px; 
                    margin-top: 20px; border: 1px solid rgba(0, 242, 254, 0.3); text-align: center;">
                    <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                    Macro-Average ROC AUC Score:
                    </span>
                    <span style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.5em; margin-left: 10px;">
                    {macro_roc_auc:.4f}
                    </span>
                    </div>
                    """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Could not generate curves: {str(e)}")
                    st.info("This error might occur if the model doesn't support probability predictions.")
        
        # Tab 4: Learning Curves
        with tabs[3]:
            st.subheader("Learning Curves Analysis")
            st.markdown("""
            <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,20,40,0.7); 
            padding: 10px; border-radius: 5px; border-left: 3px solid #00f2fe;">
            Learning curves show how model performance changes with increasing training data size,
            helping diagnose overfitting or underfitting issues.
            </div>
            """, unsafe_allow_html=True)
            
            # Model selection
            learning_model = st.selectbox(
                "Select Model for Learning Curves",
                ["Random Forest", "SVM", "KNN"],
                index=0
            )
            
            with st.spinner("Generating learning curves..."):
                # Initialize model
                if learning_model == "Random Forest":
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                elif learning_model == "SVM":
                    model = SVC(kernel='rbf', probability=True, random_state=42)
                else:  # KNN
                    model = KNeighborsClassifier(n_neighbors=5)
                
                # Define train sizes
                train_sizes = np.linspace(0.1, 1.0, 10)
                
                # Calculate learning curves
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X_scaled, y, train_sizes=train_sizes,
                        cv=5, scoring='accuracy', n_jobs=-1, random_state=42
                    )
                    
                    # Calculate mean and std
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    # Create plot
                    fig = go.Figure()
                    
                    # Add train score
                    fig.add_trace(go.Scatter(
                        x=train_sizes,
                        y=train_mean,
                        mode='lines+markers',
                        name='Training Score',
                        line=dict(color='#00f2fe', width=3),
                        marker=dict(size=8),
                        error_y=dict(
                            type='data', 
                            array=train_std,
                            visible=True,
                            color='#00f2fe'
                        )
                    ))
                    
                    # Add test score
                    fig.add_trace(go.Scatter(
                        x=train_sizes,
                        y=test_mean,
                        mode='lines+markers',
                        name='Cross-Validation Score',
                        line=dict(color='#00ff9d', width=3),
                        marker=dict(size=8),
                        error_y=dict(
                            type='data', 
                            array=test_std,
                            visible=True,
                            color='#00ff9d'
                        )
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Learning Curves - {learning_model}",
                        xaxis_title="Training Examples",
                        yaxis_title="Accuracy",
                        height=600,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpret the learning curves
                    gap = train_mean[-1] - test_mean[-1]
                    
                    if gap > 0.1:
                        situation = "high variance (overfitting)"
                        recommendation = "Try collecting more training data, using regularization, or reducing model complexity."
                    elif train_mean[-1] < 0.8:
                        situation = "high bias (underfitting)"
                        recommendation = "Try increasing model complexity, adding features, or reducing regularization."
                    else:
                        situation = "good fit"
                        recommendation = "The model fits the data well. Try fine-tuning hyperparameters for potentially small improvements."
                    
                    st.markdown(f"""
                    <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 10px; 
                    margin-top: 20px; border: 1px solid rgba(0, 242, 254, 0.3);">
                    <h4 style="color: #00f2fe; margin-top: 0;">Learning Curve Interpretation</h4>
                    <p style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                    <span style="color: #00f2fe;">></span> <strong>Training score:</strong> {train_mean[-1]:.4f}<br>
                    <span style="color: #00f2fe;">></span> <strong>Validation score:</strong> {test_mean[-1]:.4f}<br>
                    <span style="color: #00f2fe;">></span> <strong>Gap:</strong> {gap:.4f}<br>
                    <span style="color: #00f2fe;">></span> <strong>Diagnosis:</strong> {situation.title()}<br>
                    <span style="color: #00f2fe;">></span> <strong>Recommendation:</strong> {recommendation}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Validation curves for a key hyperparameter
                    st.subheader("Validation Curves")
                    
                    # Define parameter to analyze based on model
                    if learning_model == "Random Forest":
                        param_name = "max_depth"
                        param_range = [3, 5, 7, 9, 11, 13, 15, None]
                        param_display = ["3", "5", "7", "9", "11", "13", "15", "None"]
                    elif learning_model == "SVM":
                        param_name = "C"
                        param_range = [0.01, 0.1, 1, 10, 100]
                        param_display = [str(x) for x in param_range]
                    else:  # KNN
                        param_name = "n_neighbors"
                        param_range = [1, 3, 5, 7, 9, 11, 13]
                        param_display = [str(x) for x in param_range]
                    
                    st.markdown(f"""
                    <div style="font-family: 'Courier New', monospace; color: #e0e0e0; font-size: 0.9em; margin-bottom: 15px;">
                    Validation curves show model performance as a function of a specific hyperparameter,
                    helping identify the optimal parameter value.
                    <br><br>
                    Analyzing parameter: <span style="color: #00f2fe;">{param_name}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    try:
                        # Calculate validation curves
                        train_scores, test_scores = validation_curve(
                            model, X_scaled, y, param_name=param_name, param_range=param_range,
                            cv=5, scoring='accuracy', n_jobs=-1
                        )
                        
                        # Calculate mean and std
                        train_mean = np.mean(train_scores, axis=1)
                        train_std = np.std(train_scores, axis=1)
                        test_mean = np.mean(test_scores, axis=1)
                        test_std = np.std(test_scores, axis=1)
                        
                        # Create plot
                        fig = go.Figure()
                        
                        # Add train score
                        fig.add_trace(go.Scatter(
                            x=param_display,
                            y=train_mean,
                            mode='lines+markers',
                            name='Training Score',
                            line=dict(color='#00f2fe', width=3),
                            marker=dict(size=8),
                            error_y=dict(
                                type='data', 
                                array=train_std,
                                visible=True,
                                color='#00f2fe'
                            )
                        ))
                        
                        # Add test score
                        fig.add_trace(go.Scatter(
                            x=param_display,
                            y=test_mean,
                            mode='lines+markers',
                            name='Cross-Validation Score',
                            line=dict(color='#00ff9d', width=3),
                            marker=dict(size=8),
                            error_y=dict(
                                type='data', 
                                array=test_std,
                                visible=True,
                                color='#00ff9d'
                            )
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Validation Curve - {param_name}",
                            xaxis_title=param_name,
                            yaxis_title="Accuracy",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Identify best parameter value
                        best_idx = np.argmax(test_mean)
                        best_param_value = param_display[best_idx]
                        best_score = test_mean[best_idx]
                        
                        st.markdown(f"""
                        <div style="background-color: rgba(0,30,60,0.7); padding: 15px; border-radius: 10px; 
                        text-align: center; margin-top: 10px;">
                        <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                        Optimal value for {param_name}:
                        </span>
                        <span style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.2em; margin: 0 10px;">
                        {best_param_value}
                        </span>
                        <span style="font-family: 'Courier New', monospace; color: #e0e0e0;">
                        with accuracy:
                        </span>
                        <span style="font-family: 'Orbitron', sans-serif; color: #00ff9d; font-size: 1.2em; margin-left: 10px;">
                        {best_score:.4f}
                        </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Could not generate validation curves: {str(e)}")
                
                except Exception as e:
                    st.error(f"Could not generate learning curves: {str(e)}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    
    # Display error details in a futuristic style
    st.markdown(f"""
    <div style="background-color: rgba(50,0,0,0.7); padding: 20px; border-radius: 10px; 
    margin-top: 20px; border: 1px solid rgba(255, 50, 50, 0.5);">
    <h3 style="color: #ff5050; margin-top: 0;">SYSTEM ERROR DETECTED</h3>
    <div style="font-family: 'Courier New', monospace; color: #e0e0e0; background-color: rgba(0,0,0,0.5); 
    padding: 15px; border-radius: 5px; overflow-x: auto;">
    {str(e)}
    </div>
    <p style="font-family: 'Courier New', monospace; color: #e0e0e0; margin-top: 15px;">
    Please make sure the data file "Crop_recommendation.csv" is present in the "attached_assets" directory.
    </p>
    </div>
    """, unsafe_allow_html=True)