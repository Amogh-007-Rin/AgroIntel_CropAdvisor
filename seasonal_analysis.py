import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def simulate_seasonal_data(df, feature_names, n_seasons=4):
    """
    Simulate seasonal variations in agricultural data based on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    feature_names : list
        List of feature names to consider
    n_seasons : int, default=4
        Number of seasons to simulate
        
    Returns:
    --------
    seasonal_data : dict
        Dictionary containing seasonal data for each crop
    """
    # Create a dictionary to store seasonal data
    seasonal_data = {}
    
    # Define seasonal variations (simple model)
    seasonal_variations = {
        'temperature': {
            'Spring': {'mean_factor': 0.9, 'std_factor': 1.2},
            'Summer': {'mean_factor': 1.3, 'std_factor': 0.8},
            'Fall': {'mean_factor': 1.0, 'std_factor': 1.0},
            'Winter': {'mean_factor': 0.7, 'std_factor': 1.5}
        },
        'humidity': {
            'Spring': {'mean_factor': 1.1, 'std_factor': 0.9},
            'Summer': {'mean_factor': 0.8, 'std_factor': 1.3},
            'Fall': {'mean_factor': 1.2, 'std_factor': 0.8},
            'Winter': {'mean_factor': 1.0, 'std_factor': 1.0}
        },
        'rainfall': {
            'Spring': {'mean_factor': 1.2, 'std_factor': 1.1},
            'Summer': {'mean_factor': 0.7, 'std_factor': 1.5},
            'Fall': {'mean_factor': 1.1, 'std_factor': 1.2},
            'Winter': {'mean_factor': 1.0, 'std_factor': 0.9}
        }
    }
    
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    
    # Generate seasonal data for each crop
    for crop in df['label'].unique():
        crop_data = df[df['label'] == crop]
        
        seasonal_data[crop] = {}
        
        for season in seasons[:n_seasons]:
            # Create seasonal variation of the data
            seasonal_crop_data = crop_data[feature_names].copy()
            
            # Apply seasonal variations to relevant features
            for feature in ['temperature', 'humidity', 'rainfall']:
                if feature in feature_names:
                    mean = crop_data[feature].mean()
                    std = crop_data[feature].std()
                    
                    variation = seasonal_variations[feature][season]
                    
                    # Adjust mean and standard deviation
                    new_mean = mean * variation['mean_factor']
                    new_std = std * variation['std_factor']
                    
                    # Generate new values
                    seasonal_crop_data[feature] = np.random.normal(
                        loc=new_mean,
                        scale=new_std,
                        size=len(seasonal_crop_data)
                    )
                    
                    # Ensure values are within reasonable bounds
                    if feature == 'temperature':
                        seasonal_crop_data[feature] = seasonal_crop_data[feature].clip(5, 45)
                    elif feature == 'humidity':
                        seasonal_crop_data[feature] = seasonal_crop_data[feature].clip(30, 100)
                    elif feature == 'rainfall':
                        seasonal_crop_data[feature] = seasonal_crop_data[feature].clip(10, 400)
            
            seasonal_data[crop][season] = seasonal_crop_data
    
    return seasonal_data

def analyze_seasonal_suitability(seasonal_data, models, feature_names, top_n=3):
    """
    Analyze the suitability of crops across different seasons.
    
    Parameters:
    -----------
    seasonal_data : dict
        Dictionary containing seasonal data for each crop
    models : dict
        Dictionary of trained machine learning models
    feature_names : list
        List of feature names
    top_n : int, default=3
        Number of top crops to recommend for each season
        
    Returns:
    --------
    seasonal_suitability : dict
        Dictionary containing suitability scores for each crop in each season
    recommendations : dict
        Dictionary containing top crop recommendations for each season
    """
    # Use the best model for predictions
    model = models["Random Forest"]  # Assuming Random Forest is the best model
    
    # Create dictionaries to store results
    seasonal_suitability = {}
    recommendations = {}
    
    # Get all crops and seasons
    all_crops = list(seasonal_data.keys())
    all_seasons = list(next(iter(seasonal_data.values())).keys())
    
    # Analyze each season
    for season in all_seasons:
        seasonal_suitability[season] = {}
        
        # Create a list to store suitability scores
        scores = []
        
        # Evaluate each crop in the current season
        for crop in all_crops:
            # Get seasonal data for this crop
            crop_seasonal_data = seasonal_data[crop][season]
            
            # Calculate average probability predictions for this crop in this season
            y_proba = model.predict_proba(crop_seasonal_data)
            
            # Get the index of the current crop in model's classes
            crop_idx = np.where(model.classes_ == crop)[0][0]
            
            # Get probability for the correct crop
            crop_proba = y_proba[:, crop_idx]
            
            # Calculate average suitability score
            avg_score = np.mean(crop_proba)
            
            # Store the score
            seasonal_suitability[season][crop] = avg_score
            scores.append((crop, avg_score))
        
        # Sort crops by suitability score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations for this season
        recommendations[season] = scores[:top_n]
    
    return seasonal_suitability, recommendations

def plot_seasonal_suitability(seasonal_suitability, top_n=10):
    """
    Plot crop suitability across seasons.
    
    Parameters:
    -----------
    seasonal_suitability : dict
        Dictionary containing suitability scores for each crop in each season
    top_n : int, default=10
        Number of top crops to include in the plot
    """
    # Convert to DataFrame for easier plotting
    seasons = list(seasonal_suitability.keys())
    crops = list(set().union(*[set(seasonal_suitability[season].keys()) for season in seasons]))
    
    # Create data for heatmap
    data = []
    for crop in crops:
        row = {'Crop': crop}
        for season in seasons:
            row[season] = seasonal_suitability[season].get(crop, 0)
        data.append(row)
    
    df_suitability = pd.DataFrame(data)
    
    # Sort crops by average suitability
    avg_suitability = df_suitability.iloc[:, 1:].mean(axis=1)
    df_suitability['AvgSuitability'] = avg_suitability
    df_suitability = df_suitability.sort_values('AvgSuitability', ascending=False)
    
    # Select top N crops
    df_top = df_suitability.head(top_n)
    
    # Prepare data for heatmap
    df_melted = df_top.melt(
        id_vars=['Crop', 'AvgSuitability'],
        value_vars=seasons,
        var_name='Season',
        value_name='Suitability'
    )
    
    # Create heatmap
    fig = px.density_heatmap(
        df_melted,
        x='Season',
        y='Crop',
        z='Suitability',
        title=f'Top {top_n} Crops: Seasonal Suitability',
        labels={'Suitability': 'Suitability Score'},
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'array', 'categoryarray': df_top['Crop'].tolist()}
    )
    
    return fig

def plot_seasonal_feature_variations(seasonal_data, feature_names):
    """
    Plot how features vary across seasons for different crops.
    
    Parameters:
    -----------
    seasonal_data : dict
        Dictionary containing seasonal data for each crop
    feature_names : list
        List of feature names
    """
    # Select features to visualize
    seasonal_features = ['temperature', 'humidity', 'rainfall']
    
    # Create tabs for each feature
    tabs = st.tabs(seasonal_features)
    
    # Get all seasons
    seasons = list(next(iter(seasonal_data.values())).keys())
    
    # Get a subset of crops for clarity
    all_crops = list(seasonal_data.keys())
    if len(all_crops) > 5:
        selected_crops = all_crops[:5]  # Use first 5 crops
    else:
        selected_crops = all_crops
    
    # Create plots for each feature
    for i, feature in enumerate(seasonal_features):
        with tabs[i]:
            st.write(f"### {feature.capitalize()} Variations by Season")
            
            # Create data for plotting
            data = []
            
            for crop in selected_crops:
                for season in seasons:
                    # Calculate average feature value for this crop in this season
                    avg_value = seasonal_data[crop][season][feature].mean()
                    
                    data.append({
                        'Crop': crop,
                        'Season': season,
                        feature.capitalize(): avg_value
                    })
            
            # Convert to DataFrame
            df_feature = pd.DataFrame(data)
            
            # Create grouped bar chart
            fig = px.bar(
                df_feature,
                x='Season',
                y=feature.capitalize(),
                color='Crop',
                barmode='group',
                title=f'Average {feature.capitalize()} by Season for Top Crops',
                labels={feature.capitalize(): f'{feature.capitalize()} Value'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            if feature == 'temperature':
                st.write("""
                **Temperature Variation Analysis:**
                
                Temperature significantly affects crop growth rates, with each crop having optimal temperature ranges.
                - Spring and Fall typically have moderate temperatures suitable for many crops
                - Summer peaks can be too hot for some crops but ideal for others
                - Winter temperatures are often too low for active growth of most crops
                """)
            elif feature == 'humidity':
                st.write("""
                **Humidity Variation Analysis:**
                
                Humidity affects transpiration rates and disease susceptibility.
                - High humidity can increase fungal disease risk in susceptible crops
                - Low humidity can increase water stress and irrigation requirements
                - Different crops have adapted to different humidity levels
                """)
            elif feature == 'rainfall':
                st.write("""
                **Rainfall Variation Analysis:**
                
                Rainfall directly impacts water availability and soil moisture.
                - Seasonal rainfall patterns determine natural irrigation
                - Excessive rainfall can cause waterlogging and nutrient leaching
                - Insufficient rainfall requires supplemental irrigation
                - Each crop has optimal water requirements
                """)

def create_seasonal_recommendations_ui(recommendations, seasonal_data, models, feature_names):
    """
    Create an interactive UI for seasonal crop recommendations.
    
    Parameters:
    -----------
    recommendations : dict
        Dictionary containing top crop recommendations for each season
    seasonal_data : dict
        Dictionary containing seasonal data for each crop
    models : dict
        Dictionary of trained machine learning models
    feature_names : list
        List of feature names
    """
    st.subheader("Seasonal Crop Recommendation Tool")
    st.write("""
    This tool helps you find the most suitable crops for your region based on seasonal conditions.
    Select a season and customize environmental factors to get personalized recommendations.
    """)
    
    # Get all seasons
    seasons = list(recommendations.keys())
    
    # Select season
    selected_season = st.selectbox("Select a season:", seasons)
    
    # Display top recommended crops for the selected season
    st.write(f"### Top Recommended Crops for {selected_season}")
    
    top_crops = recommendations[selected_season]
    
    # Create a DataFrame for the recommendations
    df_recommendations = pd.DataFrame(
        top_crops,
        columns=['Crop', 'Suitability Score']
    )
    
    # Format suitability score as percentage
    df_recommendations['Suitability Score'] = df_recommendations['Suitability Score'].map(lambda x: f"{x:.1%}")
    
    # Display as a table
    st.table(df_recommendations)
    
    # Create a horizontal line
    st.markdown("---")
    
    # Custom seasonal conditions
    st.write("### Customize Seasonal Conditions")
    st.write("Adjust the environmental factors to match your specific conditions:")
    
    # Create columns for input fields
    col1, col2, col3 = st.columns(3)
    
    # Get the default values for the selected season
    # Calculate average values across all crops for the selected season
    avg_temp = np.mean([seasonal_data[crop][selected_season]['temperature'].mean() 
                        for crop in seasonal_data.keys()])
    avg_humidity = np.mean([seasonal_data[crop][selected_season]['humidity'].mean() 
                           for crop in seasonal_data.keys()])
    avg_rainfall = np.mean([seasonal_data[crop][selected_season]['rainfall'].mean() 
                           for crop in seasonal_data.keys()])
    
    # Input fields for environmental factors
    with col1:
        temperature = st.slider(
            "Temperature (°C):", 
            min_value=5.0, 
            max_value=45.0, 
            value=float(avg_temp),
            step=0.5
        )
    
    with col2:
        humidity = st.slider(
            "Humidity (%):", 
            min_value=30.0, 
            max_value=100.0, 
            value=float(avg_humidity),
            step=1.0
        )
    
    with col3:
        rainfall = st.slider(
            "Rainfall (mm):", 
            min_value=10.0, 
            max_value=400.0, 
            value=float(avg_rainfall),
            step=5.0
        )
    
    # Soil parameters
    st.write("### Soil Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.slider("Nitrogen (N) content:", min_value=0, max_value=150, value=80, step=5)
    
    with col2:
        p = st.slider("Phosphorus (P) content:", min_value=0, max_value=150, value=50, step=5)
    
    with col3:
        k = st.slider("Potassium (K) content:", min_value=0, max_value=150, value=40, step=5)
    
    ph = st.slider("Soil pH:", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    
    # Button to get custom recommendations
    if st.button("Get Customized Recommendations"):
        # Create input data
        input_data = pd.DataFrame({
            'N': [n], 'P': [p], 'K': [k], 
            'temperature': [temperature], 'humidity': [humidity], 
            'ph': [ph], 'rainfall': [rainfall]
        })
        
        # Select model for prediction
        model = models["Random Forest"]
        
        # Get prediction probabilities for all crops
        probabilities = model.predict_proba(input_data)[0]
        
        # Create DataFrame with crop names and probabilities
        crop_probs = pd.DataFrame({
            'Crop': model.classes_,
            'Suitability Score': probabilities
        }).sort_values('Suitability Score', ascending=False)
        
        # Display top 5 recommendations
        st.write("### Your Customized Crop Recommendations")
        st.write(f"Based on your specific {selected_season} conditions:")
        
        # Display recommendations
        top_5_crops = crop_probs.head(5).copy()
        top_5_crops['Suitability Score'] = top_5_crops['Suitability Score'].map(lambda x: f"{x:.1%}")
        
        st.table(top_5_crops)
        
        # Create a bar chart for suitability scores
        fig = px.bar(
            crop_probs.head(10),
            x='Crop',
            y='Suitability Score',
            color='Suitability Score',
            color_continuous_scale='Viridis',
            title='Top 10 Suitable Crops for Your Conditions',
            labels={'Suitability Score': 'Suitability Score'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add seasonal growing tips for the top recommended crop
        top_crop = crop_probs['Crop'].iloc[0]
        
        st.write(f"### {selected_season} Growing Tips for {top_crop}")
        
        # Season-specific tips
        if selected_season == 'Spring':
            st.write(f"""
            **Planting Time**: Early to mid-spring after last frost
            
            **{selected_season} Care Tips for {top_crop}**:
            - Prepare soil with proper drainage and organic matter
            - Monitor for early-season pests that emerge with warmer weather
            - Gradual water increases as temperatures rise
            - Consider row covers for late frost protection
            """)
        
        elif selected_season == 'Summer':
            st.write(f"""
            **Planting Time**: Late spring to early summer
            
            **{selected_season} Care Tips for {top_crop}**:
            - Implement consistent irrigation schedule, preferably in early morning
            - Apply mulch to conserve soil moisture and reduce weed competition
            - Monitor for heat stress during peak temperature periods
            - Consider partial shade during extreme heat if necessary
            """)
        
        elif selected_season == 'Fall':
            st.write(f"""
            **Planting Time**: Late summer to early fall
            
            **{selected_season} Care Tips for {top_crop}**:
            - Watch for early frost warnings and protect plants as needed
            - Reduce fertilization as growth slows
            - Monitor soil moisture as rainfall patterns change
            - Consider extending season with row covers or cold frames
            """)
        
        elif selected_season == 'Winter':
            st.write(f"""
            **Planting Time**: Late fall for cold-tolerant varieties
            
            **{selected_season} Care Tips for {top_crop}**:
            - Provide adequate protection from freezing temperatures
            - Reduce watering frequency but maintain soil moisture
            - Monitor for snow and ice damage to plants
            - Consider greenhouse or indoor cultivation in extreme conditions
            """)

def run_seasonal_analysis(df, models, feature_names, target_names):
    """
    Run a complete seasonal analysis workflow.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original dataset
    models : dict
        Dictionary of trained machine learning models
    feature_names : list
        List of feature names
    target_names : list
        List of target/crop names
    """
    st.header("Seasonal Crop Analysis")
    st.write("""
    This analysis explores how crop suitability varies across different seasons.
    It helps identify which crops are best suited for each season in your region.
    """)
    
    # Create tabs for different analysis sections
    tabs = st.tabs([
        "Seasonal Suitability",
        "Feature Variations",
        "Recommendation Tool"
    ])
    
    # Generate seasonal data
    with st.spinner("Generating seasonal data..."):
        seasonal_data = simulate_seasonal_data(df, feature_names)
    
    # Analyze seasonal suitability
    with st.spinner("Analyzing seasonal crop suitability..."):
        seasonal_suitability, recommendations = analyze_seasonal_suitability(
            seasonal_data, models, feature_names
        )
    
    # Tab 1: Seasonal Suitability
    with tabs[0]:
        st.subheader("Crop Suitability by Season")
        st.write("""
        This heatmap shows how suitable different crops are for each season.
        Darker colors indicate higher suitability scores.
        """)
        
        # Plot seasonal suitability
        fig = plot_seasonal_suitability(seasonal_suitability)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add seasonal insights
        st.write("### Key Seasonal Insights")
        
        # Find the best season for each crop
        best_seasons = {}
        for crop in target_names:
            crop_scores = {season: seasonal_suitability[season].get(crop, 0) 
                          for season in seasonal_suitability.keys()}
            best_season = max(crop_scores.items(), key=lambda x: x[1])[0]
            best_seasons[crop] = best_season
        
        # Count crops per best season
        season_counts = {}
        for season in seasonal_suitability.keys():
            season_counts[season] = sum(1 for s in best_seasons.values() if s == season)
        
        # Create a pie chart of crop distribution by best season
        fig = px.pie(
            values=list(season_counts.values()),
            names=list(season_counts.keys()),
            title='Distribution of Crops by Best Growing Season',
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # List top crops for each season
        st.write("### Top 3 Recommended Crops by Season")
        
        col1, col2, col3, col4 = st.columns(4)
        
        for i, (season, col) in enumerate(zip(seasonal_suitability.keys(), [col1, col2, col3, col4])):
            with col:
                st.write(f"**{season}**")
                for crop, score in recommendations[season]:
                    st.write(f"- {crop} ({score:.1%})")
    
    # Tab 2: Feature Variations
    with tabs[1]:
        st.subheader("Seasonal Feature Variations")
        st.write("""
        These charts show how key environmental factors vary across seasons
        and how they affect different crops.
        """)
        
        # Plot feature variations
        plot_seasonal_feature_variations(seasonal_data, feature_names)
    
    # Tab 3: Recommendation Tool
    with tabs[2]:
        create_seasonal_recommendations_ui(recommendations, seasonal_data, models, feature_names)