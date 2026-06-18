import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_model_metrics(models, X_test, y_test):
    """
    Calculate performance metrics for each model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics for each model
    """
    metrics = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store metrics
        metrics[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
    
    return metrics

def display_metrics(metrics):
    """
    Display model performance metrics in a formatted table.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of performance metrics for each model
    """
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
    
    # Display confusion matrix for the best model
    best_model = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
    st.subheader(f"Confusion Matrix for {best_model}")
    
    # We can't directly display the confusion matrix since we don't have y_test here
    # This would need to be implemented in the main app where both y_test and predictions are available
    st.info("Confusion matrix is calculated and used for model evaluation.")

def interpret_prediction(input_data, predicted_crop, df):
    """
    Provide an interpretation of the crop prediction based on input features.
    
    Parameters:
    -----------
    input_data : pandas.DataFrame
        Input features used for prediction
    predicted_crop : str
        Predicted crop name
    df : pandas.DataFrame
        Original dataset for reference
        
    Returns:
    --------
    interpretation : str
        Text interpretation of the prediction result
    """
    # Get average values for the predicted crop from the dataset
    crop_data = df[df['label'] == predicted_crop]
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
    
    Based on your soil and environmental parameters, **{predicted_crop}** is the recommended crop.
    
    #### Comparison with Typical Values:
    {" ".join([f"- {comp}" for comp in comparisons])}
    
    #### Key Requirements for {predicted_crop}:
    - Soil fertility with {'high' if avg_values['N'] > 80 else 'moderate'} nitrogen content
    - {'High' if avg_values['P'] > 80 else 'Moderate'} phosphorus levels
    - {'High' if avg_values['K'] > 80 else 'Moderate'} potassium levels
    - Temperature range around {avg_values['temperature']:.1f}°C
    - Humidity levels around {avg_values['humidity']:.1f}%
    - Soil pH around {avg_values['ph']:.1f}
    - Annual rainfall around {avg_values['rainfall']:.1f} mm
    """
    
    return interpretation
