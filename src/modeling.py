import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """
    Train multiple machine learning models and evaluate their performance.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    metrics : dict
        Dictionary of performance metrics for each model
    """
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    # Train models
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    # Evaluate models
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics
        metrics[name] = {
            'accuracy': accuracy,
            'report': report,
            'predictions': y_pred
        }
    
    return models, metrics

def predict_crop(input_data, model, target_names):
    """
    Predict the most suitable crop for given soil and environmental conditions.
    
    Parameters:
    -----------
    input_data : pandas.DataFrame
        Input features for prediction
    model : trained model
        Trained machine learning model
    target_names : list
        List of crop names corresponding to class labels
        
    Returns:
    --------
    predicted_crop : str
        Predicted crop name
    probabilities : numpy.ndarray
        Probability scores for each crop type
    """
    # Make prediction
    prediction = model.predict(input_data)[0]
    predicted_crop = prediction
    
    # Get probabilities for each class
    probabilities = model.predict_proba(input_data)
    
    return predicted_crop, probabilities

def feature_importance(model, feature_names):
    """
    Extract feature importance from a trained model.
    
    Parameters:
    -----------
    model : trained model
        Trained machine learning model with feature_importances_ attribute
    feature_names : list
        List of feature names
        
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with feature names and their importance scores
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models without built-in feature importance
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': [0] * len(feature_names)
        })
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df
