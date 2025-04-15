import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def train_advanced_models(X_train, y_train, cv=5):
    """
    Train advanced machine learning models with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target vector
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    best_models : dict
        Dictionary of best trained models
    cv_results : dict
        Cross-validation results
    """
    best_models = {}
    cv_results = {}
    
    # Create model pipelines
    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ])
    
    gb_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
    
    # Define parameter grids
    rf_param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    
    gb_param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 10]
    }
    
    # Perform grid search
    with st.spinner("Tuning Random Forest model..."):
        rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='accuracy', verbose=0)
        rf_grid.fit(X_train, y_train)
        best_models['Random Forest'] = rf_grid.best_estimator_
        cv_results['Random Forest'] = rf_grid.cv_results_
    
    with st.spinner("Tuning Gradient Boosting model..."):
        gb_grid = GridSearchCV(gb_pipeline, gb_param_grid, cv=cv, scoring='accuracy', verbose=0)
        gb_grid.fit(X_train, y_train)
        best_models['Gradient Boosting'] = gb_grid.best_estimator_
        cv_results['Gradient Boosting'] = gb_grid.cv_results_
    
    return best_models, cv_results

def evaluate_model_performance(model, X_test, y_test, class_names):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    model : trained model
        Trained machine learning model
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target vector
    class_names : list
        List of class names
        
    Returns:
    --------
    metrics : dict
        Dictionary of performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    
    # One-vs-Rest ROC curves
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=class_names)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    metrics['roc_curves'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # Precision-Recall curves
    precision = {}
    recall = {}
    pr_auc = {}
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    metrics['pr_curves'] = {'precision': precision, 'recall': recall, 'auc': pr_auc}
    
    return metrics

def plot_confusion_matrix(confusion_mat, class_names):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
    -----------
    confusion_mat : numpy.ndarray
        Confusion matrix
    class_names : list
        List of class names
    """
    # Create a DataFrame for better visualization
    conf_df = pd.DataFrame(confusion_mat, index=class_names, columns=class_names)
    
    # Calculate percentages
    conf_pct = conf_df.div(conf_df.sum(axis=1), axis=0) * 100
    
    # Plot heatmap with values
    fig = px.imshow(
        conf_pct,
        text_auto='.1f',
        labels=dict(x="Predicted", y="True", color="Percentage"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title='Confusion Matrix (% of True Class)',
        height=600,
        width=600
    )
    
    # Add annotation suffix
    for i, row in enumerate(conf_pct.values):
        for j, val in enumerate(row):
            fig.data[0].customdata[i][j] = f"{val:.1f}%"
    
    fig.update_traces(
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{customdata}<extra></extra>"
    )
    
    return fig

def plot_roc_curves(fpr, tpr, roc_auc, class_names):
    """
    Plot ROC curves for all classes.
    
    Parameters:
    -----------
    fpr : dict
        False positive rates for each class
    tpr : dict
        True positive rates for each class
    roc_auc : dict
        ROC AUC values for each class
    class_names : list
        List of class names
    """
    fig = go.Figure()
    
    for i, crop in enumerate(class_names):
        fig.add_trace(
            go.Scatter(
                x=fpr[i],
                y=tpr[i],
                name=f'{crop} (AUC = {roc_auc[i]:.3f})',
                mode='lines'
            )
        )
    
    # Add diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier',
            showlegend=True
        )
    )
    
    fig.update_layout(
        title='ROC Curves (One-vs-Rest)',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        height=500
    )
    
    return fig

def plot_precision_recall_curves(precision, recall, pr_auc, class_names):
    """
    Plot Precision-Recall curves for all classes.
    
    Parameters:
    -----------
    precision : dict
        Precision values for each class
    recall : dict
        Recall values for each class
    pr_auc : dict
        PR AUC values for each class
    class_names : list
        List of class names
    """
    fig = go.Figure()
    
    for i, crop in enumerate(class_names):
        fig.add_trace(
            go.Scatter(
                x=recall[i],
                y=precision[i],
                name=f'{crop} (AUC = {pr_auc[i]:.3f})',
                mode='lines'
            )
        )
    
    fig.update_layout(
        title='Precision-Recall Curves',
        xaxis_title='Recall',
        yaxis_title='Precision',
        legend=dict(x=0.01, y=0.01, bgcolor='rgba(255,255,255,0.8)'),
        height=500
    )
    
    return fig

def plot_learning_curves(estimator, X, y, cv=5):
    """
    Plot learning curves to evaluate model performance with varying training set sizes.
    
    Parameters:
    -----------
    estimator : trained model
        Trained machine learning model
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    cv : int, default=5
        Number of cross-validation folds
    """
    # Calculate learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create plot
    fig = go.Figure()
    
    # Add training score
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=8)
        )
    )
    
    # Add training score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False
        )
    )
    
    # Add test score
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_mean,
            mode='lines+markers',
            name='Cross-validation Score',
            line=dict(color='red'),
            marker=dict(symbol='circle', size=8)
        )
    )
    
    # Add test score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='Learning Curves',
        xaxis_title='Training Set Size',
        yaxis_title='Accuracy',
        legend=dict(x=0.01, y=0.01),
        height=400
    )
    
    return fig

def plot_validation_curves(estimator, X, y, param_name, param_range, cv=5):
    """
    Plot validation curves to evaluate model performance with varying hyperparameter values.
    
    Parameters:
    -----------
    estimator : sklearn estimator
        Untrained model
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    param_name : str
        Name of the parameter to vary
    param_range : list or array
        Range of parameter values to test
    cv : int, default=5
        Number of cross-validation folds
    """
    # Calculate validation curves
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring='accuracy', n_jobs=-1
    )
    
    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Create plot
    fig = go.Figure()
    
    # Add training score
    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            marker=dict(symbol='circle', size=8)
        )
    )
    
    # Add training score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([param_range, param_range[::-1]]),
            y=np.concatenate([train_mean + train_std, (train_mean - train_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False
        )
    )
    
    # Add test score
    fig.add_trace(
        go.Scatter(
            x=param_range,
            y=test_mean,
            mode='lines+markers',
            name='Cross-validation Score',
            line=dict(color='red'),
            marker=dict(symbol='circle', size=8)
        )
    )
    
    # Add test score error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([param_range, param_range[::-1]]),
            y=np.concatenate([test_mean + test_std, (test_mean - test_std)[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=f'Validation Curve for {param_name}',
        xaxis_title=param_name,
        yaxis_title='Accuracy',
        legend=dict(x=0.01, y=0.01),
        height=400
    )
    
    # Use log scale for certain parameters
    if 'learning_rate' in param_name or 'alpha' in param_name:
        fig.update_xaxes(type='log')
    
    return fig

def plot_feature_permutation_importance(model, X, y, feature_names, n_repeats=10, random_state=42):
    """
    Plot feature importance based on permutation importance.
    
    Parameters:
    -----------
    model : trained model
        Trained machine learning model
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    feature_names : list
        List of feature names
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state
    )
    
    # Create DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        error_x='Std',
        title='Feature Importance (Permutation-based)',
        labels={'Importance': 'Importance (decrease in model score)'},
        color='Importance',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    
    return fig

def analyze_hyperparameter_tuning(cv_results, model_name):
    """
    Analyze hyperparameter tuning results.
    
    Parameters:
    -----------
    cv_results : dict
        Cross-validation results from GridSearchCV
    model_name : str
        Name of the model
    """
    # Convert to DataFrame
    results_df = pd.DataFrame(cv_results)
    
    # Extract relevant columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    
    # Create figure
    fig = make_subplots(
        rows=len(param_cols),
        cols=1,
        subplot_titles=[col.replace('param_', '') for col in param_cols],
        vertical_spacing=0.1
    )
    
    for i, param in enumerate(param_cols):
        # Get unique parameter values
        param_values = results_df[param].astype(str).unique()
        
        # Calculate mean test scores for each parameter value
        mean_scores = []
        for value in param_values:
            mean_score = results_df[results_df[param].astype(str) == value]['mean_test_score'].mean()
            mean_scores.append(mean_score)
        
        # Sort by parameter value if numeric
        try:
            if 'None' in param_values:
                param_values = [v for v in param_values if v != 'None']
                param_values = sorted([float(v) for v in param_values]) + ['None']
                # Reorder mean_scores accordingly
                indices = [list(results_df[param].astype(str).unique()).index(str(v)) for v in param_values]
                mean_scores = [mean_scores[i] for i in indices]
            else:
                numeric_values = [float(v) for v in param_values]
                sorted_indices = np.argsort(numeric_values)
                param_values = [param_values[i] for i in sorted_indices]
                mean_scores = [mean_scores[i] for i in sorted_indices]
        except ValueError:
            # Non-numeric parameters
            pass
        
        # Add trace
        fig.add_trace(
            go.Bar(
                x=param_values,
                y=mean_scores,
                name=param.replace('param_', '')
            ),
            row=i+1,
            col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'Hyperparameter Tuning Results for {model_name}',
        height=200 * len(param_cols),
        showlegend=False
    )
    
    # Update y-axis labels
    for i in range(len(param_cols)):
        fig.update_yaxes(title_text='Mean Test Score', row=i+1, col=1)
    
    return fig

def display_model_summary(model, feature_names):
    """
    Display a summary of the model's parameters and characteristics.
    
    Parameters:
    -----------
    model : trained model
        Trained machine learning model
    feature_names : list
        List of feature names
    """
    # Extract model from pipeline if necessary
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        model_instance = model.named_steps['model']
    else:
        model_instance = model
    
    # Create a summary dictionary
    summary = {
        'Model Type': type(model_instance).__name__,
        'Parameters': dict(model_instance.get_params())
    }
    
    # Display as JSON
    st.json(summary)
    
    # Display feature importance if available
    if hasattr(model_instance, 'feature_importances_'):
        importances = model_instance.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Display as a table
        st.subheader("Feature Importance")
        st.dataframe(importance_df)
    
    # Add model-specific information
    if isinstance(model_instance, RandomForestClassifier):
        st.subheader("Random Forest Metrics")
        st.write(f"Number of trees: {model_instance.n_estimators}")
        st.write(f"Maximum tree depth: {model_instance.max_depth if model_instance.max_depth else 'None (unlimited)'}")
        st.write(f"Out-of-bag score: {model_instance.oob_score_:.4f}" if hasattr(model_instance, 'oob_score_') else "Out-of-bag score not computed")
    
    elif isinstance(model_instance, GradientBoostingClassifier):
        st.subheader("Gradient Boosting Metrics")
        st.write(f"Number of estimators: {model_instance.n_estimators}")
        st.write(f"Learning rate: {model_instance.learning_rate:.4f}")
        
        # Plot training deviance
        if hasattr(model_instance, 'train_score_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(np.arange(model_instance.n_estimators) + 1, model_instance.train_score_, 'b-', label='Training Set Deviance')
            ax.set_xlabel('Boosting Iterations')
            ax.set_ylabel('Deviance')
            ax.set_title('Training Deviance Over Boosting Iterations')
            ax.legend()
            st.pyplot(fig)

def run_advanced_model_evaluation(X, y, X_train, X_test, y_train, y_test, feature_names, class_names):
    """
    Run advanced model evaluation.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Complete feature matrix
    y : pandas.Series
        Complete target vector
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame
        Testing feature matrix
    y_train : pandas.Series
        Training target vector
    y_test : pandas.Series
        Testing target vector
    feature_names : list
        List of feature names
    class_names : list
        List of class names
    """
    st.header("Advanced Model Evaluation")
    
    # Create tabs for different analysis sections
    tabs = st.tabs([
        "Model Performance",
        "Learning Curves",
        "Hyperparameter Tuning",
        "Feature Importance"
    ])
    
    # Check if advanced models exist, otherwise train them
    model_path = "advanced_models.joblib"
    
    try:
        advanced_models = joblib.load(model_path)
        st.success("Loaded pre-trained advanced models.")
    except:
        st.info("Training advanced models with hyperparameter tuning. This may take a few minutes...")
        advanced_models, cv_results = train_advanced_models(X_train, y_train)
        
        # Save models for future use
        import joblib
        joblib.dump(advanced_models, model_path)
    
    # Select model for evaluation
    model_choice = st.selectbox(
        "Select model for evaluation:",
        list(advanced_models.keys())
    )
    
    selected_model = advanced_models[model_choice]
    
    # Tab 1: Model Performance
    with tabs[0]:
        st.subheader(f"{model_choice} Model Performance")
        
        # Display model summary
        st.write("### Model Summary")
        display_model_summary(selected_model, feature_names)
        
        # Evaluate model
        metrics = evaluate_model_performance(selected_model, X_test, y_test, class_names)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Precision", f"{metrics['precision']:.2%}")
        col3.metric("Recall", f"{metrics['recall']:.2%}")
        col4.metric("F1 Score", f"{metrics['f1']:.2%}")
        
        # Display confusion matrix
        st.write("### Confusion Matrix")
        conf_fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        st.plotly_chart(conf_fig, use_container_width=True)
        
        # Create tabs for ROC and PR curves
        curve_tabs = st.tabs(["ROC Curves", "Precision-Recall Curves"])
        
        with curve_tabs[0]:
            st.write("### ROC Curves (One-vs-Rest)")
            roc_fig = plot_roc_curves(
                metrics['roc_curves']['fpr'],
                metrics['roc_curves']['tpr'],
                metrics['roc_curves']['auc'],
                class_names
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        
        with curve_tabs[1]:
            st.write("### Precision-Recall Curves")
            pr_fig = plot_precision_recall_curves(
                metrics['pr_curves']['precision'],
                metrics['pr_curves']['recall'],
                metrics['pr_curves']['auc'],
                class_names
            )
            st.plotly_chart(pr_fig, use_container_width=True)
    
    # Tab 2: Learning Curves
    with tabs[1]:
        st.subheader("Learning Curves")
        st.write("""
        Learning curves show how the model's performance changes with increasing training data.
        A gap between training and validation scores indicates overfitting.
        """)
        
        # Plot learning curves
        learning_curve_fig = plot_learning_curves(selected_model, X, y)
        st.plotly_chart(learning_curve_fig, use_container_width=True)
        
        # Validation curves
        st.subheader("Validation Curves")
        st.write("""
        Validation curves show how the model's performance changes with different hyperparameter values.
        They help identify the optimal hyperparameter values.
        """)
        
        # Select hyperparameter to analyze
        if model_choice == "Random Forest":
            param_options = {
                "model__n_estimators": np.arange(10, 210, 20),
                "model__max_depth": [2, 4, 6, 8, 10, 15, 20, None],
                "model__min_samples_split": [2, 5, 10, 15, 20]
            }
        else:  # Gradient Boosting
            param_options = {
                "model__n_estimators": np.arange(10, 210, 20),
                "model__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5],
                "model__max_depth": [1, 2, 3, 5, 7, 10]
            }
        
        param_name = st.selectbox(
            "Select hyperparameter to analyze:",
            list(param_options.keys())
        )
        
        param_range = param_options[param_name]
        
        # Plot validation curve
        validation_curve_fig = plot_validation_curves(
            selected_model, X, y, param_name, param_range
        )
        st.plotly_chart(validation_curve_fig, use_container_width=True)
    
    # Tab 3: Hyperparameter Tuning
    with tabs[2]:
        st.subheader("Hyperparameter Tuning Results")
        
        if 'cv_results' in locals():
            # Plot hyperparameter tuning results
            tuning_fig = analyze_hyperparameter_tuning(cv_results[model_choice], model_choice)
            st.plotly_chart(tuning_fig, use_container_width=True)
            
            # Display best parameters
            st.write("### Best Parameters")
            if hasattr(selected_model, 'named_steps') and 'model' in selected_model.named_steps:
                best_params = {k.replace('model__', ''): v for k, v in selected_model.get_params().items() if k.startswith('model__')}
            else:
                best_params = selected_model.get_params()
            
            st.json(best_params)
        else:
            st.info("Hyperparameter tuning results not available. Re-run the model training to see these results.")
    
    # Tab 4: Feature Importance
    with tabs[3]:
        st.subheader("Feature Importance Analysis")
        
        # Permutation importance
        st.write("### Permutation Feature Importance")
        st.write("""
        Permutation importance measures how much the model's performance decreases when a feature is randomly shuffled.
        This is a more reliable measure of feature importance than built-in methods.
        """)
        
        perm_fig = plot_feature_permutation_importance(selected_model, X_test, y_test, feature_names)
        st.plotly_chart(perm_fig, use_container_width=True)
        
        # Feature importance from model
        if hasattr(selected_model, 'named_steps') and hasattr(selected_model.named_steps['model'], 'feature_importances_'):
            st.write("### Model-based Feature Importance")
            
            importances = selected_model.named_steps['model'].feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance (Model-based)',
                color='Importance',
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare both importance methods
            st.write("### Comparing Feature Importance Methods")
            st.write("""
            Comparing different feature importance methods can provide more robust insights.
            Consistent importance across methods indicates reliable feature relevance.
            """)
            
            # Calculate permutation importance
            from sklearn.inspection import permutation_importance
            perm_importance = permutation_importance(
                selected_model, X_test, y_test, n_repeats=5, random_state=42
            )
            
            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Feature': feature_names,
                'Model-based': importances,
                'Permutation-based': perm_importance.importances_mean
            })
            
            # Scale importances for comparison
            comparison_df['Model-based'] = comparison_df['Model-based'] / comparison_df['Model-based'].max()
            comparison_df['Permutation-based'] = comparison_df['Permutation-based'] / comparison_df['Permutation-based'].max()
            
            # Melt for plotting
            comparison_df_melted = comparison_df.melt(
                id_vars='Feature',
                var_name='Method',
                value_name='Importance'
            )
            
            # Plot comparison
            fig = px.bar(
                comparison_df_melted,
                x='Importance',
                y='Feature',
                color='Method',
                barmode='group',
                orientation='h',
                title='Feature Importance Methods Comparison (Normalized)',
                labels={'Importance': 'Normalized Importance'}
            )
            
            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)