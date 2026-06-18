import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path, test_size=0.2, random_state=42):
    """
    Load the crop recommendation dataset and preprocess it for analysis and modeling.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the crop recommendation data
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split
        
    Returns:
    --------
    df : pandas.DataFrame
        The complete dataset
    X : pandas.DataFrame
        Feature matrix (without target variable)
    y : pandas.Series
        Target vector
    X_train, X_test : pandas.DataFrame
        Training and testing feature matrices
    y_train, y_test : pandas.Series
        Training and testing target vectors
    feature_names : list
        List of feature names
    target_names : list
        List of unique crop types (target classes)
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check for missing values
    if df.isnull().any().any():
        # Handle missing values if any (can be extended based on requirements)
        df = df.dropna()
    
    # Check for duplicate rows
    df = df.drop_duplicates()
    
    # Extract features and target
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_names]
    y = df['label']
    
    # Get unique target classes
    target_names = sorted(y.unique())
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features (but return the original dataframe for EDA)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return df, X, y, X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_names

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers in a specific column of the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name to check for outliers
    method : str, default='iqr'
        Method to use for outlier detection ('iqr' or 'zscore')
    threshold : float, default=1.5
        Threshold for outlier detection
        
    Returns:
    --------
    pandas.Series
        Boolean mask where True indicates an outlier
    """
    if method == 'iqr':
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        return z_scores > threshold
    else:
        raise ValueError("Method must be either 'iqr' or 'zscore'")
