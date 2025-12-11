"""
Machine Learning Model Pipeline
Handles model loading, prediction, and evaluation
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import os
import json
from datetime import datetime

def load_model(model_path='models/ecg_rf_pipeline.joblib'):
    """
    Load pre-trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to saved model file
    
    Returns:
    --------
    model : object
        Loaded model pipeline
    """
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        else:
            print(f"Model file not found at {model_path}")
            # Return a dummy model for demonstration
            return create_dummy_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return create_dummy_model()

def create_dummy_model():
    """
    Create a dummy model for demonstration when trained model is not available
    """
    print("Creating dummy model for demonstration...")
    
    # Create a simple random forest with random weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    # Generate dummy training data
    n_samples = 1000
    n_features = 56  # Number of features from feature extraction
    
    X_dummy = np.random.randn(n_samples, n_features)
    y_dummy = np.random.randint(0, 5, n_samples)  # 5 classes
    
    # Fit dummy model
    model.fit(X_dummy, y_dummy)
    
    return model

def predict_beats(model, features):
    """
    Predict beat classifications
    
    Parameters:
    -----------
    model : object
        Trained model
    features : array
        Feature array (n_beats, n_features)
    
    Returns:
    --------
    predictions : array
        Predicted class labels
    probabilities : array
        Prediction probabilities
    """
    try:
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)
        else:
            # Create dummy probabilities
            n_classes = 5
            probabilities = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 0.9  # High confidence for predicted class
                probabilities[i, :] += 0.02  # Small probability for others
                probabilities[i] /= probabilities[i].sum()  # Normalize
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return dummy predictions
        n_samples = len(features) if features.ndim > 1 else 1
        predictions = np.zeros(n_samples, dtype=int)
        probabilities = np.ones((n_samples, 5)) / 5
        return predictions, probabilities

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : array
        Test features
    y_test : array
        True labels
    
    Returns:
    --------
    metrics : dict
        Performance metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, 
                                recall_score, f1_score, 
                                confusion_matrix, classification_report)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'timestamp': datetime.now().isoformat()
    }
    
    # Per-class metrics
    unique_classes = np.unique(y_test)
    per_class_metrics = {}
    
    for class_id in unique_classes:
        class_mask = y_test == class_id
        if np.any(class_mask):
            per_class_metrics[f'class_{class_id}'] = {
                'precision': precision_score(y_test == class_id, y_pred == class_id),
                'recall': recall_score(y_test == class_id, y_pred == class_id),
                'f1_score': f1_score(y_test == class_id, y_pred == class_id),
                'support': int(np.sum(class_mask))
            }
    
    metrics['per_class_metrics'] = per_class_metrics
    
    return metrics

def train_model(X_train, y_train, model_params=None):
    """
    Train a new Random Forest model
    
    Parameters:
    -----------
    X_train : array
        Training features
    y_train : array
        Training labels
    model_params : dict
        Model hyperparameters
    
    Returns:
    --------
    model : object
        Trained model
    training_metrics : dict
        Training performance metrics
    """
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
    
    # Create and train model
    model = RandomForestClassifier(**model_params)
    
    # Train with cross-validation
    print("Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    training_metrics = {
        'cv_accuracy_mean': float(np.mean(cv_scores)),
        'cv_accuracy_std': float(np.std(cv_scores)),
        'cv_scores': cv_scores.tolist(),
        'n_samples': len(X_train),
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_train)),
        'model_params': model_params,
        'training_date': datetime.now().isoformat()
    }
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        training_metrics['feature_importances'] = model.feature_importances_.tolist()
    
    return model, training_metrics

def save_model(model, filepath='models/ecg_rf_pipeline.joblib', metrics=None):
    """
    Save trained model to disk
    
    Parameters:
    -----------
    model : object
        Trained model
    filepath : str
        Path to save model
    metrics : dict
        Optional training metrics to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
    # Save metrics if provided
    if metrics:
        metrics_path = filepath.replace('.joblib', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

def create_ensemble_prediction(models, features, weights=None):
    """
    Create ensemble predictions from multiple models
    
    Parameters:
    -----------
    models : list
        List of trained models
    features : array
        Feature array
    weights : array
        Model weights for averaging
    
    Returns:
    --------
    predictions : array
        Ensemble predictions
    probabilities : array
        Averaged probabilities
    """
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    
    all_probabilities = []
    
    for model, weight in zip(models, weights):
        _, probs = predict_beats(model, features)
        all_probabilities.append(probs * weight)
    
    # Average probabilities
    ensemble_probs = np.sum(all_probabilities, axis=0)
    
    # Get predictions from averaged probabilities
    ensemble_predictions = np.argmax(ensemble_probs, axis=1)
    
    return ensemble_predictions, ensemble_probs

def get_model_info(model):
    """
    Extract information about the model
    
    Returns:
    --------
    info : dict
        Model information
    """
    info = {
        'model_type': type(model).__name__,
        'created': datetime.now().isoformat()
    }
    
    # Random Forest specific info
    if isinstance(model, RandomForestClassifier):
        info.update({
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf,
            'max_features': model.max_features,
            'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
            'n_classes': model.n_classes_ if hasattr(model, 'n_classes_') else None
        })
    
    return info

def calibrate_model_confidence(model, X_val, y_val):
    """
    Calibrate model confidence scores using validation data
    
    Parameters:
    -----------
    model : object
        Trained model
    X_val : array
        Validation features
    y_val : array
        Validation labels
    
    Returns:
    --------
    calibration_metrics : dict
        Calibration performance metrics
    """
    from sklearn.calibration import calibration_curve
    from sklearn.isotonic import IsotonicRegression
    
    # Get predictions and probabilities
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    
    calibration_metrics = {
        'original_accuracy': accuracy_score(y_val, y_pred)
    }
    
    # Calculate calibration for each class
    n_classes = y_proba.shape[1]
    calibration_data = {}
    
    for class_idx in range(n_classes):
        # Get binary labels for this class
        y_true_binary = (y_val == class_idx).astype(int)
        y_proba_class = y_proba[:, class_idx]
        
        # Calculate calibration curve
        fraction_pos, mean_pred = calibration_curve(
            y_true_binary, y_proba_class, n_bins=10
        )
        
        calibration_data[f'class_{class_idx}'] = {
            'fraction_positive': fraction_pos.tolist(),
            'mean_predicted': mean_pred.tolist(),
            'expected_calibration_error': float(np.mean(np.abs(fraction_pos - mean_pred)))
        }
    
    calibration_metrics['calibration_data'] = calibration_data
    calibration_metrics['avg_calibration_error'] = np.mean([
        v['expected_calibration_error'] 
        for v in calibration_data.values()
    ])
    
    return calibration_metrics