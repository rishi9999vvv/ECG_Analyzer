import os
import sys
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from backend import utils, preprocess, features, model_pipeline, visualize_ecg

# MIT-BIH records to use for training
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

def load_mitbih_record(record_name, data_path):
    """
    Load a single MIT-BIH record with annotations
    
    Args:
        record_name: Record identifier (e.g., '100')
        data_path: Path to MIT-BIH database
    
    Returns:
        Dictionary with signal and annotations
    """
    try:
        record_path = os.path.join(data_path, record_name)
        
        # Load signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # Use first channel (MLII)
        fs = record.fs
        
        # Load annotations
        annotation = wfdb.rdann(record_path, 'atr')
        
        return {
            'signal': signal,
            'fs': fs,
            'ann_sample': annotation.sample,
            'ann_symbol': annotation.symbol,
            'record_name': record_name
        }
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None

def extract_features_from_record(record_data):
    """
    Extract features and labels from a single record
    
    Args:
        record_data: Dictionary with signal and annotations
    
    Returns:
        features_df: DataFrame with features
        labels: Array of labels
    """
    signal = record_data['signal']
    fs = record_data['fs']
    ann_samples = record_data['ann_sample']
    ann_symbols = record_data['ann_symbol']
    
    # Preprocess signal
    print(f"  Preprocessing record {record_data['record_name']}...")
    preprocessed = preprocess.preprocess_ecg(signal, fs)
    
    # Extract features
    print(f"  Extracting features...")
    features_df = features.extract_all_features(
        preprocessed['beats'],
        preprocessed['r_peaks'],
        fs
    )
    
    # Get labels for each beat
    labels = []
    valid_indices = preprocessed['valid_beat_indices']
    r_peaks = preprocessed['r_peaks']
    
    for idx in valid_indices:
        r_peak = r_peaks[idx]
        
        # Find closest annotation
        distances = np.abs(ann_samples - r_peak)
        closest_idx = np.argmin(distances)
        
        # Only use if annotation is close enough (within 50 samples)
        if distances[closest_idx] < 50:
            symbol = ann_symbols[closest_idx]
            label = utils.map_beat_annotation(symbol)
            labels.append(label)
        else:
            labels.append('Q')  # Unclassifiable
    
    # Ensure same length
    min_len = min(len(features_df), len(labels))
    features_df = features_df.iloc[:min_len]
    labels = labels[:min_len]
    
    return features_df, labels

def load_mitbih_dataset(record_list, data_path):
    """
    Load multiple MIT-BIH records and extract features
    
    Args:
        record_list: List of record identifiers
        data_path: Path to MIT-BIH database
    
    Returns:
        X: Feature matrix
        y: Labels
        metadata: Dataset metadata
    """
    all_features = []
    all_labels = []
    record_info = []
    
    print(f"Loading {len(record_list)} MIT-BIH records...")
    
    for i, record_name in enumerate(record_list):
        print(f"\nProcessing record {i+1}/{len(record_list)}: {record_name}")
        
        # Load record
        record_data = load_mitbih_record(record_name, data_path)
        if record_data is None:
            print(f"  Skipping record {record_name}")
            continue
        
        # Extract features
        try:
            features_df, labels = extract_features_from_record(record_data)
            
            if len(features_df) > 0:
                all_features.append(features_df)
                all_labels.extend(labels)
                record_info.append({
                    'record_name': record_name,
                    'n_beats': len(labels)
                })
                print(f"  Extracted {len(labels)} beats")
            else:
                print(f"  No valid beats extracted")
        except Exception as e:
            print(f"  Error processing record: {e}")
            continue
    
    # Combine all features
    if len(all_features) == 0:
        raise ValueError("No valid records processed!")
    
    X = pd.concat(all_features, ignore_index=True)
    y = np.array(all_labels)
    
    # Create metadata
    metadata = {
        'n_records': len(record_info),
        'n_beats': len(y),
        'records': record_info,
        'class_distribution': {cls: int(np.sum(y == cls)) for cls in np.unique(y)}
    }
    
    print(f"\nDataset loaded:")
    print(f"  Total beats: {len(y)}")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Class distribution: {metadata['class_distribution']}")
    
    return X, y, metadata

def train_and_evaluate(X, y, test_size=0.2, val_size=0.1):
    """
    Train and evaluate model
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Proportion for test set
        val_size: Proportion for validation set
    
    Returns:
        model: Trained model
        evaluation: Evaluation metrics
    """
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} beats")
    print(f"  Validation: {len(X_val)} beats")
    print(f"  Test: {len(X_test)} beats")
    
    # Train model
    print("\nTraining model...")
    model, history = model_pipeline.train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    print("\nEvaluating model...")
    class_names = sorted(list(set(y)))
    evaluation = model_pipeline.evaluate_model(model, X_test, y_test, class_names)
    
    print(f"\nTest Accuracy: {evaluation['accuracy']:.4f}")
    print("\nClassification Report:")
    for class_name in class_names:
        metrics = evaluation['classification_report'][class_name]
        print(f"  {Config.ARRHYTHMIA_CLASSES.get(class_name, class_name)}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1-Score: {metrics['f1-score']:.4f}")
    
    return model, evaluation, class_names

def save_training_report(evaluation, metadata, feature_names, output_path):
    """
    Save training report as JSON
    
    Args:
        evaluation: Evaluation metrics
        metadata: Dataset metadata
        feature_names: List of feature names
        output_path: Path to save report
    """
    report = {
        'dataset': metadata,
        'accuracy': float(evaluation['accuracy']),
        'classification_report': evaluation['classification_report'],
        'confusion_matrix': evaluation['confusion_matrix'].tolist(),
        'feature_names': feature_names
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nTraining report saved to {output_path}")

def main():
    """Main training script"""
    print("=" * 60)
    print("ECG Arrhythmia Classifier - Training Script")
    print("=" * 60)
    
    # Ensure directories exist
    utils.ensure_directories()
    
    # Check if MIT-BIH data exists
    if not os.path.exists(Config.MITBIH_PATH):
        print(f"\nERROR: MIT-BIH database not found at {Config.MITBIH_PATH}")
        print("Please download the MIT-BIH Arrhythmia Database and place it in the data/mit-bih/ folder")
        print("Download from: https://physionet.org/content/mitdb/1.0.0/")
        return
    
    # Load dataset
    try:
        X, y, metadata = load_mitbih_dataset(MITBIH_RECORDS, Config.MITBIH_PATH)
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        print("\nFor demo purposes, creating a synthetic dataset...")
        # Create synthetic data for demo
        np.random.seed(Config.RANDOM_STATE)
        n_samples = 1000
        n_features = len(features.get_feature_names())
        X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=features.get_feature_names())
        y = np.random.choice(['N', 'V', 'S', 'F', 'Q'], size=n_samples, p=[0.7, 0.15, 0.1, 0.03, 0.02])
        metadata = {'n_beats': n_samples, 'n_records': 0, 'class_distribution': {}}
        print("Synthetic dataset created for demo")
    
    # Train and evaluate
    model, evaluation, class_names = train_and_evaluate(X.values, y)
    
    # Save model
    model_pipeline.save_model(model, Config.MODEL_PATH)
    
    # Get feature importance
    feature_names = list(X.columns)
    feature_importance = model_pipeline.get_feature_importance(model, feature_names)
    
    # Save training report
    report_path = os.path.join(Config.REPORTS_PATH, 'training_report.json')
    save_training_report(evaluation, metadata, feature_names, report_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_ecg.plot_confusion_matrix_training(
        evaluation['confusion_matrix'],
        class_names,
        os.path.join(Config.REPORTS_PATH, 'confusion_matrix.png')
    )
    
    visualize_ecg.plot_feature_importance(
        feature_importance,
        os.path.join(Config.REPORTS_PATH, 'feature_importance.png')
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {Config.MODEL_PATH}")
    print(f"Reports saved to: {Config.REPORTS_PATH}")
    print("=" * 60)

if __name__ == '__main__':
    main()