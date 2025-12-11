import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from config import Config
from backend import utils, preprocess, features, model_pipeline

def compute_hrv_metrics(rr_intervals):
    """
    Compute Heart Rate Variability metrics
    
    Args:
        rr_intervals: Array of RR intervals in seconds
    
    Returns:
        Dictionary of HRV metrics
    """
    if len(rr_intervals) < 2:
        return {
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0,
            'mean_rr': 0,
            'std_rr': 0
        }
    
    # Convert to milliseconds
    rr_ms = rr_intervals * 1000
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_ms)
    
    # RMSSD: Root mean square of successive differences
    diff_rr = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    
    # pNN50: Percentage of successive RR intervals that differ by more than 50 ms
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    
    hrv_metrics = {
        'sdnn': float(sdnn),
        'rmssd': float(rmssd),
        'pnn50': float(pnn50),
        'mean_rr': float(np.mean(rr_ms)),
        'std_rr': float(np.std(rr_ms))
    }
    
    return hrv_metrics

def compute_arrhythmia_burden(predictions):
    """
    Compute arrhythmia burden (percentage of abnormal beats)
    
    Args:
        predictions: Array of predicted classes
    
    Returns:
        Dictionary with burden metrics
    """
    total_beats = len(predictions)
    
    if total_beats == 0:
        return {
            'pvc_burden': 0,
            'sve_burden': 0,
            'normal_percentage': 0,
            'abnormal_percentage': 0
        }
    
    # Count each class
    class_counts = {}
    for cls in ['N', 'V', 'S', 'F', 'Q']:
        class_counts[cls] = np.sum(predictions == cls)
    
    # Calculate percentages
    pvc_burden = (class_counts['V'] / total_beats) * 100
    sve_burden = (class_counts['S'] / total_beats) * 100
    normal_percentage = (class_counts['N'] / total_beats) * 100
    abnormal_percentage = 100 - normal_percentage
    
    burden = {
        'pvc_burden': float(pvc_burden),
        'sve_burden': float(sve_burden),
        'fusion_percentage': float((class_counts['F'] / total_beats) * 100),
        'normal_percentage': float(normal_percentage),
        'abnormal_percentage': float(abnormal_percentage),
        'class_counts': {k: int(v) for k, v in class_counts.items()}
    }
    
    return burden

def detect_tachy_brady_episodes(heart_rates, threshold_tachy=100, threshold_brady=60):
    """
    Detect tachycardia and bradycardia episodes
    
    Args:
        heart_rates: Array of instantaneous heart rates
        threshold_tachy: Tachycardia threshold (bpm)
        threshold_brady: Bradycardia threshold (bpm)
    
    Returns:
        Dictionary with episode information
    """
    if len(heart_rates) == 0:
        return {
            'tachy_episodes': 0,
            'brady_episodes': 0,
            'tachy_duration': 0,
            'brady_duration': 0
        }
    
    # Find episodes (consecutive beats above/below threshold)
    is_tachy = heart_rates > threshold_tachy
    is_brady = heart_rates < threshold_brady
    
    # Count transitions to find episodes
    tachy_episodes = np.sum(np.diff(is_tachy.astype(int)) == 1)
    brady_episodes = np.sum(np.diff(is_brady.astype(int)) == 1)
    
    # Duration (number of beats)
    tachy_duration = np.sum(is_tachy)
    brady_duration = np.sum(is_brady)
    
    episodes = {
        'tachy_episodes': int(tachy_episodes),
        'brady_episodes': int(brady_episodes),
        'tachy_duration': int(tachy_duration),
        'brady_duration': int(brady_duration),
        'max_hr': float(np.max(heart_rates)),
        'min_hr': float(np.min(heart_rates))
    }
    
    return episodes

def compute_qrs_metrics(beats, fs):
    """
    Compute QRS complex metrics
    
    Args:
        beats: Array of beat segments
        fs: Sampling frequency
    
    Returns:
        Dictionary with QRS metrics
    """
    if len(beats) == 0:
        return {
            'mean_qrs_width': 0,
            'std_qrs_width': 0
        }
    
    qrs_widths = []
    
    for beat in beats:
        # Find R-peak
        r_peak_idx = np.argmax(beat)
        r_peak_amp = beat[r_peak_idx]
        
        # Find QRS boundaries (threshold method)
        threshold = 0.3 * r_peak_amp
        above_threshold = np.abs(beat) > threshold
        
        if np.any(above_threshold):
            qrs_start = np.where(above_threshold)[0][0]
            qrs_end = np.where(above_threshold)[0][-1]
            width_samples = qrs_end - qrs_start
            width_ms = (width_samples / fs) * 1000
            qrs_widths.append(width_ms)
    
    if len(qrs_widths) == 0:
        return {
            'mean_qrs_width': 0,
            'std_qrs_width': 0
        }
    
    metrics = {
        'mean_qrs_width': float(np.mean(qrs_widths)),
        'std_qrs_width': float(np.std(qrs_widths))
    }
    
    return metrics

def predict_ecg(signal, fs, model, metadata=None):
    """
    Main inference function - predict arrhythmias on ECG signal
    
    Args:
        signal: ECG signal array
        fs: Sampling frequency
        model: Trained model
        metadata: Optional metadata about the signal
    
    Returns:
        Dictionary with predictions and analysis
    """
    print("Starting ECG analysis...")
    
    # Preprocess signal
    print("Preprocessing signal...")
    preprocessed = preprocess.preprocess_ecg(signal, fs)
    
    # Extract features
    print("Extracting features...")
    features_df = features.extract_all_features(
        preprocessed['beats'],
        preprocessed['r_peaks'],
        fs
    )
    
    if len(features_df) == 0:
        return {
            'error': 'No valid beats detected in signal',
            'beats': [],
            'summary': {}
        }
    
    # Predict
    print("Running inference...")
    predictions, confidences, probabilities = model_pipeline.predict_with_confidence(
        model,
        features_df.values
    )
    
    # Compute heart rates
    rr_intervals = preprocessed['rr_intervals']
    heart_rates = preprocess.compute_heart_rate(rr_intervals)
    
    # Prepare per-beat results
    beats_data = []
    r_peaks = preprocessed['r_peaks']
    valid_indices = preprocessed['valid_beat_indices']
    
    for i in range(len(predictions)):
        beat_idx = valid_indices[i]
        time_sec = r_peaks[beat_idx] / fs
        
        # Get RR interval and HR for this beat
        if i < len(rr_intervals):
            rr = rr_intervals[i]
            hr = heart_rates[i]
        else:
            rr = rr_intervals[-1] if len(rr_intervals) > 0 else 0
            hr = heart_rates[-1] if len(heart_rates) > 0 else 0
        
        beat_info = {
            'index': int(beat_idx),
            'time': float(time_sec),
            'class': predictions[i],
            'class_name': Config.ARRHYTHMIA_CLASSES.get(predictions[i], 'Unknown'),
            'confidence': float(confidences[i]),
            'rr_interval': float(rr),
            'heart_rate': float(hr),
            'probabilities': {
                cls: float(probabilities[i][j]) 
                for j, cls in enumerate(model.classes_)
            }
        }
        beats_data.append(beat_info)
    
    # Compute HRV metrics
    print("Computing HRV metrics...")
    hrv_metrics = compute_hrv_metrics(rr_intervals)
    
    # Compute arrhythmia burden
    print("Computing arrhythmia burden...")
    burden = compute_arrhythmia_burden(predictions)
    
    # Detect tachycardia/bradycardia episodes
    print("Detecting tachy/brady episodes...")
    episodes = detect_tachy_brady_episodes(heart_rates)
    
    # Compute QRS metrics
    print("Computing QRS metrics...")
    qrs_metrics = compute_qrs_metrics(preprocessed['beats'], fs)
    
    # Compute signal quality
    signal_quality = utils.compute_signal_quality(signal, fs)
    
    # Compute overall statistics
    mean_hr = np.mean(heart_rates) if len(heart_rates) > 0 else 0
    
    # Summary metrics
    summary = {
        'total_beats': len(predictions),
        'duration_seconds': len(signal) / fs,
        'mean_hr': float(mean_hr),
        'signal_quality': float(signal_quality),
        'signal_quality_label': 'Excellent' if signal_quality > 80 else 'Good' if signal_quality > 60 else 'Fair' if signal_quality > 40 else 'Poor',
        **hrv_metrics,
        **burden,
        **episodes,
        **qrs_metrics
    }
    
    # Prepare result
    result = {
        'beats': beats_data,
        'summary': summary,
        'metadata': {
            'sampling_frequency': fs,
            'signal_length': len(signal),
            'n_channels': signal.shape[1] if signal.ndim > 1 else 1,
            'analysis_timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
    }
    
    print("Analysis complete!")
    return result

def generate_report(result, output_path):
    """
    Generate and save analysis report as JSON
    
    Args:
        result: Analysis result dictionary
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Report saved to {output_path}")

def analyze_uploaded_file(filepath, model_path=None, fs=None):
    """
    Analyze an uploaded ECG file
    
    Args:
        filepath: Path to uploaded .dat file
        model_path: Path to trained model
        fs: Sampling frequency (if header not available)
    
    Returns:
        Analysis result dictionary
    """
    # Load model
    if model_path is None:
        model_path = Config.MODEL_PATH
    
    if not os.path.exists(model_path):
        return {
            'error': f'Model not found at {model_path}. Please train the model first.',
            'beats': [],
            'summary': {}
        }
    
    model = model_pipeline.load_model(model_path)
    
    # Load signal
    signal, signal_fs, metadata = utils.load_ecg_signal(filepath, fs=fs)
    
    if signal is None:
        return {
            'error': 'Failed to load ECG signal',
            'beats': [],
            'summary': {}
        }
    
    # Use first channel if multi-channel
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    # Run analysis
    result = predict_ecg(signal, signal_fs, model, metadata)
    
    return result

def get_beat_annotations(result):
    """
    Extract beat annotations for visualization
    
    Args:
        result: Analysis result dictionary
    
    Returns:
        Dictionary with annotations by class
    """
    annotations = {
        'N': [], 'V': [], 'S': [], 'F': [], 'Q': []
    }
    
    for beat in result['beats']:
        cls = beat['class']
        annotations[cls].append({
            'time': beat['time'],
            'index': beat['index'],
            'hr': beat['heart_rate'],
            'confidence': beat['confidence']
        })
    
    return annotations

def get_summary_statistics(result):
    """
    Get formatted summary statistics for display
    
    Args:
        result: Analysis result dictionary
    
    Returns:
        List of formatted statistics
    """
    summary = result['summary']
    
    stats = [
        {
            'label': 'Total Beats',
            'value': summary['total_beats'],
            'unit': 'beats',
            'icon': 'heart'
        },
        {
            'label': 'Mean Heart Rate',
            'value': f"{summary['mean_hr']:.1f}",
            'unit': 'bpm',
            'icon': 'activity'
        },
        {
            'label': 'SDNN',
            'value': f"{summary['sdnn']:.1f}",
            'unit': 'ms',
            'icon': 'trending-up'
        },
        {
            'label': 'RMSSD',
            'value': f"{summary['rmssd']:.1f}",
            'unit': 'ms',
            'icon': 'bar-chart'
        },
        {
            'label': 'pNN50',
            'value': f"{summary['pnn50']:.1f}",
            'unit': '%',
            'icon': 'percent'
        },
        {
            'label': 'PVC Burden',
            'value': f"{summary['pvc_burden']:.1f}",
            'unit': '%',
            'icon': 'alert-circle',
            'alert': summary['pvc_burden'] > 5
        },
        {
            'label': 'SVE Burden',
            'value': f"{summary['sve_burden']:.1f}",
            'unit': '%',
            'icon': 'alert-triangle',
            'alert': summary['sve_burden'] > 5
        },
        {
            'label': 'Signal Quality',
            'value': summary['signal_quality_label'],
            'unit': f"({summary['signal_quality']:.0f}/100)",
            'icon': 'check-circle'
        }
    ]
    
    return stats

def get_clinical_interpretation(result):
    """
    Generate clinical interpretation of results
    
    Args:
        result: Analysis result dictionary
    
    Returns:
        Dictionary with interpretation
    """
    summary = result['summary']
    
    findings = []
    recommendations = []
    
    # Heart rate interpretation
    mean_hr = summary['mean_hr']
    if mean_hr < 60:
        findings.append(f"Bradycardia detected (mean HR: {mean_hr:.0f} bpm)")
        recommendations.append("Consider clinical correlation for symptomatic bradycardia")
    elif mean_hr > 100:
        findings.append(f"Tachycardia detected (mean HR: {mean_hr:.0f} bpm)")
        recommendations.append("Evaluate for underlying causes of tachycardia")
    else:
        findings.append(f"Normal heart rate (mean HR: {mean_hr:.0f} bpm)")
    
    # PVC burden
    if summary['pvc_burden'] > 10:
        findings.append(f"Significant PVC burden ({summary['pvc_burden']:.1f}%)")
        recommendations.append("Consider Holter monitoring and cardiac workup")
    elif summary['pvc_burden'] > 1:
        findings.append(f"Occasional PVCs detected ({summary['pvc_burden']:.1f}%)")
    
    # SVE burden
    if summary['sve_burden'] > 5:
        findings.append(f"Frequent supraventricular ectopics ({summary['sve_burden']:.1f}%)")
        recommendations.append("Consider evaluation for atrial arrhythmias")
    
    # HRV interpretation
    if summary['sdnn'] < 50:
        findings.append(f"Reduced heart rate variability (SDNN: {summary['sdnn']:.0f} ms)")
        recommendations.append("Low HRV may indicate autonomic dysfunction")
    elif summary['sdnn'] > 100:
        findings.append(f"High heart rate variability (SDNN: {summary['sdnn']:.0f} ms)")
    
    # Signal quality
    if summary['signal_quality'] < 60:
        findings.append(f"Poor signal quality ({summary['signal_quality']:.0f}/100)")
        recommendations.append("Consider repeat recording with better electrode contact")
    
    # Normal findings
    if len(findings) == 1 and "Normal heart rate" in findings[0]:
        findings.append("No significant arrhythmias detected")
        findings.append("Regular rhythm")
    
    interpretation = {
        'findings': findings,
        'recommendations': recommendations if len(recommendations) > 0 else ["Routine follow-up as clinically indicated"],
        'severity': 'high' if summary['pvc_burden'] > 10 or summary['sve_burden'] > 10 else 'moderate' if summary['pvc_burden'] > 5 or summary['sve_burden'] > 5 else 'low'
    }
    
    return interpretation