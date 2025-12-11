"""
ECG Analyzer Backend Package
Advanced arrhythmia detection and analysis system
"""

# Package metadata
__version__ = "2.0.0"
__author__ = "ECG Analyzer Team"
__description__ = "Advanced ECG arrhythmia classification system with real-time analysis"

# Import main modules
from .preprocess import (
    preprocess_signal,
    segment_beats,
    adaptive_r_peak_detection,
    calculate_signal_quality_index
)

from .features import (
    extract_features,
    extract_morphological_features,
    extract_statistical_features,
    extract_wavelet_features,
    extract_frequency_features,
    get_feature_names
)

from .model_pipeline import (
    load_model,
    predict_beats,
    evaluate_model,
    train_model,
    save_model,
    get_model_info
)

from .visualize_ecg import (
    create_ecg_plot,
    create_distribution_chart,
    create_hrv_plot,
    create_rhythm_strip
)

from .clinical_analysis import (
    generate_clinical_report,
    analyze_rhythm,
    analyze_arrhythmia_burden,
    analyze_hrv,
    calculate_arrhythmia_statistics,
    assess_clinical_urgency
)

# Module-level configuration
CONFIG = {
    'sampling_rate': 360,  # Default sampling rate (Hz)
    'beat_window': {
        'before': 0.2,  # Seconds before R-peak
        'after': 0.4    # Seconds after R-peak
    },
    'filtering': {
        'baseline_cutoff': 0.5,  # Hz
        'powerline_freq': [50, 60],  # Hz
        'high_freq_cutoff': 40  # Hz
    },
    'classification': {
        'classes': {
            0: 'Normal',
            1: 'PVC',
            2: 'SVE',
            3: 'Fusion',
            4: 'Unclassifiable'
        },
        'confidence_threshold': 0.85
    },
    'clinical': {
        'normal_hr_range': (60, 100),  # BPM
        'pvc_burden_thresholds': {
            'low': 5,
            'moderate': 10,
            'high': 20
        },
        'hrv_normal_ranges': {
            'sdnn': (50, 100),  # ms
            'rmssd': (20, 50),  # ms
            'pnn50': (5, 25)    # %
        }
    }
}

# Initialize module components
def initialize():
    """
    Initialize backend components
    """
    print(f"ECG Analyzer Backend v{__version__}")
    print(f"Sampling Rate: {CONFIG['sampling_rate']} Hz")
    print(f"Classification Classes: {list(CONFIG['classification']['classes'].values())}")
    return True

# Utility functions
def get_config():
    """Get current configuration"""
    return CONFIG.copy()

def update_config(new_config):
    """Update configuration settings"""
    CONFIG.update(new_config)
    return CONFIG

def get_version():
    """Get package version"""
    return __version__

# Export all
__all__ = [
    # Preprocessing
    'preprocess_signal',
    'segment_beats',
    'adaptive_r_peak_detection',
    'calculate_signal_quality_index',
    
    # Features
    'extract_features',
    'extract_morphological_features',
    'extract_statistical_features',
    'extract_wavelet_features',
    'extract_frequency_features',
    'get_feature_names',
    
    # Model
    'load_model',
    'predict_beats',
    'evaluate_model',
    'train_model',
    'save_model',
    'get_model_info',
    
    # Visualization
    'create_ecg_plot',
    'create_distribution_chart',
    'create_hrv_plot',
    'create_rhythm_strip',
    
    # Clinical
    'generate_clinical_report',
    'analyze_rhythm',
    'analyze_arrhythmia_burden',
    'analyze_hrv',
    'calculate_arrhythmia_statistics',
    'assess_clinical_urgency',
    
    # Configuration
    'CONFIG',
    'initialize',
    'get_config',
    'update_config',
    'get_version'
]