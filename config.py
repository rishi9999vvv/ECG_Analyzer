import os

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'ecg-analyzer-secret-key-2024'
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'data', 'user_inputs')
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'ecg_rf_pipeline.joblib')
    MITBIH_PATH = os.path.join(os.path.dirname(__file__), 'data', 'mit-bih')
    REPORTS_PATH = os.path.join(os.path.dirname(__file__), 'reports')
    OUTPUTS_PATH = os.path.join(os.path.dirname(__file__), 'outputs')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'dat', 'hea', 'atr'}
    
    # Signal processing parameters
    DEFAULT_FS = 360  # Default sampling frequency for MIT-BIH
    BANDPASS_LOW = 0.5
    BANDPASS_HIGH = 50.0
    NOTCH_FREQ = 50.0  # Powerline frequency
    
    # Model parameters
    N_ESTIMATORS = 200
    MAX_DEPTH = 20
    RANDOM_STATE = 42
    
    # Arrhythmia classes
    ARRHYTHMIA_CLASSES = {
        'N': 'Normal',
        'V': 'Ventricular Ectopic (PVC)',
        'S': 'Supraventricular Ectopic (SVE)',
        'F': 'Fusion',
        'Q': 'Unclassifiable'
    }
    
    # Color scheme for plots
    CLASS_COLORS = {
        'N': '#10b981',  # Green
        'V': '#ef4444',  # Red
        'S': '#f59e0b',  # Orange
        'F': '#8b5cf6',  # Purple
        'Q': '#6b7280'   # Gray
    }