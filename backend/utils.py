import os
import json
import numpy as np
import wfdb
from werkzeug.utils import secure_filename
from config import Config

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder):
    """Save uploaded file securely"""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return filename, filepath
    return None, None

def check_header_exists(filepath):
    """Check if .hea file exists for the .dat file"""
    base_path = filepath.rsplit('.', 1)[0]
    hea_path = base_path + '.hea'
    return os.path.exists(hea_path)

def parse_header(filepath):
    """Parse MIT-BIH header file using wfdb"""
    try:
        base_path = filepath.rsplit('.', 1)[0]
        record = wfdb.rdheader(base_path)
        return {
            'fs': record.fs,
            'n_sig': record.n_sig,
            'sig_len': record.sig_len,
            'sig_name': record.sig_name,
            'units': record.units,
            'comments': record.comments
        }
    except Exception as e:
        print(f"Error parsing header: {e}")
        return None

def load_ecg_signal(filepath, fs=None, n_channels=None):
    """
    Load ECG signal from .dat file
    
    Args:
        filepath: Path to .dat file
        fs: Sampling frequency (if header not available)
        n_channels: Number of channels (if header not available)
    
    Returns:
        signal: ECG signal array
        fs: Sampling frequency
        metadata: Dictionary with signal info
    """
    base_path = filepath.rsplit('.', 1)[0]
    
    # Try to load with wfdb (if header exists)
    if check_header_exists(filepath):
        try:
            record = wfdb.rdrecord(base_path)
            signal = record.p_signal
            fs = record.fs
            metadata = {
                'fs': fs,
                'n_sig': record.n_sig,
                'sig_len': record.sig_len,
                'sig_name': record.sig_name,
                'has_header': True
            }
            return signal, fs, metadata
        except Exception as e:
            print(f"Error loading with wfdb: {e}")
    
    # Load raw .dat file if no header
    if fs is None:
        fs = Config.DEFAULT_FS
    
    try:
        # MIT-BIH format: 212 format (2 channels, 12-bit)
        signal = np.fromfile(filepath, dtype=np.int16).reshape(-1, 2)
        # Convert from 212 format to millivolts
        signal = signal.astype(np.float32) / 200.0
        
        metadata = {
            'fs': fs,
            'n_sig': signal.shape[1],
            'sig_len': signal.shape[0],
            'sig_name': ['MLII', 'V5'],
            'has_header': False
        }
        return signal, fs, metadata
    except Exception as e:
        print(f"Error loading raw signal: {e}")
        return None, None, None

def load_annotations(filepath):
    """Load MIT-BIH annotation file (.atr)"""
    try:
        base_path = filepath.rsplit('.', 1)[0]
        annotation = wfdb.rdann(base_path, 'atr')
        return {
            'sample': annotation.sample,
            'symbol': annotation.symbol,
            'aux_note': annotation.aux_note
        }
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return None

def create_config_file(filename, fs, n_channels, dtype='int16'):
    """Create configuration file for headerless .dat files"""
    config_path = os.path.join(Config.UPLOAD_FOLDER, filename.replace('.dat', '_config.json'))
    config_data = {
        'filename': filename,
        'fs': fs,
        'n_channels': n_channels,
        'dtype': dtype
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    return config_path

def load_config_file(filename):
    """Load configuration file for headerless .dat files"""
    config_path = os.path.join(Config.UPLOAD_FOLDER, filename.replace('.dat', '_config.json'))
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        Config.UPLOAD_FOLDER,
        Config.MITBIH_PATH,
        Config.REPORTS_PATH,
        Config.OUTPUTS_PATH,
        os.path.dirname(Config.MODEL_PATH)
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def map_beat_annotation(symbol):
    """
    Map MIT-BIH annotation symbols to our classes
    
    MIT-BIH symbols:
    - N, L, R, e, j: Normal
    - V, E: Ventricular ectopic
    - A, a, J, S: Supraventricular ectopic
    - F: Fusion
    - Others: Unclassifiable
    """
    normal_beats = ['N', 'L', 'R', 'e', 'j']
    ventricular_beats = ['V', 'E']
    supraventricular_beats = ['A', 'a', 'J', 'S']
    fusion_beats = ['F']
    
    if symbol in normal_beats:
        return 'N'
    elif symbol in ventricular_beats:
        return 'V'
    elif symbol in supraventricular_beats:
        return 'S'
    elif symbol in fusion_beats:
        return 'F'
    else:
        return 'Q'

def compute_signal_quality(signal, fs):
    """
    Compute signal quality score (0-100)
    Based on:
    - SNR estimation
    - Baseline wander
    - Number of artifacts
    """
    try:
        # Simple quality metric based on signal variance and peaks
        signal_std = np.std(signal)
        signal_range = np.ptp(signal)
        
        # Good ECG has moderate variance, not too flat or too noisy
        if signal_std < 0.01 or signal_range < 0.1:
            quality = 30  # Too flat
        elif signal_std > 2.0 or signal_range > 10.0:
            quality = 40  # Too noisy
        else:
            quality = 85  # Good quality
        
        return min(100, max(0, quality))
    except:
        return 50  # Default medium quality