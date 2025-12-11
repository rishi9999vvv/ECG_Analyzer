"""
Signal Preprocessing Module
Handles ECG signal filtering, denoising, and segmentation
"""

import numpy as np
import scipy.signal as signal
from scipy.signal import butter, filtfilt, find_peaks, iirnotch, detrend
import pywt

def preprocess_signal(ecg_signal, fs=360):
    """
    Complete preprocessing pipeline for ECG signals
    
    Parameters:
    -----------
    ecg_signal : array-like
        Raw ECG signal
    fs : int
        Sampling frequency (Hz)
    
    Returns:
    --------
    filtered_signal : array
        Preprocessed ECG signal
    """
    # Remove NaN and infinite values
    ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 1. Baseline wander removal (high-pass filter at 0.5 Hz)
    filtered = remove_baseline_wander(ecg_signal, fs)
    
    # 2. Powerline interference removal (notch filter at 50/60 Hz)
    filtered = remove_powerline_interference(filtered, fs)
    
    # 3. High-frequency noise removal (low-pass filter at 40 Hz)
    filtered = remove_high_frequency_noise(filtered, fs)
    
    # 4. Normalize signal
    filtered = normalize_signal(filtered)
    
    # 5. Additional wavelet denoising (optional)
    if len(filtered) > 1000:  # Only apply for longer signals
        filtered = wavelet_denoise(filtered)
    
    return filtered

def preprocess_ecg_fixed(ecg_signal, fs=360, use_wfdb_gain=False, wfdb_record=None):
    """
    Fixed preprocessing pipeline for ECG signals using the proven approach:
    1. Bandpass filter (0.5-45 Hz, 4th order) - removes baseline wander and high-frequency noise
    2. Notch filter (50 Hz, Q=30) - removes powerline interference
    3. Detrend - removes linear trend
    4. Z-score normalization (signal - mean) / std
    
    This approach produces clean, properly scaled ECG signals without distortion.
    
    Parameters:
    -----------
    ecg_signal : array-like
        Raw ECG signal (1D array)
    fs : int
        Sampling frequency (Hz)
    use_wfdb_gain : bool
        If True and wfdb_record provided, use physical units from record
    wfdb_record : wfdb.Record
        WFDB record object (optional, for gain correction)
    
    Returns:
    --------
    filtered_signal : array
        Preprocessed ECG signal
    """
    from scipy.signal import butter, filtfilt, iirnotch, detrend
    
    # Remove NaN and infinite values
    ecg_signal = np.array(ecg_signal, dtype=np.float64)
    
    # Handle NaN values
    if np.isnan(ecg_signal).sum() > 0:
        ecg_signal = np.nan_to_num(ecg_signal, nan=np.nanmean(ecg_signal))
    ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If signal is 2D, take first channel (MLII)
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[:, 0]
    
    # Apply WFDB gain correction if available
    if use_wfdb_gain and wfdb_record is not None:
        try:
            # Get gain from record
            if hasattr(wfdb_record, 'adc_gain'):
                gain = wfdb_record.adc_gain[0] if isinstance(wfdb_record.adc_gain, (list, np.ndarray)) else wfdb_record.adc_gain
            elif hasattr(wfdb_record, 'adc'):
                gain = wfdb_record.adc[0] if isinstance(wfdb_record.adc, (list, np.ndarray)) else wfdb_record.adc
            else:
                gain = 1.0
            
            if gain > 0:
                ecg_signal = ecg_signal / gain
        except Exception as e:
            print(f"Warning: Could not apply WFDB gain correction: {e}")
    
    # Step 1: Bandpass filter (0.5-45 Hz, 4th order) - removes baseline wander and high-frequency noise
    # This is the proven approach that produces clean ECG signals
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    
    if low < 1.0 and high < 1.0:
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)  # filtfilt for forward-backward filtering (zero phase shift)
    else:
        print(f"Warning: Filter cutoff too high. Using original signal.")
        filtered_signal = ecg_signal.copy()
    
    # Step 2: Notch filter to remove powerline interference (50 Hz, Q=30)
    # This removes 50 Hz powerline interference (European standard)
    try:
        b_notch, a_notch = iirnotch(50.0 / (fs / 2), Q=30)
        notched_signal = filtfilt(b_notch, a_notch, filtered_signal)
    except Exception as e:
        print(f"Warning: Notch filter failed: {e}. Using bandpass filtered signal.")
        notched_signal = filtered_signal
    
    # Step 3: Detrend - removes linear trend
    detrended_signal = detrend(notched_signal)
    
    # Step 4: Z-score normalization (proper normalization)
    # (signal - mean) / std
    # Note: For visualization, you might want to skip normalization
    # to preserve the original amplitude scale
    mean_val = np.mean(detrended_signal)
    std_val = np.std(detrended_signal)
    
    if std_val > 1e-6:  # Avoid division by zero
        filtered = (detrended_signal - mean_val) / std_val
    else:
        print(f"Warning: Signal has zero variance. Skipping normalization.")
        filtered = detrended_signal - mean_val  # At least center the signal
    
    return filtered

def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=360, order=4):
    """
    Bandpass filter function - EXACT copy of user's working code
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq=50.0, fs=360, Q=30):
    """
    Notch filter function - EXACT copy of user's working code
    """
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, data)

def preprocess_ecg_for_visualization(ecg_signal, fs=360, use_wfdb_gain=False, wfdb_record=None):
    """
    Preprocessing pipeline for ECG signals - EXACT copy of user's working approach
    This matches the user's working code EXACTLY:
    1. bandpass_filter(signal, 0.5, 45, fs)
    2. notch_filter(filtered_signal, freq=50, fs=fs)
    3. detrend(notched_signal)
    
    Does NOT normalize - preserves original amplitude scale for clean, smooth visualization
    
    IMPORTANT: record.p_signal is already in physical units (mV), so normally we should NOT
    apply gain correction. Only convert if signal appears to be in ADC units (very large values).
    
    Parameters:
    -----------
    ecg_signal : array-like
        Raw ECG signal (1D array) - should be in physical units (mV) from p_signal
    fs : int
        Sampling frequency (Hz)
    use_wfdb_gain : bool
        If True and signal appears to be in ADC units, convert to physical units
    wfdb_record : wfdb.Record
        WFDB record object (optional, for gain correction if needed)
    
    Returns:
    --------
    filtered_signal : array
        Preprocessed ECG signal (NOT normalized - preserves amplitude)
    """
    from scipy.signal import detrend
    
    # Remove NaN and infinite values
    ecg_signal = np.array(ecg_signal, dtype=np.float64)
    
    # Handle NaN values exactly like user's code
    if np.isnan(ecg_signal).sum() > 0:
        ecg_signal = np.nan_to_num(ecg_signal, nan=np.nanmean(ecg_signal))
    ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If signal is 2D, take first channel (MLII)
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[:, 0]
    
    # Check if signal needs scaling (very large values indicate ADC units or scaling issue)
    max_abs = np.max(np.abs(ecg_signal))
    if max_abs > 1000:
        print(f"  ⚠️ Signal has very large values (max={max_abs:.0f}), attempting to scale...")
        
        # Try to convert from ADC units to physical units if WFDB record is available
        if use_wfdb_gain and wfdb_record is not None:
            try:
                # Get gain and baseline from record
                if hasattr(wfdb_record, 'adc_gain'):
                    gain = wfdb_record.adc_gain[0] if isinstance(wfdb_record.adc_gain, (list, np.ndarray)) else wfdb_record.adc_gain
                else:
                    gain = 200.0  # Default for MIT-BIH (200 ADC units per mV)
                
                if hasattr(wfdb_record, 'baseline'):
                    baseline = wfdb_record.baseline[0] if isinstance(wfdb_record.baseline, (list, np.ndarray)) else wfdb_record.baseline
                else:
                    baseline = 0.0
                
                if gain > 0:
                    # Convert from ADC units to physical units: (signal - baseline) / gain
                    ecg_signal = (ecg_signal - baseline) / gain
                    print(f"     ✓ Converted using WFDB gain: range={np.min(ecg_signal):.2f} to {np.max(ecg_signal):.2f} mV")
                else:
                    # Fallback: estimate scale factor (typical ECG is -5 to +5 mV)
                    # If signal is 30,000, we need to divide by ~6000 to get ~5 mV
                    scale_factor = max_abs / 5.0  # Scale to typical max of 5 mV
                    ecg_signal = ecg_signal / scale_factor
                    print(f"     ✓ Scaled by estimated factor ({scale_factor:.0f}): range={np.min(ecg_signal):.2f} to {np.max(ecg_signal):.2f} mV")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not convert using WFDB gain: {e}")
                # Fallback: estimate scale factor
                scale_factor = max_abs / 5.0  # Scale to typical max of 5 mV
                ecg_signal = ecg_signal / scale_factor
                print(f"     ✓ Scaled by estimated factor ({scale_factor:.0f}): range={np.min(ecg_signal):.2f} to {np.max(ecg_signal):.2f} mV")
        else:
            # No WFDB record available - use estimated scaling
            # Typical ECG is -5 to +5 mV, so if signal is 30,000, divide by ~6000
            scale_factor = max_abs / 5.0  # Scale to typical max of 5 mV
            ecg_signal = ecg_signal / scale_factor
            print(f"     ✓ Scaled by estimated factor ({scale_factor:.0f}): range={np.min(ecg_signal):.2f} to {np.max(ecg_signal):.2f} mV")
    else:
        # Signal is already in reasonable range (typical ECG: -5 to +5 mV)
        print(f"  ✓ Signal is in expected range (max={max_abs:.2f} mV)")
    
    # Step 1: Bandpass filter - EXACTLY like user's code
    filtered_signal = bandpass_filter(ecg_signal, lowcut=0.5, highcut=45.0, fs=fs, order=4)
    
    # Step 2: Notch filter - EXACTLY like user's code
    notched_signal = notch_filter(filtered_signal, freq=50.0, fs=fs, Q=30)
    
    # Step 3: Detrend - EXACTLY like user's code
    detrended_signal = detrend(notched_signal)
    
    # Return the clean signal WITHOUT normalization
    # This preserves the original amplitude scale for clean, smooth visualization
    return detrended_signal

def remove_baseline_wander(signal_data, fs):
    """
    Remove baseline wander using high-pass filter
    """
    try:
        # High-pass filter at 0.5 Hz
        nyquist = fs / 2
        low_cutoff = 0.5 / nyquist
        
        if low_cutoff < 1:
            b, a = butter(4, low_cutoff, btype='high')
            filtered = filtfilt(b, a, signal_data)
        else:
            filtered = signal_data
            
        # Additional detrending
        filtered = signal.detrend(filtered, type='linear')
        
        return filtered
    except:
        return signal_data

def remove_powerline_interference(signal_data, fs):
    """
    Remove 50/60 Hz powerline interference using notch filter
    """
    try:
        # Notch filter at 50 Hz (European) and 60 Hz (American)
        for freq in [50, 60]:
            if fs > freq * 2:  # Check Nyquist criterion
                w0 = freq / (fs / 2)
                Q = 30  # Quality factor
                b, a = signal.iirnotch(w0, Q)
                signal_data = filtfilt(b, a, signal_data)
        
        return signal_data
    except:
        return signal_data

def remove_high_frequency_noise(signal_data, fs):
    """
    Remove high-frequency noise using low-pass filter
    """
    try:
        # Low-pass filter at 40 Hz
        nyquist = fs / 2
        high_cutoff = 40 / nyquist
        
        if high_cutoff < 1:
            b, a = butter(4, high_cutoff, btype='low')
            filtered = filtfilt(b, a, signal_data)
        else:
            filtered = signal_data
        
        return filtered
    except:
        return signal_data

def normalize_signal(signal_data):
    """
    Normalize signal to zero mean and unit variance (z-score normalization)
    This is the proper way to normalize ECG signals.
    """
    if len(signal_data) == 0:
        return signal_data
    
    # Z-score normalization: (signal - mean) / std
    mean_val = np.mean(signal_data)
    std_val = np.std(signal_data)
    
    if std_val > 1e-6:  # Avoid division by zero
        signal_data = (signal_data - mean_val) / std_val
    else:
        # If zero variance, just center the signal
        signal_data = signal_data - mean_val
    
    return signal_data

def wavelet_denoise(signal_data, wavelet='db4', level=4):
    """
    Apply wavelet denoising to remove additional noise
    """
    try:
        # Decompose signal
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Estimate noise level (using MAD of finest detail coefficients)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Calculate threshold
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # Apply soft thresholding to detail coefficients
        coeffs_thresh = list(coeffs)
        for i in range(1, len(coeffs)):
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
        
        # Reconstruct signal
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # Ensure same length as input
        if len(denoised) > len(signal_data):
            denoised = denoised[:len(signal_data)]
        elif len(denoised) < len(signal_data):
            denoised = np.pad(denoised, (0, len(signal_data) - len(denoised)), mode='edge')
        
        return denoised
    except:
        return signal_data

def segment_beats(signal_data, r_peaks, fs, before=0.2, after=0.4):
    """
    Segment individual beats around R-peaks
    
    Parameters:
    -----------
    signal_data : array
        ECG signal
    r_peaks : array
        R-peak locations (sample indices)
    fs : int
        Sampling frequency
    before : float
        Time before R-peak to include (seconds)
    after : float
        Time after R-peak to include (seconds)
    
    Returns:
    --------
    beats : list of arrays
        Segmented beats
    """
    beats = []
    before_samples = int(before * fs)
    after_samples = int(after * fs)
    fixed_length = before_samples + after_samples
    
    for peak in r_peaks:
        start = peak - before_samples
        end = peak + after_samples
        
        if start >= 0 and end < len(signal_data):
            beat = signal_data[start:end]
            
            # Ensure fixed length
            if len(beat) == fixed_length:
                beats.append(beat)
            elif len(beat) < fixed_length:
                # Pad if necessary
                beat = np.pad(beat, (0, fixed_length - len(beat)), mode='edge')
                beats.append(beat)
        elif start < 0:
            # Pad at the beginning
            beat = signal_data[0:end]
            beat = np.pad(beat, (abs(start), 0), mode='edge')
            if len(beat) == fixed_length:
                beats.append(beat)
        elif end >= len(signal_data):
            # Pad at the end
            beat = signal_data[start:]
            beat = np.pad(beat, (0, end - len(signal_data) + 1), mode='edge')
            if len(beat) == fixed_length:
                beats.append(beat)
    
    return beats

def adaptive_r_peak_detection(signal_data, fs):
    """
    Advanced R-peak detection with adaptive thresholding
    """
    # Bandpass filter for QRS complex (5-15 Hz)
    b, a = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal_data)
    
    # Differentiate
    diff = np.diff(filtered)
    
    # Square
    squared = diff ** 2
    
    # Moving window integration
    window_size = int(0.150 * fs)  # 150ms window
    mwi = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Adaptive thresholding
    threshold = np.mean(mwi) + 0.5 * np.std(mwi)
    
    # Find peaks with adaptive threshold
    min_distance = int(0.200 * fs)  # Minimum 200ms between beats
    peaks, properties = find_peaks(mwi, height=threshold, distance=min_distance)
    
    # Refine to actual R-peaks
    refined_peaks = []
    search_window = int(0.040 * fs)  # 40ms window
    
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal_data), peak + search_window)
        
        # Find maximum in the window (actual R-peak)
        window_signal = signal_data[start:end]
        if len(window_signal) > 0:
            local_max = np.argmax(np.abs(window_signal))
            refined_peaks.append(start + local_max)
    
    return np.array(refined_peaks)

def calculate_signal_quality_index(signal_data, fs):
    """
    Calculate comprehensive signal quality index (SQI)
    
    Returns:
    --------
    sqi : float
        Signal quality index (0-1, higher is better)
    metrics : dict
        Individual quality metrics
    """
    metrics = {}
    
    # 1. Check signal amplitude range
    amplitude_range = np.ptp(signal_data)
    metrics['amplitude_range'] = amplitude_range
    amplitude_score = min(amplitude_range / 2.0, 1.0)  # Normalize to 0-1
    
    # 2. Check for flat lines (missing data)
    flat_segments = np.sum(np.abs(np.diff(signal_data)) < 1e-6)
    flat_ratio = flat_segments / len(signal_data)
    metrics['flat_ratio'] = flat_ratio
    flat_score = 1.0 - min(flat_ratio * 10, 1.0)
    
    # 3. Check baseline drift
    detrended = signal.detrend(signal_data)
    drift = signal_data - detrended
    drift_ratio = np.std(drift) / (np.std(signal_data) + 1e-6)
    metrics['drift_ratio'] = drift_ratio
    drift_score = 1.0 - min(drift_ratio * 2, 1.0)
    
    # 4. Check high-frequency noise
    b, a = butter(4, 40/(fs/2), btype='high')
    high_freq = filtfilt(b, a, signal_data)
    noise_ratio = np.std(high_freq) / (np.std(signal_data) + 1e-6)
    metrics['noise_ratio'] = noise_ratio
    noise_score = 1.0 - min(noise_ratio * 5, 1.0)
    
    # 5. Check for clipping
    max_val = np.max(np.abs(signal_data))
    clipping_ratio = np.sum(np.abs(signal_data) > 0.95 * max_val) / len(signal_data)
    metrics['clipping_ratio'] = clipping_ratio
    clipping_score = 1.0 - min(clipping_ratio * 10, 1.0)
    
    # Calculate weighted SQI
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    scores = [amplitude_score, flat_score, drift_score, noise_score, clipping_score]
    sqi = np.sum(np.array(weights) * np.array(scores))
    
    metrics['overall_sqi'] = sqi
    return sqi, metrics