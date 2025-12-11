"""
Feature Extraction Module
Extracts morphological and statistical features from ECG beats
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks
import pywt

def extract_features(beat_segment, fs=360):
    """
    Extract comprehensive features from a single ECG beat
    
    Parameters:
    -----------
    beat_segment : array
        Single beat waveform
    fs : int
        Sampling frequency
    
    Returns:
    --------
    features : array
        Feature vector for the beat
    """
    features = []
    
    # 1. Morphological features
    morph_features = extract_morphological_features(beat_segment, fs)
    features.extend(morph_features)
    
    # 2. Statistical features
    stat_features = extract_statistical_features(beat_segment)
    features.extend(stat_features)
    
    # 3. Wavelet features
    wavelet_features = extract_wavelet_features(beat_segment)
    features.extend(wavelet_features)
    
    # 4. Frequency domain features
    freq_features = extract_frequency_features(beat_segment, fs)
    features.extend(freq_features)
    
    # 5. RR interval features (if available)
    rr_features = extract_rr_features(beat_segment)
    features.extend(rr_features)
    
    return np.array(features)

def extract_morphological_features(beat, fs):
    """
    Extract morphological features from ECG beat
    """
    features = []
    
    # Find R-peak (should be near center)
    center = len(beat) // 2
    r_peak_idx = center
    r_peak_amp = beat[r_peak_idx]
    
    # QRS complex detection
    qrs_start, qrs_end = detect_qrs_boundaries(beat, r_peak_idx, fs)
    qrs_duration = (qrs_end - qrs_start) / fs * 1000  # Convert to ms
    
    # P-wave detection (before QRS)
    p_wave_amp, p_wave_duration = detect_p_wave(beat[:qrs_start], fs)
    
    # T-wave detection (after QRS)
    t_wave_amp, t_wave_duration = detect_t_wave(beat[qrs_end:], fs)
    
    # Features
    features.append(r_peak_amp)  # R-peak amplitude
    features.append(qrs_duration)  # QRS duration
    features.append(p_wave_amp)  # P-wave amplitude
    features.append(p_wave_duration)  # P-wave duration
    features.append(t_wave_amp)  # T-wave amplitude
    features.append(t_wave_duration)  # T-wave duration
    
    # QRS area
    qrs_area = np.trapz(np.abs(beat[qrs_start:qrs_end]))
    features.append(qrs_area)
    
    # PR interval (if P-wave detected)
    pr_interval = (qrs_start - center) / fs * 1000 if p_wave_amp > 0 else 0
    features.append(pr_interval)
    
    # QT interval
    qt_interval = (qrs_end - qrs_start + t_wave_duration * fs / 1000) / fs * 1000
    features.append(qt_interval)
    
    return features

def extract_statistical_features(beat):
    """
    Extract statistical features from ECG beat
    """
    features = []
    
    # Basic statistics
    features.append(np.mean(beat))
    features.append(np.std(beat))
    features.append(np.var(beat))
    features.append(stats.skew(beat))
    features.append(stats.kurtosis(beat))
    
    # Range and percentiles
    features.append(np.ptp(beat))  # Peak-to-peak
    features.append(np.percentile(beat, 25))
    features.append(np.percentile(beat, 50))
    features.append(np.percentile(beat, 75))
    
    # Energy
    features.append(np.sum(beat ** 2))
    
    # Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(beat)))
    features.append(zero_crossings)
    
    # Entropy
    hist, _ = np.histogram(beat, bins=10)
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    features.append(entropy)
    
    return features

def extract_wavelet_features(beat):
    """
    Extract wavelet-based features
    """
    features = []
    
    try:
        # Wavelet decomposition
        coeffs = pywt.wavedec(beat, 'db4', level=4)
        
        # Features from each level
        for i, coeff in enumerate(coeffs):
            features.append(np.mean(np.abs(coeff)))
            features.append(np.std(coeff))
            features.append(np.sum(coeff ** 2))  # Energy
            
            # Pad with zeros if needed to maintain fixed feature length
            if len(features) >= 15:  # Limit wavelet features
                break
        
        # Ensure fixed length
        while len(features) < 15:
            features.append(0)
            
    except:
        features = [0] * 15  # Default if wavelet transform fails
    
    return features[:15]

def extract_frequency_features(beat, fs):
    """
    Extract frequency domain features
    """
    features = []
    
    try:
        # FFT
        fft = np.fft.fft(beat)
        freqs = np.fft.fftfreq(len(beat), 1/fs)
        
        # Power spectral density
        psd = np.abs(fft) ** 2
        
        # Only use positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        psd_pos = psd[pos_mask]
        
        # Frequency bands
        # VLF: 0-5 Hz
        # LF: 5-15 Hz (QRS complex)
        # HF: 15-40 Hz
        
        vlf_mask = (freqs_pos >= 0) & (freqs_pos < 5)
        lf_mask = (freqs_pos >= 5) & (freqs_pos < 15)
        hf_mask = (freqs_pos >= 15) & (freqs_pos < 40)
        
        vlf_power = np.sum(psd_pos[vlf_mask]) if np.any(vlf_mask) else 0
        lf_power = np.sum(psd_pos[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.sum(psd_pos[hf_mask]) if np.any(hf_mask) else 0
        total_power = vlf_power + lf_power + hf_power
        
        # Normalized powers
        features.append(vlf_power)
        features.append(lf_power)
        features.append(hf_power)
        features.append(lf_power / (hf_power + 1e-10))  # LF/HF ratio
        features.append(total_power)
        
        # Dominant frequency
        if len(psd_pos) > 0:
            dominant_freq_idx = np.argmax(psd_pos)
            dominant_freq = freqs_pos[dominant_freq_idx]
            features.append(dominant_freq)
        else:
            features.append(0)
        
        # Spectral entropy
        psd_norm = psd_pos / (np.sum(psd_pos) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        features.append(spectral_entropy)
        
    except:
        features = [0] * 7  # Default if FFT fails
    
    return features

def extract_rr_features(beat):
    """
    Extract RR interval related features
    """
    features = []
    
    # Since we have a single beat, we'll extract position-based features
    # These would normally be calculated from RR intervals
    
    # R-peak position (relative to beat center)
    center = len(beat) // 2
    r_peak_idx = center + np.argmax(np.abs(beat[center-10:center+10])) - 10
    r_peak_position = r_peak_idx / len(beat)
    
    features.append(r_peak_position)
    
    # Pre-R and post-R energy ratio
    pre_r_energy = np.sum(beat[:r_peak_idx] ** 2)
    post_r_energy = np.sum(beat[r_peak_idx:] ** 2)
    energy_ratio = pre_r_energy / (post_r_energy + 1e-10)
    features.append(energy_ratio)
    
    # Symmetry index
    pre_r = beat[:r_peak_idx]
    post_r = beat[r_peak_idx:]
    min_len = min(len(pre_r), len(post_r))
    if min_len > 0:
        pre_r = pre_r[-min_len:]
        post_r = post_r[:min_len]
        symmetry = 1 - np.mean(np.abs(pre_r[::-1] - post_r))
    else:
        symmetry = 0
    features.append(symmetry)
    
    return features

def detect_qrs_boundaries(beat, r_peak_idx, fs):
    """
    Detect QRS complex boundaries
    """
    # Simple approach: find where signal drops to 10% of R-peak amplitude
    threshold = 0.1 * np.abs(beat[r_peak_idx])
    
    # Search backward for QRS start
    qrs_start = r_peak_idx
    for i in range(r_peak_idx, max(0, r_peak_idx - int(0.1 * fs)), -1):
        if np.abs(beat[i]) < threshold:
            qrs_start = i
            break
    
    # Search forward for QRS end
    qrs_end = r_peak_idx
    for i in range(r_peak_idx, min(len(beat), r_peak_idx + int(0.1 * fs))):
        if np.abs(beat[i]) < threshold:
            qrs_end = i
            break
    
    return qrs_start, qrs_end

def detect_p_wave(pre_qrs_segment, fs):
    """
    Detect P-wave in the segment before QRS
    """
    if len(pre_qrs_segment) < 10:
        return 0, 0
    
    # Smooth the signal
    window = min(5, len(pre_qrs_segment) // 4)
    if window > 1:
        smoothed = np.convolve(pre_qrs_segment, np.ones(window)/window, mode='valid')
    else:
        smoothed = pre_qrs_segment
    
    if len(smoothed) == 0:
        return 0, 0
    
    # Find the peak (P-wave)
    peaks, properties = find_peaks(smoothed, distance=len(smoothed)//2)
    
    if len(peaks) > 0:
        p_peak_idx = peaks[0]
        p_wave_amp = smoothed[p_peak_idx]
        
        # Estimate duration (find where amplitude drops to 50% of peak)
        threshold = 0.5 * p_wave_amp
        start = p_peak_idx
        end = p_peak_idx
        
        for i in range(p_peak_idx, -1, -1):
            if smoothed[i] < threshold:
                start = i
                break
        
        for i in range(p_peak_idx, len(smoothed)):
            if smoothed[i] < threshold:
                end = i
                break
        
        p_wave_duration = (end - start) / fs * 1000  # Convert to ms
        return p_wave_amp, p_wave_duration
    
    return 0, 0

def detect_t_wave(post_qrs_segment, fs):
    """
    Detect T-wave in the segment after QRS
    """
    if len(post_qrs_segment) < 20:
        return 0, 0
    
    # Skip the initial part (ST segment)
    st_skip = int(0.06 * fs)  # Skip 60ms
    if st_skip >= len(post_qrs_segment):
        return 0, 0
    
    t_segment = post_qrs_segment[st_skip:]
    
    # Smooth the signal
    window = min(10, len(t_segment) // 4)
    if window > 1:
        smoothed = np.convolve(t_segment, np.ones(window)/window, mode='valid')
    else:
        smoothed = t_segment
    
    if len(smoothed) == 0:
        return 0, 0
    
    # Find the peak (T-wave)
    peaks, properties = find_peaks(np.abs(smoothed), distance=len(smoothed)//2)
    
    if len(peaks) > 0:
        t_peak_idx = peaks[0]
        t_wave_amp = smoothed[t_peak_idx]
        
        # Estimate duration
        threshold = 0.5 * np.abs(t_wave_amp)
        start = t_peak_idx
        end = t_peak_idx
        
        for i in range(t_peak_idx, -1, -1):
            if np.abs(smoothed[i]) < threshold:
                start = i
                break
        
        for i in range(t_peak_idx, len(smoothed)):
            if np.abs(smoothed[i]) < threshold:
                end = i
                break
        
        t_wave_duration = (end - start) / fs * 1000  # Convert to ms
        return t_wave_amp, t_wave_duration
    
    return 0, 0

def get_feature_names():
    """
    Get names of all extracted features
    """
    names = []
    
    # Morphological features
    names.extend(['r_peak_amp', 'qrs_duration', 'p_wave_amp', 'p_wave_duration',
                  't_wave_amp', 't_wave_duration', 'qrs_area', 'pr_interval', 'qt_interval'])
    
    # Statistical features
    names.extend(['mean', 'std', 'var', 'skewness', 'kurtosis', 'ptp',
                  'percentile_25', 'percentile_50', 'percentile_75', 'energy',
                  'zero_crossings', 'entropy'])
    
    # Wavelet features
    for i in range(5):
        names.extend([f'wavelet_l{i}_mean', f'wavelet_l{i}_std', f'wavelet_l{i}_energy'])
    
    # Frequency features
    names.extend(['vlf_power', 'lf_power', 'hf_power', 'lf_hf_ratio',
                  'total_power', 'dominant_freq', 'spectral_entropy'])
    
    # RR features
    names.extend(['r_peak_position', 'energy_ratio', 'symmetry_index'])
    
    return names