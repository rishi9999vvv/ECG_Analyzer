"""
Test script to visualize ECG preprocessing fixes
Shows comparison of old vs fixed preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
from backend.preprocess import preprocess_signal, preprocess_ecg_fixed

# Set up output directory
OUTPUT_DIR = 'reports/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_test_ecg(record_name='100', data_path='data/mit-bih'):
    """
    Load a test ECG record from MIT-BIH database
    """
    try:
        record_path = os.path.join(data_path, record_name)
        record = wfdb.rdrecord(record_path)
        
        # Use MLII channel (first channel)
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        # Load annotations for visualization
        try:
            annotation = wfdb.rdann(record_path, 'atr')
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol
        except:
            ann_samples = None
            ann_symbols = None
        
        return signal, fs, record, ann_samples, ann_symbols
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        print("Generating synthetic ECG signal for testing...")
        
        # Generate synthetic ECG
        fs = 360
        duration = 10  # seconds
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create ECG-like signal
        signal = np.zeros(len(t))
        heart_rate = 75  # bpm
        beat_period = 60 / heart_rate
        
        for i in range(int(duration / beat_period)):
            beat_time = i * beat_period
            beat_idx = int(beat_time * fs)
            if beat_idx < len(t):
                # Add QRS complex
                qrs_range = np.arange(max(0, beat_idx - 20), min(len(t), beat_idx + 20))
                if len(qrs_range) > 0:
                    qrs_t = qrs_range - beat_idx
                    signal[qrs_range] += 1.5 * np.exp(-(qrs_t**2) / 50)
        
        # Add noise and artifacts
        signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Baseline wander
        signal += 0.05 * np.sin(2 * np.pi * 50 * t)  # Powerline interference
        signal += 0.1 * np.random.randn(len(t))  # Random noise
        
        return signal, fs, None, None, None


def plot_preprocessing_comparison():
    """
    Plot comparison of old vs fixed preprocessing
    """
    # Load test ECG
    signal, fs, record, ann_samples, ann_symbols = load_test_ecg()
    
    # Calculate time axis
    time = np.arange(len(signal)) / fs
    
    # Limit to first 10 seconds for visualization
    max_samples = min(len(signal), int(10 * fs))
    signal_trimmed = signal[:max_samples]
    time_trimmed = time[:max_samples]
    
    # Apply old preprocessing
    try:
        old_filtered = preprocess_signal(signal_trimmed, fs)
    except Exception as e:
        print(f"Error with old preprocessing: {e}")
        old_filtered = signal_trimmed.copy()
    
    # Apply fixed preprocessing
    try:
        if record is not None:
            fixed_filtered = preprocess_ecg_fixed(signal_trimmed, fs, 
                                                 use_wfdb_gain=True, 
                                                 wfdb_record=record)
        else:
            fixed_filtered = preprocess_ecg_fixed(signal_trimmed, fs)
    except Exception as e:
        print(f"Error with fixed preprocessing: {e}")
        fixed_filtered = signal_trimmed.copy()
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                            facecolor='#E8F4F8')
    fig.suptitle('ECG Preprocessing Comparison: Before Fix vs After Fix', 
                fontsize=16, fontweight='bold', color='#1E3A5F', y=0.98)
    
    # Plot 1: Raw signal
    ax1 = axes[0]
    ax1.plot(time_trimmed, signal_trimmed, color='#E74C3C', linewidth=1.5, alpha=0.8)
    ax1.set_title('Before Fix (Raw ECG Signal)', fontsize=14, fontweight='bold',
                 color='#1E3A5F', pad=15)
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Amplitude (mV)', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add annotations if available
    if ann_samples is not None:
        ann_times = [s / fs for s in ann_samples if s < max_samples]
        ann_amplitudes = [signal_trimmed[s] for s in ann_samples if s < max_samples]
        if ann_times:
            ax1.scatter(ann_times[:20], ann_amplitudes[:20], 
                       color='red', s=30, alpha=0.6, zorder=5,
                       label='R-peaks (annotations)')
            ax1.legend(loc='upper right')
    
    # Plot 2: Old preprocessing
    ax2 = axes[1]
    ax2.plot(time_trimmed, old_filtered, color='#F39C12', linewidth=1.5, alpha=0.8)
    ax2.set_title('Old Preprocessing (Multiple Filters + Min-Max Normalization)', 
                 fontsize=14, fontweight='bold', color='#1E3A5F', pad=15)
    ax2.set_xlabel('Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Amplitude (normalized)', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Fixed preprocessing
    ax3 = axes[2]
    ax3.plot(time_trimmed, fixed_filtered, color='#2ECC71', linewidth=1.5, alpha=0.8)
    ax3.set_title('After Fix (Bandpass Filter + Z-score Normalization)', 
                 fontsize=14, fontweight='bold', color='#1E3A5F', pad=15)
    ax3.set_xlabel('Time (seconds)', fontweight='bold')
    ax3.set_ylabel('Amplitude (normalized)', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ecg_preprocessing_comparison.png', 
               dpi=300, bbox_inches='tight', facecolor='#E8F4F8')
    print(f"✓ Saved: {OUTPUT_DIR}/ecg_preprocessing_comparison.png")
    plt.close()
    
    # Create overlay comparison
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#E8F4F8')
    fig.suptitle('Preprocessing Comparison: Overlay View', 
                fontsize=16, fontweight='bold', color='#1E3A5F', y=0.98)
    
    ax.plot(time_trimmed, signal_trimmed, label='Raw ECG', 
           color='#E74C3C', linewidth=1.5, alpha=0.6)
    ax.plot(time_trimmed, old_filtered, label='Old Preprocessing', 
           color='#F39C12', linewidth=1.5, alpha=0.8)
    ax.plot(time_trimmed, fixed_filtered, label='Fixed Preprocessing', 
           color='#2ECC71', linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_title('Signal Comparison: Raw vs Old vs Fixed', 
                fontsize=14, fontweight='bold', color='#1E3A5F', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/ecg_preprocessing_overlay.png', 
               dpi=300, bbox_inches='tight', facecolor='#E8F4F8')
    print(f"✓ Saved: {OUTPUT_DIR}/ecg_preprocessing_overlay.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("Preprocessing Statistics:")
    print("="*60)
    print(f"Raw Signal:")
    print(f"  Mean: {np.mean(signal_trimmed):.4f}, Std: {np.std(signal_trimmed):.4f}")
    print(f"  Min: {np.min(signal_trimmed):.4f}, Max: {np.max(signal_trimmed):.4f}")
    print(f"\nOld Preprocessing:")
    print(f"  Mean: {np.mean(old_filtered):.4f}, Std: {np.std(old_filtered):.4f}")
    print(f"  Min: {np.min(old_filtered):.4f}, Max: {np.max(old_filtered):.4f}")
    print(f"\nFixed Preprocessing:")
    print(f"  Mean: {np.mean(fixed_filtered):.4f}, Std: {np.std(fixed_filtered):.4f}")
    print(f"  Min: {np.min(fixed_filtered):.4f}, Max: {np.max(fixed_filtered):.4f}")
    print("="*60)


if __name__ == '__main__':
    print("="*60)
    print("Testing ECG Preprocessing Fix")
    print("="*60)
    print()
    
    plot_preprocessing_comparison()
    
    print("\n✅ Preprocessing comparison complete!")

