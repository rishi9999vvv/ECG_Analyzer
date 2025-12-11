"""
Standalone test script for ECG preprocessing
Tests the fixed preprocessing approach with MIT-BIH data
"""

import numpy as np
import matplotlib.pyplot as plt
import wfdb
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.preprocess import preprocess_ecg_fixed

def test_preprocessing():
    """Test preprocessing with MIT-BIH record 100"""
    
    print("=" * 60)
    print("Testing ECG Preprocessing with MIT-BIH Record 100")
    print("=" * 60)
    
    try:
        # Load MIT-BIH record 100
        record = wfdb.rdrecord('100', pn_dir='mitdb')
        annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')
        
        # Extract signal from first channel (MLII)
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        print(f"\n✓ Loaded Record 100:")
        print(f"  Signal shape: {signal.shape}")
        print(f"  Sampling frequency: {fs} Hz")
        print(f"  Number of annotations: {len(annotation.sample)}")
        print(f"  Missing values: {np.isnan(signal).sum()}")
        
        # Handle NaN values
        if np.isnan(signal).sum() > 0:
            signal = np.nan_to_num(signal, nan=np.nanmean(signal))
        
        # Apply fixed preprocessing
        print(f"\n✓ Applying fixed preprocessing...")
        filtered_signal = preprocess_ecg_fixed(signal, fs, use_wfdb_gain=False, wfdb_record=None)
        
        print(f"  Original signal - Mean: {np.mean(signal):.4f}, Std: {np.std(signal):.4f}")
        print(f"  Filtered signal - Mean: {np.mean(filtered_signal):.4f}, Std: {np.std(filtered_signal):.4f}")
        
        # Create comparison plots
        print(f"\n✓ Creating visualization plots...")
        
        # Plot 1: Raw signal
        plt.figure(figsize=(15, 5))
        plt.plot(signal[:1500], color='purple', linewidth=1)
        plt.title("Raw ECG Signal (Record 100, first 1500 samples)", fontsize=14)
        plt.xlabel("Samples (≈ 2000 / 360 Hz ≈ 5.5 seconds)", fontsize=12)
        plt.ylabel("Amplitude (mV)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('reports/plots/test_raw_signal.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: reports/plots/test_raw_signal.png")
        plt.close()
        
        # Plot 2: ECG with annotations
        plt.figure(figsize=(15, 5))
        plt.plot(signal[:3000], label='ECG', color='blue', linewidth=1)
        
        # Add annotation markers
        for s, l in zip(annotation.sample, annotation.symbol):
            if s < 3000:
                plt.axvline(x=s, color='r', linestyle='--', alpha=0.6)
                plt.text(s, np.max(signal[:3000]) * 0.95, l, fontsize=8, rotation=0)
        
        plt.title("ECG with Annotations (first 3000 samples)", fontsize=14)
        plt.xlabel("Samples (≈ 8.3 seconds)", fontsize=12)
        plt.ylabel("Amplitude (mV)", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('reports/plots/test_ecg_with_annotations.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: reports/plots/test_ecg_with_annotations.png")
        plt.close()
        
        # Plot 3: Raw vs Filtered comparison (zoomed view)
        plt.figure(figsize=(15, 5))
        plt.plot(signal[1000:2000], label="Raw Signal", alpha=0.7, color='purple', linewidth=1.5)
        plt.plot(filtered_signal[1000:2000], label="Filtered Signal", alpha=0.9, color='green', linewidth=1.5)
        plt.legend()
        plt.title("Raw vs Filtered ECG (Zoomed 1000–2000 samples)", fontsize=14)
        plt.xlabel("Samples", fontsize=12)
        plt.ylabel("Amplitude (normalized)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('reports/plots/test_raw_vs_filtered.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: reports/plots/test_raw_vs_filtered.png")
        plt.close()
        
        # Plot 4: Distribution comparison
        import seaborn as sns
        plt.figure(figsize=(12, 5))
        sns.histplot(signal, bins=100, kde=True, color='blue', label='Raw', alpha=0.6)
        sns.histplot(filtered_signal, bins=100, kde=True, color='green', label='Filtered', alpha=0.6)
        plt.title("Distribution of ECG Amplitudes: Raw vs Filtered", fontsize=14)
        plt.xlabel("Amplitude", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('reports/plots/test_distribution_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: reports/plots/test_distribution_comparison.png")
        plt.close()
        
        print(f"\n✅ All test plots generated successfully!")
        print(f"   Check the 'reports/plots/' folder for output files.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Make sure MIT-BIH database is in the correct location.")
        print("      Expected: mitdb/ folder in current directory or data/mit-bih/")


if __name__ == '__main__':
    # Create output directory
    os.makedirs('reports/plots', exist_ok=True)
    
    test_preprocessing()

