"""
ECG Analyzer - Professional Medical-Themed Visualizations
For Presentation and Academic Posters

This module generates publication-ready figures with consistent medical styling:
- Light blue backgrounds
- Clean white cards
- Professional fonts
- Medical aesthetic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
from scipy import stats

# Set up medical theme
MEDICAL_COLORS = {
    'background': '#E8F4F8',  # Light blue
    'card': '#FFFFFF',        # White
    'primary': '#1E3A5F',     # Dark navy
    'accent': '#4A90A4',      # Medical blue
    'success': '#2ECC71',     # Green
    'warning': '#F39C12',     # Orange
    'danger': '#E74C3C',      # Red
    'text': '#2C3E50',        # Dark gray
    'grid': '#D5E8F0'        # Light grid
}

BEAT_COLORS = {
    'N': '#10b981',  # Green - Normal
    'V': '#ef4444',  # Red - PVC
    'S': '#3b82f6',  # Blue - SVE
    'F': '#8b5cf6',  # Purple - Fusion
    'Q': '#6b7280'   # Gray - Unclassifiable
}

# Configure matplotlib style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'figure.facecolor': MEDICAL_COLORS['background'],
    'axes.facecolor': MEDICAL_COLORS['card'],
    'savefig.facecolor': MEDICAL_COLORS['background'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Set seaborn theme
sns.set_theme(style="whitegrid", palette="pastel", rc={
    'figure.facecolor': MEDICAL_COLORS['background'],
    'axes.facecolor': MEDICAL_COLORS['card'],
    'grid.color': MEDICAL_COLORS['grid']
})

# Create output directory
OUTPUT_DIR = 'reports/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_medical_axes(fig, ax, title=None):
    """Style axes with medical theme"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(MEDICAL_COLORS['text'])
    ax.spines['bottom'].set_color(MEDICAL_COLORS['text'])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_facecolor(MEDICAL_COLORS['card'])
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', 
                    color=MEDICAL_COLORS['primary'], pad=15)
    
    return ax


def plot_class_distribution():
    """
    1️⃣ Dataset Composition:
    - Bar chart for class counts (N, V, S, F, Q)
    - Pie chart showing proportion of each class
    """
    # Generate dummy data
    classes = ['N', 'V', 'S', 'F', 'Q']
    class_names = ['Normal', 'PVC', 'SVE', 'Fusion', 'Unclassifiable']
    counts = [45000, 3200, 1800, 500, 200]
    
    fig = plt.figure(figsize=(14, 6), facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('Dataset Composition - MIT-BIH Arrhythmia Database', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    # Bar chart
    ax1 = fig.add_subplot(121)
    bars = ax1.bar(class_names, counts, color=[BEAT_COLORS[c] for c in classes], 
                  edgecolor=MEDICAL_COLORS['primary'], linewidth=1.5, alpha=0.8)
    ax1 = create_medical_axes(fig, ax1, 'Class Distribution (Bar Chart)')
    ax1.set_xlabel('Beat Type', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_ylabel('Count', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2 = fig.add_subplot(122)
    colors_pie = [BEAT_COLORS[c] for c in classes]
    wedges, texts, autotexts = ax2.pie(counts, labels=class_names, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Class Proportion (Pie Chart)', fontsize=14, fontweight='bold',
                 color=MEDICAL_COLORS['primary'], pad=15)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/1_dataset_composition.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/1_dataset_composition.png")
    plt.close()


def plot_signal_preprocessing():
    """
    2️⃣ Signal Preprocessing:
    Line plot comparing raw vs filtered ECG signals
    """
    # Generate dummy ECG signal
    fs = 360
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Raw signal with noise
    raw_signal = (np.sin(2 * np.pi * 1.2 * t) + 
                  0.3 * np.sin(2 * np.pi * 0.5 * t) +  # Baseline wander
                  0.2 * np.sin(2 * np.pi * 50 * t) +  # Powerline interference
                  0.15 * np.random.randn(len(t)))      # Noise
    
    # Filtered signal (simulated)
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [0.5/(fs/2), 40/(fs/2)], btype='band')
    filtered_signal = filtfilt(b, a, raw_signal)
    filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('Signal Preprocessing: Raw vs Filtered ECG', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    ax.plot(t, raw_signal, label='Raw ECG Signal', color='#E74C3C', 
           linewidth=1.5, alpha=0.7)
    ax.plot(t, filtered_signal, label='Filtered ECG Signal', color='#2ECC71', 
           linewidth=2, alpha=0.9)
    
    ax = create_medical_axes(fig, ax, 'Signal Comparison')
    ax.set_xlabel('Time (seconds)', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax.set_ylabel('Amplitude (mV)', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/2_signal_preprocessing.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/2_signal_preprocessing.png")
    plt.close()


def plot_feature_extraction():
    """
    3️⃣ Feature Extraction:
    - Heatmap for feature correlations
    - Boxplot comparing one key feature across all arrhythmia classes
    """
    # Generate dummy feature correlation data
    feature_names = ['RR Interval', 'QRS Width', 'HRV', 'R Amplitude', 
                    'P Amplitude', 'T Amplitude', 'ST Segment', 'QT Interval']
    n_features = len(feature_names)
    
    # Create correlation matrix
    np.random.seed(42)
    corr_matrix = np.random.rand(n_features, n_features)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Create feature data for boxplot
    classes = ['N', 'V', 'S', 'F', 'Q']
    class_names = ['Normal', 'PVC', 'SVE', 'Fusion', 'Unclassifiable']
    n_samples_per_class = 100
    
    fig = plt.figure(figsize=(16, 7), facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('Feature Extraction Analysis', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    # Heatmap
    ax1 = fig.add_subplot(121)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', 
               center=0, vmin=-1, vmax=1, square=True,
               xticklabels=feature_names, yticklabels=feature_names,
               cbar_kws={'label': 'Correlation Coefficient'},
               ax=ax1, linewidths=0.5, linecolor='white')
    ax1.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold',
                 color=MEDICAL_COLORS['primary'], pad=15)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Boxplot
    ax2 = fig.add_subplot(122)
    data_for_boxplot = []
    labels_for_boxplot = []
    
    for i, (cls, cls_name) in enumerate(zip(classes, class_names)):
        # Generate dummy data with different distributions per class
        if cls == 'N':
            data = np.random.normal(800, 50, n_samples_per_class)
        elif cls == 'V':
            data = np.random.normal(600, 80, n_samples_per_class)
        elif cls == 'S':
            data = np.random.normal(750, 60, n_samples_per_class)
        elif cls == 'F':
            data = np.random.normal(700, 70, n_samples_per_class)
        else:
            data = np.random.normal(650, 90, n_samples_per_class)
        
        data_for_boxplot.append(data)
        labels_for_boxplot.append(cls_name)
    
    bp = ax2.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, cls in zip(bp['boxes'], classes):
        patch.set_facecolor(BEAT_COLORS[cls])
        patch.set_alpha(0.7)
        patch.set_edgecolor(MEDICAL_COLORS['primary'])
        patch.set_linewidth(1.5)
    
    ax2 = create_medical_axes(fig, ax2, 'RR Interval Distribution by Class')
    ax2.set_xlabel('Beat Type', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.set_ylabel('RR Interval (ms)', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.set_xticklabels(labels_for_boxplot, rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/3_feature_extraction.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/3_feature_extraction.png")
    plt.close()


def plot_training_validation():
    """
    4️⃣ Training & Validation:
    Accuracy vs Epochs and Loss vs Epochs curves (side by side)
    """
    # Generate dummy training data
    np.random.seed(42)
    epochs = np.arange(1, 51)
    
    # Training curves (simulated with realistic behavior)
    train_acc = 0.5 + 0.45 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 0.01, len(epochs))
    train_acc = np.clip(train_acc, 0, 1)
    
    val_acc = 0.5 + 0.43 * (1 - np.exp(-epochs / 15)) + np.random.normal(0, 0.015, len(epochs))
    val_acc = np.clip(val_acc, 0, 1)
    
    train_loss = 1.5 * np.exp(-epochs / 12) + 0.1 + np.random.normal(0, 0.02, len(epochs))
    train_loss = np.clip(train_loss, 0, None)
    
    val_loss = 1.5 * np.exp(-epochs / 12) + 0.15 + np.random.normal(0, 0.025, len(epochs))
    val_loss = np.clip(val_loss, 0, None)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), 
                                   facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('Training & Validation Curves', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, label='Training Accuracy', 
            color='#2ECC71', linewidth=2.5, marker='o', markersize=4, alpha=0.8, markevery=5)
    ax1.plot(epochs, val_acc, label='Validation Accuracy', 
            color='#3498DB', linewidth=2.5, marker='s', markersize=4, alpha=0.8, markevery=5)
    ax1 = create_medical_axes(fig, ax1, 'Model Accuracy')
    ax1.set_xlabel('Epoch', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_ylabel('Accuracy', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_ylim([0.4, 1.0])
    ax1.legend(loc='lower right', framealpha=0.95, shadow=True)
    
    # Format y-axis as percentage
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Loss plot
    ax2.plot(epochs, train_loss, label='Training Loss', 
            color='#E74C3C', linewidth=2.5, marker='o', markersize=4, alpha=0.8, markevery=5)
    ax2.plot(epochs, val_loss, label='Validation Loss', 
            color='#F39C12', linewidth=2.5, marker='s', markersize=4, alpha=0.8, markevery=5)
    ax2 = create_medical_axes(fig, ax2, 'Model Loss')
    ax2.set_xlabel('Epoch', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.set_ylabel('Loss', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.legend(loc='upper right', framealpha=0.95, shadow=True)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/4_training_validation.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/4_training_validation.png")
    plt.close()


def plot_model_evaluation():
    """
    5️⃣ Model Evaluation:
    - Confusion matrix (counts) and normalized version (side-by-side)
    - ROC curves per class
    """
    # Generate dummy confusion matrix
    classes = ['N', 'V', 'S', 'F', 'Q']
    class_names = ['Normal', 'PVC', 'SVE', 'Fusion', 'Unclassifiable']
    
    # Create realistic confusion matrix
    cm_counts = np.array([
        [4200, 50, 30, 10, 10],
        [80, 2800, 20, 15, 5],
        [40, 25, 1500, 10, 5],
        [15, 10, 5, 450, 5],
        [20, 5, 5, 5, 150]
    ])
    
    cm_normalized = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
    
    fig = plt.figure(figsize=(16, 7), facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('Model Evaluation Metrics', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    # Confusion matrix - counts
    ax1 = fig.add_subplot(121)
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'},
               ax=ax1, linewidths=0.5, linecolor='white',
               annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold',
                 color=MEDICAL_COLORS['primary'], pad=15)
    ax1.set_xlabel('Predicted Label', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_ylabel('True Label', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right')
    
    # Confusion matrix - normalized
    ax2 = fig.add_subplot(122)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized'},
               ax=ax2, linewidths=0.5, linecolor='white',
               annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold',
                 color=MEDICAL_COLORS['primary'], pad=15)
    ax2.set_xlabel('Predicted Label', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.set_ylabel('True Label', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5a_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/5a_confusion_matrix.png")
    plt.close()
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=MEDICAL_COLORS['background'])
    fig.suptitle('ROC Curves per Class', 
                fontsize=16, fontweight='bold', color=MEDICAL_COLORS['primary'], y=1.02)
    
    # Generate dummy ROC curves
    np.random.seed(42)
    for i, (cls, cls_name) in enumerate(zip(classes, class_names)):
        # Generate dummy ROC data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) + np.random.normal(0, 0.02, len(fpr))
        tpr = np.clip(tpr, 0, 1)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, label=f'{cls_name} (AUC = {roc_auc:.3f})',
               color=BEAT_COLORS[cls], linewidth=2.5, alpha=0.8)
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random Classifier')
    
    ax = create_medical_axes(fig, ax, 'ROC Curves')
    ax.set_xlabel('False Positive Rate', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax.set_ylabel('True Positive Rate', fontweight='bold', color=MEDICAL_COLORS['text'])
    ax.legend(loc='lower right', framealpha=0.95, shadow=True)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/5b_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/5b_roc_curves.png")
    plt.close()


def plot_annotated_ecg():
    """
    6️⃣ Annotated ECG Visualization (Bonus):
    Plotly graph showing ECG waveform with color-coded beat classifications
    """
    # Generate dummy ECG signal
    fs = 360
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create ECG-like signal
    ecg_signal = np.zeros(len(t))
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
                ecg_signal[qrs_range] += 1.5 * np.exp(-(qrs_t**2) / 50)
    
    # Add some variation
    ecg_signal += 0.1 * np.sin(2 * np.pi * 0.5 * t)
    ecg_signal += 0.05 * np.random.randn(len(t))
    
    # Generate beat classifications
    n_beats = int(duration / beat_period)
    beat_types = ['N', 'N', 'N', 'N', 'V', 'N', 'S', 'N', 'N', 'F', 'N', 'N', 'Q']
    beat_times = np.arange(0, duration, beat_period)[:n_beats]
    beat_indices = [int(bt * fs) for bt in beat_times[:len(beat_types)]]
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add ECG signal
    fig.add_trace(go.Scatter(
        x=t,
        y=ecg_signal,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#1e40af', width=1.5),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ))
    
    # Add beat annotations
    beat_labels_map = {
        'N': 'Normal Beats',
        'V': 'PVC',
        'S': 'SVE',
        'F': 'Fusion Beats',
        'Q': 'Unclassifiable'
    }
    
    for beat_type in ['N', 'V', 'S', 'F', 'Q']:
        type_indices = [i for i, bt in enumerate(beat_types[:len(beat_indices)]) 
                       if bt == beat_type and i < len(beat_indices)]
        
        if type_indices:
            x_vals = [t[beat_indices[i]] for i in type_indices]
            y_vals = [ecg_signal[beat_indices[i]] for i in type_indices]
            
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                name=beat_labels_map[beat_type],
                marker=dict(
                    color=BEAT_COLORS[beat_type],
                    size=12,
                    line=dict(color='white', width=1.5),
                    symbol='diamond' if beat_type == 'V' else 'circle'
                ),
                hovertemplate=f'<b>{beat_labels_map[beat_type]}</b><br>Time: %{{x:.2f}}s<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Annotated ECG Signal with Beat Classifications',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': MEDICAL_COLORS['primary']}
        },
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridcolor=MEDICAL_COLORS['grid'],
            titlefont=dict(size=12, color=MEDICAL_COLORS['text'])
        ),
        yaxis=dict(
            title='Amplitude (mV)',
            showgrid=True,
            gridcolor=MEDICAL_COLORS['grid'],
            titlefont=dict(size=12, color=MEDICAL_COLORS['text'])
        ),
        plot_bgcolor=MEDICAL_COLORS['card'],
        paper_bgcolor=MEDICAL_COLORS['background'],
        height=500,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)'
        ),
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    # Save as HTML
    fig.write_html(f'{OUTPUT_DIR}/6_annotated_ecg_plotly.html')
    print(f"✓ Saved: {OUTPUT_DIR}/6_annotated_ecg_plotly.html")
    
    # Also save as static image (optional - requires kaleido)
    try:
        fig.write_image(f'{OUTPUT_DIR}/6_annotated_ecg_plotly.png', width=1200, height=600)
        print(f"✓ Saved: {OUTPUT_DIR}/6_annotated_ecg_plotly.png")
    except Exception as e:
        print(f"⚠ Note: Could not save PNG image (kaleido not installed). HTML version saved.")
        print(f"   To enable PNG export, install: pip install kaleido")


def generate_all_plots():
    """Generate all visualization plots"""
    print("=" * 60)
    print("🩺 Generating Medical-Themed Visualizations")
    print("=" * 60)
    print()
    
    plot_class_distribution()
    plot_signal_preprocessing()
    plot_feature_extraction()
    plot_training_validation()
    plot_model_evaluation()
    plot_annotated_ecg()
    
    print()
    print("=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    generate_all_plots()

