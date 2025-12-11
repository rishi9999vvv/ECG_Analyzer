"""
ECG Visualization Module
Creates interactive plots and charts for ECG analysis results
"""

import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import plotly.utils

def create_ecg_plot(signal_data, r_peaks=None, beat_classifications=None, 
                    fs=360, duration=10):
    """
    Create interactive ECG signal plot with annotations
    
    Parameters:
    -----------
    signal_data : array
        ECG signal data
    r_peaks : array
        R-peak locations
    beat_classifications : list
        Beat classification results
    fs : int
        Sampling frequency
    duration : int
        Duration to display (seconds)
    
    Returns:
    --------
    plot_json : str
        JSON representation of plotly figure
    """
    # Limit signal to specified duration
    max_samples = min(len(signal_data), duration * fs)
    time = np.arange(max_samples) / fs
    signal_slice = signal_data[:max_samples]
    
    # Create figure
    fig = go.Figure()
    
    # Add ECG trace
    fig.add_trace(go.Scatter(
        x=time,
        y=signal_slice,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#3b82f6', width=1.5),
        hovertemplate='Time: %{x:.2f}s<br>Amplitude: %{y:.3f} mV<extra></extra>'
    ))
    
    # Add R-peaks with classifications
    if r_peaks is not None and len(r_peaks) > 0:
        # Filter peaks within display window
        visible_peaks = r_peaks[r_peaks < max_samples]
        
        if len(visible_peaks) > 0:
            peak_times = visible_peaks / fs
            peak_amplitudes = signal_slice[visible_peaks]
            
            # Color-code by classification
            colors = []
            hover_texts = []
            
            if beat_classifications:
                for i, peak_idx in enumerate(visible_peaks):
                    if i < len(beat_classifications):
                        beat = beat_classifications[i]
                        label = beat.get('label', 'Unknown')
                        confidence = beat.get('confidence', 0)
                        
                        # Assign colors
                        color_map = {
                            'Normal': '#10b981',
                            'PVC': '#ef4444',
                            'SVE': '#3b82f6',
                            'Fusion': '#8b5cf6',
                            'Unclassifiable': '#6b7280'
                        }
                        colors.append(color_map.get(label, '#6b7280'))
                        
                        hover_texts.append(
                            f"Beat #{i+1}<br>"
                            f"Type: {label}<br>"
                            f"Confidence: {confidence:.2%}<br>"
                            f"Time: {peak_times[i]:.2f}s"
                        )
                    else:
                        colors.append('#6b7280')
                        hover_texts.append(f"Beat #{i+1}")
            else:
                colors = ['#10b981'] * len(visible_peaks)
                hover_texts = [f"R-peak at {t:.2f}s" for t in peak_times]
            
            # Add R-peak markers
            fig.add_trace(go.Scatter(
                x=peak_times,
                y=peak_amplitudes,
                mode='markers',
                name='R-peaks',
                marker=dict(
                    color=colors,
                    size=10,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))
    
    # Add grid lines for clinical measurements
    add_clinical_grid(fig, time[-1] if len(time) > 0 else duration)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'ECG Signal Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f2937'}
        },
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            minor=dict(showgrid=True, gridcolor='#f3f4f6', gridwidth=0.5),
            zeroline=False,
            tickformat='.1f'
        ),
        yaxis=dict(
            title='Amplitude (mV)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            minor=dict(showgrid=True, gridcolor='#f3f4f6', gridwidth=0.5),
            zeroline=True,
            zerolinecolor='#9ca3af',
            zerolinewidth=1
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        margin=dict(l=60, r=30, t=60, b=60)
    )
    
    # Add range selector buttons
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1s", step="second", stepmode="backward"),
                dict(count=5, label="5s", step="second", stepmode="backward"),
                dict(count=10, label="10s", step="second", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor='white',
            activecolor='#e0e7ff',
            x=0,
            y=1.15
        )
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_distribution_chart(beat_distribution):
    """
    Create beat distribution pie chart
    
    Parameters:
    -----------
    beat_distribution : dict
        Dictionary with beat types and counts
    
    Returns:
    --------
    plot_json : str
        JSON representation of plotly figure
    """
    if not beat_distribution:
        return None
    
    # Prepare data
    labels = list(beat_distribution.keys())
    values = list(beat_distribution.values())
    
    # Define colors
    color_map = {
        'Normal': '#10b981',
        'PVC': '#ef4444',
        'SVE': '#3b82f6',
        'Fusion': '#8b5cf6',
        'Unclassifiable': '#6b7280'
    }
    colors = [color_map.get(label, '#9ca3af') for label in labels]
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='white', width=2)),
        textfont=dict(size=12, color='white'),
        textposition='inside',
        texttemplate='%{label}<br>%{value}<br>(%{percent})',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Beat Classification Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#1f2937'}
        },
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        margin=dict(l=20, r=120, t=60, b=20),
        height=350,
        paper_bgcolor='white'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_hrv_plot(rr_intervals, fs=360):
    """
    Create HRV analysis plots
    
    Parameters:
    -----------
    rr_intervals : array
        RR intervals in milliseconds
    fs : int
        Sampling frequency
    
    Returns:
    --------
    plot_json : str
        JSON representation of plotly figure
    """
    if len(rr_intervals) < 2:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RR Intervals Over Time', 'RR Interval Distribution',
                       'Poincaré Plot', 'RR Interval Differences'),
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. RR intervals over time
    time_rr = np.cumsum(rr_intervals) / 1000  # Convert to seconds
    fig.add_trace(
        go.Scatter(
            x=time_rr[:-1],
            y=rr_intervals[:-1],
            mode='lines+markers',
            name='RR Intervals',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4),
            hovertemplate='Time: %{x:.2f}s<br>RR Interval: %{y:.0f}ms<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. RR interval histogram
    fig.add_trace(
        go.Histogram(
            x=rr_intervals,
            nbinsx=30,
            name='Distribution',
            marker_color='#10b981',
            hovertemplate='RR Interval: %{x:.0f}ms<br>Count: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Poincaré plot
    if len(rr_intervals) > 1:
        fig.add_trace(
            go.Scatter(
                x=rr_intervals[:-1],
                y=rr_intervals[1:],
                mode='markers',
                name='Poincaré',
                marker=dict(
                    size=5,
                    color=rr_intervals[:-1],
                    colorscale='Viridis',
                    showscale=False
                ),
                hovertemplate='RR(n): %{x:.0f}ms<br>RR(n+1): %{y:.0f}ms<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add diagonal reference line
        min_rr = min(rr_intervals)
        max_rr = max(rr_intervals)
        fig.add_trace(
            go.Scatter(
                x=[min_rr, max_rr],
                y=[min_rr, max_rr],
                mode='lines',
                name='Identity',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 4. RR differences
    rr_diff = np.diff(rr_intervals)
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rr_diff)),
            y=rr_diff,
            mode='lines',
            name='RR Differences',
            line=dict(color='#ef4444', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(239, 68, 68, 0.2)',
            hovertemplate='Beat: %{x}<br>ΔRR: %{y:.0f}ms<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1, showgrid=True, gridcolor='#e5e7eb')
    fig.update_xaxes(title_text="RR Interval (ms)", row=1, col=2, showgrid=True, gridcolor='#e5e7eb')
    fig.update_xaxes(title_text="RR(n) (ms)", row=2, col=1, showgrid=True, gridcolor='#e5e7eb')
    fig.update_xaxes(title_text="Beat Number", row=2, col=2, showgrid=True, gridcolor='#e5e7eb')
    
    fig.update_yaxes(title_text="RR Interval (ms)", row=1, col=1, showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(title_text="Count", row=1, col=2, showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(title_text="RR(n+1) (ms)", row=2, col=1, showgrid=True, gridcolor='#e5e7eb')
    fig.update_yaxes(title_text="ΔRR (ms)", row=2, col=2, showgrid=True, gridcolor='#e5e7eb')
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Heart Rate Variability Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1f2937'}
        },
        showlegend=False,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def add_clinical_grid(fig, max_time):
    """
    Add clinical measurement grid to ECG plot
    """
    # Add 200ms grid lines (5 large squares at 25mm/s)
    for i in np.arange(0, max_time, 0.2):
        fig.add_vline(
            x=i,
            line=dict(color='#fca5a5', width=0.5, dash='dot'),
            layer='below'
        )
    
    # Add 40ms grid lines (1 small square at 25mm/s)
    for i in np.arange(0, max_time, 0.04):
        fig.add_vline(
            x=i,
            line=dict(color='#fecaca', width=0.3, dash='dot'),
            layer='below'
        )

def create_rhythm_strip(signal_data, r_peaks, fs=360, duration=30):
    """
    Create a longer rhythm strip for detailed analysis
    """
    # Similar to create_ecg_plot but optimized for longer duration
    return create_ecg_plot(signal_data, r_peaks, None, fs, duration)

def create_interactive_ecg_plot(signal_data, r_peaks=None, beat_classifications=None, 
                                fs=360, duration=10):
    """
    Create enhanced interactive Plotly ECG graph with zoomable features and beat annotations
    
    Parameters:
    -----------
    signal_data : array
        ECG signal data
    r_peaks : array
        R-peak locations
    beat_classifications : list
        Beat classification results with type (N, V, S, F, Q)
    fs : int
        Sampling frequency
    duration : int
        Duration to display (seconds)
    
    Returns:
    --------
    plot_json : str
        JSON representation of plotly figure
    """
    # Limit signal to specified duration
    max_samples = min(len(signal_data), duration * fs)
    time = np.arange(max_samples) / fs
    signal_slice = signal_data[:max_samples]
    
    # Create figure
    fig = go.Figure()
    
    # Add ECG trace with better styling
    fig.add_trace(go.Scatter(
        x=time,
        y=signal_slice,
        mode='lines',
        name='ECG Signal',
        line=dict(color='#1e40af', width=1.5),
        hovertemplate='<b>Time:</b> %{x:.3f}s<br><b>Amplitude:</b> %{y:.3f} mV<extra></extra>',
        showlegend=True
    ))
    
    # Add beat annotations with color coding
    if r_peaks is not None and len(r_peaks) > 0 and beat_classifications:
        # Filter peaks within display window
        visible_peaks = r_peaks[r_peaks < max_samples]
        
        if len(visible_peaks) > 0:
            peak_times = visible_peaks / fs
            peak_amplitudes = signal_slice[visible_peaks]
            
            # Color map for beat types
            color_map = {
                'N': '#10b981',  # Green - Normal
                'V': '#ef4444',  # Red - PVC
                'S': '#3b82f6',  # Blue - SVE
                'F': '#8b5cf6',  # Purple - Fusion
                'Q': '#6b7280'   # Gray - Unclassifiable
            }
            
            # Symbol map for beat types
            symbol_map = {
                'N': 'circle',
                'V': 'diamond',
                'S': 'square',
                'F': 'star',
                'Q': 'x'
            }
            
            # Group beats by type for separate traces
            # Valid beat types: N, V, S, F, Q only (no P or other annotations)
            valid_beat_types = ['N', 'V', 'S', 'F', 'Q']
            beat_groups = {}
            for i, peak_idx in enumerate(visible_peaks):
                if i < len(beat_classifications):
                    beat = beat_classifications[i]
                    beat_type = beat.get('type', 'Q')
                    # Filter out invalid beat types (like 'P' which is a P-wave, not a beat type)
                    if beat_type not in valid_beat_types:
                        beat_type = 'Q'  # Map invalid types to Unclassifiable
                    beat_label = beat.get('label', 'Unknown')
                    confidence = beat.get('confidence', 0)
                    
                    if beat_type not in beat_groups:
                        beat_groups[beat_type] = {
                            'times': [],
                            'amplitudes': [],
                            'hover_texts': []
                        }
                    
                    beat_groups[beat_type]['times'].append(peak_times[i])
                    beat_groups[beat_type]['amplitudes'].append(peak_amplitudes[i])
                    
                    hover_text = (
                        f"<b>Beat #{i+1}</b><br>"
                        f"Type: {beat_label} ({beat_type})<br>"
                        f"Confidence: {confidence:.1%}<br>"
                        f"Time: {peak_times[i]:.3f}s"
                    )
                    beat_groups[beat_type]['hover_texts'].append(hover_text)
            
            # Add trace for each beat type
            beat_labels = {
                'N': 'Normal (N)',
                'V': 'PVC (V)',
                'S': 'SVE (S)',
                'F': 'Fusion (F)',
                'Q': 'Unclassifiable (Q)'
            }
            
            for beat_type, data in beat_groups.items():
                fig.add_trace(go.Scatter(
                    x=data['times'],
                    y=data['amplitudes'],
                    mode='markers',
                    name=beat_labels.get(beat_type, beat_type),
                    marker=dict(
                        color=color_map.get(beat_type, '#6b7280'),
                        size=12,
                        symbol=symbol_map.get(beat_type, 'circle'),
                        line=dict(color='white', width=1.5),
                        opacity=0.9
                    ),
                    text=data['hover_texts'],
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))
    
    # Add clinical grid lines
    add_clinical_grid(fig, time[-1] if len(time) > 0 else duration)
    
    # Update layout with enhanced interactivity
    fig.update_layout(
        title={
            'text': 'Interactive ECG Signal with Beat Annotations',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1f2937', 'family': 'Inter'}
        },
        xaxis=dict(
            title='Time (seconds)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            minor=dict(showgrid=True, gridcolor='#f3f4f6', gridwidth=0.5),
            zeroline=False,
            tickformat='.2f',
            rangeslider=dict(visible=False),  # Can enable for range slider
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1s", step="second", stepmode="backward"),
                    dict(count=5, label="5s", step="second", stepmode="backward"),
                    dict(count=10, label="10s", step="second", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(255,255,255,0.8)',
                activecolor='#3b82f6',
                x=0,
                y=1.15
            )
        ),
        yaxis=dict(
            title='Amplitude (mV)',
            showgrid=True,
            gridcolor='#e5e7eb',
            gridwidth=1,
            minor=dict(showgrid=True, gridcolor='#f3f4f6', gridwidth=0.5),
            zeroline=True,
            zerolinecolor='#9ca3af',
            zerolinewidth=1
        ),
        hovermode='closest',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1,
            font=dict(size=11)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        margin=dict(l=70, r=30, t=80, b=70),
        # Enable zoom and pan
        dragmode='zoom',
        # Add zoom buttons
        modebar_add=[
            'zoom2d',
            'pan2d',
            'select2d',
            'lasso2d',
            'zoomIn2d',
            'zoomOut2d',
            'autoScale2d',
            'resetScale2d'
        ]
    )
    
    # Add annotations for beat types in legend area
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'ecg_plot',
            'height': 500,
            'width': 1200,
            'scale': 2
        }
    }
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_annotated_ecg_plot(signal_data, r_peaks=None, beat_classifications=None, 
                              fs=360, duration=10):
    """
    Create a high-quality annotated ECG plot using matplotlib with proper beat annotations
    
    This function creates a static, consistent, and well-annotated ECG graph with:
    - Proper beat type labels (N, V, S, F, Q)
    - Color-coded annotations
    - Clear legend
    - Clinical grid lines
    
    Parameters:
    -----------
    signal_data : array
        ECG signal data
    r_peaks : array
        R-peak locations
    beat_classifications : list
        Beat classification results with type (N, V, S, F, Q)
    fs : int
        Sampling frequency
    duration : int
        Duration to display (seconds)
    
    Returns:
    --------
    plot_base64 : str
        Base64 encoded PNG image of the plot
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO
    import base64
    
    # Limit signal to specified duration
    max_samples = min(len(signal_data), duration * fs)
    time = np.arange(max_samples) / fs
    signal_slice = signal_data[:max_samples].copy()  # Explicit copy to avoid any modifications
    
    # Debug: Verify signal properties (should NOT be normalized - std should be > 0.1 typically)
    print(f"  📈 Plotting ECG signal:")
    print(f"     Length: {len(signal_slice)} samples")
    print(f"     Mean: {np.mean(signal_slice):.4f}")
    print(f"     Std: {np.std(signal_slice):.4f} (should NOT be ~1.0 if not normalized)")
    print(f"     Min: {np.min(signal_slice):.4f}, Max: {np.max(signal_slice):.4f}")
    print(f"     Range: {np.max(signal_slice) - np.min(signal_slice):.4f}")
    
    # Create figure with proper sizing
    fig, ax = plt.subplots(figsize=(14, 6), dpi=100)
    fig.patch.set_facecolor('white')
    
    # Plot ECG signal - use the clean preprocessed signal directly (NO normalization)
    ax.plot(time, signal_slice, color='#1e40af', linewidth=1.2, label='ECG Signal', zorder=1)
    
    # Beat type definitions
    beat_type_labels = {
        'N': 'Normal Beats',
        'V': 'PVC',
        'S': 'SVE',
        'F': 'Fusion Beats',
        'Q': 'Unclassifiable'
    }
    
    beat_type_colors = {
        'N': '#10b981',  # Green - Normal
        'V': '#ef4444',  # Red - PVC
        'S': '#3b82f6',  # Blue - SVE
        'F': '#8b5cf6',  # Purple - Fusion
        'Q': '#6b7280'   # Gray - Unclassifiable
    }
    
    beat_type_markers = {
        'N': 'o',   # Circle
        'V': 'D',   # Diamond
        'S': 's',   # Square
        'F': '*',   # Star
        'Q': 'x'    # X
    }
    
    # Group beats by type for annotation
    if r_peaks is not None and len(r_peaks) > 0 and beat_classifications:
        visible_peaks = r_peaks[r_peaks < max_samples]
        
        if len(visible_peaks) > 0:
            peak_times = visible_peaks / fs
            peak_amplitudes = signal_slice[visible_peaks]
            
            # Group beats by type
            # Valid beat types: N, V, S, F, Q only (no P or other annotations)
            valid_beat_types = ['N', 'V', 'S', 'F', 'Q']
            beat_groups = {}
            for i, peak_idx in enumerate(visible_peaks):
                # Find the corresponding beat classification index
                # visible_peaks are filtered from r_peaks, so we need to find the original index
                original_idx = np.where(r_peaks == peak_idx)[0]
                beat_idx = original_idx[0] if len(original_idx) > 0 else i
                
                if beat_idx < len(beat_classifications):
                    beat = beat_classifications[beat_idx]
                    beat_type = beat.get('type', 'Q')
                    # Filter out invalid beat types (like 'P' which is a P-wave, not a beat type)
                    if beat_type not in valid_beat_types:
                        beat_type = 'Q'  # Map invalid types to Unclassifiable
                    
                    if beat_type not in beat_groups:
                        beat_groups[beat_type] = {
                            'times': [],
                            'amplitudes': [],
                            'indices': []
                        }
                    
                    beat_groups[beat_type]['times'].append(peak_times[i])
                    beat_groups[beat_type]['amplitudes'].append(peak_amplitudes[i])
                    beat_groups[beat_type]['indices'].append(beat_idx)
            
            # Plot each beat type with annotations
            for beat_type, data in beat_groups.items():
                if len(data['times']) > 0:
                    color = beat_type_colors.get(beat_type, '#6b7280')
                    marker = beat_type_markers.get(beat_type, 'o')
                    label = beat_type_labels.get(beat_type, beat_type)
                    
                    # Plot markers
                    ax.scatter(data['times'], data['amplitudes'], 
                             c=color, marker=marker, s=100, 
                             edgecolors='white', linewidths=1.5,
                             label=label, zorder=3, alpha=0.9)
                    
                    # Add text annotations for each beat (only first few to avoid clutter)
                    max_annotations = min(10, len(data['times']))
                    for j in range(max_annotations):
                        time_val = data['times'][j]
                        amp_val = data['amplitudes'][j]
                        idx = data['indices'][j]
                        
                        # Add annotation above the marker
                        ax.annotate(f"{beat_type}\n#{idx+1}", 
                                  xy=(time_val, amp_val),
                                  xytext=(time_val, amp_val + 0.3),
                                  fontsize=8, fontweight='bold',
                                  color=color,
                                  ha='center',
                                  bbox=dict(boxstyle='round,pad=0.3', 
                                          facecolor='white', 
                                          edgecolor=color,
                                          alpha=0.8),
                                  arrowprops=dict(arrowstyle='->', 
                                                color=color, 
                                                lw=1.5, 
                                                alpha=0.6),
                                  zorder=4)
    
    # Add clinical grid lines
    # Major grid lines (200ms intervals)
    for t in np.arange(0, time[-1] + 0.2, 0.2):
        ax.axvline(x=t, color='#fecaca', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Minor grid lines (40ms intervals)
    for t in np.arange(0, time[-1] + 0.04, 0.04):
        ax.axvline(x=t, color='#fee2e2', linestyle=':', linewidth=0.3, alpha=0.3, zorder=0)
    
    # Add horizontal grid lines
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    for y in np.arange(y_min, y_max + y_range/10, y_range/10):
        ax.axhline(y=y, color='#e5e7eb', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amplitude (mV)', fontsize=12, fontweight='bold')
    ax.set_title('ECG Signal with Beat Annotations', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, which='major', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.3)
    
    # Legend
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                      fancybox=True, shadow=True, ncol=2)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#e5e7eb')
    
    # Set axis limits
    ax.set_xlim([0, time[-1] if len(time) > 0 else duration])
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    
    return image_base64