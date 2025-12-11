# 🫀 ECG Analyzer - Advanced Arrhythmia Detection System

A full-stack web application for analyzing ECG signals and detecting arrhythmias using machine learning. Built with Flask, Python, and modern web technologies.

![ECG Analyzer](https://img.shields.io/badge/ECG-Analyzer-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0-red)
![License](https://img.shields.io/badge/License-Educational-yellow)

## 🌟 Features

### Core Functionality
- **Real-time ECG Analysis**: Upload and analyze ECG signals instantly
- **Arrhythmia Detection**: Detects 5 types of arrhythmias (Normal, PVC, SVE, Fusion, Unclassifiable)
- **Heart Rate Variability**: Computes SDNN, RMSSD, pNN50 metrics
- **Interactive Visualizations**: Plotly-based interactive charts with zoom and pan
- **Clinical Interpretation**: Automated clinical findings and recommendations
- **Beat-Level Analysis**: Detailed analysis of every heartbeat with confidence scores

### Advanced Features
- **Signal Quality Assessment**: Automated signal quality scoring
- **Arrhythmia Burden Calculation**: PVC and SVE burden percentages
- **Tachycardia/Bradycardia Detection**: Episode detection and duration
- **QRS Complex Analysis**: QRS width estimation
- **Export Capabilities**: Download results as JSON reports

### User Interface
- **Beautiful Modern Design**: Gradient backgrounds, card-based layout
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Drag & Drop Upload**: Easy file upload with drag and drop
- **Interactive Tables**: Searchable and filterable beat-level data
- **Real-time Updates**: Loading indicators and status messages

## 🏗️ Project Structure

```
ecg_analyzer/
│
├── app.py                          # Flask main application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── static/                         # Frontend assets
│   ├── css/
│   │   └── style.css              # Custom styles
│   └── js/
│       └── main.js                # Frontend JavaScript
│
├── templates/
│   └── index.html                 # Main dashboard
│
├── backend/                        # Backend processing modules
│   ├── __init__.py
│   ├── utils.py                   # Helper functions
│   ├── preprocess.py              # Signal preprocessing
│   ├── features.py                # Feature extraction
│   ├── model_pipeline.py          # ML model
│   ├── infer_ecg.py              # Inference engine
│   ├── visualize_ecg.py          # Visualization
│   └── main_train_eval.py        # Training script
│
├── models/                         # Trained models
│   └── ecg_rf_pipeline.joblib    # Random Forest model
│
├── reports/                        # Training reports
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── training_report.json
│
├── outputs/                        # Inference outputs
│   └── (user analysis results)
│
└── data/
    ├── mit-bih/                   # Training dataset
    │   ├── 100.dat
    │   ├── 100.hea
    │   └── ...
    └── user_inputs/               # Uploaded files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) MIT-BIH Arrhythmia Database for training

### Installation

1. **Clone or create the project**
```bash
mkdir ecg_analyzer
cd ecg_analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Create necessary directories**
```bash
mkdir -p data/mit-bih data/user_inputs models reports outputs
mkdir -p static/css static/js templates backend
```

5. **Set up the files**
Copy all the provided files into their respective directories as shown in the project structure.

### Training the Model (Optional)

If you have the MIT-BIH Arrhythmia Database:

1. **Download MIT-BIH Database**
   - Visit: https://physionet.org/content/mitdb/1.0.0/
   - Download .dat, .hea, and .atr files
   - Place them in `data/mit-bih/` folder

2. **Train the model**
```bash
python backend/main_train_eval.py
```

This will:
- Load and preprocess all MIT-BIH records
- Extract features from ECG beats
- Train a Random Forest classifier
- Save the model to `models/ecg_rf_pipeline.joblib`
- Generate training reports and visualizations

**Note**: If you don't have the MIT-BIH database, the training script will create a synthetic dataset for demonstration purposes.

### Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to the URL to start analyzing ECG signals!

## 📖 Usage Guide

### Uploading ECG Files

1. **Click on the upload area** or **drag and drop** a `.dat` file
2. Supported formats:
   - `.dat` - ECG signal data (MIT-BIH format)
   - `.hea` - Header file (optional, contains metadata)
   - `.atr` - Annotation file (for training only)

3. **If header file is missing:**
   - A modal will appear asking for signal parameters
   - Enter sampling frequency (default: 360 Hz)
   - Enter number of channels (default: 2)
   - Select data type (default: 16-bit Integer)

### Analyzing ECG

1. After upload, click **"Analyze ECG"** button
2. Wait for processing (typically 5-30 seconds depending on signal length)
3. View results in multiple sections:

### Understanding the Results

#### 1. Clinical Interpretation
- **Severity Level**: Low, Moderate, or High
- **Findings**: List of detected conditions
- **Recommendations**: Clinical suggestions

#### 2. Summary Metrics
- **Total Beats**: Number of heartbeats detected
- **Mean Heart Rate**: Average HR in bpm
- **SDNN**: Standard deviation of RR intervals (HRV metric)
- **RMSSD**: Root mean square of successive differences
- **pNN50**: Percentage of successive RR intervals differing by >50ms
- **PVC Burden**: Percentage of premature ventricular contractions
- **SVE Burden**: Percentage of supraventricular ectopic beats
- **Signal Quality**: Quality score out of 100

#### 3. ECG Waveform
- Interactive plot showing the entire ECG signal
- Beats are color-coded by class:
  - 🟢 Green: Normal beats
  - 🔴 Red: PVC (Ventricular ectopic)
  - 🟠 Orange: SVE (Supraventricular ectopic)
  - 🟣 Purple: Fusion beats
  - ⚪ Gray: Unclassifiable
- Hover over beats to see details
- Use zoom and pan tools

#### 4. Detailed Analysis
- **Heart Rate Trend**: HR over time with tachycardia/bradycardia thresholds
- **RR Interval Distribution**: Histogram of beat-to-beat intervals
- **Beat Classification**: Pie chart showing arrhythmia distribution
- **Arrhythmia Timeline**: Timeline view of abnormal beats

#### 5. Beat-Level Table
- Detailed information for each beat
- Search and filter capabilities
- Paginated view (50 beats per page)
- Columns:
  - Index: Beat number
  - Time: Timestamp in seconds
  - Class: Beat classification
  - Confidence: Model confidence score
  - RR Interval: Time since previous beat
  - Heart Rate: Instantaneous HR

### Downloading Results

- **JSON Report**: Complete analysis results in JSON format
- **PDF Report**: Print-friendly report (use browser's print function)

## 🧠 Technical Details

### Machine Learning Model

**Algorithm**: Random Forest Classifier
- 200 trees
- Max depth: 20
- Balanced class weights
- Features: 39 time, frequency, and morphological features

**Feature Categories**:
1. **Time Domain** (12 features)
   - Duration, amplitude statistics
   - Energy, zero-crossing rate
   - Peak detection

2. **Frequency Domain** (7 features)
   - FFT-based features
   - Power in frequency bands
   - Spectral entropy and centroid

3. **Morphological** (10 features)
   - QRS complex characteristics
   - R, Q, S wave positions and amplitudes
   - Waveform shape descriptors

4. **RR Intervals** (10 features)
   - RR interval and ratios
   - Heart rate calculations
   - RR differences

### Signal Processing Pipeline

1. **Baseline Wander Removal**: High-pass filter at 0.5 Hz
2. **Bandpass Filtering**: 0.5-50 Hz to remove noise
3. **Notch Filtering**: Remove 50/60 Hz powerline interference
4. **R-Peak Detection**: Using Pan-Tompkins or biosppy algorithm
5. **Beat Segmentation**: Extract 0.2s before to 0.4s after R-peak
6. **Normalization**: Z-score normalization

### HRV Analysis

**Time Domain Metrics**:
- **SDNN**: Standard deviation of normal-to-normal intervals
  - Normal: >100 ms
  - Reduced: 50-100 ms
  - Severely reduced: <50 ms

- **RMSSD**: Root mean square of successive differences
  - Indicates parasympathetic activity
  - Normal: >30 ms

- **pNN50**: % of successive RR intervals differing by >50ms
  - Normal: >10%
  - Reduced: <5%

## 🎨 Customization

### Changing Color Scheme

Edit `config.py`:
```python
CLASS_COLORS = {
    'N': '#your-color',  # Normal
    'V': '#your-color',  # PVC
    'S': '#your-color',  # SVE
    'F': '#your-color',  # Fusion
    'Q': '#your-color'   # Unclassifiable
}
```

### Adjusting Model Parameters

Edit `config.py`:
```python
N_ESTIMATORS = 200      # Number of trees
MAX_DEPTH = 20          # Maximum tree depth
BANDPASS_LOW = 0.5      # Low frequency cutoff
BANDPASS_HIGH = 50.0    # High frequency cutoff
```

### Adding New Arrhythmia Classes

1. Update `utils.py`:
   - Modify `map_beat_annotation()` function
   - Add new symbol mappings

2. Update `config.py`:
   - Add to `ARRHYTHMIA_CLASSES` dictionary
   - Add color to `CLASS_COLORS`

3. Retrain model with new classes

## 🐛 Troubleshooting

### Model Not Found Error
```
Model not found at models/ecg_rf_pipeline.joblib
```
**Solution**: Run the training script:
```bash
python backend/main_train_eval.py
```

### File Upload Fails
- Check file format (.dat files only)
- Ensure file size is under 50MB
- Verify file permissions

### No Beats Detected
- Check if signal is valid ECG data
- Verify sampling frequency is correct
- Try different preprocessing parameters

### Poor Signal Quality
- Ensure good electrode contact
- Check for motion artifacts
- Filter out noisy segments

## 📊 Performance

- **Analysis Speed**: ~1-5 seconds for 30-minute ECG
- **Accuracy**: >95% on MIT-BIH test set (typical)
- **Memory Usage**: ~200-500MB during analysis
- **Supported File Size**: Up to 50MB (configurable)

## 🔒 Security Notes

**For College Project Use**:
- No authentication/authorization implemented
- File uploads not sanitized beyond basic checks
- Suitable for local/demo use only
- **Do NOT deploy to production without security hardening**

## 📝 License

This project is for educational purposes only. Not intended for clinical use.

## 🤝 Contributing

This is a college project. Feel free to:
- Report bugs
- Suggest features
- Submit improvements

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Contact your project advisor

## 🎓 Academic Use

### Citation
If you use this project in your research or coursework, please cite:
```
ECG Analyzer - Advanced Arrhythmia Detection System
[Your Name], [Year]
[Your Institution]
```

### Dataset
The MIT-BIH Arrhythmia Database is available at:
https://physionet.org/content/mitdb/1.0.0/

## 🚀 Future Enhancements

Possible improvements for future versions:
- [ ] Real-time ECG monitoring
- [ ] Multi-lead ECG support
- [ ] Deep learning models (CNN, LSTM)
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] User authentication
- [ ] Database integration
- [ ] Advanced HRV analysis
- [ ] Report generation in multiple formats
- [ ] Batch processing support

## ✨ Acknowledgments

- MIT-BIH Arrhythmia Database (PhysioNet)
- scikit-learn for machine learning
- Plotly for visualizations
- Flask for web framework
- biosppy for signal processing

---

**Made with ❤️ for Academic Excellence**