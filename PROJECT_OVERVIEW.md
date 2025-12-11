# 🫀 ECG Analyzer - Complete Project Overview

## 📋 Project Summary

**ECG Analyzer** is a full-stack web application for analyzing ECG signals and detecting arrhythmias using machine learning. The system provides real-time ECG analysis, arrhythmia detection, heart rate variability (HRV) metrics, and comprehensive clinical reports.

---

## 🏗️ Project Structure

```
ecg_analyzer/
│
├── 📄 Main Application Files
│   ├── app.py                          # Flask main application (846 lines)
│   ├── config.py                       # Configuration settings
│   ├── requirements.txt                # Python dependencies
│   ├── README.md                       # Comprehensive documentation
│   ├── setup.bat                       # Setup script for Windows
│   └── evaluate_model.py               # Model evaluation script
│
├── 📁 backend/                         # Backend processing modules
│   ├── __init__.py
│   ├── utils.py                        # Helper functions (file handling, utilities)
│   ├── preprocess.py                   # Signal preprocessing & filtering
│   ├── features.py                     # Feature extraction (39 features)
│   ├── model_pipeline.py               # ML model loading & prediction
│   ├── infer_ecg.py                    # Inference engine & HRV analysis
│   ├── visualize_ecg.py                # Visualization (Plotly, Matplotlib)
│   ├── clinical_analysis.py            # Clinical interpretation & reports
│   ├── ai_module.py                    # AI-powered insights
│   └── main_train_eval.py              # Training & evaluation script
│
├── 📁 templates/                       # HTML templates
│   ├── index.html                      # Main dashboard (844 lines)
│   └── learn.html                      # Educational ECG basics page
│
├── 📁 static/                          # Frontend assets
│   ├── css/
│   │   └── style.css                   # Custom styles
│   ├── js/
│   │   └── main.js                     # Frontend JavaScript (530+ lines)
│   └── images/
│       └── logo.png                    # Logo
│
├── 📁 models/                          # Trained ML models
│   └── ecg_rf_pipeline.joblib         # Random Forest model
│
├── 📁 data/                            # Data directories
│   ├── mit-bih/                        # MIT-BIH Arrhythmia Database
│   │   ├── 100.dat, 100.hea, 100.atr   # Training records
│   │   ├── 101.dat, 101.hea, 101.atr
│   │   └── ... (many more records)
│   └── user_inputs/                    # Uploaded user files
│
├── 📁 reports/                          # Training reports & visualizations
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── training_report.json
│   └── plots/                          # Training visualization plots
│
├── 📁 outputs/                          # Analysis results
│   └── *_report.json                   # Generated reports
│
└── 📁 venv/                            # Virtual environment (Python packages)
```

---

## 🔧 Core Components

### 1. **Flask Application (`app.py`)**
- **Main Features:**
  - File upload handling (drag & drop support)
  - ECG signal loading (MIT-BIH, CSV, TXT formats)
  - Signal analysis pipeline
  - Model metrics API
  - PDF report generation
  - Error handling & fallbacks

- **Key Endpoints:**
  - `GET /` - Main dashboard
  - `POST /analyze` - ECG analysis endpoint
  - `GET /model_metrics` - Model performance metrics
  - `POST /ai_insight` - AI-powered insights
  - `POST /export_pdf` - PDF report generation
  - `GET /learn` - Educational page

### 2. **Signal Preprocessing (`backend/preprocess.py`)**
- **Functions:**
  - `preprocess_signal()` - Complete preprocessing pipeline
  - `preprocess_ecg_fixed()` - Fixed preprocessing (bandpass, notch, normalization)
  - `preprocess_ecg_for_visualization()` - Visualization-specific preprocessing
  - `adaptive_r_peak_detection()` - R-peak detection
  - `segment_beats()` - Beat segmentation
  - Baseline wander removal
  - Powerline interference removal (50/60 Hz notch)
  - High-frequency noise filtering
  - Wavelet denoising

### 3. **Feature Extraction (`backend/features.py`)**
- **39 Features Extracted:**
  - **Time Domain (12):** Duration, amplitude statistics, energy, zero-crossing rate
  - **Frequency Domain (7):** FFT features, power in bands, spectral entropy
  - **Morphological (10):** QRS complex, R/Q/S wave positions, waveform shape
  - **RR Intervals (10):** RR intervals, ratios, heart rate, differences

### 4. **Machine Learning Model (`backend/model_pipeline.py`)**
- **Algorithm:** Random Forest Classifier
- **Parameters:**
  - 200 trees
  - Max depth: 20
  - Balanced class weights
- **Classes:** 5 arrhythmia types (N, V, S, F, Q)
- **Performance:** ~98.7% accuracy, 97.2% precision, 96.5% recall

### 5. **Inference Engine (`backend/infer_ecg.py`)**
- HRV metrics computation (SDNN, RMSSD, pNN50)
- Arrhythmia burden calculation
- Tachycardia/Bradycardia detection
- QRS complex analysis
- Signal quality assessment

### 6. **Visualization (`backend/visualize_ecg.py`)**
- Interactive Plotly charts
- Matplotlib static plots
- Annotated ECG waveforms
- Heart rate trends
- RR interval distributions
- Beat classification pie charts
- Arrhythmia timelines

### 7. **Clinical Analysis (`backend/clinical_analysis.py`)**
- Clinical findings generation
- Risk level assessment
- Recommendations generation
- Follow-up scheduling
- Comprehensive clinical reports

### 8. **Frontend (`templates/index.html` + `static/js/main.js`)**
- **UI Features:**
  - Modern, responsive design (Tailwind CSS)
  - Drag & drop file upload
  - Real-time loading indicators
  - Interactive data tables
  - Searchable/filterable beat data
  - Model performance metrics display
  - Interactive visualizations

---

## 🎯 Key Features

### Core Functionality
✅ Real-time ECG Analysis  
✅ Arrhythmia Detection (5 types: Normal, PVC, SVE, Fusion, Unclassifiable)  
✅ Heart Rate Variability (SDNN, RMSSD, pNN50)  
✅ Interactive Visualizations (Plotly)  
✅ Clinical Interpretation  
✅ Beat-Level Analysis with Confidence Scores  

### Advanced Features
✅ Signal Quality Assessment  
✅ Arrhythmia Burden Calculation  
✅ Tachycardia/Bradycardia Detection  
✅ QRS Complex Analysis  
✅ Export Capabilities (JSON, PDF)  
✅ AI-Powered Insights  

### User Interface
✅ Modern Gradient Design  
✅ Responsive (Desktop/Tablet/Mobile)  
✅ Drag & Drop Upload  
✅ Interactive Tables  
✅ Real-time Updates  

---

## 📊 Technical Stack

### Backend
- **Framework:** Flask 3.0.0
- **ML:** scikit-learn 1.3.1
- **Signal Processing:** scipy 1.11.3, biosppy 0.8.0, PyWavelets 1.4.1
- **ECG Data:** wfdb 4.1.2 (PhysioNet)
- **Visualization:** plotly 5.17.0, matplotlib 3.7.2
- **Data:** numpy 1.24.3, pandas 2.0.3

### Frontend
- **Framework:** Vanilla JavaScript
- **Styling:** Tailwind CSS (CDN)
- **Charts:** Plotly.js
- **Icons:** Font Awesome 6.4.0

### Development
- **Python:** 3.8+
- **Package Manager:** pip
- **Virtual Environment:** venv

---

## 🔄 Data Flow

```
1. User Uploads ECG File (.dat)
   ↓
2. File Saved to data/user_inputs/
   ↓
3. Signal Loaded (wfdb or raw binary)
   ↓
4. Preprocessing Pipeline:
   - Baseline removal
   - Noise filtering
   - Normalization
   ↓
5. R-Peak Detection
   ↓
6. Beat Segmentation
   ↓
7. Feature Extraction (39 features per beat)
   ↓
8. ML Model Prediction (Random Forest)
   ↓
9. HRV Analysis
   ↓
10. Clinical Interpretation
   ↓
11. Visualization Generation
   ↓
12. Results Returned to Frontend
```

---

## 📁 File Details

### Main Files
- **`app.py`** (846 lines): Flask application with all endpoints
- **`config.py`** (41 lines): Configuration settings
- **`requirements.txt`** (42 lines): All Python dependencies

### Backend Modules
- **`backend/preprocess.py`**: Signal preprocessing (~500+ lines)
- **`backend/features.py`**: Feature extraction (~395 lines)
- **`backend/model_pipeline.py`**: ML model (~375 lines)
- **`backend/infer_ecg.py`**: Inference engine (~523 lines)
- **`backend/visualize_ecg.py`**: Visualization (~825 lines)
- **`backend/clinical_analysis.py`**: Clinical analysis (~684 lines)
- **`backend/utils.py`**: Utilities (~196 lines)

### Frontend Files
- **`templates/index.html`** (844 lines): Main dashboard
- **`static/js/main.js`** (530+ lines): Frontend JavaScript
- **`static/css/style.css`**: Custom styles

---

## 🎓 Machine Learning Details

### Model Architecture
- **Type:** Random Forest Classifier
- **Classes:** 5 (Normal, PVC, SVE, Fusion, Unclassifiable)
- **Features:** 39 per beat
- **Training Data:** MIT-BIH Arrhythmia Database
- **Performance:**
  - Accuracy: 98.7%
  - Precision: 97.2%
  - Recall: 96.5%
  - F1-Score: 96.8%

### Feature Categories
1. **Time Domain** (12 features)
2. **Frequency Domain** (7 features)
3. **Morphological** (10 features)
4. **RR Intervals** (10 features)

---

## 🚀 Usage

### Installation
```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train model
python backend/main_train_eval.py

# 4. Run application
python app.py
```

### Access Application
- URL: `http://localhost:5000`
- Upload ECG files (.dat format)
- View analysis results

---

## 📈 Project Statistics

- **Total Python Files:** ~15
- **Total Lines of Code:** ~5,000+
- **Backend Modules:** 9
- **Frontend Files:** 3
- **ML Model:** Random Forest (200 trees)
- **Features per Beat:** 39
- **Arrhythmia Classes:** 5
- **Supported Formats:** .dat, .csv, .txt, .hea

---

## 🔐 Security Notes

⚠️ **For Educational Use Only**
- No authentication/authorization
- File uploads not fully sanitized
- Suitable for local/demo use only
- **Do NOT deploy to production without security hardening**

---

## 📝 License

This project is for **educational purposes only**. Not intended for clinical use.

---

## 🎯 Future Enhancements

Possible improvements:
- [ ] Real-time ECG monitoring
- [ ] Multi-lead ECG support
- [ ] Deep learning models (CNN, LSTM)
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] User authentication
- [ ] Database integration
- [ ] Advanced HRV analysis
- [ ] Batch processing support

---

## 📚 Key Dependencies

### Core
- Flask==3.0.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.1
- scipy==1.11.3

### Signal Processing
- wfdb==4.1.2
- biosppy==0.8.0
- PyWavelets==1.4.1

### Visualization
- plotly==5.17.0
- matplotlib==3.7.2

### Utilities
- joblib==1.3.2
- python-dotenv==1.0.0
- reportlab==4.0.7

---

## 📞 Support

For questions or issues:
1. Check the troubleshooting section in README.md
2. Review the code comments
3. Contact your project advisor

---

**Made with ❤️ for Academic Excellence**

*Last Updated: 2024*





