# 📊 Instructions for Running ppt_diagram.py in IDLE

## Quick Start

### Option 1: Run in IDLE (Recommended for Windows)

1. **Open IDLE:**
   - Press `Windows Key` and type "IDLE"
   - Open "IDLE (Python 3.11)" or your Python version

2. **Open the File:**
   - In IDLE, go to `File` → `Open`
   - Navigate to: `C:\Users\rd414\ecg_analyzer\ppt_diagram.py`
   - Click `Open`

3. **Run the Script:**
   - Press `F5` or go to `Run` → `Run Module`
   - The script will generate all plots in `reports/plots/` folder

4. **View Results:**
   - All plots will be saved as PNG files in `reports/plots/`
   - The Plotly interactive graph will be saved as HTML

### Option 2: Run from Command Line

```bash
cd C:\Users\rd414\ecg_analyzer
python ppt_diagram.py
```

### Option 3: Run from Python Interactive Shell

```python
# In IDLE or Python shell:
import sys
sys.path.append(r'C:\Users\rd414\ecg_analyzer')
from ppt_diagram import generate_all_plots
generate_all_plots()
```

## 📁 Output Files

After running, you'll find these files in `reports/plots/`:

1. `1_dataset_composition.png` - Bar chart and pie chart
2. `2_signal_preprocessing.png` - Raw vs filtered signal
3. `3_feature_extraction.png` - Correlation heatmap and boxplot
4. `4_training_validation.png` - Accuracy and loss curves
5. `5a_confusion_matrix.png` - Confusion matrices
6. `5b_roc_curves.png` - ROC curves
7. `6_annotated_ecg_plotly.html` - Interactive ECG plot (open in browser)

## ⚠️ Troubleshooting

### Warning: FixedFormatter
- **Fixed**: The code now properly sets tick locations before labels
- This warning has been resolved

### Error: kaleido not installed
- **Not critical**: The script will still save the HTML version
- To enable PNG export: `pip install kaleido`
- The HTML file can be opened in any browser

### Missing Dependencies
If you get import errors, install:
```bash
pip install matplotlib seaborn plotly numpy pandas scipy
```

## 🎨 Customization

To modify the plots, edit `ppt_diagram.py`:
- Colors: Modify `MEDICAL_COLORS` dictionary
- Dummy data: Modify the data generation sections
- Plot styles: Modify `create_medical_axes()` function

## 📝 Notes

- All plots use consistent medical styling (light blue background, white cards)
- Dummy data is used for demonstration
- Replace dummy data with real data from your analysis
- All plots are saved at 300 DPI for publication quality

