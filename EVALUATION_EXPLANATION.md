# How Model Evaluation Works

## Current Implementation

### What We're Currently Doing

**Right now, we're NOT actually re-evaluating the model.** Instead, we're:

1. **Loading pre-calculated metrics** from `reports/training_report.json`
2. This file was created **during training** when you ran `backend/main_train_eval.py`
3. The metrics are **static** - they represent the model's performance on the test set from training time

### How Evaluation Was Done During Training

When you ran `python backend/main_train_eval.py`, here's what happened:

```
1. Load MIT-BIH Dataset (108,370 beats from 48 records)
   ↓
2. Split Data:
   - Training: 80% (~86,696 beats)
   - Validation: 10% (~10,837 beats) 
   - Test: 20% (~21,674 beats)
   ↓
3. Train Random Forest Model on training set
   ↓
4. Evaluate on Test Set:
   - Load test data (X_test, y_test)
   - Make predictions: y_pred = model.predict(X_test)
   - Calculate metrics:
     * Accuracy = accuracy_score(y_test, y_pred)
     * Precision = precision_score(y_test, y_pred, average='weighted')
     * Recall = recall_score(y_test, y_pred, average='weighted')
     * F1-Score = f1_score(y_test, y_pred, average='weighted')
   - Generate confusion matrix
   - Generate per-class metrics
   ↓
5. Save Results to reports/training_report.json
```

### The Evaluation Code (from `backend/model_pipeline.py`)

```python
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    return metrics
```

### Current Metrics Source

The metrics displayed on the frontend come from:
- **File**: `reports/training_report.json`
- **Generated**: During training (when you ran `main_train_eval.py`)
- **Test Set Size**: 21,674 beats
- **Results**:
  - Accuracy: 98.80%
  - Precision: 98.77%
  - Recall: 98.80%
  - F1-Score: 98.77%

## What We're NOT Doing (But Could)

Currently, `evaluate_model_on_test_data()` in `backend/evaluate_model.py` is **not implemented**. It just returns `None`.

To actually re-evaluate the model, we would need to:

1. **Load the test set** (or hold it in memory)
2. **Load the trained model**
3. **Run predictions** on the test set
4. **Calculate metrics** fresh
5. **Return the metrics**

This would be useful for:
- Re-evaluating after model updates
- Checking if performance has changed
- Getting fresh metrics on-demand

## Summary

| Aspect | Current Implementation |
|--------|----------------------|
| **Evaluation Method** | Loading from pre-generated JSON file |
| **When Evaluation Happens** | During training (one-time) |
| **Test Set** | 21,674 beats from MIT-BIH (20% split) |
| **Metrics Calculation** | Done during training, saved to JSON |
| **Frontend Display** | Reads from JSON file via API |
| **Real-time Evaluation** | ❌ Not implemented |

## Options to Improve

### Option 1: Keep Current Approach (Recommended)
- ✅ Fast (no computation needed)
- ✅ Metrics are from actual test set
- ✅ No need to reload test data
- ❌ Metrics are static (don't update)

### Option 2: Implement Real-time Evaluation
- ✅ Get fresh metrics on-demand
- ✅ Can evaluate after model updates
- ❌ Requires loading test data each time
- ❌ Slower (need to run predictions)
- ❌ Need to store/load test set

Would you like me to implement Option 2 (real-time evaluation)?





