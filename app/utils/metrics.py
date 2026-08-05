"""
Metrics Calculation - Pure Python implementations
"""

import numpy as np
import math
from collections import Counter


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    # Convert to list if numpy arrays
    if hasattr(y_true, 'tolist'):
        y_true = y_true.tolist()
    if hasattr(y_pred, 'tolist'):
        y_pred = y_pred.tolist()
    
    # Ensure same length
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    n = len(y_true)
    if n == 0:
        return {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'support': 0
        }
    
    # Get unique classes
    classes = sorted(set(y_true + y_pred))
    
    # Build confusion matrix
    cm = {}
    for c1 in classes:
        cm[c1] = {}
        for c2 in classes:
            cm[c1][c2] = 0
    
    for t, p in zip(y_true, y_pred):
        cm[t][p] = cm[t].get(p, 0) + 1
    
    # Calculate metrics per class
    metrics = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for c in classes:
        tp = cm.get(c, {}).get(c, 0)
        fp = sum(cm.get(other, {}).get(c, 0) for other in classes if other != c)
        fn = sum(cm.get(c, {}).get(other, 0) for other in classes if other != c)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[c] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(cm.get(c, {}).values())
        }
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # Calculate overall metrics
    total_correct = sum(cm.get(c, {}).get(c, 0) for c in classes)
    accuracy = total_correct / n if n > 0 else 0
    
    macro_precision = sum(m['precision'] for m in metrics.values()) / len(classes) if classes else 0
    macro_recall = sum(m['recall'] for m in metrics.values()) / len(classes) if classes else 0
    macro_f1 = sum(m['f1'] for m in metrics.values()) / len(classes) if classes else 0
    
    # Micro metrics (weighted by support)
    total_support = sum(m['support'] for m in metrics.values())
    if total_support > 0:
        micro_precision = sum(m['precision'] * m['support'] for m in metrics.values()) / total_support
        micro_recall = sum(m['recall'] * m['support'] for m in metrics.values()) / total_support
        if micro_precision + micro_recall > 0:
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        else:
            micro_f1 = 0
    else:
        micro_precision = micro_recall = micro_f1 = 0
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(micro_precision, 4),
        'recall': round(micro_recall, 4),
        'f1': round(micro_f1, 4),
        'macro_precision': round(macro_precision, 4),
        'macro_recall': round(macro_recall, 4),
        'macro_f1': round(macro_f1, 4),
        'classes': metrics,
        'confusion_matrix': cm,
        'total_samples': n
    }


def calculate_auc_roc(y_true, y_scores):
    """Calculate AUC-ROC"""
    # Convert to list if numpy arrays
    if hasattr(y_true, 'tolist'):
        y_true = y_true.tolist()
    if hasattr(y_scores, 'tolist'):
        y_scores = y_scores.tolist()
    
    # Create pairs and sort by score
    pairs = sorted(zip(y_scores, y_true), key=lambda x: x[0], reverse=True)
    
    n_positive = sum(y_true)
    n_negative = len(y_true) - n_positive
    
    if n_positive == 0 or n_negative == 0:
        return 0.5
    
    # Calculate ROC
    tp = 0
    fp = 0
    tpr_list = []
    fpr_list = []
    
    prev_score = None
    for score, label in pairs:
        if score != prev_score:
            tpr_list.append(tp / n_positive if n_positive > 0 else 0)
            fpr_list.append(fp / n_negative if n_negative > 0 else 0)
            prev_score = score
        
        if label == 1:
            tp += 1
        else:
            fp += 1
    
    tpr_list.append(tp / n_positive if n_positive > 0 else 0)
    fpr_list.append(fp / n_negative if n_negative > 0 else 0)
    
    # Calculate AUC using trapezoidal rule
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return min(max(auc, 0), 1)