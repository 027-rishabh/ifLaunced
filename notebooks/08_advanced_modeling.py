"""
SpaceX Falcon 9 Booster Landing Prediction - Advanced ML Pipeline
=================================================================
This script implements a comprehensive ML pipeline with:
- 5 different ML models (Logistic Regression, Random Forest, XGBoost, SVM, MLP)
- Advanced evaluation metrics (ROC-AUC, PR-AUC, Cross-Validation)
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Model persistence and inference script

Author: Rishabh Singh
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix, 
    classification_report
)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import os
from pathlib import Path

# Get the project root (parent of notebooks directory)
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Create directories if they don't exist
os.makedirs('src/models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

print("="*80)
print("SPACEX FALCON 9 BOOSTER LANDING PREDICTION - ADVANCED ML PIPELINE")
print("="*80)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("\n[1/8] Loading and preparing data...")

df = pd.read_csv("data/processed/spacex_enriched.csv")

# Feature selection and cleaning
# Handle both old and new data formats
if 'orbit_y' in df.columns and 'orbit_x' in df.columns:
    df['orbit'] = df['orbit_y'].combine_first(df['orbit_x'])
    df['launch_site'] = df['launch_site_x'].combine_first(df['launch_site_y'])
    df = df[['rocket_name', 'payload_mass', 'orbit', 'landing_success']].dropna()
else:
    df = df[['rocket_name', 'payload_mass', 'orbit', 'landing_success']].dropna()

# Fix types
df['landing_success'] = df['landing_success'].map({
    True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0
})

print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Target distribution:\n{df['landing_success'].value_counts()}")
print(f"✓ Missing values: {df.isnull().sum().sum()}")

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
print("\n[2/8] Setting up preprocessing pipeline...")

X = df[['rocket_name', 'payload_mass', 'orbit']]
y = df['landing_success']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Preprocessor
numeric_features = ['payload_mass']
categorical_features = ['rocket_name', 'orbit']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ==========================================
# 3. TRAIN MULTIPLE MODELS
# ==========================================
print("\n[3/8] Training 5 ML models...")

models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    'XGBoost': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        ))
    ]),
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ]),
    'Neural Network (MLP)': Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(100, 50), 
            max_iter=500, 
            random_state=42
        ))
    ])
}

results = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'cv_roc_auc_mean': cv_mean,
        'cv_roc_auc_std': cv_std,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'model': model
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  CV Score:  {cv_mean:.4f} (+/- {cv_std:.4f})")

print("\n✓ All models trained successfully!")

# ==========================================
# 4. MODEL COMPARISON TABLE
# ==========================================
print("\n[4/8] Generating model comparison table...")

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()],
    'PR-AUC': [r['pr_auc'] for r in results.values()],
    'CV ROC-AUC': [f"{r['cv_roc_auc_mean']:.4f} (+/- {r['cv_roc_auc_std']:.4f})" 
                   for r in results.values()]
})

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('../reports/model_comparison.csv', index=False)
print(f"\n✓ Saved model comparison to ../reports/model_comparison.csv")

# ==========================================
# 5. VISUALIZATIONS
# ==========================================
print("\n[5/8] Generating visualizations...")

# 5.1 ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    auc = roc_auc_score(y_test, result['y_proba'])
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - SpaceX Landing Success Prediction', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved ROC curves plot")

# 5.2 Precision-Recall Curves
fig, ax = plt.subplots(figsize=(10, 8))

for name, result in results.items():
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, result['y_proba'])
    pr_auc = average_precision_score(y_test, result['y_proba'])
    ax.plot(recall_curve, precision_curve, 
            label=f'{name} (AP = {pr_auc:.3f})', linewidth=2)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - SpaceX Landing Success Prediction', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved Precision-Recall curves plot")

# 5.3 Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
model_names = list(results.keys())
metrics = ['accuracy', 'f1', 'roc_auc', 'pr_auc']
titles = ['Accuracy', 'F1-Score', 'ROC-AUC', 'PR-AUC']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    values = [results[name][metric] for name in model_names]
    bars = ax.barh(model_names, values, color=colors[:len(model_names)])
    ax.set_xlim(0, 1.0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('../reports/figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved model comparison bar charts")

# 5.4 Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar=False, annot_kws={'size': 14})
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10)
    axes[idx].set_ylabel('Actual', fontsize=10)

# Hide the last subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('../reports/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved confusion matrices plot")

# ==========================================
# 6. HYPERPARAMETER TUNING
# ==========================================
print("\n[6/8] Performing hyperparameter tuning (top 2 models)...")

# Find best model based on ROC-AUC
best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
print(f"\nBest model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.4f})")

# Get top 2 models
top_2_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:2]

tuned_results = {}

for name, _ in top_2_models:
    print(f"\n{'='*60}")
    print(f"Tuning: {name}")
    print(f"{'='*60}")
    
    if 'Random Forest' in name:
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [10, 20, 30, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    elif 'XGBoost' in name:
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    elif 'Logistic Regression' in name:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear']
        }
    elif 'SVM' in name:
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        }
    elif 'Neural Network' in name:
        param_grid = {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }
    else:
        continue
    
    grid_search = GridSearchCV(
        models[name],
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    y_pred_tuned = grid_search.predict(X_test)
    y_proba_tuned = grid_search.predict_proba(X_test)[:, 1]
    
    tuned_results[f'{name} (Tuned)'] = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned, zero_division=0),
        'recall': recall_score(y_test, y_pred_tuned, zero_division=0),
        'f1': f1_score(y_test, y_pred_tuned, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba_tuned),
        'pr_auc': average_precision_score(y_test, y_proba_tuned),
        'y_pred': y_pred_tuned,
        'y_proba': y_proba_tuned,
        'model': grid_search.best_estimator_,
        'best_params': grid_search.best_params_
    }
    
    print(f"  Test Accuracy:  {tuned_results[f'{name} (Tuned)']['accuracy']:.4f}")
    print(f"  Test ROC-AUC:   {tuned_results[f'{name} (Tuned)']['roc_auc']:.4f}")

# Update results with tuned models
results.update(tuned_results)

print("\n✓ Hyperparameter tuning complete!")

# ==========================================
# 7. FEATURE IMPORTANCE + SHAP ANALYSIS
# ==========================================
print("\n[7/8] Analyzing feature importance...")

# Always train a dedicated Random Forest for feature importance
print("\nTraining dedicated Random Forest for feature importance analysis...")
rf_explain = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42))
])
rf_explain.fit(X_train, y_train)

# Get feature names after preprocessing
ohe = rf_explain.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + cat_feature_names

# Feature importances from Random Forest
importances = rf_explain.named_steps['classifier'].feature_importances_
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n" + "="*60)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*60)
print(importance_df.to_string(index=False))

# Plot feature importance (top 15)
fig, ax = plt.subplots(figsize=(10, 6))
top_n = min(15, len(importance_df))
sns.barplot(data=importance_df.head(top_n), x='Importance', y='Feature',
            palette='viridis', ax=ax)
ax.set_title('Feature Importance — Random Forest (300 trees)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('reports/figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved feature importance plot")

# SHAP analysis
print("\nComputing SHAP values for model interpretation...")
try:
    import shap

    # Get transformed training data for SHAP
    X_train_transformed = rf_explain.named_steps['preprocessor'].transform(X_train)

    # Use TreeExplainer on the trained Random Forest
    explainer = shap.TreeExplainer(rf_explain.named_steps['classifier'])
    shap_values = explainer.shap_values(X_train_transformed)

    # SHAP expects list for binary classification (values for positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class

    # Summary plot
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train_transformed, feature_names=all_feature_names,
                     show=False, plot_size=(10, 6))
    plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved SHAP summary plot")

    # Beeswarm-style bar chart
    fig = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, feature_names=all_feature_names, show=False)
    plt.title('SHAP Mean Absolute Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/figures/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved SHAP bar chart")

except ImportError:
    print("⚠️  SHAP not installed. Install with: pip install shap")
except Exception as e:
    print(f"⚠️  SHAP analysis skipped: {e}")

# Save feature importance to CSV
importance_df.to_csv('reports/feature_importance.csv', index=False)
print("✓ Saved feature importance to reports/feature_importance.csv")

# ==========================================
# 8. SAVE BEST MODEL
# ==========================================
print("\n[8/8] Saving best model and results...")

# Find overall best model
best_overall_name = max(results, key=lambda k: results[k]['roc_auc'])
best_overall_result = results[best_overall_name]

# Save model
joblib.dump(best_overall_result['model'], '../src/models/best_model.pkl')
print(f"✓ Saved best model: {best_overall_name}")

# Save preprocessor separately
joblib.dump(preprocessor, '../src/models/preprocessor.pkl')
print("✓ Saved preprocessor")

# Save all results to JSON
results_summary = {}
for name, result in results.items():
    results_summary[name] = {
        'accuracy': round(result['accuracy'], 4),
        'precision': round(result['precision'], 4),
        'recall': round(result['recall'], 4),
        'f1': round(result['f1'], 4),
        'roc_auc': round(result['roc_auc'], 4),
        'pr_auc': round(result['pr_auc'], 4)
    }
    if 'best_params' in result:
        results_summary[name]['best_params'] = result['best_params']

with open('../reports/model_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Saved results to ../reports/model_results.json")

# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\n🏆 Best Model: {best_overall_name}")
print(f"   ROC-AUC: {best_overall_result['roc_auc']:.4f}")
print(f"   Accuracy: {best_overall_result['accuracy']:.4f}")
print(f"   F1-Score: {best_overall_result['f1']:.4f}")
print("\n📊 Generated Files:")
print("   - reports/model_comparison.csv")
print("   - reports/model_results.json")
print("   - reports/feature_importance.csv")
print("   - reports/figures/roc_curves.png")
print("   - reports/figures/pr_curves.png")
print("   - reports/figures/model_comparison.png")
print("   - reports/figures/confusion_matrices.png")
print("   - reports/figures/feature_importance.png")
print("   - src/models/best_model.pkl")
print("   - src/models/preprocessor.pkl")
print("\n✅ All tasks completed successfully!")
