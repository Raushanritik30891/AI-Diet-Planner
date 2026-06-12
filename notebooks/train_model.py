"""
AI Diet Planner - ML Training Pipeline
=======================================
Trains Linear Regression vs Random Forest Regressor,
compares metrics, saves best model, generates all visualizations.
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE, '../dataset/diet_dataset.csv')
MODELS_DIR   = os.path.join(BASE, '../models')
VIZ_DIR      = os.path.join(BASE, '../notebooks/visualizations')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

print("=" * 60)
print("  AI DIET PLANNER — ML TRAINING PIPELINE")
print("=" * 60)

# ─── 1. Load & Inspect ────────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df.columns)}")
print(f"    Nulls:\n{df.isnull().sum()}")

# ─── 2. Data Cleaning ─────────────────────────────────────────────────────────
print("\n[2] Cleaning data...")
df.dropna(inplace=True)
df = df[(df['calories'] >= 1200) & (df['calories'] <= 5000)]
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
print(f"    Clean shape: {df.shape}")

# ─── 3. Feature Engineering ───────────────────────────────────────────────────
print("\n[3] Feature engineering...")

# BMI Category
def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

df['bmi_category'] = df['bmi'].apply(bmi_category)

# Encode categoricals
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_goal = LabelEncoder()
le_bmi_cat = LabelEncoder()

df['gender_enc']    = le_gender.fit_transform(df['gender'])
df['activity_enc']  = le_activity.fit_transform(df['activity_level'])
df['goal_enc']      = le_goal.fit_transform(df['goal'])
df['bmi_cat_enc']   = le_bmi_cat.fit_transform(df['bmi_category'])

# Save encoders
joblib.dump(le_gender,   os.path.join(MODELS_DIR, 'le_gender.pkl'))
joblib.dump(le_activity, os.path.join(MODELS_DIR, 'le_activity.pkl'))
joblib.dump(le_goal,     os.path.join(MODELS_DIR, 'le_goal.pkl'))

FEATURES = ['age', 'height_cm', 'weight_kg', 'bmi', 'bmr',
            'activity_score', 'gender_enc', 'activity_enc', 'goal_enc']
TARGET   = 'calories'

X = df[FEATURES]
y = df[TARGET]
print(f"    Features: {FEATURES}")

# ─── 4. Train-Test Split ──────────────────────────────────────────────────────
print("\n[4] Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

# ─── 5. Model Training & Comparison ──────────────────────────────────────────
print("\n[5] Training models...")

results = {}

# --- Linear Regression ---
print("    → Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

mae_lr  = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr   = r2_score(y_test, y_pred_lr)
cv_lr   = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()

results['Linear Regression'] = {
    'model': lr, 'pred': y_pred_lr,
    'MAE': mae_lr, 'RMSE': rmse_lr, 'R2': r2_lr, 'CV_R2': cv_lr
}
print(f"      MAE={mae_lr:.1f}  RMSE={rmse_lr:.1f}  R²={r2_lr:.4f}  CV-R²={cv_lr:.4f}")

# --- Random Forest ---
print("    → Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                            min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf  = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf   = r2_score(y_test, y_pred_rf)
cv_rf   = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()

results['Random Forest'] = {
    'model': rf, 'pred': y_pred_rf,
    'MAE': mae_rf, 'RMSE': rmse_rf, 'R2': r2_rf, 'CV_R2': cv_rf
}
print(f"      MAE={mae_rf:.1f}  RMSE={rmse_rf:.1f}  R²={r2_rf:.4f}  CV-R²={cv_rf:.4f}")

# ─── 6. Model Selection ───────────────────────────────────────────────────────
print("\n[6] Selecting best model...")
best_name = max(results, key=lambda k: results[k]['R2'])
best = results[best_name]
print(f"    ✓ Best model: {best_name}")
print(f"      MAE={best['MAE']:.1f} | RMSE={best['RMSE']:.1f} | R²={best['R2']:.4f}")

# Save best model + feature list
joblib.dump(best['model'], os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump({'features': FEATURES, 'model_name': best_name},
            os.path.join(MODELS_DIR, 'model_meta.pkl'))
print(f"    Model saved → models/best_model.pkl")

# ─── 7. Visualizations ────────────────────────────────────────────────────────
print("\n[7] Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#6C63FF', '#FF6584', '#43B89C', '#F5A623', '#4A90D9']

# --- Fig 1: BMI + Calorie Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('AI Diet Planner — Data Distributions', fontsize=16, fontweight='bold', y=1.02)

axes[0].hist(df['bmi'], bins=40, color=PALETTE[0], alpha=0.85, edgecolor='white')
for v, label in [(18.5,'Underweight'), (25,'Normal'), (30,'Overweight')]:
    axes[0].axvline(v, color='#333', linestyle='--', linewidth=1.2)
    axes[0].text(v+0.3, axes[0].get_ylim()[1]*0.85, label, fontsize=8, color='#555')
axes[0].set_title('BMI Distribution', fontweight='bold')
axes[0].set_xlabel('BMI'); axes[0].set_ylabel('Count')

axes[1].hist(df['calories'], bins=40, color=PALETTE[1], alpha=0.85, edgecolor='white')
axes[1].set_title('Daily Calorie Distribution', fontweight='bold')
axes[1].set_xlabel('Calories (kcal)'); axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '01_data_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ 01_data_distributions.png")

# --- Fig 2: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
numeric_cols = ['age', 'height_cm', 'weight_kg', 'bmi', 'bmr', 'activity_score', 'calories']
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '02_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ 02_correlation_heatmap.png")

# --- Fig 3: Feature Importance (RF) ---
fig, ax = plt.subplots(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors = [PALETTE[i % len(PALETTE)] for i in range(len(importances))]
importances.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
for i, (val, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '03_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ 03_feature_importance.png")

# --- Fig 4: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Model Comparison: Linear Regression vs Random Forest',
             fontsize=14, fontweight='bold')

metrics = ['MAE', 'RMSE', 'R2']
labels  = ['MAE (kcal)', 'RMSE (kcal)', 'R² Score']
model_names = list(results.keys())

for idx, (metric, label) in enumerate(zip(metrics, labels)):
    vals = [results[m][metric] for m in model_names]
    bars = axes[idx].bar(model_names, vals, color=[PALETTE[0], PALETTE[1]],
                          edgecolor='white', width=0.5)
    axes[idx].set_title(label, fontweight='bold')
    for bar, val in zip(bars, vals):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height()*0.95,
                       f'{val:.2f}', ha='center', va='top', color='white',
                       fontweight='bold', fontsize=11)
    axes[idx].set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '04_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ 04_model_comparison.png")

# --- Fig 5: Actual vs Predicted (Best Model) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Actual vs Predicted — {best_name}', fontsize=14, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    ax.scatter(y_test, res['pred'], alpha=0.3, s=10, color=PALETTE[0])
    mn, mx = y_test.min(), y_test.max()
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('Actual Calories'); ax.set_ylabel('Predicted Calories')
    ax.set_title(f'{name}  (R²={res["R2"]:.4f})', fontweight='bold')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '05_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ 05_actual_vs_predicted.png")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"  Best Model : {best_name}")
print(f"  MAE        : {best['MAE']:.1f} kcal")
print(f"  RMSE       : {best['RMSE']:.1f} kcal")
print(f"  R² Score   : {best['R2']:.4f}")
print(f"  CV R²      : {best['CV_R2']:.4f}")
print(f"\n  Artifacts saved in: models/  &  notebooks/visualizations/")
print("=" * 60)
