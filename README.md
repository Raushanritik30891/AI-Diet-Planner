# 🥗 AI Diet Planner

> A portfolio-ready Machine Learning project that predicts personalized daily calorie requirements and generates diet recommendations using Random Forest Regression.

---

## 📌 Problem Statement

Millions of people follow generic calorie guides that ignore individual factors like age, gender, body composition, and activity level. This project builds a machine learning model trained on 5,000 records to predict **precise daily calorie targets** for each individual, then wraps it in a clean web UI with rule-based diet suggestions.

---

## 📂 Project Structure

```
ai-diet-planner/
│
├── dataset/
│   ├── generate_dataset.py     ← Synthetic dataset generation script
│   └── diet_dataset.csv        ← 5,000-record dataset (auto-generated)
│
├── notebooks/
│   ├── train_model.py          ← Full ML pipeline script
│   └── visualizations/         ← All 5 output charts (PNG)
│
├── models/
│   ├── best_model.pkl          ← Saved Random Forest model
│   ├── le_gender.pkl           ← LabelEncoder for gender
│   ├── le_activity.pkl         ← LabelEncoder for activity level
│   ├── le_goal.pkl             ← LabelEncoder for goal
│   └── model_meta.pkl          ← Feature names & model info
│
├── backend/
│   └── app.py                  ← Flask REST API
│
├── frontend/
│   ├── index.html              ← Main page
│   ├── style.css               ← Dark-themed responsive design
│   └── script.js               ← Prediction logic + local fallback
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

**Source:** Synthetically generated using validated medical formulas (no real personal data)

| Column          | Description                                           |
|-----------------|-------------------------------------------------------|
| `age`           | Age in years (18–69)                                  |
| `gender`        | Male / Female                                         |
| `height_cm`     | Height in centimetres                                 |
| `weight_kg`     | Weight in kilograms                                   |
| `activity_level`| Sedentary / Lightly Active / Moderately Active / ...  |
| `goal`          | Weight Loss / Maintain Weight / Weight Gain           |
| `bmi`           | Body Mass Index (engineered feature)                  |
| `bmr`           | Basal Metabolic Rate — Mifflin-St Jeor equation       |
| `activity_score`| Numerical activity multiplier (1.2 – 1.9)            |
| `tdee`          | Total Daily Energy Expenditure (bmr × activity_score) |
| `calories`      | **Target** — TDEE ± goal adjustment + noise           |

**Size:** 5,000 rows × 11 columns  
**Formula used for BMR:**
- Male:   `10×weight + 6.25×height − 5×age + 5`
- Female: `10×weight + 6.25×height − 5×age − 161`

---

## 🤖 ML Workflow

```
Raw Data
    │
    ▼
Data Cleaning (nulls, outlier clipping)
    │
    ▼
Feature Engineering
  • BMI  = weight / (height/100)²
  • BMR  = Mifflin-St Jeor
  • TDEE = BMR × activity multiplier
  • Label Encoding (gender, activity, goal)
    │
    ▼
Train-Test Split  (80% / 20%  →  4,000 / 1,000 samples)
    │
    ├── Linear Regression
    │     MAE=287.9  RMSE=335.4  R²=0.692
    │
    └── Random Forest Regressor  ← WINNER
          MAE=43.6   RMSE=56.4   R²=0.9913
    │
    ▼
Model Selection → Random Forest saved with joblib
    │
    ▼
Flask API  →  POST /predict
    │
    ▼
Frontend (HTML/CSS/JS)
```

---

## 🏆 Results

| Model             | MAE (kcal) | RMSE (kcal) | R² Score | CV R² |
|-------------------|-----------|------------|---------|-------|
| Linear Regression | 287.9     | 335.4      | 0.6922  | 0.6909|
| **Random Forest** | **43.6**  | **56.4**   | **0.9913** | **0.9916** |

**Winner:** Random Forest Regressor — 99.1% variance explained, ±44 kcal average error.

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
cd dataset
python generate_dataset.py
```

### 3. Train the model
```bash
cd notebooks
python train_model.py
```

### 4. Start the Flask backend
```bash
cd backend
python app.py
# → Listening on http://localhost:5000
```

### 5. Open the frontend
```bash
# Simply open in browser:
open frontend/index.html
# or serve with Python:
cd frontend && python -m http.server 8080
```

> ⚠️ If Flask is not running, the frontend falls back to a **local formula** so the UI still works.

---

## 🔌 API Reference

### `POST /predict`

**Request body:**
```json
{
  "age": 25,
  "gender": "Male",
  "height_cm": 175,
  "weight_kg": 70,
  "activity_level": "Moderately Active",
  "goal": "Weight Loss"
}
```

**Response:**
```json
{
  "bmi": 22.9,
  "bmi_category": "Normal Weight",
  "bmr": 1724.0,
  "tdee": 2672.0,
  "predicted_calories": 2174,
  "model_used": "Random Forest",
  "macros": { "protein_g": 163, "carbs_g": 245, "fat_g": 60 },
  "diet_plan": {
    "headline": "Calorie Deficit Plan",
    "emoji": "🔥",
    "tips": ["...", "..."],
    "foods": { "eat_more": [...], "eat_less": [...] }
  }
}
```

---

## 📈 Visualizations

All charts saved in `notebooks/visualizations/`:

| File | Description |
|------|-------------|
| `01_data_distributions.png` | BMI + Calorie histograms |
| `02_correlation_heatmap.png` | Feature correlation matrix |
| `03_feature_importance.png` | Random Forest feature importances |
| `04_model_comparison.png` | MAE / RMSE / R² bar charts |
| `05_actual_vs_predicted.png` | Scatter plot (actual vs predicted) |

---

## 💡 Future Improvements

- [ ] Add micronutrient tracking (vitamins, minerals)
- [ ] Integrate a real Kaggle dataset (e.g. Fitness Trackers dataset)
- [ ] Deploy to Render / Railway with Docker
- [ ] Add meal plan generator using Claude API
- [ ] Include XGBoost / LightGBM for comparison
- [ ] User authentication + history tracking

---

## 🎤 Interview Q&A (20 Questions)

### Q1. What problem does this project solve?
**A:** Generic calorie calculators use fixed multipliers ignoring individual variation. This project trains an ML model on 5,000 diverse records to predict personalised daily calorie requirements accounting for age, gender, height, weight, activity, and goal.

### Q2. Why did you choose Random Forest over Linear Regression?
**A:** Linear Regression gave R²=0.69, meaning it struggled with non-linear interactions (e.g., how activity level and weight interact). Random Forest captures these by averaging predictions from 100 decision trees, yielding R²=0.99 and MAE of just 44 kcal.

### Q3. Explain the Mifflin-St Jeor equation.
**A:** It calculates Basal Metabolic Rate (calories burned at complete rest):
- Male: `10×weight + 6.25×height − 5×age + 5`
- Female: `10×weight + 6.25×height − 5×age − 161`
It's the most accurate BMR formula validated for diverse populations.

### Q4. What is feature engineering and what did you do here?
**A:** Feature engineering transforms raw inputs into more predictive representations. I computed BMI (weight/height²), BMR (Mifflin-St Jeor), activity_score (numeric multiplier), and TDEE (BMR × activity_score). These derived features carry much more predictive signal than raw inputs alone.

### Q5. What is R² score and what does 0.99 mean?
**A:** R² (coefficient of determination) measures the proportion of variance in the target explained by the model. R²=0.9913 means our Random Forest explains 99.13% of calorie variance — very high, indicating excellent fit.

### Q6. What is MAE vs RMSE?
**A:** MAE (Mean Absolute Error) is the average absolute difference between predicted and actual values — easy to interpret (44 kcal average error). RMSE (Root Mean Squared Error) penalises large errors more heavily by squaring them first. RMSE=56 here means occasional predictions are somewhat worse, but still small relative to calorie ranges of 1200–5000.

### Q7. How did you prevent overfitting in Random Forest?
**A:** Three approaches: (1) `max_depth=15` limits tree depth, (2) `min_samples_split=5` prevents splitting on tiny subsets, (3) 5-fold cross-validation confirms CV R²=0.9916 ≈ test R²=0.9913, confirming the model generalises well.

### Q8. What is cross-validation and why use it?
**A:** Cross-validation splits data into k folds, trains on k−1 and tests on 1 repeatedly. It gives a more reliable performance estimate than a single split. Our 5-fold CV R² closely matches the test R², confirming no overfitting.

### Q9. What is Label Encoding and when would you use One-Hot Encoding instead?
**A:** Label Encoding maps categories to integers (Male=0, Female=1). It's appropriate for tree-based models because they split on thresholds, so ordinal relationships don't matter. One-Hot Encoding (creating dummy columns) is better for linear models to avoid implying false ordinal relationships.

### Q10. Why does TDEE matter more than BMR for calorie planning?
**A:** BMR is calories burned at complete rest. TDEE multiplies BMR by an activity factor to reflect real daily energy expenditure including exercise and movement. Using only BMR would underestimate calories for active people by 20–90%.

### Q11. How does your diet recommendation system work?
**A:** It's rule-based logic on top of the ML prediction. The model predicts calories; a Python dictionary maps the user's goal (Weight Loss/Gain/Maintain) to curated tips and food lists. The BMI category adds conditional warnings (e.g., Obese triggers a health advisory).

### Q12. What is joblib and why use it over pickle?
**A:** Joblib is optimised for saving NumPy arrays and scikit-learn models — it compresses arrays efficiently and is faster than standard pickle for large models. It's the recommended way to persist sklearn estimators.

### Q13. Explain the Flask API design.
**A:** The API has one main endpoint `POST /predict` that accepts JSON user inputs, computes derived features (BMI, BMR, activity_score), encodes categoricals using saved LabelEncoders, runs the model, and returns a structured JSON response with calories, macros, and diet plan. CORS is enabled so the frontend can call it from a browser.

### Q14. What's the BMI classification you used?
**A:** WHO standard: Underweight (<18.5), Normal (18.5–24.9), Overweight (25–29.9), Obese (≥30). BMI has known limitations (doesn't account for muscle mass) which I mention in the UI disclaimer.

### Q15. How would you deploy this project?
**A:** (1) Containerise with Docker, (2) Push to GitHub, (3) Deploy Flask backend to Render/Railway (free tier), (4) Serve frontend via GitHub Pages or Netlify. Alternatively, package as a single Flask app serving static files.

### Q16. What are the limitations of using a synthetic dataset?
**A:** The model learns formulas it was generated from, so R² appears artificially high. On real-world data with measurement noise, genetic variability, and metabolic differences, performance would be lower (~0.75–0.85 R² is realistic). For production use, a real dataset (e.g., NHANES) should be used.

### Q17. What is the 80/20 train-test split and why that ratio?
**A:** 80% (4,000 samples) for training gives the model sufficient data to learn patterns. 20% (1,000 samples) is held out, never seen during training, to measure true generalisation performance. 80/20 is the standard default balancing training data size vs evaluation reliability.

### Q18. Which feature was most important in the Random Forest and why?
**A:** `activity_score` (TDEE multiplier) and `bmr` dominate, because calorie requirements are directly computed from them. This makes intuitive sense — a very active person burns 60–90% more calories than a sedentary person of the same size.

### Q19. How would you improve model performance on real data?
**A:** (1) Add more features: body fat %, sleep, stress level, metabolic history, (2) Use gradient boosting (XGBoost/LightGBM), (3) Hyperparameter tuning with GridSearchCV, (4) Collect a larger real-world dataset, (5) Consider a separate model per goal type.

### Q20. How do you explain this project in 60 seconds?
**A:** "I built AI Diet Planner — a machine learning web app that predicts your personalised daily calorie needs. I generated a 5,000-record dataset using validated nutrition formulas, engineered features like BMI and BMR, then compared Linear Regression versus Random Forest. Random Forest won with 99.1% accuracy and an average error of just 44 kcal. The system wraps the model in a Flask API with a clean dark-themed UI that shows predicted calories, macros breakdown, and a rule-based diet plan based on your goal. It's deployable locally and designed to be portfolio-ready."

---

## 👤 Author

**Ritik Raushan**  
B.Tech AI & ML | Campus Placement Prep 2025  
GitHub · LinkedIn

---

*Built with Python, Flask, Scikit-learn, Vanilla JS*
"# AI-Diet-Planner" 
