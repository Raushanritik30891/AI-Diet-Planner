"""
AI Diet Planner — Flask Backend API
POST /predict  →  returns BMI, calories, weight category, diet recommendations
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, '../models')

# ─── Load Artifacts ───────────────────────────────────────────────────────────
model      = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
le_gender  = joblib.load(os.path.join(MODELS_DIR, 'le_gender.pkl'))
le_activity= joblib.load(os.path.join(MODELS_DIR, 'le_activity.pkl'))
le_goal    = joblib.load(os.path.join(MODELS_DIR, 'le_goal.pkl'))
meta       = joblib.load(os.path.join(MODELS_DIR, 'model_meta.pkl'))
print(f"✓ Model loaded: {meta['model_name']}")

# ─── Helper Functions ─────────────────────────────────────────────────────────
ACTIVITY_MULTIPLIERS = {
    'Sedentary': 1.2,
    'Lightly Active': 1.375,
    'Moderately Active': 1.55,
    'Very Active': 1.725,
    'Extra Active': 1.9
}

def compute_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return round(weight_kg / (h * h), 1)

def bmi_category(bmi):
    if bmi < 18.5: return 'Underweight'
    elif bmi < 25: return 'Normal Weight'
    elif bmi < 30: return 'Overweight'
    else: return 'Obese'

def compute_bmr(gender, weight_kg, height_cm, age):
    if gender == 'Male':
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def get_diet_tips(goal, bmi_cat, calories):
    tips = {
        'Weight Loss': {
            'headline': 'Calorie Deficit Plan',
            'emoji': '🔥',
            'color': '#FF6B6B',
            'tips': [
                'Aim for a 500 kcal daily deficit from TDEE',
                'Prioritize high-protein foods (chicken, eggs, legumes)',
                'Cut added sugars and ultra-processed foods',
                'Fill half your plate with non-starchy vegetables',
                'Drink 8–10 glasses of water daily',
                'Include 150+ min/week of moderate cardio'
            ],
            'foods': {
                'eat_more': ['Lean protein', 'Leafy greens', 'Berries', 'Greek yogurt', 'Oats'],
                'eat_less': ['Sugary drinks', 'White bread', 'Fried foods', 'Alcohol', 'Sweets']
            }
        },
        'Weight Gain': {
            'headline': 'Calorie Surplus Plan',
            'emoji': '💪',
            'color': '#43B89C',
            'tips': [
                'Aim for a 300–500 kcal daily surplus',
                'Eat every 3–4 hours — never skip meals',
                'Prioritize complex carbohydrates for energy',
                'Get 1.6–2.2g of protein per kg body weight',
                'Include resistance training 3–4x per week',
                'Add healthy fats: nuts, avocado, olive oil'
            ],
            'foods': {
                'eat_more': ['Brown rice', 'Chicken/fish', 'Nuts & seeds', 'Whole milk', 'Bananas'],
                'eat_less': ['Diet foods', 'Very low-fat items', 'Excessive cardio', 'Skipped meals']
            }
        },
        'Maintain Weight': {
            'headline': 'Balanced Maintenance Plan',
            'emoji': '⚖️',
            'color': '#6C63FF',
            'tips': [
                'Eat at TDEE to maintain current weight',
                'Follow the plate method: 50% veggies, 25% protein, 25% carbs',
                'Vary your protein sources (plant & animal)',
                'Limit saturated fat and sodium intake',
                'Include 2 servings of fatty fish per week',
                'Practice mindful eating and portion control'
            ],
            'foods': {
                'eat_more': ['Whole grains', 'Mixed vegetables', 'Legumes', 'Fatty fish', 'Fruits'],
                'eat_less': ['Processed snacks', 'Excess salt', 'Trans fats', 'Sugary drinks']
            }
        }
    }

    plan = tips.get(goal, tips['Maintain Weight'])

    # Extra advice if BMI category warrants it
    extra = []
    if bmi_cat == 'Underweight':
        extra.append('⚠️ Your BMI is low — consider consulting a dietitian for a structured gain plan.')
    elif bmi_cat == 'Obese':
        extra.append('⚠️ A BMI above 30 carries health risks — consider speaking with a healthcare provider.')

    if extra:
        plan = dict(plan)
        plan['tips'] = plan['tips'] + extra

    return plan

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'running',
        'project': 'AI Diet Planner',
        'endpoints': {'POST /predict': 'Get personalized calorie & diet recommendations'}
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # ── Validate required fields ──────────────────────────────────────────
        required = ['age', 'gender', 'height_cm', 'weight_kg', 'activity_level', 'goal']
        missing  = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        age      = float(data['age'])
        gender   = data['gender']
        height   = float(data['height_cm'])
        weight   = float(data['weight_kg'])
        activity = data['activity_level']
        goal     = data['goal']

        # ── Derived features ──────────────────────────────────────────────────
        bmi   = compute_bmi(weight, height)
        bmr   = compute_bmr(gender, weight, height, age)
        act_score = ACTIVITY_MULTIPLIERS.get(activity, 1.375)

        gender_enc   = int(le_gender.transform([gender])[0])
        activity_enc = int(le_activity.transform([activity])[0])
        goal_enc     = int(le_goal.transform([goal])[0])

        # ── Model prediction ──────────────────────────────────────────────────
        FEATURES = meta['features']
        X = pd.DataFrame([[age, height, weight, bmi, bmr,
                           act_score, gender_enc, activity_enc, goal_enc]],
                         columns=FEATURES)
        predicted_calories = int(round(model.predict(X)[0]))

        bmi_cat  = bmi_category(bmi)
        diet     = get_diet_tips(goal, bmi_cat, predicted_calories)

        # ── Response ──────────────────────────────────────────────────────────
        return jsonify({
            'bmi': bmi,
            'bmi_category': bmi_cat,
            'bmr': round(bmr, 0),
            'tdee': round(bmr * act_score, 0),
            'predicted_calories': predicted_calories,
            'model_used': meta['model_name'],
            'diet_plan': diet,
            'macros': {
                'protein_g': round(predicted_calories * 0.30 / 4),
                'carbs_g':   round(predicted_calories * 0.45 / 4),
                'fat_g':     round(predicted_calories * 0.25 / 9)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': meta['model_name']})


if __name__ == '__main__':
    print("\n🥗 AI Diet Planner API running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
