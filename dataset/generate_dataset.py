"""
AI Diet Planner - Dataset Generator
Generates a realistic synthetic dataset using validated medical formulas:
- BMR: Mifflin-St Jeor Equation
- TDEE: Harris-Benedict Activity Multipliers
- BMI: Standard WHO formula
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

# Demographics
ages = np.random.randint(18, 70, N)
genders = np.random.choice(['Male', 'Female'], N)

# Heights & Weights (realistic distributions by gender)
heights = np.where(
    genders == 'Male',
    np.random.normal(175, 8, N),
    np.random.normal(162, 7, N)
)
weights = np.where(
    genders == 'Male',
    np.random.normal(78, 14, N),
    np.random.normal(63, 12, N)
)
heights = np.clip(heights, 145, 210)
weights = np.clip(weights, 40, 160)

# Activity levels
activity_levels = np.random.choice(
    ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active', 'Extra Active'],
    N, p=[0.25, 0.30, 0.25, 0.15, 0.05]
)

# Goals
goals = np.random.choice(['Weight Loss', 'Maintain Weight', 'Weight Gain'], N, p=[0.40, 0.35, 0.25])

# ---- Feature Engineering ----

# BMI
bmi = weights / ((heights / 100) ** 2)

# BMR (Mifflin-St Jeor)
bmr = np.where(
    genders == 'Male',
    10 * weights + 6.25 * heights - 5 * ages + 5,
    10 * weights + 6.25 * heights - 5 * ages - 161
)

# Activity multipliers
activity_map = {
    'Sedentary': 1.2,
    'Lightly Active': 1.375,
    'Moderately Active': 1.55,
    'Very Active': 1.725,
    'Extra Active': 1.9
}
activity_scores = np.array([activity_map[a] for a in activity_levels])

# TDEE (Total Daily Energy Expenditure)
tdee = bmr * activity_scores

# Goal adjustment
goal_adjustment = np.where(
    goals == 'Weight Loss', -500,
    np.where(goals == 'Weight Gain', +500, 0)
)

# Target calories (with small noise for realism)
calories = tdee + goal_adjustment + np.random.normal(0, 50, N)
calories = np.clip(calories, 1200, 5000).round(0)

# Build DataFrame
df = pd.DataFrame({
    'age': ages,
    'gender': genders,
    'height_cm': heights.round(1),
    'weight_kg': weights.round(1),
    'activity_level': activity_levels,
    'goal': goals,
    'bmi': bmi.round(2),
    'bmr': bmr.round(2),
    'activity_score': activity_scores,
    'tdee': tdee.round(2),
    'calories': calories.astype(int)
})

df.to_csv('diet_dataset.csv', index=False)
print(f"Dataset generated: {len(df)} rows")
print(df.describe())
print("\nClass distribution:")
print(df['activity_level'].value_counts())
