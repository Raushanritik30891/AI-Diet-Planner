/* AI Diet Planner — frontend/script.js */

const API = 'http://localhost:5000/predict';

// ─── State ────────────────────────────────────────────────────────────────────
let selectedGender = 'Male';
let selectedGoal   = 'Weight Loss';

// ─── Toggle buttons ───────────────────────────────────────────────────────────
document.querySelectorAll('#gender-toggle .toggle').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#gender-toggle .toggle').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedGender = btn.dataset.val;
  });
});

document.querySelectorAll('#goal-group .goal-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#goal-group .goal-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    selectedGoal = btn.dataset.val;
  });
});

// ─── Main predict function ────────────────────────────────────────────────────
async function predict() {
  const age    = parseFloat(document.getElementById('age').value);
  const height = parseFloat(document.getElementById('height').value);
  const weight = parseFloat(document.getElementById('weight').value);
  const activity = document.getElementById('activity').value;

  // Validation
  if (!age || !height || !weight || !activity) {
    shakeCard(); return;
  }
  if (age < 15 || age > 90)         { alert('Age must be between 15 and 90.'); return; }
  if (height < 140 || height > 220) { alert('Height must be between 140–220 cm.'); return; }
  if (weight < 30  || weight > 200) { alert('Weight must be between 30–200 kg.'); return; }

  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div><span>Analysing…</span>';

  try {
    const res = await fetch(API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        age, height_cm: height, weight_kg: weight,
        gender: selectedGender, activity_level: activity, goal: selectedGoal
      })
    });

    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    renderResult(data);

  } catch (err) {
    // Fallback: compute locally if backend is not running
    const localResult = computeLocally(age, height, weight, selectedGender, activity, selectedGoal);
    renderResult(localResult);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="cta-label">Analyse My Diet</span><span class="cta-arrow">→</span>';
  }
}

// ─── Local fallback (runs without backend) ────────────────────────────────────
function computeLocally(age, height, weight, gender, activity, goal) {
  const activityMap = {
    'Sedentary': 1.2, 'Lightly Active': 1.375,
    'Moderately Active': 1.55, 'Very Active': 1.725, 'Extra Active': 1.9
  };
  const bmi = +(weight / ((height / 100) ** 2)).toFixed(1);
  const bmr = gender === 'Male'
    ? 10 * weight + 6.25 * height - 5 * age + 5
    : 10 * weight + 6.25 * height - 5 * age - 161;
  const tdee = bmr * (activityMap[activity] || 1.375);
  const adj  = goal === 'Weight Loss' ? -500 : goal === 'Weight Gain' ? 500 : 0;
  const calories = Math.round(tdee + adj);

  const bmi_category = bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal Weight' : bmi < 30 ? 'Overweight' : 'Obese';
  const dietPlans = {
    'Weight Loss': {
      headline: 'Calorie Deficit Plan', emoji: '🔥', color: '#FF6B6B',
      tips: ['Aim for a 500 kcal daily deficit from TDEE','Prioritize high-protein foods (chicken, eggs, legumes)','Cut added sugars and ultra-processed foods','Fill half your plate with non-starchy vegetables','Drink 8–10 glasses of water daily'],
      foods: { eat_more: ['Lean protein','Leafy greens','Berries','Greek yogurt','Oats'], eat_less: ['Sugary drinks','White bread','Fried foods','Alcohol','Sweets'] }
    },
    'Weight Gain': {
      headline: 'Calorie Surplus Plan', emoji: '💪', color: '#43B89C',
      tips: ['Aim for a 300–500 kcal daily surplus','Eat every 3–4 hours — never skip meals','Prioritize complex carbohydrates for energy','Get 1.6–2.2g of protein per kg body weight','Include resistance training 3–4x per week'],
      foods: { eat_more: ['Brown rice','Chicken/fish','Nuts & seeds','Whole milk','Bananas'], eat_less: ['Diet foods','Very low-fat items','Excessive cardio','Skipped meals'] }
    },
    'Maintain Weight': {
      headline: 'Balanced Maintenance Plan', emoji: '⚖️', color: '#6C63FF',
      tips: ['Eat at TDEE to maintain current weight','Follow the plate method: 50% veggies, 25% protein, 25% carbs','Vary your protein sources (plant & animal)','Limit saturated fat and sodium intake','Include 2 servings of fatty fish per week'],
      foods: { eat_more: ['Whole grains','Mixed vegetables','Legumes','Fatty fish','Fruits'], eat_less: ['Processed snacks','Excess salt','Trans fats','Sugary drinks'] }
    }
  };

  return {
    bmi, bmi_category, bmr: Math.round(bmr), tdee: Math.round(tdee),
    predicted_calories: calories, model_used: 'Local Formula (backend offline)',
    diet_plan: dietPlans[goal] || dietPlans['Maintain Weight'],
    macros: {
      protein_g: Math.round(calories * 0.30 / 4),
      carbs_g:   Math.round(calories * 0.45 / 4),
      fat_g:     Math.round(calories * 0.25 / 9)
    }
  };
}

// ─── Render results ───────────────────────────────────────────────────────────
function renderResult(data) {
  // Hide placeholder, show output
  document.getElementById('placeholder').classList.add('hidden');
  const out = document.getElementById('output');
  out.classList.remove('hidden');

  // Calories hero
  animateNumber('cal-num', data.predicted_calories);
  document.getElementById('cal-model').textContent = `Model: ${data.model_used}`;

  // Metrics chips
  const bmiEl = document.getElementById('bmi-val');
  bmiEl.textContent = data.bmi;
  const catEl = document.getElementById('bmi-cat');
  catEl.textContent = data.bmi_category;
  catEl.className = 'mc-tag ' + bmiColorClass(data.bmi_category);

  document.getElementById('bmr-val').textContent  = data.bmr;
  document.getElementById('tdee-val').textContent = data.tdee;

  // Macros bar
  const { protein_g, carbs_g, fat_g } = data.macros;
  const totalCal = protein_g * 4 + carbs_g * 4 + fat_g * 9;
  const pPct = ((protein_g * 4 / totalCal) * 100).toFixed(0);
  const cPct = ((carbs_g * 4 / totalCal) * 100).toFixed(0);
  const fPct = (100 - pPct - cPct);

  document.getElementById('macro-bar').innerHTML = `
    <div style="width:${pPct}%;background:#FF6B6B"></div>
    <div style="width:${cPct}%;background:#43B89C"></div>
    <div style="width:${fPct}%;background:#6C63FF"></div>
  `;
  document.getElementById('macro-legend').innerHTML = `
    <div class="legend-item"><div class="legend-dot" style="background:#FF6B6B"></div>Protein ${protein_g}g (${pPct}%)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#43B89C"></div>Carbs ${carbs_g}g (${cPct}%)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#6C63FF"></div>Fat ${fat_g}g (${fPct}%)</div>
  `;

  // Diet plan
  const dp = data.diet_plan;
  document.getElementById('diet-emoji').textContent    = dp.emoji;
  document.getElementById('diet-headline').textContent = dp.headline;
  document.getElementById('diet-goal').textContent     = selectedGoal;
  document.getElementById('diet-card').style.borderLeftColor = dp.color;
  document.getElementById('diet-card').style.borderLeftWidth = '4px';

  const tipList = document.getElementById('tip-list');
  tipList.innerHTML = dp.tips.map(t => `<li>${t}</li>`).join('');

  // Foods
  document.getElementById('eat-more').innerHTML = dp.foods.eat_more.map(f => `<li>🟢 ${f}</li>`).join('');
  document.getElementById('eat-less').innerHTML = dp.foods.eat_less.map(f => `<li>🔴 ${f}</li>`).join('');

  // Scroll to results
  out.classList.add('fade-in');
  document.getElementById('results-col').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
function animateNumber(id, target) {
  const el = document.getElementById(id);
  const start = 0;
  const duration = 800;
  const startTime = performance.now();
  const update = (now) => {
    const progress = Math.min((now - startTime) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(start + (target - start) * ease);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function bmiColorClass(cat) {
  const map = {
    'Underweight': 'bmi-underweight',
    'Normal Weight': 'bmi-normal',
    'Overweight': 'bmi-overweight',
    'Obese': 'bmi-obese'
  };
  return map[cat] || '';
}

function shakeCard() {
  const card = document.querySelector('.card--input');
  card.style.animation = 'none';
  card.offsetHeight; // reflow
  card.style.animation = 'shake .35s ease';
  setTimeout(() => card.style.animation = '', 400);
}

// Shake keyframe via JS (injected once)
const shakeStyle = document.createElement('style');
shakeStyle.textContent = `@keyframes shake {
  0%,100%{transform:translateX(0)}
  20%{transform:translateX(-8px)}
  40%{transform:translateX(8px)}
  60%{transform:translateX(-6px)}
  80%{transform:translateX(6px)}
}`;
document.head.appendChild(shakeStyle);

// Allow Enter key
document.addEventListener('keydown', e => {
  if (e.key === 'Enter') predict();
});
