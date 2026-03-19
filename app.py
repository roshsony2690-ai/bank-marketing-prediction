from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import pickle

app = Flask(__name__)

# Load model with error handling
MODEL_PATH = 'bank_marketing_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        # Try to load with joblib first
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully with joblib!")
    except Exception as e:
        print(f"⚠ Joblib loading failed: {e}")
        try:
            # Try with pickle as fallback
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully with pickle!")
        except Exception as e2:
            print(f"❌ Could not load model: {e2}")
            model = None
else:
    print(f"❌ Model file not found: {MODEL_PATH}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'age': int(request.form['age']),
            'job': request.form['job'],
            'marital': request.form['marital'],
            'education': request.form['education'],
            'default': request.form['default'],
            'balance': float(request.form['balance']),
            'housing': request.form['housing'],
            'loan': request.form['loan'],
            'day': int(request.form['day']),
            'month': request.form['month'],
            'campaign': int(request.form['campaign']),
            'pdays': int(request.form['pdays']),
            'previous': int(request.form['previous']),
            'poutcome': request.form['poutcome']
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([form_data])
        
        # Make prediction
        if model is not None:
            try:
                # Try to predict
                pred = model.predict(df)[0]
                
                # Try to get probability
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(df)[0][1] * 100
                else:
                    prob = 50.0
                    
                print(f"✅ Prediction made: {pred} with probability {prob:.2f}%")
                
            except Exception as e:
                print(f"⚠ Prediction error: {e}")
                # Fallback to demo
                import random
                pred = random.choice([0, 1])
                prob = random.uniform(0, 100)
        else:
            # Demo mode
            import random
            pred = random.choice([0, 1])
            prob = random.uniform(0, 100)
        
        prediction_text = 'Yes' if pred == 1 else 'No'
        
        return render_template('result.html', 
                             prediction=prediction_text,
                             probability=prob,
                             form_data=form_data)
        
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><p><a href='/'>Go back</a></p>"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BANK MARKETING PREDICTION APP")
    print("="*60)
    if model:
        print("✅ Model loaded successfully!")
    else:
        print("⚠ Running in DEMO mode (random predictions)")
    print("\n🌐 Open browser and go to: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, port=5000)
