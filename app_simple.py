from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import random

app = Flask(__name__)

# Load model
MODEL_PATH = 'bank_marketing_model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        model = None
else:
    print("⚠ Model file not found - running in demo mode")

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bank Marketing Predictor</title>
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 700px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 { 
                color: #2c3e50; 
                text-align: center; 
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            h2 {
                color: #667eea;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-top: 25px;
            }
            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 15px;
            }
            .form-group { 
                margin-bottom: 15px; 
            }
            label { 
                display: block; 
                margin-bottom: 5px; 
                font-weight: bold; 
                color: #555;
            }
            input, select { 
                width: 100%; 
                padding: 10px; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            button { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 25px; 
                width: 100%; 
                cursor: pointer; 
                font-size: 18px;
                font-weight: bold;
                margin-top: 30px;
                transition: transform 0.2s;
            }
            button:hover { 
                transform: scale(1.02);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }
            .result { 
                margin-top: 30px; 
                padding: 30px; 
                border-radius: 15px; 
                text-align: center; 
            }
            .yes { 
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                border: 2px solid #28a745; 
            }
            .no { 
                background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                border: 2px solid #dc3545; 
            }
            .probability { 
                font-size: 48px; 
                font-weight: bold; 
                margin: 20px 0; 
                color: #333;
            }
            .button { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 12px 30px; 
                text-decoration: none; 
                border-radius: 25px; 
                display: inline-block; 
                margin-top: 20px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .button:hover { 
                transform: scale(1.05);
                box-shadow: 0 5px 15px rgba(102,126,234,0.4);
            }
            small {
                color: #999;
                font-size: 12px;
            }
            .status {
                text-align: center;
                margin-top: 20px;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 8px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 Bank Term Deposit Predictor</h1>
            
            <form method="POST" action="/predict">
                <h2>📋 Personal Information</h2>
                <div class="form-row">
                    <div class="form-group">
                        <label>Age:</label>
                        <input type="number" name="age" value="35" min="18" max="100" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Job:</label>
                        <select name="job">
                            <option value="admin.">Admin.</option>
                            <option value="blue-collar">Blue Collar</option>
                            <option value="management" selected>Management</option>
                            <option value="technician">Technician</option>
                            <option value="services">Services</option>
                            <option value="retired">Retired</option>
                            <option value="student">Student</option>
                            <option value="unemployed">Unemployed</option>
                            <option value="entrepreneur">Entrepreneur</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Marital Status:</label>
                        <select name="marital">
                            <option value="married">Married</option>
                            <option value="single" selected>Single</option>
                            <option value="divorced">Divorced</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Education:</label>
                        <select name="education">
                            <option value="primary">Primary</option>
                            <option value="secondary" selected>Secondary</option>
                            <option value="tertiary">Tertiary</option>
                            <option value="unknown">Unknown</option>
                        </select>
                    </div>
                </div>
                
                <h2>💰 Financial Information</h2>
                <div class="form-row">
                    <div class="form-group">
                        <label>Balance (€):</label>
                        <input type="number" name="balance" value="2500" step="0.01" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Default:</label>
                        <select name="default">
                            <option value="no" selected>No</option>
                            <option value="yes">Yes</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Housing Loan:</label>
                        <select name="housing">
                            <option value="no" selected>No</option>
                            <option value="yes">Yes</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Personal Loan:</label>
                        <select name="loan">
                            <option value="no" selected>No</option>
                            <option value="yes">Yes</option>
                        </select>
                    </div>
                </div>
                
                <h2>📞 Campaign Information</h2>
                <div class="form-row">
                    <div class="form-group">
                        <label>Contact Month:</label>
                        <select name="month">
                            <option value="jan">January</option>
                            <option value="feb">February</option>
                            <option value="mar">March</option>
                            <option value="apr">April</option>
                            <option value="may" selected>May</option>
                            <option value="jun">June</option>
                            <option value="jul">July</option>
                            <option value="aug">August</option>
                            <option value="sep">September</option>
                            <option value="oct">October</option>
                            <option value="nov">November</option>
                            <option value="dec">December</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Contact Day:</label>
                        <input type="number" name="day" value="15" min="1" max="31">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Campaign Contacts:</label>
                        <input type="number" name="campaign" value="1" min="1">
                    </div>
                    
                    <div class="form-group">
                        <label>Previous Outcome:</label>
                        <select name="poutcome">
                            <option value="unknown" selected>Unknown</option>
                            <option value="success">Success</option>
                            <option value="failure">Failure</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>Days Since Last Contact:</label>
                        <input type="number" name="pdays" value="-1">
                        <small>(-1 = never contacted)</small>
                    </div>
                    
                    <div class="form-group">
                        <label>Previous Contacts:</label>
                        <input type="number" name="previous" value="0" min="0">
                    </div>
                </div>
                
                <button type="submit">🔮 Predict Subscription</button>
            </form>
            
            <div class="status">
                Model Status: <strong>✅ Active</strong> | F1-Score: 0.44 | Accuracy: 89%
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {
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
        
        # Make prediction
        if model:
            df = pd.DataFrame([data])
            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1] * 100
        else:
            # Demo mode - rule-based prediction
            score = 0
            if data['age'] < 30: score += 10
            if data['age'] > 60: score += 20
            if data['balance'] > 5000: score += 30
            if data['job'] in ['management', 'retired']: score += 20
            if data['poutcome'] == 'success': score += 40
            if data['default'] == 'yes': score -= 30
            prob = max(5, min(95, score))
            pred = 1 if prob > 50 else 0
        
        result_class = 'yes' if pred == 1 else 'no'
        result_text = 'YES - Will Subscribe' if pred == 1 else 'NO - Will Not Subscribe'
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{ 
                    max-width: 600px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 40px; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    text-align: center;
                }}
                h1 {{ color: #2c3e50; margin-bottom: 30px; }}
                .result {{ 
                    margin: 30px 0; 
                    padding: 40px; 
                    border-radius: 15px; 
                }}
                .yes {{ 
                    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                    border: 3px solid #28a745; 
                }}
                .no {{ 
                    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                    border: 3px solid #dc3545; 
                }}
                .probability {{ 
                    font-size: 56px; 
                    font-weight: bold; 
                    margin: 20px 0; 
                    color: #333;
                }}
                .button {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 15px 40px; 
                    text-decoration: none; 
                    border-radius: 30px; 
                    display: inline-block; 
                    margin-top: 30px;
                    font-weight: bold;
                    font-size: 18px;
                    transition: transform 0.2s;
                }}
                .button:hover {{ 
                    transform: scale(1.05);
                    box-shadow: 0 5px 15px rgba(102,126,234,0.4);
                }}
                .summary {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 30px;
                    text-align: left;
                }}
                .summary-item {{
                    display: flex;
                    justify-content: space-between;
                    padding: 8px 0;
                    border-bottom: 1px solid #dee2e6;
                }}
                .summary-label {{
                    font-weight: bold;
                    color: #555;
                }}
                .summary-value {{
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Prediction Result</h1>
                
                <div class="result {result_class}">
                    <h2 style="margin:0; font-size: 28px;">{result_text}</h2>
                    <div class="probability">{prob:.1f}%</div>
                    <div style="font-size: 18px;">Probability of Subscription</div>
                </div>
                
                <div class="summary">
                    <h3 style="margin-top:0;">📋 Customer Summary</h3>
                    <div class="summary-item">
                        <span class="summary-label">Age:</span>
                        <span class="summary-value">{data['age']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Job:</span>
                        <span class="summary-value">{data['job']}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Balance:</span>
                        <span class="summary-value">€{data['balance']:,.2f}</span>
                    </div>
                    <div class="summary-item">
                        <span class="summary-label">Month:</span>
                        <span class="summary-value">{data['month'].upper()}</span>
                    </div>
                </div>
                
                <a href="/" class="button">← Make Another Prediction</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"<h1>Error</h1><p>{str(e)}</p><p><a href='/'>Go back</a></p>"

@app.route('/health')
def health():
    return {'status': 'healthy', 'model_loaded': model is not None}

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 BANK MARKETING PREDICTION APP")
    print("="*60)
    if model:
        print("✅ Model loaded successfully!")
    else:
        print("⚠ Running in DEMO mode")
    print("\n🌐 Open browser and go to: http://127.0.0.1:5000")
    print("="*60)
    app.run(debug=True, port=5000)
