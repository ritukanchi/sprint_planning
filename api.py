import json
import os
import joblib
from flask_cors import CORS
from app.models.Model_training import recommend_employees, agg, employee_skills
from flask import Flask, jsonify, request, redirect, url_for, render_template
import pandas as pd
from dotenv import load_dotenv 
import sqlite3 
import logging

logging.basicConfig(level=logging.DEBUG)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'app', 'templates')
MODELS_PATH = os.path.join(BASE_DIR, 'app', 'models', 'trained_models')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

lr, rf, xgb_model, team_encoder = None, None, None, None

try:
    lr = joblib.load(os.path.join(MODELS_PATH, 'lr.joblib'))
    rf = joblib.load(os.path.join(MODELS_PATH, 'rf.joblib'))
    xgb_model = joblib.load(os.path.join(MODELS_PATH, 'xgb.joblib'))
    team_encoder = joblib.load(os.path.join(MODELS_PATH, 'team_encoder.joblib'))
    logging.info("ML models loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load one or more ML models from {MODELS_PATH}: {e}")

@app.route("/")
def home():
    """Redirect root to dashboard"""
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route('/api/recommendations', methods=['GET'])
def get_static_recommendations():
    path = os.path.join(BASE_DIR, 'app', 'models', 'recommendations.json')
    
    if not os.path.exists(path):
        return jsonify({"error": "Static recommendations file not found."}), 404
        
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        logging.error(f"Error loading recommendations.json: {e}")
        return jsonify({"error": "Failed to load static recommendations."}), 500


@app.route('/api/recommendations', methods=['POST'])
def recommend_employees_api():
    if not all([lr, rf, xgb_model, team_encoder]):
        return jsonify({"error": "ML models are not available on the server. Deployment failed to load assets."}), 503
        
    data = request.get_json()
    task_skills = data.get('task_skills', '')
    top_n = data.get('top_n', 10)
    
    if agg is None or employee_skills is None:
         return jsonify({"error": "ML backend data (agg/skills) failed to load. Check database connectivity."}), 500

    try:
        recommendations_df = recommend_employees(
            task_skills, top_n=top_n, models=[lr, rf, xgb_model], team_encoder=team_encoder
        )
        
        recommendations_df['Team'] = recommendations_df['Employee_ID'].map(lambda eid: agg.loc[eid, 'Team'])
        recommendations_df['Skills'] = recommendations_df['Employee_ID'].map(lambda eid: list(employee_skills.get(eid, [])))

        recommendations_df['Team'] = recommendations_df['Team'].fillna('')
        
        return jsonify(recommendations_df.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"Error during recommendation generation: {e}")
        return jsonify({"error": "Failed to process recommendation request.", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
