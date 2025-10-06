import json
import os
import joblib
from flask_cors import CORS
from app.models.Model_training import recommend_employees, agg, employee_skills
from flask import Flask, jsonify, request, redirect, url_for, render_template
import pandas as pd
from dotenv import load_dotenv 
import sqlite3 

app = Flask(__name__)
CORS(app)
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr = joblib.load(os.path.join(BASE_DIR, 'app/models/trained_models/lr.joblib'))
rf = joblib.load(os.path.join(BASE_DIR, 'app/models/trained_models/rf.joblib'))
xgb_model = joblib.load(os.path.join(BASE_DIR, 'app/models/trained_models/xgb.joblib'))
team_encoder = joblib.load(os.path.join(BASE_DIR, 'app/models/trained_models/team_encoder.joblib'))


@app.route("/")
def home():
    """Redirect root to dashboard"""
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    """Serve the dashboard frontend"""
    return render_template("dashboard.html")

@app.route('/api/recommendations', methods=['GET'])
def get_static_recommendations():
    path = os.path.join('models', 'recommendations.json')
    if not os.path.exists(path):
        return jsonify([])
    with open(path, 'r') as f:
        data = json.load(f)
    return jsonify(data)



@app.route('/api/recommendations', methods=['POST'])
def recommend_employees_api():
    data = request.get_json()
    task_skills = data.get('task_skills', '')
    top_n = data.get('top_n', 10)
    recommendations_df = recommend_employees(
        task_skills, top_n=top_n, models=[lr, rf, xgb_model], team_encoder=team_encoder
    )
    recommendations_df['Team'] = recommendations_df['Employee_ID'].map(lambda eid: agg.loc[eid]['team'])
    recommendations_df['Skills'] = recommendations_df['Employee_ID'].map(lambda eid: list(employee_skills[eid]))
    return jsonify(recommendations_df.to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
