import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import os
import warnings
import sqlite3

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

def get_data_from_db(db_path='app/database.db'):
    """Fetches employee and skill data from the SQLite database."""
    conn = sqlite3.connect(db_path)
    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
    skills_df = pd.read_sql_query("SELECT * FROM skills", conn)
    conn.close()
    
    employee_skills = skills_df.groupby('Employee_ID')['Skill'].apply(set).to_dict()
    
    # Re-build the full dataframe for feature engineering
    df = pd.read_excel('app/models/employee_performance_cleaned_v2.xlsx')
    def score_feedback(feedback):
        if pd.isna(feedback):
            return 3
        f = str(feedback).lower()
        if "excellent" in f or "very efficient" in f:
            return 5
        elif "good" in f or "clear" in f or "creative" in f:
            return 4
        elif "average" in f:
            return 3
        elif "needs improvement" in f or "issues" in f or "delayed" in f:
            return 2
        else:
            return 3
    df['feedback_score'] = df['Feedback'].apply(score_feedback)
    
    def normalize_skillset(text):
        if pd.isna(text): 
            return []
        return [p.strip().lower() for p in str(text).split(",")]
    df['skill_list'] = df['Skillset'].apply(normalize_skillset)
    
    return employees_df, employee_skills, df

def train_and_save_models():
    """Trains the ML models and saves them to disk."""
    print("Loading data from database and training models...")
    employees_df, employee_skills, df = get_data_from_db()

    def skill_match_ratio(task_skills, emp_skills):
        if len(task_skills) == 0:
            return 0.0
        return len(set(task_skills).intersection(emp_skills)) / len(task_skills)

    rows = []
    for _, r in df.iterrows():
        emp_id = r['Employee_ID']
        emp_profile = employees_df[employees_df['Employee_ID'] == emp_id].iloc[0]

        rows.append({
            'employee_id': emp_id,
            'skill_match_ratio': skill_match_ratio(r['skill_list'], employee_skills.get(emp_id, set())),
            'employee_avg_efficiency': emp_profile['employee_avg_efficiency'],
            'employee_feedback_mean': emp_profile['employee_feedback_mean'],
            'tasks_done_count': emp_profile['tasks_done_count'],
            'team': emp_profile['team'],
            'target_efficiency': r['Efficiency']
        })

    features_df = pd.DataFrame(rows)
    team_encoder = LabelEncoder()
    features_df['team_label'] = team_encoder.fit_transform(features_df['team'])

    X = features_df[['skill_match_ratio', 'employee_avg_efficiency', 'employee_feedback_mean', 'tasks_done_count', 'team_label']]
    y = features_df['target_efficiency']

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0, objective='reg:squarederror')

    lr.fit(X, y)
    rf.fit(X, y)
    xgb_model.fit(X, y)

    # Save models and encoder
    models_dir = 'app/models/trained_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    joblib.dump(lr, os.path.join(models_dir, 'lr.joblib'))
    joblib.dump(rf, os.path.join(models_dir, 'rf.joblib'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgb.joblib'))
    joblib.dump(team_encoder, os.path.join(models_dir, 'team_encoder.joblib'))
    
    print(" Models and encoder saved successfully.")
employees_df, employee_skills, df = get_data_from_db()

def recommend_employees(new_task_skill_text, top_n=5, models=None, team_encoder=None):
    if models is None:
        models = [joblib.load('app/models/trained_models/lr.joblib'),
                  joblib.load('app/models/trained_models/rf.joblib'),
                  joblib.load('app/models/trained_models/xgb.joblib')]
    if team_encoder is None:
        team_encoder = joblib.load('app/models/trained_models/team_encoder.joblib')

    def normalize_skillset(text):
        if pd.isna(text): 
            return []
        return [p.strip().lower() for p in str(text).split(",")]

    new_task_skills = normalize_skillset(new_task_skill_text)
    rows = []
    for _, prof in employees_df.iterrows():
        X_row = pd.DataFrame([{
            'skill_match_ratio': len(set(new_task_skills).intersection(employee_skills.get(prof['Employee_ID'], set()))) / len(new_task_skills) if new_task_skills else 0.0,
            'employee_avg_efficiency': prof['employee_avg_efficiency'],
            'employee_feedback_mean': prof['employee_feedback_mean'],
            'tasks_done_count': prof['tasks_done_count'],
            'team_label': team_encoder.transform([prof['team']])[0] if prof['team'] in team_encoder.classes_ else 0
        }])
        preds = [m.predict(X_row)[0] for m in models]
        avg_score = float(np.mean(preds))
        rows.append({
            'Employee_ID': prof['Employee_ID'],
            'Employee_Name': prof['employee_name'],
            'Email': prof['email'],
            'Predicted_Efficiency': round(avg_score, 2),
            'Skill_Match': round(X_row['skill_match_ratio'].iloc[0], 2)
        })
    return pd.DataFrame(rows).sort_values('Predicted_Efficiency', ascending=False).head(top_n)

agg = employees_df.set_index('Employee_ID')
if __name__ == '__main__':
    train_and_save_models()
