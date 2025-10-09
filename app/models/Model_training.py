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
import logging

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database.db')
EXCEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'employee_performance_cleaned_v2.xlsx')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')


def get_data_from_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    employees_df = pd.read_sql_query("SELECT * FROM employees", conn)
    skills_df = pd.read_sql_query("SELECT * FROM skills", conn)
    conn.close()
    
    employee_skills = skills_df.groupby('Employee_ID')['Skill'].apply(set).to_dict()
    
    df_raw = pd.read_excel(EXCEL_PATH)
    
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
    df_raw['feedback_score'] = df_raw['Feedback'].apply(score_feedback)
    
    def normalize_skillset(text):
        if pd.isna(text): 
            return []
        return [p.strip().lower() for p in str(text).split(",")]
    df_raw['skill_list'] = df_raw['Skillset'].apply(normalize_skillset)
    
    return employees_df, employee_skills, df_raw

def train_and_save_models():
    logging.info("Loading data from database and training models...")
    employees_df, employee_skills, df_raw = get_data_from_db()

    def skill_match_ratio(task_skills, emp_skills):
        if len(task_skills) == 0:
            return 0.0
        return len(set(task_skills).intersection(emp_skills)) / len(task_skills)

    rows = []
    
    for _, r in df_raw.iterrows():
        emp_id = r['Employee_ID']
        emp_profile = employees_df[employees_df['Employee_ID'] == emp_id]
        
        if emp_profile.empty:
            continue
            
        emp_profile = emp_profile.iloc[0] 
        
        rows.append({
            'employee_id': emp_id,
            'skill_match_ratio': skill_match_ratio(r['skill_list'], employee_skills.get(emp_id, set())),
            'employee_avg_efficiency': emp_profile['employee_avg_efficiency'],
            'employee_feedback_mean': emp_profile['employee_feedback_mean'],
            'tasks_done_count': emp_profile['tasks_done_count'],
            'team': emp_profile['Team'], 
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

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    joblib.dump(lr, os.path.join(MODELS_DIR, 'lr.joblib'))
    joblib.dump(rf, os.path.join(MODELS_DIR, 'rf.joblib'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb.joblib'))
    joblib.dump(team_encoder, os.path.join(MODELS_DIR, 'team_encoder.joblib'))
    
    logging.info(" Models and encoder saved successfully.")


try:
    _employees_df_loaded, employee_skills, _ = get_data_from_db()
    agg = _employees_df_loaded.set_index('Employee_ID') 
    logging.info("Global employee data (agg) and skills loaded for API use.")
except Exception as e:
    logging.error(f"Failed to load global data structures for API: {e}")
    agg = None
    employee_skills = None


def recommend_employees(new_task_skill_text, top_n=5, models=None, team_encoder=None):
    if models is None:
        models = [
            joblib.load(os.path.join(MODELS_DIR, 'lr.joblib')),
            joblib.load(os.path.join(MODELS_DIR, 'rf.joblib')),
            joblib.load(os.path.join(MODELS_DIR, 'xgb.joblib'))
        ]
    if team_encoder is None:
        team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))

    def normalize_skillset(text):
        if pd.isna(text): 
            return []
        return [p.strip().lower() for p in str(text).split(",")]

    new_task_skills = normalize_skillset(new_task_skill_text)
    rows = []
    
    employees_df_for_api = agg.reset_index()

    for _, prof in employees_df_for_api.iterrows():
        skill_match_ratio_val = len(set(new_task_skills).intersection(employee_skills.get(prof['Employee_ID'], set()))) / len(new_task_skills) if new_task_skills else 0.0
        
        team_val = prof['Team'] 
        team_label = team_encoder.transform([team_val])[0] if team_val in team_encoder.classes_ else 0

        X_row = pd.DataFrame([{
            'skill_match_ratio': skill_match_ratio_val,
            'employee_avg_efficiency': prof['employee_avg_efficiency'],
            'employee_feedback_mean': prof['employee_feedback_mean'],
            'tasks_done_count': prof['tasks_done_count'],
            'team_label': team_label
        }])
        
        preds = [m.predict(X_row)[0] for m in models]
        avg_score = float(np.mean(preds))
        
        final_score = np.clip(avg_score, 0, 100)
        
        rows.append({
            'Employee_ID': prof['Employee_ID'],
            'Employee_Name': prof['Employee_Name'],
            'Email': prof['Email'],
            'Predicted_Efficiency': round(final_score, 2),
            'Skill_Match': round(skill_match_ratio_val, 2)
        })
    return pd.DataFrame(rows).sort_values('Predicted_Efficiency', ascending=False).head(top_n)


if __name__ == '__main__':
    train_and_save_models()
