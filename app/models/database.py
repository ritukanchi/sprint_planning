import sqlite3
import pandas as pd
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

DATABASE = 'app/database.db'
DATA_EXCEL_PATH = 'app/models/employee_performance_cleaned_v2.xlsx'

def get_db_connection(db_path=DATABASE):
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row 
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database at {db_path}: {e}")
        return None

def init_db(db_path=DATABASE, file_path=DATA_EXCEL_PATH):
    logging.info("Starting database initialization process (ETL)...")

    if not os.path.exists(file_path):
        logging.error(f"FATAL: Excel file not found at {file_path}. Database cannot be populated.")
        return False

    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully read data from {file_path}. Rows found: {len(df)}")
    except Exception as e:
        logging.error(f"Error reading Excel file: {e}")
        return False

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

    def normalize_skillset(text):
        if pd.isna(text): 
            return []
        return [p.strip().lower() for p in str(text).split(",") if p.strip()]
    df['feedback_score'] = df['Feedback'].apply(score_feedback)
    
    agg = df.groupby('Employee_ID').agg(
        employee_avg_efficiency=('Efficiency', 'mean'),
        employee_feedback_mean=('feedback_score', 'mean'),
        tasks_done_count=('Task', 'count'), 
        employee_name=('Employee Name', 'first'),
        email=('Email_ID', 'first'),
        team=('Team', 'first')
    ).reset_index()
    
    agg = agg.rename(columns={
        'employee_name': 'Employee_Name',
        'email': 'Email',
        'team': 'Team'
    })

    skills_list = []
    for _, row in df.iterrows():
        for skill in normalize_skillset(row['Skillset']):
            skills_list.append({'Employee_ID': row['Employee_ID'], 'Skill': skill})
    
    skills_df = pd.DataFrame(skills_list).drop_duplicates()
    
    logging.info(f"Data transformed. Aggregated employees: {len(agg)}, Unique skill links: {len(skills_df)}")

    conn = get_db_connection(db_path)
    if conn is None:
        return False

    try:
        agg.to_sql('employees', conn, if_exists='replace', index=False)
        skills_df.to_sql('skills', conn, if_exists='replace', index=False)
        
        logging.info("Data successfully loaded into 'employees' and 'skills' tables.")
        return True
    
    except Exception as e:
        logging.error(f"Error during SQLite data loading: {e}")
        return False
    
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    init_db()
