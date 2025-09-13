import sqlite3
import pandas as pd
import os

def create_db_and_tables(db_path='app/database.db'):
    """Creates the SQLite database and necessary tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            Employee_ID TEXT PRIMARY KEY,
            Employee_Name TEXT,
            Email TEXT,
            Team TEXT,
            employee_avg_efficiency REAL,
            employee_feedback_mean REAL,
            tasks_done_count INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS skills (
            Employee_ID TEXT,
            Skill TEXT,
            FOREIGN KEY (Employee_ID) REFERENCES employees (Employee_ID)
        )
    ''')

    conn.commit()
    conn.close()
    print(" Database and tables created successfully.")


def load_data_from_excel(db_path='app/database.db', file_path='app/models/employee_performance_cleaned_v2.xlsx'):
# loads data into sqlite database from the excel sheet
    if not os.path.exists(file_path):
        print(f"Error: Excel file not found at {file_path}. Cannot load data.")
        return False

    df = pd.read_excel(file_path)

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

    agg = df.groupby('Employee_ID').agg(
        employee_avg_efficiency=('Efficiency', 'mean'),
        employee_feedback_mean=('feedback_score', 'mean'),
        tasks_done_count=('Task', 'count'),
        employee_name=('Employee Name', 'first'),
        email=('Email_ID', 'first'),
        team=('Team', 'first')
    ).reset_index()

    skills_list = []
    for _, row in df.iterrows():
        for skill in normalize_skillset(row['Skillset']):
            skills_list.append({'Employee_ID': row['Employee_ID'], 'Skill': skill})
    skills_df = pd.DataFrame(skills_list)
    
    conn = sqlite3.connect(db_path)
    agg.to_sql('employees', conn, if_exists='replace', index=False)
    skills_df.to_sql('skills', conn, if_exists='replace', index=False)
    conn.close()

    print(" Data successfully loaded into SQLite database.")
    return True

if __name__ == '__main__':
    create_db_and_tables()
    load_data_from_excel()
