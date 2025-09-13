import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import joblib
from flask_cors import CORS
from app.models.Model_training import recommend_employees, agg, employee_skills
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)
CORS(app)


lr = joblib.load('app/models/trained_models/lr.joblib')
rf = joblib.load('app/models/trained_models/rf.joblib')
xgb_model = joblib.load('app/models/trained_models/xgb.joblib')
team_encoder = joblib.load('app/models/trained_models/team_encoder.joblib')

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

@app.route('/api/send_email', methods=['POST'])
def send_email():
    # email sent ideally by smtp but gotta change for outlook i think
    data = request.json
    employee_id = data.get('employeeId')
    task_skills = data.get('taskSkills', [])

    if employee_id not in EMPLOYEE_DATA:
        return jsonify({"status": "error", "message": "Employee not found."}), 404

    employee = EMPLOYEE_DATA[employee_id]
    recipient_email = employee['email']
    employee_name = employee['name']

    sender_email = "#" 
    sender_password = "#"  
    smtp_server = "#"
    smtp_port = 587

    subject = f"New Task Recommendation: {employee_name}"
    body = (
        f"Hi {employee_name},\n\n"
        f"You have been recommended for a new task based on your skills.\n\n"
        f"Required skills for the task: {', '.join(task_skills)}\n\n"
        f"Please check with your manager for more details.\n\n"
        f"Best regards,\n"
        f"The Nokia Sprint Planning System"
    )

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return jsonify({"status": "success", "message": f"Email sent successfully to {employee_name}."}), 200
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({"status": "error", "message": f"Failed to send email. Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
