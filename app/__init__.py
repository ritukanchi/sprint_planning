from flask import Flask, render_template, request, jsonify
from models.Model_training import recommend_employees  # Assuming the function is accessible

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    task_skills = request.json.get('task_skills', '')
    top_n = request.json.get('top_n', 5)
    recommendations = recommend_employees(task_skills, top_n)
    return jsonify(recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)