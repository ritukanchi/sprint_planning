from flask import Flask, jsonify
import pandas as pd
import os
import json

app = Flask(__name__)

@app.route('/api/recommendations', methods=['GET'])
def get_static_recommendations():
    path = os.path.join('models', 'recommendations.json')
    if not os.path.exists(path):
        return jsonify([])
    with open(path, 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)