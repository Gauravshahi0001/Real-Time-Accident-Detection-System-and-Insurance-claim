from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///accidents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Model
class Accident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    location = db.Column(db.String(200), nullable=False)
    severity = db.Column(db.Integer)
    description = db.Column(db.Text)
    vehicle_count = db.Column(db.Integer)
    injuries = db.Column(db.Integer)
    status = db.Column(db.String(50))
    gender = db.Column(db.String(10))  # Adding gender field

# Create tables
with app.app_context():
    db.create_all()

# Load ML model (will be trained separately)
def load_model():
    try:
        return joblib.load('model/accident_classifier.joblib')
    except:
        return RandomForestClassifier()

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/report', methods=['POST'])
def report_accident():
    data = request.json
    try:
        new_accident = Accident(
            location=data['location'],
            severity=data.get('severity', 0),
            description=data['description'],
            vehicle_count=data.get('vehicle_count', 1),
            injuries=data.get('injuries', 0),
            status='reported',
            gender=data.get('gender', 'not_specified')  # Adding gender to the report
        )
        db.session.add(new_accident)
        db.session.commit()
        
        # Predict severity if not provided
        if 'severity' not in data:
            features = [[
                data.get('vehicle_count', 1),
                data.get('injuries', 0),
                # Add more relevant features
            ]]
            predicted_severity = model.predict(features)[0]
            new_accident.severity = predicted_severity
            db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'Accident reported successfully',
            'id': new_accident.id
        }), 201
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/accidents', methods=['GET'])
def get_accidents():
    accidents = Accident.query.order_by(Accident.timestamp.desc()).all()
    return jsonify([{
        'id': a.id,
        'timestamp': a.timestamp.isoformat(),
        'location': a.location,
        'severity': a.severity,
        'description': a.description,
        'vehicle_count': a.vehicle_count,
        'injuries': a.injuries,
        'status': a.status,
        'gender': a.gender  # Adding gender to the response
    } for a in accidents])

if __name__ == '__main__':
    app.run(debug=True)
