from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import json
import os
import joblib
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'fraud_detection_secret_key_2024'

# Global variables to store model and data
model = None
model_trained = False
model_metrics = {}

def load_data(filepath="creditcard.csv", sample_size=10000):
    """Load and optionally sample the dataset for faster processing"""
    try:
        print(f"Loading dataset from {filepath}...")
        data = pd.read_csv(filepath)
        
        # Sample data for faster processing in web interface
        if len(data) > sample_size:
            # Ensure we get both fraud and non-fraud samples
            fraud_data = data[data['Class'] == 1]
            non_fraud_data = data[data['Class'] == 0].sample(n=sample_size-len(fraud_data), random_state=42)
            data = pd.concat([fraud_data, non_fraud_data]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """Preprocess the data for training"""
    print("Preprocessing data...")
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Balance dataset with SMOTE
    print("Applying SMOTE to balance the dataset...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"Resampled dataset shape: {X_res.shape}, {y_res.shape}")
    
    return X_res, y_res

def train_model_func(X_train, y_train):
    """Train the Random Forest model"""
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=50)  # Reduced for faster training
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def evaluate_model_func(model, X_test, y_test):
    """Evaluate the model and return metrics"""
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'], output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return metrics

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html', model_trained=model_trained, metrics=model_metrics)

@app.route('/train')
def train_page():
    """Model training page"""
    return render_template('train.html')

@app.route('/predict')
def predict_page():
    """Transaction prediction page"""
    return render_template('predict.html', model_trained=model_trained)

@app.route('/api/train', methods=['POST'])
def train_model_api():
    """API endpoint to train the model"""
    global model, model_trained, model_metrics
    
    try:
        # Load data
        data = load_data()
        if data is None:
            return jsonify({'error': 'Failed to load dataset. Make sure creditcard.csv is in the project directory.'}), 400
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model_func(X_train, y_train)
        
        # Evaluate model
        model_metrics = evaluate_model_func(model, X_test, y_test)
        model_trained = True
        
        # Save model
        joblib.dump(model, 'fraud_model.pkl')
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully!',
            'metrics': model_metrics
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict_transaction():
    """API endpoint to predict a single transaction"""
    global model
    
    if not model_trained or model is None:
        return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
    
    try:
        data = request.json
        
        # Extract features from request
        time = float(data.get('time', 0))
        amount = float(data.get('amount', 0))
        
        # Create feature vector - set V1 to V28 to 0 for simplicity
        # In a real application, you would collect all 30 features
        features = [time] + [0]*28 + [amount]
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'fraud_probability': float(probability[1]),
            'legitimate_probability': float(probability[0]),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/model-status')
def model_status():
    """Get current model status"""
    return jsonify({
        'model_trained': model_trained,
        'metrics': model_metrics if model_trained else None
    })

if __name__ == '__main__':
    # Try to load existing model
    if os.path.exists('fraud_model.pkl'):
        try:
            model = joblib.load('fraud_model.pkl')
            model_trained = True
            print("Loaded existing model from fraud_model.pkl")
        except:
            print("Failed to load existing model")
    
    app.run(debug=True, host='0.0.0.0', port=5000)