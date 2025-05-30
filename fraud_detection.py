import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def load_data(filepath):
    print("Loading dataset...")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    return data

def preprocess_data(data):
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

def train_model(X_train, y_train):
    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

def test_transaction(model):
    print("\nTest your own transaction (enter feature values):")
    
    # You can customize the features you want the user to input here.
    # Credit Card Fraud dataset has 30 features named V1, V2, ..., V28, plus 'Time' and 'Amount'.
    # For simplicity, just ask for 'Amount' and a few example features, or all 30 features if you prefer.
    
    # For demo, let's ask for Amount and Time, and set other features to 0 for simplicity.
    
    try:
        time = float(input("Enter Time (seconds since first transaction): "))
        amount = float(input("Enter Amount: "))
        
        # Create feature vector - set V1 to V28 to 0 for simplicity (replace with real values if you want)
        features = [time] + [0]*28 + [amount]
        
        # The order of columns in the dataset is: Time, V1,...,V28, Amount
        # So features length = 30
        
        import numpy as np
        features = np.array(features).reshape(1, -1)
        
        pred = model.predict(features)[0]
        if pred == 1:
            print("Prediction: Fraudulent transaction ⚠️")
        else:
            print("Prediction: Legitimate transaction ✅")
    except Exception as e:
        print("Error in input:", e)

def main():
    # Load data
    data = load_data("creditcard.csv")
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model & vectorizer if needed (optional)
    # e.g. use joblib.dump(model, 'rf_model.pkl')
    
    # Test interface - keep asking user if they want to test
    while True:
        test_transaction(model)
        cont = input("\nTest another transaction? (y/n): ").strip().lower()
        if cont != 'y':
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
