import pandas as pd
import numpy as np
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- STEP 1: GENERATE DATA & TRAIN MODEL ---
def train_model():
    # Generate dummy data for demonstration
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n),
        'Age': np.random.randint(18, 90, n),
        'Tenure': np.random.randint(0, 10, n),
        'Balance': np.random.uniform(0, 100000, n),
        'NumOfProducts': np.random.randint(1, 4, n),
        'HasCrCard': np.random.randint(0, 2, n),
        'Exited': np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    })

    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Train the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- STEP 2: DEFINE PREDICTION FUNCTION ---
def predict_churn(credit_score, age, tenure, balance, products, has_card):
    input_data = pd.DataFrame([[credit_score, age, tenure, balance, products, has_card]],
                              columns=['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        return f"⚠️ RISK ALERT! Probability of Churn: {probability:.1%}"
    else:
        return f"✅ SAFE. Probability of Churn: {probability:.1%}"

# --- STEP 3: LAUNCH WEB APP ---
interface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Slider(300, 850, label="Credit Score"),
        gr.Slider(18, 90, label="Age"),
        gr.Slider(0, 10, label="Tenure (Years)"),
        gr.Number(label="Account Balance"),
        gr.Slider(1, 4, step=1, label="Number of Products"),
        gr.Radio([0, 1], label="Has Credit Card? (0=No, 1=Yes)")
    ],
    outputs="text",
    title="Customer Churn Prediction System",
    description="Adjust the sliders to predict if a customer is likely to leave the bank."
)

if __name__ == "__main__":
    interface.launch()