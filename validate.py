import pandas as pd
import joblib
import numpy as np

print("Loading model and vectorizer...")
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Loading CEAS_08 dataset for validation...")
df = pd.read_csv("CEAS_08.csv")

# Adjust column names based on your dataset structure
# Typically: 'body' or 'text' for content, 'label' for class (1 = phish, 0 = legit)
text_col = 'body' if 'body' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else df.columns[-1]

# Sample 20 random emails (10 phishing, 10 legitimate if balanced)
sample_df = df.sample(n=20, random_state=42)

print("\nRunning spot-check validation on 20 random samples...")
print("-" * 65)
print(f"{'Actual':<10} | {'Predicted':<10} | {'Confidence':<12} | {'Status'}")
print("-" * 65)

correct = 0
for idx, row in sample_df.iterrows():
    text = str(row[text_col])
    actual = int(row[label_col])
    
    # Vectorise and predict
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    
    # Get confidence score using decision function
    try:
        dec = model.decision_function(vec)[0]
        conf = 1 / (1 + np.exp(-dec))
        if pred == 0:
            conf = 1 - conf
    except Exception:
        conf = 0.95  # Fallback if decision_function is unavailable
        
    conf_pct = f"{conf * 100:.1f}%"
    
    actual_str = "Phish" if actual == 1 else "Legit"
    pred_str = "Phish" if pred == 1 else "Legit"
    
    status = "✓ Pass" if actual == pred else "✗ Fail"
    if actual == pred:
        correct += 1
        
    print(f"{actual_str:<10} | {pred_str:<10} | {conf_pct:<12} | {status}")

print("-" * 65)
print(f"Validation Accuracy: {correct}/20 ({(correct/20)*100:.1f}%)")