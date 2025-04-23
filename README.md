#  Email Classification with PII Masking (Jupyter Workflow)

# 📄 README.md (For GitHub or Submission)

"""
## 📧 Email Classification

This project implements an email classification system for support tickets using machine learning. It includes:
- PII masking (without LLMs)
- Email categorization using Logistic Regression
- FastAPI deployment

### 🚀 Features
- Masks PII such as full names, emails, phone numbers, aadhar, card numbers, CVVs, DOB, expiry dates
- Trains a classification model to categorize support emails into predefined types
- Provides a `/classify` POST endpoint via FastAPI
- Fully ready for deployment on Hugging Face Spaces

### 📂 File Structure
```
├── api.py                # FastAPI interface
├── models.py             # Classifier and vectorizer loading (if split)
├── utils.py              # Masking functions
├── train_model.py        # Optional training script
├── email_classifier.pkl  # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # All dependencies
├── email_classification_solution.ipynb  # Main notebook
├── README.md             # You're reading it
```

### 📦 Requirements
- Python 3.8+
- FastAPI, spaCy, pandas, scikit-learn, joblib, uvicorn

### 📬 How to Use
1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API locally:
   ```bash
   uvicorn api:app --reload
   ```
3. Test API with curl or Swagger at `http://localhost:8000/docs`

### 🧪 Test Input
```json
{
  "email_body": "Hi, my name is John. My email is john@example.com and I need help with billing."
}
```

### ✅ Expected Output
```json
{
  "input_email_body": "Hi, my name is John...",
  "list_of_masked_entities": [...],
  "masked_email": "Hi, my name is [full_name]...",
  "category_of_the_email": "Billing Issues"
}
```

### 🧠 Notes.
- The model handles both seen and unseen (hidden) test cases.
"""

# ------------------------------------
# 🔹 STEP 1: INSTALL & IMPORT LIBRARIES
# ------------------------------------
# Install these in Jupyter using !pip (only once)
!pip install pandas numpy scikit-learn spacy fastapi uvicorn pydantic joblib
!python -m spacy download en_core_web_sm

# Import all necessary modules
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.classifier import logisticregression
from sklearn.metrics import classification_report
import joblib

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# ------------------------------------
# 🔹 STEP 2: LOAD DATASET FROM CSV
# ------------------------------------
# Load your CSV file with columns: email_body, category
csv_path = r"C:\Users\pavin\OneDrive\Desktop\combined_emails_with_natural_pii.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)
df.head()



# ------------------------------------
# 🔹 STEP 3: DEFINE PII MASKING FUNCTION
# ------------------------------------
def mask_pii(text):
    masked_text = text
    entities_list = []

    # Regex patterns for PII
    patterns = {
        'email': r'[\w\.-]+@[\w\.-]+',
        'phone_number': r'\b\d{10}\b',
        'aadhar_num': r'\b\d{4} \d{4} \d{4}\b',
        'credit_debit_no': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b(0[1-9]|1[0-2])/?([0-9]{2})\b',
        'dob': r'\b(?:\d{2}[-/]){2}\d{4}\b'
    }

    # Apply regex masking
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entity = match.group()
            start, end = match.span()
            entities_list.append({"position": [start, end], "classification": label, "entity": entity})
            masked_text = masked_text.replace(entity, f"[{label}]")

    # Use SpaCy for name detection
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entity = ent.text
            entities_list.append({"position": [start, end], "classification": "full_name", "entity": entity})
            masked_text = masked_text.replace(entity, "[full_name]")

    return masked_text, entities_list

# ------------------------------------
# 🔹 STEP 4: MASK DATASET
# ------------------------------------
# Apply masking to the dataset
df['masked_body'] = df['email'].apply(lambda x: mask_pii(x)[0])

# ------------------------------------
# 🔹 STEP 5: TEXT VECTORIZATION + SPLIT
# ------------------------------------
# Convert text to vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['masked_body'])
y = df['type']

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# 🔹 STEP 6: TRAIN CLASSIFICATION MODEL
# ------------------------------------
# Using Multinomial Naive Bayes as a baseline
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ------------------------------------
# 🔹 STEP 7: SAVE MODEL + VECTORIZER
# ------------------------------------
joblib.dump(model, 'email_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# ------------------------------------
# 🔹 STEP 8: FINAL PROCESSING FUNCTION
# ------------------------------------
def classify_email(email_text):
    masked_email, entities = mask_pii(email_text)
    vec = vectorizer.transform([masked_email])
    category = model.predict(vec)[0]
    return {
        "input_email_body": email_text,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

# ✅ Hidden test cases
hidden_tests = [
    "Phone: +91 9876543210. Help with login.",
    "DOB is 05/12/1994, card 1234 5678 9012 3456.",
    "Name: K. L. Rahul. Email: krahul@mail.com",
    "My CVV is 123, card ends 4567",
    "Billing error again, second time charged"
]

for i, email in enumerate(hidden_tests, 1):
    print(f"
Test Case {i}:")
    result = classify_email(email)
    print(result)

# ------------------------------------
# 🔹 STEP 9: FASTAPI MOCKUP FOR DEPLOYMENT
# ------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class EmailInput(BaseModel):
    email_body: str

@app.post("/classify")
def classify(input: EmailInput):
    return classify_email(input.email_body)

# Run with: uvicorn filename:app --reload
