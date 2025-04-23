#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import re
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Load multilingual SpaCy model
nlp = spacy.load("xx_ent_wiki_sm")

# Initialize FastAPI app
app = FastAPI()

# ------------------------------------
# 🔹 STEP 1: LOAD DATASET FROM CSV
# ------------------------------------
csv_path = r"C:\Users\pavin\OneDrive\Desktop\combined_emails_with_natural_pii.csv"  # file path
df = pd.read_csv(csv_path)

# ------------------------------------
# 🔹 STEP 2: PII MASKING FUNCTION WITH CONTEXT-BASED ENTITY DETECTION
# ------------------------------------
def mask_pii(text):
    """
    Function to mask Personally Identifiable Information (PII) entities
    in the given email text, considering the context before the entity.
    Args:
        text (str): The input email body text.
    Returns:
        tuple: A tuple containing the masked text and a list of masked entities.
    """
    masked_text = text
    entities_list = []

    # Define patterns for each entity type
    patterns = {
        'email': r'[\w\.-]+@[\w\.-]+',
        'phone_number': r'\+?\d{1,4}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',
        # Updated pattern to handle various formats of Aadhar number
        'aadhar_num': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'credit_debit_no': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b(0[1-9]|1[0-2])/?([0-9]{2})\b',
        'dob': r'\b(?:\d{2}[-/]){2}\d{4}\b'
    }

    # Keywords preceding the entities
    keywords = {
        'aadhar_num': r'(?:aadhar|aadhaar|adhar|adaar|uidai)',  # Updated to capture all variations
        'credit_debit_no': r'(?:credit|debit|card)',
        'cvv_no': r'(?:cvv)',
        'expiry_no': r'(?:expiry|exp date)',
        'dob': r'(?:dob|date of birth|birthday)',
        'email': r'(?:email|mail id|email id)',
        'phone_number': r'(?:phone|mobile|contact)',
    }

    # Iterate through patterns and match them with the keywords
    for label, pattern in patterns.items():
        # Combine the keyword and the pattern for the entity
        combined_pattern = r'\b(?:' + keywords[label] + r')\b.*?(' + pattern + r')'

        # Search for the combined pattern in the text
        for match in re.finditer(combined_pattern, text):
            entity = match.group(1)  # Get the matched entity (the actual number or email)
            start, end = match.span(1)  # Get start and end positions of the entity

            # Append the matched entity to the entities list with classification
            entities_list.append({
                "position": [start, end],
                "classification": label,
                "entity": entity
            })

            # Mask entity in the text
            masked_text = masked_text.replace(entity, f"[{label}]")

    # Use spaCy for detecting full names
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entity = ent.text
            entities_list.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": entity
            })
            masked_text = masked_text.replace(entity, "[full_name]")

    return masked_text, entities_list


# Apply masking to dataset
masked_data = df['email'].apply(mask_pii)
df['masked_body'] = masked_data.apply(lambda x: x[0])
df['entities'] = masked_data.apply(lambda x: x[1])

# ------------------------------------
# 🔹 STEP 3: LABEL ENCODING
# ------------------------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['type'])

# ------------------------------------
# 🔹 STEP 4: TEXT VECTORIZATION
# ------------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['masked_body']).toarray()
y = df['label']

# ------------------------------------
# 🔹 STEP 5: SPLIT AND TRAIN
# ------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# ------------------------------------
# 🔹 STEP 6: PREDICT FUNCTION
# ------------------------------------
def predict_email(email):
    masked_email, entities = mask_pii(email)
    email_vec = vectorizer.transform([masked_email]).toarray()
    predicted_label = classifier.predict(email_vec)[0]
    original_label = label_encoder.inverse_transform([predicted_label])[0]
    return {
        "input_email_body": email,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": original_label
    }

# ------------------------------------
# 🔹 STEP 7: FASTAPI INFERENCE
# ------------------------------------
class EmailRequest(BaseModel):
    subject: str
    body: str

@app.post("/predict")
async def classify_email(request: EmailRequest):
    combined_email = f"Subject: {request.subject}\n{request.body}"
    result = predict_email(combined_email)
    return result
# ------------------------------------
# 🔹 STEP 8: CLASSIFICATION ACCURACY REPORT
# ------------------------------------
# Predict on the test set
y_pred = classifier.predict(X_test)

# Print classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# ------------------------------------
# 🔹 STEP 8: LOCAL TEST
# ------------------------------------
if __name__ == "__main__":
    test_email = {
        "subject": "Ratung für Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen",
        "body": "Ratung, ob es möglich ist, Sicherung medizinischer Daten in HubSpot CRM PostgreSQL-Umgebungen durchzuführen? Danke.. My contact number is +971-50-123-4567."
    }
    combined = f"Subject: {test_email['subject']}\n{test_email['body']}"
    output = predict_email(combined)
    print(json.dumps(output, ensure_ascii=False, indent=2))


# In[13]:





# In[9]:





# In[11]:


import pickle

# Save classifier
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


# In[ ]:




