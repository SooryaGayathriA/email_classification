#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import spacy

# Load models and spaCy
model = joblib.load("email_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")
nlp = spacy.load("en_core_web_sm")

# FastAPI app instance
app = FastAPI()

# Input format
class EmailInput(BaseModel):
    email_body: str

# Masking function
def mask_pii(text):
    masked_text = text
    entities_list = []

    patterns = {
        'email': r'[\w\.-]+@[\w\.-]+',
        'phone_number': r'\b\d{10}\b',
        'aadhar_num': r'\b\d{4} \d{4} \d{4}\b',
        'credit_debit_no': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
        'cvv_no': r'\b\d{3}\b',
        'expiry_no': r'\b(0[1-9]|1[0-2])/?([0-9]{2})\b',
        'dob': r'\b(?:\d{2}[-/]){2}\d{4}\b'
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entity = match.group()
            start, end = match.span()
            entities_list.append({"position": [start, end], "classification": label, "entity": entity})
            masked_text = masked_text.replace(entity, f"[{label}]")

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char
            entity = ent.text
            entities_list.append({"position": [start, end], "classification": "full_name", "entity": entity})
            masked_text = masked_text.replace(entity, "[full_name]")

    return masked_text, entities_list

# Classification endpoint
@app.post("/classify")
def classify(input: EmailInput):
    text = input.email_body
    masked_text, entities = mask_pii(text)
    X = vectorizer.transform([masked_text])
    category = model.predict(X)[0]

    return {
        "input_email_body": text,
        "list_of_masked_entities": entities,
        "masked_email": masked_text,
        "category_of_the_email": category
    }

