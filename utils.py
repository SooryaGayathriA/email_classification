import re
import spacy
import pickle

# Load multilingual SpaCy model
nlp = spacy.load("xx_ent_wiki_sm")

# Load trained model components
with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define keyword-based regex patterns
patterns = {
    'email': r'[\w\.-]+@[\w\.-]+',
    'phone_number': r'\+?\d{1,4}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}',
    'aadhar_num': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b|\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    'credit_debit_no': r'\b(?:\d{4}[- ]?){3}\d{4}\b',
    'cvv_no': r'\b\d{3}\b',
    'expiry_no': r'\b(0[1-9]|1[0-2])/?([0-9]{2})\b',
    'dob': r'\b(?:\d{2}[-/]){2}\d{4}\b'
}

keywords = {
    'aadhar_num': r'(?:aadhar|aadhaar|adhar|adaar|uidai)',
    'credit_debit_no': r'(?:credit|debit|card)',
    'cvv_no': r'(?:cvv)',
    'expiry_no': r'(?:expiry|exp date)',
    'dob': r'(?:dob|date of birth|birthday)',
    'email': r'(?:email|mail id|email id)',
    'phone_number': r'(?:phone|mobile|contact)',
}

def mask_pii(text):
    masked_text = text
    entities_list = []

    for label, pattern in patterns.items():
        combined_pattern = r'\b(?:' + keywords[label] + r')\b.*?(' + pattern + r')'

        for match in re.finditer(combined_pattern, text):
            entity = match.group(1)
            start, end = match.span(1)
            entities_list.append({
                "position": [start, end],
                "classification": label,
                "entity": entity
            })
            masked_text = masked_text.replace(entity, f"[{label}]")

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

def predict_email(email_body):
    masked_email, entities = mask_pii(email_body)
    email_vec = vectorizer.transform([masked_email]).toarray()
    predicted_label = classifier.predict(email_vec)[0]
    original_label = label_encoder.inverse_transform([predicted_label])[0]

    return {
        "input_email_body": email_body,
        "list_of_masked_entities": entities,
        "masked_email": masked_email,
        "category_of_the_email": original_label
    }
