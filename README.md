
# PII Masking and Email Classification API

This project provides an API to mask Personally Identifiable Information (PII) in email bodies and classify the email content. It uses a combination of regex-based entity detection and SpaCy for named entity recognition.

## Features

- **PII Masking**: Masks sensitive information like phone numbers, email addresses, Aadhar numbers, credit/debit card details, CVV, expiry dates, and date of birth.
- **Entity Detection**: Identifies and masks full names using SpaCy's named entity recognition.
- **Email Classification**: Classifies the email based on the content after masking PII entities using a trained machine learning model.

## Technologies Used

- **FastAPI**: Web framework to create the API.
- **SpaCy**: NLP library for named entity recognition (NER).
- **Regex**: Used for detecting and masking PII entities in text.
- **Scikit-learn**: Machine learning library for email classification.
- **Pickle**: Serialization library for loading pre-trained model components.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SpaCy Model

```bash
python -m spacy download xx_ent_wiki_sm
```

### 3. Load Pre-trained Model Components

Ensure the following files are available in your project directory:

- `classifier.pkl`: Trained model for email classification.
- `vectorizer.pkl`: Trained vectorizer for transforming text data.
- `label_encoder.pkl`: Label encoder for mapping predicted labels back to their original form.

### 4. Deploy on Hugging Face

To deploy the FastAPI app on Hugging Face Spaces, follow these steps:

1. **Create a Hugging Face account** (if you don’t have one).
2. **Create a new Space**:
   - Go to your [Hugging Face Spaces](https://huggingface.co/spaces) and create a new space.
   - Select the **FastAPI** template.

3. **Push Your Code to Hugging Face**:
   - Clone your newly created Hugging Face Space repository.
   - Push your local code (including `api.py`, `requirements.txt`, and model files like `classifier.pkl`, `vectorizer.pkl`, `label_encoder.pkl`) to the repository.

4. **Start the FastAPI App**:
   - Hugging Face will automatically detect the FastAPI app and deploy it.
   - Your FastAPI server will be available at the Hugging Face URL provided for your space.

## API Endpoints

### POST `/predict`

Classify an email and mask PII entities.

#### Request Body:

```json
{
  "email_body": "Your email body content goes here."
}
```

#### Response:

```json
{
    "input_email_body": "Original email body",
    "list_of_masked_entities": [
        {
            "position": [start, end],
            "classification": "entity_type",
            "entity": "masked_entity"
        }
    ],
    "masked_email": "Masked email body",
    "category_of_the_email": "Predicted email category"
}
```

## Example Usage

Example request to classify an email:

```json
{
  "email_body": "Subject: i am rahul my phone number is +91-9999034566 and adhar number 55554444 5555 3333"
}
```

The response will contain masked entities like phone number and Aadhar number, and the predicted category of the email.

## File Structure

```plaintext
project/
│
├── api.py                  # FastAPI application containing routes and logic
├── classifier.pkl          # Pre-trained email classifier model
├── vectorizer.pkl          # Pre-trained vectorizer for email text transformation
├── label_encoder.pkl       # Pre-trained label encoder
├── requirements.txt        # Python dependencies for the project
├── README.md               # Project documentation
└── Dockerfile              # (Optional) Dockerfile for containerized deployment (if needed)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
