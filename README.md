Email Classification and PII Masking API
This project provides an API that classifies emails and masks Personally Identifiable Information (PII) entities in the email text. It uses machine learning to classify emails into various categories and performs PII masking using predefined regular expressions and a multilingual SpaCy model.

File Structure
graphql
Copy
Edit
.
├── app.py            # FastAPI application to serve the model
├── api.py            # API handler for predictions
├── models.py         # Contains model training and utility functions
├── utils.py          # Utility functions for processing and masking PII
├── requirements.txt  # Required Python libraries
└── README.md         # Project documentation
Setup and Requirements
Clone the repository or download the necessary files.

Install the required dependencies by running the following command:

bash
Copy
Edit
pip install -r requirements.txt
Hugging Face Deployment
You can use the API hosted on Hugging Face. It accepts POST requests with the email body and returns a response with classified information and masked PII entities.

Hugging Face URL
The deployed API is available at:

https://gayathrisoorya-email-classification-new.hf.space/predict

Example Usage with Postman
Make a POST request to the URL:

nginx
Copy
Edit
POST https://gayathrisoorya-email-classification-new.hf.space/predict
Set the request body with JSON content like below:

json
Copy
Edit
{
    "email_body": "Subject: Inquiry About Data Analytics Services for Investment Portfolio Optimization\n\nHello Customer Support, I hope this message finds you well. I am writing to seek detailed information about your company's data analytics services aimed at enhancing the optimization of investment portfolios. You can reach me at elena.ivanova@support.org. I am keen to understand how your services can aid in making prudent investment decisions and in boosting potential returns. Could you provide more details about the analytics services you offer, including portfolio optimization, risk management, and performance measurement? I would additionally appreciate any examples or case studies illustrating how these services have been beneficial to other clients. Furthermore, could you inform me of the specific data and information you need to initiate our service cooperation? I am eagerly looking forward to your response and to learning more about how your company can optimize my investment portfolio. My name is Omar Hassan. Thanks for your support and time."
}
Response will include:

The original email text with PII masked.

A list of the detected PII entities (e.g., phone number, email, full name).

The category (e.g., Request).

Example Response
json
Copy
Edit
{
  "input_email_body": "Subject: Inquiry About Data Analytics Services for Investment Portfolio Optimization\n\nHello Customer Support, I hope this message finds you well. I am writing to seek detailed information about your company's data analytics services aimed at enhancing the optimization of investment portfolios. You can reach me at elena.ivanova@support.org. I am keen to understand how your services can aid in making prudent investment decisions and in boosting potential returns. Could you provide more details about the analytics services you offer, including portfolio optimization, risk management, and performance measurement? I would additionally appreciate any examples or case studies illustrating how these services have been beneficial to other clients. Furthermore, could you inform me of the specific data and information you need to initiate our service cooperation? I am eagerly looking forward to your response and to learning more about how your company can optimize my investment portfolio. My name is Omar Hassan. Thanks for your support and time.",
  "list_of_masked_entities": [
    {"position": [124, 149], "classification": "email", "entity": "elena.ivanova@support.org"},
    {"position": [601, 614], "classification": "full_name", "entity": "Omar Hassan"}
  ],
  "masked_email": "Subject: Inquiry About Data Analytics Services for Investment Portfolio Optimization\n\nHello Customer Support, I hope this message finds you well. I am writing to seek detailed information about your company's data analytics services aimed at enhancing the optimization of investment portfolios. You can reach me at [email]. I am keen to understand how your services can aid in making prudent investment decisions and in boosting potential returns. Could you provide more details about the analytics services you offer, including portfolio optimization, risk management, and performance measurement? I would additionally appreciate any examples or case studies illustrating how these services have been beneficial to other clients. Furthermore, could you inform me of the specific data and information you need to initiate our service cooperation? I am eagerly looking forward to your response and to learning more about how your company can optimize my investment portfolio. My name is [full_name]. Thanks for your support and time.",
  "category_of_the_email": "Request"
}
Running the Model Locally
For local deployment or testing, you can follow these steps to download and save the model files (classifier.pkl, vectorizer.pkl, label_encoder.pkl) from the Hugging Face space:

Download the model files using the script provided in models.py. This will save the files locally and can be used for local testing or deployment.

Run the FastAPI application using uvicorn or any other ASGI server:

bash
Copy
Edit
uvicorn app:app --reload
Send POST requests as outlined above.

File Descriptions
app.py
This is the main FastAPI application file that serves the email classification and PII masking API.

api.py
Handles the logic for predicting email classifications and masking PII.

models.py
Contains functions for downloading and saving the trained model files (.pkl files).

utils.py
Provides utility functions for PII masking and entity detection.

requirements.txt
Lists the dependencies required to run the project.
