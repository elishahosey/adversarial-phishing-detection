import streamlit as st
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import torch.multiprocessing as mp
from multiprocessing import freeze_support
import subprocess
import timm
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
#from model import modelBertLoaded, tokenizerLoaded

#Logging in to HuggingFace Hub
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

modelBertLoaded = AutoModelForSequenceClassification.from_pretrained('hosephoo/capstone3680')
tokenizerLoaded = AutoTokenizer.from_pretrained('hosephoo/capstone3680')

def predictWithBert(token):
    with torch.no_grad():
                outputs = modelBertLoaded(**token)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1)
                
    print(f"Predicted class: {prediction.item()}")
    return prediction.item()

def outputResult(p):
    result = "phishing" if p == 1 else "legit"
    st.write(f"The email is predicted to be {result}")
      

def main():
    st.write("Welcome to the Phishing Email Classifier ")

    # Get subject and email content
    input_subject = st.text_input("Subject line")
    input_email = st.text_area("Email")

    combined_input = input_subject + input_email
    print(combined_input)
    tokenized_input = tokenizerLoaded(combined_input, padding="max_length", return_tensors="pt", truncation=True)
    print(tokenized_input)

    # Predict button
    predict = st.button("PREDICT")

#TODO possibly include in the future
    # Check if the model is ready for prediction
    # if not os.path.exists(model_file) and not os.path.exists(token_file):
    #     st.write("Model is not ready. Please train the model first.")
    #     return

    if predict:
        predicted = predictWithBert(tokenized_input)
        outputResult(predicted)


if __name__ == "__main__":
    main()