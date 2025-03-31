import streamlit as st
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import torch.multiprocessing as mp
from multiprocessing import freeze_support
import subprocess
import google.generativeai as genai
import timm
import time
from google.generativeai.types.generation_types import GenerateContentResponse
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
#from model import modelBertLoaded, tokenizerLoaded

#Logging in to HuggingFace Hub
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
api_key_genai = os.getenv("GOOGLE_API_KEY")
google_creds_path = os.getenv("GOOGLE_AUTH_CRED")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path

modelBertLoaded = AutoModelForSequenceClassification.from_pretrained('hosephoo/capstone3680')
tokenizerLoaded = AutoTokenizer.from_pretrained('hosephoo/capstone3680')

def generateReason(result,subject,email):
    st.write(f"generating result: {result}")
    if result == 'phishing':
        analysis = f"""
        Analyze the following email and explain why it is likely a phishing attempt. Identify any red flags such as urgency, fake links, impersonation, or suspicious requests. Provide a clear and concise explanation.

        **Subject:** {subject}
        **Body:** {email}
        """
    else:
         analysis = f"""
        Analyze the following email and explain why it is legitimate and safe. Provide a confident and well-reasoned explanation that confirms its authenticity and reliability.

        **Subject:** {subject}
        **Body:** {email}
        """
    genai.configure(api_key=os.environ['GOOGLE_API_KEY']) 
    modelReason = genai.GenerativeModel('gemini-1.5-pro')  
    start = time.time()
    analyzedRes = modelReason.generate_content(analysis)
  # Building out into new dataframes

    return analyzedRes
         

def parseResponse(response):
     if isinstance(response, GenerateContentResponse):
          res_str = response.candidates[0].content.parts[0].text
          return res_str
     

def predictWithBert(token):
    with torch.no_grad():
                outputs = modelBertLoaded(**token)
                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1)
                
    #print(f"Predicted class: {prediction.item()}")
    return prediction.item()

def outputResult(p):
    print("This is p",p)
    result = "phishing" if p == 1 else "legit"
    st.write(f"The email is predicted to be {result}")
    return result

      

def main():
    st.write("Welcome to the Phishing Email Classifier ")

    # Get subject and email content
    input_subject = st.text_input("Subject line")
    input_email = st.text_area("Email")

    combined_input = input_subject + input_email
    tokenized_input = tokenizerLoaded(combined_input, padding="max_length", return_tensors="pt", truncation=True)
 

    # Predict button
    predict = st.button("PREDICT")

#TODO possibly include in the future
    # Check if the model is ready for prediction
    # if not os.path.exists(model_file) and not os.path.exists(token_file):
    #     st.write("Model is not ready. Please train the model first.")
    #     return

    if predict:
        predicted = predictWithBert(tokenized_input)
        res=outputResult(predicted)
        reason=generateReason(res,input_subject,input_email)
        parsedReason = parseResponse(reason)
        st.write(f"This is a reason: {parsedReason}")



if __name__ == "__main__":
    main()