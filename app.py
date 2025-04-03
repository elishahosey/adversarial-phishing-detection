import streamlit as st
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
import torch.multiprocessing as mp
from multiprocessing import freeze_support
from huggingface_hub import hf_hub_download
import subprocess
import google.generativeai as genai
import timm
import time
from google.generativeai.types.generation_types import GenerateContentResponse
import shutil
from textattack.augmentation import EasyDataAugmenter
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
#from model import modelBertLoaded, tokenizerLoaded

#Logging in to HuggingFace Hub
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_AUTH_CRED")

login(HF_TOKEN)


filename="text_combined_Df_20250403_2635.csv"
csv_path = hf_hub_download(repo_id='hosephoo/capstone3860v2', filename=filename)
modelBertLoaded = AutoModelForSequenceClassification.from_pretrained('hosephoo/capstone3860v2')
tokenizerLoaded = AutoTokenizer.from_pretrained('hosephoo/capstone3860v2')


# os.makedirs("data", exist_ok=True)
# shutil.copy(csv_path, "./" + filename)

# print("Saved to:", os.path.abspath("data/" + filename))

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
        Analyze the following email and explain confidently why it is legitimate and safe. 
Focus only on positive indicators of legitimacy such as clear sender intent, appropriate tone, and normal language patterns.

Do not mention uncertainty, phishing possibilities, or provide disclaimers.

Respond with a short, direct justification confirming why this email appears trustworthy.

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


def swap_user_text(text):
    augmenter = EasyDataAugmenter()
    augmented = augmenter.augment(text)
    return augmented[0] if isinstance(augmented, list) else augmented


def main():
    st.write("Welcome to the Phishing Email Classifier ")

    # Get subject and email content
    input_subject = st.text_input("Subject line")
    input_email = st.text_area("Email")

    combined_input = input_subject + input_email
    tokenized_input = tokenizerLoaded(combined_input, padding="max_length", return_tensors="pt", truncation=True)
 

    # Predict button
    predict = st.button("PREDICT")

    testRobust=st.button("Test Robustness (Swap Attack)")


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

    
    if testRobust:
         combined_testinput = input_subject +" "+ input_email
         if combined_testinput.strip():
            attacked_text = swap_user_text(combined_testinput)

            st.subheader("Original Input")
            st.write(combined_testinput)

            st.subheader("Adversarial Input (Swap Attack)")
            st.write(attacked_text)

            tokenized_testinput = tokenizerLoaded(combined_testinput, padding="max_length", return_tensors="pt", truncation=True)
            tokenized_attackinput = tokenizerLoaded(attacked_text, padding="max_length", return_tensors="pt", truncation=True)

            predictOriginal = predictWithBert(tokenized_testinput)
            predictAttack= predictWithBert(tokenized_attackinput)
            st.write("Prediction (before):")
            st.write( outputResult(predictOriginal))
            st.write("Prediction (before):")
            st.write(outputResult(predictAttack))
         


if __name__ == "__main__":
    main()