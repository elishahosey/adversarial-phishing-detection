import opendatasets as od
import pandas as pd
import torch
import optuna
import os
import wandb
import pathlib
import textwrap
import time
import google.generativeai as genai
import json
import requests
from huggingface_hub import login
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from google.generativeai.types.generation_types import GenerateContentResponse
from textattack.augmentation import EasyDataAugmenter
from IPython.display import display
from IPython.display import Markdown
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
import tensorflow as tf
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
api_key_genai = os.getenv("GOOGLE_API_KEY")
google_creds_path = os.getenv("GOOGLE_AUTH_CRED")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_creds_path

#TODO improve model prediction 
'''
PROMPTS
'''

phish_prompt = """
Generate 10 unique phishing emails impersonating major companies such as Amazon, PayPal, or Microsoft. Each response should contain exactly 10 emails. Do **not** return a list. The response should be in the 'GenerateContentResponse' format.
Each email must have:
- A different pretext (e.g., security warning, payment failure, order confirmation, fake promotion).
- Variations in tone and professionalism (some should be highly convincing, while others should have subtle grammatical or stylistic errors).
- A fake but realistic-looking phishing link (e.g., "http://amazon-security-check.com").
- Different psychological manipulation tactics:
  1. Urgency: "Act now to avoid suspension!"
  2. Fear: "Suspicious activity detected on your account."
  3. Reward: "Congratulations! You've won a $100 gift card."
  4. Curiosity: "Unusual login detected. Click here for details."
  5. Authority: "Official notice from the PayPal security team."
- Ensure each email looks distinct from the others.
- Avoid using placeholders like '[insert link here]'; instead, generate actual phishing-style links that look deceptive.

Output the emails in a structured JSON format like and do not include comments:
[
  {
    "subject": "...",
    "body": "...",
    "phishing_tactic": "Urgency",
    "fake_link": "http://amazon-security-check.com"
  },
  ...
]


"""

legit_prompt = """
Generate 10 unique legitimate emails from major companies such as Amazon, PayPal, or Microsoft. Each response should contain exactly 10 emails. Do **not** return a list. The response should be in the 'GenerateContentResponse' format.

Output the emails in a structured JSON format like and do not include comments:
[
  {
    "subject": "...",
    "body": "...",
  },
  ...
]


"""

'''
END OF PROMPTS
'''



'''
Model Related Stuff
'''
#Defining Bert model and tokenizer
def load_model():
    token = AutoTokenizer.from_pretrained("bert-base-uncased")
    mB = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    return mB, token

# Objective function for Optuna
def compute_metrics(pred):
      logits, labels = pred
      predictions = torch.argmax(torch.tensor(logits), dim=-1)
      accuracy = accuracy_score(labels, predictions)
      f1 = f1_score(labels, predictions, average="weighted")
      return {"accuracy": accuracy, "f1": f1}


def final_training(mB,token_train,token_test):
  # Training arguments https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/trainer#transformers.TrainingArguments
  training_args = TrainingArguments(
      output_dir="./results",
      evaluation_strategy="steps",
      eval_steps=250,  # evaluate less frequently
      learning_rate=4e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=2,
      weight_decay=0.01,
      logging_dir="./logs",
      save_strategy="steps",
      save_steps=500,  # save frequently to prevent unnecessary retraining
      load_best_model_at_end=True,
      fp16=True,  # keep mixed precision training
      gradient_accumulation_steps=2,
      dataloader_num_workers=4,  # speed up data loading
      run_name= 'test_streamlit_app_model'
  )
  trainer = Trainer(
      model=mB,
      args=training_args,
      train_dataset=token_train,
      eval_dataset=token_test,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics
  )

  trainer.train()
  trainer.evaluate()
  #TODO save model with date time too
  modelBert.save_pretrained("./model/my_trainedBert_model")
  tokenizer.save_pretrained("./model/my_trainedBertToken_model")

'''
End of Model Stuff
'''

'''
Parsing new data sets from Gemini API
'''
def preprocess_emails(phish,legit,cand):
  phish_emails = []
  legit_emails = []
  #Verify type of response to determine parsing
  #print(f"Phish Type: {type(phish)}, Legit Type: {type(legit)}")
  if isinstance(phish, GenerateContentResponse):
    #convert each emails in proper format for df conversion
    for c in range(0,cand):
      phish_str = phish.candidates[c].content.parts[0].text
      legit_str = legit.candidates[c].content.parts[0].text

      phish_parse = phish_str.replace("```json", "").replace("```", "").strip()
      legit_parse = phish_str.replace("```json", "").replace("```", "").strip()

      phish_emails.extend(json.loads(phish_parse))
      legit_emails.extend(json.loads(legit_parse))
    return phish_emails, legit_emails
  else:
     return phish, legit

def create_email_df(phish,legit):
  phish_df = pd.DataFrame(phish)
  phish_df['label'] = 1
  legit_df = pd.DataFrame(legit)
  legit_df['label'] = 0

  #remove potential duplicate responses
  phish_df = phish_df.drop_duplicates(subset=['body'])
  legit_df = legit_df.drop_duplicates(subset=['body'])
  return phish_df, legit_df

def prepForTest(s):
  test_df = s.drop(columns=["phishing_tactic", "fake_link","label"])
  return test_df

def combineDf(phish_df,legit_df):
  combined_df = pd.concat([phish_df, legit_df], ignore_index=True)
  shuffled_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
  #createFile(shuffled_df,"shuffled_Before_Drop")
  #shuffled_df = shuffled_df.drop(columns=["phishing_tactic", "fake_link","label"])
  return shuffled_df

#<Swap Text Method Attack>#
def swapAttack(shuffled_df):
  swapAttack_df = shuffled_df.copy()
  augmenter = EasyDataAugmenter()
  swapAttack_df['body'] = swapAttack_df['body'].apply(augmenter.augment)
  swapAttack_df['body'] = swapAttack_df['body'].apply(lambda x: x[0] if isinstance(x, list) else x)
  return swapAttack_df

def createFile(df,name):
  timestamp = datetime.now().strftime("%Y%m%d_%M%S")
  filename = f"{name}_{timestamp}.csv"
  df.to_csv(filename, index=False)
  
'''
END
'''

'''
Preprocessing
'''
#convert to hugging face dataset format
def load_and_preprocess_data():
    file="./dataset/phishing_email.csv"
    df=pd.read_csv(file)
    X = df.iloc[:50, :-1].squeeze()  # text_combined
    Y = df.iloc[:50, -1]  # label

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    train_df = pd.DataFrame({"text_combined": X_train, "label": Y_train})
    test_df = pd.DataFrame({"text_combined": X_test, "label": Y_test})

    # Convert to Hugging Face dataset format
    train_huggy = Dataset.from_pandas(train_df)
    test_huggy = Dataset.from_pandas(test_df)

    return train_huggy, test_huggy
'''
END
'''


def checkGPUstatus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    torch.cuda.is_available()
    print("GPUS",gpus)
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def generateNewTestData():
  # Prompt and build new test set with generative ai
  generation_config = genai.GenerationConfig(candidate_count=8) #Batch processing at most 8 responses per request
  genai.configure(api_key=os.environ['GOOGLE_API_KEY']) 
  modelPhishing = genai.GenerativeModel('gemini-1.5-pro')


  # Generate at least 100 new test sets
  start = time.time()
  phishing_response = modelPhishing.generate_content(phish_prompt,generation_config=generation_config)
  time.sleep(10) # avoid bottleneck of requests
  modelLegit = genai.GenerativeModel('gemini-1.5-pro')
  legit_response = modelLegit.generate_content(legit_prompt,generation_config=generation_config)

  # Building out into new dataframes
  phishing_converted_response, legit_converted_response = preprocess_emails(phishing_response,legit_response,8)
  phish_df, legit_df = create_email_df(phishing_converted_response,legit_converted_response)
  # print(phish_df.shape)
  # print(legit_df.shape)
  combined_df = combineDf(phish_df,legit_df)
  test_combined_df = prepForTest(combined_df)
  swapAttack_df = swapAttack(test_combined_df)
  createFile(swapAttack_df,"swapAttack")
  createFile(combined_df,"combinedDf")
  createFile(test_combined_df,"testcombinedDf")
    

checkGPUstatus()

#TODO Export environment to yaml for others to use

def tokenize_function(email):
    return tokenizer(email["text_combined"], padding="max_length", truncation=True)

def tokenize_testfunction(email):
    return tokenizerLoaded(email["body"], padding="max_length", truncation=True)

train_huggydataset, test_huggydataset = load_and_preprocess_data() 
modelBert,tokenizer = load_model()

#Tokenize the sets
tokenized_train = train_huggydataset.map(tokenize_function, batched=True)
tokenized_test = test_huggydataset.map(tokenize_function, batched=True)

#Finally train the model
#final_training(modelBert,tokenized_train,tokenized_test)



# Load trained model and tokenizer for evaluation
modelBertLoaded = AutoModelForSequenceClassification.from_pretrained("./model/my_trainedBert_model")
tokenizerLoaded = AutoTokenizer.from_pretrained("./model/my_trainedBertToken_model")

#testAttack_huggydataset = Dataset.from_pandas(swapAttack_df)
testOriginal_huggydataset = Dataset.from_pandas(test_combined_df)

# Tokenize dataset
# tokenized_test = testAttack_huggydataset.map(
#     tokenize_testfunction,
#     batched=True,
#     remove_columns=testAttack_huggydataset.column_names
# )
tokenized_test_original = testOriginal_huggydataset.map(
    tokenize_testfunction,
    batched=True,
    remove_columns=testOriginal_huggydataset.column_names
)

# Convert tokenized data to PyTorch tensors
tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_test_original.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelBertLoaded.to(device)

#test_dataloader = DataLoader(tokenized_test, batch_size=8)
test_dataloader = DataLoader(tokenized_test_original, batch_size=8)

modelBertLoaded.eval()  # Set to evaluation mode
all_predictions = []

with torch.no_grad():  # Disable gradient computation for inference
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move to device
        outputs = modelBertLoaded(**batch)  # Forward pass
        logits = outputs.logits  # Extract logits
        predictions = torch.argmax(logits, dim=-1)  # Get predicted class
        all_predictions.extend(predictions.cpu().numpy())  # Move to CPU for printing


swapAttack_df['predictions'] = all_predictions
createFile(swapAttack_df,"swapAttack_predictions")

Test with original data before attack
test_combined_df['predictions'] = all_predictions
createFile(test_combined_df,"combined_predictions")


'''
Check Accuracy
'''
y_true = combined_df['label']
y_pred = test_combined_df['predictions']

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
'''
END
'''
login() #login to external model on huggingface hub
generateNewTestData()

if __name__ == '__main__':
    # Put the code that starts your training here
    final_training(modelBert, tokenized_train, tokenized_test)