# adversarial-phishing-detection

# dependencies
textattack, opendatasets,pandas, transformers, tokenizers,datasets,wandb,google-generativeai,optuna,Keras,torch

Dataset used for model:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

General techniques of AI attacks on emails
'''Modify emails with AI attack techniques
Goal: Confuse the AI model by subtly changing the input
  + Token modification ->
    -add in discrete changes, looks fine to humans but will confuse the model
    -suble character swapping
    -split the words a bit
    - replace some words with synonyms
  + Word-Level Rewriting->
    -use synonyms
    -change sentence structures
    -paraphrase
  +Add extra noise to the emails
  +Change tone of email -> change prompts occasionally (adjust prompts basically)
'''
 