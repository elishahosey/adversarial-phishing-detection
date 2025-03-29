import os
import torch
import sys
import time
import torch.multiprocessing as mp
from multiprocessing import freeze_support
from model import modelBertLoaded, tokenizerLoaded, modelBert,tokenizer,tokenize_function,final_training, load_model, load_and_preprocess_data

#TODO Figure out SOC of streamlit-run model, when file is not here 
def run_model():
    print("Starting model training...")
    # modelB, tokenizer = load_model()
    # train_dataset, test_dataset = load_and_preprocess_data()
    # tokenized_train = train_dataset.map(tokenize_function, batched=True)
    # tokenized_test = test_dataset.map(tokenize_function, batched=True)
    # final_training(modelB, tokenizer, tokenized_train, tokenized_test)
    print("Model training finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    freeze_support()  # Required for Windows
    print(f"Using multiprocessing start method: {mp.get_start_method()}")
    time.sleep(5)
    sys.stdout.flush() 
    run_model()