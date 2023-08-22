import os  # Import the os module for interacting with the operating system
import pandas as pd  # Import the pandas library for data manipulation
import numpy as np  # Import the numpy library for numerical operations
import torch  # Import the PyTorch library for deep learning
from transformers import BertTokenizer, BertModel, BertForSequenceClassification  # Import components from Transformers
import torch.nn as nn  # Import the PyTorch module for neural networks
from dataclasses import dataclass  # Import the dataclass decorator for creating structured classes
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler  # Import data-related classes
from transformers import AdamW  # Import the AdamW optimizer from Transformers
from utils import save_object  # Import a custom utility function
from tqdm import trange  # Import trange for progress tracking

# Load BERT model and tokenizer via HuggingFace Transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased")

# Define a data class for model trainer configuration
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "BERT_model_trainer.pkl")
    
# Define a class named 'BERTModelTrainer' for training a BERT-based model
class BERTModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def tokenizer(self, x_train, y_train, x_val, y_val):
        MAX_LENGTH = 42 # Maximum sequence length for tokenization
        # Tokenize and encode sequences in the train set
        tokens_train = tokenizer.batch_encode_plus(
            x_train,
            max_length = MAX_LENGTH,
            padding=True,
            truncation=True
        )
        # tokenize and encode sequences in the validation set
        tokens_val = tokenizer.batch_encode_plus(
            x_val,
            max_length = MAX_LENGTH,
            padding=True,
            truncation=True
        )  
        
        print(len(tokens_train['input_ids']))
        print(len(tokens_train['attention_mask']))
        print(len(y_train))
        print(len(tokens_val['input_ids']))
        print(len(tokens_val['attention_mask']))
        print(len(y_val))
        # Convert lists to tensors
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(y_train)

        val_seq = torch.tensor(tokens_val['input_ids'])
        val_mask = torch.tensor(tokens_val['attention_mask'])
        val_y = torch.tensor(y_val)

        batch_size = 32                                               #define a batch size

        train_data = TensorDataset(train_seq, train_mask, train_y)    # wrap tensors
        train_sampler = RandomSampler(train_data)                     # sampler for sampling the data during training
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                                                                      # dataLoader for train set
        val_data = TensorDataset(val_seq, val_mask, val_y)            # wrap tensors
        val_sampler = SequentialSampler(val_data)                     # sampler for sampling the data during training
        val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
                                                                      # dataLoader for validation set
        
        return train_dataloader, val_dataloader
 
    def initiate_model_trainer(self, train_dataloader):
        
        try:
            # Load the BERT model for sequence classification
            model = BertForSequenceClassification.from_pretrained(
                bert,
                num_labels = 2,  # Number of output labels
                output_attentions = False,
                output_hidden_states = False,
            )
            optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-5, eps = 1e-08)        # learning rate
            # Define the loss function
            # Number of training epochs
            epochs = 2
            for _ in trange(epochs, desc = 'Epoch'):
    
                # ========== Training ==========

                # Set model to training mode
                model.train()

                # Tracking variables
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                for step, batch in enumerate(train_dataloader):
                    batch = tuple(t for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch
                    optimizer.zero_grad()
                    # Forward pass
                    train_output = model(b_input_ids, 
                                         token_type_ids = None, 
                                         attention_mask = b_input_mask, 
                                         labels = b_labels)
                    # Backward pass
                    train_output.loss.backward()
                    optimizer.step()
                    # Update tracking variables
                    tr_loss += train_output.loss.item()
                    nb_tr_examples += b_input_ids.size(0)
                    nb_tr_steps += 1

                    
                print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
                

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
        except Exception as e:
            print("Error in training the model:", str(e))

if __name__ == "__main__":
    # Load training and validation data from CSV files
    train = pd.read_csv("artifacts/train.csv")
    x_train = train['title'].values.tolist()
    y_train = train['true'].values.tolist()
    val = pd.read_csv("artifacts/validation.csv")
    x_val = val['title'].values.tolist()
    y_val = val['true'].values.tolist()
    print(len(x_train), len(y_train), len(x_val), len(y_val))
    
    BERT = BERTModelTrainer()  # Create an instance of the BERTModelTrainer class
    train_dataloader, validation_dataloader = BERT.tokenizer(x_train,y_train, x_val, y_val)     
    BERT.initiate_model_trainer(train_dataloader) # Initiate the training process

