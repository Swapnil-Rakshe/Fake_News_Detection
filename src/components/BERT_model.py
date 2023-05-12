import os
import pandas as pd
import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from utils import save_object

# Load BERT model and tokenizer via HuggingFace Transformers
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "BERT_model_trainer.pkl")
    

class BERTModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def tokenizer(self, x_train, y_train, x_val, y_val):
        # Majority of titles above have word length under 15. So, we set max title length as 15
        MAX_LENGHT = 42
        # Tokenize and encode sequences in the train set
        tokens_train = tokenizer(
            x_train.tolist(),
            max_length = MAX_LENGHT,
            padding=True,
            truncation=True
        )
        # tokenize and encode sequences in the validation set
        tokens_val = tokenizer(
            x_val.tolist(),
            max_length = MAX_LENGHT,
            padding=True,
            truncation=True
        )  
            
        # Convert lists to tensors
        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(y_train.tolist())

        val_seq = torch.tensor(tokens_val['input_ids'])
        val_mask = torch.tensor(tokens_val['attention_mask'])
        val_y = torch.tensor(y_val.tolist())

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
        
        for param in bert.parameters():
            param.requires_grad = False 
        try:
            model = nn.Sequential(
                bert,
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 2),
                nn.LogSoftmax(dim=1)
            )
            optimizer = torch.optim.AdamW(model.parameters(),lr = 1e-5)          # learning rate
            # Define the loss function
            cross_entropy  = nn.NLLLoss() 
            # Number of training epochs
            epochs = 2
            total_loss, total_accuracy = 0, 0
            for epoch in range(epochs):
                print(epoch)
                total_loss = 0
                model.train()
                for step,batch in enumerate(train_dataloader):                # iterate over batches
                    if step % 50 == 0 and not step == 0:                        # progress update after every 50 batches.
                        print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
                batch = [r for r in batch]                                  # push the batch to gpu
                sent_id, mask, labels = batch 
                model.zero_grad()                                           # clear previously calculated gradients
                preds = model(sent_id, mask)                                # get model predictions for current batch
                loss = cross_entropy(preds, labels)                         # compute loss between actual & predicted values
                total_loss = total_loss + loss.item()                       # add on to the total loss
                loss.backward()                                             # backward pass to calculate the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # clip gradients to 1.0. It helps in preventing exploding gradient problem
                optimizer.step()                                            # update parameters
                preds=preds.detach().cpu().numpy()                          # model predictions are stored on GPU. So, push it to CPU

                avg_loss = total_loss / len(train_dataloader)                 # compute training loss of the epoch  
                                                                          # reshape predictions in form of (# samples, # classes)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            return avg_loss  
        except:
            print("Error in training the model")

if __name__ == "__main__":
    train = pd.read_csv("artifacts/train.csv")
    x_train = train['title']
    y_train = train['true']
    val = pd.read_csv("artifacts/validation.csv")
    x_val = val['title']
    y_val = val['true']
    
    BERT = BERTModelTrainer()
    train_dataloader, validation_dataloader = BERT.tokenizer(x_train,y_train, x_val, y_val)     
    loss = BERT.initiate_model_trainer(train_dataloader)
    print(f'\nTraining Loss: {loss}')
