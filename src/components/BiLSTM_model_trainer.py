# Import required libraries
import os 
from dataclasses import dataclass
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
# Import a custom utility function for saving objects
from utils import save_object
# Define a data class to hold configuration settings
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "BiLSTM_model_trainer.pkl")
    
# Class to train and save a Bidirectional LSTM model
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, x_train, y_train, x_val, y_val):
        # Assign input and output data
        x_train = x_train
        y_train = y_train
        x_val = x_val
        y_val = y_val
        
        try:
            ## Creating model Using LSTM
            embedding_vector_features=40
            voc_size = 13225
            model=Sequential()
            model.add(Embedding(voc_size,embedding_vector_features,input_length=42))
            model.add(Dropout(0.3))
            model.add(Bidirectional(LSTM(100)))
            model.add(Dropout(0.3))
            model.add(Dense(1,activation='sigmoid'))
            model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            
            # Training the model
            model.fit(x_train, y_train, batch_size = 64, epochs = 5, validation_data=(x_val, y_val))
            
             # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
        except:
            print("Error in training the model")