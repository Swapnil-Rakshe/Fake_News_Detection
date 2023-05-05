import pandas as pd
import os

from utils import save_object
from data_transformation import DataTransformation
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from utils import load_object


class ModelTrainer:
    def __init__(self):
        pass
        
    def initiate_model_trainer(self, x_train, y_train, x_val, y_val):
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
            
            return model()
            
        except:
            print("Error in training the model")
            
if __name__ == "__main__":
    model = ModelTrainer()
    y_1 = pd.read_csv("artifacts/train.csv")
    x_train = y_1['title']
    y_train = y_1['true']
    preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
    preprocessor=load_object(file_path=preprocessor_path)
    x_train=preprocessor.get_data_transformer_object(x_train)
    y_2 = pd.read_csv("artifacts/validation.csv")
    x_val = y_2['title']
    y_val = y_2['true']
    x_val=preprocessor.get_data_transformer_object(x_val)
    model_trained = model.initiate_model_trainer(x_train, y_train, x_val, y_val)
    
    
    
    test_data = pd.read_csv("artifacts/test.csv")
    input_feature_test_df = test_data['title']
    y_test = test_data['true']
    preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
    preprocessor=load_object(file_path=preprocessor_path)
    print("After Loading")
    x_test=preprocessor.get_data_transformer_object(input_feature_test_df)
    y_pred = model_trained.predict(x_test)
    print(y_pred)

