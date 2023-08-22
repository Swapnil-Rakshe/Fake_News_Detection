# Import necessary libraries and modules
import os
import pandas as pd
from dataclasses import dataclass
from nltk.corpus import stopwords
import nltk
import re
import os
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from utils import save_object
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Define a data class to hold configuration settings for data transformation
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    
# Class for data transformation and preprocessing
class DataTransformation:
    def __init__(self):
        self.train_df = pd.read_csv("artifacts/train.csv")
    # Method to preprocess and transform text data
    def get_data_transformer_object(self, messages):
        try:
            corpus = []
             # Preprocess each message in the given messages
            for i in range(len(messages)):
               review = re.sub('[^a-zA-Z]', ' ', messages[i]) # Remove non-alphabetic characters
               review = review.lower()   # Convert to lowercase
               review = review.split()   # Split into words
                # Apply stemming and remove stopwords
               review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
               review = ' '.join(review)
               corpus.append(review)
             # Apply stemming and remove stopwords
            tokenizer = Tokenizer(num_words = 13225)  # Initialize a tokenizer with a maximum vocabulary size
            tokenizer.fit_on_texts(self.train_df['title'])  # Fit the tokenizer on training data
            train_sequences = tokenizer.texts_to_sequences(corpus) # Convert text to sequences
            input_feature_train_arr = pad_sequences(train_sequences,maxlen = 42, padding = 'pre', truncating = 'post')  # Pad sequences to a fixed length
            
            return input_feature_train_arr
    
        except:
            print("Error in getting data transformer object")

# Class for initiating data transformation        
class InitiateDataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
     # Method to initiate data transformation
    def initiate_data_transformation(self, train_path, validation_path, test_path):
         # Read data from CSV files
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)
        # Create an instance of the DataTransformation class
        preprocessing_obj=DataTransformation()
         # Extract input feature data from dataframes
        input_feature_train_df = train_df['title']
        
        input_feature_validation_df = validation_df['title']

        input_feature_test_df = test_df['title']
         # Preprocess and transform the input feature data
        train_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_train_df)
        validation_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_validation_df)
        test_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_test_df)
        # Save the preprocessing object
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )
         # Return the preprocessed data and the path to the saved preprocessing object
        return (
        train_preprocessed,
        validation_preprocessed,
        test_preprocessed,
        self.data_transformation_config.preprocessor_obj_file_path,
        )
        
        
        
        
        
        
        

        
        
        
    