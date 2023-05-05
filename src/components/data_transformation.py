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


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.train_df = pd.read_csv("artifacts/train.csv")
    
    def get_data_transformer_object(self, messages):
        try:
            corpus = []

            for i in range(len(messages)):
               review = re.sub('[^a-zA-Z]', ' ', messages[i])
               review = review.lower()
               review = review.split()

               review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
               review = ' '.join(review)
               corpus.append(review)

            tokenizer = Tokenizer(num_words = 13225)
            tokenizer.fit_on_texts(self.train_df['title'])
            train_sequences = tokenizer.texts_to_sequences(corpus)
            input_feature_train_arr = pad_sequences(train_sequences,maxlen = 42, padding = 'pre', truncating = 'post')
            
            return input_feature_train_arr
    
        except:
            print("Error in getting data transformer object")
            
class InitiateDataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self, train_path, validation_path, test_path):
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)
        
        preprocessing_obj=DataTransformation()
        
        input_feature_train_df = train_df['title']
        
        input_feature_validation_df = validation_df['title']

        input_feature_test_df = test_df['title']
        
        train_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_train_df)
        validation_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_validation_df)
        test_preprocessed=preprocessing_obj.get_data_transformer_object(input_feature_test_df)
    
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )
        
        return (
        train_preprocessed,
        validation_preprocessed,
        test_preprocessed,
        self.data_transformation_config.preprocessor_obj_file_path,
        )
        
        
        
        
        
        
        

        
        
        
    