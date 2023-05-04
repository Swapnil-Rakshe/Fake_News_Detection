import os
import pandas as pd
from dataclasses import dataclass
from nltk.corpus import stopwords
import nltk
import re
import os
import sys
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
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

            return corpus
        except:
            print("Error in getting data transformer object")
        
    def initiate_data_transformation(self, train_path, test_path, validation_path):
        train_df = pd.read_csv(train_path)
        validation_df = pd.read_csv(validation_path)
        test_df = pd.read_csv(test_path)
        
        preprocessing_obj=self.get_data_transformer_object(train_df['title'])
        
        input_feature_train_df = train_df.drop(columns="true", axis=1)
        target_feature_train_df = train_df["true"]
        
        input_feature_validation_df = validation_df.drop(columns="true", axis=1)
        target_feature_validation_df = validation_df["true"]

        input_feature_test_df = test_df.drop(columns="true", axis=1)
        target_feature_test_df = test_df["true"]
        
        input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_validation_arr=preprocessing_obj.transform(input_feature_validation_df)
        input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
    
        save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )
        
        return (
        input_feature_train_arr,
        input_feature_validation_arr,
        input_feature_test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
        )
        
        
        
        
        
        
        

        
        
        
    