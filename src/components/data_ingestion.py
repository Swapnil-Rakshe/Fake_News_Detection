import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from wordcloud import WordCloud
from dataclasses import dataclass
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os


from data_transformation import InitiateDataTransformation
from data_transformation import DataTransformationConfig

from BiLSTM_model_trainer import ModelTrainerConfig
from BiLSTM_model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    validation_data_path: str=os.path.join('artifacts',"validation.csv")
    
class DataAnalysis:
    def __init__(self):
        self.true_data = pd.read_csv("dataset/True.csv")
        self.fake_data = pd.read_csv("dataset/Fake.csv")
    
    def data_visualization(self):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
        
        sns.countplot(y="subject", palette="coolwarm", data=self.true_data, ax=ax1)
        ax1.set_title('True News Subject Distribution')
        
        sns.countplot(y="subject", palette="coolwarm", data=self.fake_data, ax=ax2)
        ax2.set_title('Fake News Subject Distribution')
        
        plt.tight_layout()
        plt.show()
        
    def create_wordcloud(self, data, title):
        all_titles = data.title.str.cat(sep=' ')
        wordcloud = WordCloud(background_color='white', width=800, height=500,
                              max_font_size=180, collocations=False).generate(all_titles)
        plt.figure(figsize=(10,7))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title, fontsize=20)
        plt.show()
    def visualize_wordclouds(self):
        self.create_wordcloud(self.true_data, 'Real News Title WordCloud')
        self.create_wordcloud(self.fake_data, 'Fake News Title WordCloud')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.true_data = pd.read_csv("dataset/True.csv")
        self.fake_data = pd.read_csv("dataset/Fake.csv")
    
    def data_combination(self):
        # Add Labels to both df
        self.true_data['true'] = 1
        self.fake_data['true'] = 0
        # Concat
        df = pd.concat([self.true_data, self.fake_data])
        # Purify
        df = df.iloc[:,[0, -1]]

        # Shuffle
        df = shuffle(df).reset_index(drop=True)
        
        # Splitting the data into training and test sets
        train_set_1,test_set=train_test_split(df,test_size=0.2,random_state=42)
        
        # Splitting the training data into training and validation sets
        train_set, validation_set=train_test_split(train_set_1,test_size=0.2,random_state=42)
        
        train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
        validation_set.to_csv(self.ingestion_config.validation_data_path,index=False,header=True)
        test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
        
        return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path

            )
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, validation_data, test_data = obj.data_combination()
    
    data_transformation = InitiateDataTransformation()
    x_train, x_val, x_test,_=data_transformation.initiate_data_transformation(train_data, validation_data, test_data)
    
    y_1 = pd.read_csv("artifacts/train.csv")
    y_train = y_1['true']
    y_2 = pd.read_csv("artifacts/validation.csv")
    y_val = y_2['true']
    
    print(len(x_train))
    print(len(y_train))
    print(len(x_val))
    print(len(y_val))
    
    modeltrainer=ModelTrainer()
    modeltrainer.initiate_model_trainer(x_train,y_train, x_val, y_val)