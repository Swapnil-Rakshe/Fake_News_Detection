import os
import numpy as np
import pandas as pd
from utils import load_object
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","BiLSTM_model_trainer.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.get_data_transformer_object(features)
            print("Data Transformed")
            preds=(model.predict(data_scaled) > 0.5).astype("int32")
            return preds
        
        except:
            print("Error in prediction")

if __name__ == "__main__":
    predict_pipeline=PredictPipeline()
    test_data = pd.read_csv("artifacts/test.csv")
    input_feature_test_df = test_data['title']
    y_pred = predict_pipeline.predict(input_feature_test_df)
    y_test = test_data['true']
    y_test = np.array(y_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    print(classification_report(y_test, y_pred))
    