# Import necessary libraries and modules
import os
import numpy as np
import pandas as pd
from utils import load_object
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Class for making predictions using a trained model and preprocessor
class PredictPipeline:
    def __init__(self):
        pass
    
    # Method to make predictions
    def predict(self,features):
        try:
            # Define file paths for the trained model and preprocessor objects
            model_path=os.path.join("artifacts","BiLSTM_model_trainer.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
              # Load the trained model and preprocessor
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
             # Transform input data using the preprocessor
            data_scaled=preprocessor.get_data_transformer_object(features)
            print("Data Transformed")
             # Make predictions using the model
            preds=(model.predict(data_scaled) > 0.5).astype("int32")
            return preds
        
        except:
            print("Error in prediction")
# Main execution starts here
if __name__ == "__main__":
    # Initialize the prediction pipeline
    predict_pipeline=PredictPipeline()
      # Read test data from CSV file
    test_data = pd.read_csv("artifacts/test.csv")
    input_feature_test_df = test_data['title']
    # Make predictions on test data
    y_pred = predict_pipeline.predict(input_feature_test_df)
    # Prepare ground truth label
    y_test = test_data['true']
    y_test = np.array(y_test)
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Print classification report with precision, recall, and F1-score
    print(accuracy)
    print(classification_report(y_test, y_pred))
    