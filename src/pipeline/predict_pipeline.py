import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class predictpipeline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except CustomException as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__( self,
        gender:str,
        race_ethnicity:int,
        parental_level_of_education,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int,):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_socre = reading_score

        self.writing_socre = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_score":[self.test_preparation_course],
                "reading_socre":[self.reading_socre],
                "writing_socre":[self.writing_socre],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
