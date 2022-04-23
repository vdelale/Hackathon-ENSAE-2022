from computer_vision.main import poster_analysis
from gender.gender_feature import *
import pandas as pd
import pickle

df = pd.read_json("data/tmdb_data.json")
df = df.head()
clf = pickle.load("xgb_reg.pkl")

def preprocessing(df):
    merged_df = poster_analysis(df)
    merged_df = add_infos(merged_df)
    merged_df = genderAnalysis(merged_df)
    return merged_df


# require : imdb_id
def predict_bechdel(df):
    df = preprocessing(df)
    

if __name__ == "__main__":
    predict_bechdel(df)
