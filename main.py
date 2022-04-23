from computer_vision.main import poster_analysis
from gender.gender_feature import *
import pandas as pd

df = pd.read_json("data/tmdb_data.json")
df = df.head()


def preprocessing(df):
    merged_df = poster_analysis(df)
    print(merged_df.shape)
    merged_df = add_infos(merged_df)
    print(merged_df.shape)
    merged_df = genderAnalysis(merged_df)
    print(merged_df.shape)
    return merged_df


# require : imdb_id
def predict(df):
    df = preprocessing(df)

if __name__ == "__main__":
    predict(df)
