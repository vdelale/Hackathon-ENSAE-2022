from computer_vision.main import poster_analysis
import pandas as pd

df = pd.read_json("data/tmdb_data.json")
df = df.head()


def preprocessing(df):
    merged_df = poster_analysis(df)
    return df


# require : imdb_id
def predict(df):
    df = preprocessing(df)


if __name__ == "__main__":
    predict(df)
