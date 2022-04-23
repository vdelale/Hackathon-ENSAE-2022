from computer_vision import PosterAnalysis
import pandas as pd

df = pd.DataFrame(
    data=["0047034", "7816420", "10293406", "8356942", "2592084"], columns=["imdb_id"]
)
print(df)


def preprocessing(df):
    merged_df = PosterAnalysis(df)

    return df


# require : imdb_id
def predict(df):
    df = preprocessing(df)


if __name__ == "__main__":
    predict(df)
