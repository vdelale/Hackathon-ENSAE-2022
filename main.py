from computer_vision.main import PosterAnalysis


def preprocessing(df):
    merged_df = PosterAnalysis(df)

    return df


# require : imdb_id
def predict(df):
    df = preprocessing(df)
