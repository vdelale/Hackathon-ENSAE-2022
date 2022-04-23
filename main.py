from computer_vision.main import poster_analysis
from gender.gender_feature import *
import pandas as pd
import numpy as np
import pickle
from analysis.NLP_PCA import add_NLP_cols
from sklearn.preprocessing import StandardScaler

df = pd.read_json("data/tmdb_data.json")
df = df.head(31)
with open("xgb_reg.pkl","rb") as f:
    clf = pickle.load(f)
    print(clf)


N_PCA_NLP = 30  # Number of vectors in the PCA
YEARLY_INFLATION = 1.036
genres_global = []

def preprocessing(df):
    global genres_global
    merged_df = poster_analysis(df)
    merged_df = genderAnalysis(merged_df)
    print(merged_df.columns)
    merged_df = add_NLP_cols(merged_df, N_PCA_NLP)
    
    def parse_genres(x):
        global genres_global
        if x is np.nan:
            return []
        genres = []
        for d in x:
            genres.append(d["name"])
            if d["name"] not in genres_global:
                genres_global.append(d["name"])
        return genres
    
    merged_df["Genres"] = merged_df["genres"].apply(parse_genres)

    for genre in genres_global:
        merged_df[f"Is_" + genre] = merged_df["Genres"].apply(lambda x: genre in x)

    merged_df["release_month"] = pd.to_datetime(merged_df["release_date"]).dt.month
    merged_df["collection"] = merged_df["belongs_to_collection"] is None
    merged_df["revenue_is_available"] = merged_df["revenue"] != 0
    merged_df["budget is available"] = merged_df["budget"] != 0
    merged_df["year"] = pd.to_datetime(merged_df["release_date"]).dt.year.astype(int)
    merged_df["budget"] = merged_df["budget"] * (YEARLY_INFLATION ** (2022 - merged_df["year"]))
    merged_df["revenue"] = merged_df["revenue"] * (YEARLY_INFLATION ** (2022 - merged_df["year"]))
    
    columns_to_remove = [
        "title",
        "adult",
        "imdb_id",
        "overview",
        "backdrop_path",
        "genres",
        "Genres",
        "belongs_to_collection",
        "homepage",
        "original_language",
        "original_title",
        "poster_path",
        "status",
        "video",
        "spoken_languages",
        "tagline",
        "release_date",
        "directors",
        "writers",
        "cast",
        "id"
    ]

    columns_to_maybe_add_back = ["production_companies", "production_countries"]

    merged_df = merged_df.drop(columns=columns_to_remove + columns_to_maybe_add_back)

    columns_to_scale = [
        "year",
        "budget",
        "popularity",
        "revenue",
        "runtime",
        "vote_average",
        "vote_count",
        "release_month",
        "directors_male",
        "directors_female",
        "writers_male",
        "writers_female",
        "cast_male",
        "cast_female",
        "nb_women",
        "nb_men",
        "area_women",
        "area_men",
    ]
    scaler = StandardScaler()
    merged_df[columns_to_scale] = scaler.fit_transform(merged_df[columns_to_scale])
    return merged_df
    


# require : imdb_id
def predict_bechdel(df):
    df = preprocessing(df)
    print(df.columns)
    print(df.isna().any())
    print(df.dtypes)
    return clf.predict(df)
    

if __name__ == "__main__":
    predict_bechdel(df)
