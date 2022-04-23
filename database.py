import pandas as pd
import numpy as np
from analysis.NLP_PCA import add_NLP_cols
from sklearn.preprocessing import StandardScaler

DATA_FOLDER = "data"
N_PCA_NLP = 30  # Number of vectors in the PCA
YEARLY_INFLATION = 1.036


def inflation(row):
    return row["budget"] * (YEARLY_INFLATION ** (2022 - row["year"]))


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


def main():
    df_bechdel = pd.read_json("data/bechdel_data.json")
    df_bechdel = df_bechdel[df_bechdel["imdbid"] != ""]
    df_bechdel["imdbid"] = df_bechdel["imdbid"].astype(float).astype(str)

    df_tmdb = pd.read_json("data/tmdb_data.json")
    df_tmdb["imdbid"] = df_tmdb["imdb_id"].str[2:].astype(float).astype(str)

    df_poster = pd.read_csv("computer_vision/PosterAnalysis.csv")
    df_poster["imdbid"] = df_poster["imbdid"].astype(float).astype(str)

    df_genderCount = pd.read_csv("data/directors_writers_cast_score.csv")
    df_genderCount["imdbid"] = df_genderCount["imdbid"].astype(float).astype(str)
    df_genderCount.drop(columns=["title", "year", "id", "rating"], inplace=True)

    df_temp_1 = pd.merge(df_bechdel, df_tmdb, "inner", "imdbid")
    df_temp_2 = pd.merge(df_poster, df_temp_1, "inner", "imdbid")
    df = pd.merge(df_temp_2, df_genderCount, "inner", "imdbid")

    df = df[df["overview"].notna()]

    df = add_NLP_cols(df, N_PCA_NLP)

    genres_global = []

    df["Genres"] = df["genres"].apply(parse_genres)

    for genre in genres_global:
        df[f"Is_" + genre] = df["Genres"].apply(lambda x: genre in x)

    df["release_month"] = pd.to_datetime(df["release_date"]).dt.month
    df["collection"] = df["belongs_to_collection"] is None
    df["revenue_is_available"] = df["revenue"] != 0
    df["budget is available"] = df["budget"] != 0
    df["budget"] = df["budget"] * (YEARLY_INFLATION ** (2022 - df["year"]))
    df["revenue"] = df["revenue"] * (YEARLY_INFLATION ** (2022 - df["year"]))
    columns_to_remove = [
        "title_x",
        "imdbid",
        "imbdid",
        "id_x",
        "adult",
        "imdb_id",
        "overview",
        "backdrop_path",
        "genres",
        "Genres",
        "belongs_to_collection",
        "homepage",
        "id_y",
        "original_language",
        "original_title",
        "poster_path",
        "status",
        "video",
        "spoken_languages",
        "tagline",
        "title_y",
        "release_date",
        "directors",
        "writers",
        "cast",
    ]

    columns_to_maybe_add_back = ["production_companies", "production_countries"]

    df = df.drop(columns=columns_to_remove + columns_to_maybe_add_back)

    df.to_csv("data/final_database.csv", index=False)

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
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    return df


if __name__ == "__main":
    df = main()
