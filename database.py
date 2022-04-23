# %%
import pandas as pd
import numpy as np
from analysis.NLP_PCA import add_NLP_cols
from sklearn.preprocessing import StandardScaler


DATA_FOLDER = "data"
N_PCA_NLP = 10 # Number of vectors in the PCA
YEARLY_INFLATION = 1.036


df_bechdel = pd.read_json("data/bechdel_data.json")
df_tmdb = pd.read_json("data/tmdb_data.json")
df_tmdb["imdbid"] = df_tmdb["imdb_id"].str[2:]
df = pd.merge(df_bechdel, df_tmdb, "left", "imdbid")
df = df[df['overview'].notna()]

#%% Adding NLP columns based on the overviews (+ PCA)
df = add_NLP_cols(df, N_PCA_NLP)


#%% Creating dummy variables for Genres
genres_global = []

def parse_genres(x):
    global genres_global
    if x is np.nan:
        return []
    genres = []
    for d in x:
        genres.append(d['name'])
        if d['name'] not in genres_global:
            genres_global.append(d['name'])
    return genres

df["Genres"] = df["genres"].apply(parse_genres)

for genre in genres_global:
    df[f"Is_" + genre] = df["Genres"].apply(lambda x: genre in x)
    

def inflation(row):
    return row["budget"] * (YEARLY_INFLATION ** (2022-row["year"]))

#%% NLP analysis + PCA
df["release_month"] = pd.to_datetime(df["release_date"]).dt.month
df["collection"] = (df["belongs_to_collection"] is None)
df["revenue_is_available"] = (df["revenue"] != 0)
df["budget is available"] = (df["budget"] != 0)
df["budget"] = df["budget"] * (YEARLY_INFLATION ** (2022-df["year"]))
df["revenue"] = df["revenue"] * (YEARLY_INFLATION ** (2022-df["year"]))
columns_to_remove = ["title_x", 
                     "imdbid", 
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
                     "release_date"]

columns_to_maybe_add_back = ["production_companies",
                             "production_countries"]

df = df.drop(columns=columns_to_remove + columns_to_maybe_add_back)
# %%
columns_to_scale = ["year",
                    "budget",
                    "popularity",
                    "revenue",
                    "runtime",
                    "vote_average",
                    "vote_count",
                    "release_month"]
scaler = StandardScaler()
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
# %%
