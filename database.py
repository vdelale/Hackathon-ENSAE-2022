# %%
import pandas as pd
import numpy as np
from analysis.NLP_PCA import add_NLP_cols


DATA_FOLDER = "data"
N_PCA_NLP = 10 # Number of vectors in the PCA


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
    
#%% NLP analysis + PCA



df["collection"] = (df["belongs_to_collection"] is None)
columns_to_remove = ["title_x", 
                     "imdbid", 
                     "id_x",
                     "adult",
                     "imdb_id",
                     
                     "backdrop_path", 
                     "genres", 
                     "Genres", 
                     "budget", 
                     "belongs_to_collection", 
                     "budget", 
                     "homepage", 
                     "id_y", 
                     "original_language", 
                     "original_title", ]

df.drop(columns=columns_to_remove)