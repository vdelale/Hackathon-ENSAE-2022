from imdb import Cinemagoer, IMDbDataAccessError
import gender_guesser.detector as gg
from tqdm import TqdmExperimentalWarning as TEW
import pandas as pd
import warnings

warnings.filterwarnings(action="ignore", category=TEW)
from tqdm.autonotebook import tqdm

gender_to_score = {
    "male": 1,
    "mostly_male": 0.5,
    "andy": 0,
    "unknown": 0,
    "mostly_female": -0.5,
    "female": -1,
}


def add_infos(df):
    """
    It takes a dataframe, and adds the directors, writers, and cast to the dataframe

    :param df: the dataframe you want to add the information to
    :param start_index: the index of the first movie to add info to, defaults to 0 (optional)
    :param end_index: the index of the last movie you want to add info to, defaults to 2000 (optional)
    :return: A dataframe with the directors, writers, and cast added.
    """
    data = df.copy()
    data.index = range(0, len(data))
    data["directors"] = pd.NA
    data["writers"] = pd.NA
    data["cast"] = pd.NA

    movie_fetcher = Cinemagoer()
    index = 0
    movie_ids = data["imdb_id"].to_list()

    for imdb_id in tqdm(movie_ids):
        imdb_id = imdb_id[2:] 
        try:
            movie = movie_fetcher.get_movie_full_credits(imdb_id)
        except IMDbDataAccessError:
            movie = {"data": {}}
        try:
            directors = movie["data"]["director"]
            directors = [director["name"] for director in directors]
        except KeyError:
            directors = []
        try:
            cast = movie["data"]["cast"]
            cast = [actor["name"] for actor in cast]
        except KeyError:
            cast = []
        try:
            writers = movie["data"]["writer"]
            writers = [writer["name"] for writer in writers if len(writer) > 0]
        except:
            writers = []
        data.loc[index, "directors"] = "\n".join(directors)
        data.loc[index, "writers"] = "\n".join(writers)
        data.loc[index, "cast"] = "\n".join(cast)
        index += 1

    return data


def gender_score(string, detector):
    """
    It takes a string of names, splits it into a list of first names, and then uses the gender detector
    to get the gender of each name. It then converts the gender to a score, and returns a list of scores

    :param string: a string of names, separated by newlines
    :param detector: the gender detector object
    :return: A list of scores for each name in the list.
    """
    list_first_names = string.split("\n")
    list_first_names = [name.split(" ")[0] for name in list_first_names]
    list_genders = [detector.get_gender(firstname) for firstname in list_first_names]
    return [gender_to_score[gender] for gender in list_genders]


def genderAnalysis(df):
    """
    It takes a dataframe, adds the gender scores for each of the three columns, and returns a dataframe
    with the gender scores for each of the three columns

    :param df: the dataframe you want to analyze
    :return: A dataframe with the gender scores for directors, writers, and cast.
    """
    detector = gg.Detector()
    data = add_infos(df)
    df = data.copy()

    for col in ("directors", "writers", "cast"):
        df[col] = data[col].apply(lambda x: gender_score(x, detector=detector))
        df[col + "_male"] = df[col].apply(lambda x: sum(el for el in x if el > 0))
        df[col + "_female"] = df[col].apply(lambda x: sum(-el for el in x if el < 0))
        df[col] = df[col].apply(sum)
    return df
