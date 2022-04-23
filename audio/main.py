from fileinput import filename
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from tqdm import tqdm
import os

from pytube import YouTube

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid


def get_url(search):
    driver = webdriver.Chrome()
    base_url = "https://www.youtube.com/results?search_query=bande+annonce+vf"
    keywords = search.split()
    url = "+".join((base_url, *keywords))
    driver.get(url)
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "video-title"))
    )
    return element.get_attribute("href")


def calculate_ratio(row, i):
    title = row["title_x"]
    video_url = get_url(title)
    selected_video = YouTube(video_url)
    selected_video.streams.filter(
        only_audio=True, file_extension="mp4"
    ).first().download(filename=f"{str(i)}.mp4")
    segmentation = Segmenter()(f"{i}.mp4")
    df = pd.DataFrame.from_records(segmentation, columns=["labels", "start", "stop"])
    df["delay"] = df.stop - df.start
    df1 = df.groupby(by="labels").sum()
    os.remove(f"{i}.mp4")
    if "female" not in df1.index.unique():
        return row.loc["imdb_id"], 0
    else:
        return row.loc["imdb_id"], df1.loc["female", "delay"] / df1.loc["male", "delay"]


def run():
    bechdel_df = pd.read_json("bechdel_data.json")
    imdb_df = pd.read_json("tmdb_data.json")
    imdb_df["imdb_id"] = imdb_df["imdb_id"].apply(lambda x: x[2:])
    merged_df = pd.merge(bechdel_df, imdb_df, left_on="imdbid", right_on="imdb_id")
    i = 0
    applied_df = merged_df.apply(lambda row: calculate_ratio(row, i + 1), axis=1)

    applied_df.columns = ["imdb_id", "ratio"]


if __name__ == "__main__":
    run()
