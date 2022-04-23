from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from tqdm import tqdm

from pytube import YouTube

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from inaSpeechSegmenter import Segmenter
from inaSpeechSegmenter.export_funcs import seg2csv, seg2textgrid

bechdel_df = pd.read_json("bechdel_data.json")
imdb_df = pd.read_json("tmdb_data.json")
imdb_df["imdb_id"] = imdb_df["imdb_id"].apply(lambda x: x[2:])
merged_df = pd.merge(bechdel_df, imdb_df, left_on="imdbid", right_on="imdb_id")


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


def run():

    for i in tqdm(range(0, len(merged_df) - 1, 1)):

        line = merged_df.iloc[i : i + 1, :]

        video_url = get_url(line["title_x"])
        selected_video = YouTube(video_url)
        selected_video.streams.filter(
            only_audio=True, file_extension="mp4"
        ).first().download(f"{line['title_x']}")

        segmentation = Segmenter()(f"{line['title_x']}/*.mp4")
        seg2csv(segmentation, f"{line['title_x']}.csv")
        df = pd.read_csv(f"{line['title_x']}.csv", sep="\t")
        df["delay"] = df.stop - df.start
        df1 = df.groupby(by="labels").sum()
        applied_df = line.apply(
            lambda row: (
                row["imdbid"],
                df1.loc["female", "delay"] / df1.loc["male", "delay"],
            )
        )
        applied_df.columns = ["imbdid", "ratio"]

        with open("PosterAnalysis.csv", "a") as csv_file:
            applied_df.to_csv(csv_file, header=False, index=False)


if __name__ == "__main__":
    run()
