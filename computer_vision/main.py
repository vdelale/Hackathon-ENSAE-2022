from PosterAnalysis import analyse_img_from_url
import pandas as pd
from deepface import DeepFace
from deepface.detectors import FaceDetector
import yaml
from tqdm import tqdm

tqdm.pandas()

with open("config.yml") as yml_file:
    config = yaml.safe_load(yml_file)

bechdel_df = pd.read_json("bechdel_data.json")
imdb_df = pd.read_json("tmdb_data.json")
imdb_df["imdb_id"] = imdb_df["imdb_id"].apply(lambda x: x[2:])
merged_df = pd.merge(bechdel_df, imdb_df, left_on="imdbid", right_on="imdb_id")

gender_detector = DeepFace.build_model("Gender")
face_detector = FaceDetector.build_model(config["detector_backend"])
age_detector = DeepFace.build_model("Age")

for i in tqdm(range(0, len(merged_df) - 10, 10)):

    line = merged_df.iloc[i : i + 10, :]
    applied_df = line.apply(
        lambda row: (
            row["imdbid"],
            *analyse_img_from_url(
                row["poster_path"],
                gender_detector=gender_detector,
                face_detector=face_detector,
                age_detector=age_detector,
            ),
        ),
        axis="columns",
        result_type="expand",
    )
    applied_df.columns = ["imbdid", "nb_women", "nb_men", "area_women", "area_men"]

    with open("PosterAnalysis.csv", "a") as csv_file:
        applied_df.to_csv(csv_file, header=False, index=False)

