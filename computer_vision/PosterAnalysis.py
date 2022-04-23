from operator import ge
from threading import Thread
import numpy as np
import requests
import yaml
from deepface import DeepFace
from deepface.detectors import FaceDetector
from PIL import Image

with open("computer_vision/config.yml") as yml_file:
    config = yaml.safe_load(yml_file)


def analyse_one_face_task(face: np.ndarray, outputs, gender_detector, age_detector):
    """
    It takes a face image and returns a list of 4 values:

    - The dominant emotion
    - The gender
    - The dominant race
    - The face area in the image

    The face area is calculated by dividing the area of the face in the image by the area of the image

    :param face: the face to be analysed
    :type face: np.ndarray
    :param outputs: a list to store the results of each face
    """
    output = DeepFace.analyze(
        face,
        enforce_detection=False,
        detector_backend=config["detector_backend"],
        actions=("gender", "age"),
        models={"age": age_detector, "gender": gender_detector},
        prog_bar=False,
    )
    if len(output) > 0:
        one_face_analyse = [
            output["gender"],
            float(output["region"]["h"]) * float(output["region"]["w"]),
        ]
    else:
        one_face_analyse = [0, 0]
    outputs.append(one_face_analyse)


def get_image(poster_path: str) -> np.ndarray:
    """
    It takes a poster path, and returns the image as a numpy array

    :param poster_path: The path to the poster image
    :type poster_path: str
    :return: A numpy array of the image
    """
    url = config["base_url_for_poster"] + poster_path
    return np.asarray(Image.open(requests.get(url, stream=True).raw))


def analyse_img(img: np.ndarray, gender_detector, face_detector, age_detector) -> list:
    """
    It takes an image as input and returns a list of the image's properties.

    :param img: The image to be analysed
    :type img: np.ndarray
    """

    faces = FaceDetector.detect_faces(face_detector, config["detector_backend"], img)
    outputs = []
    threads = []
    for face in faces:
        threads.append(
            Thread(
                target=analyse_one_face_task,
                args=(face[0], outputs, gender_detector, age_detector),
            )
        )
        threads[-1].start()
    for thread in threads:
        thread.join()

    return outputs


def clean_outputs(
    face_analysis_results, w_img: int, l_img: int
) -> tuple[int, int, float, float]:
    """
    It takes a list of tuples, and returns a tuple of four integers

    :param face_analysis_results: list of tuples, each tuple contains the following:
    :return: The number of women, the number of men, the area of women and the area of men.
    """
    nb_women, nb_men, area_men, area_women = 0, 0, 0, 0
    for result in face_analysis_results:
        if result[-1] / (w_img * l_img) < 1.0:
            if result[0] == "Woman":
                nb_women += 1
                area_women += result[-1]
            else:
                nb_men += 1
                area_men += result[-1]

    return (
        nb_women,
        nb_men,
        area_women / float((w_img * l_img)),
        area_men / float((w_img * l_img)),
    )


def analyse_img_from_url(
    poster_path: str, face_detector, gender_detector, age_detector
) -> tuple[int, int, float, float]:
    """
    It takes a URL, downloads the image, and then runs the image through the model

    :param poster_path: the url of the image you want to analyse
    :type poster_path: str
    :return: A tuple of the form (width, height, aspect_ratio, average_color)
    """
    try:
        img = get_image(poster_path)
    except:
        return (0, 0, 0, 0)
    try:
        analysis = analyse_img(
            img,
            gender_detector=gender_detector,
            face_detector=face_detector,
            age_detector=age_detector,
        )
    except:
        return (0, 0, 0, 0)
    return clean_outputs(analysis, img.shape[0], img.shape[1])
