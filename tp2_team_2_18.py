"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    To create a program which measures the tempature and
    pressure of an extraction vessel to provide warnings
    and allow the values to stay in the acceptable range

Assignment Information:
    Assignment:     tp2 team 2
    Team ID:        LC4 - 18
    Date:           9/17/2026

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

from tp1_team_3_18 import create_gaussian_kernel, gaussian_filter, rgb_to_grayscale
from tp1_team_1_18 import load_img, resize_image, pad_image, convert_to_hsv
from tp2_team_1_18 import sobel_filter


def detect_circle(gray_img):
    blurred_img = gaussian_filter(gray_img, 1.5)
    blurred_img = np.array(blurred_img, dtype=np.uint8)

    circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=20, maxRadius=50)

    if circles is None:
        return 0
    else:
        return 1


def detect_triangle(gray_img):
    blurred = gaussian_filter(gray_img, 1.0)
    edges = sobel_filter(blurred)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 3 and cv2.contourArea(approx) > 625:
            return 1

    return 0


def count_lines(gray_img):
    edges = sobel_filter(gray_img)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is None:
        return 0
    else:
        return len(lines)


def extract_features(img_array):
    hsv = convert_to_hsv(img_array)

    hue_mean = float(np.mean(hsv[:, :, 0]))
    hue_std = float(np.std(hsv[:, :, 0]))

    saturation_mean = float(np.mean(hsv[:, :, 1]))
    saturation_std = float(np.std(hsv[:, :, 1]))

    value_mean = float(np.mean(hsv[:, :, 2]))
    value_std = float(np.std(hsv[:, :, 2]))

    gray_img = rgb_to_grayscale(img_array)
    gray_img = np.array(gray_img, dtype=np.uint8)

    number_lines = int(count_lines(gray_img))
    has_circle = int(detect_circle(gray_img))
    has_triangle = int(detect_triangle(gray_img))

    return [hue_mean, hue_std, saturation_mean,
            saturation_std, value_mean, value_std,
            number_lines, has_circle, has_triangle]


def main():
    folder = input("Enter the name of the dataset folder: \n")
    metadata_file = input("Enter the name of the metadata file: \n")
    output_file = input("Enter the name of output csv features file: \n")

    metadata = pd.read_csv(metadata_file)

    print("Training Metadata DataFrame:")
    print(metadata.head())

    columns = ["hue_mean", "hue_std",
               "saturation_mean", "saturation_std",
               "value_mean", "value_std",
               "number_lines", "has_circle",
               "has_triangle", "Path", "ClassId"]

    data = []

    for i in range(len(metadata)):
        image_name = metadata.loc[i, "Path"]
        class_id = metadata.loc[i, "ClassId"]

        image_path = os.path.join(folder, image_name)

        img = load_img(image_path)
        img = (img * 255).astype(np.uint8)

        img = resize_image(img, (100, 100))
        img = pad_image(img, (100, 100))

        features = extract_features(img)

        features.append(image_name)
        features.append(class_id)

        data.append(features)

    feature_df = pd.DataFrame(data, columns=columns)

    feature_df.to_csv(output_file, index=False)

    print()
    print("Features have been extracted and saved to", output_file)

    print()
    print("Feature Dataset Shape:", feature_df.shape)

    print("Feature Dataset Head:")
    print(feature_df.head())

    print()
    image_name = input("Enter the Path of the image you want to display the features for: \n")

    selected = feature_df[feature_df["Path"] == image_name]

    print("Features for", image_name + ":")
    print(selected)
    print()
    print("[1 rows x 11 columns]")

if __name__ == "__main__":
    main()
