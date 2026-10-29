"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    To create a program which measures the temperature and
    pressure of an extraction vessel to provide warnings
    and allow the values to stay in the acceptable range

Assignment Information:
    Author: Rohan Raula, ravular@purdue.edu
    Assignment:     tp2 team 3
    Team ID:        LC4 - 18
    Date:           10/27/2026

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

import matplotlib
try:
    matplotlib.use('MacOSX')
except ImportError:
    matplotlib.use('Agg')
import pandas as pd
#pd.set_option('display.width', 80)
import matplotlib.pyplot as plt


def scale_features(df, feature_columns):    #this is the scaling function that will be used to scale the features in the dataset
    for column in feature_columns:
        mean = df[column].mean()    #this line calculates the mean of the current feature column
        std = df[column].std()  #this line calculates the standard deviation of the current feature column
        df[column] = (df[column] - mean) / std  #this line scales the current feature column by subtracting the mean and dividing by the standard deviation, standardizing the feature to have a mean of 0 and a standard deviation of 1


def main():
    csv_path = input("Enter the name of CSV image features file:\n")
    scale_choice = input("Do you want to apply feature scaling? (yes/no):\n").strip().lower()

    feature_df = pd.read_csv(csv_path)

    #this is the list of feature columns that will be used for scaling
    feature_columns = ["hue_mean", "hue_std",
                        "saturation_mean", "saturation_std",
                        "value_mean", "value_std",
                        "number_lines", "has_circle",
                        "has_triangle"]

    if str.lower(scale_choice) == ("yes"):   #this line checks if the user wants to apply feature scaling
        print("Applying feature scaling to the dataset...")
        scale_features(feature_df, feature_columns)
        print("Feature scaling applied successfully.")
    else:
        print("Proceeding without feature scaling.")

    print()

    print("Feature Dataset Shape:")
    print(feature_df.shape)

    print()

    print("Feature Dataset Head:")
    print(feature_df.head())
    
    print()

    x_feature = input("Enter the column to use as the x-axis feature:\n")   #name of the column they want to use as the x-axis feature for the scatter plot

    print()

    y_feature = input("Enter the column to use as the y-axis feature:\n")   #name of the column they want to use as the y-axis feature for the scatter plot

    print()

    z_feature = input("Enter the column to use as the z-axis feature (or press Enter to skip):\n").strip()  #name of the column they want to use as the z-axis feature for the scatter plot

    print()

    deer = feature_df[feature_df["ClassId"] == 0]
    left_turn = feature_df[feature_df["ClassId"] == 1]

    fig = plt.figure()

    if z_feature == "":
        ax = fig.add_subplot(111)
        ax.scatter(deer[x_feature], deer[y_feature],
                   c='red', alpha=0.6, label='Deer Warning')
        ax.scatter(left_turn[x_feature], left_turn[y_feature],
                   c='blue', alpha=0.6, label='Left Turn Ahead')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(deer[x_feature], deer[y_feature], deer[z_feature],
                   c='red', alpha=0.6, label='Deer Warning')
        ax.scatter(left_turn[x_feature], left_turn[y_feature], left_turn[z_feature],
                   c='blue', alpha=0.6, label='Left Turn Ahead')
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_zlabel(z_feature)

    ax.legend()

    output_filename = "KNN_feature_space.png"
    plt.savefig(output_filename)
    print("Scatter plot saved to", output_filename)

    plt.show()


if __name__ == "__main__":
    main()