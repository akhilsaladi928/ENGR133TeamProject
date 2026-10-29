
"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    KNN classifiers, etc. (need to update)

Assignment Information:
    Assignment:     tp2 team 2
    Team ID:        LC4 - 18
    Author:         Akhil Saladi, asaladi@purdue.edu
                    Rohan Ravula
                    Thomas Morgan
                    Oscar Anderson
    Date:           10/29/2026

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
import pandas as pd
from tp2_team_3_18 import scale_features

#PARAMS
#file_path: str: Path to csv file
#feature_cols: [str]: Names of cols to use as features
#label_col: str: Name of col containing binary labels
#shuffle: bool: whether to shuffle data
#seed: int: default 42, seed for reproducibility
#RET
#x: np[float][float]: feature matrix
#y np[int]: Label vector
#PURPOSE
# Loads a dataset from a CSV file, separates features and labels, and optionally shuffles the data.
def load_dataset(file_path, feature_cols, label_col, shuffle, seed=42):
    
    df = pd.read_csv(file_path)
    # Use the existing feature scaling function from Checkpoint 2 Task 3
    scale_features(df, feature_cols)

    x = df[feature_cols].to_numpy()
    y = df[label_col].to_numpy()

    if shuffle:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(y))
        x = x[shuffled_indices]
        y = y[shuffled_indices]

    return(x, y)

#PARAMS
#X: array of image features
#y: array of image labels
#train_ratio: float: proportion of data to use for training
#val_ratio: float: proportion of data to use for validation
#test_ratio: float: proportion of data to use for testing
#RET
#X_train: array of training features
#y_train: array of training labels
#X_val: array of validation features
#y_val: array of validation labels
#X_test: array of testing features
#y_test: array of testing labels
#PURPOSE
#Splits the dataset into training, validation, and testing sets based on the given ratios.
def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    #check that the splits all add up to 100%
    if (train_ratio + val_ratio + test_ratio) != 1.0:
        raise ValueError("The sum of train, validation, and test ratios must equal 1.0")
    #gets the length of the array which corrosponds to ratios given above
    firstsplitX = int(len(X) * train_ratio)
    secondsplitX = int(len(X) * (train_ratio + val_ratio))
    firstsplity = int(len(y) * train_ratio)
    secondsplity = int(len(y) * (train_ratio + val_ratio))\
    
    #splits the arrays by the given lengths above
    X_train, X_val, X_test = np.split(X, [firstsplitX, secondsplitX])
    y_train, y_val, y_test = np.split(y, [firstsplity, secondsplity])   
    return(X_train, y_train, X_val, y_val, X_test, y_test)

#PARAMS
#new_example: np[float]: features of one image
#X_train: np[float][float]: training feature matrix
#y_train: np[int]: training labels
#k: int: number of nearest neighbors
#RET
#predicted_label: int: 0 for deer warning sign, 1 for left turn ahead sign
#PURPOSE
#predicts the class for a single data point given its features
def knn_single_prediction(new_example, X_train, y_train, k):
    #compute the euclidean distance between the new example and the training matrix set
    distances = np.linalg.norm(X_train - new_example, axis=1)
    #get indices of the k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]
    #get labels of the k nearest neighbors
    k_nearest_labels = y_train[k_nearest_indices]
    #Find the mode of the k_nearest_labels array, which is the predicted label
    predicted_label = np.bincount(k_nearest_labels).argmax()
    return predicted_label

#PARAMS
#X_new: new feature data to classify
#X_train: training feature matrix
#y_train: training labels
#k: int: number of nearest neighbors
#RET
#predicted_labels: [int]: list of predicted classID labels (0,1)
#PURPOSE
#loop through training set to predict labels for new data points
def predict_labels_knn(X_new, X_train, y_train, k):
    predicted_labels = []
    #loop through each new data point and predict its label using KNN
    for i in range(len(X_new)):
            #runs single prediction on each image
            prediction = knn_single_prediction(X_new[i], X_train, y_train, k)
            #adds the predicted value (0, 1) to predicted labels
            predicted_labels.append(prediction)
    return(predicted_labels)

#PARAMS
#predicted_labels: predicted classID labels (0,1)
#true_labels: true classID labels
#RETURNS
#accuracy: proportion of predictions correct
#error: proportion of predictions incorrect
def calculate_metrics(predicted_labels, true_labels):
    correct = 0
    #represents how many images we are guessing
    number_predictions = len(predicted_labels)
    #itterates across all predicted images
    for i in range(number_predictions):
        #adds 1 to the correct answers if predict the correct classID
        if predicted_labels[i] == true_labels[i]:
            correct += 1
    #converts accuracy to a proportion
    accuracy = correct / number_predictions
    error = 1 - accuracy
    return(accuracy, error)

def main():
    path1 = input("Enter the path to the feature dataset: ")
    shufBool = input("Shuffle the dataset? (yes/no): ").lower() in ('y', 'yes')
    seed = int(input("Enter a seed for loading the dataset: "))
    kVal = int(input("Enter the value of k to use for KNN: "))

    #give the load function the names of the features to use
    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"]
    x, y = load_dataset(path1, feature_cols, "ClassId", shufBool, seed)

    #splits data into seperate sections
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(x, y)

    predicted_labels = predict_labels_knn(X_test, X_train, y_train, kVal)
    
    #prints summary of set lengths
    print("\nData loaded and split into\n")
    print("Training set:\n  size: {%d}", len(X_train))
    print("Validation set:\n  size: {%d}", len(X_val))
    print("Test set:\n  size: {%d}", len(X_test))

    #calculates the training, validation, and testing predictions and gets their corrosponding accuracy and error
    training = predict_labels_knn(X_train, X_train, y_train, kVal)
    train_accuracy, train_error = calculate_metrics(training, y_train)
    validation = predict_labels_knn(X_val, X_train, y_train, kVal)
    val_accuracy, val_error = calculate_metrics(validation, y_val)
    testing = predict_labels_knn(X_test, X_train, y_train, kVal)
    test_accuracy, test_error = calculate_metrics(testing, y_test)

    #prints accuracy and error of training data
    print("Evaluating KNN model on Training set with k = {kVal}...")
    print("\n--- Model Performance on Training Set ---")
    print("Training Set Accuracy: {%.4f}", train_accuracy)
    print("Training Set Error Rate: {%.4f}", train_error)

    #prints accuracy and error of validation data
    print("Evaluating KNN model on validation set with k = {kVal}...")
    print("\n--- Model Performance on Validation Set ---")
    print("Validation Set Accuracy: {%.4f}", val_accuracy)
    print("Validation Set Error Rate: {%.4f}", val_error) 

    #prints accuracy and error of testing data
    print("Evaluating KNN model on test set with k = {kVal}...")
    print("\n--- Model Performance on Test Set ---")
    print("Test Set Accuracy: {%.4f}", test_accuracy)
    print("Test Set Error Rate: {%.4f}", test_error)


if __name__ == "__main__":
    main()
    
