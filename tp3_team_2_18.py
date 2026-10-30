
"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    Need Description

Assignment Information:
    Assignment:     tp3 team 2
    Team ID:        LC4 - 18
    Author:         Akhil Saladi, asaladi@purdue.edu
                    Rohan Ravula
                    Thomas Morgan
                    Oscar Anderson, ande2059@purdue.edu
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

#PARAMS
#z: float or np float arr: input to the sigmoid function
#RET
#sigmoid: float or np float arr: output of the sigmoid function
#PURPOSE
#Calculates the sigmoid function for the given input.
def calculate_sigmoid(z):
    #return sigmoid, making sure to keep extra large vaLues between range
    return 1 / (1 + np.exp(-np.clip(z, -10.0, 10.0)))

#PARAMS
#X: np[float][float]: feature matrix
#w: np[float]: weight vector
#b: float: bias term
#RET
#proba: np[float]: predicted probabilities
#PURPOSE
#Predicts the probabilities for the given feature matrix using the logistic regression model.
def predict_proba(X, w, b):
    #compute dot product plus added bias
    linear_model = np.dot(X, w) + b
    #return the values passed thru sigmoid function
    return calculate_sigmoid(linear_model)

#PARAMS
#X: np[float][float]: feature matrix
#w: np[float]: weight vector
#b: float: bias term
#threshold: float: decision threshold
#RET
#predicted_labels: np[int]: predicted labels
#PURPOSE
#Predicts the labels for the given feature matrix using the logistic regression model.
def predict_labels(X, w, b, threshold=0.5):
    proba = predict_proba(X, w, b)
    predicted_labels = (proba >= threshold).astype(int)
    return predicted_labels


def main():
    pass

if __name__ == "__main__":
    main()
    
