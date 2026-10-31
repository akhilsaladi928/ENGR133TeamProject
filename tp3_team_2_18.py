
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
from tp3_team_1_18 import load_dataset, train_val_test_split, calculate_metrics 

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
    yhat = predict_proba(X, w, b)
    predicted_labels = (yhat >= threshold).astype(int)
    return predicted_labels

#PARAMS
#X: np[float][float]: feature matrix
#y: np[int]: true labels
#w: np[float]: current weight vector
#b: float: current bias 
#RET
#loss: float: Scalar training loss 
#dw: np[float]: Gradient with respect to w
#db: float: Gradient with respect to b
def compute_loss_and_grads(X, y, w, b):
    #get the predicted probabilities, ensure clipping to avoid log(0) issues
    yhat = np.clip(predict_proba(X, w, b), 1e-15, 1 - 1e-15)
    #compute the average loss across all samples using formula 9.5
    Loss = -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))
    #compute the gradients with respect to w and b using formulas 9.8 and 9.9
    dw = np.dot(X.T, (yhat - y)) / X.shape[0] #9.8
    db = np.sum(yhat - y) / X.shape[0] #9.9
    return Loss, dw, db

#PARAMS
#X_train: np[float][float]: training feature matrix
#y_train: np[int]: training labels
#X_val: np[float][float]: validation feature matrix
#y_val: np[int]: validation labels
#learning_rate: float: step size for gradient descent
#num_iterations: int: number of iterations for training
#RET
#w_best: np[float]: weight vector with lowest validation loss
#b_best: float: bias with lowest validation loss
#loss_history: [float]: training loss at each iteration
#val_loss_history: [float]: validation loss at each iteration
def train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate=0.01, num_iterations=1000):
    # Initialize weights as random numbers between -0.01 and 0.01, and bias as 0
    w = np.random.randn(X_train.shape[1]) * 0.01
    b = 0.0
    w_best = w.copy()
    b_best = b
    best_val_loss = float('inf')
    # Store loss values for plotting or analysis
    loss_history = []
    val_loss_history = []
    #repeat for each iteration
    for i in range(num_iterations):
        #compute loss and gradients
        train_loss, dw, db = compute_loss_and_grads(X_train, y_train, w, b)
        #update w and b using gradient descent equations
        w -= learning_rate * dw
        b -= learning_rate * db
        #append training loss to loss_history
        loss_history.append(train_loss)
        #calculate and append validation loss to val_loss_history
        val_loss, _, _ = compute_loss_and_grads(X_val, y_val, w, b)
        val_loss_history.append(val_loss)
        #check if the loss is the best so far and add if so
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            w_best = w.copy()
            b_best = b
        #print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    return w_best, b_best, loss_history, val_loss_history

#PARAMS
#X: np[float][float]: feature matrix
#y: np[int]: true labels
#w: np[float]: trained weight vector
#b: float: trained bias
#RET
#accuracy: float: proportion of correct predictions
#error_rate: float: proportion of incorrect predictions
def evaluate_logistic_regression(X, y, w, b):
    #predict labels using the trained model
    y_pred = predict_labels(X, w, b)
    #calculate accuracy and error rate using previous calculate_metrics function
    accuracy, error_rate = calculate_metrics(y, y_pred)
    return accuracy, error_rate

def main():
    path1 = input("Enter the path to the feature dataset: ")
    shufBool = input("Shuffle the dataset? (yes/no): ").lower() in ('y', 'yes')
    seed = int(input("Enter a seed for loading the dataset: "))
    #seed for reproducibility (shuffle and weight initialization)
    np.random.seed(seed)
 
    #give the load function the names of the features to use
    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"]
    x, y = load_dataset(path1, feature_cols, "ClassId", shufBool, seed)
 
    #display shapes of the loaded data
    print(f"\nFeatures shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
 
    #splits data into seperate sections
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(x, y)
    #prints summary of set lengths
    print("Data loaded and split into\n")
    print(f"Training set:\n  size: {len(X_train)}")
    print(f"Validation set:\n  size: {len(X_val)}")
    print(f"Test set:\n  size: {len(X_test)}")
 
    #set vals for learning rate and train number
    learning_rate = 0.01
    num_iterations = 5000
 
    print(f"\nTraining logistic regression model with learning rate = {learning_rate} and num_iterations = {num_iterations}...\n")
 
    #actually train the model to get best weights
    w_best, b_best, loss_history, _ = train_logistic_regression(X_train, y_train, X_val, y_val, learning_rate, num_iterations)
 
    print("\nFinal Model Evaluation with the best parameters:")
    print("--------------------------------------------------")
 
    #evaluate the model for training
    accuracy_train, error_rate_train = evaluate_logistic_regression(X_train, y_train, w_best, b_best)
    #evaluate the model for validation
    accuracy_val, error_rate_val = evaluate_logistic_regression(X_val, y_val, w_best, b_best)
    #evaluate the model for testing
    accuracy_test, error_rate_test = evaluate_logistic_regression(X_test, y_test, w_best, b_best)
 
    #report as percentages to 2 decimal places
    print(f"Training Set - Accuracy: {accuracy_train * 100:.2f}%, Error Rate: {error_rate_train * 100:.2f}%")
    print(f"Validation Set - Accuracy: {accuracy_val * 100:.2f}%, Error Rate: {error_rate_val * 100:.2f}%")
    print(f"Test Set - Accuracy: {accuracy_test * 100:.2f}%, Error Rate: {error_rate_test * 100:.2f}%")
 
    print("\nFinal Parameters:")
    print(f"Weights shape: {w_best.shape}")
    print(f"Bias: {b_best:.6f}")
    print(f"Final training loss: {loss_history[-1]:.6f}")
 
    #feature importance: absolute value of each weight
    print("\nFeature Importance (absolute weight values):")
    for i, weight in enumerate(w_best):
        print(f"Feature {i}: {abs(weight):.6f}")

if __name__ == "__main__":
    main()
    
