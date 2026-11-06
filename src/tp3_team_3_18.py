"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    Evaluate both the logistic regression and KNN models, give evaluation outputs, print the corresponding graphs

Assignment Information:
    Assignment:     tp3 team 3
    Team ID:        LC4 - 18
    Author:         Thomas Morgan, morga487@purdue.edu
                    Oscar Anderson, ande2059@purdue.edu
    Date:           11/2/2026

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
import matplotlib.pyplot as plt
from tp3_team_1_18 import load_dataset, predict_labels_knn,  train_val_test_split, calculate_metrics
from tp3_team_2_18 import calculate_sigmoid, train_logistic_regression, evaluate_logistic_regression


#PARAMS
#k_values: [int]: list of k values to test
#X_train: np[float][float]: training feature matrix
#y_train: np[int]: training labels
#X_val: np[float][float]: validation feature matrix
#y_val: np[int]: validation labels
#RET
#metrics: dict: double dictionary with keys acc and error, each containing a dictionary
#with train_key and val_key, each containing a list of corresponding acc or err for 
#each k value 
#PURPOSE
#evaluates the knn model for each k value given, which is the amount of nearest neighbors visited
def evaluate_knn_model(k_values, X_train, y_train, X_val, y_val):
    metrics = {"acc": {"train_acc": [], "val_acc": []}, "error": {"train_error": [], "val_error": []}}
    #loop thru all integer values of nearest neighbors to visit
    for k in k_values:
        #predict the labels for the training set and validation set
        predicted_train = predict_labels_knn(X_train, X_train, y_train, k)
        predicted_val = predict_labels_knn(X_val, X_train, y_train, k)
        #get the accuracy and error for training (should be very close to one)
        trainAccuracy, trainError = calculate_metrics(predicted_train, y_train)
        #get the accuracy and error for validation (should be lower than training)
        valAccuracy, valError = calculate_metrics(predicted_val, y_val)
        metrics["acc"]["train_acc"].append(trainAccuracy)
        metrics["acc"]["val_acc"].append(valAccuracy)
        metrics["error"]["train_error"].append(trainError)
        metrics["error"]["val_error"].append(valError)
    return(metrics)

#PARAMS
#metrics: dict: dictionary returned by KNN model
#k_values: [int]: List of tested k values
#PURPOSE
#prints a performance graph of the knn model
def plot_knn_performance(metrics, k_values):

    fig, ax = plt.subplots(1, 2, figsize=(12,6))

    #accuracy subplot
    ax[0].set_ylim(0, 1)
    ax[0].set_xlim(-2, 55)
    ax[0].plot(k_values, metrics["acc"]["train_acc"], color='cyan', marker='o', label='Training Accuracy')
    ax[0].plot(k_values, metrics["acc"]["val_acc"], color='orange', marker='o', label='Validation Accuracy')
    ax[0].set_title("Accuracy vs. Number of Neighbors (k)")
    ax[0].set_xlabel("k (Number of Neighbors)")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(True)
    ax[0].legend(loc='lower left')

    #error rate subplot
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(-2, 55)
    ax[1].plot(k_values, metrics["error"]["train_error"], color='cyan', marker='o', label='Training Error Rate')
    ax[1].plot(k_values, metrics["error"]["val_error"], color='orange', marker='o', label='Validation Error Rate')
    ax[1].set_title("Error Rate vs. Number of Neighbors (k)")
    ax[1].set_xlabel("k (Number of Neighbors)")
    ax[1].set_ylabel("Error Rate")
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("train_val.png")

#PARAMS
#k_values: [int]: list of tested k values
# #metrics: dict: dicionary returned by KNN model
#RET
#best_k: int: the k value with highest validation accuracy, tie is broken by lower k val
def select_best_k(k_values, metrics):
    val_acc = metrics["acc"]["val_acc"]
    best_acc = max(val_acc)
    return min(k for k, acc in zip(k_values, val_acc) if acc == best_acc)

#PARAMS
#X_test: np[float][float]: test feature matrix
#y_test: np[int]: test labels
#X_train: np[float][float]: training feature matrix
#y_train: np[int]: training labels
#best_k: int: selected value of k
#RET
#test_accuracy: float: test-set accuracy
#test_error: float: test-set error rate
def evaluate_knn_test_set(X_test, y_test, X_train, y_train, best_k):
    predicted_val = predict_labels_knn(X_test, X_train, y_train, best_k)
    test_accuracy, test_error = calculate_metrics(predicted_val, y_test)
    return test_accuracy, test_error

#PARAMS
#X_train: np[float][float]: training feature matrix
#y_train: np[int]: training labels
#X_val: np[float][float]: validation feature matrix
#y_val: np[int]: validation labels
#X_test: np[float][float]: test feature matrix
#y_test: np[int]: test labels
#w: np[float]: trarined weights
#b: float: trained bias
#RET
#metrics: dict: dictionary with train, val, and test keys, containing accuracy and error
def evaluate_logistic_model(X_train, y_train, X_val, y_val, X_test, y_test, w, b):
    train_accuracy, train_error = evaluate_logistic_regression(X_train, y_train, w, b)
    val_accuracy, val_error = evaluate_logistic_regression(X_val, y_val, w, b)
    test_accuracy, test_error = evaluate_logistic_regression(X_test, y_test, w, b)
    return {
        "train": {"accuracy": train_accuracy, "error": train_error},
        "val": {"accuracy": val_accuracy, "error": val_error},
        "test": {"accuracy": test_accuracy, "error": test_error},
    }

#PARAMS
#loss_history: [float]: training loss values produced during training
#val_loss_history: [float]: validation loss values produced during training
#RET
#PURPOSE
#Plot loss v iteration
def plot_logistic_training(loss_history, val_loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('training_progress.png')


def main():
    path1 = input("Enter the path to the feature dataset: ")
    shufBool = input("Shuffle the dataset? (yes/no): ").lower() in ('y', 'yes')
    seed = int(input("Enter a seed for loading the dataset: "))
    #seed for reproducibility (shuffle and weight initialization)
    np.random.seed(seed)

    feature_cols = ["hue_mean", "hue_std", "saturation_mean", "saturation_std", "value_mean", "value_std", "number_lines", "has_circle", "has_triangle"]
    x, y = load_dataset(path1, feature_cols, "ClassId", shufBool, seed)

    #splits data into seperate sections
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(x, y)
    print("\nData loaded and split into\n")
    print(f"Training set:\n  size: {X_train.shape[0]}")
    print(f"Validation set:\n  size: {X_val.shape[0]}")
    print(f"Test set:\n  size: {X_test.shape[0]}")
    print()
    print("KNN Validation Curve Summary")        
    k_values = [1, 3, 5, 7, 9, 15, 31, 53]
    metrics = evaluate_knn_model(k_values, X_train, y_train, X_val, y_val)
    for index, k in enumerate(k_values):
        print(f"k = {k}: train accuracy = {metrics['acc']['train_acc'][index]:.4f}, validation accuracy = {metrics['acc']['val_acc'][index]:.4f}")

    best_k = select_best_k(k_values, metrics)

    print()
    plot_knn_performance(metrics, k_values)
    print(f"Based on the plots, the best k appears to be: {best_k}")
    print()
    print("--- Final KNN Model Performance ---")
    test_accuracy, test_error = evaluate_knn_test_set(X_test, y_test, X_train, y_train, best_k)
    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Test Set Error Rate: {test_error:.4f}")
    print()
    print("Training logistic regression model with learning rate = 0.01 and num_iterations = 5000...")
    print()

    w, b, loss_history, val_history = train_logistic_regression(X_train, y_train, X_val, y_val, 0.01, 5000, False)
    plot_logistic_training(loss_history, val_history)
    log_train_accuracy, log_train_error = evaluate_logistic_regression(X_train, y_train, w, b)
    log_val_accuracy, log_val_error = evaluate_logistic_regression(X_val, y_val, w, b)
    log_test_accuracy, log_test_error = evaluate_logistic_regression(X_test, y_test, w, b)

    print("\nFinal Logistic Regression Evaluation:")
    print("--------------------------------------------------")
    print(f"Training Set - Accuracy: {log_train_accuracy:.4f} ({log_train_accuracy * 100:.2f}%), Error Rate: {log_train_error:.4f} ({log_train_error * 100:.2f}%)")
    print(f"Validation Set - Accuracy: {log_val_accuracy:.4f} ({log_val_accuracy * 100:.2f}%), Error Rate: {log_val_error:.4f} ({log_val_error * 100:.2f}%)")
    print(f"Test Set - Accuracy: {log_test_accuracy:.4f} ({log_test_accuracy * 100:.2f}%), Error Rate: {log_test_error:.4f} ({log_test_error * 100:.2f}%)")

    #get the accuracy of the best k value for KNN
    knn_best = metrics["acc"]["val_acc"][k_values.index(best_k)]

    print("\nModel Summary")
    print(f"Best KNN validation accuracy: {knn_best:.4f}")
    print(f"Logistic regression validation accuracy: {log_val_accuracy:.4f}")
    print(f"KNN test accuracy: {test_accuracy:.4f}")
    print(f"Logistic regression test accuracy: {log_test_accuracy:.4f}")

if __name__ == "__main__":
    main()

