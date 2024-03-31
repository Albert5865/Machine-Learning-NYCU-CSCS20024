# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate, iteration):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        # Initialize weights and intercept
        self.weights = np.zeros(X.shape[1])
        self.intercept = 0

        for _ in range(self.iteration):
            # Calculate predicted probabilities
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.intercept)

            # Compute gradients
            gradient_weights = np.dot(X.T, (y_pred - y)) 
            gradient_intercept = np.sum(y_pred - y) 

            # Update weights and intercept
            self.weights -= self.learning_rate * gradient_weights
            self.intercept -= self.learning_rate * gradient_intercept
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        # Use the learned weights and intercept to make predictions
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.intercept)
        return np.where(y_pred >= 0.5, 1, 0)

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]

        self.m0 = np.mean(X_class0, axis = 0)
        self.m1 = np.mean(X_class1, axis = 0)

        self.sw = np.dot((X_class0 - self.m0).T , (X_class0 - self.m0)) + np.dot((X_class1 - self.m1).T , (X_class1 - self.m1))
        self.sb = np.dot((self.m1 - self.m0).reshape(-1, 1), (self.m1 - self.m0).reshape(1, -1))

        eig_vals, eig_vecs = np.linalg.eig(np.dot(np.linalg.inv(self.sw), self.sb))
        self.w = eig_vecs[:, np.argmax(eig_vals)]  
        self.slope = self.w[1] / self.w[0]
        self.intercept = -self.slope * np.mean(X[:, 0]) + np.mean(X[:, 1])

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        projected = np.dot(X, self.w)
        projected_m0 = np.dot(self.m0, self.w)
        projected_m1 = np.dot(self.m1, self.w)
        
        distances_m0 = np.abs(projected - projected_m0)
        distances_m1 = np.abs(projected - projected_m1)
        
        predictions = np.where(distances_m0 < distances_m1, 0, 1)
        
        return predictions.tolist()

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X_train, y_train, X_test, y_test):

        x_ax = np.linspace(np.min(X_train[:, 0]), np.max(X_train[:, 1]), 100)
        y_ax = self.slope * x_ax + self.intercept
        plt.plot(x_ax, y_ax, 'g', label='Projection Line')

        predicted_test = self.predict(X_test)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_test, cmap=plt.cm.Spectral, label='Predicted Testing Set')

        projected_x = (X_test[:, 0] + self.slope * X_test[:,1] - self.slope * self.intercept) / (self.slope**2 + 1)
        projected_y = self.slope * projected_x + self.intercept

        # Connect the dots and their projections with dashed lines
        for i in range(len(X_test)):
            plt.plot([X_test[i, 0], projected_x[i]], [X_test[i, 1], projected_y[i]], color='blue', linewidth=0.5)

        plt.axis('equal')
        plt.xlim(0, 150)
        plt.ylim(50,250)
        plt.title(f'Fisher Linear (Slope: {self.slope:.2f}, Intercept: {self.intercept:.2f})')
        plt.xlabel('Feature: Age')
        plt.ylabel('Feature: Thalach')
        plt.legend()
        plt.show()

     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.000001, iteration=120000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    FLD.plot_projection(X_train, y_train, X_test, y_test)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"