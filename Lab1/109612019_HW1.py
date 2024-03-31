# You are not allowed to import any additional packages/libraries.
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv

class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None

    def closed_form_fit(self, X, y):
        # Closed-form solution: w = (X^T * X)^(-1) * X^T * y
        skew = np.ones((X.shape[0], 1))  
        skew_X = np.concatenate((skew, X), axis = 1)
        X_transpose = np.transpose(skew_X)
        XTX = np.dot(X_transpose, skew_X)
        XTX_inv = np.linalg.inv(XTX)
        XTY = np.dot(X_transpose, y)
        weights = np.dot(XTX_inv, XTY)

        self.closed_form_weights = weights[1:]
        self.closed_form_intercept = weights[0]

    def gradient_descent_fit(self, X, y, lr, epochs):
        # Initialize weights and intercept
        num_samples, num_features = X.shape
        self.gradient_descent_weights = np.zeros(num_features)
        self.gradient_descent_intercept = 0

        # Gradient Descent
        for epoch in range(epochs):
            predictions = np.dot(X, self.gradient_descent_weights) + self.gradient_descent_intercept
            error = predictions - y

            # Update weights and intercept
            gradient_weights = (2 / num_samples) * np.dot(X.T, error)
            gradient_intercept = (2 / num_samples) * np.sum(error)

            self.gradient_descent_weights -= lr * gradient_weights
            self.gradient_descent_intercept -= lr * gradient_intercept
            #print(self.gradient_descent_weights)

            loss = LR.gradient_descent_evaluate(train_x, train_y)
            loss_history.append(loss)

    def get_mse_loss(self, prediction, ground_truth):
        #print(ground_truth)
        mse = np.mean((prediction - ground_truth) ** 2)
        return mse

    def closed_form_predict(self, X):
        return np.dot(X, self.closed_form_weights) + self.closed_form_intercept

    def gradient_descent_predict(self, X):
        return np.dot(X, self.gradient_descent_weights) + self.gradient_descent_intercept

    def closed_form_evaluate(self, X, y):
        prediction = self.closed_form_predict(X)
        return self.get_mse_loss(prediction, y)

    def gradient_descent_evaluate(self, X, y):
        prediction = self.gradient_descent_predict(X)
        return self.get_mse_loss(prediction, y)

    def plot_learning_curve(self, loss_history):
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training loss')
        plt.show()

if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    train_x = train_df.drop(["Performance Index"], axis=1)
    train_y = train_df["Performance Index"]
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    # Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    loss_history = []

    LR.gradient_descent_fit(train_x, train_y, lr=0.000185, epochs=500000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv"))
    test_x = test_df.drop(["Performance Index"], axis=1)
    test_y = test_df["Performance Index"]
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    closed_form_loss = LR.closed_form_evaluate(test_x, test_y)
    gradient_descent_loss = LR.gradient_descent_evaluate(test_x, test_y)
    print(f"Error Rate: {((gradient_descent_loss - closed_form_loss) / closed_form_loss * 100):.1f}%")

    #print(learning_rate)
    LR.plot_learning_curve(loss_history)
