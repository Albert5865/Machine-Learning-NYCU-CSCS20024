# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def gini(sequence):
    pj = np.unique(sequence, return_counts=True)[1] / len(sequence)
    return 1 - np.sum(pj**2)

def entropy(sequence):
    pj = np.unique(sequence, return_counts=True)[1] / len(sequence)
    return - np.sum(np.dot(pj, np.log2(pj)))

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, class_index=None):
        self.feature_index = feature_index
        self.class_index = class_index
        self.threshold = threshold    
        self.left = None           
        self.right = None        

class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None

    def gini_entropy_common(self, y_data, sample_weight, is_gini=True):
        if len(y_data) == 0 or len(np.unique(y_data)) < 2:
            return 0
        pj = np.array([np.sum(sample_weight[y_data == 0]), np.sum(sample_weight[y_data == 1])]) / np.sum(sample_weight)
        if is_gini:
            return 1 - np.sum(pj**2)
        else:
            return -np.sum(np.dot(pj, np.log2(pj)))

    def gini(self, y_data, sample_weight):
        return self.gini_entropy_common(y_data, sample_weight, is_gini=True)

    def entropy(self, y_data, sample_weight):
        return self.gini_entropy_common(y_data, sample_weight, is_gini=False)

    def impurity(self, y, sample_w):
        if self.criterion == 'gini':
            return self.gini(y,sample_w)
        elif self.criterion == 'entropy':
            return self.entropy(y,sample_w)

    def tree_generate(self, X, y, depth, sample_w):
        unique_classes, counts = np.unique(y, return_counts=True)

        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return Node(class_index=unique_classes[counts.argmax()])

        num_features = X.shape[1]
        best_feature, best_threshold, min_imp = None, None, float('inf')

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask, right_mask = X[:, feature] <= threshold, X[:, feature] > threshold
                left_impurity = self.gini(y[left_mask], sample_w[left_mask]) if self.criterion == 'gini' else self.entropy(y[left_mask], sample_w[left_mask])
                right_impurity = self.gini(y[right_mask], sample_w[right_mask]) if self.criterion == 'gini' else self.entropy(y[right_mask], sample_w[right_mask])
                impurity = (len(y[left_mask]) * left_impurity + len(y[right_mask]) * right_impurity) / len(y)

                if impurity < min_imp:
                    min_imp, best_feature, best_threshold = impurity, feature, threshold

        new_node = Node(feature_index=best_feature, threshold=best_threshold)
        imp = self.impurity(y, sample_w)
        
        if min_imp == imp:
            return Node(class_index=unique_classes[counts.argmax()])

        left, right = X[:, best_feature] <= best_threshold, X[:, best_feature] > best_threshold
        if len(left) > 0 and len(right) > 0:
            new_node.left = self.tree_generate(X[left], y[left], depth + 1, sample_w[left])
            new_node.right = self.tree_generate(X[right], y[right], depth + 1, sample_w[right])

        return new_node

    def fit(self, X, y, sample_w=None):
        sample_w = np.ones(len(y)) / len(y) if sample_w is None else sample_w
        self.root = self.tree_generate(X, y, depth=0, sample_w=sample_w)

    def predict_tree(self, node, x):
        if node.class_index is not None:
            return node.class_index
        next_node = node.left if x[node.feature_index] <= node.threshold else node.right
        return self.predict_tree(next_node, x)

    def predict(self, X):
        y_pred = np.array([self.predict_tree(self.root, x) for x in X])
        return y_pred
    
    def plot(self, columns):
        def get_importance(node, importance):
            if node and node.feature_index is not None:
                importance[node.feature_index] += 1
            if node:
                get_importance(node.left, importance)
                get_importance(node.right, importance)

        feature_importance = np.zeros(len(columns))
        get_importance(self.root, feature_importance)
        plt.barh(range(len(columns)), feature_importance, align='center')
        plt.yticks(range(len(columns)), columns)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Decision Tree Feature Importance')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        n_samples, _ = X.shape
        w = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            weak_learner = DecisionTree(criterion=self.criterion, max_depth=1)
            weak_learner.fit(X, y, sample_w=w)
            
            predictions = weak_learner.predict(X)
            weighted_error = np.sum(w[predictions != y])
            
            if weighted_error >= 1:
                alpha = 0
            elif weighted_error <= 0.5:
                alpha = 20
            else:
                alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)

            temp = np.where(y != predictions, -1, 1)
            w *= np.exp(-alpha * temp)
            w /= np.sum(w)
            
            self.models.append(weak_learner)
            self.alphas.append(alpha)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        if not self.models:
            raise ValueError("The model has not been trained yet. Call fit() first.")
        predictions = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            y_pred = model.predict(X)
            y_pred[y_pred == 0] = -1
            predictions += alpha * y_pred
        final_predictions = np.sign(predictions)
        final_predictions[final_predictions <= 0] = 0
        return final_predictions

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    root = DecisionTree(criterion='gini', max_depth=7)
    root.fit(X_train, y_train)
    y_pred = root.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    root = DecisionTree(criterion='entropy', max_depth=7)
    root.fit(X_train, y_train)
    y_pred = root.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))
    root = DecisionTree(criterion='gini', max_depth=15)
    root.fit(X_train, y_train)
    root.plot(columns=["age","sex","cp","fbs","thalach","thal"])

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=200)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
