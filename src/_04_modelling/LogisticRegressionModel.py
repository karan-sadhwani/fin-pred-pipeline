
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, C=1e6, max_iter=100000, multi_class="ovr"):
        # Initialize the logistic regression model with specified parameters
        self.C = C
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.model = self.initialise_model()
        print("after initialising in logreg")
        print(self.C)

    def initialise_model(self):
        return LogisticRegression(C=self.C, max_iter=self.max_iter, multi_class=self.multi_class)
    
    def fit(self, X_train, y_train):
        # Call the fit method of the scikit-learn LogisticRegression model
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
    
    