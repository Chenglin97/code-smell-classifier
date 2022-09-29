from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import warnings
import time
warnings.filterwarnings("ignore")


class Classifier:
    def __init__(self, X, y, classifier='rf'):
        self.training_data = X
        self.training_label = y
        self.classifier = classifier
    
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_label, test_size=0.15)
        
        ''' Use oversampling to balance labels for training set '''
        for i, y in enumerate(y_train):
            if y == 1:
                for _ in range(4):
                    newRow = X_train[i]
                    X_train = np.vstack((X_train, newRow))
                    y_train = np.append(y_train, y_train[i])

        ''' Random Forest '''
        if self.classifier == 'rf':
            clf = RandomForestClassifier(random_state=0)
            param_grid = {
                "n_estimators" : [1, 10, 50, 100],  
                "max_depth": [1, 2, 5],
            }
        
        ''' Neural Network (MLP) '''
        if self.classifier == 'mlp':
            clf = MLPClassifier(alpha=1e-5, random_state=0, max_iter=100)
            param_grid = {
                "hidden_layer_sizes": [(64, 32, 16), (32, 16, 8), (64, 32)],
            }
        
        """ SVM """
        if self.classifier == 'svm':
            clf = svm.SVC()
            param_grid = {
                'C': [0.1,1, 10, 100], 
                'gamma': [1,0.1,0.01,0.001]
            }
        
        if not self.classifier:
            print(f"Error: classifier {self.classifier} not found.")
            exit()
            
        start = time.time()

        grid_search = GridSearchCV(
            clf,
            param_grid = param_grid,
            scoring = 'f1',
            return_train_score = True,
            cv = 5,
            verbose=10
        ) 
        grid_search.fit(X_train, y_train)

        y_train_pred = grid_search.predict(X_train)

        y_pred = grid_search.predict(X_test)
        print("\nbest estimator is", grid_search.best_estimator_, "\n")
        stop = time.time()
        
        training_time = stop - start

        ac_train, f1_train = accuracy_score(y_train, y_train_pred), f1_score(y_train, y_train_pred, average="weighted")

        ac, f1 = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="weighted")

        ac_formatted = str(float(ac) * 100)[:2] + "%"
        f1_formatted = str(float(f1) * 100)[:2] + "%"

        ac_train_formatted = str(float(ac_train) * 100)[:2] + "%"
        f1_train_formatted = str(float(f1_train) * 100)[:2] + "%"
        training_time_formatted = str(int(training_time)) + " seconds"

        return ac_formatted, f1_formatted, ac_train_formatted, f1_train_formatted, training_time_formatted