import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, smell_type='dc'):
        self.smell_type = smell_type

    def load_data(self):
        data = None
        smell_type_to_names = {
            'gc': 'god-class',
            'dc': 'data-class',
            'fe': 'feature-envy',
            'lm': 'long-method'
        }
        if self.smell_type in smell_type_to_names:
            try:
                data = arff.loadarff('datasets/'+ smell_type_to_names[self.smell_type] +'.arff')
            except FileNotFoundError:
                print("Dataset for " + smell_type_to_names[self.smell_type] + " does not exist!")
                exit()
        else:
            print("Smell type is not found!")
            exit()

        return data

    def process_data(self, data):
        df = pd.DataFrame(data[0])
        Y_data = df.iloc[:, -1].values
        encoder = preprocessing.LabelEncoder()
        y = encoder.fit_transform(Y_data)
        
        X_copy = df.iloc[:, :-1].copy()

        imputer = SimpleImputer(strategy="median")
        new_X = imputer.fit_transform(X_copy)
        
        X_scaled = self.preprocess_x(new_X)

        self.viewFeatureDistribution(X_scaled)

        return X_scaled, y

    def preprocess_x(self, X):
        scaler = preprocessing.MinMaxScaler().fit(X)
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def viewFeatureDistribution(self, X):
        import matplotlib.pyplot as plt
        df = pd.DataFrame(X)
        del df[0]
        df.to_csv('features.csv', index=False)
        fig, axis = plt.subplots(9, 9, figsize=(10,10))
        axes = df.hist(ax=axis, bins=20, xlabelsize=1, ylabelsize=1)
        plt.tight_layout()
        plt.show()