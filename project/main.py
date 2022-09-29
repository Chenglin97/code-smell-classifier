import argparse
from models import Classifier, DataLoader, ClassifierGui
import tkinter as tk
from tkinter import *
import pandas as pd


class Main:
    def __init__(self):
        self.smell_type = None

        self.smell_type_to_names = {
            'gc': 'God Class',
            'dc': 'Data Class',
            'fe': 'Feature Envy',
            'lm': 'Long Method'
        }
        self.classifier_type_to_names = {
            'rf': 'Random Forest',
            'mlp': 'Neural Network',
            'svm': 'SVC(Linear)'
        }

    def _get_args(self):
        p = argparse.ArgumentParser()
        p.add_argument(
            '-l',
            "--smell_types",
            nargs='+',
            help="smell types: gc=God Class, dc=Data Class, fe=Feature Envy, lm=Long Method"
        )
        p.add_argument(
            "--classifier",
            help="classifiers available: rf=Random Forest, mlp=Neural Network (MLP), svm=SVC(Linear)",
            type=str, 
            default="rf"
        )
        p.add_argument(
            "--gui",
            help="Set gui to False if you want to run it in console",
            type=str, 
            default="True"
        )
        return p.parse_args()

    def start(self):
        """ Parse args to decide which training omde to use """

        args = self._get_args()
        if args.gui == "True":
            self.window = ClassifierGui()
            self.window.initView()
            # Exit the application when Tkinter window is closed
            exit()
        else:
            self.smell_types = args.smell_types
            self.classifier = args.classifier
        self.startTrainingCli(self.smell_types, self.classifier)

    def startTrainingCli(self, smell_types, classifier='rf'):
        """ Training mode in CLI if the GUI_argument is set to False  """

        valid_smells = ['gc', 'dc', 'fe', 'lm']
        valid_classifiers = ['rf', 'mlp', 'svm']
        if classifier not in valid_classifiers:
            print(f"Error: Invalid classifier type: {classifier}")
            exit()
        # Validate smell type input
        for smell_type in smell_types:
            if smell_type not in valid_smells:
                print("Error: Invalid smell type:", smell_type)
                exit()

        results = []

        for smell_type in smell_types:
            data_loader = DataLoader(smell_type)
            data = data_loader.load_data()
            X, y = data_loader.process_data(data)
            # run data with classifier model
            model = Classifier(X, y, classifier)
            ac, f1, ac_train, f1_train, training_time = model.train()
            print(f"Testing Accuracy: {ac}")
            print(f"F1 Score: {f1}")
            classifier_name = self.classifier_type_to_names[classifier]
            results.append((self.smell_type_to_names[smell_type], ac, f1, ac_train, f1_train, training_time, classifier_name))
        
        self.updateResultCSV(results)
        self.displayResultsCli(results)



    def updateResultCSV(self, results):
        """ Update training results in results.csv """

        try:
            df = pd.read_csv('results.csv')
        except FileNotFoundError:
            # Construct new dataframe if results.csv does not exist 
            df = pd.DataFrame(data={
                'smell_type': [],
                'accuracy': [],
                'f1': [],
                'accuracy_training': [],
                'f1_training': [],
                'training_time': [],
                'classifier': []
            })
        for row in results:
            smell_type, ac, f1, ac_train, f1_train, training_time, classifier = row
            if not df.loc[(df['smell_type'] == smell_type) & (df['classifier'] == classifier)].empty:
                df.loc[(df['smell_type'] == smell_type) & (df['classifier'] == classifier)] = row
            else:
                df.loc[len(df)] = row
        df.to_csv('results.csv', index=False)


    def displayResultsCli(self, results):
        """ Display training results in command line """

        print("=============================================================================================\n Results:")
        print("")
        print ("{:<15} {:<10} {:<10} {:<12} {:<12} {:<14} {:<15}".format( \
            'Smell Type','Accuracy','F1 Score', 'Train ACC','Training F1', 'Training Time', 'Classifier'))
        for smell_type, ac, f1, ac_train, f1_train, training_time, classifier in results:
            print ("{:<15} {:<10} {:<10} {:<12} {:<12} {:<14} {:<15}".format( \
                smell_type, ac, f1, ac_train, f1_train, training_time, classifier))
        print("\n\n")


if __name__ == '__main__':
    main = Main()
    main.start()