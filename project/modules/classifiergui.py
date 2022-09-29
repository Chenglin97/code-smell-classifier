import tkinter as tk
from tkinter import *
import pandas as pd
from .classifier import Classifier
from .dataloader import DataLoader
from scipy.io import arff

class ClassifierGui:
    def __init__(self):
        self.trainingLabel = None
        self.errorLabel = None
        self.resultLabelR1 = None
        self.resultLabelR2 = None
        self.resultLabelR3 = None
        self.resultLabelR4 = None
        self.reportLabels = []

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
        self.smell_type_to_file_names = {
            'gc': 'god-class',
            'dc': 'data-class',
            'fe': 'feature-envy',
            'lm': 'long-method'
        }

    def initView(self):
        self.window = tk.Tk()
        self.window.title("Chenglin's Smell Classifier") 
        self.window.geometry("520x620+10+20")

        self.gc_check_val = IntVar()
        self.dc_check_val = IntVar()
        self.fe_check_val = IntVar()
        self.lm_check_val = IntVar()

        gc_check=Checkbutton(self.window, text="God Class", variable=self.gc_check_val)
        dc_check=Checkbutton(self.window, text="Data Class", variable=self.dc_check_val)
        fe_check=Checkbutton(self.window, text="Feature Envy", variable=self.fe_check_val)
        lm_check=Checkbutton(self.window, text="Long Method", variable=self.lm_check_val)
        gc_check.grid(row=0, column=1)
        dc_check.grid(row=1, column=1)
        fe_check.grid(row=2, column=1)
        lm_check.grid(row=3, column=1)
        self.tk_labels = []
        self.startBtn = Button(self.window, text="Start Training")
        self.startBtn.grid(row=4, column=1)
        self.startBtn.bind('<Button-1>', self.startTraining)

        self.classifier_model = StringVar() 
        self.classifier_model.set('rf')
        r1 = Radiobutton(self.window, text="Random Forest", variable=self.classifier_model, value='rf')
        r2 = Radiobutton(self.window, text="Neural Network", variable=self.classifier_model, value='mlp')
        r3 = Radiobutton(self.window, text="SVC(Linear)", variable=self.classifier_model, value='svm')
        r1.grid(row=5, column=0)
        r2.grid(row=5, column=1)
        r3.grid(row=5, column=2)

        self.resultLabelR1 = Label(self.window, text="Smell Type", width=9)
        self.resultLabelR2 = Label(self.window, text="Accuracy" )
        self.resultLabelR3 = Label(self.window, text="F-1 Score" )
        self.resultLabelR4 = Label(self.window, text="Classifier" )

        self.resultLabelR1.grid(row=6, column=0)
        self.resultLabelR2.grid(row=6, column=1)
        self.resultLabelR3.grid(row=6, column=2)
        self.resultLabelR4.grid(row=6, column=3)
        self.initReport()
        self.window.mainloop()

    def startTraining(self, event):
        if self.errorLabel:
            self.errorLabel.destroy()
        ''' Training in GUI view '''
        self.smell_types = []
        if self.gc_check_val.get():
            self.smell_types.append('gc')
        if self.dc_check_val.get():
            self.smell_types.append('dc')
        if self.fe_check_val.get():
            self.smell_types.append('fe')
        if self.lm_check_val.get():
            self.smell_types.append('lm')
        
        if not self.smell_types:
            # display empty seletion message
            self.displayErrorMessage('Error: \nPlease choose at lease 1 training set!')
            return

        print("started training", self.smell_types)

        # display training message
        self.displayTrainingMessage()

        results = []
        for smell in self.smell_types:
            # import dataset 
            try:
                data = arff.loadarff('datasets/'+ self.smell_type_to_file_names[smell] +'.arff')
            except FileNotFoundError:
                self.deleteTrainingMessage()
                self.displayErrorMessage(f"Dataset for {self.smell_type_to_names[smell]} is not found")
                return
            data_loader = DataLoader(smell)
            data = data_loader.load_data()
            X, y = data_loader.process_data(data)
            # run data with classifier model
            model = Classifier(X, y, classifier=self.classifier_model.get())
            ac, f1, ac_train, f1_train, training_time = model.train()
            print("Testing Accuracy: " + str(ac))
            print("F1 Score: " + str(f1))
            classifier_name = self.classifier_type_to_names[self.classifier_model.get()]
            results.append((self.smell_type_to_names[smell], ac, f1, ac_train, f1_train, training_time, classifier_name))
        
        # delete training message
        self.deleteTrainingMessage()

        self.displayResultMessage(results)

        self.updateResultCSV(results)


    def initReport(self):
        if self.errorLabel:
            self.errorLabel.destroy()
        linebreak = Label(self.window, text="==========================================================")
        linebreak.grid(row=11, columnspan=10)
        filterButton = Button(self.window, text="Filter Models")
        filterButton.grid(row=12, column=1)
        filterButton.bind('<Button-1>', self.filterReport)

        resetButton = Button(self.window, text="Show All Models")
        resetButton.grid(row=12, column=2)
        resetButton.bind('<Button-1>', self.resetReport)

        compareButton = Button(self.window, text="Compare Model")
        compareButton.grid(row=12, column=3)
        compareButton.bind('<Button-1>', self.compareReport)

        reportLabel1 = Label(self.window, text="Smell Type", width=9)
        reportLabel2 = Label(self.window, text="Accuracy" )
        reportLabel3 = Label(self.window, text="F-1 Score" )
        reportLabel4 = Label(self.window, text="Classifier" )

        reportLabel1.grid(row=13, column=0)
        reportLabel2.grid(row=13, column=1)
        reportLabel3.grid(row=13, column=2)
        reportLabel4.grid(row=13, column=3)


    def displayErrorMessage(self, message="Error!"):
        if self.errorLabel:
            self.errorLabel.destroy()
        for label in self.tk_labels:
            label.destroy()
        self.tk_labels = []
        self.errorLabel = Label(self.window, text=message, fg='#f00')
        self.errorLabel.grid(row=7, columnspan=10)

    def displayTrainingMessage(self):
        for label in self.tk_labels:
            label.destroy()
        self.tk_labels = []
        if self.errorLabel:
            self.errorLabel.destroy()
        self.trainingLabel = Label(self.window, text="Training...\nPlease Wait...")
        self.trainingLabel.grid(row=7, column=1)
        self.window.update()

    def deleteTrainingMessage(self):
        if self.errorLabel:
            self.errorLabel.destroy()
        self.trainingLabel.destroy()
        self.window.update()
    
    def displayResultMessage(self, results):
        if self.errorLabel:
            self.errorLabel.destroy()
        self.tk_row = 7
        for smell, ac, f1, ac_train, f1_train, training_time, classifier in results:

            smell_label = Label(self.window, text=smell)
            ac_label = Label(self.window, text=ac )
            f1_label = Label(self.window, text=f1 )
            classifier_label = Label(self.window, text=classifier)

            self.tk_labels.append(smell_label)
            self.tk_labels.append(ac_label)
            self.tk_labels.append(f1_label)
            self.tk_labels.append(classifier_label)

            smell_label.grid(row=self.tk_row, column=0)
            ac_label.grid(row=self.tk_row, column=1)
            f1_label.grid(row=self.tk_row, column=2)
            classifier_label.grid(row=self.tk_row, column=3)

            self.tk_row += 1
        self.window.update()

    def filterReport(self, event):
        if self.errorLabel:
            self.errorLabel.destroy()
        smell_types = []
        if self.gc_check_val.get():
            smell_types.append('God Class')
        if self.dc_check_val.get():
            smell_types.append('Data Class')
        if self.fe_check_val.get():
            smell_types.append('Feature Envy')
        if self.lm_check_val.get():
            smell_types.append('Long Method')
        
        if not smell_types:
            # display empty seletion message
            self.displayErrorMessage('Error: \nPlease choose at lease 1 training set!')
            return

        classifier_filter = self.classifier_model.get()
        classifier_filter_dict = {
            'rf': 'Random Forest',
            'mlp': 'Neural Network',
            'svm': 'SVC(Linear)'
        }
        classifier_filter = classifier_filter_dict[classifier_filter]
        for label in self.reportLabels:
            label.destroy()

        df = pd.read_csv('results.csv')

        self.reportLabels = []
        curr_row = 14
        for _, row in df.iterrows():
            smell, ac, f1, ac_train, f1_train, training_time, classifier = row
            if smell not in smell_types or classifier != classifier_filter:
                continue
            smell_label = Label(self.window, text=smell)
            ac_label = Label(self.window, text=ac )
            f1_label = Label(self.window, text=f1 )
            classifier_label = Label(self.window, text=classifier)

            self.reportLabels.append(smell_label)
            self.reportLabels.append(ac_label)
            self.reportLabels.append(f1_label)
            self.reportLabels.append(classifier_label)

            smell_label.grid(row=curr_row, column=0)
            ac_label.grid(row=curr_row, column=1)
            f1_label.grid(row=curr_row, column=2)
            classifier_label.grid(row=curr_row, column=3)

            curr_row += 1
        if curr_row == 14:
            self.displayErrorMessage("Model not found. Please train before filtering")
        self.window.update()

    def resetReport(self, event):
        if self.errorLabel:
            self.errorLabel.destroy()
        for label in self.reportLabels:
            label.destroy()
        self.reportLabels = []
        try:
            df = pd.read_csv('results.csv')
        except FileNotFoundError:
            self.displayErrorMessage('Cannot find Trained Models. Please train first!')
        self.reportLabels = []
        curr_row = 14
        for _, row in df.iterrows():
            smell, ac, f1, ac_train, f1_train, training_time, classifier = row
            smell_label = Label(self.window, text=smell)
            ac_label = Label(self.window, text=ac )
            f1_label = Label(self.window, text=f1 )
            classifier_label = Label(self.window, text=classifier)

            self.reportLabels.append(smell_label)
            self.reportLabels.append(ac_label)
            self.reportLabels.append(f1_label)
            self.reportLabels.append(classifier_label)

            smell_label.grid(row=curr_row, column=0)
            ac_label.grid(row=curr_row, column=1)
            f1_label.grid(row=curr_row, column=2)
            classifier_label.grid(row=curr_row, column=3)

            curr_row += 1
        self.window.update()

    def compareReport(self, event):
        if self.errorLabel:
            self.errorLabel.destroy()
        smell_types = []
        if self.gc_check_val.get():
            smell_types.append('God Class')
        if self.dc_check_val.get():
            smell_types.append('Data Class')
        if self.fe_check_val.get():
            smell_types.append('Feature Envy')
        if self.lm_check_val.get():
            smell_types.append('Long Method')
        
        if not smell_types or len(smell_types) > 1:
            # display empty seletion message
            self.displayErrorMessage('Error: \nPlease choose ONLY 1 dataset!')
            return

        classifier_filter = self.classifier_model.get()
        classifier_filter_dict = {
            'rf': 'Random Forest',
            'mlp': 'Neural Network',
            'svm': 'SVC(Linear)'
        }
        classifier_filter = classifier_filter_dict[classifier_filter]
        for label in self.reportLabels:
            label.destroy()

        df = pd.read_csv('results.csv')

        self.reportLabels = []
        curr_row = 14
        for _, row in df.iterrows():
            smell, ac, f1, ac_train, f1_train, training_time, classifier = row
            if smell not in smell_types or classifier != classifier_filter:
                continue
                
            smell_label_training = Label(self.window, text=smell + "(Training)")
            ac_label_training = Label(self.window, text=ac_train )
            f1_label_training = Label(self.window, text=f1_train )
            classifier_label_training = Label(self.window, text=classifier)

            self.reportLabels.append(smell_label_training)
            self.reportLabels.append(ac_label_training)
            self.reportLabels.append(f1_label_training)
            self.reportLabels.append(classifier_label_training)

            smell_label_training.grid(row=curr_row, column=0)
            ac_label_training.grid(row=curr_row, column=1)
            f1_label_training.grid(row=curr_row, column=2)
            classifier_label_training.grid(row=curr_row, column=3)

            curr_row += 1
            smell_label = Label(self.window, text=smell + "(Testing)")
            ac_label = Label(self.window, text=ac )
            f1_label = Label(self.window, text=f1 )
            classifier_label = Label(self.window, text=classifier)

            self.reportLabels.append(smell_label)
            self.reportLabels.append(ac_label)
            self.reportLabels.append(f1_label)
            self.reportLabels.append(classifier_label)

            smell_label.grid(row=curr_row, column=0)
            ac_label.grid(row=curr_row, column=1)
            f1_label.grid(row=curr_row, column=2)
            classifier_label.grid(row=curr_row, column=3)

            curr_row += 1
        if curr_row == 14:
            self.displayErrorMessage("Model not found. Please train before filtering")

        self.window.update()

    def updateResultCSV(self, results):
        try:
            df = pd.read_csv('results.csv')
        except FileNotFoundError:
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