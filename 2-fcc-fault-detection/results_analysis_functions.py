import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

class detection_results():

    def __init__(self, df_detect):
        
        # df_detect columns: timestamp, y, y_pred, fault_score, y_label, y_label_pred
        # if bayesian df_detect columns: timestamp, y, y_pred_mean, y_pred_std, fault_score, y_label, y_label_pred

        self.df = df_detect

        # detection classification

        self.tp = self.df[(self.df['y_label'] == 1) & (self.df['y_label_pred'] == 1)]
        self.fp = self.df[(self.df['y_label'] == 0) & (self.df['y_label_pred'] == 1)]
        self.fn = self.df[(self.df['y_label'] == 1) & (self.df['y_label_pred'] == 0)]

    def detection_result_plot(self, plot_title, plot_x_axis, plot_y_axis):

        plt.figure(figsize=(20,10))
        plt.plot(self.df['timestamp'], self.df['y'], label=plot_y_axis, color='navy')
        plt.scatter(self.tp['timestamp'], self.tp['y'], marker='o', s=100, color='lightgreen', label='True positive')
        plt.scatter(self.fp['timestamp'], self.fp['y'], marker='X', s=100, color='black', label='False positive')
        plt.scatter(self.fn['timestamp'], self.fn['y'], marker='X', s=100, color='red', label='False negative')
        plt.xlabel(plot_x_axis)
        plt.ylabel(plot_y_axis)
        plt.title(plot_title)
        plt.legend()
        plt.show()

    def detection_result_prob_plot(self, plot_title, plot_x_axis, plot_y_axis, z_threshold):

        plt.figure(figsize=(20,10))
        plt.fill_between(self.df['timestamp'], 
                         self.df['y_pred_mean'] + z_threshold*(self.df['y_pred_std']), 
                         self.df['y_pred_mean'] - z_threshold*(self.df['y_pred_std']), 
                         alpha=0.5, color ='lightblue', label='Predictive distribution')
        plt.plot(self.df['timestamp'], self.df['y_pred_mean'], color='steelblue', label='Prediction mean')                 
        plt.plot(self.df['timestamp'], self.df['y'], label=plot_y_axis, color='navy')
        plt.scatter(self.tp['timestamp'], self.tp['y'], marker='o', s=100, color='lightgreen', label='True positive')
        plt.scatter(self.fp['timestamp'], self.fp['y'], marker='X', s=100, color='black', label='False positive')
        plt.scatter(self.fn['timestamp'], self.fn['y'], marker='X', s=100, color='red', label='False negative')
        plt.xlabel(plot_x_axis)
        plt.ylabel(plot_y_axis)
        plt.title(plot_title)
        plt.legend()
        plt.show()

    def detection_confusion_matrix(self):

        self.confusion_matrix = confusion_matrix(self.df['y_label'], self.df['y_label_pred'])

        sns.heatmap(self.confusion_matrix / np.sum(self.confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.show()

        return self.confusion_matrix


    def detection_result_metrics(self):

        tn = self.confusion_matrix[0,0]
        fp = self.confusion_matrix[0,1]
        fn = self.confusion_matrix[1,0]
        tp = self.confusion_matrix[1,1]

        tpr = tp / (tp + fn)
        tnr = tn / (fp + tn)
        fpr = fp / (fp + tn)

        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        balanced_accuracy = (tpr + tnr) / 2

        f1_score = 2*precision*tpr/(precision + tpr)

        print('Detection performance metrics:\n')
        print(f'True negative: {tn}')
        print(f'False positive: {fp}')
        print(f'False negative: {fn}')
        print(f'True positive: {tp}')
        print(f'True positive rate (TPR): {tpr}')
        print(f'True negative rate (TNR): {tnr}')
        print(f'False positive rate (FPR): {fpr}')
        print(f'Precision: {precision}')
        print(f'Accuracy: {accuracy}')
        print(f'Balanced accuracy: {balanced_accuracy}')
        print(f'F1 score: {f1_score}')