import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

class AlarmNet(nn.Module):

    @classmethod
    def compare_results(cls, results1, results2):
        print('Comparing results:')
        comparisons = {
            'accuracy': 100*(results1['accuracy'] - results2['accuracy'])/results1['accuracy'],
            'precision': 100*(results1['precision'] - results2['precision'])/results1['precision'],
            'recall': 100*(results1['recall'] - results2['recall'])/results1['recall'],
            'f1': 100*(results1['f1'] - results2['f1'])/results1['f1']
        }
        for key, value in comparisons.items():
            print(f'{key}: {value} %')
    def __init__(self, num_features=0, activation=nn.ReLU, hidden_layers = [64, 32, 16], pass_through=False):
        super().__init__()
        if pass_through:
            return
        self.stack_list = [nn.Linear(num_features, hidden_layers[0]), activation()]
        for i in range(1, len(hidden_layers)):
            self.stack_list.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), activation()])  # Use extend instead of assignment
        
        self.stack_list.extend([nn.Linear(hidden_layers[-1], 1), nn.Sigmoid()])  # Use extend instead of assignment
        self.stack = nn.Sequential(*self.stack_list)
    def forward(self, x):
        return self.stack(x)
    def predict(self, x):
        return self.forward(x).round()
    def train(self, epochs, X_train, X_test, Y_train, Y_test, alpha, loss_fn=nn.BCELoss(), print_epoch=500, optimizer=torch.optim.Adam):
        if print_epoch:
            print("-"*55)
            print(f"| {'Epoch':5} | {'Training Loss':5} | {' Test Loss  ':5} |    Recall    |")
            print("-"*55)
        train_hist = np.zeros(epochs)
        test_hist = np.zeros(epochs)
        optimizer = optimizer(self.parameters(), lr=alpha)

        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = self.forward(X_train)
            loss = loss_fn(Y_pred, Y_train)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                test_loss = loss_fn(self.forward(X_test), Y_test).item()
            test_hist[epoch] = test_loss
            if epoch % print_epoch == 0 or epoch == epochs-1:
                # Center values between pipes so that the printed output looks like a table
                recall = recall_score(Y_test.cpu().detach().numpy(), self.predict(X_test).cpu().detach().numpy())
                print(f"| {epoch:5} | {loss.item():13.10f} | {test_loss:10.10f} | {recall:5.10f} |")
                print("-"*55)
            train_hist[epoch] = loss.item()

        Y_pred = self.predict(X_test)
        self.last_pred = Y_pred
        self.last_test = Y_test
        self.test_hist = test_hist
        self.train_hist = train_hist
        return [Y_test,Y_pred]
    
    def get_results(self, Y_test=None, Y_pred=None):
        if Y_test is None:
            Y_test = self.last_test
        if Y_pred is None:
            Y_pred = self.last_pred
        Y_test = Y_test.cpu().detach().numpy()
        Y_pred = Y_pred.cpu().detach().numpy()
        results = {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred),
            'recall': recall_score(Y_test, Y_pred),
            'f1': f1_score(Y_test, Y_pred),
            'confusion_matrix': confusion_matrix(Y_test, Y_pred),
            'classification_report': classification_report(Y_test, Y_pred)
        }
        self.last_results = results
        return results
    def print_results(self, results=None):
        if results is None:
            try: 
                results = self.last_results
            except:
                results = self.get_results()
        for key, value in results.items():
            if key not in ['confusion_matrix', 'classification_report']:
                print(f'{key.capitalize()}: {value}')
            else:
                print(f'{key.capitalize()}:\n{value}')
    
    def train_and_print(self, epochs, X_train, X_test, Y_train, Y_test, alpha):
        Y_pred = self.train(epochs, X_train, X_test, Y_train, Y_test, alpha).cpu().detach().numpy().round().astype(int)
        self.print_results(Y_test, Y_pred)
        
    def plot_training(self, title):
        plt.plot(self.train_hist, label='Training Loss')
        plt.plot(self.test_hist, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.show()
        
class ConstantPredictor(AlarmNet):
    def __init__(self, val):
        self.val = val
    def predict(self, x):
        return torch.tensor([self.val]*x.shape[0]).reshape(-1, 1).float()
    def train(self, *args, **kwargs):
        pass
    def get_results(self):
        return {
            'accuracy': None,
            'precision': None,
            'recall': None,
            'f1': None,
            'confusion_matrix': None,
            'classification_report': None
        }
    def print_results(self):
        super().print_results(self.get_results())
        
        
        