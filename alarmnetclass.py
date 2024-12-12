import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from torch.amp import autocast, GradScaler
#import svm 
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import time



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
            self.stack_list.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), activation()]) 
        
        self.stack_list.extend([nn.Linear(hidden_layers[-1], 1)])  
        self.stack = nn.Sequential(*self.stack_list)
    def forward(self, x):
        return self.stack(x)
    def predict(self, x):
        out = F.sigmoid(self.forward(x))
        return out.round()
    def train_model(
        self,
        epochs,
        X_train,
        X_test,
        Y_train,
        Y_test,
        loss_fn=nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.Adam,
        optimizer_kwargs = {},
        print_epoch=10,
        header_epoch = 15,
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs = {'mode': 'max', 'factor': 0.1, 'patience': 10, 'threshold': 0.01, 'threshold_mode': 'rel', 'cooldown': 0, 'min_lr': 0, 'eps': 1e-08},
        device = 'cuda'
    ):  
        
        scaler = GradScaler("cuda")
        optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_kwargs)
        training_time = 0
        train_hist = torch.zeros(epochs, device=device)
        test_hist = torch.zeros(epochs, device=device)
        recall_hist = torch.zeros(epochs, device=device)
        
        cell_width = 20
        header_form_spec = f'^{cell_width}'
        
        epoch_inspection = {
            "Epoch": 0,
            "Epoch Time (s)": 0,
            "Inf Time (s/samp)": 0,
            "Training Loss": 0,
            "Test Loss ": 0,
            "Overfit (%)": 0,
            "Recall (%)": 0,
            "Δ Recall (%)": 0,
        }

        header_string = "|"
        for key in epoch_inspection.keys():
            header_string += (f"{key:{header_form_spec}}|")
        
        divider_string = '-'*len(header_string)
        if print_epoch:
            print(f'Training {self.__class__.__name__}\n')
            print(divider_string)
        max_recall = torch.zeros(1, device=device)            
        for epoch in range(epochs):
            begin_epoch = time.time()
            self.train()
            start_time = time.time()
            
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                Y_pred = self.forward(X_train)
                loss = loss_fn(Y_pred, Y_train)
            train_loss = loss.detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            training_time += time.time() - start_time
            
            train_hist[epoch] = train_loss
            
            
            self.eval()
            with torch.no_grad():
                begin_eval = time.time()
                test_loss = torch.zeros(1, device=device)
                
                out = self.forward(X_test)
                test_loss += loss_fn(out, Y_test)
                out = F.sigmoid(out).round()
                end_eval = time.time()
                eval_time = (end_eval - begin_eval)/len(Y_test)
                    
            test_hist[epoch] = test_loss
            
            true_pos = torch.sum(out * Y_test)
            false_neg = torch.sum((1 - out) * Y_test)
            recall = true_pos / (true_pos + false_neg)
            
            recall_hist[epoch] = recall
            
            scheduler.step(recall)
            
            end_epoch = time.time()
            if print_epoch and (epoch % print_epoch == 0 or epoch == epochs - 1) :
                if header_epoch and epoch % header_epoch == 0:
                    print(header_string)
                    print(divider_string)
                epoch_duration = end_epoch - begin_epoch
                overfit = 100 * (test_loss - train_loss) / train_loss
                d_accuracy = torch.zeros(1) if max_recall == 0 else 100 * (recall - max_recall) / max_recall
                if recall > max_recall and epoch > 0:
                    max_recall = recall
                
                epoch_inspection['Epoch'] = f'{epoch}'
                epoch_inspection['Epoch Time (s)'] = f'{epoch_duration:4f}'
                epoch_inspection['Training Loss'] = f'{train_loss.item():8f}'
                epoch_inspection['Test Loss '] = f'{test_loss.item():8f}'
                epoch_inspection['Overfit (%)'] = f'{overfit.item():4f}'
                epoch_inspection['Recall (%)'] = f'{recall.item()*100:4f}'
                epoch_inspection['Δ Recall (%)'] = f'{d_accuracy.item():4f}'
                epoch_inspection["Inf Time (s/samp)"] = f'{eval_time:4e}'
                for value in epoch_inspection.values():
                    print(f"|{value:^{cell_width}}", end='')
                print('|')
                print(divider_string)
            

        print(f'\nTraining Time: {training_time} seconds\n')
        
        self.train_hist = train_hist
        self.test_hist = test_hist
        self.accuracy_hist = recall_hist
        self.last_test = Y_test
        self.last_pred = self.predict(X_test)
    
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
        
class ConstantPredictor(AlarmNet):
    def __init__(self, val):
        self.val = val
    def predict(self, x):
        return self.forward(x).round()
    def train(self, epochs, X_train, X_test, Y_train, Y_test, alpha, loss_fn=nn.BCELoss(), print_epoch=500, optimizer=torch.optim.Adam):
        optimizer = optimizer(self.parameters(), lr=alpha)

        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = self.forward(X_train)
            loss = loss_fn(Y_pred, Y_train)
            loss.backward()
            optimizer.step()
            if epoch % print_epoch == 0:
                print(f'Epoch {epoch} Loss: {loss.item()}')
        Y_pred = self.predict(X_test)
        self.last_pred = Y_pred
        self.last_test = Y_test
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
    def print_results(self):
        super().print_results(self.get_results())
        
        
class SVMAlarmNet(AlarmNet):
    def __init__(self, kernel='rbf', C=1.0, degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=1e-3, cache_size=200, class_weight=None):
        super().__init__(pass_through=True)
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight)
    def train(self, 
              X_train: np.ndarray,
              X_test: np.ndarray,
              Y_train: np.ndarray,
              Y_test: np.ndarray,
              *args,
              **kwargs
        ):
        self.model.fit(X_train, Y_train)
        self.last_test = Y_test
        self.last_pred = self.model.predict(X_test).reshape(-1, 1)
    def get_results(self):
        self.last_results = {
            'accuracy': accuracy_score(self.last_test, self.last_pred),
            'precision': precision_score(self.last_test, self.last_pred),
            'recall': recall_score(self.last_test, self.last_pred),
            'f1': f1_score(self.last_test, self.last_pred),
            'confusion_matrix': confusion_matrix(self.last_test, self.last_pred),
            'classification_report': classification_report(self.last_test, self.last_pred)
        }
        return self.last_results
    def predict(self, x):
        return self.model.predict(x)
    def plot_confusion_matrix(self, title, color='Reds'):
        cm = self.last_results['confusion_matrix']
        disp = ConfusionMatrixDisplay.from_predictions(
            y_pred = self.last_pred,
            y_true = self.last_test,
            display_labels=["No Fire", "Fire"],
            cmap = color
        )
        plt.title(title)
class AlarmNetNoCuda(nn.Module):

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
        optimizer = optimizer(self.parameters(), lr=alpha)

        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = self.forward(X_train)
            loss = loss_fn(Y_pred, Y_train)
            loss.backward()
            optimizer.step()
            if epoch % print_epoch == 0:
                print(f'Epoch {epoch} Loss: {loss.item()}')
        Y_pred = self.predict(X_test)
        self.last_pred = Y_pred
        self.last_test = Y_test
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
            if key != 'confusion_matrix':
                print(f'{key.capitalize()}: {value}')
            else:
                print(f'{key.capitalize()}:\n{value}')
    
    def train_and_print(self, epochs, X_train, X_test, Y_train, Y_test, alpha):
        Y_pred = self.train(epochs, X_train, X_test, Y_train, Y_test, alpha).cpu().detach().numpy().round().astype(int)
        self.print_results(Y_test, Y_pred)

        