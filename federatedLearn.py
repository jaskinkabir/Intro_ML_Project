import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from alarmnetclass import AlarmNetNoCuda as AlarmNet
from sklearn.metrics import classification_report, confusion_matrix

class FederatedLearning:
    def __init__(self, global_model, n_clients):
        self.global_model = global_model
        self.n_clients = n_clients
        self.client_data = []
        self.training_losses = []  # To store training losses
        self.test_losses = []  # To store test losses

    def split_data(self, X, Y):
        split_size = len(X) // self.n_clients
        self.client_data = []
        for i in range(self.n_clients):
            start = i * split_size
            end = (i + 1) * split_size if i != self.n_clients - 1 else len(X)
            self.client_data.append((X[start:end], Y[start:end]))

    def local_train(self, client_idx, epochs, lr):
        X_client, Y_client = self.client_data[client_idx]
        local_model = AlarmNet(
            num_features=self.global_model.stack[0].in_features,
            hidden_layers=[32, 16, 8],
        )
        
        local_model.load_state_dict(self.global_model.state_dict())  # Copy global weights
        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        epoch_losses = []  # Track the loss per epoch
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = local_model.forward(X_client)
            loss = loss_fn(Y_pred, Y_client)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())  # Store the loss for each epoch

            if (epoch + 1) % 100 == 0:
                print(f"Client {client_idx + 1} - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        # Store training loss for plotting
        self.training_losses.append(epoch_losses)

        # Predict using the local model
        Y_pred_class = (Y_pred > 0.5).float()  # Convert to binary predictions
        report = classification_report(Y_client, Y_pred_class, output_dict=True)
        cm = confusion_matrix(Y_client, Y_pred_class)  # Compute confusion matrix

        print(f"Client {client_idx + 1} training completed.")
        print(f"Classification Report for Client {client_idx + 1}:\n{report}")
        print(f"Confusion Matrix for Client {client_idx + 1}:\n{cm}")

        return report, cm, local_model.state_dict()

    def aggregate_models(self, client_states):
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_state[key] for client_state in client_states], dim=0).mean(dim=0)
        self.global_model.load_state_dict(global_state)

    def evaluate_global_model(self, X_test, Y_test, conf_title):
        self.global_model.train(epochs=0, X_train=None, X_test=X_test, Y_train=None, Y_test=Y_test, alpha=0)  # Dummy call
        results = self.global_model.get_results()
        self.global_model.print_results(results)

        # After evaluating the global model, show the confusion matrix graph
        global_pred = self.global_model.forward(X_test)
        global_pred_class = (global_pred > 0.5).float()
        loss_fn = torch.nn.BCELoss()
        test_loss = loss_fn(global_pred, Y_test).item()  # Calculate test loss
        self.test_losses.append(test_loss)  # Store test loss for plotting

        cm_global = confusion_matrix(Y_test, global_pred_class)
        print(f"Confusion Matrix for Global Model:\n{cm_global}")
        self.plot_confusion_matrix(cm_global, conf_title)  # Plot confusion matrix for global model

    def plot_loss(self):
        # Plot training and test loss
        plt.figure(figsize=(10, 6))
        
        # Plot Training Losses
        for i, client_losses in enumerate(self.training_losses):
            plt.plot(client_losses, label=f"Client {i + 1} - Training Loss")
        
        # Plot Test Loss
        plt.plot(self.test_losses, label="Global Model - Test Loss", color="red", linestyle="--")
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Over Time')
        plt.legend()
        plt.show()

    def federated_training(self, epochs_per_round, lr, n_rounds, X_test, Y_test, conf_title):
        for round in range(n_rounds):
            print(f"--- Federated Training Round {round + 1} ---")
            client_reports = []  # Store reports of each client
            client_cm = []  # Store confusion matrices
            client_states = []
            for client_idx in range(self.n_clients):
                report, cm, client_state = self.local_train(client_idx, epochs_per_round, lr)
                client_reports.append(report)
                client_cm.append(cm)
                client_states.append(client_state)
            self.aggregate_models(client_states)

            # Evaluate global model after each round
            self.evaluate_global_model(X_test, Y_test, conf_title=conf_title)

            # Display results for each client
            print("\n--- Client Reports ---")
            for report in client_reports:
                print(report)

            print("\n--- Client Confusion Matrices ---")
            for idx, cm in enumerate(client_cm):
                print(f"Confusion Matrix for Client {idx + 1}:\n{cm}")

        # Plot Losses after training
        self.plot_loss()

    def dump_data_to_csv(self, X, Y, output_dir='client_data'):
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Split the data among clients
        self.split_data(X, Y)

        # Save each client's data to a separate CSV file
        for client_idx, (X_client, Y_client) in enumerate(self.client_data):
            client_data = pd.DataFrame(X_client)
            client_data['Label'] = Y_client  # Adding labels as a column
            client_data.to_csv(f"{output_dir}/client_{client_idx + 1}_data.csv", index=False)
            print(f"Data for client {client_idx + 1} saved to {output_dir}/client_{client_idx + 1}_data.csv")

    def plot_confusion_matrix(self, cm, title):
        plt.figure(figsize=(6, 5))
        plt.title(title)
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire']) # Dont think this is super helpful
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
