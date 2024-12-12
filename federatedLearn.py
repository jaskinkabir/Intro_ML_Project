import torch
import seaborn as sns
import matplotlib.pyplot as plt
from alarmnetclass import AlarmNet 

class FederatedLearning:
    def __init__(self, global_model, n_clients):
        self.global_model = global_model
        self.n_clients = n_clients
        self.client_data = []

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
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            Y_pred = local_model.forward(X_client)
            loss = loss_fn(Y_pred, Y_client)
            loss.backward()
            optimizer.step()

            # Report every 25 epochs
            if (epoch + 1) % 25 == 0:
                print(f"Client {client_idx + 1} - Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        return local_model.state_dict()

    def aggregate_models(self, client_states):
        global_state = self.global_model.state_dict()
        for key in global_state.keys():
            global_state[key] = torch.stack([client_state[key] for client_state in client_states], dim=0).mean(dim=0)
        self.global_model.load_state_dict(global_state)

    def display_network_weights(self, model):
        print("\n--- Neural Network Weights ---")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data.numpy()}")

    def plot_weight_heatmap(self, model):
        plt.figure(figsize=(10, 8))
        named_params = list(model.named_parameters())
        
        for i, (name, param) in enumerate(named_params):
            if 'weight' in name:  
                weights = param.data.numpy()
                plt.subplot(len(named_params), 1, i + 1)
                sns.heatmap(weights, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                plt.title(f"Heatmap for {name}")
        plt.tight_layout()
        plt.show()

    def federated_training(self, epochs_per_round, lr, n_rounds):
        for round in range(n_rounds):
            print(f"--- Federated Training Round {round + 1} ---")
            client_states = []
            for client_idx in range(self.n_clients):
                client_state = self.local_train(client_idx, epochs_per_round, lr)
                client_states.append(client_state)
            self.aggregate_models(client_states)

            self.display_network_weights(self.global_model)
            self.plot_weight_heatmap(self.global_model)

    def evaluate_global_model(self, X_test, Y_test):
        self.global_model.train(epochs=0, X_train=None, X_test=X_test, Y_train=None, Y_test=Y_test, alpha=0)  # Dummy call
        results = self.global_model.get_results()
        self.global_model.print_results(results)
