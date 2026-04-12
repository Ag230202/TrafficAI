import torch
import torch.nn as nn
import os

class DQNNetwork(nn.Module):
    def __init__(self, input_dim=10, output_dim=4):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.out(x)

class DQNAgent:
    """
    Loads pre-trained weights and exposes predict() for live traffic signal control.
    """
    def __init__(self, weights_path="dqn_weights.pt"):
        self.device = torch.device("cpu") # Force CPU for fast inference ~1ms
        self.model = DQNNetwork().to(self.device)
        self.is_loaded = False
        
        if os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            self.model.eval()
            self.is_loaded = True
            print(f"[DQNAgent] Successfully loaded offline weights from {weights_path}")
        else:
            print(f"[DQNAgent] WARNING: Weights file {weights_path} not found. Using untrained weights!")
            self.model.eval()

    def get_q_values(self, state_vector):
        """
        Returns all 4 Q-values as a list of floats.
        state_vector: list of 10 floats
        """
        tensor_state = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(tensor_state)
        return q_values.squeeze(0).tolist()

    def predict(self, state_vector):
        """
        Returns argmax of Q-values (0, 1, 2, or 3) representing the best phase.
        state_vector: list of 10 floats
        """
        q_vals = self.get_q_values(state_vector)
        return q_vals.index(max(q_vals))

# Helper singleton instance if needed
# agent = DQNAgent()
