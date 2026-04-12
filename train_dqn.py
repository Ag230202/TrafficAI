import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse

# ── MODEL ARCHITECTURE ───────────────────────────────────────────
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

# ── DATASET PREPARATION ──────────────────────────────────────────
def load_and_preprocess_data(csv_path: str):
    """
    Reads traffic_log.csv and builds (state, action, reward, next_state) tuples.
    Detects sequence resets by checking if frame_id drops.
    """
    df = pd.read_csv(csv_path)

    # State features
    feature_cols = [
        "count_left", "count_bottom", "count_right", "count_top",
        "wait_left", "wait_bottom", "wait_right", "wait_top",
        "current_phase", "time_in_phase"
    ]
    
    # Robustness: If header is wrong, re-read with explicit names
    if "frame_id" not in df.columns:
        names = ["timestamp", "frame_id"] + feature_cols + ["action", "reward", "next_state"]
        df = pd.read_csv(csv_path, header=None, names=names[:len(df.columns)])
    
    states = []
    actions = []
    rewards = []
    next_states = []

    # Iterate row by row to detect sequence boundaries
    for i in range(len(df) - 1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Reset sequence if frame_id jumps backwards
        if next_row["frame_id"] <= curr_row["frame_id"]:
            continue
            
        s = curr_row[feature_cols].values.astype(np.float32)
        a = int(curr_row["action"])
        r = float(curr_row["reward"])
        s_next = next_row[feature_cols].values.astype(np.float32)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

    print(f"Loaded {len(states)} valid transitions from {len(df)} total rows.")
    
    if len(states) == 0:
         raise ValueError("No valid transitions found. Run the pipeline to collect more data.")

    return (
        torch.tensor(np.array(states)),
        torch.tensor(np.array(actions), dtype=torch.int64),
        torch.tensor(np.array(rewards), dtype=torch.float32),
        torch.tensor(np.array(next_states))
    )

# ── TRAINING LOOP ────────────────────────────────────────────────
def train(csv_path="traffic_log.csv", epochs=100, batch_size=32, save_path="dqn_weights.pt"):
    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} not found. Run the pipeline first to generate data.")
        return

    # 1. Load Data
    print(f"Loading data from {csv_path}...")
    states, actions, rewards, next_states = load_and_preprocess_data(csv_path)
    
    dataset = TensorDataset(states, actions, rewards, next_states)
    
    # 80/20 train/test split
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size
    
    if test_size == 0 or train_size == 0:
         # extremely small dataset (e.g. initial snapshot)
         train_size = len(dataset)
         test_size = 0
         train_dataset = dataset
         test_dataset = None
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Setup Model & Optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    policy_net = DQNNetwork().to(device)
    target_net = DQNNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.99  # discount factor

    # 3. Epoch Loop
    print("Starting training...")
    for epoch in range(epochs):
        policy_net.train()
        total_loss = 0.0
        
        for batch_idx, (b_states, b_actions, b_rewards, b_next_states) in enumerate(train_loader):
            b_states = b_states.to(device)
            b_actions = b_actions.to(device)
            b_rewards = b_rewards.to(device)
            b_next_states = b_next_states.to(device)

            # Current Q-values
            q_values = policy_net(b_states)
            q_values_for_actions = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(-1)

            # Target Q-values
            with torch.no_grad():
                max_next_q_values = target_net(b_next_states).max(1)[0]
                target_q_values = b_rewards + (gamma * max_next_q_values)

            # Gradient Descent
            loss = loss_fn(q_values_for_actions, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Soft update target network periodically
        if epoch % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f}")

    # 4. Save best model
    torch.save(policy_net.state_dict(), save_path)
    print(f"Training complete. Weights saved to {save_path}")

    # 5. Evaluate on Test Set
    if test_dataset:
        policy_net.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for b_states, b_actions, _, _ in test_loader:
                b_states = b_states.to(device)
                b_actions = b_actions.to(device)
                q_vals = policy_net(b_states)
                preds = q_vals.argmax(dim=1)
                correct += (preds == b_actions).sum().item()
                total += b_actions.size(0)
        
        match_rate = (correct / total) * 100
        print(f"Evaluation against Rule-based Policy (Test Set): {match_rate:.2f}% Match")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="traffic_log.csv", help="Path to training data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    args = parser.parse_args()
    
    train(csv_path=args.csv, epochs=args.epochs)
