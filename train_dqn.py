import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import argparse
import copy

# ── MODEL ARCHITECTURE ───────────────────────────────────────────
class DQNNetwork(nn.Module):
    """Deeper network with Batch Normalisation for stable training."""
    def __init__(self, input_dim=10, output_dim=4):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ── DATASET PREPARATION ──────────────────────────────────────────
def load_and_preprocess_data(csv_path: str):
    """
    Reads traffic_log.csv and builds (state, action, reward, next_state) tuples.
    Rewards are z-score normalised to prevent Q-value explosion and loss divergence.
    Features are also normalised so all inputs are on the same scale.
    """
    df = pd.read_csv(csv_path)

    feature_cols = [
        "count_left", "count_bottom", "count_right", "count_top",
        "wait_left",  "wait_bottom",  "wait_right",  "wait_top",
        "current_phase", "time_in_phase"
    ]

    # Robustness: re-read with explicit names if header is missing
    if "frame_id" not in df.columns:
        names = ["timestamp", "frame_id"] + feature_cols + ["action", "reward", "next_state"]
        df = pd.read_csv(csv_path, header=None, names=names[:len(df.columns)])

    states, actions, rewards, next_states = [], [], [], []

    for i in range(len(df) - 1):
        curr_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        # Skip sequence boundary resets
        if next_row["frame_id"] <= curr_row["frame_id"]:
            continue

        s      = curr_row[feature_cols].values.astype(np.float32)
        a      = int(curr_row["action"])
        r      = float(curr_row["reward"])
        s_next = next_row[feature_cols].values.astype(np.float32)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_next)

    print(f"Loaded {len(states)} valid transitions from {len(df)} total rows.")

    if len(states) == 0:
        raise ValueError("No valid transitions found. Run the pipeline to collect more data.")

    rewards_arr = np.array(rewards, dtype=np.float32)

    # ── Z-score reward normalisation ─────────────────────────────
    # Prevents Q-value explosion (the primary cause of loss divergence)
    r_mean, r_std = rewards_arr.mean(), rewards_arr.std() + 1e-8
    rewards_arr = (rewards_arr - r_mean) / r_std
    print(f"Reward stats  →  mean: {r_mean:.2f}, std: {r_std:.2f}  (normalised to ~N(0,1))")

    # ── Feature normalisation ─────────────────────────────────────
    states_arr      = np.array(states,      dtype=np.float32)
    next_states_arr = np.array(next_states, dtype=np.float32)
    feat_mean = states_arr.mean(axis=0)
    feat_std  = states_arr.std(axis=0) + 1e-8
    states_arr      = (states_arr      - feat_mean) / feat_std
    next_states_arr = (next_states_arr - feat_mean) / feat_std

    return (
        torch.tensor(states_arr),
        torch.tensor(np.array(actions), dtype=torch.int64),
        torch.tensor(rewards_arr),
        torch.tensor(next_states_arr),
    )


# ── TRAINING LOOP ────────────────────────────────────────────────
def train(csv_path="traffic_log.csv", epochs=500, batch_size=64, save_path="dqn_weights.pt"):
    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} not found. Run the pipeline first to generate data.")
        return

    # 1. Load & Preprocess Data
    print(f"Loading data from {csv_path}...")
    states, actions, rewards, next_states = load_and_preprocess_data(csv_path)

    dataset = TensorDataset(states, actions, rewards, next_states)

    # 80/20 train/test split
    train_size = int(0.8 * len(dataset))
    test_size  = len(dataset) - train_size

    if test_size == 0 or train_size == 0:
        train_size    = len(dataset)
        test_size     = 0
        train_dataset = dataset
        test_dataset  = None
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Setup Model & Optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_net = DQNNetwork().to(device)
    target_net = DQNNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Lower LR + cosine decay for stable long training
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    mse_loss = nn.MSELoss()
    gamma    = 0.99

    # ── Early stopping & best-model tracking ─────────────────────
    best_loss        = float("inf")
    best_weights     = copy.deepcopy(policy_net.state_dict())
    patience         = 60   # epochs to wait without loss improvement
    no_improve_count = 0

    # Target network update frequency
    target_update_freq = 5

    print("Starting training...")
    print(f"{'Epoch':>8} | {'DQN Loss':>12} | {'LR':>10}")
    print("-" * 40)

    for epoch in range(epochs):
        policy_net.train()
        total_loss = 0.0

        for b_states, b_actions, b_rewards, b_next_states in train_loader:
            b_states      = b_states.to(device)
            b_actions     = b_actions.to(device)
            b_rewards     = b_rewards.to(device)
            b_next_states = b_next_states.to(device)

            # Current Q-values for taken actions
            q_values             = policy_net(b_states)
            q_values_for_actions = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(-1)

            # Target Q-values via Bellman equation
            with torch.no_grad():
                max_next_q = target_net(b_next_states).max(1)[0]
                target_q   = b_rewards + gamma * max_next_q

            # Pure DQN (Bellman) loss — genuine RL, no supervision cheat
            loss = mse_loss(q_values_for_actions, target_q)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping: prevents exploding gradients
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Soft-copy target network periodically
        if epoch % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Save best weights based on training loss
        if avg_loss < best_loss:
            best_loss        = avg_loss
            best_weights     = copy.deepcopy(policy_net.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == epochs - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1:>5}/{epochs} | Loss: {avg_loss:>12.4f} | LR: {lr_now:.2e}")

        # Early stopping
        if no_improve_count >= patience:
            print(f"\n[Early Stop] No loss improvement for {patience} epochs. Stopping at epoch {epoch+1}.")
            break

    # 3. Restore & Save Best Weights
    policy_net.load_state_dict(best_weights)
    torch.save(best_weights, save_path)
    print(f"\nTraining complete. Best DQN loss: {best_loss:.4f}")
    print(f"Weights saved to {save_path}")

    # 4. Evaluate on Test Set
    if test_dataset:
        policy_net.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct  = 0
        total    = 0
        total_q_max        = 0.0
        ai_action_counts   = {0: 0, 1: 0, 2: 0, 3: 0}
        rule_action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        with torch.no_grad():
            for b_states, b_actions, _, _ in test_loader:
                b_states  = b_states.to(device)
                b_actions = b_actions.to(device)
                q_vals    = policy_net(b_states)
                preds     = q_vals.argmax(dim=1)
                max_q     = q_vals.max(dim=1)[0]

                correct     += (preds == b_actions).sum().item()
                total       += b_actions.size(0)
                total_q_max += max_q.sum().item()

                for p, a in zip(preds.cpu().numpy(), b_actions.cpu().numpy()):
                    ai_action_counts[p]   += 1
                    rule_action_counts[a] += 1

        match_rate     = (correct / total) * 100
        avg_confidence = total_q_max / total

        print("\n" + "=" * 50)
        print("          === ADVANCED EVALUATION METRICS ===")
        print("=" * 50)
        print(f"1. Rule-Based Match Rate:      {match_rate:.2f}%")
        print(f"2. Mean AI Confidence (Q-val): {avg_confidence:.4f}")
        print("\n3. Action Selection Distribution (Rule-Based vs AI):")
        for i in range(4):
            print(f"   Lane {i}:  Rule={rule_action_counts[i]:<5} | AI={ai_action_counts[i]:<5}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str, default="traffic_log.csv",
                        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Maximum training epochs")
    args = parser.parse_args()

    train(csv_path=args.csv, epochs=args.epochs)
