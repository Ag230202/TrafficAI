"""
run_ai_priority.py
------------------
Dedicated entrypoint for running the AI-driven traffic prioritization.
This script uses the DQN model (dqn_weights.pt) trained on historical data.

Usage:
    python run_ai_priority.py
"""

import os
from example_usage import main as run_pipeline_main, CUSTOM_SIGNAL_CONFIG

def main():
    # 1. Force Enable DQN in the configuration
    print("\n[AI Setup] Enabling DQN Prioritization Mode...")
    CUSTOM_SIGNAL_CONFIG["use_dqn"] = True
    
    # 2. Check for weights
    if not os.path.exists("dqn_weights.pt"):
        print("[ERROR] dqn_weights.pt not found. Please run training first:")
        print("        python train_dqn.py")
        return

    print("[AI Setup] Weights found. Loading policy...")
    
    # 3. Execute the standard integrated pipeline
    # We keep all logging active so you can audit the AI's decisions.
    print("[AI Setup] Logging active:")
    print("  • ai_signal_log.txt   (Signal phase decisions)")
    print("  • ai_traffic_log.csv  (State-action-reward data)")
    
    try:
        # Override log paths before running to keep AI logs separate from training logs
        import example_usage
        from data_logger import DATA_LOGGER_CONFIG
        
        example_usage.SIGNAL_LOG_FILE = "ai_signal_log.txt"
        DATA_LOGGER_CONFIG["log_path"] = "ai_traffic_log.csv"
        
        run_pipeline_main()
    except KeyboardInterrupt:
        print("\n[User Exit] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")

if __name__ == "__main__":
    main()
