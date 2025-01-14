import torch
import numpy as np
from rich import print
from rich.traceback import install
from .train_exp_1b import main_t5_lora

install()

def run_experiment_t5_lora(n_runs=3):
    """
    Runs Experiment 2 (length split) with T5 + LoRA,
    using the main_t5_lora function from your existing LoRA script.
    """
    # Hyperparameters (you can adjust them as needed)
    hyperparams = {
        "learning_rate": 5e-4,
        "batch_size": 64,
        "epochs": 4, 
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Paths for the SCAN length split
    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    exp_name = "length"

    results = {}

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting T5-LoRA run {run+1}/{n_runs} on the length split with seed {seed}")
        print("=" * 70)

        # Call the existing function from train_exp_1_t5_lora.py
        # Adjust the call signature if your function differs
        model, token_acc, seq_acc = main_t5_lora(
            train_path=train_path,
            test_path=test_path,
            model_suffix=f"T5_LoRA_{exp_name}",
            hyperparams=hyperparams,
            random_seed=seed,
        )

        results[f"run_{run}"] = (token_acc, seq_acc)

    # Convert results dict to list of (token_acc, seq_acc)
    all_accuracies = list(results.values())  # e.g. [(0.0, 0.1), (0.05, 0.2), ...]
    # Ensure floats
    all_accuracies = [(float(a), float(s)) for (a, s) in all_accuracies]

    # Compute mean and std along axis=0 => (token_acc, seq_acc)
    mean = np.mean(all_accuracies, axis=0)
    std = np.std(all_accuracies, axis=0)

    # Print a summary like in experiment 1
    print("\nAggregated Stats for All Runs:")
    print("=" * 50)
    print(f"Mean Token Acc : {mean[0]:.4f} ± {std[0]:.4f}")
    print("Individual runs: " + ", ".join(f"{acc[0]:.4f}" for acc in all_accuracies))
    print(f"Mean Seq  Acc  : {mean[1]:.4f} ± {std[1]:.4f}")
    print("Individual runs: " + ", ".join(f"{acc[1]:.4f}" for acc in all_accuracies))
    print("-" * 50)


if __name__ == "__main__":
    run_experiment_t5_lora()
