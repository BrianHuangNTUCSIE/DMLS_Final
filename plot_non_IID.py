import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_non_IID.py <dataset> <model>")
        sys.exit(1)

    dataset = sys.argv[1]
    model = sys.argv[2]

    deltas = [0.0, 0.2, 0.4, 1.0]
    colors = {0.0: "red", 0.2: "blue", 0.4: "orange", 1.0: "black"}  # Color mapping
    result_dir = "./result2"
    plt.figure(figsize=(10, 6))

    for delta in deltas:
        file_path = os.path.join(result_dir, f"non_IID_{dataset}_{model}_{delta}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue

        df = pd.read_csv(file_path)
        if 'round' not in df.columns or 'accuracy' not in df.columns:
            print(f"Warning: File {file_path} does not contain 'round' and 'accuracy' columns.")
            continue

        plt.plot(df['round'], df['accuracy'], label=f"δ = {delta}", color=colors[delta])

    plt.title(f"Accuracy vs Round for {model} on {dataset}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dataset}_plot/non_IID_{model}.png")

if __name__ == "__main__":
    main()