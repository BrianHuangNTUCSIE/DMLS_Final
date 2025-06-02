import pandas as pd
import matplotlib.pyplot as plt

# File names and labels
files = {
    "cifar10_LR.csv": "Logistic Regression",
    "cifar10_DNN.csv": "DNN",
    "cifar10_CNN.csv": "CNN",
    "cifar10_DenseNet.csv": "DenseNet"
}

colors = ["blue", "green", "red", "purple"]

plt.figure(figsize=(10, 6))

for (file, label), color in zip(files.items(), colors):
    df = pd.read_csv(file)
    plt.plot(df["round"], df["accuracy"], label=label, color=color)

plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Round for Different Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("no_data_sharing.png")