import os
import pickle
from utils.visualization import *
import matplotlib.pyplot as plt

# Options to generate the graph
date = "11"
month = "4"
model = "ConvNet2.100"

# Load all the relevant stats, if they exist
with open(f"model_training_stats/{month}-{date}_{model}_masked.train_loss.pkl", "rb") as f:
    train_loss = pickle.load(f)
with open(f"model_training_stats/{month}-{date}_{model}_masked.test_loss.pkl", "rb") as f:
    test_loss = pickle.load(f)
with open(f"model_training_stats/{month}-{date}_{model}_masked.train_acc.pkl", "rb") as f:
    train_acc = pickle.load(f)
with open(f"model_training_stats/{month}-{date}_{model}_masked.test_acc.pkl", "rb") as f:
    test_acc = pickle.load(f)

fig1 = plot_train_test_stats(train_loss, test_loss, metric="Loss")
fig1.savefig(f"figures/{month}-{date}_{model}_loss.png")

fig2 = plot_train_test_stats(train_acc, test_acc, metric="Accuracy")
fig2.savefig(f"figures/{month}-{date}_{model}_acc.png")