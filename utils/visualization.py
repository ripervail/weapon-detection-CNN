'''
Functions to visualize stuff from loss to accuracy across different models
'''

import matplotlib.pyplot as plt
import os
import random

def plot_train_test_stats(train_stats, test_stats, metric):
    '''
    Plot train/test loss or accuracy. Best used for 1 model
    - metric: "Loss" or "Accuracy". It is for the graph labels
    '''
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the train and test losses
    ax.plot(train_stats, label=f"Train {metric}")
    ax.plot(test_stats, label=f"Test {metric}")

    # Set the title and axis labels
    ax.set_title(f"Train and Test {metric} Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)

    # Add a legend
    ax.legend()

    # Return the plot object
    return fig

def plot_stats_comparison(model_stats, model_names, metric):
    '''
    Plots model stats across different models
    - metric: "Test" or "Train" with "Loss" or "Accuracy". It is for the graph labels
    '''
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Plot the test losses for each model on the same graph
    for i, loss in enumerate(model_stats):
        ax.plot(loss, label=model_names[i])

    # Set the title and axis labels
    ax.set_title(f"{metric} Comparison Across Models")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)

    # Add a legend
    ax.legend()

    return fig

def get_random_pic(modifier1, modifier2, modifier3):
    '''
    - modifier1: "data" or "rembg-data"
    - modifier2: "train" or "test"
    - modifier3: "norm" or "weap"
    '''
    file_path = f"/illumina/scratch/deep_learning/jneo1/cs5242_project/weapon-detection-CNN/{modifier1}/{modifier2}/{modifier3}/"
    files = os.listdir(file_path)
    random_file = file_path + random.choice(files)
    return random_file