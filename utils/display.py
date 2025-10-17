import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


# debugging visualization
def display_results(all_labels, all_preds):
    labels = np.array(all_labels)
    preds = np.array(all_preds)

    plt.hist([preds[labels==0], preds[labels==1]], bins=40, label=['normal', 'abnormal'], color=["#3989c3", "#f47a7a"], alpha=0.7)
    plt.legend()
    # plt.show()
    os.makedirs("runs/fig", exist_ok=True)
    plt.tight_layout()
    plt.savefig("runs/fig/" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")
    plt.close()

    # plt.figure()
    # plt.scatter(np.arange(len(preds)), preds, c=labels, cmap='coolwarm', alpha=0.6, label='preds')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Predicted Score')
    # plt.title('Scatter Plot of Predictions')
    # plt.colorbar(label='Label (0=normal, 1=abnormal)')
    # plt.show()