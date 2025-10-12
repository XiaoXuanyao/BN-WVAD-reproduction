import matplotlib.pyplot as plt
import numpy as np


# debugging visualization
def display_results(all_labels, all_preds):
    labels = np.array(all_labels)
    preds = np.array(all_preds)

    plt.hist([preds[labels==0], preds[labels==1]], bins=20, label=['normal', 'abnormal'])
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.scatter(np.arange(len(preds)), preds, c=labels, cmap='coolwarm', alpha=0.6, label='preds')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Predicted Score')
    # plt.title('Scatter Plot of Predictions')
    # plt.colorbar(label='Label (0=normal, 1=abnormal)')
    # plt.show()