import os
from matplotlib import pyplot as plt 
import torch


 # Save Model
def save_model(model, acc_test, args):
    PATH = "./model"; caption = "/{}_epoch{}_lr{}_{}_acc{}".format(args['model'], args['epoch'], args['lr'], args['path'], acc_test)
    if not os.path.exists(PATH): os.mkdir(PATH)
    torch.save(model.state_dict(), PATH + caption)
    
    return model, acc_test # return the model with weight selected by best performance 

def plot_loss_acc(training_stats, acc, args):
    loss_train, acc_train, loss_valid, acc_valid = [], [], [], []


    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(len(loss_train)), loss_train, color='tab:blue', label="Train")
    ax1.plot(range(len(loss_valid)), loss_valid, color='tab:orange', label="Validation")
    ax1.set_title("Training and Validation Losses Across Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross-entropy Loss")
    ax1.legend(loc="upper right", frameon=False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(len(acc_train)), acc_train, color='tab:blue', label="Train")
    ax2.plot(range(len(acc_valid)), acc_valid, color='tab:orange', label="Validation")
    ax2.set_title("Training and Validation Accuracies Across Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy %")
    ax2.legend(loc="upper right", frameon=False)

    path = "./images/"; caption = "{}_acc{}_{}_epoch{}_lr{}_freeze{}.png".format(args['model'], acc, args['path'], args['epoch'], args['lr'],args['pt_ft']); 
    if not os.path.exists(path): os.mkdir(path)
    plt.savefig(path + caption)