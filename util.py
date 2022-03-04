import os, torch
from matplotlib import pyplot as plt 
import numpy as np
from sklearn.metrics import matthews_corrcoef 

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def mcc_score(preds, true_labels):
    '''Computes mcc score
    @param
        preds: Raw model output float 2D array. Shape (num_samples, num_classes)
               Convert to prediction using argmax
        true_labels: Ground truth int 2D array. Shape (num_samples)
    @return 
        mcc
    '''
    preds = np.argmax(preds, axis=1).flatten()
    mcc = matthews_corrcoef(true_labels, preds)
    return mcc


 # Save Model
def save_model(model, mcc, args):
    PATH = "./models/"
    caption = "{}_{}_{}_mcc{:.2f}.pt".format(args['model'], args['dataset'] ,args['log_path'], mcc)
    if not os.path.exists(PATH): os.mkdir(PATH)
    torch.save(model.state_dict(), PATH + caption)

def plot_loss_acc(training_stats, mcc, args):
    loss_train, acc_train, loss_valid, acc_valid = [], [], [], []
    for stat in training_stats:
        loss_train.append(stat['train_loss'])
        acc_train.append(stat['train_acc'])
        loss_valid.append(stat['valid_loss'])
        acc_valid.append(stat['valid_acc'])


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

    PATH = "./images/"
    caption = "{}_{}_{}_mcc{:.2f}_epoch{}_lr{}.png".format(args['model'], args['dataset'] ,args['log_path'], mcc, args['epoch'], args['lr'])
    if not os.path.exists(PATH): os.mkdir(PATH)
    plt.savefig(PATH + caption)

def plot_confusion_matrix(preds, true_labels):
    ''' TODO: Plot confusion matrix. Code snipped right now is from original matthews score calculation
    @param
        preds: Raw model output float 2D array. Shape (num_samples, num_classes)
               Convert to prediction using argmax on axis 1
        true_labels: Ground truth int 2D array. Shape (num_samples)
    '''
    # matthews_set = []
    # for i in range(len(true_labels)):
    #     pred_labels_i = np.argmax(preds[i], axis=1).flatten()
    #     # Calculate and store the coef for this batch.  
    #     matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
    #     matthews_set.append(matthews)

    # ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)
    # plt.title('MCC Score per Batch')
    # plt.ylabel('MCC Score (-1 to +1)')
    # plt.xlabel('Batch #')
    # plt.show()
    raise NotImplementedError()