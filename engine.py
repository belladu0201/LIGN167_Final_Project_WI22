'''
Train and Evaluate Model
'''
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef 
from model import get_bert_custom, get_bert_pretrained
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time, datetime
from tqdm import tqdm
import random
import data
from util import *

writer = SummaryWriter()

def prepare_model(device, train_length, args=None):
    total_steps = train_length * args['epoch']
    
    if args['model'] == 'pretrained':
        model = get_bert_pretrained(args)
    elif args['model'] == 'custom':
        model = get_bert_custom(args)
        
    # NOTE:Optim from huggingface (not pytorch). 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                    lr = args['lr'], # args.learning_rate - default is 5e-5, our notebook had 2e-5
                    eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    return model, scheduler, optimizer


def prepare_data(args):
    # @ return train, val, test dataloaders
    PATH = {'twitter': 'datasets/twitter.csv',  # 'hate_speech' col: # of users who marked it as hateful
            'gab': 'datasets/gab.csv',          # Total 45601. 
            'reddit': 'datasets/reddit.csv',
            'parler': 'datasets/parler_pretrain.ndjson'
            }
    if not PATH[args['dataset']]: raise NotImplementedError()
    
    if args['dataset'] == 'twitter':
        df = data.parse_twitter(PATH['twitter'], args['sample'], args['verbose'])
    elif args['dataset'] == 'gab':
        df = data.parse_reddit_gab(PATH['gab'], args['sample'], args['verbose'])
    elif args['dataset'] == 'reddit':
        df = data.parse_reddit_gab(PATH['reddit'], args['sample'], args['verbose'])
    elif args['dataset'] == 'parler':
        # TODO: Parler ds pipeline
        raise NotImplementedError()
    else:
        raise NotImplementedError('No such dataset')

    input_ids, attention_masks, labels = data.tokenize_dataframe(df, args['verbose'], args['model'])
    train_loader, valid_loader, test_loader \
        = data.create_dataloaders(input_ids, attention_masks, labels, args['bz'])

    return train_loader, valid_loader, test_loader, df


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

def train_model(model, scheduler, optimizer, dataloaders, args):

    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    model = model.to(device) # TODO: Uncomment this when have GPU access

    training_stats = [] # Store training and validation loss, validation accuracy, and timings.
    total_t0 = time.time() # Measure the total training time for the whole run.

    epochs = args['epoch'] # The BERT authors recommend between 2 and 4.
    print('Training...')
    for epoch_i in range(0, epochs):
        print('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))
        t0 = time.time()
        total_train_loss = 0

        model.train()

        for batch in tqdm(dataloaders[0]):
            # if step % 40 == 0 and not step == 0:
            #     elapsed = format_time(time.time() - t0)
            #     print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloaders[0]), elapsed))

            # batch[0] = input ids, batch[1] = attention masks, batch[2] = labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            result = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            loss, logits = result.loss, result.logits

            label_ids = b_labels.to('cpu').numpy()

            total_train_loss += loss.item()

            loss.backward()

            # Clip the norm of the gradients to 1.0. To prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(dataloaders[0]) # avg. loss over all batches.

        writer.add_scalar("Loss/train", avg_train_loss, epoch_i)
        
        # Measure how long this epoch took.
        # training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        # print("  Training epoch took: {:}".format(training_time))
        print("\n Validation...")

        # t0 = time.time()

        model.eval() # dropout layers behave differently during evaluation.
        total_eval_accuracy = 0
        total_eval_loss = 0
        
        for batch in tqdm(dataloaders[1]): # Validation data for one epoch
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():        
                result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss, logits = result.loss, result.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Accuracy for this batch of test sentences, accumulated over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
            

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(dataloaders[1])
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dataloaders[1])

        writer.add_scalar("Loss/val", avg_train_loss, epoch_i)
        writer.add_scalar("Accuracy/val",avg_val_accuracy, epoch_i)
        
        # Measure how long the validation run took.
        # validation_time = format_time(time.time() - t0)
        
        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        # print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy
                # 'Training Time': training_time,
                # 'Validation Time': validation_time 
            }
        )

    print("")
    print("Training complete!")
    # print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    return model


def test_model(model, dataloaders, device, args):
    # print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
    print('Predicting labels for {:,} test sentences...'.format(len(dataloaders[2])))

    model.eval()
    predictions, true_labels = [], []

    for batch in dataloaders[2]:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch 
        
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

        print('    DONE.')

        df = dataloaders[3]
        print('Positive samples: %d of %d (%.2f%%)' % (df['class'].sum(), len(df['class']), (df['class'].sum() / len(df['class']) * 100.0)))

    matthews_set = []
    print('Calculating Matthews Corr. Coef. for each batch...')

    for i in range(len(true_labels)):
        # The predictions for this batch are a 2-column ndarray (one column for "0" 
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        # Calculate and store the coef for this batch.  
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
        matthews_set.append(matthews)

    # Create a barplot showing the MCC score for each batch of test samples.
    # ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    # plt.title('MCC Score per Batch')
    # plt.ylabel('MCC Score (-1 to +1)')
    # plt.xlabel('Batch #')

    # plt.show()

    # Combine the results across all batches. 
    flat_predictions = np.concatenate(predictions, axis=0)

    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    # Calculate the MCC
    mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

    print('Total MCC: %.3f' % mcc)

    
        