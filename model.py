'''
Define Model
'''
from transformers import BertForSequenceClassification
from tqdm import tqdm
import torch
import numpy as np
from util import *


def get_bert_pretrained(args):
    model = BertForSequenceClassification.from_pretrained( #  Pretrained BERT with a single linear classification layer on top. 
        "bert-base-uncased", 
        num_labels = args['num_classes'], 
        output_attentions = False,      # Whether model returns attentions weights.
        output_hidden_states = False,   # Whether model returns all hidden-states.
    )
    return model 

def get_bert_custom(args):
    # TODO
    raise NotImplementedError()

def train_epoch(model, dataloader, device, optimizer, scheduler):
    ''' Train model for 1 epoch
    @param
        model: BERT model
        dataloader: Training set dataloader
        dvice: CPU or CUDA
        optimizer, scheduler: Optimizer and lr scheduler
    @return
        avg_train_loss: Average training loss across batches
    '''
    total_train_loss, total_train_acc = 0, 0
    for batch in tqdm(dataloader):
        input_id, input_mask, label = \
            batch[0].to(device), batch[1].to(device), batch[2].to(device)

        model.zero_grad()
        result = model(input_id, token_type_ids=None, attention_mask=input_mask, labels=label)
        loss, logits = result.loss, result.logits

        loss.backward()

        label_ids = label.to('cpu').numpy()
        logits = logits.detach().cpu().numpy()
        total_train_loss += loss.item()
        total_train_acc  += flat_accuracy(logits, label_ids)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip the norm of the gradients to 1.0. To prevent "exploding gradients"
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(dataloader) # avg. loss over all batches.
    avg_train_acc  = total_train_acc / len(dataloader)  # avg. acc over all batches.

    print("\tAverage training loss: {0:.2f}".format(avg_train_loss))
    print("\t                 acc : {0:.2f}".format(avg_train_acc))
    return avg_train_loss, avg_train_acc

def eval_epoch(model, dataloader, device):
    ''' Evaluate model for 1 epoch
    @param
        model: BERT model
        dataloader: Training set dataloader
        dvice: CPU or CUDA
    @return
        avg_eval_loss: Average evaluation loss across batches
        avg_eval_acc:  Average evaluation acc  across batches
    '''
    total_eval_loss, total_eval_acc = 0, 0
    for batch in tqdm(dataloader):
        input_id, input_mask, label = \
            batch[0].to(device), batch[1].to(device), batch[2].to(device)

        with torch.no_grad():
            result = model(input_id, token_type_ids=None, attention_mask=input_mask, labels=label)
            loss, logits = result.loss, result.logits

        label_ids = label.to('cpu').numpy()
        logits = logits.detach().cpu().numpy()
        total_eval_loss += loss.item()
        total_eval_acc += flat_accuracy(logits, label_ids) # Accuracy for this batch of test sentences, accumulated over all batches.

    avg_eval_loss = total_eval_loss / len(dataloader) # avg. loss over all batches.
    avg_eval_acc  = total_eval_acc  / len(dataloader) # avg. acc over all batches.

    print("\tAverage evaluation loss: {0:.2f}".format(avg_eval_loss))
    print("\t                   acc : {0:.2f}".format(avg_eval_acc))
    return avg_eval_loss, avg_eval_acc

def test(model, dataloader, device):
    ''' Test the model
    @return:
        preds: Model predictions
        true_labels: Ground truth
    '''
    preds, true_labels = [], []
    for batch in dataloader:
        input_id, input_mask, label = \
            batch[0].to(device), batch[1].to(device), batch[2].to(device)
        
        with torch.no_grad():
            results = model(input_id, token_type_ids=None, attention_mask=input_mask)

        label_ids = label.to('cpu').numpy()
        logits = results[0].detach().cpu().numpy()

        preds.append(logits)
        true_labels.append(label_ids)
    
    preds = np.concatenate(preds, axis=0) # Combine across all batches
    true_labels = np.concatenate(true_labels, axis=0)

    return preds, true_labels




