'''
Train and Evaluate Model
'''
from transformers import AdamW, get_linear_schedule_with_warmup
from model import get_bert_custom, get_bert_pretrained
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score
import random
import data
from util import *
from model import *

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
        df = data.parse_parler(PATH['parler'])
    else:
        raise NotImplementedError('No such dataset')

    input_ids, attention_masks, labels = data.tokenize_dataframe(df, args['verbose'])
    train_loader, valid_loader, test_loader \
        = data.create_dataloaders(input_ids, attention_masks, labels, args['bz'])

    return train_loader, valid_loader, test_loader, df


def train_model(model, scheduler, optimizer, dataloaders, args):
    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    model = model.to(device) 

    training_stats = [] # Store training and validation loss, validation accuracy, and timings.

    epochs = args['epoch'] # The BERT authors recommend between 2 and 4.
    print('Training...')
    for epoch_i in range(0, epochs):
        print('\n======== Epoch {:} / {:} ========\n'.format(epoch_i + 1, epochs))

        model.train()
        avg_train_loss, avg_train_acc = train_epoch(model, dataloaders[0], device, optimizer, scheduler)
        writer.add_scalar("Loss/train", avg_train_loss, epoch_i)
        writer.add_scalar("Accuracy/train", avg_train_acc, epoch_i)

        model.eval()
        print('\tValidation...')
        avg_val_loss, avg_val_acc = eval_epoch(model, dataloaders[1], device)
        writer.add_scalar("Loss/val", avg_val_loss, epoch_i+1)
        writer.add_scalar("Accuracy/val",avg_val_acc, epoch_i+1)

        training_stats.append({'epoch': epoch_i + 1,        # Record training state
                            'train_loss': avg_train_loss, 'train_acc': avg_train_acc,
                            'valid_loss': avg_val_loss, 'valid_acc': avg_val_acc})
    
    return model, training_stats


def test_model(model, dataloaders, device, args, training_stats):
    print('Teseting...')
    print('Test dataset size: {:,}'.format(len(dataloaders[2])))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    model = model.to(device) 

    model.eval()
    preds, true_labels= test(model, dataloaders[2], device)

    f1 = f1_score(true_labels, np.argmax(preds, axis=1).flatten(), average='macro')
    print("F1 SCORE: %.3f" % f1)

    mcc = mcc_score(preds, true_labels)
    print('Total MCC: %.3f' % mcc)
    print('\tDONE')

    plot_confusion_matrix(preds, true_labels, args, mcc)
    if training_stats: 
        plot_loss_acc(training_stats, mcc, args)
    if args['save']:
        save_model(model, mcc, args)