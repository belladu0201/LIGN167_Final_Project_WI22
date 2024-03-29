'''
1. Parse arguments
2. Define and Run experiments by calling engine.py
'''
import argparse
from datetime import datetime
import os, sys, json, pathlib
from data import create_dataloaders
from engine import *
from model import * 
import pretrain_parler
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

parser = argparse.ArgumentParser()

parser.add_argument('--log_path', default=datetime.now().strftime('%Y-%m-%d-%H%M%S'), type=str,
                    help='Default log output path if not specified')
parser.add_argument('--model_path', default=None, type=str,
                    help='path to saved model')        
parser.add_argument('--exp_pretrain',dest='exp_pretrain', action='store_true', default=False,
                help='data augmentation')             

parser.add_argument('--sample',dest='sample', action='store_true', default=False,
                help='data augmentation') 
parser.add_argument('--verbose',dest='verbose', action='store_true', default=False,
                help='data augmentation') 
parser.add_argument('--save',dest='save', action='store_true', default=False,
                help='save model or not') 
parser.add_argument('--dataset', default='twitter', type=str,
                    help='select dataset: [\'twitter\', \'gab\', \'reddit\', \'parler\']')

parser.add_argument('--test',dest='test', action='store_true', default=False,
                help='test - evaluate model without training') 
parser.add_argument('--model', default='pretrained', type=str,
                    help='select model: [\'pretrained\', \'custom\'')

parser.add_argument('--num_classes', default=3, type=int,
                    help='number of class')
parser.add_argument('--bz', default=32, type=int,
                    help='batch size')
parser.add_argument('--epoch', default=2, type=int,
                    help='number of epochs')
parser.add_argument('--lr', default=2e-5, type=float,
                    help='learning rate')


parser.add_argument('--device_id', default=0, type=int,
                    help='CUDA device id')
parser.add_argument('--random_seed', default=42, type=int,
                    help='CUDA device id')


args = vars(parser.parse_args())

def main(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    print(args, device)
    dataloaders = prepare_data(args)
    if args['model'] == 'pretrained':
        if not args['test']:
            model, scheduler, optimizer = prepare_model(device, len(dataloaders[0]), args)
            model, train_stats = train_model(model, scheduler, optimizer, dataloaders, args)
            test_model(model, dataloaders, device, args, training_stats=train_stats)
        else:
            model, _, _ = prepare_model(device, train_length=0, args=args)
            model.load_state_dict(torch.load('./models/' + args['model_path']))
            test_model(model, dataloaders, device, args, training_stats=None)
    elif args['model'] == 'custom':
        # The full pre-training + fine-tuning pipeline
        pretrain_parler.run(args)


if __name__ == '__main__':
    main(args)

# ''' Incremental Tests '''
# args['dataset'] = 'reddit'
# args['sample'] = True
# args['epoch'] = 1
# main(args)