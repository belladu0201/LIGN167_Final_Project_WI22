# %%
from pkgutil import get_data
import pandas as pd
import numpy as np
import ujson as json
import torch
from nltk.stem.porter import *
from torch.utils.data import TensorDataset, DataLoader
import ujson as json
from engine import *
import util
from transformers import AdamW
from tqdm import tqdm  

def get_dataloader():
    records = map(json.loads, open('datasets/parler_pretrain.ndjson'))
    df = pd.DataFrame.from_records(records)
    len_orig = len(df)
    df = df[['body']]
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    print("{}/{} ({:2f})% of samples are non-empty".format(len(df), len_orig, len(df)/len_orig*100))

    # %%
    from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertConfig, BertForSequenceClassification

    sentences = df['body'].values.tolist()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Done parsing sentences.")

    # 1. Create inputs, i.e. tokenized sentences
    inputs = tokenizer(sentences, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    print('Done Tokenizing sentences')
    print(inputs.keys())

    # 2. Create labels: a copy of input_ids, i.e. tokenized input
    inputs['labels'] = inputs.input_ids.detach().clone()
    print(inputs.keys())

    # 3. Mask 15% of words in labels
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
            (inputs.input_ids != 102) * (inputs.input_ids != 0)
                # 101: input_id for [CLS] | 102: input_id for [SEP] | 0: input_id for [PAD]
    # Select positions to be masked
    selection = [torch.flatten(mask_arr[i].nonzero()).tolist() \
                for i in range(inputs.input_ids.shape[0])]
    # Mask-out selected positions
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103 # input_id for [MASK]
    # print(inputs.input_ids[:10])
    dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, inputs.labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader


def pretrain(epochs):
    dataloader = get_dataloader()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            optim.zero_grad()
            input_id, input_mask, label = \
                batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(input_ids = input_id, attention_mask=input_mask,
                            labels=label)

            loss = outputs.loss
            loss.backward()
            optim.step()
        print('epoch: {} - loss: {}'.format(epoch, loss))

    # PATH = "./models/pretrain/" 
    # if not os.path.exists(PATH): os.mkdir(PATH)
    model.save_pretrained("model-custom-" + str(epochs))

def run(args):
    device = torch.device("cuda:{}".format(args['device_id']) if torch.cuda.is_available() else "cpu")
    
    args['pretrain_epochs'] = 1
    # for ds in ['twitter', 'gab', 'reddit']:
    for ds in ['gab', 'reddit']:
        args['num_classes'] = 3 if ds == 'twitter' else 2
        args['dataset'] = ds 
        dataloaders = prepare_data(args)
        model, scheduler, optimizer = prepare_model(device, len(dataloaders[0]), args)
        util.save_model(model, args) # Save the pretrained model before fine-tuning
        model, train_stats = train_model(model, scheduler, optimizer, dataloaders, args)
        test_model(model, dataloaders, device, args, training_stats=train_stats)
        del model

'''Pretrain'''
# pretrain(1)
# pretrain(2)


