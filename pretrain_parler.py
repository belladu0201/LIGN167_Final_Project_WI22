# %%
import pandas as pd
import numpy as np
import ujson as json
import torch
from nltk.stem.porter import *
from torch.utils.data import TensorDataset, DataLoader
import ujson as json

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# %% [markdown]
# ## 1. Create Dataframe for Parler Dataset

# %%
records = map(json.loads, open('datasets/parler_pretrain.ndjson'))
df = pd.DataFrame.from_records(records)
len_orig = len(df)
df = df[['body']]
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)
print("{}/{} ({:2f})% of samples are non-empty".format(len(df), len_orig, len(df)/len_orig*100))
df.head(10)

# %%
from transformers import BertTokenizer, BertForMaskedLM, BertModel, BertConfig

sentences = df['body'].values.tolist()[:10]
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 1. Create inputs, i.e. tokenized sentences
inputs = tokenizer(sentences, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
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
print(inputs.input_ids[:10])


# %%
dataset = TensorDataset(inputs.input_ids, inputs.attention_mask, inputs.labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)


# %%
from transformers import AdamW
from tqdm import tqdm  
import os

def train(epochs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.bert = BertModel(BertConfig.from_pretrained("bert-base-cased"), add_pooling_layer=True) # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/bert/modeling_bert.py#L1286
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

    PATH = "./models/pretrain/" 
    if not os.path.exists(PATH): os.mkdir(PATH)
    torch.save(model.state_dict(), PATH + str(epochs) + '.pt')

train(1)
train(3)


# %%



