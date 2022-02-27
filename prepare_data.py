import pandas as pd
from transformers import BertTokenizer
import torch

def prep_data():
    PATH = {'twitter': 'datasets/twitter.csv',  # 'hate_speech' col: # of users who marked it as hateful
            'gab': 'datasets/gab.csv',
            'reddit': 'datasets/reddit.csv',
            'parler': 'datasets/parler_pretrain.ndjson'
            }

    df = pd.read_csv(PATH['twitter'])

    tweets = df.loc[:,'tweet'].values
    labels = df.loc[:,'hate_speech'].values

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ids = []
    atn_mask = []
    for tweet in tweets:
        encoded = tokenizer.encode_plus(tweet, 
                        add_special_tokens=True, 
                        return_tensors = 'pt',
                        return_return_attention_mask=True,
                        pad_to_max_length = True)
                        
        ids.append(encoded['input_ids'])
        atn_mask.append(encoded['attention_mask'])
    
    inputs = torch.cat(ids)
    masks = torch.cat(atn_mask)
    labels = torch.cat(labels)

    return inputs, masks, labels
