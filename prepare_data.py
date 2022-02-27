import pandas as pd
from transformers import BertTokenizer
import torch
import re

PATH = {'twitter': 'datasets/twitter.csv',  # 'hate_speech' col: # of users who marked it as hateful
        'gab': 'datasets/gab.csv',
        'reddit': 'datasets/reddit.csv',
        'parler': 'datasets/parler_pretrain.ndjson'
        }

def prep_twitter_data(path = PATH['twitter']):
    df = pd.read_csv(path)

    def preprocess(text_string): # Ref: https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/classifier/final_classifier.ipynb
        """ Accepts a text string and replaces:
            1) urls with URLHERE
            2) lots of whitespace with one instance
            3) mentions with MENTIONHERE
        Get standardized counts of urls and mentions w/o caring about specific people mentioned
        """
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        return parsed_text

    df = pd.concat([df['tweet'].apply(preprocess), df['class'].astype(int)], axis = 1)

    tweets = df.loc[:,'tweet'].values
    labels = df.loc[:,'class'].values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Ref: https://mccormickml.com/2019/07/22/BERT-fine-tuning/

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

