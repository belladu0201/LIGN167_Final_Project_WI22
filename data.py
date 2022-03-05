'''
Create dataloaders
'''
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
from numpy import genfromtxt
from transformers import BertTokenizer
from nltk.stem.porter import *

PATH = {'twitter': 'datasets/twitter.csv',  # 'hate_speech' col: # of users who marked it as hateful
        'gab': 'datasets/gab.csv',  # Total 45601. 
        'reddit': 'datasets/reddit.csv',
        'parler': 'datasets/parler_pretrain.ndjson'
        }

def preprocess(text_string): # Ref: https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/classifier/final_classifier.ipynb
    """ Accepts a text string and replaces:
        1) urls with URLHERE
        2) lots of whitespace with one instance
        3) mentions with MENTIONHERE
        Get standardized counts of urls and mentions w/o caring about specific people mentioned
    @ retrun 
       List of stemmed words in a sentence
    """
    stemmer = PorterStemmer()
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)

    parsed_text = " ".join(re.split("[.,!?:\t\n\"]", parsed_text.lower()))  # Doc: https://docs.python.org/3/library/re.html?highlight=split#re.split
    stemmed_text = [stemmer.stem(t) for t in parsed_text.split()]

    return stemmed_text 

def parse_twitter(dataset_path=PATH['twitter'], sample_df = False, verbose = False):
    df = pd.read_csv(dataset_path) 
    print('Tweet categories: {}'.format(df['class'].unique())) # 0 - hate speech | 1 - offensive language | 2 - neither

    if sample_df:
        df = df.iloc[:20]
    
    df = pd.concat([df['tweet'].apply(preprocess), df['class'].astype(int)], axis = 1)
    df = df.rename(columns = {'tweet':'text', 'class':'class'})
    
    if verbose: # Print out sample text if verbose
        print('{} Dataset Length: [{}]'.format(dataset_path, len(df)))
        # None hateful tweets (some may contain "sensitive" words)
        df_subclass = df.loc[df['class']==2]
        print('---- Number of [neutral] tweets: {} ({}%)'.format(len(df_subclass), round(len(df_subclass)/len(df), 4) * 100))
        print(df_subclass['text'].head(5))

        # Offensive tweets - not-quite hateful
        df_subclass = df.loc[df['class']==1]
        print('---- Number of [offensive] tweets: {} ({}%)'.format(len(df_subclass), round(len(df_subclass)/len(df), 3) * 100))
        print(df_subclass['text'].head(5))

        # Hateful tweets
        df_subclass = df.loc[df['class']==0]
        print('---- Number of [HATEFUL] tweets: {} ({}%)'.format(len(df_subclass), round(len(df_subclass)/len(df), 3) * 100))
        print(df_subclass['text'].head(5))
    
    return df

def parse_reddit_gab(dataset_path=PATH['reddit'], sample_df = False, verbose = False):
    df = pd.read_csv(dataset_path) 
    df = df[['text', 'hate_speech_idx']]

    if sample_df:
        df = df.iloc[:20]

    # Expand intertwined rows
    for i, row in df.iterrows():
        text = re.sub('[1-9.>]', '', row['text'])
        text = text.strip().split('\n')
        # Replace NaN with 0 for hate_speech_idx column.
        type = '0' if pd.isnull(df.iloc[i, 1]) else row['hate_speech_idx'].strip('[]').split(',')[0]
        row['text'], row['hate_speech_idx'] = text, type
    df = df.explode('text', ignore_index=True) # https://stackoverflow.com/questions/39011511/pandas-expand-rows-from-list-data-available-in-column

    df = pd.concat([df['text'].apply(preprocess), df['hate_speech_idx'].astype(int)], axis = 1)
    df = df.rename(columns = {'text':'text', 'hate_speech_idx':'class'})
    df = df.replace({'class': list(range(100))[1:]}, 1)    # Turn class labels into binary

    if verbose: # Print out sample text if verbose
        print('{} Dataset Length: [{}]'.format(dataset_path, len(df)))
        df_subclass = df.loc[df['class']==0]
        print('---- Number of [neutral] tweets: {} ({}%)'.format(len(df_subclass), round(len(df_subclass)/len(df), 4) * 100))
        print(df_subclass.head(5))
        df_subclass = df.loc[df['class']!=0]
        print('---- Number of [HATEFUL] tweets: {} ({}%)'.format(len(df_subclass), round(len(df_subclass)/len(df), 4) * 100))
        print(df_subclass.head(7))
    
    return df

def tokenize_dataframe(df, verbose = False, model='pretrained'): # Tokenize text in a dataframe
    # @reutrn: input_ids, attention_masks, labels
    sentences, labels = df['text'].values, df['class'].values
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # TODO: Add option to use Custom-trained Tokenizer

    input_ids = []  # Tokenize all of the sentences and map the tokens to thier word IDs.
    attention_masks = []
    for i, sent in enumerate(sentences): 
        if len(sent) == 0:  # Remove empty sentence
            sentences = np.delete(sentences, i)
            labels = np.delete(labels, i)
            continue
        encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 48,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    if verbose:
        print('1 Sample -- Words: {}\nTokens: {}\nLabel: {}'.format(sentences[0], input_ids[0].numpy(), labels[0]))
        print('Unique labels: {}'.format(np.unique(labels.numpy())))
    return input_ids, attention_masks, labels

def create_dataloaders(input_ids, attention_masks, labels, batch_size=32):
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Split: Train 70% | Val 15% | Test 15%
    train_size, val_size = int(0.7 * len(dataset)), int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print("Total Samples: {}\n\tTrain {} | Val {} | Test {}".format(len(dataset), train_size, val_size, test_size))
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],  
                                        generator=torch.Generator().manual_seed(42))

    show_distribution(train_dataset, val_dataset, test_dataset)
                                    
    train_loader = DataLoader(train_dataset,  
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size)

    valid_loader = DataLoader(val_dataset, 
                sampler = SequentialSampler(val_dataset), # Select batches sequentially.
                batch_size = batch_size)

    test_loader = DataLoader(test_dataset, 
                sampler = SequentialSampler(test_dataset),  # Select batches sequentially.
                batch_size = batch_size)

    return train_loader, valid_loader, test_loader

def show_distribution(train_dataset, val_dataset, test_dataset):
    print("Train/Val/Test Lens: {}, {}, {}".format(len(train_dataset), len(val_dataset), len(test_dataset)))
    train_dataset, val_dataset, test_dataset = \
        train_dataset.dataset.tensors[2], val_dataset.dataset.tensors[2], test_dataset.dataset.tensors[2]
    nonzero_count_train = np.count_nonzero(train_dataset.numpy())
    nonzero_count_val = np.count_nonzero(val_dataset.numpy())
    nonzero_count_test = np.count_nonzero(test_dataset.numpy())

    print(f"Train Distribution: nonezeros: {int(nonzero_count_train) / len(train_dataset)*100}, zeros: {(len(train_dataset) - nonzero_count_train) / len(train_dataset) * 100}")
    print(f"Valid Distribution: nonezeros: {int(nonzero_count_val) / len(val_dataset)*100}, zeros: {(len(val_dataset) - nonzero_count_val) / len(val_dataset) * 100}")
    print(f"Test Distribution: nonezeros: {int(nonzero_count_test) / len(test_dataset)*100}, zeros: {(len(test_dataset) - nonzero_count_test) / len(test_dataset) * 100}")


'''
Incremental Tests
'''
# df = parse_twitter(sample_df=False, verbose=True)
# df = parse_reddit_gab(PATH['gab'], sample_df=False, verbose=True)
# df = parse_reddit_gab(sample_df=True, verbose=False)

# input_ids, attention_masks, labels = tokenize_dataframe(df, verbose=True)
# a,b,c = create_dataloaders(input_ids, attention_masks, labels)


