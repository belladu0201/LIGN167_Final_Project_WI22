'''
Define Model
'''
from transformers import BertForSequenceClassification

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
   