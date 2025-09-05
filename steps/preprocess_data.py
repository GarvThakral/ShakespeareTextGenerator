from zenml import step
import datasets as ds
import logging 
from typing import Tuple , Any , Dict
import re

@step
def preprocessing_data(ds:ds.DatasetDict,maxInDs:int)->Tuple[ds.Dataset,ds.Dataset,int,Dict[int , str],Dict[str,int],int]:
    newVocab = []

    def create_new_vocab(sentence,newVocab):
        sentence_array = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,-]", sentence.lower())
        new_words = [x for x in sentence_array if x not in newVocab]
        new_words_set = set(new_words)
        new_words = list(new_words_set)
        newVocab += new_words

    ds['train'].map(lambda x : create_new_vocab(x["Text"],newVocab))
    ds['test'].map(lambda x : create_new_vocab(x["Text"],newVocab))

    newVocab.insert(0,"<UNK>")
    newVocab.insert(0,"<PAD>")
    
    vocab_size = len(newVocab)
    index_to_word = {i:k for i,k in enumerate(newVocab)}
    word_to_index = {k:i for i,k in enumerate(newVocab)}

    def preprocessing_layer(sentence):
        sentence_array = re.findall(r"\b\w+(?:'\w+)?\b|[.!?,-]", sentence.lower())
        # print(sentence_array)
        wor_to_in = [word_to_index[x] for x in sentence_array]
        # print(wor_to_in)
        pad_len = maxInDs - len(wor_to_in)
        zero_array = [0]*pad_len
        # print(len(wor_to_in))
        padded_wor_to_in = wor_to_in+zero_array
        # print(len(padded_wor_to_in))
        if(len(padded_wor_to_in) > maxInDs):
            logging.info("Sentence size mismatch")
            return "STOP"
        labels = padded_wor_to_in[1:] + [0]
        return {"features":padded_wor_to_in,"labels":labels}

    processed_ds_train = ds['train'].map(lambda x : preprocessing_layer(x["Text"]))
    processed_ds_test = ds['test'].map(lambda x : preprocessing_layer(x["Text"]))

    return processed_ds_train , processed_ds_test , maxInDs , index_to_word , word_to_index , vocab_size