import pandas as pd
import os
import gzip
import csv
from typing import Union, List
from torch.utils.data import Dataset

from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import time
import random
import os



class MyDataset(data.Dataset):
    name = 'Grand Dataset'

    @staticmethod
    def sort_key(ex):
        return  len(ex.text)

    def __init__(self, path, text_field_a,text_field_b, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None),  # we won't be needing the id, so we pass in None as the field
                  ("sentence_a", text_field_a), ("sentence_b", text_field_b), ("similarity", label_field)]

        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))
        if test:
            for text_a ,text_b in tqdm(zip(csv_data["sentence_a"],csv_data["sentence_b"])):
                examples.append(data.Example.fromlist([None, text_a, text_b,None], fields))
        else:
            for text_a ,text_b ,label in tqdm(zip(csv_data["sentence_a"],csv_data["sentence_b"],csv_data["similarity"])):
                examples.append(data.Example.fromlist([None, text_a, text_b,label], fields))
        super(MyDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)



class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))




