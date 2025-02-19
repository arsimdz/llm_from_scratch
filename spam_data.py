import torch
from torch.utils import Dataset
import pandas as pd 
class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer,max_length=None,pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        if max_length==None:
            self.max_length = self.longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ] 

        self.encoded_texts = [
            encoded_text + [pad_token_id]*(self.max_length-len(encoded_text))
            for encoded_text in self.encoded_texts
        ]       