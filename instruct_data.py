import json
import os
import urllib
import urllib.request
import torch
from torch.utils.data import Dataset



def download_and_load_file(file_path,url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path,"w",encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path,"r",encoding="utf-8") as file:
            text_data = file.read()

    with open(file_path,"r") as file:
        data = json.load(file)

    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes request."
        f"\n\n#### Instruction:\n{entry['instruction']}"
    )

    input_text = (
        f"\n\n#### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text+input_text


class InstructDataset(Dataset):
    


    def __init__(self,tokenizer,data):
        self.data =data
        self.encoded_text = []
        for entry in data:
            instruct_input = format_input(entry)
            output = f"\n\n#### Response:\n{entry['output']}"
            text = instruct_input + output
            encoded = tokenizer.encode(text)
            self.encoded_text.append(encoded)

    def __getitem__(self, index):
        return self.encoded_text[index]
    
    def __len__(self):
        return len(self.data)
    


def custom_collate(batch,pad_token=50256,device="cpu",ignore_index=-100,allowed_max_length=None):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_list,target_list = [],[]
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token]
        padded = (
            new_item +[pad_token]*(batch_max_length-len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_list.append(inputs)
        target_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    target_tensor = torch.stack(target_list).to(device)
    return inputs_tensor,target_tensor    
            