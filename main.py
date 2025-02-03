import tiktoken
import torch
from gpt_config_124M import GPT_Config_124M
import gpt

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch,dim=0)
print(batch) 
       
torch.manual_seed(123)
model = gpt.DummyGPTModel(GPT_Config_124M)
logits = model(batch)
print("Output Shape:",logits.shape)
print(logits)