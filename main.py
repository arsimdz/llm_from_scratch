import tiktoken
import torch
from gpt_config_124M import GPT_Config_124M
from gpt import GPTModel,GELU
import gpt
from multihead_attention import MulitHeadAttention
import matplotlib.pyplot as plt
       
def generate_text_simple(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]    
        prob = torch.softmax(logits,dim=-1)
        idx_next = torch.argmax(prob,dim=-1,keepdim=True)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx    

def text_to_token(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor   

def token_ids_to_text(token_ids,tokenizer):
    decode_txt = tokenizer.decode(token_ids.squeeze(0).tolist())
    return decode_txt


tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Every effort moves you"

encoded_tensor = text_to_token(start_context,tokenizer)

 
model = GPTModel(GPT_Config_124M)
model.eval()

token_ids = generate_text_simple(model=model,idx=encoded_tensor,max_new_tokens=6,context_size=GPT_Config_124M["context_length"])
decoded_txt = token_ids_to_text(token_ids,tokenizer)
print(decoded_txt)

