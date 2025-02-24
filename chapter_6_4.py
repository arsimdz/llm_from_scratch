from gpt_download import download_and_load_gpt2
from chapter_5_5 import load_weights_into_gpt
from gpt_config_124M import GPT_Config_124M as config
from gpt import GPTModel
import torch 
import tiktoken

model = GPTModel(config)
tokenizer = tiktoken.get_encoding("gpt2")
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
load_weights_into_gpt(model,params)
model.eval()
num_classes = 2
model.out_head = torch.nn.Linear(in_features=config["emb_dim"],out_features=num_classes)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time.")
inputs = torch.tensor(inputs).unsqueeze(0)
with torch.no_grad():
    outputs = model(inputs)

print("output:\n",outputs)
print("output dimension",outputs.shape)