from instruct_data import download_and_load_file, format_input
from instruct_data import InstructDataset,custom_collate
from functools import partial
import torch
import tiktoken
from torch.utils.data import DataLoader
from gpt_download import download_and_load_gpt2
from chapter_5_5 import load_weights_into_gpt
from gpt import GPTModel
from gpt_config_124M import GPT_Config_124M as config
from train import generate,text_to_token,token_ids_to_text,calc_loss_loader,train_model_simple
import time


file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path,url)




train_length = int(len(data)*0.85)
test_length= int(len(data)*0.1)
train_data = data[:train_length]
test_data = data[train_length:test_length+train_length]
val_data = data[test_length+train_length:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
customized_collate_fn = partial(
    custom_collate,
    device = device,
    allowed_max_length = 1024
)
num_workers = 0
batch_size = 8
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

train_dataset = InstructDataset(tokenizer,train_data)
train_loader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last= True,
    num_workers=num_workers
)

val_dataset = InstructDataset(tokenizer,val_data)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructDataset(tokenizer,test_data)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)




model = GPTModel(cfg=config)
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
load_weights_into_gpt(model,params)

start_time = time.time()

optimizer = torch.optim.AdamW(
    model.parameters(),lr=0.00005, weight_decay=0.1
)
num_epochs = 2

train_losses,val_losses,tokens_seen = train_model_simple(
    model,train_loader,val_loader,optimizer,device,
    num_empochs=num_epochs,eval_freq=5,eval_iter=5,
    start_context=format_input(val_data[0]),tokenizer=tokenizer
)

end_time = time.time()
execution_time = (end_time-start_time)/60
print(f"Training completed in {execution_time:.2f} minutes.")

torch.save(model.state_dict(), 'instruct_weights.pth')