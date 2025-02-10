import tiktoken
import torch
from gpt_config_124M import GPT_Config_124M
from gpt import GPTModel
import gpt

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from dataloader_gpt import create_data_loader_v1
from train import train_model_simple




with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read()

split_ratio = 0.9
split_idx = int(split_ratio * len(raw_text))

tokenizer = tiktoken.get_encoding("gpt2")
train_set = raw_text[:split_idx]
val_set = raw_text[split_idx:]

dataloader_train = create_data_loader_v1(train_set,batch_size=2,max_length=GPT_Config_124M["context_length"],
                                         stride=GPT_Config_124M["context_length"],shuffle=False)
dataloader_val =  create_data_loader_v1(val_set,batch_size=2,max_length=GPT_Config_124M["context_length"],
                                         stride=GPT_Config_124M["context_length"],shuffle=False)

torch.manual_seed(123)
model = GPTModel(GPT_Config_124M)
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
model = model.to(device)
num_empochs=10
train_losses,val_losses,tokens_seen = train_model_simple(
    model,dataloader_train, dataloader_val,optimizer,device,
    num_empochs=num_empochs,eval_freq=5,eval_iter=5,
    start_context="Every efforts moves you",tokenizer=tokenizer
)


def plot_losses(epochs_seen,tokens_seen,train_losses,val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label="Training Loss")
    ax1.plot(
        epochs_seen,val_losses,linestyle="-.",label= "Validation Loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen,train_losses,alpha=0)
    ax2.set_xlabel("Tokens Seen")
    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, num_empochs, len(train_losses))
plot_losses(epochs_tensor,tokens_seen,train_losses,val_losses)

 





