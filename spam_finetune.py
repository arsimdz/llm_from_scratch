from gpt_download import download_and_load_gpt2
from chapter_5_5 import load_weights_into_gpt
from gpt_config_124M import GPT_Config_124M as config
from gpt import GPTModel
import torch 
import tiktoken
from torch.utils.data import DataLoader
from spam_data import SpamDataset
import time
from spam_data import train_classifier_simple
model = GPTModel(config)
tokenizer = tiktoken.get_encoding("gpt2")
settings, params = download_and_load_gpt2(
    model_size="124M", models_dir="gpt2"
)
load_weights_into_gpt(model,params)
num_classes = 2

for param in model.parameters():
    param.requires_grad = False

model.out_head = torch.nn.Linear(in_features=config["emb_dim"],out_features=num_classes)



for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True




train_dataset = SpamDataset(csv_file="train.csv",tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="validation.csv",tokenizer=tokenizer,max_length=train_dataset.max_length)
test_dataset = SpamDataset(csv_file="test.csv",tokenizer=tokenizer,max_length=train_dataset.max_length)

num_workers=0
batch_size = 8

train_loader  =  DataLoader(train_dataset,batch_size=batch_size,num_workers=num_workers
                            ,shuffle=True,drop_last=True)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False)


start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=0.1)
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

train_losses,val_losses,train_accs,val_accs,examples = train_classifier_simple(
    model,train_loader,val_loader,optimizer,device,num_epochs,eval_freq=50,eval_iter=5
)

torch.save(model.state_dict(), 'model_weights.pth')

end_time = time.time()
execution_time = (end_time-start_time)/60
print(f"Training completed in {execution_time:.2f} minutes")


