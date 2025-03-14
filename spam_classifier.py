import torch
import tiktoken
from gpt_config_124M import GPT_Config_124M
from gpt import GPTModel
from spam_data import SpamDataset


device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
model_state_dict = torch.load("model_weights.pth",weights_only=True)
model = GPTModel(GPT_Config_124M)
model.out_head = torch.nn.Linear(in_features=GPT_Config_124M["emb_dim"],out_features=2)
model.load_state_dict(model_state_dict,strict=False)
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv",tokenizer=tokenizer)

def classify(text,model,tokenizer,device,max_length=None,pad_token_id=50256):
    model.eval()

    ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    ids = ids[:min(max_length,supported_context_length)]
    ids += [pad_token_id]*(max_length-len(ids))

    input_tensor = torch.tensor(
        ids,device=device
    ).unsqueeze(0)


    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    pred_label = torch.argmax(logits,dim=-1).item()
    if pred_label==1:
        return "spam"
    else:
        return "not spam"    


text_1 = (
    "You are  a winner you have been specially"
    "selected to receive $1000 cash or a $2000 award."
)

print(classify(text_1,model,tokenizer,device,max_length=train_dataset.max_length))

text_2 = (
    "Hey, just wanted to check if we're still on"
    "for dinner tonight? Let me know!"
)

print(classify(text_2,model,tokenizer,device,max_length=train_dataset.max_length))