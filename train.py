
from torch.nn.functional import cross_entropy
import torch
from torch import nn
def calc_batch_loss(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

def calc_loss_loader(dataloader,model,device,num_batches=None):
    loss = 0
    if(len(dataloader)==0):
        return float("nan")
    elif(num_batches==None):
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches,len(dataloader))
    for i, (input_batch,target_batch) in enumerate(dataloader):
        if(i<num_batches):
            batch_loss = calc_batch_loss(input_batch,target_batch,model,device)
            loss += batch_loss.item()
        else:
            break    

        
    return loss/num_batches            


def train_model_simple(model,train_loader,val_loader,
                       optimizer,device,num_empochs,
                       eval_freq,eval_iter,start_context,tokenizer):
    
    train_losses, val_losses, track_tokens_seen = [],[],[]
    tokens_seen, global_step = 0,-1
    for epoch in range(num_empochs):
        model.train()
        for input_batch,target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(input_batch,target_batch,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(model, train_loader,val_loader,device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep:{epoch+1} (Step {global_step:06d}): "
                      f"Train loss:{train_loss:.3f}, "
                       f"Val Loss: {val_loss:.3f}" )
            
        generate_and_print_sample(
            model,tokenizer,device,start_context
        )   
    return train_losses,val_losses,track_tokens_seen     


def evaluate_model(model, train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        val_loss = calc_loss_loader(dataloader=val_loader,model=model,device=device,num_batches=eval_iter)
        train_loss = calc_loss_loader(dataloader=train_loader,model=model,device=device,num_batches=eval_iter)
    model.train()
    return train_loss, val_loss
    

def generate_and_print_sample(model, tokenizer,device,start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token(start_context,tokenizer=tokenizer)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,idx=encoded,context_size=context_size,max_new_tokens=50)
    gen_txt = token_ids_to_text(token_ids=token_ids,tokenizer=tokenizer)    
    print(gen_txt.replace("\n"," "))
    model.train() 


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