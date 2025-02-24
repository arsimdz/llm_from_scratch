from torch.utils.data import DataLoader
from spam_data import SpamDataset
import tiktoken 


tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv",tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="validation.csv",tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="test.csv",tokenizer=tokenizer)

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


for input_batch,target_batch in train_loader:
    pass

print("Input batch dimension:",input_batch.shape)
print("Label batch dimension:",target_batch.shape)
print(f"{len(train_loader)} traiining batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")