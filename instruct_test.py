import torch
from gpt import GPTModel
from gpt_config_124M import GPT_Config_124M
from instruct_data import download_and_load_file
from instruct_data import format_input
from train import token_ids_to_text,text_to_token,generate
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
model_state_dict = torch.load("instruct_weights.pth",weights_only=True)
model = GPTModel(GPT_Config_124M)

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path,url)

train_length = int(len(data)*0.85)
test_length= int(len(data)*0.1)
test_data = data[train_length:test_length+train_length]

for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model,
        idx = text_to_token(input_text,tokenizer),
        max_new_tokens=256,
        context_size=GPT_Config_124M["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids,tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:","")
        .strip()
    )
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>>{response_text.strip()}")
    print("------------------------------------------")

