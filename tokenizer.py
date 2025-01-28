import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = "Akwirw ier"
tokens = tokenizer.encode(text,allowed_special={"<|endofzext|>"})
print(tokens)
strings = tokenizer.decode(tokens)
print(strings)