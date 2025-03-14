from instruct_data import download_and_load_file, format_input

file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

data = download_and_load_file(file_path,url)
print("Number of entries:",len(data))
print("Example entry:\n",data[50])
input = format_input(data[50])
desired_output = f"\n\n#### Response:\n{data[50]['output']}"
print(input+desired_output)

train_length = int(len(data)*0.85)
test_length= int(len(data)*0.1)
train_data = data[:train_length]
test_data = data[train_length:test_length+train_length]
val_data = data[test_length+train_length:]

print("Train data size:",len(train_data))
print("Test data size:",len(test_data))
print("Val data size:",len(val_data))
