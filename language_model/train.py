import torch

with open('language_model/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #Create's a sorted list of all of the characters in the set 
vocab_size = len(chars)
print(''.join(chars)) #prints all available characters
print(vocab_size) #prints how many vocabulary are used



#Example of a character-level Tokenizer - converts the raw text as a string into a sequence of integers

#Creates a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } #A lookup table for a character to an integer
itos = { i:ch for i,ch in enumerate(chars) } #A lookup table for an integer to a character
encode = lambda s: [stoi[c] for c in s] #Encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder: takes a list of integers, outputs a string

print(encode("hii there"))
print(decode(encode("hii there")))



data = torch.tensor(encode(text), dtype = torch.long) #Wraps all of the tiny shakesphere dataset encodings into a torch tensor
print(data.shape, data.dtype)
print(data[:1000]) #First 1000 characters of the encoding

n = int(0.9 * len(data)) #First 90% of the data will be for training, the rest is validation
train_data = data[:n]
val_data = data[n:]



block_size = 8 #Maximum size allowed for batches used for training
train_data[:block_size + 1] #In a chunk of 9 characters, there are 8 individual examples that can come after elements
print(train_data[:block_size + 1])

x = train_data[:block_size] #The first block_size amount of characters in the dataset
y = train_data[1: block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")



torch.manual_seed(1337)
batch_size = 4
block_size = 8

#Progress: 18 minutes into the video





