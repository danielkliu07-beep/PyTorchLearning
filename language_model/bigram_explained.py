import torch

with open('language_model/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #Create's a sorted list of all of the characters in the set 
vocab_size = len(chars)
print(''.join(chars)) #prints all available characters
print(vocab_size) #prints how many vocabulary are used



#Example of a character-level Tokenizer - converts the raw text as a string into a sequence of integers
#Google uses sentence piece for their tokenizer

#Creates a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } #A lookup table for a character to an integer
itos = { i:ch for i,ch in enumerate(chars) } #A lookup table for an integer to a character
encode = lambda s: [stoi[c] for c in s] #Encoder: takes a string, outputs a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #Decoder: takes a list of integers, outputs a string

#print(encode("hii there"))
#print(decode(encode("hii there")))



data = torch.tensor(encode(text), dtype = torch.long) #Wraps all of the tiny shakesphere dataset encodings into a torch tensor
#print(data.shape, data.dtype)
#print(data[:1000]) #First 1000 characters of the encoding

n = int(0.9 * len(data)) #First 90% of the data will be for training, the rest is validation
train_data = data[:n]
val_data = data[n:]



block_size = 8 #Maximum size allowed for batches used for training
train_data[:block_size + 1] #In a chunk of 9 characters, there are 8 individual examples that can come after elements
#print(train_data[:block_size + 1])

x = train_data[:block_size] #The first block_size amount of characters in the dataset
y = train_data[1: block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    #print(f"when input is {context} the target: {target}")



torch.manual_seed(1337) 
batch_size = 4 #How many independent sequences every forward and backward pass of the transformer
block_size = 8 #Maximum context length/size allowed for batches used for training

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #Generates batch_size number of random offsets (0 to len(data) - block_size) 
    x = torch.stack([data[i:i+block_size] for i in ix]) #Every block_size sequence of characters starting at i
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #(The offset by 1 of x) Every block_size sequence of characters starting at i + 1
    #Torch.stack() combines all of these block_size character sequences into a single matrix (4 by 8 since batch_size x block_size)

    return x, y


xb, yb = get_batch('train')
#print('inputs:')
#print(xb.shape)
#print(xb)
#print('targets:')
#print('yb.shape')
#print(yb)

#As shown in the print statements, all 8 elements in a row of the inputs (xb) produce the last element of the targets (yb)

#print('-----')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1] #bth row, everything before t + 1
        target = yb[b, t] #bth row, t
        #print(f"when input is {context.tolist()} the target: {target}")

print(xb) #Our input to the transformer





import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):

        super().__init__()

        #Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):

        # idx and targets are both (Batch, Time) tensor of integers
        logits = self.token_embedding_table(idx) # (Batch (batch_Size), Time (block_Size), Channel (vocab_size)) tensor

        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B * T, C) #Stretches B and T into a 1 dimensional sequence while preserving C as the second dimension
            targets = targets.view(B * T) #Turns the targets into a 1 dimensional sequence to match the logits

            loss = F.cross_entropy(logits, targets) #How well are we finding the 'targets' based on the 'logits'

        return logits, loss 

    def generate(self, idx, max_new_tokens): #idx is a (B, T) array of indices in the current context

        for _ in range(max_new_tokens):

            #get the predictions of the current indicies
            logits, loss = self(idx) #calls the forward function, no error is returned because 'targets = None' makes targets not required

            logits = logits[:, -1, :] #Turns a (B, T, C) tensor into a (B, C) tensor
            probs = F.softmax(logits, dim = -1)

            #idx_next = (B, 1) tensor, because in each batch dimension, we're recieving a single prediction on what happens next
            idx_next = torch.multinomial(probs, num_samples = 1) 

            idx = torch.cat((idx, idx_next), dim = 1) #Creates a (B, T + 1) tensor
        
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape) #The x-dimension for logits is 32 because we multiplied B by T in our logits.view(), the y-dimension is the channel
print(loss)

#Takes in a 1 by 1 zero matrix as the idx with 100 max tokens, and adds the next predictions to the idx 100 times to form a response
print(decode(m.generate(idx = torch.zeros((1, 1), dtype = torch.long), max_new_tokens=100)[0].tolist()))

#Optimizer:
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

batch_size = 32
for steps in range(15000):

    #get a batch of data for training
    xb, yb = get_batch('train')

    #evaluating the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) #zeroing out the gradients
    loss.backward() #getting the gradients
    optimizer.step() #applying the gradients to the values

print(loss.item())


#idx is zero -> generates response from zero -> decodes response -> creates a python list of response
print(decode(m.generate(idx = torch.zeros((1, 1), dtype = torch.long), max_new_tokens=500)[0].tolist()))



