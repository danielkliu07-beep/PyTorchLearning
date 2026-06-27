import torch
import torch.nn as nn
import torch.nn.functional as F #Access the softmax() function for calculating attention

from torch.optim import Adam #For backpropogation
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

#The word embedding function nn.Embedding() only accepts numbers as inputs, so we do this:
token_to_id = {
    'what': 0,
    'is': 1,
    'statquest': 2,
    'awesome': 3,
    '<EOS>': 4,
}

id_to_token = dict(map(reversed, token_to_id.items())) #Dictionary that can go from id numbers back to the original tokens

inputs = torch.tensor([[token_to_id["what"],
                        token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"]],
                        
                        [token_to_id["statquest"],
                         token_to_id["is"],
                         token_to_id["what"],
                         token_to_id["<EOS>"],
                         token_to_id["awesome"]]])

labels = torch.tensor([[token_to_id["is"],
                        token_to_id["statquest"],
                        token_to_id["<EOS>"],
                        token_to_id["awesome"],
                        token_to_id["<EOS>"]],
                        
                        [token_to_id["is"],
                         token_to_id["what"],
                         token_to_id["<EOS>"],
                         token_to_id["awesome"],
                         token_to_id["<EOS>"]]])

dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

class PositionEncoding(nn.Module):

    def __init__(self, d_model = 2, max_len = 6): #d_model - dimension of the model (number of word embedding values per token), max_len - max number of tokens our Transformer can process (input and output combined)

        super().__init__()

        pe = torch.zeros(max_len, d_model) #6 rows by 2 columns matrix of zeros

        position = torch.arange(start = 0, end = max_len, step = 1).float().unsqueeze(1) #creates a tensor of [0, 1, 2, 3, 4, 5], .unsqueeze(1) turns the tensor into a column matrix
        embedding_index = torch.arange(start = 0, end = d_model, step = 2).float() #tensor of [0, 2], same as 2i

        div_term = 1 / torch.tensor(10000.0)**(embedding_index / d_model) #tensor of all 10000^(2i/d_model))

        pe[:, 0::2] = torch.sin(position * div_term) #Applies sin to even columns - 'start:stop:step -> 0::2'
        pe[:, 1::2] = torch.cos(position * div_term) #Applies sin to odd columns

        self.register_buffer('pe', pe) #Ensures pe gets moved to a GPU if we use one 
    
    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :] #Adds positional encoding values to the word embedding values


class Attention(nn.Module):

    def __init__(self, d_model = 2):

        super().__init__()

        self.W_q = nn.Linear(in_features = d_model, out_features = d_model, bias = False) #Creates a 2 by 2 matrix for the Query Weights
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model, bias = False) #2 by 2 matrix for Key Weights
        self.W_v = nn.Linear(in_features = d_model, out_features = d_model, bias = False) #2 by 2 matrix for Value Weights

        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):

        q = self.W_q(encodings_for_q) #Since self.W_q is a nn.Linear object, it automatically multiplies the 2 matricies together to form q
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        #Calculating the Attention for Q, K, V:
        #The formula is SoftMax((QK^T / sqrt(d_k)) + M) * V
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1 = self.col_dim)) #Similarity between the Queries and the Keys
        scaled_sims = sims / torch.tensor(k.size(self.col_dim) ** 0.5) #Scale the similarities to the square root of the number of values used in each key

        #Masks are used to prevent early tokens from looking at later tokens
        if mask is not None: #Add the masks to the scaled similarities if there is one 
            scaled_sims = scaled_sims.masked_fill(mask = mask, value = -1e9) #masked_fill creates a matrix with the 'Trues' in the matrix set to -1e9, and the 'Falses' set to 0, and adds that matrix to scaled_sims

        attention_percents = F.softmax(scaled_sims, dim = self.col_dim) #Determines the percentages of influence each token should have on the others

        attention_scores = torch.matmul(attention_percents, v) #multiplies the attention percents by the Values in v

        return attention_scores
    
class DecoderOnlyTransformer(L.LightningModule):

    def __init__(self, num_tokens, d_model = 2, max_len = 6):

        super().__init__()

        self.we = nn.Embedding(num_embeddings = num_tokens, embedding_dim = d_model)

        self.pe = PositionEncoding(d_model = d_model, max_len = max_len)

        self.self_attention = Attention(d_model = d_model)

        self.fc_layer = nn.Linear(in_features=d_model, out_features = num_tokens) #Fully connected layer - inputs: d_model, outputs: num_tokens

        self.loss = nn.CrossEntropyLoss() #Cross Entropy Loss function - automatically applies the loss function for us
    
    def forward(self, token_ids):

        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim = 0), token_ids.size(dim = 0)))) #First creates a one-only matrix of size (number of token_ids) x (number of token_ids)
        #Then passes that matrix to torch.tril (Tri-L - Lower Triangle), which sets values not in the Lower Triangle to 0
        mask = mask == 0 #Turns 0s into Trues and 1s into Falses

        self_attention_values = self.self_attention(position_encoded, position_encoded, position_encoded, mask = mask)

        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr = 0.1)
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss

        





