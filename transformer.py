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


