import torch
import torch.nn as nn

#torch.nn.Linear() example

N = 10
D_in = 1
D_out = 1
X = torch.randn(N, D_in)

linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

print(f"Layer's Weight (W): {linear_layer.weight}\n")
print(f"Layer's Bias (b): {linear_layer.bias}\n")


y_hat_nn = linear_layer(X) #forward pass

print(f"Output of nn.Linear (first 3 rows): \n {y_hat_nn[:3]}")



#Activation functions add non-linearities between linear layers, allowing the model to learn the complex patterns of the real world

#1 - ReLU activation function
relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = relu(sample_data)

print(f"Original Data: {sample_data}")
print(f"Data after ReLU: {activated_data}")

#2 - Gaussian Error Linear Unit activation function - modern standard for transformers
gelu = torch.nn.GELU()
sample_data = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
activated_data = gelu(sample_data)

print(f"Original Data: {sample_data}")
print(f"Data after GELU: {activated_data}")

#3 - Softmax activation function
softmax = torch.nn.Softmax(dim = -1)
logits = torch.tensor([[1.0, 3.0, 0.5, 1.5], [-1.0, 2.0, 1.0, 0.0]])
probabilities = softmax(logits)

print(f"Output Probabilities:\n {probabilities}\n")
print(f"Sum of probabilities for item 1: {probabilities[0].sum()}")



#Essential Layers for LLMs:

#1 - torch.nn.Embedding()
vocab_size = 10 #Our language has 10 unique words
embedding_dim = 3 #We'll represent each word with a 3d vector

embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)

input_ids = torch.tensor([[1, 5, 0, 8]])
word_vectors = embedding_layer(input_ids)

#2 - torch.nn.LayerNorm() - prevents values from exploding, rescales values to stable range
norm_layer = torch.nn.LayerNorm(normalized_shape=3)
input_features = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
normalized_features = norm_layer(input_features)

print(f"Mean (should be ~0): {normalized_features.mean(dim = -1)}")
print(f"Std Dev (should be ~1): {normalized_features.std(dim = -1)}")

#3 - torch.nn.Dropout() - prevents overfitting, randomly zeros neurons during training
dropout_layer = torch.nn.Dropout(p = 0.5) #Creates a dropout layer that zeros out 50% of inputs
input_tensor = torch.ones(1, 10)

dropout_layer.train()
output_during_train = dropout_layer(input_tensor)

dropout_layer.eval()
output_during_eval = dropout_layer(input_tensor)






