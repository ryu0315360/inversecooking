import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.utils as utils
from modules.multihead_attention import MultiheadAttention
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Quantity_DecoderTransformer(nn.Module):
    def __init__(self, embed_size, output_size, num_layers, nhead, dropout=0.1):
        super().__init__()

        self.embed_size = embed_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nhead = nhead

        # create a transformer decoder with specified number of layers and attention heads
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # create output linear layer
        self.linear = nn.Linear(embed_size, output_size)

    def forward(self, input):
        # input: tensor of shape (batch_size, embed_size)

        # add a sequence dimension to the input
        input = input.unsqueeze(0)

        # create a target mask for the decoder
        mask = torch.ones((1, 1)).to(input.device)

        # pass input through the decoder
        output = self.transformer_decoder(input, None, tgt_mask=mask)

        # apply linear layer to generate output sequence
        output = self.linear(output)

        return output.squeeze(0)

    def sample(self, input): ## forward랑 같음
        # input: tensor of shape (batch_size, embed_size)

        # add a sequence dimension to the input
        input = input.unsqueeze(0)

        # create a target mask for the decoder
        mask = torch.ones((1, 1)).to(input.device)

        # pass input through the decoder
        output = self.transformer_decoder(input, None, tgt_mask=mask)

        # apply linear layer to generate predicted quantities
        output = self.linear(output)

        return output.squeeze(0)

class Quantity_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
