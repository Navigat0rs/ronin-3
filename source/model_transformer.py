import torch
import torch.nn as nn


class TransformerRegression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.001):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Define the input embedding layer
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Define the transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Reshape the input tensor to (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)

        # Apply the input embedding layer
        x = self.embedding(x)

        # Apply the transformer encoder layers
        x = self.transformer(x)

        # Take the mean of the encoder outputs across the sequence dimension
        x = x.mean(dim=0)

        # Apply the output layer
        x = self.output_layer(x)

        return x
