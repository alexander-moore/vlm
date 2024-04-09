import torch.nn as nn


"""
Currently just using simple linear projection.
Literature often uses tiny attention models like xformer (?) mini-gpt etc to get output tokens and then project to new LLM dim
"""

class ImageTokenizer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ImageTokenizer, self).__init__()
        self.projection = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        """
        Forward maps vision_encoder outputs to llm_token input size
        (bs, seq_length, in_size) -> (bs, seq_length, out_size)
        """
        return self.projection(x)
    
    