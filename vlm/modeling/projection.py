import torch.nn as nn


"""
Currently just using simple linear projection.
Literature often uses tiny attention models like xformer (?) mini-gpt etc to get output tokens and then project to new LLM dim
"""

def get_image_tokenizer(projector_str, insize, outsize):
    if 'qformer' in projector_str:
        #config = BlipConfig(in_dim = insize,
        #                    out_dim = outsize)
        #return QFormer(config)
        pass
    else:
        return ImageTokenizer(insize, outsize)
    
def get_in_size(image_encoder):
    return image_encoder.config.hidden_size

def get_out_size(lang_tokenizer, llm):
    input_ids = lang_tokenizer.encode('hi', add_special_tokens=True, return_tensors="pt")
    vec = llm.get_input_embeddings()(input_ids)
    embedded_tokens_size = vec.size()[-1]
    return embedded_tokens_size

class ImageTokenizer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ImageTokenizer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.activ = nn.GELU()
    
    def forward(self, x):
        """
        Forward maps vision_encoder outputs to llm_token input size
        (bs, seq_length, in_size) -> (bs, seq_length, out_size)
        """
        return self.fc2(self.activ(self.fc1(x)))
    
# class BlipConfig():
#     """
#     Build the config to pass to Blip2Qformer
#     """
#     def __init__(self, in_dim, out_dim):
#         self.hidden_size = in_dim
#         self.num_attention_heads = 8
        

# def QFormer(config):
#     from transformers import Blip2QFormerModel
#     return Blip2QFormerModel()

def qformer():
    import torch
    from qformer import QFormer

    # Create a random tensor of shape (1, 32, 512)
    x = torch.randn(1, 32, 512)

    # Create a random image tensor of shape (1, 3, 224, 224)
    img = torch.randn(1, 3, 224, 224)

    # Create an instance of the QFormer model with the following parameters:
    # - input_size: 512
    # - num_heads: 8
    # - num_layers: 8
    # - dropout: 0.1
    # - num_classes: 2
    # - num_patches: 2
    qformer = QFormer(512, 8, 8, 0.1, 2, 2)

    # Apply the QFormer model to the input tensors x and img
    y = qformer(x, img)
    
    
    # Print the shape of the output tensor y
    print(y.shape)
    
    # Then I think we literally jam the QFORMER output into the LLM? 
    # I think? lol. that ounds crazty

