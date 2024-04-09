"""
Seems like we want our own module so we can train multiple elements at once
"""
import torch
import torch.nn as nn
import torchvision

from modeling import vision_encoder, llm, projection

def build_vlm():
    """
    Assemble a VLM from an image encoder, projection, and llm
    """
    # Image processor and image encoder model - loaded from a Huggingface ViT
    image_processor, image_encoder = vision_encoder.get_image_encoder()
    
    # Language model tokenizer and llm
    language_tokenizer, language_model = llm.get_llm("mistralai/Mistral-7B-v0.1")
    
    # "Image tokenizer" projects from image encoder transformer activations to LLM input dimension
    image_tokenizer = projection.ImageTokenizer(in_dim = 77, out_dim = 88)
    
    vlm = VisionLanguageModel(image_processor,
                              image_encoder,
                              image_tokenizer,
                              language_tokenizer,
                              language_model
                              )
    
    for name, param in vlm.vision_encoder.named_parameters():
        param.requires_grad = False
        
    for name, param in vlm.language_model.named_parameters():
        param.requires_grad = False
    
    return vlm
    

class VisionLanguageModel(nn.Module):
    """
    SAM for images
    """
    def __init__(self,
                 vision_processor,
                 vision_encoder,
                 vision_tokenizer,
                 language_tokenizer,
                 language_model,
                 ):
        super(VisionLanguageModel, self).__init__() # initialize self._modules as OrderedDict - enables nested nn modules
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model structural components
        self.vision_processor = vision_processor
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        
        self.language_model = language_model
        self.language_tokenizer = language_tokenizer
        
        # Language components - custom tokens we use to format the prompt
        imstart_str = '<|imstart|>'
        imend_str = '<|imend|>'
        #textend_str = '<|endoftext|>'
        language_prompt = "This image contains"
        

        language_tokenizer.add_tokens(imstart_str, special_tokens = True)
        language_tokenizer.add_tokens(imend_str, special_tokens = True)
        
        # Update langauge model's token embedding matrix
        self.language_model.resize_token_embeddings(len(self.language_tokenizer))
        
        # Hold vector representing elements of our custom prompt
        self.start_vec = self.embed_ints(torch.tensor(self.language_tokenizer(imstart_str)['input_ids'], device = self.device)).unsqueeze(0)
        self.end_vec = self.embed_ints(torch.tensor(self.language_tokenizer(imend_str)['input_ids'], device = self.device)).unsqueeze(0)
        self.query_vec = self.embed_ints(torch.tensor(self.language_tokenizer(language_prompt)['input_ids'], device = self.device)).unsqueeze(0)
        #self.textend_vec = self.embed_ints(torch.tensor(self.language_tokenizer(textend_str)['input_ids'], device = self.device)).unsqueeze(0)
        
        # Loss
        self.bceloss = nn.BCEWithLogitsLoss()
        
    # def forward(self, inputs_embeds):
    #     """
    #     Given aggregated dataloader batch containing:
    #     batch['image']: PIL Image (bs, 3, H, W)
    #     batch['label]: str sentence image caption
    #     """
    #     # batch image to image tokens using vision encoder and image tokenizer
    #     tokenized_image = self.image_forward(batch['image'])
        
    #     # get prompt for llm
    #     #llm_vector_prompt = self.format_forward(tokenized_image)
        
    #     output = language_model.forward(inputs_embeds = inputs_embeds)
        
    #     return output['logits'][:, -1] # Last logit
    
    def forward(self, batch):
        """
        Given a batch, format the input, do the forward, get the logits, return logits, loss
        """
        logits = self.pred_next(batch)
        last_logit = logits[:, -1]
        
        target = self.make_target(batch)
        loss = self.loss_function(last_logit, target)
        
        return logits, loss
    
    def pred_next(self, batch):
                
        output = self.language_model.forward(inputs_embeds = inputs_embeds)
    
        tokenized_image = self.image_forward(batch['image'])
        
        llm_vector_prompt = self.format_forward(tokenized_image).to(self.device)
        
        logit_output = self.forward(inputs_embeds = llm_vector_prompt)
        
        return logit_output

    def generate(self, batch, max_new_tokens = 10):
        # idx is (B, T) array of indices in the current context
        tokenized_image = self.image_forward(batch['image'])
        
        # get initial prompt for llm
        llm_vector_prompt = self.format_forward(tokenized_image).to(self.device)
        
        logit_outputs = []
        for _ in range(max_new_tokens):
            # Forward on prompt
            logit_output = self.forward(inputs_embeds = llm_vector_prompt)
            logit_outputs.append(logit_output)
            
            #print(llm_vector_prompt.shape, logit_output.shape)
            # Add EMBEDDED output to current sequence
            int_output = logit_output.argmax(dim = 1)
            #print('int output', int_output)
            new_vec = self.embed_ints(int_output).unsqueeze(0)
            llm_vector_prompt = torch.cat((llm_vector_prompt, new_vec), dim = 1)
        
        logit_outputs = torch.stack(logit_outputs, dim = 1)
        print('generate constructed outputs', logit_outputs.shape)
        return logit_outputs
    
    def image_forward(self, image):
        """
        Set of PIL images?
        Or single pil image?
        """
        # Vision processor should have all necessary transforms and normalize
        image = self.vision_processor(image)
        
        # Encode from image pixels to token sequence
        encoded_image = self.vision_encoder(image)
        
        # Project representation to language tokens with trainable 'image tokenizer'
        tokenized_image = self.vision_tokenizer(encoded_image)
        
        return tokenized_image
    
    def format_forward(self, tokenized_image):
        """
        To account for our unique vision-language component, we need to format the input vectors as:
        [\start_image, image_tokens, \end_image, prompt]
        
        In order for the language model to understand what is going on with our od prompting scheme
        """
        llm_input = torch.cat((self.start_vec, tokenized_image, self.end_vec, self.query_vec), dim = 1)#.permute(0,2,1)

        return llm_input
        
    def embed_ints(self, tokens):
        """
        Use the model's existing integer tokens to return vector embeddings:
        """
        return self.language_model.get_input_embeddings().to(self.device)(tokens)
    
    def loss_function(self, logits, int_labels):
        """
        logits FloatTensor shape: [B*T, vocab_size] (sequence of probabilities over vocab)
        labels intTensor shape: [B*T] (sequence of int vocab positions)
        - what is b*t? Shouldn't loss be (b, vocab) (b, 1) -> ints
        """
        
        return torch.nn.functional.cross_entropy(logits, int_labels)
        
        
if __name__ == '__main__':
    """
    Do a forward to check vlm works
    """