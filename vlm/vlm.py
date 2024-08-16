"""
Seems like we want our own module so we can train multiple elements at once
"""
import torch
import torch.nn as nn
import torchvision
import random

from modeling import vision_encoder, llm, projection

def build_vlm(image_encoder_str = 'google/vit-base-patch16-224',
              llm_str = "microsoft/Phi-3-mini-4k-instruct",
              projector_str = 'twoLayerLinear'):
    """
    Assemble a VLM from an image encoder, projection, and llm
    """
    # Image processor and image encoder model - loaded from a Huggingface ViT
    image_processor, image_encoder = vision_encoder.get_image_encoder(image_encoder_str, peft = True)
    
    # Language model tokenizer and llm
    #language_tokenizer, language_model = llm.get_llm("mistralai/Mistral-7B-v0.1")
    language_tokenizer, language_model = llm.get_llm(llm_str, peft = True)
    
    # "Image tokenizer" projects from image encoder transformer activations to LLM input dimension
    #image_tokenizer = projection.ImageTokenizer(in_dim = 768, out_dim = 3072)
    insize = projection.get_in_size(image_encoder)
    outsize = projection.get_out_size(language_tokenizer, language_model)
    image_tokenizer = projection.get_image_tokenizer(projector_str, insize, outsize)
    
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
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model structural components
        self.vision_processor = vision_processor
        self.vision_encoder = vision_encoder
        self.vision_tokenizer = vision_tokenizer
        
        self.language_model = language_model
        self.language_tokenizer = language_tokenizer
        
        # Language components - custom tokens we use to format the prompt
        # bos_token = tokenizer.bos_token # begin sentence
        imstart_str = '<|imstart|>'
        imend_str = '<|imend|>'
        #textend_str = '<|endoftext|>'
        language_prompt = "This image contains: "

        language_tokenizer.add_tokens(imstart_str, special_tokens = True)
        language_tokenizer.add_tokens(imend_str, special_tokens = True)
        
        # Update langauge model's token embedding matrix
        self.language_model.resize_token_embeddings(len(self.language_tokenizer))
        
        # Hold vector representing elements of our custom prompt
        self.start_vec = self.embed_ints(torch.tensor(self.language_tokenizer(imstart_str)['input_ids'])).unsqueeze(0)
        self.end_vec = self.embed_ints(torch.tensor(self.language_tokenizer(imend_str)['input_ids'])).unsqueeze(0)
        self.query_vec = self.embed_ints(torch.tensor(self.language_tokenizer(language_prompt)['input_ids'])).unsqueeze(0)
        #self.textend_vec = self.embed_ints(torch.tensor(self.language_tokenizer(textend_str)['input_ids'], device = self.device)).unsqueeze(0)
        
        # Loss
        self.bceloss = nn.BCEWithLogitsLoss()
    
    def forward(self, batch):
        """
        Given a batch, format the input, do the forward, get the logits, return logits, loss
        Predicts the next token given an image, random substring of caption
        """
        device = batch['image'].device
        tokenized_image = self.image_forward(batch['image'])
        
        # Tokenize string
        int_captions = torch.LongTensor(self.language_tokenizer(batch['caption'])['input_ids']).to(device)
        
        predict_at_index = random.randint(1, int_captions.shape[1] - 2)
        caption_prefix = self.embed_ints(int_captions[:, :predict_at_index])
        caption_target = int_captions[:, predict_at_index]
        
        self.start_vec = self.start_vec.to(device)
        self.end_vec = self.end_vec.to(device)
        self.query_vec = self.query_vec.to(device)
        
        # Structure token sequence

        # Here, rather than simple concat, need to use the LLM text formatting, then break up the ids and embed them
        # then cat the embeddings which use the conversation formatting to the image embeddings to the suffix embeddings
        llm_input = torch.cat((self.start_vec, tokenized_image, self.end_vec, self.query_vec, caption_prefix), dim = 1)#.permute(0,2,1)
        
        # Forward with frozen llm
        output = self.language_model.forward(inputs_embeds = llm_input)
        logits = output.logits
        
        #print(logits.shape)
        last_logit = logits[:, -1, :]
        
        loss = self.loss_function(last_logit, caption_target)
        
        return logits, loss
    
    def generate(self, batch, max_new_tokens):
        device = batch['image'].device
        #self.language_model.assisted_decoding
        
        # idx is (B, T) array of indices in the current context
        tokenized_image = self.image_forward(batch['image'])
        
        # get initial prompt for llm
        # Tokenize string
        int_captions = torch.LongTensor(self.language_tokenizer(batch['caption'])['input_ids']).to(device)
        
        predict_at_index = 0
        caption_prefix = self.embed_ints(int_captions[:, :predict_at_index])
        
        self.start_vec = self.start_vec.to(device)
        self.end_vec = self.end_vec.to(device)
        self.query_vec = self.query_vec.to(device)
        
        # Structure token sequence
        llm_input = torch.cat((self.start_vec, tokenized_image, self.end_vec, self.query_vec, caption_prefix), dim = 1)#.permute(0,2,1)
        
        logit_outputs = []
        for _ in range(max_new_tokens):
            # Forward on prompt
            outputs = self.language_model.forward(inputs_embeds = llm_input)
            logit_output = outputs.logits[:, -1, :]
            logit_outputs.append(logit_output)
            
            #print(llm_vector_prompt.shape, logit_output.shape)
            # Add EMBEDDED output to current sequence
            int_output = logit_output.argmax(dim = 1)
            #print('int output', int_output)
            new_vec = self.embed_ints(int_output).unsqueeze(0)
            llm_input = torch.cat((llm_input, new_vec), dim = 1)
        
        logit_outputs = torch.stack(logit_outputs, dim = 1)
        #print('generate constructed outputs', logit_outputs.shape)
        
        # Logits to ints
        int_outputs = logit_outputs.argmax(dim = 2)
        str_outputs = self.language_tokenizer.decode(int_outputs[0])
        print('Caption: [', batch['caption'], ']')
        print('Str out: [', str_outputs, ']')
        return str_outputs
    
    def image_forward(self, image):
        """
        Set of PIL images?
        Or single pil image?
        """
        # Vision processor should have all necessary transforms and normalize
        #print('in img forward', image.shape)
        inputs = self.vision_processor(image, return_tensors='pt').to(self.vision_encoder.device)
        
        #print('what is this', type(image), image.shape)
        # Encode from image pixels to token sequence
        
        encoded_image = self.vision_encoder(**inputs, output_hidden_states = True)
        encoded_image = encoded_image.hidden_states[-1]
        
        #print('encoded', encoded_image.shape)
        # Project representation to language tokens with trainable 'image tokenizer'
        tokenized_image = self.vision_tokenizer(encoded_image)
        
        return tokenized_image
    
    def embed_ints(self, tokens):
        """
        Use the model's existing integer tokens to return vector embeddings:
        """
        return self.language_model.get_input_embeddings()(tokens)
    
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
    model = build_vlm()
    #model.print_trainable_parameters()
    
