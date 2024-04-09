"""
Vision Encoder
--------------

This vision model will encode images into representations which can be sequenced/patched into tokens under our custom tokenizer.
This will encode visual information for the language model to decode.
"""

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def get_image_encoder():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            
    return processor, model
    
if __name__ == '__main__':
    """
    Test image encoder, verify encoded size
    """
        
    

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    print(processor, model)
    
    for name, parameter in model.named_parameters():
        #print(name, parameter.requires_grad)
        parameter.requires_grad = False
        
    print('image', type(image))
    inputs = processor(images=image, return_tensors="pt")
    print('processed inputs', inputs['pixel_values'].shape)
    outputs = model(**inputs, output_hidden_states = True)
    
    # for tens in outputs.hidden_states:
    #     print(tens.shape)
    
    #print(outputs)
    embeddings = outputs.hidden_states[-1]
    print('made embedings', embeddings.shape)
    
    # Token projection (trainable)
    image_tokenizer = ImageTokenizer(768, 256)
    
    print(image_tokenizer, embeddings.shape)
    tokenized_embeddings = image_tokenizer(embeddings)
    print('tokenzied embeddings', tokenized_embeddings.shape)
    
    