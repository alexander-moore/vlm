"""
Vision Encoder
--------------

This vision model will encode images into representations which can be sequenced/patched into tokens under our custom tokenizer.
This will encode visual information for the language model to decode.
"""

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

def get_image_encoder(image_encoder_str, peft = False):
    processor = ViTImageProcessor.from_pretrained(image_encoder_str)
    model = ViTForImageClassification.from_pretrained(image_encoder_str)
    
    if peft:
        
        from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=None, inference_mode=False, r=8, 
            lora_alpha=32, lora_dropout=0.1, target_modules=['dense']
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    else:
        model.requires_grad = False
        
    return processor, model
    
if __name__ == '__main__':
    """
    Test image encoder, verify encoded size
    """
    #url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    #image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open('/data/coco2017/val2017/000000187990.jpg')
    
    processor, model = get_image_encoder('google/vit-base-patch16-224', peft = True)
        
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states = True)
    
    embeddings = outputs.hidden_states[-1]
    print('made embedings', embeddings.shape)
    
