"""
Language Model Base
-------------------

We need a language model to serve as the initial point. This language model will be frozen during training and used to decode custom language-vision inputs
"""
import huggingface_hub
import transformers
#from transformers import PhiForCausalLM, PhiForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional peft component for lora
from peft import LoraConfig, PeftModel, get_peft_model


def get_llm(llm_name):
    """
    
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(llm_name)
    
    # Freeze LLM
    for name, param in model.named_parameters():
        param.requires_grad = False
            
    # Add PEFT?
    print('peft requires target modules, gate_proj, up_proj, down_proj')
    if False:
        peft_config = LoraConfig(
        task_type=None, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=['lin1', 'lin2']
        )
        
        model = get_peft_model(model, peft_config)

    return tokenizer, model

if __name__ == '__main__':
    """
    Be able to call llm.py to test things
    """
    
    # Load model directly

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    
    # Load model directly
    # Rather than instruct model, likely want to fine-tune our own base model?
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # 
    from transformers import AutoTokenizer, MistralForCausalLM

    #model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    #tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
