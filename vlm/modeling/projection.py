"""
Language Model Base
-------------------

We need a language model to serve as the initial point. This language model will be frozen during training and used to decode custom language-vision inputs
"""
import huggingface_hub
import transformers
#from transformers import PhiForCausalLM, PhiForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_llm(llm_name, peft = False):
    """
    
    """
    tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code = True)
    
    # Freeze LLM
    for name, param in model.named_parameters():
        param.requires_grad = False
            
    # Add PEFT?
    if peft is True:
        # Optional peft component for lora
        from peft import LoraConfig, PeftModel, get_peft_model
        
        peft_config = LoraConfig(
        task_type=None, inference_mode=False, r=64, 
        lora_alpha=32, lora_dropout=0.1, 
        target_modules=['gate_up_proj', 'down_proj']
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    return tokenizer, model

if __name__ == '__main__':
    """
    This function is used to test llm.py correctly builds a model
    """
    tokenizer, model = get_llm("microsoft/Phi-3-mini-4k-instruct")
    
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate
    generate_ids = model.generate(inputs.input_ids, max_length=30)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
