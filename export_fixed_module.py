import torch
import json
from llama import ModelArgs, Tokenizer

tokenizer_path = './tokenizer.model'
params_path = './params.json'

## Initialized Tokenizer
tokenizer = Tokenizer(tokenizer_path)

## Load model args
with open(params_path, "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(
    max_seq_len=512, max_batch_size=1, **params
)

model_args.vocab_size = tokenizer.n_words

### Set dummy input of Transformer block

x = torch.randint(1, 100, (1, 1, 4096))

### ---------------------------------------

from llama.model_fixed import TransformerBlock

### Initialize Transformer Block
tfb = TransformerBlock(31 ,model_args)

### Export 
with torch.no_grad():
    tfblock = torch.onnx.export(tfb, x, 'tfblock_fixed.onnx', opset_version=14)
    
### -----------------------------------------
    
##### If hopes to export other modules list att_norm, attation, ffn could work like following code

### from llama.model_fixed import Attention
### attention =Attention(model_args)
### with torch.no_grad():
    ### torch.onnx.export(attention, x, 'att_fixed.onnx', opset_version = 14) ### opset could also set as 11,12,13