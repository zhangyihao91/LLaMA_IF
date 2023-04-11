# %%
from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

#from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama.model import precompute_freqs_cis

# %%
from llama.generation import sample_top_p

# %% [markdown]
# ### Load Model/Tokenizer

# %%


# %%
checkpoint = torch.load('/workspace/llm/llama_fb/download/7B/consolidated.00.pth', map_location="cpu")

with open('/workspace/llm/llama_fb/download/7B/params.json', "r") as f:
    params = json.loads(f.read())

model_args: ModelArgs = ModelArgs(
    max_seq_len=512, max_batch_size=1, **params
)
tokenizer = Tokenizer('/workspace/llm/llama_fb/download/tokenizer.model')
model_args.vocab_size = tokenizer.n_words
torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)
model.load_state_dict(checkpoint,strict= False)

generator = LLaMA(model, tokenizer)

# %% [markdown]
# ### Initialization

# %%
temperature: float = 0.8
top_p: float = 0.95
max_seq_len=512
max_batch_size=1

# %%
prompts = ['I believe the meaning of life is']

# %%
max_gen_len = 256

bsz = 1 
params = params
prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
min_prompt_size = min([len(t) for t in prompt_tokens])
max_prompt_size = max([len(t) for t in prompt_tokens])

total_len = min(max_seq_len, max_gen_len + max_prompt_size)

tokens = torch.full((bsz, total_len),tokenizer.pad_id).cuda().long()

for k, t in enumerate(prompt_tokens):
    tokens[k, : len(t)] = torch.tensor(t).long()
input_text_mask = tokens != generator.tokenizer.pad_id
start_pos = min_prompt_size
prev_pos = 0

# %% [markdown]
# ### Export

# %%
seqlen = 1
cur_pos = prev_pos + seqlen

mask = torch.full((1, 1, 1, model_args.max_seq_len), float("-inf"), device=tokens.device)
for i in range(start_pos + 1):
    mask[0, 0, 0, i] = 0


freqs_cis_all = precompute_freqs_cis(
    generator.model.params.dim // generator.model.params.n_heads, generator.model.params.max_seq_len * 2
)
freqs_cis = freqs_cis_all[start_pos : start_pos + seqlen]

# dummy_inputs = torch.cat((tokens[:, prev_pos:cur_pos], torch.tensor([[prev_pos]]).cuda()), 1)
# dummy_inputs = (tokens[:, prev_pos:cur_pos].cuda(), torch.tensor([[prev_pos]]).cuda(), mask.cuda(), freqs_cis.cuda())
dummy_inputs = (tokens[:, prev_pos:cur_pos].cuda(), prev_pos, freqs_cis.cuda(), mask.cuda())

# %%
# with torch.no_grad():
#     model.eval()
#     trace = torch.jit.trace(generator.model, dummy_inputs)
#     torch.jit.save(trace, './llama7B_static.trace.pt')

# with torch.no_grad():
#     model.eval()
#     script = torch.jit.script(generator.model, dummy_inputs)
#     torch.jit.save(script, './transformer_static.script.pt')

# with torch.no_grad():
#     model = model.cuda()
#     model.eval()
#     torch.onnx.export(model, dummy_inputs, 'onnx/llama7B_static.onnx/llama7B_static.onnx', export_params=True, verbose=True, opset_version=14)


x = torch.full((1, 1, 4096), 0, device=tokens.device)
transformer_block_inputs = (x.cuda(), prev_pos, freqs_cis.cuda(), mask.cuda())
with torch.no_grad():
    model = model.cuda()
    model.eval()
    torch.onnx.export(model.layers[0], transformer_block_inputs, 'onnx/llama7B_transformer_block.onnx/transformer_block.onnx', export_params=True, verbose=True, opset_version=14)