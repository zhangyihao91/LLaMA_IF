import torch
import json
import tqdm

from llama import ModelArgs, Tokenizer, Transformer, LLaMA
from llama.generation import sample_top_p

### Initializaiton
params_path = '../7B/params.json'
tokenizer_path = '../tokenizer.model'
checkpoint_path = '../7B/consolidated.00.pth'
prompts = ["I believe the meaning of life is"]

### Load model args and checkpoint
with open(params_path, "r") as f:
    params = json.loads(f.read())

checkpoint = torch.load(checkpoint_path, map_location="cpu")

model_args: ModelArgs = ModelArgs(
    max_seq_len=512, max_batch_size=1, **params
)

### Load model

tokenizer = Tokenizer(tokenizer_path)
model_args.vocab_size = tokenizer.n_words
torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = Transformer(model_args)
torch.set_default_tensor_type(torch.FloatTensor)
model.load_state_dict(checkpoint,strict= False)

generator = LLaMA(model, tokenizer)

### Hyperparameters

temperature: float = 0.8
top_p: float = 0.95
max_batch_size=1

### Inference

results = generator.generate(
    prompts, max_gen_len=256, temperature=temperature, top_p=top_p
)

print(results)