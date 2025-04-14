import time

import torch
from tokenization import Lang
from model import GPTLanguage


checkpoint = torch.load("model_epoch_1.pth")
hyperparameters = checkpoint["hyperparameters"]
loaded_model = GPTLanguage(
    vocab_size=hyperparameters["vocab_size"],
    embd_size=hyperparameters["embedding_size"],
    heads_num=hyperparameters["heads_num"],
    layers_num=hyperparameters["layers_num"],
    block_size=hyperparameters["block_size"],
)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
print("Model loaded successfully")

lang = Lang.load_state_dict(checkpoint["lang_state_dict"])


@torch.no_grad()
def generate_text(model, prompt, max_new_tokens):
    model.eval()
    encoded = lang.encode(prompt)
    context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        tokens = model.generate(context, max_new_tokens=max_new_tokens)
    return lang.decode(tokens[0])


prompt_text = "Ziman hebûn e"
start = time.time()
generated_text = generate_text(loaded_model, prompt_text, max_new_tokens=2000)
print(f"Prompt: {prompt_text}")
print(f"Generated text: {generated_text}")
stop = time.time()
print(f"Time taken: {stop - start:.2f} seconds")
