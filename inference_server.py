# === inference_server.py ===
import socket
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import send_tensor, recv_tensor

MODEL_DIR = "./distilgpt2_finetuned"
NUM_SHARDS = 5
SHARD_ADDRS = {
    0: ("192.168.1.45", 9000),
    1: ("192.168.1.45", 9001),
    2: ("192.168.1.45", 9002),
    3: ("192.168.1.45", 9003),
    4: ("192.168.1.45", 9004),
}

# === Load tokenizer and server-side head ===
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
wte = model.transformer.wte
wpe = model.transformer.wpe
drop = model.transformer.drop
ln_f = model.transformer.ln_f
lm_head = model.lm_head
model.eval()

def run_shard(shard_id, tensor):
    ip, port = SHARD_ADDRS[shard_id]
    with socket.create_connection((ip, port), timeout=10) as sock:
        sock.send(b"data")
        send_tensor(sock, tensor.cpu())
        result = recv_tensor(sock)
        return result

def generate_token(input_text, max_new_tokens=20):
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        position_ids = torch.arange(generated.size(1)).unsqueeze(0)
        hidden = wte(generated[:, -1:]) + wpe(position_ids[:, -1:])
        hidden = drop(hidden)

        for shard_id in range(NUM_SHARDS):
            hidden = run_shard(shard_id, hidden)

        logits = lm_head(ln_f(hidden))[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0])

if __name__ == "__main__":
    prompt = "Once upon a time"
    output = generate_token(prompt, max_new_tokens=30)
    print("\n📝 Generated Text:")
    print(output)
