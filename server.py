# === server_debug.py ===
import os
import socket
import time
import torch
import pandas as pd
import psutil
from torch import nn, optim
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import send_tensor, recv_tensor

# === Config ===
MODEL_ID = "distilgpt2"
MODEL_DIR = "./distilgpt2_local"
CSV_PATH = "wikitext_small.csv"
SAVE_DIR = "./distilgpt2_finetuned"

EPOCHS = 20
BATCH_SIZE = 4
MAX_SEQ_LEN = 16
NUM_SHARDS = 5

SHARD_ADDRS = {
    0: ("192.168.1.47", 9000),
    1: ("192.168.1.47", 9001),
    2: ("192.168.1.48", 9002),
    3: ("192.168.1.48", 9003),
    4: ("192.168.1.48", 9004),
}

def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEM] {note} RAM used: {mem_mb:.2f} MB")

if os.path.exists(MODEL_DIR):
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True)
else:
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
    model = GPT2LMHeadModel.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

tokenizer.pad_token = tokenizer.eos_token

df = pd.read_csv(CSV_PATH)
all_texts = df["text"].tolist()

wte = model.transformer.wte
wpe = model.transformer.wpe
drop = model.transformer.drop
ln_f = model.transformer.ln_f
lm_head = model.lm_head

optimizer = optim.Adam(list(ln_f.parameters()) + list(lm_head.parameters()), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()
training_log = []

# === Initialize shard connections ===
shard_sockets = {}
def connect_shards():
    for shard_id in range(NUM_SHARDS):
        ip, port = SHARD_ADDRS[shard_id]
        try:
            sock = socket.create_connection((ip, port), timeout=10)
            shard_sockets[shard_id] = sock
            print(f"[shard {shard_id}] Connected")
        except Exception as e:
            print(f"[shard {shard_id}] Connection error: {e}")
            shard_sockets[shard_id] = None

def close_shards():
    for shard_id, sock in shard_sockets.items():
        if sock:
            sock.close()
            print(f"[shard {shard_id}] Closed")

def save_shard_weights():
    for shard_id, sock in shard_sockets.items():
        if sock:
            try:
                print(f"[to shard {shard_id}] Sending save command")
                sock.send(b"save")
                ack = sock.recv(2)
                if ack == b"OK":
                    print(f"[from shard {shard_id}] Save acknowledged")
            except Exception as e:
                print(f"[shard {shard_id}] Save failed: {e}")

def run_shard(shard_id, tensor=None, grad=None):
    sock = shard_sockets.get(shard_id)
    if not sock:
        print(f"[shard {shard_id}] No connection")
        return torch.zeros_like(tensor if tensor is not None else grad)
    try:
        if tensor is not None:
            print(f"[to shard {shard_id}] Sending tensor, shape: {list(tensor.shape)}")
            sock.send(b"data")
            send_tensor(sock, tensor.cpu())
            result = recv_tensor(sock)
            print(f"[from shard {shard_id}] Received result, shape: {list(result.shape)}")
            return result
        elif grad is not None:
            print(f"[to shard {shard_id}] Sending gradient, shape: {list(grad.shape)}")
            sock.send(b"grad")
            send_tensor(sock, grad.cpu())
            result = recv_tensor(sock)
            print(f"[from shard {shard_id}] Received backward gradient, shape: {list(result.shape)}")
            return result
    except Exception as e:
        print(f"[shard {shard_id}] Communication error: {e}")
        return torch.zeros_like(tensor if tensor is not None else grad)

model.train()
connect_shards()

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    start_time = time.time()
    total_loss = 0.0
    log_memory_usage("start epoch")

    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = inputs["input_ids"]
        position_ids = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1)

        embedding = (wte(input_ids) + wpe(position_ids)).detach().requires_grad_()
        hidden_states = drop(embedding)
        log_memory_usage("after embedding")

        for shard_id in range(NUM_SHARDS):
            hidden_states = run_shard(shard_id, tensor=hidden_states)
            log_memory_usage(f"after forward shard {shard_id}")

        hidden_states = ln_f(hidden_states)
        logits = lm_head(hidden_states[:, :-1, :])

        labels = input_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad()

        try:
            loss.backward()
            log_memory_usage("after backward")
        except RuntimeError as e:
            print(f"Backward error: {e}")
            continue

        grad = embedding.grad if embedding.grad is not None else torch.zeros_like(embedding)
        for shard_id in reversed(range(NUM_SHARDS)):
            grad = run_shard(shard_id, grad=grad)
            log_memory_usage(f"after backward shard {shard_id}")

        optimizer.step()
        log_memory_usage("after optimizer step")
        print(f"\n[Batch {i // BATCH_SIZE + 1}] Loss = {loss.item():.4f}")
        total_loss += loss.item()

    duration = time.time() - start_time
    avg_loss = total_loss / (len(all_texts) // BATCH_SIZE)
    print(f"\nEpoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}, Time: {duration:.2f} sec")

    training_log.append({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "duration_sec": round(duration, 2)
    })

    log_df = pd.DataFrame(training_log)
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_df.to_csv(os.path.join(SAVE_DIR, "training_log.csv"), index=False)
    print("Training log saved")

    save_shard_weights()

close_shards()
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model and tokenizer saved to {SAVE_DIR}")
