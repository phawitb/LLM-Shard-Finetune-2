# === shard_worker.py ===
import sys
import os
import socket
import torch
import torch.optim as optim
from transformers import GPT2LMHeadModel
from utils import split_model, send_tensor, recv_tensor

# === Config ===
SHARD_INDEX = int(sys.argv[1])
PORT = 9000 + SHARD_INDEX
DEVICE = "cpu"
MODEL_PATH = "./distilgpt2_local"
NUM_SHARDS = 5

# === Load model blocks for this shard ===
print(f"[SHARD {SHARD_INDEX}] Loading model from {MODEL_PATH}...")
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
blocks = split_model(model, f"device{SHARD_INDEX}", num_shards=NUM_SHARDS)

# === Try load fine-tuned weights ===
weights_path = f"shard_{SHARD_INDEX}_weights.pt"
if os.path.exists(weights_path):
    weights = torch.load(weights_path)
    for block, weight in zip(blocks, weights):
        block.load_state_dict(weight)
    print(f"[SHARD {SHARD_INDEX}] Loaded fine-tuned weights")
else:
    print(f"[SHARD {SHARD_INDEX}] No fine-tuned weights found, using pretrained")

# === Setup optimizer ===
params = []
for block in blocks:
    params += list(block.parameters())
optimizer = optim.Adam(params, lr=5e-5)

# === TCP Server Setup ===
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('0.0.0.0', PORT))
server.listen(1)
print(f"[SHARD {SHARD_INDEX}] Ready and listening on port {PORT}")

while True:
    conn, addr = server.accept()
    print(f"[SHARD {SHARD_INDEX}] Connection from {addr}")
    with conn:
        try:
            input_tensor = None
            output_tensor = None
            while True:
                header = conn.recv(4)
                if not header or len(header) < 4:
                    print(f"[SHARD {SHARD_INDEX}] Empty or incomplete header")
                    break

                if header == b'save':
                    weights = [block.state_dict() for block in blocks]
                    torch.save(weights, weights_path)
                    print(f"[SHARD {SHARD_INDEX}] Weights saved to {weights_path}")
                    conn.send(b"OK")

                elif header == b'data':
                    print(f"[SHARD {SHARD_INDEX}] Receiving forward tensor")
                    input_tensor = recv_tensor(conn).to(DEVICE)
                    input_tensor.requires_grad = True
                    output_tensor = input_tensor
                    for block in blocks:
                        output_tensor = block(output_tensor)[0]
                    send_tensor(conn, output_tensor.detach().cpu())
                    print(f"[SHARD {SHARD_INDEX}] Sent forward output")

                elif header == b'grad':
                    print(f"[SHARD {SHARD_INDEX}] Receiving gradient")
                    grad_tensor = recv_tensor(conn).to(DEVICE)
                    if output_tensor is None or input_tensor is None:
                        raise ValueError("Forward pass must occur before backward")
                    output_tensor.backward(grad_tensor)
                    grad_input = input_tensor.grad.detach()
                    send_tensor(conn, grad_input.cpu())
                    optimizer.step()
                    optimizer.zero_grad()
                    print(f"[SHARD {SHARD_INDEX}] Backward pass and update complete")

                else:
                    print(f"[SHARD {SHARD_INDEX}] Unknown header: {header}")
                    break

        except Exception as e:
            print(f"[SHARD {SHARD_INDEX}] ERROR: {e}")
