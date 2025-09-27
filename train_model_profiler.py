import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import time,os
from pure_pytorch_class import TransformerSeq2Seq

def make_model(device, vocab=5000, d_model=512, n_layers =2, n_heads=8,d_ff = 2048, max_len=512):
    return TransformerSeq2Seq(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, max_len=max_len, pad_token_id=0).to(device)

def make_batch(B=32, T_src=128, T_tgt=128, vocab=5000, device="cuda"):
    src = torch.randint(1, vocab, (B,T_src), device=device)
    tgt_in = torch.randint(1, vocab,(B, T_tgt), device=device)
    tgt_out = torch.randint(1, vocab,(B, T_tgt), device=device)
    return src,tgt_in, tgt_out

device = "cuda" if torch.cuda.is_available() else "cpu"

B=32
T_src = 128
T_tgt = 128
vocab = 5000
d_model = 512
n_layers = 4
n_heads = 8
d_ff = 2048

log_dir = "./tb_prof"
os.makedirs(log_dir, exist_ok=True)

print("Device:", device)
model = make_model(device, vocab=vocab, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, max_len=max(T_src, T_tgt))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

warmup = 8
print(f"Warming up ({warmup} steps)...")
for i in range(warmup):
    src, tgt_in, tgt_out = make_batch(B, T_src, T_tgt, vocab=vocab, device=device)
    opt.zero_grad()
    logits = model(src, tgt_in)
    loss = F.cross_entropy(logits.view(-1, vocab), tgt_out.view(-1))
    loss.backward()
    opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
        
        
# Profiling schedule: wait=1, warmup=1, active=6 -> captures a few iterations
print("Starting profiler...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=6, repeat=1),
    on_trace_ready=tensorboard_trace_handler(log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    iters = 10
    for step in range(iters):
        src, tgt_in, tgt_out = make_batch(B, T_src, T_tgt, vocab=vocab, device=device)
        opt.zero_grad()
        with record_function("model_forward"):
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.view(-1, vocab), tgt_out.view(-1))
        with record_function("model_backward"):
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        prof.step()
        
# Save a text summary: top ops by self CUDA time
print("\nProfiler run finished — printing top CUDA ops (self_cuda_time total):\n")
try:
    key_averages = prof.key_averages(group_by_input_shape=True)
except Exception:
    key_averages = prof.key_averages()

def get_cuda_time(item):
    # try a few common attribute names, fall back to 0
    return float(getattr(item, "self_cuda_time_total", 
               getattr(item, "cuda_time_total", 
               getattr(item, "cuda_time", 0))))

def get_cpu_time(item):
    return float(getattr(item, "self_cpu_time_total",
               getattr(item, "cpu_time_total",
               getattr(item, "cpu_time", 0))))

def get_mem_usage(item):
    # sum of cpu+cuda memory usage if available (best-effort)
    cpu_mem = float(getattr(item, "self_cpu_memory_usage", 
                  getattr(item, "cpu_memory_usage", 0)))
    cuda_mem = float(getattr(item, "self_cuda_memory_usage",
                   getattr(item, "cuda_memory_usage", 0)))
    return cpu_mem + cuda_mem

# Sort by CUDA time (descending). If no CUDA times present, fall back to CPU time.
sorted_ops = sorted(
    key_averages,
    key=lambda x: (get_cuda_time(x) if get_cuda_time(x) > 0 else get_cpu_time(x)),
    reverse=True
)

print(f"{'op_name':45s} {'count':>6s} {'cuda_ms':>12s} {'cpu_ms':>10s} {'mem_MB':>12s}")
for item in sorted_ops[:30]:
    name = item.key
    count = int(getattr(item, "count", 0))
    cuda_ms = get_cuda_time(item) / 1000.0
    cpu_ms = get_cpu_time(item) / 1000.0
    mem_mb = get_mem_usage(item) / (1024**2)
    print(f"{name:45.45s} {count:6d} {cuda_ms:12.3f} {cpu_ms:10.3f} {mem_mb:12.2f}")

print(f"\nTensorBoard traces are in: {log_dir}")
print("Run: tensorboard --logdir", log_dir, "and open the 'Profile' / 'Trace' view.")
