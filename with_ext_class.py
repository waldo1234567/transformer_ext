import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
import sys 
from torch.amp import autocast, GradScaler

_ext_names = ["matmul_gelu_ext", "custom_extension", "matmul_gelu", "matmul_gelu_module"]
ext = None
for name in _ext_names:
    try:
        ext = __import__(name)
        print("Loaded extension module:", name)
        break
    except Exception:
        pass
if ext is None:
    raise ImportError("Could not import compiled extension. Make sure your extension is on PYTHONPATH and name is one of: " + ", ".join(_ext_names))

assert hasattr(ext, "forward") and hasattr(ext, "backward"), "Extension must expose forward and backward functions"

scaler = GradScaler()

class MatmulGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A,B,scale:float):
        C,Cpre = ext.forward(A,B,float(scale))
        Cpre_for_backward = Cpre
        if Cpre.dtype != A.dtype:
            Cpre_for_backward = Cpre.to(A.dtype)
        ctx.save_for_backward(A,B,Cpre_for_backward)
        ctx.scale = float(scale)
        return C
    
    @staticmethod
    def backward(ctx, grad_outputs):
        A,B,Cpre = ctx.saved_tensors
        target_dtype = A.dtype
        if Cpre.dtype != target_dtype:
            Cpre = Cpre.to(target_dtype)
        if grad_outputs.dtype != target_dtype:
            grad_outputs = grad_outputs.contiguous().to(target_dtype)
        scale = ctx.scale
        grads = ext.backward(A,B,Cpre, grad_outputs, float(scale))
        gradA, gradB = grads[0], grads[1]
        return gradA, gradB, None
    
def matmul_gelu(A,B,scale=1.0):
    return MatmulGELUFunction.apply(A , B, float(scale))

class FusedFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)
        w1_t = self.w1.weight.detach().t().contiguous()
        self.register_buffer("w1_t", w1_t)
        self.w1_t.requires_grad = False
    
    def sync_weight_t(self):
        with torch.no_grad():
            self.w1_t.copy_(self.w1.weight.detach().t().contiguous())
        
    def forward(self , x):
        B,T,D = x.shape
        M = B * T
        A = x.reshape(M,D)
        w1_t = self.w1_t
        C = matmul_gelu(A, w1_t, 1.0)
        if self.w1.bias is not None:
            C = C + self.w1.bias.unsqueeze(0)
        C = C.view(B,T,-1)
        return self.w2(C)
    

class TheTansformer(nn.Module):
    def __init__(self, vocab=100, d_model = 128, n_heads=8, d_ff=512, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model))
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn =  FusedFFN(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, src):
        B,T = src.shape
        x = self.token_emb(src) + self.pos_emb[:, :T, :]
        x2,_ = self.attn(x,x,x)
        x = x + x2
        x = self.ln1(x)
        x = x + self.ffn(x)
        x = self.ln2(x)
        return x
    

def compute_token_seq_metrics(logits, targets):
    B,T,V = logits.shape
    preds = logits.argmax(dim=-1)
    correct_tokens = (preds == targets).sum().item()
    total_tokens = B * T
    exact_matches = (preds == targets).all(dim=1).sum().item()
    return correct_tokens, total_tokens, exact_matches, preds  

from torch.utils.data import Dataset, DataLoader,random_split

class CopyDataset(Dataset):
    def __init__(self, vocab_size = 100 , seq_len=6, size=2000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.data=[]
        for _ in range(size):
            src = torch.randint(1, vocab_size, (seq_len,), dtype=torch.long)
            self.data.append(src)
    
    def __len__ (self): return self.size
    def __getitem__(self, index):
        return self.data[index]
    
def collate_batch(batch):
    src = torch.stack(batch)
    return src

def train_and_profile(
    device="cuda",
    d_model=128,
    vocab=1000,
    seq_len=20,
    pool_size=5000,
    train_frac=0.9,
    batch_size=32,
    num_epochs=50,
    lr=1e-3,
    print_every=1
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    
    full  = CopyDataset(vocab, seq_len=seq_len, size=pool_size)
    train_n = int(pool_size * train_frac)
    val_n = pool_size - train_n
    
    train_ds, val_ds = random_split(full, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader =  DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)    
    
    model = TheTansformer(vocab=vocab, d_model=128, n_heads=8 ,d_ff=512, max_len=seq_len).to(device)

    head = nn.Linear(d_model, vocab).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr = lr)
    
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state=None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_exact = 0
        steps = 0
        for src in train_loader:
            src = src.to(device)
            opt.zero_grad()
            with autocast(device_type="cuda"):
                out = model(src)
                logits = head(out)
                loss = criterion(logits.view(-1, logits.size(-1)), src.view(-1))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.ffn.sync_weight_t()
            
            with torch.no_grad():
                ct, tt, ex, _ = compute_token_seq_metrics(logits, src)
                total_correct += ct
                total_tokens += tt
                total_exact += ex
            total_loss += loss.item()
            steps += 1
            
        train_loss = total_loss / steps
        train_token_acc = total_correct / total_tokens
        train_seq_exact = total_exact / (steps * batch_size)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_tokens = 0
        val_correct = 0
        val_exact = 0
        sample_preds = []
        
        with torch.no_grad():
            for src in val_loader:
                src = src.to(device)
                out = model(src)
                logits = head(out)
                loss = criterion(logits.view(-1, logits.size(-1)), src.view(-1))
                val_loss += loss.item()
                val_steps += 1
                ct, tt, ex, preds = compute_token_seq_metrics(logits, src)
                val_correct += ct
                val_tokens += tt
                val_exact += ex
                if len(sample_preds) < 3:
                    sample_preds.append((src.cpu(), preds.cpu()))
                    
        val_loss = val_loss / max(1, val_steps)
        val_token_acc = val_correct / val_tokens
        val_seq_exact = val_exact / (val_steps * batch_size)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model": model.state_dict(),
                "head": head.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
            }
        if epoch % print_every == 0:
            print(f"Epoch {epoch:02d} train_loss {train_loss:.4f} val_loss {val_loss:.4f}"
                  f" train_token_acc {train_token_acc:.4f} val_token_acc {val_token_acc:.4f}"
                  f" train_seq_exact {train_seq_exact:.4f} val_seq_exact {val_seq_exact:.4f}")

            for src_cpu, preds in sample_preds:
                for i in range(min(3, src_cpu.shape[0])):
                    print("SRC  :", src_cpu[i].tolist())
                    print("PRED :", preds[i].tolist())
                    print("---")

    if best_state:
        model.load_state_dict(best_state["model"])
        head.load_state_dict(best_state["head"])
        print("Restored best model from epoch", best_state["epoch"], "val_loss", best_val_loss)

    return model, head, train_loader, val_loader


def profile_model(model, head, num_epochs ,loader, device="cuda", log_dir="./tb_prof_ext"):
    import torch.cuda.nvtx as nvtx
    model.eval()
    head.eval()
  
    print("Warming up 10 steps...") 
    it = iter(loader)
    for _ in range(10):
        src = next(it).to(device)
        with torch.no_grad():
            nvtx.range_push("encoder_block")
            out = model(src)
            nvtx.range_pop()
            nvtx.range_push("head_linear")
            logits = head(out)
            nvtx.range_pop()
        
    print("Starting profiling run (writes tb traces to", log_dir, ")")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=6, repeat=1),
        on_trace_ready=tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        
        for step in range(num_epochs):
            try:
                src = next(it).to(device)
            except StopIteration:
                it = iter(loader)
                src = next(it).to(device)
            opt = None
           
            opt = torch.optim.SGD(model.parameters(), lr=1e-3) 

            with record_function("model_forward"):
                nvtx.range_push("encoder_block")
                out = model(src)
                nvtx.range_pop()
                nvtx.range_push("head_linear")
                logits = head(out)
                nvtx.range_pop()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), src.view(-1))
            with record_function("model_backward"):
                loss.backward()
                opt.step()
                model.ffn.sync_weight_t()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prof.step()

    ka = prof.key_averages()
    print("=== sample key_averages[0] attrs ===")
    if len(ka) > 0:
        item0 = ka[0]
        print("repr:", item0)
        print("dir keys:", [k for k in dir(item0) if not k.startswith("_")][:120])
        for name in ("self_cuda_time_total", "cuda_time_total", "cuda_time", "self_cpu_time_total", "cpu_time_total"):
            print(f"{name}: ", getattr(item0, name, None))
    print("=== end sample ===\n")

    try:
        print("Profiler table sorted by CUDA time (if available):")
        print(ka.table(sort_by="self_cuda_time_total", row_limit=20))
    except Exception as e:
        try:
            print("Fallback table sorted by cuda_time_total:")
            print(ka.table(sort_by="cuda_time_total", row_limit=20))
        except Exception:
            print("Could not use key_averages.table(), falling back to manual aggregation below.")

    def get_cuda_ms(ev):
        return float(getattr(ev, "self_cuda_time_total",
                    getattr(ev, "cuda_time_total",
                    getattr(ev, "cuda_time", 0)))) / 1000.0

    def get_cpu_ms(ev):
        return float(getattr(ev, "self_cpu_time_total",
                    getattr(ev, "cpu_time_total",
                    getattr(ev, "cpu_time", 0)))) / 1000.0

    from collections import defaultdict
    agg = defaultdict(lambda: {"count":0, "cuda":0.0, "cpu":0.0})
    total_cuda = 0.0
    for ev in ka:
        base = ev.key.split("(")[0].strip()
        c = int(getattr(ev, "count", 0))
        cuda_ms = get_cuda_ms(ev)
        cpu_ms  = get_cpu_ms(ev)
        agg[base]["count"] += c
        agg[base]["cuda"]  += cuda_ms
        agg[base]["cpu"]   += cpu_ms
        total_cuda += cuda_ms

    rows = sorted(agg.items(), key=lambda kv: kv[1]["cuda"], reverse=True)
    print(f"{'op':35s} {'count':>6s} {'cuda_ms':>10s} {'cuda_%':>8s} {'cpu_ms':>10s}")
    for name, v in rows[:30]:
        cuda_ms = v["cuda"]
        print(f"{name:35.35s} {v['count']:6d} {cuda_ms:10.3f} {100.0 * (cuda_ms/total_cuda if total_cuda>0 else 0):7.2f}% {v['cpu']:10.3f}")
    print(f"\nTotal CUDA ms summed across ops: {total_cuda:.3f} ms")

    
torch.backends.cudnn.benchmark = True
def new_bench(A, W, W1_t=None, W2=None, repeats=80, warmup=10, mode="forward", fused_fn=None):
    import time, torch, statistics
    device = A.device
    for _ in range(warmup):
        if mode=='forward':
            if fused_fn:
                _ = fused_fn(A, W)
            else:
                C = A.mm(W)
                _ = 0.5 * C * (1 + torch.tanh(0.7978845608 * (C + 0.044715 * C**3)))
        else: 
            A2 = A.clone().requires_grad_()
            W2c = W.clone().requires_grad_()
            if fused_fn:
                C = fused_fn(A2, W2c)
                L = C.sum()
                L.backward()
            else:
                C = A2.mm(W2c)
                C = 0.5 * C * (1 + torch.tanh(0.7978845608 * (C + 0.044715 * C**3)))
                L = C.sum()
                L.backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if mode=='forward':
            if fused_fn:
                _ = fused_fn(A, W)
            else:
                C = A.mm(W)
                _ = 0.5 * C * (1 + torch.tanh(0.7978845608 * (C + 0.044715 * C**3)))
        else:
            A2 = A.clone().requires_grad_()
            W2c = W.clone().requires_grad_()
            if fused_fn:
                C = fused_fn(A2, W2c)
                L = C.sum()
                L.backward()
            else:
                C = A2.mm(W2c)
                C = 0.5 * C * (1 + torch.tanh(0.7978845608 * (C + 0.044715 * C**3)))
                L = C.sum()
                L.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times)>1 else 0.0,
        "min_s": min(times),
        "max_s": max(times),
        "samples": times
    }
if __name__ == "__main__":
    # choose run mode: "micro" or "train"
    import sys
    mode = sys.argv[1] if len(sys.argv)>1 else "micro"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, head, train_loader, val_loader = None, None, None, None
    if mode == "micro":
        import csv
        sweeps = [
        {"B":32,"T":6,"d_model":128,"d_ff":512,"vocab":100},
        {"B":32,"T":6,"d_model":128,"d_ff":512,"vocab":1000},
        {"B":32,"T":20,"d_model":128,"d_ff":512,"vocab":1000},
        {"B":64,"T":20,"d_model":256,"d_ff":1024,"vocab":5000},
            ]
        rows = []
        for cfg in sweeps:
            B,T,d_model,d_ff,vocab = cfg["B"],cfg["T"],cfg["d_model"],cfg["d_ff"],cfg["vocab"]
            M = B*T
            A = torch.randn(M,d_model, device=device)
            W = torch.randn(d_model, d_ff, device=device)
            
            fb = new_bench(A, W, repeats=40, warmup=10, mode='forward', fused_fn=None)
            ff = new_bench(A, W, repeats=40, warmup=10, mode='forward', fused_fn=matmul_gelu)
            bb = new_bench(A, W, repeats=40, warmup=10, mode='forward+back', fused_fn=None)
            bf = new_bench(A, W, repeats=40, warmup=10, mode='forward+back', fused_fn=matmul_gelu)
            rows.append((cfg, fb["median_s"], ff["median_s"], bb["median_s"], bf["median_s"]))
        with open("bench_sweep.csv","w",newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["B","T","d_model","d_ff","vocab","fwd_baseline_s","fwd_fused_s","fwdback_baseline_s","fwdback_fused_s"])
            for cfg, fb, ff, bb, bf in rows:
                writer.writerow([cfg["B"],cfg["T"],cfg["d_model"],cfg["d_ff"],cfg["vocab"],fb,ff,bb,bf])
        print("Saved bench_sweep.csv")
    else:
        num_epochs=50
        model, head, train_loader, val_loader = train_and_profile(device=device,
            vocab=2000, seq_len=30, pool_size=5000, train_frac=0.9,
            batch_size=64, num_epochs=num_epochs, lr=1e-3)
        
        profile_model(model, head, num_epochs,train_loader if train_loader is not None else val_loader, device=device, log_dir="./tb_prof_ext")
    
        
