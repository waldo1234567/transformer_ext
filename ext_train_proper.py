import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split,Dataset
from with_ext_class import TheTansformer
import requests
import re
from collections import Counter
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler, schedule
import torch.cuda.nvtx as nvtx
import time, statistics

seq_len = 64
batch_size = 128
gpu_times = []
host_times = []

def simple_tokenizer(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        toks = text.split()
        return toks

class SimpleVocab:
    def __init__(self, stoi, unk_index):
        self.stoi = stoi
        self.unk_index = unk_index
    def __call__(self, tokens):
        return [self.stoi.get(t, self.unk_index) for t in tokens]
    def __len__(self):
        return len(self.stoi)
    
class FastTextDataset(Dataset):
        def __init__(self, all_ids_tensor):
            self.data = all_ids_tensor
        def __len__(self):
            return self.data.size(0)
        def __getitem__(self, idx):
            return self.data[idx]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Downloading text...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text_lines = requests.get(url).text.splitlines()
    print("Lines:", len(text_lines))
            
    counter = Counter()
    for line in text_lines:
        counter.update(simple_tokenizer(line))
    tokens = [tok for tok, _ in counter.most_common()]
    all_tokens = ["<unk>"] + tokens
    stoi = {t: i for i, t in enumerate(all_tokens)}
    vocab = SimpleVocab(stoi, unk_index=0)

    tokenized_ids = []
    for line in text_lines:
        toks = simple_tokenizer(line)
        ids = vocab(toks)
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [vocab.unk_index] * (seq_len - len(ids))
        tokenized_ids.append(torch.tensor(ids, dtype=torch.long))
    all_ids = torch.stack(tokenized_ids)

    dataset = FastTextDataset(all_ids)
    train_iter = int(0.8 * len(dataset))
    val_iter = len(dataset) - train_iter
    train_ds, val_ds = random_split(dataset, [train_iter, val_iter], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    vocab_size = len(vocab)
    
    torch.backends.cudnn.benchmark = True
    d_model = 128
    model = TheTansformer(vocab=vocab_size, d_model=d_model, max_len=seq_len).to(device)
    head = nn.Linear(d_model, vocab_size).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    cutoffs = [2000,6000]
    adp = nn.AdaptiveLogSoftmaxWithLoss(in_features=d_model, n_classes=vocab_size,
                                        cutoffs=cutoffs, div_value=4.0).to(device)
    
    WARMUP = 5
    MEASURE = 50
    step = 0
    FULL_VAL_EVERY = 10
    QUICK_VAL_BATCHES = 2
    it = iter(train_loader)

    for _ in range(5):
        b = next(it); b = b.to(device, non_blocking=True)
        _ = head(model(b))

    for i in range(50):
        b = next(it)
        t0 = time.perf_counter()
        b = b.to(device, non_blocking=True)
        t1 = time.perf_counter()
        out = model(b)
        logits = head(out)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        host_times.append((t1 - t0) * 1000)
        gpu_times.append((t2 - t1) * 1000)
    print("host avg ms:", statistics.mean(host_times))
    print("gpu  avg ms:", statistics.mean(gpu_times))
    print("total avg ms:", statistics.mean([h+g for h,g in zip(host_times,gpu_times)]))
    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch = batch.to(device, non_blocking=True) 
            opt.zero_grad()
            out = model(batch)        
            pooled = out             
            logits = head(pooled)     
            outs = out.view(-1, d_model)
            targets = batch.view(-1).to(device)
            loss = adp(outs, targets).loss
            loss.backward()
            opt.step()
            if step % 8 == 0:
                model.ffn.sync_weight_t()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{10}  train loss: {avg_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= QUICK_VAL_BATCHES:
                    break
                batch = batch.to(device, non_blocking=True)
                out = model(batch)
                logits = head(out)
                outs = out.view(-1, d_model)
                targets = batch.view(-1).to(device)
                val_loss += adp(outs, targets).loss
        print(f"Quick val (first {QUICK_VAL_BATCHES} batches) loss: {val_loss/QUICK_VAL_BATCHES:.4f}")
        
        if(epoch + 1) % FULL_VAL_EVERY == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    out = model(batch)
                    logits = head(out)
                    outs = out.view(-1, d_model)
                    targets = batch.view(-1).to(device)
                    vloss = adp(outs, targets).loss
                    val_loss += vloss.item()
            print("  val loss:", val_loss / len(val_loader))
        
    log_dir = "./profiler_logs"
    prof_schedule = schedule(wait=1, warmup=1, active=3, repeat=1)

    it = iter(train_loader)
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
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
    ) as prof:
            
        for step in range(10):
            try:
                src = next(it).to(device)
            except StopIteration:
                it = iter(train_loader)
                src = next(it).to(device)
            opt = None
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)  
            opt.zero_grad()
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