from pure_pytorch_class import TransformerSeq2Seq
import random 
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch

class CopyDataset(Dataset):
    def __init__(self, vocab_size= 50, seq_len= 6, size=200):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size
        self.data=[]
        for _ in range(size):
            src = torch.randint(1, vocab_size, (seq_len,))
            tgt_in = torch.cat([torch.tensor([1]), src[:-1]])
            tgt_out = src.clone()
            self.data.append((src, tgt_in, tgt_out))
            
    def __len__(self): return self.size
    def __getitem__(self, index):
        return self.data[index]
    
def collate_batch(batch):
    srcs, tgt_incs, tgts = zip(*batch)
    return torch.stack(srcs), torch.stack(tgt_incs), torch.stack(tgts)

#hyperparams
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab = 100
pad_id = 0
seq_len = 6
full = CopyDataset(vocab_size=vocab, seq_len=seq_len, size=5000)  # one big pool
from torch.utils.data import random_split
train_ds, val_ds = random_split(full, [4800, 200], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_loader  = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_batch)

model = TransformerSeq2Seq(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, d_ff = 256, max_len=64, pad_token_id=pad_id)
model.to(device)

opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
from torch.optim.lr_scheduler import ReduceLROnPlateau
sched = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)


def eval_one_batch_on_model(model, src, tgt_in, tgt_out, vocab):
    model.eval()
    src,tgt_in, tgt_out = src.to(next(model.parameters()).device), tgt_in.to(next(model.parameters()).device), tgt_out.to(next(model.parameters()).device)
    with torch.no_grad():
        logits = model(src, tgt_in)
        preds = logits.argmax(dim = -1)
        loss = F.cross_entropy(logits.view(-1, vocab), tgt_out.view(-1)).item()
    seq_exact = (preds == tgt_out).all(dim=1).float().mean().item()
    token_acc = (preds == tgt_out).float().mean().item() 
    return loss, token_acc, seq_exact, preds, tgt_out  

def validate():
    model.eval()
    total, correct = 0,0
    val_loss = 0.0
    with torch.no_grad():
        for src, tgt_in, tgt_out in test_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.view(-1, vocab), tgt_out.view(-1))
            val_loss += float(loss) * src.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == tgt_out).all(dim=1).sum().item()
            total += src.size(0)
    return val_loss/total, correct/total


# src, tgt_in, tgt_out = next(iter(train_loader))
# loss_train_batch, token_acc_train, seq_exact_train, preds_train, tgt_train = eval_one_batch_on_model(model, src, tgt_in, tgt_out, vocab)
# print("TRAIN BATCH: loss", loss_train_batch, "token_acc", token_acc_train, "seq_exact", seq_exact_train)
# print("Preds (first 3), Targets (first 3):")
# for i in range(min(3, src.size(0))):
#     print("pred:", preds_train[i].tolist())
#     print(" tgt:", tgt_train[i].tolist())
#     print("---")


# src_v, tgt_in_v, tgt_out_v = next(iter(test_loader))
# loss_val_batch, token_acc_val, seq_exact_val, preds_val, tgt_val = eval_one_batch_on_model(model, src_v, tgt_in_v, tgt_out_v, vocab)
# print("VAL BATCH: loss", loss_val_batch, "token_acc", token_acc_val, "seq_exact", seq_exact_val)
# print("Preds (first 3), Targets (first 3):")
# for i in range(min(3, src_v.size(0))):
#     print("pred:", preds_val[i].tolist())
#     print(" tgt:", tgt_val[i].tolist())
#     print("---")

best_val = 1e9
for epoch in range(1,51):
    model.train()
    running_loss = 0.0
    for i,(src, tgt_in, tgt_out) in enumerate(train_loader, 1):
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        opt.zero_grad()
        logits=  model(src, tgt_in)
        loss = F.cross_entropy(logits.view(-1, vocab), tgt_out.view(-1))
        loss.backward()
        opt.step()
        running_loss += loss.item()
    
    device = next(model.parameters()).device
    train_loss = running_loss / len(train_loader)
    val_loss, val_acc = validate()
    sched.step(val_loss)
    print(f"Epoch {epoch:02d} train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_exact_acc {val_acc:.3f}")
    # checkpoint
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_simple_transformer.pt")

# print the first 20 mismatches from val
device = next(model.parameters()).device
model.eval()
count = 0
with torch.no_grad():
    for src, tgt_in, tgt_out in test_loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        preds = logits.argmax(dim=-1)
        for i in range(src.size(0)):
            if not (preds[i] == tgt_out[i]).all():
                print("SRC   :", src[i].tolist())
                print("TGT_IN:", tgt_in[i].tolist())
                print("TGT_OUT:", tgt_out[i].tolist())
                print("PRED  :", preds[i].tolist())
                print("---")
                count += 1
                if count >= 20:
                    break
        if count >= 20:
            break
print("printed", count, "mismatches")
