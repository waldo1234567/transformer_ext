import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


def causal_mask(T, device=None):
    #"""Lower-triangular boolean mask [T, T]"""
    return torch.tril(torch.ones(T,T,dtype=torch.bool, device=device))


def padding_mask_from_tokens(tokens, pad_token_id = 0):
    # tokens: [B, T]
    # returns: mask [B, T] boolean True = NOT padding (i.e., valid token)
    return(tokens != pad_token_id)
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_models ,num_heads):
        super().__init__()
        assert d_models % num_heads == 0
        self.h = num_heads
        self.dk = d_models // num_heads
        self.qkv = nn.Linear(d_models, 3 * d_models)
        self.out = nn.Linear(d_models, d_models)
        
    def _split_heads(self, x):
        B,T,D = x.shape
        return x.view(B,T, self.h, self.dk).transpose(1,2)

        
    def forward(self, q_in, kv_in, attn_mask = None):
        B, Tq, D = q_in.shape
        Tk = kv_in.shape[1]
        qkv = self.qkv(q_in)           # shape: [B, Tq, 3*D]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # split into Q,K,V
        q = self._split_heads(q)
        k = self._split_heads(kv_in)   # or split from qkv if computing K,V from same input
        v = self._split_heads(kv_in)
        
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.dk)
        
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-1e9"))
            
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        out = ctx.transpose(1,2).contiguous().view(B, Tq, D)
        return self.out(out)

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, src_mask = None):
        x = x + self.self_attn(self.ln1(x) , self.ln1(x), attn_mask=src_mask)
        x = x + self.ff(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )  
        
    def forward(self,x,enc_out, self_mask=None, enc_mask=None):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), attn_mask=self_mask)
        x = x + self.cross_attn(self.ln2(x), enc_out, attn_mask = enc_mask)
        x = x + self.ff(self.ln3(x))
        return x
    
# Full Transformer
      
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model = 64, n_layers = 2, n_heads=4, d_ff= 256, max_len=128, pad_token_id = 0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        #embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len , d_model)
        
        #encoder decoder stacks
        self.encoder_layers = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers))
        
        self.out_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, src_tokens, tgt_tokens):
        B, T_src = src_tokens.shape
        _, T_tgt = tgt_tokens.shape
        device = src_tokens.device
        
        #masks
        src_key_mask = padding_mask_from_tokens(src_tokens, pad_token_id=self.pad_token_id)
        src_mask = src_key_mask.unsqueeze(1).expand(B, T_src,T_src)
        
        tgt_key_mask = padding_mask_from_tokens(tgt_tokens, pad_token_id=self.pad_token_id)
        
        causal = causal_mask(T_tgt, device=device)
        
        tgt_key_mask_for_keys = tgt_key_mask.unsqueeze(1).expand(B,T_tgt, T_tgt)
        
        tgt_mask = tgt_key_mask_for_keys & causal.unsqueeze(0)
        
        #encoder embeddings
        src_pos = torch.arange(0, T_src, device=device).unsqueeze(0).expand(B,T_src)
        src = self.token_emb(src_tokens) * math.sqrt(self.d_model) + self.pos_emb(src_pos)
        
        enc = src
        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)
            
        tgt_pos = torch.arange(0, T_tgt, device=device).unsqueeze(0).expand(B, T_tgt)
        dec = self.token_emb(tgt_tokens) * math.sqrt(self.d_model) + self.pos_emb(tgt_pos)
        
        enc_key_mask = src_key_mask.unsqueeze(1).expand(B , T_tgt, T_src)
        
        for layer in self.decoder_layers:
            dec = layer(dec, enc, self_mask=tgt_mask, enc_mask = enc_key_mask)
            
        logits = self.out_linear(dec)
        return logits
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = 50
    pad_id = 0
    
    model = TransformerSeq2Seq(vocab_size=vocab, d_model=64, n_layers = 2, n_heads=4, d_ff = 256, max_len=64, pad_token_id=pad_id)
    model = model.to(device)
    
    # toy batch: random ints (avoid pad id for simplicity)
    B, T_src, T_tgt = 4, 7, 6
    src = torch.randint(1, vocab, (B, T_src), device=device)
    tgt_input = torch.randint(1, vocab, (B, T_tgt), device=device)
    
    tgt_labels = torch.randint(1, vocab, (B, T_tgt), device=device)
    
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    model.train()
    opt.zero_grad()
    
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    logits = model(src, tgt_input)  # [B, T_tgt, V]
    loss = F.cross_entropy(logits.view(-1, vocab), tgt_labels.view(-1))
    loss.backward()
    opt.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"loss: {loss.item():.4f}, fwd+back (s): {t1 - t0:.4f}")
    print("logits shape:", logits.shape)