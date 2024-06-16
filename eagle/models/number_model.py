import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FeedFoward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    # https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
    def __init__(self, n_embed, n_head, dropout=0.0):
        super().__init__()
        assert n_embed % n_head == 0, "embed_dim is indivisible by num_heads"
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.n_embed = n_embed
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, num_tokens, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv
        use_dropout = 0.0 if not self.training else self.dropout
        context_vec = nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.n_embed)
        context_vec = self.proj(context_vec)
        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, dropout=0.0):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, n_head, dropout=dropout)
        self.ffwd = FeedFoward(n_embed, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# 0..9, 10 is EOS, 11 is PAD
class NumberModel(nn.Module):
    def __init__(self, decoder_dim: int = 256, decoder_num_heads: int = 8):
        super().__init__()
        self.cnn = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.last_cnn_dim = self.cnn.feature_info[-1]["num_chs"]
        self.proj = nn.Linear(self.last_cnn_dim, decoder_dim)
        self.decoder = TransformerBlock(n_embed=decoder_dim, n_head=decoder_num_heads)
        self.embedding = nn.Embedding(12, decoder_dim, padding_idx=11)
        self.head = nn.Linear(decoder_dim, 12)

    def forward(self, image, targets=None):
        img_embed = self.cnn(image)
        x = self.proj(img_embed).unsqueeze(1)  # on seq_len dimension
        if targets is not None:
            target_embed = self.embedding(targets)
            x = torch.cat([x, target_embed], axis=1)  # on seq_len dimension
        # x shape batch_size, 5, decoder_dim
        out = self.decoder(x)
        logits = self.head(out)
        return logits

    @torch.no_grad()
    def generate(self, image, threshold=0.5):
        batch_size = image.shape[0]
        img_embed = self.cnn(image)
        cond = self.proj(img_embed).unsqueeze(1)
        gen_results = torch.empty((batch_size, 4), dtype=torch.int64, device=image.device)  # initialize tensor for results
        for i in range(4):
            out = self.decoder(cond)
            logits = self.head(out)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1)
            print(probs.shape)  # torch.Size([1, 12])
            if probs[0, idx_next] < threshold:
                break
            gen_results[:, i] = idx_next
            cond = torch.cat((cond, self.embedding(idx_next).unsqueeze(1)), dim=1)

        return gen_results

    def postprocess(self, results_tensor):
        res = []
        results_tensor = results_tensor.cpu().numpy()
        for item in results_tensor:
            indiv = []
            for x in item:
                x = int(x)
                if x >= 10:
                    break
                else:
                    indiv.append(str(x))
            indiv = "".join(indiv)
            if len(indiv) == 0:
                res.append("")
            else:
                res.append(int(indiv))
        return res
