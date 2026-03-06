import torch
import torch.nn as nn
import torch.nn.functional as F

text = open("data.txt", encoding="utf-8").read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 128
embed_size = 128
heads = 4
layers = 2


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        T = x.shape[0]

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) / (k.shape[-1]**0.5)

        weights = weights.masked_fill(self.mask[:T,:T]==0,float("-inf"))

        weights = F.softmax(weights, dim=-1)

        v = self.value(x)

        return weights @ v


class MultiHead(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(head_size*num_heads, embed_size)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size)
        )

    def forward(self,x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = embed_size // heads

        self.attn = MultiHead(heads, head_size)
        self.ff = FeedForward()

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self,x):

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x


class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])

        self.ln = nn.LayerNorm(embed_size)

        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self,x):

        x = self.embedding(x)

        x = self.blocks(x)

        x = self.ln(x)

        return self.head(x)


model = MiniGPT()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

steps = 20000

for step in range(steps):

    start = torch.randint(len(data)-block_size,(1,)).item()

    x = data[start:start+block_size]
    y = data[start+1:start+block_size+1]

    logits = model(x)

    loss = loss_fn(logits.view(-1,vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%500==0:
        print(step,loss.item())


torch.save({
    "model_state":model.state_dict(),
    "stoi":stoi,
    "itos":itos
},"model.pth")

print("training complete")