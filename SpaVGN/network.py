import torch
import math


class PatchEmbed(torch.nn.Module):
    def __init__(self, patch_size=16, in_channel=1, embed_dim=1024):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (b, 1, h, w) -> (b, embed_dim, h/patch_size, w/patch_size)
        # flatten 并 transpose 得 (b, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(torch.nn.Module):
    def __init__(self, dim=1024, num_heads=8, drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = torch.nn.Linear(dim, dim*3, bias=True)
        self.drop1 = torch.nn.Dropout(drop_ratio)
        self.proj = torch.nn.Linear(dim, dim)
        self.drop2 = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, int(D / self.num_heads)).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        att = att.softmax(dim=-1)
        att = self.drop1(att)
        x = (att @ v).transpose(1, 2).flatten(2)  # (B, N, D)
        x = self.drop2(self.proj(x))
        return x


class Mlp(torch.nn.Module):
    def __init__(self, in_dim=1024, drop_ratio=0.):
        super(Mlp, self).__init__()
        self.fc1 = torch.nn.Linear(in_dim, in_dim*2)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(in_dim*2, in_dim)
        self.drop = torch.nn.Dropout(drop_ratio)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, sigma=1.0):

        super(GNNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.sigma = sigma



    def forward(self, x, pos):
        B, N, _ = x.shape

        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.sqrt((pos_diff ** 2).sum(dim=-1) + 1e-8)  # (B, N, N)

        adj = torch.exp(- (dist ** 2) / (2 * self.sigma ** 2))  # (B, N, N)
 
        top_k = 4  
        values, indices = torch.topk(adj, k=top_k, dim=-1)  # (B, N, top_k)
        mask = torch.zeros_like(adj)
        mask = mask.scatter_(-1, indices, 1.0) 
    

        adj = adj * mask
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-8)
    

        out = torch.bmm(adj, x)
        out = self.linear(out)
        return out
 

class BlockWithGNN(torch.nn.Module):
    def __init__(self, in_dim=1024, num_heads=8, drop_ratio=0., sigma=1.0):
        super(BlockWithGNN, self).__init__()
        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.attn = Attention(dim=in_dim, num_heads=num_heads, drop_ratio=drop_ratio)
        self.norm_gnn = torch.nn.LayerNorm(in_dim)
        self.gnn = GNNLayer(in_dim, in_dim, sigma=sigma)
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = Mlp(in_dim=in_dim, drop_ratio=drop_ratio)
        self.drop = torch.nn.Dropout(0.)

    def forward(self, x, pos):
        # Transformer Attention 
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.gnn(self.norm_gnn(x), pos))
        # MLP 
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(torch.nn.Module):
    def __init__(self, patch_size=16, in_c=1, embed_dim=1024, depth=12, num_heads=8, drop_ratio=0., sigma=1.0):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_c, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, in_c, kernel_size=3, padding=1),  
            torch.nn.ReLU()
        )
        
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channel=in_c, embed_dim=embed_dim)
        self.pos_drop = torch.nn.Dropout(p=drop_ratio)
        
        self.blocks = torch.nn.ModuleList([
            BlockWithGNN(in_dim=embed_dim, num_heads=num_heads, drop_ratio=drop_ratio, sigma=sigma)
            for _ in range(depth)
        ])
        

        self.final_conv = torch.nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, x):
        device = x.device
        b, _, h, w = x.shape
        
        x = self.cnn(x)
        pri_x = x  
        

        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w
        pos_enc = absolute_position_encoding(num_patches, self.embed_dim).to(device)
        x = self.patch_embed(x)
        x = self.pos_drop(x + pos_enc)  # (B, N, embed_dim)
        

        grid_y, grid_x = torch.meshgrid(torch.arange(num_patches_h), torch.arange(num_patches_w), indexing='ij')
        pos = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).float().to(device)  # (N, 2)
        pos = pos.unsqueeze(0).expand(b, -1, -1)  # (B, N, 2)

        for block in self.blocks:
            x = block(x, pos)
        

        x = x.reshape(b, -1, int(self.embed_dim//4), 4).transpose(1, 3).reshape(
            b, 4, int(self.embed_dim//4), int(h/self.patch_size), int(w/self.patch_size))
        

        fina_x = torch.zeros((b, 4, h, w), device=device)
        k = 0
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                fina_x[:, :, i::self.patch_size, j::self.patch_size] = x[:, :, k, :, :]
                k += 1
                

        fina_x = self.final_conv(fina_x)  
        out = pri_x + fina_x
        
        return out


def absolute_position_encoding(seq_len, embed_dim):
    seq_len = int(seq_len)
    pos_enc = torch.zeros((seq_len, embed_dim))
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pos_enc[pos, i] = math.sin(pos / (10000 ** (2*i / embed_dim)))
            if i + 1 < embed_dim:
                pos_enc[pos, i + 1] = math.cos(pos / (10000 ** (2*i / embed_dim)))
    return pos_enc
