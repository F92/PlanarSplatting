import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ====== 手动伪数据 ======
def create_manual_data():
    # B=1, M=3
    g = torch.tensor([[
        [0.0, 0.0, 0.1,   0, 0, 1,   0.05, 0.01, 100, 0.02],   # 平面1
        [0.2, 0.0, 0.0,   0, 1, 0,   0.20, 0.05, 200, 0.10],   # 平面2
        [0.0, 0.3, 0.0,   1, 0, 0,   0.15, 0.02, 150, 0.05],   # 平面3
    ]], dtype=torch.float32)

    c = torch.tensor([[
        [1, 0.8, 0.1, 0],   # 平面1
        [0, 0.0, 1.0, 1],   # 平面2
        [1, 0.6, 0.3, 0],   # 平面3
    ]], dtype=torch.float32)

    p = torch.tensor([[
        [0.01, 0.02, 0.10,   0, 0, 1,   0.01, 120],   # 点1
        [0.20, 0.01, 0.00,   0, 1, 0,   0.05, 180],   # 点2
        [0.00, 0.29, 0.00,   1, 0, 0,   0.02, 160],   # 点3
    ]], dtype=torch.float32)

    r = torch.tensor([[
        [[0,0,0,0], [1,0,0,0], [0,1,0,0]],   # 平面1 vs others
        [[1,0,0,0], [0,0,0,0], [0,0,1,0]],   # 平面2 vs others
        [[0,1,0,0], [0,0,1,0], [0,0,0,0]]    # 平面3 vs others
    ]], dtype=torch.float32)

    pos = torch.tensor([[
        [0.0, 0.0, 0.1],
        [0.2, 0.0, 0.0],
        [0.0, 0.3, 0.0],
    ]], dtype=torch.float32)

    normal = torch.tensor([[
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]], dtype=torch.float32)

    score = torch.tensor([[1, 0, 1]], dtype=torch.float32)  # (1, M)

    return {"g": g, "c": c, "p": p, "r": r, "pos": pos, "normal": normal, "score": score}


# ====== 数据集类 ======
class GraspDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data["g"].shape[0]  # B
    def __getitem__(self, idx):
        return (
            self.data["g"][idx],
            self.data["c"][idx],
            self.data["p"][idx],
            self.data["r"][idx],
            self.data["pos"][idx],
            self.data["normal"][idx],
            self.data["score"][idx],
        )


# ====== 模型定义（简化版 Attention+MLP） ======
class GraspNetWithAttention(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.map_g = nn.Linear(10, d)
        self.map_c = nn.Linear(4, d)
        self.map_p = nn.Linear(8, d)

        self.attn1 = nn.MultiheadAttention(embed_dim=3*d, num_heads=4, batch_first=True)
        self.map_r = nn.Linear(4, 1)
        self.qkv = nn.Linear(3*d, 3*d)

        self.mlp_pos = nn.Sequential(nn.Linear(3*d, 64), nn.ReLU(), nn.Linear(64, 6))
        self.mlp_score = nn.Sequential(nn.Linear(4*d, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, g, c, p, r):
        B, M, _ = g.shape
        g, c, p = self.map_g(g), self.map_c(c), self.map_p(p)
        feat = torch.cat([g, c, p], dim=-1)  # (B, M, 3d)

        f_prime, _ = self.attn1(feat, feat, feat)

        bias = self.map_r(r).squeeze(-1)  # (B, M, M)
        Q, K, V = self.qkv(f_prime).chunk(3, dim=-1)
        attn_weights = (Q @ K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
        attn_weights = attn_weights + bias
        attn_weights = torch.softmax(attn_weights, dim=-1)
        z = attn_weights @ V
        z_global = z.mean(dim=1)

        pos_pred = self.mlp_pos(f_prime)
        z_expand = z_global.unsqueeze(1).expand(B, M, -1)
        score_pred = self.mlp_score(torch.cat([f_prime, z_expand], dim=-1))

        return pos_pred, score_pred.squeeze(-1)


# ====== 训练循环 ======
def train(model, dataloader, epochs=5, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for g, c, p, r, gt_pos, gt_normal, gt_score in dataloader:
            pos_pred, score_pred = model(g, c, p, r)
            pred_pos, pred_norm = pos_pred[..., :3], pos_pred[..., 3:]

            # 损失函数
            loss_pos = ((pred_pos - gt_pos) ** 2).mean()
            cos_sim = (pred_norm * gt_normal).sum(-1) / (
                pred_norm.norm(dim=-1) * gt_normal.norm(dim=-1) + 1e-6)
            loss_normal = (1 - cos_sim).mean()
            loss_score = nn.BCELoss()(score_pred, gt_score)

            loss = 1.0 * loss_pos + 0.5 * loss_normal + 1.0 * loss_score

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Epoch {epoch+1}] Loss = {loss.item():.4f}")
        print("预测点位:", pred_pos.detach().cpu().numpy().round(3))
        print("真实点位:", gt_pos.cpu().numpy())
        print("预测分数:", score_pred.detach().cpu().numpy().round(3))
        print("真实分数:", gt_score.cpu().numpy())


# ====== 主程序 ======
if __name__ == "__main__":
    data = create_manual_data()
    dataset = GraspDataset(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = GraspNetWithAttention()
    train(model, dataloader, epochs=10, lr=1e-3)
