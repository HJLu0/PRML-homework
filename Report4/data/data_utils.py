import torch
from torch.utils.data import Dataset, DataLoader
import random


PAD_IDX = 0
SOS_IDX = 1   # start-of-sequence
EOS_IDX = 2   # end-of-sequence
VOCAB_SIZE = 13   # 0(pad),1(sos),2(eos) + 数字 3~12（10个）
NUM_OFFSET = 3    # 数字 i 对应 token id = i + NUM_OFFSET


class ReverseDataset(Dataset):
    def __init__(self, num_samples=8000, min_len=3, max_len=12, seed=42):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            length = random.randint(min_len, max_len)
            nums   = [random.randint(0, 9) for _ in range(length)]
            # 数字 → token id
            src = [SOS_IDX] + [n + NUM_OFFSET for n in nums] + [EOS_IDX]
            tgt = [SOS_IDX] + [n + NUM_OFFSET for n in reversed(nums)] + [EOS_IDX]
            self.samples.append((src, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """将变长序列 pad 成等长 tensor"""
    srcs, tgts = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src  = max(src_lens)
    max_tgt  = max(tgt_lens)

    src_pad = torch.zeros(len(srcs), max_src, dtype=torch.long)
    tgt_pad = torch.zeros(len(tgts), max_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :len(s)] = torch.tensor(s)
        tgt_pad[i, :len(t)] = torch.tensor(t)

    return src_pad, tgt_pad


def get_dataloaders(batch_size=64, num_train=8000, num_val=1000, num_test=1000):
    """返回 train / val / test DataLoader"""
    train_ds = ReverseDataset(num_train,  seed=42)
    val_ds   = ReverseDataset(num_val,    seed=99)
    test_ds  = ReverseDataset(num_test,   seed=2024)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

# 序列准确率计算

def sequence_accuracy(logits, tgt, pad_idx=PAD_IDX):
    pred = logits.argmax(dim=-1)              # (batch, seq)
    mask = tgt != pad_idx
    correct = ((pred == tgt) & mask).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0