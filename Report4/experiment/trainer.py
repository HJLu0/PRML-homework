import time
import torch
import torch.nn as nn
from data_utils import PAD_IDX, sequence_accuracy

class WarmupScheduler:
    """
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
    """
    def __init__(self, optimizer, d_model, warmup_steps=200):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        self.step_num     = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']



class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, padding_idx=PAD_IDX, smoothing=0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.padding_idx = padding_idx
        self.vocab_size  = vocab_size

    def forward(self, logits, target):
        """
        logits: (N, vocab)
        target: (N,)
        """
        log_prob = torch.log_softmax(logits, dim=-1)
        # 平滑目标分布
        smooth_val = self.smoothing / (self.vocab_size - 2)
        with torch.no_grad():
            true_dist = torch.full_like(log_prob, smooth_val)
            true_dist[:, self.padding_idx] = 0
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
            mask = (target == self.padding_idx)
            true_dist[mask] = 0

        loss = -(true_dist * log_prob).sum(dim=-1)
        return loss[~mask].mean()



def train_epoch(model, loader, optimizer, criterion, scheduler, device, is_transformer=True):
    model.train()
    total_loss, total_acc, n_batch = 0., 0., 0

    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)

        # 解码器输入 = tgt 去掉最后一个 token（teacher forcing）
        tgt_in  = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        if is_transformer:
            logits = model(src, tgt_in, pad_idx=PAD_IDX)
        else:
            logits = model(src, tgt_in, pad_idx=PAD_IDX)

        # logits: (batch, seq, vocab) → (batch*seq, vocab)
        b, s, v = logits.size()
        loss = criterion(logits.reshape(b * s, v), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        acc = sequence_accuracy(logits, tgt_out)
        total_loss += loss.item()
        total_acc  += acc
        n_batch    += 1

    return total_loss / n_batch, total_acc / n_batch


def evaluate(model, loader, criterion, device, is_transformer=True):
    model.eval()
    total_loss, total_acc, n_batch = 0., 0., 0

    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            if is_transformer:
                logits = model(src, tgt_in, pad_idx=PAD_IDX)
            else:
                logits = model(src, tgt_in, pad_idx=PAD_IDX)

            b, s, v = logits.size()
            loss = criterion(logits.reshape(b * s, v), tgt_out.reshape(-1))
            acc  = sequence_accuracy(logits, tgt_out)

            total_loss += loss.item()
            total_acc  += acc
            n_batch    += 1

    return total_loss / n_batch, total_acc / n_batch



def train_model(model, train_loader, val_loader,
                d_model, num_epochs, device,
                label_smoothing=0.1, warmup_steps=200,
                is_transformer=True, verbose=True):
    """
    训练指定模型，返回每轮的 (train_loss, val_loss, train_acc, val_acc) 历史记录
    """
    from data_utils import VOCAB_SIZE
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = WarmupScheduler(optimizer, d_model, warmup_steps)
    criterion = LabelSmoothingLoss(VOCAB_SIZE, PAD_IDX, label_smoothing)

    history = {'train_loss': [], 'val_loss': [],
               'train_acc':  [], 'val_acc':  []}

    best_val_loss = float('inf')
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            scheduler, device, is_transformer
        )
        va_loss, va_acc = evaluate(
            model, val_loader, criterion, device, is_transformer
        )

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"TrainLoss={tr_loss:.4f}  ValLoss={va_loss:.4f} | "
                  f"TrainAcc={tr_acc:.4f}  ValAcc={va_acc:.4f} | "
                  f"LR={scheduler.get_lr():.6f} | {elapsed:.1f}s")

    return history