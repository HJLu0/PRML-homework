# =============================================================================
# experiments.py —— 实验主入口
#
# 实验一：位置编码对比
#   - 无位置编码 (none)
#   - 正弦位置编码 (sinusoidal) ← 原论文
#   - 可学习位置编码 (learnable)
#
# 实验二：CNN 重构网络 vs Transformer（自注意力）
#   均使用正弦位置编码，对比序列建模能力
#
# 运行方式（Spyder 中直接 F5，或终端）：
#   python experiments.py
# =============================================================================

import os
import torch
import matplotlib
matplotlib.use('Agg')          # 非交互后端，保证 Spyder 里不阻塞
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from data_utils  import get_dataloaders, VOCAB_SIZE
from model       import Transformer
from cnn_model   import CNNSeq2Seq
from trainer     import train_model

# ─────────────────────────────────────────────
# 全局配置
# ─────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 40          # 可根据机器性能调大（推荐 60）
WARMUP     = 200

# Transformer 超参（轻量版，适合 CPU）
TF_D_MODEL    = 128
TF_HEADS      = 4
TF_D_FF       = 256
TF_LAYERS     = 3
TF_DROPOUT    = 0.1

# CNN 超参（参数量尽量与 Transformer 相当）
CNN_D_MODEL   = 128
CNN_LAYERS    = 4
CNN_KERNEL    = 3
CNN_DROPOUT   = 0.1

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 中文字体（Windows 环境）
# ─────────────────────────────────────────────
def setup_chinese_font():
    """尝试设置中文字体，失败则使用英文标签"""
    candidates = [
        'SimHei', 'Microsoft YaHei', 'FangSong', 'KaiTi', 'SimSun',
        'STSong', 'STHeiti', 'Arial Unicode MS'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"  [字体] 使用: {font}")
            return True
    print("  [字体] 未找到中文字体，图表标签将使用英文")
    return False

USE_CN = setup_chinese_font()

def label(cn, en):
    return cn if USE_CN else en


# ─────────────────────────────────────────────
# 绘图工具
# ─────────────────────────────────────────────

def plot_results(histories: dict, title_cn: str, title_en: str,
                 filename: str, colors=None):
    """
    histories: {'模型名': {'train_loss':..., 'val_loss':...,
                           'train_acc': ..., 'val_acc': ...}}
    """
    if colors is None:
        colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = title_cn if USE_CN else title_en
    fig.suptitle(title, fontsize=14, fontweight='bold')

    for idx, (name, hist) in enumerate(histories.items()):
        c = colors[idx % len(colors)]
        epochs = range(1, len(hist['train_loss']) + 1)

        # 损失
        axes[0].plot(epochs, hist['train_loss'], '--', color=c, alpha=0.6,
                     label=f"{name} ({label('训练','Train')})")
        axes[0].plot(epochs, hist['val_loss'],   '-',  color=c,
                     label=f"{name} ({label('验证','Val')})")
        # 准确率
        axes[1].plot(epochs, hist['train_acc'],  '--', color=c, alpha=0.6,
                     label=f"{name} ({label('训练','Train')})")
        axes[1].plot(epochs, hist['val_acc'],    '-',  color=c,
                     label=f"{name} ({label('验证','Val')})")

    axes[0].set_xlabel(label('轮次', 'Epoch'))
    axes[0].set_ylabel(label('损失', 'Loss'))
    axes[0].set_title(label('训练 / 验证损失', 'Train / Val Loss'))
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(label('轮次', 'Epoch'))
    axes[1].set_ylabel(label('准确率', 'Accuracy'))
    axes[1].set_title(label('训练 / 验证 Token 准确率', 'Train / Val Token Accuracy'))
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {path}")


def plot_summary_bar(results: dict, metric: str,
                     title_cn: str, title_en: str, filename: str):
    """最终测试指标柱状图"""
    names  = list(results.keys())
    values = [results[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    title = title_cn if USE_CN else title_en
    bars = ax.bar(names, values,
                  color=plt.cm.tab10.colors[:len(names)],
                  edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(label('验证集准确率', 'Val Accuracy') if 'acc' in metric
                  else label('验证集损失', 'Val Loss'))
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  柱状图已保存: {path}")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────
# ===== 实验一：位置编码对比 =====
# ─────────────────────────────────────────────

def experiment1():
    print("\n" + "="*60)
    print("实验一：位置编码对比")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE)

    pos_configs = {
        label('无位置编码','No PE'):          'none',
        label('可学习位置编码','Learnable PE'): 'learnable',
        label('正弦位置编码（原论文）','Sinusoidal PE'): 'sinusoidal',
    }

    histories  = {}
    final_accs = {}

    for name, pe_type in pos_configs.items():
        print(f"\n▶ 训练: {name}  (pe_type={pe_type})")
        model = Transformer(
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            d_model=TF_D_MODEL,
            num_heads=TF_HEADS,
            d_ff=TF_D_FF,
            num_layers=TF_LAYERS,
            dropout=TF_DROPOUT,
            pos_encoding=pe_type
        )
        print(f"  参数量: {count_params(model):,}")
        hist = train_model(
            model, train_loader, val_loader,
            d_model=TF_D_MODEL,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            warmup_steps=WARMUP,
            verbose=True
        )
        histories[name]  = hist
        final_accs[name] = {'val_acc': max(hist['val_acc']),
                             'val_loss': min(hist['val_loss'])}

    # ── 绘图 ──
    plot_results(
        histories,
        title_cn='实验一：不同位置编码方式对比',
        title_en='Exp1: Comparison of Positional Encodings',
        filename='exp1_pos_encoding_curves.png'
    )
    plot_summary_bar(
        final_accs, metric='val_acc',
        title_cn='实验一：各位置编码方式最佳验证准确率',
        title_en='Exp1: Best Val Accuracy by PE Type',
        filename='exp1_pos_encoding_bar.png'
    )

    print("\n[实验一总结]")
    for name, res in final_accs.items():
        print(f"  {name:30s} | 最佳验证准确率={res['val_acc']:.4f}  "
              f"最低验证损失={res['val_loss']:.4f}")

    return histories, final_accs


# ─────────────────────────────────────────────
# ===== 实验二：CNN 重构网络 vs Transformer =====
# ─────────────────────────────────────────────

def experiment2():
    print("\n" + "="*60)
    print("实验二：CNN 重构网络 vs Transformer（均使用正弦位置编码）")
    print("="*60)

    train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE)

    models = {
        label('Transformer（自注意力）','Transformer (Self-Attn)'): {
            'model': Transformer(
                src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
                d_model=TF_D_MODEL, num_heads=TF_HEADS,
                d_ff=TF_D_FF, num_layers=TF_LAYERS,
                dropout=TF_DROPOUT, pos_encoding='sinusoidal'
            ),
            'd_model': TF_D_MODEL,
            'is_tf':   True,
        },
        label('CNN Seq2Seq（卷积+位置编码）','CNN Seq2Seq (Conv+PE)'): {
            'model': CNNSeq2Seq(
                src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
                d_model=CNN_D_MODEL, num_layers=CNN_LAYERS,
                kernel_size=CNN_KERNEL, dropout=CNN_DROPOUT,
                pos_encoding='sinusoidal'
            ),
            'd_model': CNN_D_MODEL,
            'is_tf':   False,
        },
    }

    histories  = {}
    final_accs = {}

    for name, cfg in models.items():
        print(f"\n▶ 训练: {name}")
        print(f"  参数量: {count_params(cfg['model']):,}")
        hist = train_model(
            cfg['model'], train_loader, val_loader,
            d_model=cfg['d_model'],
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            warmup_steps=WARMUP,
            is_transformer=cfg['is_tf'],
            verbose=True
        )
        histories[name]  = hist
        final_accs[name] = {'val_acc': max(hist['val_acc']),
                             'val_loss': min(hist['val_loss'])}

    # ── 绘图 ──
    plot_results(
        histories,
        title_cn='实验二：Transformer vs CNN Seq2Seq（含位置编码）',
        title_en='Exp2: Transformer vs CNN Seq2Seq (with PE)',
        filename='exp2_cnn_vs_transformer_curves.png'
    )
    plot_summary_bar(
        final_accs, metric='val_acc',
        title_cn='实验二：最佳验证准确率对比',
        title_en='Exp2: Best Val Accuracy Comparison',
        filename='exp2_cnn_vs_transformer_bar.png'
    )

    print("\n[实验二总结]")
    for name, res in final_accs.items():
        print(f"  {name:40s} | 最佳验证准确率={res['val_acc']:.4f}  "
              f"最低验证损失={res['val_loss']:.4f}")

    return histories, final_accs


# ─────────────────────────────────────────────
# ===== 综合对比图 =====
# ─────────────────────────────────────────────

def plot_combined_summary(exp1_accs, exp2_accs):
    """汇总两个实验的最终验证准确率，生成一张总览图"""
    all_results = {}
    for k, v in exp1_accs.items():
        all_results[f"Exp1: {k}"] = v['val_acc']
    for k, v in exp2_accs.items():
        all_results[f"Exp2: {k}"] = v['val_acc']

    names  = list(all_results.keys())
    values = list(all_results.values())
    colors = plt.cm.Set2.colors

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, values,
                   color=[colors[i % len(colors)] for i in range(len(names))],
                   edgecolor='black', alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    title = '两项实验综合验证准确率汇总' if USE_CN else 'Combined Val Accuracy Summary'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel(label('验证集最佳 Token 准确率', 'Best Val Token Accuracy'))
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'summary_combined.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  综合汇总图已保存: {path}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print(f"运行设备: {DEVICE}")
    print(f"PyTorch 版本: {torch.__version__}")

    # ── 实验一 ──
    hist1, accs1 = experiment1()

    # ── 实验二 ──
    hist2, accs2 = experiment2()

    # ── 综合图 ──
    print("\n生成综合汇总图...")
    plot_combined_summary(accs1, accs2)

    print("\n" + "="*60)
    print("全部实验完成！")
    print(f"输出文件保存在: {SAVE_DIR}")
    print("="*60)