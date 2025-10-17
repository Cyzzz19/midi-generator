import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

# --- 配置参数 ---
ROOT_DIR = "D:\maestro-v3.0.0"  # 请替换为你的 MIDI 文件夹路径
MAX_SEQ_LEN = 512  # 模型一次处理的最大 token 长度
SLIDE_STEP = 256   # 滑动窗口步长（非重叠为 MAX_SEQ_LEN）
NUM_WORKERS = 16    # 用于预处理的进程数

# --- 词表和特殊 token ---
PAD = 0
BOS = 1
EOS = 2
SEP = 3

# --- 字段范围 ---
PITCH_RANGE = (0, 87)  # 0-87 映射 MIDI 21-108
MAX_TIME_SHIFT = 1000
MAX_DURATION = 1000

# --- 字段偏移 ---
PITCH_OFFSET = 4
TIME_SHIFT_OFFSET = PITCH_OFFSET + 88
DURATION_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_SHIFT + 1
VOCAB_SIZE = DURATION_OFFSET + MAX_DURATION + 1

# --- MIDI 解析与事件序列化 ---
def midi_to_events(midi_path, ticks_per_beat_target=480):
    """
    将 MIDI 转为事件序列：每个事件为 (pitch, time_shift, duration)
    - pitch: 0–87 (对应 MIDI 21–108)
    - time_shift: 当前事件 start 与上一事件 start 的 tick 差（>=0）
    - duration: 音符持续 tick 数（>=1）
    所有时间已归一化到 120 BPM, ticks_per_beat=480
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None

    if len(midi.instruments) == 0:
        return None

    # 合并所有轨道（假设单旋律或多轨合并）
    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    if not notes:
        return None

    # 按 start 排序
    notes.sort(key=lambda x: x.start)

    # 计算时间缩放因子：将实际时间映射到 120 BPM, tpb=480 下的 tick
    events = []
    last_tick = 0
    for note in notes:
        if note.pitch < 21 or note.pitch > 108:
            continue  # 超出钢琴范围
        # 转换为 120 BPM 下的 tick
        start_120 = int(note.start * 2 * 480)
        end_120 = int(note.end * 2 * 480)
        duration = max(1, end_120 - start_120)
        time_shift = start_120 - last_tick
        if time_shift < 0:
            time_shift = 0  # 防止负值（理论上不会）
        pitch_idx = note.pitch - 21  # 0–87
        events.append((pitch_idx, time_shift, duration))
        last_tick = start_120

    return events

# --- 事件 → token ID ---
def event_to_token(pitch, time_shift, duration):
    ts = min(time_shift, MAX_TIME_SHIFT)
    dur = min(duration, MAX_DURATION)
    return [
        PITCH_OFFSET + pitch,
        TIME_SHIFT_OFFSET + ts,
        DURATION_OFFSET + dur
    ]

# --- 用于多进程的辅助函数 ---
def process_midi_file(args):
    path, max_seq_len, slide_step = args
    events = midi_to_events(path)
    if events is not None and len(events) > 0:
        # 将事件序列转换为 token 序列
        tokens = [BOS]
        for pitch, ts, dur in events:
            tokens.extend(event_to_token(pitch, ts, dur))
        tokens.append(EOS)
        
        # 滑动窗口切分
        all_sequences = []
        start = 0
        while start < len(tokens):
            end = start + max_seq_len
            chunk = tokens[start:end]
            if len(chunk) < max_seq_len:
                chunk = chunk + [PAD] * (max_seq_len - len(chunk))
            all_sequences.append(chunk)
            start += slide_step
            if start + max_seq_len > len(tokens):
                if start < len(tokens):
                    last_chunk = tokens[start:]
                    last_chunk = last_chunk + [PAD] * (max_seq_len - len(last_chunk))
                    all_sequences.append(last_chunk)
                break
        return all_sequences
    return []

# --- 数据集类 ---
class MIDIDataset(Dataset):
    def __init__(self, root_dir, max_seq_len, slide_step, num_workers):
        self.max_seq_len = max_seq_len
        
        # 递归搜索所有 .mid 文件
        file_paths = []
        for ext in ['*.mid', '*.midi']:
            file_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"Found {len(file_paths)} MIDI files.")
        
        # 准备多进程参数
        args_list = [(path, max_seq_len, slide_step) for path in file_paths]
        
        # 使用多进程池处理 MIDI 文件
        print(f"Preprocessing MIDI files using {num_workers} processes...")
        with mp.Pool(processes=num_workers) as pool:
            # imap_unordered 可以在任务完成时立即获取结果，可能更快
            results = list(tqdm(pool.imap_unordered(process_midi_file, args_list), total=len(args_list), desc="Processing MIDI files"))
        
        # 合并所有结果
        self.all_tokenized_sequences = []
        for result in results:
            if result: # 检查是否有返回的序列
                self.all_tokenized_sequences.extend(result)
        
        print(f"Generated {len(self.all_tokenized_sequences)} training sequences.")

    def __len__(self):
        return len(self.all_tokenized_sequences)

    def __getitem__(self, idx):
        tokens = self.all_tokenized_sequences[idx]
        return torch.tensor(tokens, dtype=torch.long)

# --- 模型定义 ---
class TransformerMusic(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(d_model, max_len=2048)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1), :].to(src.device)
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :].to(tgt.device)
        output = self.transformer(
            src_emb, tgt_emb,
            src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

# --- 训练辅助函数 ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

def create_padding_mask(seq, pad_token=PAD):
    return (seq == pad_token)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        # 划分 src 和 tgt
        src = batch[:, :-1]  # 输入（去掉最后一个 token）
        tgt = batch[:, 1:]   # 目标（去掉第一个 token）

        # 生成 masks
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(device)
        src_padding_mask = create_padding_mask(src, PAD).to(device)
        tgt_padding_mask = create_padding_mask(tgt, PAD).to(device)

        optimizer.zero_grad()
        output = model(src, tgt, tgt_mask=tgt_mask,
                       src_key_padding_mask=src_padding_mask,
                       tgt_key_padding_mask=tgt_padding_mask)
        # 修复：使用 .reshape(...) 替代 .view(...)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 主训练脚本 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据集和数据加载器
    # 注意：num_workers 指的是预处理的进程数，不是 DataLoader 的 num_workers
    dataset = MIDIDataset(ROOT_DIR, MAX_SEQ_LEN, SLIDE_STEP, NUM_WORKERS)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0) # DataLoader 本身不使用多进程，避免冲突

    # 初始化模型
    model = TransformerMusic(vocab_size=VOCAB_SIZE, d_model=512).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Average Loss: {loss:.4f}")
        # 保存模型检查点
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")

    print("\nTraining finished.")