import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# --- 配置参数 ---
ROOT_DIR = r"D:\maestro-v3.0.0"
MAX_SEQ_LEN = 2048
SLIDE_STEP = 1024
NUM_WORKERS = 16
VALIDATION_SPLIT = 0.1

# --- 特殊 token ---
PAD = 0
BOS = 1
EOS = 2
SEP = 3

# --- 字段范围 ---
PITCH_RANGE = (0, 87)  # MIDI 21–108 → 0–87
MAX_TIME_SHIFT_BUCKET = 10
MAX_DURATION_BUCKET = 10

# --- 词表大小 ---
VOCAB_SIZE = 4 + 88 * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)

# --- 时间量化 ---
def quantize_time_to_bucket(time_value, max_bucket):
    if time_value == 0:
        return 0
    return min(max_bucket, int(np.log2(time_value)))

def dequantize_bucket_to_time(bucket_idx):
    return 2 ** bucket_idx

# --- Token 映射 ---
def event_to_single_token(pitch, time_shift, duration):
    ts_bucket = quantize_time_to_bucket(time_shift, MAX_TIME_SHIFT_BUCKET)
    dur_bucket = quantize_time_to_bucket(duration, MAX_DURATION_BUCKET)
    return 4 + pitch * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1) \
           + ts_bucket * (MAX_DURATION_BUCKET + 1) + dur_bucket

def single_token_to_event(token_id):
    if token_id < 4:
        return None
    token_id -= 4
    pitch = token_id // ((MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1))
    token_id %= (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)
    ts_bucket = token_id // (MAX_DURATION_BUCKET + 1)
    dur_bucket = token_id % (MAX_DURATION_BUCKET + 1)
    return pitch, dequantize_bucket_to_time(ts_bucket), dequantize_bucket_to_time(dur_bucket)

# --- 声部分离 ---
def separate_voices(notes):
    if not notes:
        return []
    # 按 start 排序，同 start 时高音优先
    notes = sorted(notes, key=lambda n: (n.start, -n.pitch))
    voices = []

    for note in notes:
        if note.pitch < 21 or note.pitch > 108:
            continue
        assigned = False
        for voice in voices:
            conflict = False
            for v_note in voice:
                # 检查时间重叠且 v_note 更高
                if v_note.start < note.end and note.start < v_note.end:
                    if v_note.pitch > note.pitch:
                        conflict = True
                        break
            if not conflict:
                voice.append(note)
                assigned = True
                break
        if not assigned:
            voices.append([note])
    
    if not voices:
        return []
    # 按最高音降序：voices[0] = 最高声部
    voices.sort(key=lambda v: -max(n.pitch for n in v))
    return voices

# --- MIDI 转多声部事件 ---
def midi_to_voices_events(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return None

    all_notes = []
    for inst in midi.instruments:
        all_notes.extend(inst.notes)
    if not all_notes:
        return None

    voices = separate_voices(all_notes)
    if not voices or len(voices) < 2:
        return None  # 至少两个声部才有效

    voices_events = []
    for voice in voices:
        voice = [n for n in voice if 21 <= n.pitch <= 108]
        if not voice:
            continue
        voice.sort(key=lambda n: n.start)
        events = []
        last_tick = 0
        for note in voice:
            start_120 = int(note.start * 2 * 480)
            end_120 = int(note.end * 2 * 480)
            duration = max(1, end_120 - start_120)
            time_shift = max(0, start_120 - last_tick)
            pitch_idx = note.pitch - 21
            events.append((pitch_idx, time_shift, duration))
            last_tick = start_120
        if events:
            voices_events.append(events)
    return voices_events if len(voices_events) >= 2 else None

# --- 多进程处理：生成层级样本 ---
def process_midi_file_for_hierarchy(args):
    path, max_seq_len, slide_step = args
    voices_events = midi_to_voices_events(path)
    if voices_events is None:
        return []

    all_samples = []
    num_voices = len(voices_events)

    # 声部 0（最高）：自回归
    melody_tokens = [BOS]
    for pitch, ts, dur in voices_events[0]:
        melody_tokens.append(event_to_single_token(pitch, ts, dur))
    melody_tokens.append(EOS)

    # 切分旋律（自回归样本）
    start = 0
    while start < len(melody_tokens):
        in_chunk = melody_tokens[start:start + max_seq_len]
        tgt_chunk = melody_tokens[start + 1:start + max_seq_len + 1]  # 目标是下一个token
        if len(in_chunk) < max_seq_len:
            in_chunk += [PAD] * (max_seq_len - len(in_chunk))
        if len(tgt_chunk) < max_seq_len:
            tgt_chunk += [PAD] * (max_seq_len - len(tgt_chunk))
        all_samples.append((in_chunk, tgt_chunk))
        start += slide_step
        if start >= len(melody_tokens):
            break

    # 声部 1 到 N-1（条件生成）
    for i in range(1, num_voices):
        # 输入：声部 0（旋律）
        input_tokens = [BOS]
        for pitch, ts, dur in voices_events[0]:
            input_tokens.append(event_to_single_token(pitch, ts, dur))
        input_tokens.append(EOS)

        # 目标：声部 i
        target_tokens = [BOS]
        for pitch, ts, dur in voices_events[i]:
            target_tokens.append(event_to_single_token(pitch, ts, dur))
        target_tokens.append(EOS)

        # 滑动窗口对齐
        min_len = min(len(input_tokens), len(target_tokens))
        if min_len <= 2:
            continue
        start = 0
        while start < min_len:
            in_chunk = input_tokens[start:start + max_seq_len]
            tgt_chunk = target_tokens[start + 1:start + max_seq_len + 1]  # 目标是下一个token
            if len(in_chunk) < max_seq_len:
                in_chunk += [PAD] * (max_seq_len - len(in_chunk))
            if len(tgt_chunk) < max_seq_len:
                tgt_chunk += [PAD] * (max_seq_len - len(tgt_chunk))
            all_samples.append((in_chunk, tgt_chunk))
            start += slide_step
            if start >= min_len:
                break

    return all_samples

# --- 数据集 ---
class HierarchyMIDIDataset(Dataset):
    def __init__(self, root_dir, max_seq_len, slide_step, num_workers, split='train'):
        file_paths = []
        for ext in ['*.mid', '*.midi']:
            file_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        print(f"Found {len(file_paths)} MIDI files.")

        np.random.seed(42)
        np.random.shuffle(file_paths)
        split_idx = int((1 - VALIDATION_SPLIT) * len(file_paths))
        file_paths = file_paths[:split_idx] if split == 'train' else file_paths[split_idx:]
        print(f"Using {len(file_paths)} files for {split} set.")

        args_list = [(path, max_seq_len, slide_step) for path in file_paths]
        print(f"Preprocessing with {num_workers} workers...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_midi_file_for_hierarchy, args_list),
                               total=len(args_list), desc=f"Processing {split}"))

        self.samples = []
        for res in results:
            if res:
                self.samples.extend(res)
        print(f"Generated {len(self.samples)} {split} samples.")
        print(f"Vocabulary size: {VOCAB_SIZE}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# --- 模型（与训练一致）---
class MusicTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(d_model, max_len=2048)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_positional_encoding(self, d_model, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, tgt_mask=None, tgt_key_padding_mask=None):
        x_emb = self.embedding(x) + self.pos_encoding[:, :x.size(1), :].to(x.device)
        output = self.transformer_decoder(x_emb, x_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(output)

# --- 辅助函数 ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# --- 训练与验证 ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_inp, batch_tgt in tqdm(dataloader, desc="Training"):
        batch_inp, batch_tgt = batch_inp.to(device), batch_tgt.to(device)
        # 输入: [BOS, t1, t2, ...]
        # 目标: [t1, t2, ..., EOS]
        x = batch_inp[:, :-1]
        y = batch_tgt[:, 1:]

        tgt_mask = generate_square_subsequent_mask(x.size(1)).to(device)
        padding_mask = (x == PAD).to(device)

        optimizer.zero_grad()
        output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_inp, batch_tgt in tqdm(dataloader, desc="Evaluating"):
            batch_inp, batch_tgt = batch_inp.to(device), batch_tgt.to(device)
            x = batch_inp[:, :-1]
            y = batch_tgt[:, 1:]

            tgt_mask = generate_square_subsequent_mask(x.size(1)).to(device)
            padding_mask = (x == PAD).to(device)

            output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 主程序 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # === 数据集 ===
    train_dataset = HierarchyMIDIDataset(ROOT_DIR, MAX_SEQ_LEN, SLIDE_STEP, NUM_WORKERS, 'train')
    val_dataset = HierarchyMIDIDataset(ROOT_DIR, MAX_SEQ_LEN, SLIDE_STEP, NUM_WORKERS, 'val')

    if len(train_dataset) == 0:
        raise ValueError("No valid training samples generated.")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # === 模型 ===
    model = MusicTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # === 训练 ===
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_hierarchy_model.pt")
            print("  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    print("\nTraining finished. Best model: 'best_hierarchy_model.pt'")