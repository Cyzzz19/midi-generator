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
ROOT_DIR = r"D:\maestro-v3.0.0"  # 请替换为你的 MIDI 文件夹路径
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

# --- 偏移与词表大小 ---
PITCH_OFFSET = 4
TIME_SHIFT_OFFSET = PITCH_OFFSET + 88
DURATION_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_SHIFT_BUCKET + 1
VOCAB_SIZE = 4 + 88 * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)

# --- MIDI 解析 ---
def midi_to_events(midi_path, ticks_per_beat_target=480):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        return None
    if len(midi.instruments) == 0:
        return None

    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    if not notes:
        return None

    notes.sort(key=lambda x: x.start)
    events = []
    last_tick = 0
    for note in notes:
        if note.pitch < 21 or note.pitch > 108:
            continue
        start_120 = int(note.start * 2 * 480)
        end_120 = int(note.end * 2 * 480)
        duration = max(1, end_120 - start_120)
        time_shift = max(0, start_120 - last_tick)
        pitch_idx = note.pitch - 21
        events.append((pitch_idx, time_shift, duration))
        last_tick = start_120
    return events

# --- 时间量化 ---
def quantize_time_to_bucket(time_value, max_bucket):
    if time_value == 0:
        return 0
    return min(max_bucket, int(np.log2(time_value)))

def dequantize_bucket_to_time(bucket_idx):
    return 2 ** bucket_idx

# --- 三元组 ↔ 单 token ---
def event_to_single_token(pitch, time_shift, duration):
    ts_bucket = quantize_time_to_bucket(time_shift, MAX_TIME_SHIFT_BUCKET)
    dur_bucket = quantize_time_to_bucket(duration, MAX_DURATION_BUCKET)
    token_id = 4 + pitch * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1) \
               + ts_bucket * (MAX_DURATION_BUCKET + 1) + dur_bucket
    return token_id

def single_token_to_event(token_id):
    if token_id < 4:
        return None
    token_id -= 4
    pitch = token_id // ((MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1))
    token_id %= (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)
    ts_bucket = token_id // (MAX_DURATION_BUCKET + 1)
    dur_bucket = token_id % (MAX_DURATION_BUCKET + 1)
    time_shift = dequantize_bucket_to_time(ts_bucket)
    duration = dequantize_bucket_to_time(dur_bucket)
    return pitch, time_shift, duration

# --- 修复后的 MIDI 处理（禁止跨文件切分）---
def process_midi_file(args):
    path, max_seq_len, slide_step = args
    events = midi_to_events(path)
    if events is None or len(events) == 0:
        return []

    tokens = [BOS]
    for pitch, ts, dur in events:
        tokens.append(event_to_single_token(pitch, ts, dur))
    tokens.append(EOS)

    all_sequences = []
    start = 0
    while start < len(tokens):
        end = start + max_seq_len
        chunk = tokens[start:end]
        if len(chunk) < max_seq_len:
            chunk += [PAD] * (max_seq_len - len(chunk))
        all_sequences.append(chunk)
        start += slide_step
    return all_sequences

# --- 数据集 ---
class MIDIDataset(Dataset):
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
        print(f"Preprocessing MIDI files using {num_workers} processes...")
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_midi_file, args_list),
                               total=len(args_list), desc=f"Processing {split} MIDI files"))

        self.all_tokenized_sequences = []
        for result in results:
            if result:
                self.all_tokenized_sequences.extend(result)

        print(f"Generated {len(self.all_tokenized_sequences)} {split} sequences.")
        print(f"Vocabulary size: {VOCAB_SIZE}")

    def __len__(self):
        return len(self.all_tokenized_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.all_tokenized_sequences[idx], dtype=torch.long)

# --- 模型（Decoder-only）---
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

def create_padding_mask(seq, pad_token=PAD):
    return (seq == pad_token)

# --- 训练与验证 ---
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        x, y = batch[:, :-1], batch[:, 1:]
        tgt_mask = generate_square_subsequent_mask(x.size(1)).to(device)
        tgt_padding_mask = create_padding_mask(x, PAD).to(device)

        optimizer.zero_grad()
        output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            tgt_mask = generate_square_subsequent_mask(x.size(1)).to(device)
            tgt_padding_mask = create_padding_mask(x, PAD).to(device)
            output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
            loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

# --- 主训练流程 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset = MIDIDataset(ROOT_DIR, MAX_SEQ_LEN, SLIDE_STEP, NUM_WORKERS, split='train')
    val_dataset = MIDIDataset(ROOT_DIR, MAX_SEQ_LEN, SLIDE_STEP, NUM_WORKERS, split='val')
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # === 关键：更小模型 + 正则 ===
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

    # === 训练循环 + 早停 ===
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print("  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print("\nTraining finished. Best model saved as 'best_model.pt'.")