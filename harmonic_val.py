import os
import torch
import torch.nn as nn
import pretty_midi
import numpy as np
from tqdm import tqdm

# --- 配置参数 ---
MODEL_PATH = "best_hierarchy_model.pt"
INPUT_MELODY_PATH = "3.mid"          # 输入旋律 MIDI
OUTPUT_FULL_PATH = "hierarchical_harmony.mid"     # 完整和声

MAX_SEQ_LEN = 2048
MAX_GENERATE_TOKENS = 2048

# --- 采样参数 ---
TEMPERATURE = 1.2
TOP_K = 25
TOP_P = 0.92
REPETITION_PENALTY = 1.8
REPETITION_PENALTY_WINDOW = 40

# --- 词表和特殊 token ---
PAD = 0
BOS = 1
EOS = 2
SEP = 3

# --- 字段范围 ---
MAX_TIME_SHIFT_BUCKET = 10
MAX_DURATION_BUCKET = 10
VOCAB_SIZE = 4 + 88 * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)

# --- 与训练完全一致的量化函数 ---
def quantize_time_to_bucket(time_value, max_bucket):
    if time_value == 0:
        return 0
    return min(max_bucket, int(np.log2(time_value)))

def dequantize_bucket_to_time(bucket_idx):
    return 2 ** bucket_idx

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

# --- MIDI 解析（仅旋律）---
def midi_to_melody_events(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error loading MIDI: {e}")
        return None

    notes = []
    for inst in midi.instruments:
        notes.extend(inst.notes)
    if not notes:
        return None

    notes = [n for n in notes if 21 <= n.pitch <= 108]
    if not notes:
        return None

    notes.sort(key=lambda x: x.start)
    events = []
    last_tick = 0
    for note in notes:
        start_120 = int(note.start * 2 * 480)
        end_120 = int(note.end * 2 * 480)
        duration = max(1, end_120 - start_120)
        time_shift = max(0, start_120 - last_tick)
        pitch_idx = note.pitch - 21
        events.append((pitch_idx, time_shift, duration))
        last_tick = start_120
    return events

# --- 采样函数 ---
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def apply_repetition_penalty(logits, generated_tokens, penalty, window_size):
    if len(generated_tokens) == 0:
        return logits
    recent_tokens = list(set(generated_tokens[-window_size:]))
    for token_id in recent_tokens:
        if 0 <= token_id < logits.size(0):
            logits[token_id] /= penalty
    return logits

# --- 模型定义（与训练完全一致）---
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

# --- 因果掩码 ---
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

# --- 生成函数（层级）---
def generate_hierarchical_harmony(model, melody_tokens, num_voices=3, max_generate_tokens=2048,
                                  temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0,
                                  repetition_penalty_window=30):
    """
    生成层级和声：先生成 melody，再逐层生成低音声部
    """
    device = next(model.parameters()).device
    full_harmony_notes = []

    # 1. 生成旋律（自回归）
    print("Generating melody...")
    melody_input = torch.tensor(melody_tokens, dtype=torch.long, device=device).unsqueeze(0)
    melody_generated = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)

    for _ in tqdm(range(max_generate_tokens), desc="Generating Melody"):
        x = melody_generated[:, -MAX_SEQ_LEN:]
        seq_len = x.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
        padding_mask = (x == PAD).to(device)

        with torch.no_grad():
            output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
            next_token_logits = output[0, -1, :] / temperature

        if repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                melody_generated[0].tolist(),
                repetition_penalty,
                repetition_penalty_window
            )

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == EOS:
            break

        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        melody_generated = torch.cat([melody_generated, next_token_tensor], dim=1)

    melody_tokens_gen = melody_generated[0].cpu().tolist()
    melody_notes = tokens_to_notes(melody_tokens_gen)
    full_harmony_notes.extend(melody_notes)
    print(f"Generated melody with {len(melody_notes)} notes.")

    # 2. 生成低音声部（条件于旋律）
    for i in range(1, num_voices):
        print(f"Generating voice {i} (conditioned on melody)...")
        # 输入：旋律
        cond_input = torch.tensor(melody_tokens, dtype=torch.long, device=device).unsqueeze(0)
        # 生成目标：声部 i
        voice_generated = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)

        for _ in tqdm(range(max_generate_tokens), desc=f"Generating Voice {i}"):
            x = voice_generated[:, -MAX_SEQ_LEN:]
            seq_len = x.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            padding_mask = (x == PAD).to(device)

            with torch.no_grad():
                output = model(x, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
                next_token_logits = output[0, -1, :] / temperature

            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits,
                    voice_generated[0].tolist(),
                    repetition_penalty,
                    repetition_penalty_window
                )

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == EOS:
                break

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            voice_generated = torch.cat([voice_generated, next_token_tensor], dim=1)

        voice_tokens_gen = voice_generated[0].cpu().tolist()
        voice_notes = tokens_to_notes(voice_tokens_gen)
        full_harmony_notes.extend(voice_notes)
        print(f"Generated voice {i} with {len(voice_notes)} notes.")

    return full_harmony_notes

# --- Token 转 MIDI ---
def tokens_to_notes(tokens):
    """将 token 序列转为 (pitch, start_tick, end_tick) 列表"""
    notes = []
    current_tick = 0
    for token_id in tokens:
        if token_id in (BOS, EOS, PAD, SEP):
            continue
        event = single_token_to_event(token_id)
        if event is None:
            continue
        pitch, time_shift, duration = event
        start_tick = current_tick + time_shift
        end_tick = start_tick + duration
        notes.append((pitch + 21, start_tick, end_tick))
        current_tick = start_tick
    return notes

def notes_to_midi(notes, output_path, ticks_per_beat=480):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=ticks_per_beat)
    inst = pretty_midi.Instrument(program=0)
    for pitch, start_tick, end_tick in notes:
        start_time = start_tick / (2 * ticks_per_beat)
        end_time = end_tick / (2 * ticks_per_beat)
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=pitch, start=start_time, end=end_time))
    pm.instruments.append(inst)
    pm.write(output_path)

# --- 主程序 ---
if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = MusicTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.3
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}")

    # 加载旋律
    melody_events = midi_to_melody_events(INPUT_MELODY_PATH)
    if melody_events is None:
        raise ValueError(f"Failed to parse melody MIDI: {INPUT_MELODY_PATH}")

    # 构建旋律 token
    melody_tokens = [BOS]
    for pitch, ts, dur in melody_events:
        melody_tokens.append(event_to_single_token(pitch, ts, dur))
    melody_tokens.append(EOS)
    print(f"Input melody length: {len(melody_tokens)} tokens")

    # 生成层级和声
    full_notes = generate_hierarchical_harmony(
        model=model,
        melody_tokens=melody_tokens,
        num_voices=3,  # 生成旋律 + 2 个伴奏声部
        max_generate_tokens=MAX_GENERATE_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        repetition_penalty_window=REPETITION_PENALTY_WINDOW
    )

    # 保存完整和声
    if full_notes:
        notes_to_midi(full_notes, OUTPUT_FULL_PATH)
        print(f"Full hierarchical harmony saved to {OUTPUT_FULL_PATH}")
    else:
        print("No notes generated.")