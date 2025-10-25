import os
import torch
import torch.nn as nn
import pretty_midi
import numpy as np
from tqdm import tqdm

# --- 配置参数 ---
MODEL_PATH = "best_harmony_model_v2.pt"
INPUT_MELODY_PATH = "3.mid"
OUTPUT_FULL_PATH = "harmonized.mid"

MAX_SEQ_LEN = 2048
MAX_GENERATE_TOKENS = 2048

# --- 采样参数 ---
TEMPERATURE = 1.2
TOP_K = 20
TOP_P = 0.90
REPETITION_PENALTY = 1.8
REPETITION_PENALTY_WINDOW = 30
MAX_VOICES = 8

# --- 特殊 token ---
PAD = 0
BOS = 1
EOS = 2

# --- 字段范围 ---
MAX_DURATION_BUCKET = 10
VOCAB_SIZE = 89 * (MAX_DURATION_BUCKET + 1)

# --- 与训练一致的量化函数 ---
def quantize_time_to_bucket(time_value, max_bucket):
    if time_value == 0:
        return 0
    return min(max_bucket, int(np.log2(time_value)))

def dequantize_bucket_to_time(bucket_idx):
    return 2 ** bucket_idx

def event_to_token(pitch, duration):
    dur_bucket = quantize_time_to_bucket(duration, MAX_DURATION_BUCKET)
    return pitch * (MAX_DURATION_BUCKET + 1) + dur_bucket

def token_to_event(token_id):
    dur_bucket = token_id % (MAX_DURATION_BUCKET + 1)
    pitch = token_id // (MAX_DURATION_BUCKET + 1)
    duration = dequantize_bucket_to_time(dur_bucket)
    return pitch, duration

# --- MIDI 解析（旋律）---
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
    for note in notes:
        pitch_idx = note.pitch - 21 + 1  # 1-88
        duration = max(1, int((note.end - note.start) * 2 * 480))
        events.append((pitch_idx, duration))
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

# --- 模型定义 ---
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

# --- 生成函数（旋律 + 和声）---
def generate_harmony(model, melody_tokens, max_voices, max_generate_tokens, temperature=1.0,
                     top_k=0, top_p=0.0, repetition_penalty=1.0, repetition_penalty_window=30):
    model.eval()
    device = next(model.parameters()).device

    # 1. 生成旋律（自回归）
    melody_seq = torch.tensor(melody_tokens, dtype=torch.long, device=device).unsqueeze(0)
    print("Step 1: Generating melody...")
    generated_melody = [BOS]
    x = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)
    for _ in tqdm(range(max_generate_tokens), desc="Generating Melody"):
        current_input = x[:, -MAX_SEQ_LEN:]
        seq_len = current_input.size(1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
        padding_mask = (current_input == PAD).to(device)

        with torch.no_grad():
            output = model(current_input, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
            next_token_logits = output[0, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        if next_token_id == EOS:
            break
        generated_melody.append(next_token_id)
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        x = torch.cat([x, next_token_tensor], dim=1)
        if len(generated_melody) >= max_generate_tokens:
            break

    print(f"Generated melody length: {len(generated_melody)} tokens")
    all_voices = [generated_melody]  # 0 = melody

    # 2. 逐层生成和声（条件于旋律）
    for voice_idx in range(1, max_voices):
        print(f"Step {voice_idx+1}: Generating harmony voice {voice_idx}...")
        generated_voice = [BOS]
        x = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)
        melody_ctx = torch.tensor(generated_melody, dtype=torch.long, device=device).unsqueeze(0)

        for _ in tqdm(range(max_generate_tokens), desc=f"Generating Voice {voice_idx}"):
            # 模型输入：当前已生成的和声
            current_input = x[:, -MAX_SEQ_LEN:]
            seq_len = current_input.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len).to(device)
            padding_mask = (current_input == PAD).to(device)

            with torch.no_grad():
                output = model(current_input, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
                next_token_logits = output[0, -1, :] / temperature

            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits,
                    x[0].tolist(),
                    repetition_penalty,
                    repetition_penalty_window
                )

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            if next_token_id == EOS:
                break
            generated_voice.append(next_token_id)
            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            x = torch.cat([x, next_token_tensor], dim=1)
            if len(generated_voice) >= max_generate_tokens:
                break

        if len(generated_voice) > 1:  # 至少有 BOS 和一个 token
            all_voices.append(generated_voice)
        else:
            print(f"Voice {voice_idx} generation failed, stopping.")
            break

    return all_voices

# --- Token 转 MIDI ---
def tokens_to_midi(voices_tokens, output_path, ticks_per_beat=480):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=ticks_per_beat)
    for i, voice_tokens in enumerate(voices_tokens):
        inst = pretty_midi.Instrument(program=0 + i * 8)  # 不同声部用不同音色
        current_time = 0.0
        for token_id in voice_tokens:
            if token_id in (BOS, EOS, PAD):
                continue
            pitch, duration = token_to_event(token_id)
            if pitch == 0:  # note_off
                continue
            duration_sec = duration / (2 * ticks_per_beat)
            inst.notes.append(pretty_midi.Note(velocity=70, pitch=pitch + 20, start=current_time, end=current_time + duration_sec))
            current_time += duration_sec  # 简化处理：假设音符是连续的
        if inst.notes:
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

    melody_tokens = [BOS]
    for pitch, dur in melody_events:
        melody_tokens.append(event_to_token(pitch, dur))
    # melody_tokens.append(EOS) # 不加 EOS，让模型自己生成

    # 生成
    all_voices = generate_harmony(
        model=model,
        melody_tokens=melody_tokens,
        max_voices=MAX_VOICES,
        max_generate_tokens=MAX_GENERATE_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        repetition_penalty_window=REPETITION_PENALTY_WINDOW
    )

    # 保存
    tokens_to_midi(all_voices, OUTPUT_FULL_PATH)
    print(f"Harmonized MIDI saved to {OUTPUT_FULL_PATH}")
    print(f"Generated {len(all_voices)} voices.")