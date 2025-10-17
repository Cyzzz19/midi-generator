import os
import torch
import torch.nn as nn
import pretty_midi
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- 配置参数 ---
MODEL_PATH = "model_epoch_3.pt"  # 模型文件路径
INPUT_MIDI_PATH = "1.mid"  # 输入 MIDI 文件路径
OUTPUT_MIDI_PATH = "output.mid"  # 输出 MIDI 文件路径
MAX_SEQ_LEN = 512  # 模型一次处理的最大 token 长度
MAX_GENERATE_TOKENS = 512  # 生成多少个 token
TEMPERATURE = 1.0  # 采样温度，控制随机性
TOP_K = 0  # Top-k 采样，0 表示不使用
TOP_P = 0.9  # Nucleus sampling，0 表示不使用

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

# --- MIDI 解析与事件序列化 (与训练时相同) ---
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

# --- 事件 → token ID (与训练时相同) ---
def event_to_token(pitch, time_shift, duration):
    ts = min(time_shift, MAX_TIME_SHIFT)
    dur = min(duration, MAX_DURATION)
    return [
        PITCH_OFFSET + pitch,
        TIME_SHIFT_OFFSET + ts,
        DURATION_OFFSET + dur
    ]

# --- Token ID → 事件 (用于生成) ---
def token_to_event(token_id):
    if token_id >= DURATION_OFFSET and token_id < DURATION_OFFSET + MAX_DURATION + 1:
        dur = token_id - DURATION_OFFSET
        # 这是一个 duration token，需要结合上一个 pitch 和 time_shift
        # 在生成逻辑中处理
        return 'duration', dur
    elif token_id >= TIME_SHIFT_OFFSET and token_id < DURATION_OFFSET:
        ts = token_id - TIME_SHIFT_OFFSET
        return 'time_shift', ts
    elif token_id >= PITCH_OFFSET and token_id < TIME_SHIFT_OFFSET:
        pitch = token_id - PITCH_OFFSET
        return 'pitch', pitch
    else:
        return 'special', token_id
    return None

# --- 采样辅助函数 ---
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    根据 top_k 和 top_p 过滤 logits
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # logits shape: (vocab_size,)
        # We need to reshape for scatter operations if needed, but topk returns indices for the last dim
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1] # [0] are values, [1] are indices
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True) # Both shape (vocab_size,)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) # shape (vocab_size,)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p # shape (vocab_size,)

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        # Create a mask for original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# --- 模型定义 (与训练时相同) ---
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

# --- 生成函数 ---
def generate(model, context_tokens, max_generate_tokens, temperature=1.0, top_k=0, top_p=0.0):
    model.eval()
    device = next(model.parameters()).device
    
    # Encoder 输入：原始上下文（固定）
    src = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, src_len)
    
    # Decoder 输入：从 <BOS> 开始
    tgt = torch.tensor([BOS], dtype=torch.long, device=device).unsqueeze(0)  # (1, 1)

    for _ in tqdm(range(max_generate_tokens), desc="Generating"):
        # 创建 decoder mask
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        with torch.no_grad():
            # 正确调用：src 固定，tgt 是已生成序列
            output = model(src, tgt, tgt_mask=tgt_mask)
        
        # 预测下一个 token
        next_token_logits = output[0, -1, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        if next_token_id == EOS:
            print("EOS token generated, stopping.")
            break

        # 将新 token 添加到 tgt
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        tgt = torch.cat((tgt, next_token_tensor), dim=1)

    # 返回：原始上下文 + 生成部分（可选），或仅生成部分
    # 这里我们返回完整的序列用于保存
    full_sequence = torch.cat((src, tgt[:, 1:]), dim=1)  # 去掉 tgt 的 BOS 重复
    return full_sequence[0].cpu().tolist()

# --- 将生成的 tokens 转换为 MIDI 并保存 ---
def tokens_to_midi(tokens, output_path, ticks_per_beat=480):
    """
    将 token 序列转换为 MIDI 文件并保存
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=ticks_per_beat)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    current_time = 0.0
    i = 0
    # 存储临时的 pitch 和 time_shift，等待 duration
    temp_pitch = None
    temp_time_shift = 0

    while i < len(tokens):
        token_id = tokens[i]
        event_type, value = token_to_event(token_id)

        if event_type == 'pitch':
            if temp_pitch is not None and temp_time_shift is not None:
                # 上一个事件的 duration 尚未处理，将其写入
                if temp_pitch is not None:
                    start_time = current_time + temp_time_shift / (2 * ticks_per_beat)
                    end_time = start_time + temp_duration / (2 * ticks_per_beat)
                    note = pretty_midi.Note(velocity=64, pitch=temp_pitch + 21, start=start_time, end=end_time)
                    piano.notes.append(note)
                    current_time = start_time # 更新时间到音符开始
            # 设置当前事件的 pitch，等待 time_shift 和 duration
            temp_pitch = value
            temp_time_shift = 0
            temp_duration = 0
        elif event_type == 'time_shift':
            temp_time_shift = value
        elif event_type == 'duration':
            temp_duration = value
            # 如果 pitch 和 duration 都已知，则写入音符
            if temp_pitch is not None:
                start_time = current_time + temp_time_shift / (2 * ticks_per_beat)
                end_time = start_time + temp_duration / (2 * ticks_per_beat)
                note = pretty_midi.Note(velocity=64, pitch=temp_pitch + 21, start=start_time, end=end_time)
                piano.notes.append(note)
                current_time = start_time # 更新时间到音符开始
            temp_pitch = None # 重置临时变量
        # else: # special token, ignore
        i += 1
    
    # 处理最后一个可能未完成的事件
    if temp_pitch is not None and temp_time_shift is not None and temp_duration is not None:
        start_time = current_time + temp_time_shift / (2 * ticks_per_beat)
        end_time = start_time + temp_duration / (2 * ticks_per_beat)
        note = pretty_midi.Note(velocity=64, pitch=temp_pitch + 21, start=start_time, end=end_time)
        piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(output_path)
    print(f"Generated MIDI saved to {output_path}")


# --- 主预测脚本 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = TransformerMusic(vocab_size=VOCAB_SIZE, d_model=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}")

    # 加载并处理输入 MIDI
    input_events = midi_to_events(INPUT_MIDI_PATH)
    if input_events is None:
        print("Failed to load or parse input MIDI file.")
        exit()

    input_tokens = [BOS]
    for pitch, ts, dur in input_events:
        input_tokens.extend(event_to_token(pitch, ts, dur))
    # input_tokens.append(EOS) # 不需要在输入末尾添加 EOS

    print(f"Input sequence length: {len(input_tokens)} tokens.")

    # 生成
    generated_tokens = generate(model, input_tokens, MAX_GENERATE_TOKENS, TEMPERATURE, TOP_K, TOP_P)

    # 保存生成的 MIDI
    tokens_to_midi(generated_tokens, OUTPUT_MIDI_PATH)