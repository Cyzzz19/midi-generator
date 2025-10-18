import os
import torch
import torch.nn as nn
import pretty_midi
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# --- 配置参数 ---
MODEL_PATH = "model_epoch_7.pt"  # 模型文件路径 (训练后生成的文件)
INPUT_MIDI_PATH = "2.mid"  # 输入 MIDI 文件路径
OUTPUT_MIDI_PATH = "output.mid"  # 输出 MIDI 文件路径
MAX_SEQ_LEN = 512  # 模型一次处理的最大 token 长度
MAX_GENERATE_TOKENS = 512  # 生成多少个 token
# --- 增加随机性 ---
TEMPERATURE = 1.5  # 从 1.0 提高到 1.5
TOP_K = 20         # 限制候选 token 的数量
TOP_P = 0.95       # 保持 nucleus sampling
# --- 重复惩罚 ---
REPETITION_PENALTY = 2.0  # 重复 token 的惩罚系数
REPETITION_PENALTY_WINDOW = 10  # 检查最后 N 个 token

# --- 词表和特殊 token ---
PAD = 0
BOS = 1
EOS = 2
SEP = 3

# --- 字段范围 ---
PITCH_RANGE = (0, 87)  # 0-87 映射 MIDI 21-108
# --- 指数时间量化 ---
MAX_TIME_SHIFT_BUCKET = 10  # 桶的数量，time_shift 用 0-10 表示 (2^0 到 2^10)
MAX_DURATION_BUCKET = 10    # 桶的数量，duration 用 0-10 表示 (2^0 到 2^10)

# --- 字段偏移 ---
PITCH_OFFSET = 4
TIME_SHIFT_OFFSET = PITCH_OFFSET + 88
DURATION_OFFSET = TIME_SHIFT_OFFSET + MAX_TIME_SHIFT_BUCKET + 1
# --- 新的词表大小：三元组合并 ---
VOCAB_SIZE = 4 + 88 * (MAX_TIME_SHIFT_BUCKET + 1) * (MAX_DURATION_BUCKET + 1)

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

# --- 事件 → 单 token ID (三元组合并) (与训练时相同) ---
def quantize_time_to_bucket(time_value, max_bucket):
    """将时间值转换为桶索引，使用指数映射"""
    if time_value == 0:
        return 0
    bucket = min(max_bucket, int(np.log2(time_value)))
    return bucket

def dequantize_bucket_to_time(bucket_idx):
    """将桶索引转换回时间值，使用指数映射"""
    return 2 ** bucket_idx

def event_to_single_token(pitch, time_shift, duration):
    """将 (pitch, time_shift, duration) 三元组映射为一个唯一的 token ID"""
    # --- 指数量化 ---
    ts_bucket = quantize_time_to_bucket(time_shift, MAX_TIME_SHIFT_BUCKET)
    dur_bucket = quantize_time_to_bucket(duration, MAX_DURATION_BUCKET)
    
    token_id = 4 + pitch * (MAX_TIME_SHIFT_BUCKET+1) * (MAX_DURATION_BUCKET+1) + ts_bucket * (MAX_DURATION_BUCKET+1) + dur_bucket
    return token_id

def single_token_to_event(token_id):
    """将单 token ID 解析回 (pitch, time_shift, duration) 三元组"""
    if token_id < 4:
        return None # 特殊 token
    token_id -= 4
    pitch = token_id // ((MAX_TIME_SHIFT_BUCKET+1) * (MAX_DURATION_BUCKET+1))
    token_id %= (MAX_TIME_SHIFT_BUCKET+1) * (MAX_DURATION_BUCKET+1)
    time_shift_bucket = token_id // (MAX_DURATION_BUCKET+1)
    duration_bucket = token_id % (MAX_DURATION_BUCKET+1)
    
    # --- 反向指数量化 ---
    time_shift = dequantize_bucket_to_time(time_shift_bucket)
    duration = dequantize_bucket_to_time(duration_bucket)
    
    return pitch, time_shift, duration

# --- 采样辅助函数 ---
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    根据 top_k 和 top_p 过滤 logits
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][-1]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0

        # Create a mask for original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

# --- 重复惩罚函数 ---
def apply_repetition_penalty(logits, generated_tokens, penalty, window_size):
    """
    对最近生成的 tokens 应用惩罚
    """
    if len(generated_tokens) < 2:
        return logits
    
    # 取最近的 window_size 个 token
    recent_tokens = generated_tokens[-window_size:]
    
    # 对这些 token 的 logits 施加惩罚
    for token_id in recent_tokens:
        if token_id < logits.size(-1): # 确保 token_id 在范围内
            logits[token_id] = logits[token_id] / penalty
    
    return logits

# --- 修复后的模型：仅 Decoder (GPT-style) ---
class MusicTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=1024, nhead=8, num_layers=12, dim_feedforward=4096, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._generate_positional_encoding(d_model, max_len=2048)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
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

# --- 修复后的生成函数 ---
def generate(model, context_tokens, max_generate_tokens, temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0, repetition_penalty_window=3):
    """
    根据上下文 tokens 生成新的 tokens
    修复：使用 Decoder-only 模型进行自回归生成
    修复：增加重复惩罚
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 初始化生成序列：从上下文开始
    generated = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, L)

    for i in tqdm(range(max_generate_tokens), desc="Generating"):
        # 获取模型当前能看到的输入（限制长度）
        current_input = generated[:, -MAX_SEQ_LEN:]
        
        # 创建因果掩码
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(current_input.size(1)).to(device)
        # 创建 padding mask
        padding_mask = (current_input == PAD)

        with torch.no_grad():
            # 修复：仅使用 decoder，输入为 current_input
            output = model(current_input, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        
        # 预测下一个 token
        next_token_logits = output[0, -1, :] / temperature
        
        # --- 应用重复惩罚 ---
        if repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits, 
                generated[0].tolist(), 
                repetition_penalty, 
                repetition_penalty_window
            )
        
        # --- 注意：对于巨大词表，top_k/top_p 可能很慢 ---
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # 检查是否生成了 EOS
        if next_token_id == EOS:
            print("EOS token generated, stopping.")
            break

        # 将新 token 添加到生成序列中
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        generated = torch.cat((generated, next_token_tensor), dim=1)

    # 返回完整的生成序列（上下文 + 生成部分）
    return generated[0].cpu().tolist()

# --- 将生成的 tokens 转换为 MIDI 并保存 ---
def tokens_to_midi(tokens, output_path, ticks_per_beat=480):
    """
    将 token 序列转换为 MIDI 文件并保存
    修复：使用 single_token_to_event 解析
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=120, resolution=ticks_per_beat)
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    current_time = 0.0
    for token_id in tokens:
        if token_id == BOS or token_id == EOS or token_id == PAD or token_id == SEP:
            continue # 忽略特殊 token

        event_tuple = single_token_to_event(token_id)
        if event_tuple is not None:
            pitch, time_shift, duration = event_tuple
            # 计算实际时间（秒）
            start_time = current_time + time_shift / (2 * ticks_per_beat)
            end_time = start_time + duration / (2 * ticks_per_beat)
            
            # 创建音符
            note = pretty_midi.Note(
                velocity=64,
                pitch=pitch + 21,  # 转回 MIDI 音高
                start=start_time,
                end=end_time
            )
            piano.notes.append(note)
            current_time = start_time  # 更新当前时间
        # else: # token_id 是无效的，忽略

    pm.instruments.append(piano)
    pm.write(output_path)
    print(f"Generated MIDI saved to {output_path}")


# --- 主预测脚本 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型 (使用修复后的 Decoder-only 模型)
    model = MusicTransformerDecoder(vocab_size=VOCAB_SIZE, d_model=512)
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
        input_tokens.append(event_to_single_token(pitch, ts, dur)) # 单一 token
    # input_tokens.append(EOS) # 不需要在输入末尾添加 EOS

    print(f"Input sequence length: {len(input_tokens)} tokens.")

    # 生成
    generated_tokens = generate(
        model, 
        input_tokens, 
        MAX_GENERATE_TOKENS, 
        TEMPERATURE, 
        TOP_K, 
        TOP_P,
        REPETITION_PENALTY,
        REPETITION_PENALTY_WINDOW
    )

    # 保存生成的 MIDI
    tokens_to_midi(generated_tokens, OUTPUT_MIDI_PATH)