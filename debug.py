import os
import torch
import torch.nn as nn
import pretty_midi
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 配置参数 ---
MODEL_PATH = "model_epoch_12.pt"  # 模型文件路径 (训练后生成的文件)
# 如果想测试生成，可以提供一个输入 MIDI
INPUT_MIDI_PATH = "1.mid" # "path/to/your/input.mid"  # 设为 None 则从随机 token 开始
MAX_GENERATE_TOKENS = 1536  # 生成多少个 token (例如 512 个三元组)
TEMPERATURE = 1.0  # 采样温度，控制随机性
TOP_K = 0  # Top-k 采样，0 表示不使用
TOP_P = 0.9  # Nucleus sampling，0 表示不使用
MAX_TIME_STEPS = 2000  # 图像显示的最大时间步（tick）
MAX_PITCH = 108      # 图像显示的最大音高（MIDI 108 为最高）
MIN_PITCH = 21       # 图像显示的最小音高（MIDI 21 为最低 A0）

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
def event_to_token_ids(pitch, time_shift, duration):
    """返回一个包含 3 个 token ID 的列表"""
    ts = min(time_shift, MAX_TIME_SHIFT)
    dur = min(duration, MAX_DURATION)
    return [
        PITCH_OFFSET + pitch,
        TIME_SHIFT_OFFSET + ts,
        DURATION_OFFSET + dur
    ]

# --- Token ID → 事件 (用于生成) ---
def token_to_event_type_and_value(token_id):
    """辅助函数，用于调试，确定 token 类型"""
    if token_id >= DURATION_OFFSET and token_id < DURATION_OFFSET + MAX_DURATION + 1:
        return 'duration', token_id - DURATION_OFFSET
    elif token_id >= TIME_SHIFT_OFFSET and token_id < DURATION_OFFSET:
        return 'time_shift', token_id - TIME_SHIFT_OFFSET
    elif token_id >= PITCH_OFFSET and token_id < TIME_SHIFT_OFFSET:
        return 'pitch', token_id - PITCH_OFFSET
    else:
        return 'special', token_id

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

# --- 修复后的模型：仅 Decoder (GPT-style) ---
class MusicTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
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
def generate(model, context_tokens, max_generate_tokens, temperature=1.0, top_k=0, top_p=0.0):
    """
    根据上下文 tokens 生成新的 tokens
    修复：使用 Decoder-only 模型进行自回归生成
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 初始化生成序列：从上下文开始
    generated = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, L)

    for _ in tqdm(range(max_generate_tokens), desc="Generating"):
        # 获取模型当前能看到的输入（限制长度）
        current_input = generated[:, -512:] # 使用一个固定的上下文长度（例如 512）
        
        # 创建因果掩码
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(current_input.size(1)).to(device)
        # 创建 padding mask
        padding_mask = (current_input == PAD)

        with torch.no_grad():
            # 修复：仅使用 decoder，输入为 current_input
            output = model(current_input, tgt_mask=tgt_mask, tgt_key_padding_mask=padding_mask)
        
        # 预测下一个 token
        next_token_logits = output[0, -1, :] / temperature
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

# --- 将生成的 tokens 转换为 MIDI notes (用于可视化) ---
def tokens_to_notes(tokens):
    """
    将 token 序列转换为 [(start_tick, end_tick, pitch), ...] 列表
    用于可视化，不保存 MIDI 文件
    """
    notes = []
    current_time = 0.0
    i = 0

    while i < len(tokens):
        token_id = tokens[i]
        
        # 检查是否为 pitch token
        if PITCH_OFFSET <= token_id < TIME_SHIFT_OFFSET:
            # 尝试读取接下来的两个 token 作为 time_shift 和 duration
            if i + 2 < len(tokens):
                next1_id = tokens[i + 1]
                next2_id = tokens[i + 2]
                
                # 验证 next1 是否为 time_shift, next2 是否为 duration
                if (TIME_SHIFT_OFFSET <= next1_id < DURATION_OFFSET and
                    DURATION_OFFSET <= next2_id < VOCAB_SIZE):
                    
                    # 解析三元组
                    pitch = token_id - PITCH_OFFSET + 21 # 转回 MIDI 音高
                    time_shift = next1_id - TIME_SHIFT_OFFSET
                    duration = next2_id - DURATION_OFFSET
                    
                    # 计算实际时间（tick）
                    start_tick = current_time + time_shift
                    end_tick = start_tick + duration
                    
                    notes.append((start_tick, end_tick, pitch))
                    current_time = start_tick  # 更新当前时间
                    
                    i += 3  # 跳过已处理的 3 个 token
                    continue
                else:
                    # 三元组不完整，跳过当前 token
                    print(f"Warning: Invalid token sequence at index {i}: {token_id}, {next1_id}, {next2_id}")
                    i += 1
            else:
                # 序列结尾，不足以形成三元组，跳过
                i += 1
        else:
            # 不是 pitch token，跳过
            i += 1

    return notes

# --- 可视化钢琴卷帘图 ---
def plot_piano_roll(notes, output_path="debug_piano_roll.png", max_time_steps=MAX_TIME_STEPS, min_pitch=MIN_PITCH, max_pitch=MAX_PITCH):
    """
    绘制钢琴卷帘图
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for start_tick, end_tick, pitch in notes:
        if min_pitch <= pitch <= max_pitch and start_tick < max_time_steps:
            # 创建一个矩形代表音符
            rect = patches.Rectangle(
                (start_tick, pitch - 0.5),  # (x, y) bottom-left corner
                end_tick - start_tick,      # width
                1,                          # height
                linewidth=0, alpha=0.8, facecolor='blue'
            )
            ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xlim(0, max_time_steps)
    ax.set_ylim(min_pitch - 1, max_pitch + 1)
    ax.set_xlabel('Time (ticks, 120 BPM)')
    ax.set_ylabel('MIDI Pitch')
    ax.set_title('Piano Roll Visualization of Generated MIDI')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # 为了更清晰，可以只标注部分 y 轴刻度
    y_ticks = range(min_pitch, max_pitch + 1, 12) # 每 12 个音符（一个八度）标注一次
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([pretty_midi.note_number_to_name(p) for p in y_ticks])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Piano roll saved to {output_path}")
    plt.show() # 显示图像

# --- 主调试脚本 ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型 (使用修复后的 Decoder-only 模型)
    model = MusicTransformerDecoder(vocab_size=VOCAB_SIZE, d_model=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print(f"Model loaded from {MODEL_PATH}")

    # 准备上下文
    if INPUT_MIDI_PATH:
        input_events = midi_to_events(INPUT_MIDI_PATH)
        if input_events is None:
            print("Failed to load or parse input MIDI file.")
            exit()
        context_tokens = [BOS]
        for pitch, ts, dur in input_events:
            context_tokens.extend(event_to_token_ids(pitch, ts, dur)) # [pitch, ts, dur]
        print(f"Input sequence length: {len(context_tokens)} tokens.")
    else:
        # 从随机 token 开始生成
        context_tokens = [BOS]
        print("Starting generation from BOS token.")

    # 生成
    generated_tokens = generate(model, context_tokens, MAX_GENERATE_TOKENS, TEMPERATURE, TOP_K, TOP_P)
        # 在 generated_tokens = generate(...) 之后添加
    print("\n--- Generated Tokens (first 30) ---")
    for i, token in enumerate(generated_tokens[:30]):
        token_type, value = token_to_event_type_and_value(token)
        print(f"Index {i}: TokenID={token}, Type={token_type}, Value={value}")

    print(f"\nTotal generated tokens: {len(generated_tokens)}")

    # 将生成的 tokens 转换为 notes
    notes = tokens_to_notes(generated_tokens)
    print(f"Generated {len(notes)} notes.")

    # 可视化
    plot_piano_roll(notes)

    # (可选) 保存 MIDI 文件
    # if INPUT_MIDI_PATH:
    #     output_midi_path = f"generated_from_{os.path.basename(INPUT_MIDI_PATH)}"
    # else:
    #     output_midi_path = "generated_from_scratch.mid"
    # tokens_to_midi(generated_tokens, output_midi_path)