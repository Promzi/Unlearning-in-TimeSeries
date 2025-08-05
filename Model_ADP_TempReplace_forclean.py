import os
import glob
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, fftfreq
import sys
import shutil
import torch
import torch.nn.functional as F

# Add Autoformer to path
sys.path.insert(0, os.path.abspath('./Autoformer'))
from exp.exp_main import Exp_Main
from models.Autoformer import Model as Autoformer
import torch.nn as nn

# ========== TEMPORAL REPLACEMENT FUNCTIONS ==========
def extract_temporal_features(segment):
    """Extract temporal features: slope, mean, std, dominant frequency"""
    T = len(segment)
    x = np.array(segment)
    t = np.arange(T)
    
    # Handle edge cases
    if T < 2:
        return np.array([0.0, np.mean(x), 0.0, 0.0])
    
    slope = np.polyfit(t, x, 1)[0] if T > 1 else 0.0
    mean = np.mean(x)
    std = np.std(x)
    
    # FFT for dominant frequency
    if T > 2:
        fft_vals = np.abs(fft(x))
        fft_freqs = fftfreq(T)
        # Find dominant frequency (excluding DC component)
        if T // 2 > 1:
            fft_dom = fft_freqs[np.argmax(fft_vals[1:T // 2]) + 1]
        else:
            fft_dom = 0.0
    else:
        fft_dom = 0.0
    
    return np.array([slope, mean, std, np.abs(fft_dom)])

def attention_similarity(a_j, a_k, context_j, context_k, lam, eps):
    """Compute attention similarity with context decay"""
    # Handle edge cases
    if len(a_j) == 0 or len(a_k) == 0:
        return 0.0
        
    # Ensure same length
    min_len = min(len(a_j), len(a_k))
    a_j = a_j[:min_len]
    a_k = a_k[:min_len]
    
    # Cosine similarity
    norm_j = np.linalg.norm(a_j)
    norm_k = np.linalg.norm(a_k)
    
    if norm_j == 0 or norm_k == 0:
        return 0.0
        
    cosine_sim = np.dot(a_j, a_k) / (norm_j * norm_k + 1e-8)
    
    # Context distance and decay
    context_dist = np.linalg.norm(context_j - context_k)
    decay = np.exp(-lam * min(context_dist, np.sqrt(eps)))
    
    return cosine_sim * decay

def autocorr_score(x_orig, x_repl, tau_dom):
    """Compute autocorrelation preservation score"""
    T = len(x_orig)
    if T < 2:
        return 1.0
        
    max_lag = max(1, int(min(T // 4, max(1, round(1 / (tau_dom + 1e-8))))))
    scores = []
    
    for tau in range(1, min(max_lag + 1, T)):
        if T - tau > 0:
            r_orig = np.mean(x_orig[:-tau] * x_orig[tau:])
            r_repl = np.mean(x_repl[:-tau] * x_repl[tau:])
            scores.append(abs(r_orig - r_repl))
    
    if not scores:
        return 1.0
        
    return np.exp(-np.mean(scores))

def trend_continuity(pre, post):
    """Compute trend continuity between segments"""
    if len(pre) < 2 or len(post) < 2:
        return 1.0
        
    slope_pre = extract_temporal_features(pre)[0]
    slope_post = extract_temporal_features(post)[0]
    return np.exp(-abs(slope_pre - slope_post))

def season_alignment(pre, post, eps_amp):
    """Compute seasonal alignment between segments"""
    if len(pre) < 2 or len(post) < 2:
        return 1.0
        
    try:
        amp_pre = np.abs(fft(pre))
        amp_post = np.abs(fft(post))
        
        if np.abs(np.max(amp_pre) - np.max(amp_post)) > eps_amp:
            return 0.0
            
        phase_pre = np.angle(fft(pre))
        phase_post = np.angle(fft(post))
        
        phase_diff = np.mean(phase_pre - phase_post)
        return np.cos(phase_diff)
    except:
        return 1.0

def temporal_replacement(spurious_segments, clean_pool, attention_maps, tau_tc, eps_context,
                          lam=1.0, alpha=0.5, omega=(0.33, 0.33, 0.34), eps_amp=0.1):
    """
    Main temporal replacement function
    
    Args:
        spurious_segments: List of spurious time series segments
        clean_pool: List of clean segments for replacement candidates
        attention_maps: Dict mapping segment_key -> attention_vector
        tau_tc: Temporal consistency threshold
        eps_context: Context similarity threshold
        lam: Context decay parameter
        alpha: Balance between attention similarity and temporal consistency
        omega: Weights for (autocorr, trend, seasonal) components
        eps_amp: Amplitude threshold for seasonal alignment
    """
    replacement_map = {}
    quality_metrics = {}
    att_sims_global = []
    tc_scores_global = []

    for i, s_j in enumerate(spurious_segments):
        s_j_key = f"spurious_{i}"
        context_j = extract_temporal_features(s_j)
        
        # Get attention for this spurious segment
        if s_j_key in attention_maps:
            a_j = attention_maps[s_j_key]
        else:
            # Use mean attention if specific mapping not available
            a_j = np.ones(len(s_j))
            
        # Find candidates of similar length
        candidates = [(j, r) for j, r in enumerate(clean_pool) if abs(len(r) - len(s_j)) <= 2]
        
        if not candidates:
            replacement_map[s_j_key] = s_j  # No suitable candidates
            continue
            
        att_sims = {}
        tc_scores = {}

        # Compute attention similarities
        for j, r_k in candidates:
            r_k_key = f"clean_{j}"
            context_k = extract_temporal_features(r_k)
            
            if r_k_key in attention_maps:
                a_k = attention_maps[r_k_key]
            else:
                a_k = np.ones(len(r_k))
                
            sim = attention_similarity(a_j, a_k, context_j, context_k, lam, eps_context)
            att_sims[r_k_key] = sim

        # Find best replacement
        best_r = None
        best_r_key = None
        max_score = -np.inf

        for (j, r_k), (r_k_key, sim) in zip(candidates, att_sims.items()):
            if sim <= 0:
                continue
                
            # Adjust length if needed
            if len(r_k) != len(s_j):
                if len(r_k) > len(s_j):
                    r_k = r_k[:len(s_j)]
                else:
                    # Pad with edge values
                    pad_len = len(s_j) - len(r_k)
                    r_k = np.concatenate([r_k, np.full(pad_len, r_k[-1])])
            
            replaced = r_k
            tau_dom = context_j[3]
            
            # Compute temporal consistency components
            ac_score = autocorr_score(s_j, replaced, tau_dom)
            
            # For trend and seasonal, split segments in half
            mid_orig = len(s_j) // 2
            mid_repl = len(replaced) // 2
            
            if mid_orig > 0 and mid_repl > 0:
                tr_score = trend_continuity(s_j[:mid_orig], s_j[mid_orig:])
                sa_score = season_alignment(s_j[:mid_orig], s_j[mid_orig:], eps_amp)
            else:
                tr_score = 1.0
                sa_score = 1.0
            
            # Combine temporal consistency components
            tc = omega[0] * ac_score + omega[1] * tr_score + omega[2] * sa_score
            
            # Final combined score
            combined = alpha * sim + (1 - alpha) * tc
            tc_scores[r_k_key] = tc

            if combined > max_score:
                max_score = combined
                best_r = replaced
                best_r_key = r_k_key

        # Accept replacement if TC threshold is met
        if best_r is not None and best_r_key in tc_scores and tc_scores[best_r_key] >= tau_tc:
            replacement_map[s_j_key] = best_r
            quality_metrics[s_j_key] = {
                "AttSim": att_sims[best_r_key],
                "TC": tc_scores[best_r_key],
                "CombinedScore": max_score
            }
            att_sims_global.append(att_sims[best_r_key])
            tc_scores_global.append(tc_scores[best_r_key])
        else:
            # Fallback: keep original (could implement interpolation here)
            replacement_map[s_j_key] = s_j

    # Compute overall effectiveness
    if att_sims_global and tc_scores_global:
        avg_att = np.mean(att_sims_global)
        avg_tc = np.mean(tc_scores_global)
        effectiveness = 1 - np.exp(-1.0 * avg_att * avg_tc * np.sqrt(len(spurious_segments)))
    else:
        effectiveness = 0.0

    return replacement_map, quality_metrics, effectiveness

# ========== APD SCORING FUNCTIONS (EXISTING) ==========
def standardize(x):
    mu = np.mean(x)
    sigma = np.std(x) + 1e-6
    return (x - mu) / sigma

def compute_avg_attention(attention_map):
    if attention_map.ndim != 3:
        if attention_map.ndim == 4:
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 2:
            attention_map = attention_map[np.newaxis, :, :]
    return np.mean(np.sum(attention_map, axis=1), axis=0)

def compute_deviation(x, window=25):
    x = np.array(x)
    T = len(x)
    if x.ndim == 1:
        x = x[:, None]
    
    deviations = []
    for t in range(T):
        start = max(0, t - window)
        end = min(T, t + window + 1)
        local_mean = np.mean(x[start:end], axis=0)
        deviation = np.linalg.norm(x[t] - local_mean)
        deviations.append(deviation)
    
    return np.array(deviations)

def compute_marginal_loss_simple(y_true, y_pred):
    return np.abs(y_true - y_pred)

def compute_apd_score(bar_A, L_t, Dev_t, beta1=1.0, beta2=1.0, beta3=1.0):
    A_std = standardize(bar_A)
    L_std = standardize(L_t)
    D_std = standardize(Dev_t)
    return beta1 * A_std + beta2 * L_std + beta3 * D_std

def detect_spurious(apd_scores, adaptive=True, multiplier=2.0):
    if adaptive:
        median = np.median(apd_scores)
        mad = np.median(np.abs(apd_scores - median))
        threshold = median + multiplier * mad
    else:
        mu = np.mean(apd_scores)
        sigma = np.std(apd_scores)
        threshold = mu + multiplier * sigma
    
    return np.where(apd_scores > threshold)[0], threshold

def run_apd_pipeline(attention_map, y_true, y_pred, beta1=1.0, beta2=1.0, beta3=1.0, window=25):
    bar_A = compute_avg_attention(attention_map)
    Dev = compute_deviation(y_true, window=window)
    L = compute_marginal_loss_simple(y_true, y_pred)
    
    min_len = min(len(bar_A), len(Dev), len(L))
    bar_A = bar_A[:min_len]
    Dev = Dev[:min_len]
    L = L[:min_len]
    
    apd_scores = compute_apd_score(bar_A, L, Dev, beta1, beta2, beta3)
    spurious_idx, threshold = detect_spurious(apd_scores)
    
    components = {
        'attention': bar_A,
        'loss': L,
        'deviation': Dev
    }
    
    return apd_scores, spurious_idx, threshold, components

# ========== SEGMENT EXTRACTION AND REPLACEMENT FUNCTIONS ==========
def extract_segments_from_indices(time_series, spurious_indices, segment_size=5):
    """Extract segments around spurious indices"""
    segments = []
    T = len(time_series)
    
    for idx in spurious_indices:
        # Define segment boundaries
        start = max(0, idx - segment_size // 2)
        end = min(T, idx + segment_size // 2 + 1)
        
        segment = time_series[start:end]
        segments.append(segment)
    
    return segments

def extract_clean_segments(time_series, spurious_indices, segment_size=5, num_clean=20):
    """Extract clean segments for replacement pool"""
    T = len(time_series)
    clean_segments = []
    
    # Create exclusion zones around spurious indices
    exclusion_zones = set()
    for idx in spurious_indices:
        for i in range(max(0, idx - segment_size), min(T, idx + segment_size + 1)):
            exclusion_zones.add(i)
    
    # Extract clean segments
    potential_starts = [i for i in range(T - segment_size) if i not in exclusion_zones]
    
    if len(potential_starts) > num_clean:
        selected_starts = np.random.choice(potential_starts, num_clean, replace=False)
    else:
        selected_starts = potential_starts
    
    for start in selected_starts:
        end = start + segment_size
        if end <= T and all(i not in exclusion_zones for i in range(start, end)):
            clean_segments.append(time_series[start:end])
    
    return clean_segments

def create_attention_mapping(spurious_segments, clean_segments, attention_weights):
    """Create attention mapping for segments"""
    attention_maps = {}
    
    # Map spurious segments
    for i, segment in enumerate(spurious_segments):
        key = f"spurious_{i}"
        # Use average attention for the segment length
        if len(attention_weights) >= len(segment):
            attention_maps[key] = attention_weights[:len(segment)]
        else:
            # Pad with mean attention if needed
            padded = np.concatenate([attention_weights, 
                                   np.full(len(segment) - len(attention_weights), 
                                          np.mean(attention_weights))])
            attention_maps[key] = padded
    
    # Map clean segments
    for i, segment in enumerate(clean_segments):
        key = f"clean_{i}"
        # Use random subsection of attention weights
        if len(attention_weights) >= len(segment):
            start_idx = np.random.randint(0, len(attention_weights) - len(segment) + 1)
            attention_maps[key] = attention_weights[start_idx:start_idx + len(segment)]
        else:
            # Pad with mean attention if needed
            padded = np.concatenate([attention_weights, 
                                   np.full(len(segment) - len(attention_weights), 
                                          np.mean(attention_weights))])
            attention_maps[key] = padded
    
    return attention_maps

def apply_replacements_to_series(original_series, spurious_indices, replacement_map, segment_size=5):
    """Apply temporal replacements to the original time series"""
    modified_series = original_series.copy()
    T = len(original_series)
    
    for i, idx in enumerate(spurious_indices):
        spurious_key = f"spurious_{i}"
        
        if spurious_key in replacement_map:
            replacement = replacement_map[spurious_key]
            
            # Define replacement boundaries
            start = max(0, idx - segment_size // 2)
            end = min(T, idx + segment_size // 2 + 1)
            segment_len = end - start
            
            # Adjust replacement length to match segment
            if len(replacement) > segment_len:
                replacement = replacement[:segment_len]
            elif len(replacement) < segment_len:
                # Pad with edge values
                pad_len = segment_len - len(replacement)
                replacement = np.concatenate([replacement, np.full(pad_len, replacement[-1])])
            
            # Apply replacement
            modified_series[start:end] = replacement
    
    return modified_series

def visualize_replacement_results(ts_id, original_series, modified_series, spurious_indices, 
                                replacement_map, quality_metrics, effectiveness, output_dir):
    """Visualize temporal replacement results"""
    os.makedirs(os.path.join(output_dir, "temporal_replacement"), exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. Original vs Modified Series
    axes[0].plot(original_series, label='Original Series', alpha=0.7, linewidth=2)
    axes[0].plot(modified_series, label='After Replacement', alpha=0.7, linewidth=2, linestyle='--')
    axes[0].scatter(spurious_indices, original_series[spurious_indices], 
                   color='red', s=50, label=f'Spurious Points ({len(spurious_indices)})', zorder=5)
    axes[0].set_title(f'{ts_id} - Temporal Replacement Results (Effectiveness: {effectiveness:.3f})')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Difference plot
    diff = modified_series - original_series
    axes[1].plot(diff, color='purple', linewidth=2)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].scatter(spurious_indices, diff[spurious_indices], color='red', s=50, zorder=5)
    axes[1].set_title('Difference (Modified - Original)')
    axes[1].set_ylabel('Difference')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Quality metrics visualization
    if quality_metrics:
        positions = []
        att_sims = []
        tc_scores = []
        combined_scores = []
        
        for key, metrics in quality_metrics.items():
            # Extract position from spurious index
            spurious_idx = int(key.split('_')[1])
            if spurious_idx < len(spurious_indices):
                positions.append(spurious_indices[spurious_idx])
                att_sims.append(metrics['AttSim'])
                tc_scores.append(metrics['TC'])
                combined_scores.append(metrics['CombinedScore'])
        
        x_pos = np.arange(len(positions))
        width = 0.25
        
        axes[2].bar(x_pos - width, att_sims, width, label='Attention Similarity', alpha=0.7)
        axes[2].bar(x_pos, tc_scores, width, label='Temporal Consistency', alpha=0.7)
        axes[2].bar(x_pos + width, combined_scores, width, label='Combined Score', alpha=0.7)
        
        axes[2].set_xlabel('Replacement Index')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Replacement Quality Metrics')
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels([f'Pos {pos}' for pos in positions], rotation=45)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No replacements applied', 
                    transform=axes[2].transAxes, ha='center', va='center', fontsize=14)
        axes[2].set_title('Replacement Quality Metrics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_replacement", f"{ts_id}_replacement_results.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created temporal replacement visualizations for {ts_id}")

# ========== EXISTING CODE (ATTENTION CAPTURE, ETC.) ==========
attention_maps = {"encoder": {}, "decoder": []}

def make_encoder_hook(layer_idx):
    def hook_encoder_attention(module, input, output):
        try:
            if hasattr(module, 'last_attn'):
                attn = module.last_attn
                if isinstance(attn, torch.Tensor):
                    attn_np = attn.detach().cpu().numpy()
                    if attn_np.ndim == 4:
                        avg_attn = np.mean(attn_np, axis=0)
                        if layer_idx not in attention_maps["encoder"]:
                            attention_maps["encoder"][layer_idx] = []
                        attention_maps["encoder"][layer_idx].append(avg_attn)
        except Exception as e:
            print(f"[Hook ❌] Failed at layer {layer_idx}: {e}")
    return hook_encoder_attention

def hook_decoder_attention(module, input, output):
    try:
        attn = None
        if hasattr(module, 'cross_attention'):
            if hasattr(module.cross_attention, 'inner_correlation'):
                if hasattr(module.cross_attention.inner_correlation, 'last_attn'):
                    attn = module.cross_attention.inner_correlation.last_attn
        
        if attn is not None:
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().cpu().numpy()
                attention_maps["decoder"].append(attn_np)
    except Exception as e:
        print(f"[Hook ⚠️] Decoder attention hook failed: {e}")

# ========== CONFIG AND ARGS ==========
root_path = './Autoformer/dataset/m5_series_split'
output_dir = './autoformer_m5_outputs'
os.makedirs(output_dir, exist_ok=True)

series_ids = [f"FOODS_1_0{str(i).zfill(2)}_CA_1" for i in range(1, 2)]

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def create_args(ts_id, window=28, horizon=28):
    return Struct(
        model='Autoformer',
        data='custom',
        features='S',
        seq_len=window,
        label_len=window,
        pred_len=horizon,
        e_layers=2,
        d_layers=1,
        factor=3,
        enc_in=1, dec_in=1, c_out=1,
        d_model=512, d_ff=2048, n_heads=8,
        dropout=0.05, embed='timeF', freq='d',
        activation='gelu', output_attention=True,
        do_predict=True,
        moving_avg=25, use_gpu=True, gpu=0, use_multi_gpu=False,
        target='value', root_path=root_path,
        data_path=f'{ts_id}.csv',
        batch_size=16, learning_rate=0.0001, train_epochs=2,
        num_workers=0, des='Exp', itr=1,
        patience=3, lradj='type1', inverse=False,
        mix=True, cols=None,
        checkpoints='./checkpoints/', detail_freq='h',
        is_training=1, model_id=f'Autoformer_{ts_id}', loss='mse',
        use_amp=False, seasonal_patterns='Monthly'
    )

def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.maximum(y_true, 1)) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape}

# ========== MAIN LOOP WITH COMPLETE PIPELINE ==========
all_metrics = []
apd_results = []
replacement_results = []

for ts_id in series_ids[:1]:
    try:
        print(f"\n===== Running Complete Pipeline on {ts_id} =====")
        args = create_args(ts_id)
        exp = Exp_Main(args)

        # Clear previous attention maps
        attention_maps["encoder"].clear()
        attention_maps["decoder"].clear()

        # Register hooks (existing code)
        hooks = []
        for i, layer in enumerate(exp.model.encoder.attn_layers):
            try:
                module = layer.attention.inner_correlation
                hook = module.register_forward_hook(make_encoder_hook(i))
                hooks.append(hook)
                print(f"✅ Registered encoder hook on layer {i}")
            except Exception as e:
                print(f"❌ Failed to hook layer {i}: {e}")

        if hasattr(exp.model, 'decoder') and hasattr(exp.model.decoder, 'layers'):
            last_decoder_layer = exp.model.decoder.layers[-1]
            hook = last_decoder_layer.register_forward_hook(hook_decoder_attention)
            hooks.append(hook)
            print(f"✅ Registered decoder hook")

        # Train and test
        setting = f"{args.model_id}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}"
        exp.train(setting)
        exp.test(setting, test=1)

        # Create output directory
        series_output_dir = os.path.join(output_dir, ts_id, "results")
        os.makedirs(series_output_dir, exist_ok=True)

        # Extract predictions (existing prediction extraction code)
        y_pred = None
        y_true = None
        
        # Try to find predictions from files first
        possible_result_paths = [
            f'./results/{setting}/',
            f'./results/',
            f'./checkpoints/{setting}/',
            './outputs/',
            './exp_results/'
        ]
        
        for result_path in possible_result_paths:
            if os.path.exists(result_path):
                files = os.listdir(result_path)
                pred_patterns = ['pred.npy', 'preds.npy', 'prediction.npy', 'test_pred.npy']
                true_patterns = ['true.npy', 'trues.npy', 'ground_truth.npy', 'test_true.npy']
                
                pred_file = next((p for p in pred_patterns if p in files), None)
                true_file = next((t for t in true_patterns if t in files), None)
                
                if pred_file and true_file:
                    try:
                        pred_data = np.load(os.path.join(result_path, pred_file))
                        true_data = np.load(os.path.join(result_path, true_file))
                        y_pred = pred_data.reshape(-1)
                        y_true = true_data.reshape(-1)
                        print(f"✅ Found predictions in {result_path}")
                        break
                    except Exception as e:
                        continue

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if y_pred is None or y_true is None:
            print(f"❌ Could not extract predictions for {ts_id}")
            continue

        # Standard metrics
        metrics = evaluate_metrics(y_true, y_pred)
        metrics["series"] = ts_id
        all_metrics.append(metrics)

        # ========== STEP 1: APD ANALYSIS ==========
        print(f"🔍 Step 1: Running APD analysis for {ts_id}...")
        
        combined_attention = None
        if attention_maps["encoder"]:
            all_attentions = []
            for layer_idx, attn_list in attention_maps["encoder"].items():
                if attn_list:
                    layer_avg = np.mean(np.stack(attn_list, axis=0), axis=0)
                    all_attentions.append(layer_avg)
            
            if all_attentions:
                combined_attention = np.mean(np.stack(all_attentions, axis=0), axis=0)
                print(f"✅ Combined attention shape: {combined_attention.shape}")

        if combined_attention is not None:
            # Run APD analysis
            pred_len = min(len(y_true), len(y_pred), combined_attention.shape[-1])
            y_true_apd = y_true[:pred_len]
            y_pred_apd = y_pred[:pred_len]
            
            apd_scores, spurious_idx, threshold, components = run_apd_pipeline(
                attention_map=combined_attention,
                y_true=y_true_apd,
                y_pred=y_pred_apd,
                beta1=1.0, beta2=1.0, beta3=1.0, window=25
            )
            
            # Store APD results
            apd_result = {
                'series': ts_id,
                'num_spurious': len(spurious_idx),
                'spurious_ratio': len(spurious_idx) / len(apd_scores),
                'avg_apd_score': np.mean(apd_scores),
                'max_apd_score': np.max(apd_scores),
                'threshold': threshold,
                'spurious_indices': spurious_idx.tolist()
            }
            apd_results.append(apd_result)
            
            print(f"✅ APD Analysis Complete: {len(spurious_idx)} spurious patterns detected")

            # ========== STEP 2: TEMPORAL REPLACEMENT ==========
            if len(spurious_idx) > 0:
                print(f"🔄 Step 2: Running Temporal Replacement for {len(spurious_idx)} spurious patterns...")
                
                # Extract spurious and clean segments
                segment_size = 5  # Configurable segment size
                spurious_segments = extract_segments_from_indices(y_true_apd, spurious_idx, segment_size)
                clean_segments = extract_clean_segments(y_true_apd, spurious_idx, segment_size, num_clean=20)
                
                print(f"   📊 Extracted {len(spurious_segments)} spurious segments")
                print(f"   📊 Extracted {len(clean_segments)} clean segments")
                
                if clean_segments:
                    # Create attention mapping for segments
                    attention_weights = components['attention']  # Use average attention from APD
                    attention_segment_maps = create_attention_mapping(
                        spurious_segments, clean_segments, attention_weights
                    )
                    
                    # Run temporal replacement
                    replacement_map, quality_metrics, effectiveness = temporal_replacement(
                        spurious_segments=spurious_segments,
                        clean_pool=clean_segments,
                        attention_maps=attention_segment_maps,
                        tau_tc=0.7,        # Temporal consistency threshold
                        eps_context=0.01,  # Context similarity threshold
                        lam=1.0,          # Context decay parameter
                        alpha=0.6,        # Balance: 60% attention, 40% temporal consistency
                        omega=(0.33, 0.33, 0.34),  # Weights for autocorr, trend, seasonal
                        eps_amp=0.1       # Amplitude threshold
                    )
                    
                    # Apply replacements to the time series
                    modified_series = apply_replacements_to_series(
                        y_true_apd, spurious_idx, replacement_map, segment_size
                    )
                    
                    # Calculate improvement metrics
                    original_mse = mean_squared_error(y_true_apd, y_pred_apd)
                    # For modified series comparison, we'd need to re-run the model, 
                    # but for now we'll compare against original
                    modification_impact = np.mean(np.abs(modified_series - y_true_apd))
                    
                    # Store replacement results
                    replacement_result = {
                        'series': ts_id,
                        'effectiveness': effectiveness,
                        'num_replacements': sum(1 for k, v in replacement_map.items() 
                                              if not np.array_equal(v, spurious_segments[int(k.split('_')[1])])),
                        'avg_att_sim': np.mean([m['AttSim'] for m in quality_metrics.values()]) if quality_metrics else 0,
                        'avg_tc_score': np.mean([m['TC'] for m in quality_metrics.values()]) if quality_metrics else 0,
                        'modification_impact': modification_impact,
                        'original_mse': original_mse
                    }
                    replacement_results.append(replacement_result)
                    
                    # Create visualizations
                    visualize_replacement_results(
                        ts_id, y_true_apd, modified_series, spurious_idx,
                        replacement_map, quality_metrics, effectiveness, series_output_dir
                    )
                    
                    # Save replacement data
                    np.save(os.path.join(series_output_dir, "modified_series.npy"), modified_series)
                    np.save(os.path.join(series_output_dir, "original_series.npy"), y_true_apd)
                    
                    # Save replacement details
                    replacement_df = pd.DataFrame({
                        'spurious_index': spurious_idx,
                        'was_replaced': [f"spurious_{i}" in replacement_map and 
                                       not np.array_equal(replacement_map[f"spurious_{i}"], spurious_segments[i])
                                       for i in range(len(spurious_segments))],
                        'original_value': y_true_apd[spurious_idx],
                        'modified_value': modified_series[spurious_idx],
                        'improvement': np.abs(y_true_apd[spurious_idx] - modified_series[spurious_idx])
                    })
                    replacement_df.to_csv(os.path.join(series_output_dir, "replacement_details.csv"), index=False)
                    
                    print(f"✅ Temporal Replacement Complete:")
                    print(f"   🎯 Effectiveness: {effectiveness:.3f}")
                    print(f"   🔄 Replacements applied: {replacement_result['num_replacements']}/{len(spurious_segments)}")
                    print(f"   📊 Avg Attention Similarity: {replacement_result['avg_att_sim']:.3f}")
                    print(f"   📊 Avg Temporal Consistency: {replacement_result['avg_tc_score']:.3f}")
                    
                else:
                    print("⚠️ No clean segments available for replacement")
            else:
                print("ℹ️ No spurious patterns detected, skipping temporal replacement")

            # Save all analysis data
            np.save(os.path.join(series_output_dir, "apd_scores.npy"), apd_scores)
            np.save(os.path.join(series_output_dir, "spurious_indices.npy"), spurious_idx)
            
            # Detailed APD results
            apd_df = pd.DataFrame({
                'time_step': range(len(apd_scores)),
                'apd_score': apd_scores,
                'attention': components['attention'],
                'prediction_error': components['loss'],
                'local_deviation': components['deviation'],
                'is_spurious': [i in spurious_idx for i in range(len(apd_scores))]
            })
            apd_df.to_csv(os.path.join(series_output_dir, "apd_detailed_results.csv"), index=False)
            
        else:
            print(f"⚠️ No attention maps available for analysis on {ts_id}")

        # Save standard predictions
        np.save(os.path.join(series_output_dir, "pred.npy"), y_pred)
        np.save(os.path.join(series_output_dir, "true.npy"), y_true)
        
        df_series = pd.DataFrame({"GroundTruth": y_true, "Prediction": y_pred})
        df_series.to_csv(os.path.join(series_output_dir, "pred_vs_true.csv"), index=False)

    except Exception as e:
        print(f"❌ Error processing {ts_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ========== FINAL COMPREHENSIVE RESULTS ==========
print(f"\n{'='*60}")
print(f"COMPLETE AUTOFORMER + APD + TEMPORAL REPLACEMENT ANALYSIS")
print(f"{'='*60}")

# Standard metrics
if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(output_dir, "standard_metrics.csv"), index=False)
    print(f"\n📊 STANDARD FORECASTING METRICS:")
    print(df_metrics.to_string(index=False))

# APD results
if apd_results:
    df_apd = pd.DataFrame(apd_results)
    df_apd.to_csv(os.path.join(output_dir, "apd_analysis_results.csv"), index=False)
    print(f"\n🎯 APD SPURIOUS PATTERN DETECTION:")
    print(df_apd.to_string(index=False))
    
    print(f"\n📈 APD SUMMARY:")
    print(f"   • Spurious patterns detected: {df_apd['num_spurious'].sum()} total")
    print(f"   • Average spurious ratio: {df_apd['spurious_ratio'].mean()*100:.1f}%")
    print(f"   • Average APD score: {df_apd['avg_apd_score'].mean():.3f}")

# Temporal replacement results
if replacement_results:
    df_replacement = pd.DataFrame(replacement_results)
    df_replacement.to_csv(os.path.join(output_dir, "temporal_replacement_results.csv"), index=False)
    print(f"\n🔄 TEMPORAL REPLACEMENT RESULTS:")
    print(df_replacement.to_string(index=False))
    
    print(f"\n📈 REPLACEMENT SUMMARY:")
    print(f"   • Average effectiveness: {df_replacement['effectiveness'].mean():.3f}")
    print(f"   • Total replacements applied: {df_replacement['num_replacements'].sum()}")
    print(f"   • Average attention similarity: {df_replacement['avg_att_sim'].mean():.3f}")
    print(f"   • Average temporal consistency: {df_replacement['avg_tc_score'].mean():.3f}")
    print(f"   • Average modification impact: {df_replacement['modification_impact'].mean():.3f}")

# Create summary visualization
if all_metrics and apd_results and replacement_results:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Standard metrics
    metrics_cols = ['MSE', 'MAE', 'RMSE', 'MAPE', 'WAPE']
    available_metrics = [col for col in metrics_cols if col in df_metrics.columns]
    if available_metrics:
        df_metrics[available_metrics].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Standard Forecasting Metrics')
        axes[0,0].set_ylabel('Metric Value')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. APD detection rates
    axes[0,1].bar(df_apd['series'], df_apd['spurious_ratio'] * 100)
    axes[0,1].set_title('Spurious Pattern Detection Rate')
    axes[0,1].set_ylabel('Spurious Ratio (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Replacement effectiveness
    axes[1,0].bar(df_replacement['series'], df_replacement['effectiveness'])
    axes[1,0].set_title('Temporal Replacement Effectiveness')
    axes[1,0].set_ylabel('Effectiveness Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Quality metrics comparison
    x_pos = np.arange(len(df_replacement))
    width = 0.35
    axes[1,1].bar(x_pos - width/2, df_replacement['avg_att_sim'], width, 
                  label='Attention Similarity', alpha=0.7)
    axes[1,1].bar(x_pos + width/2, df_replacement['avg_tc_score'], width, 
                  label='Temporal Consistency', alpha=0.7)
    axes[1,1].set_title('Replacement Quality Metrics')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(df_replacement['series'], rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complete_analysis_summary.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()

print(f"\n✅ COMPLETE PIPELINE FINISHED!")
print(f"📁 All results saved to: {output_dir}")
print(f"🔍 Check subdirectories for:")
print(f"   • APD analysis visualizations")
print(f"   • Temporal replacement results")
print(f"   • Modified time series data")
print(f"   • Quality metrics and effectiveness scores")

# ========== FINAL RECOMMENDATIONS ==========
if replacement_results:
    avg_effectiveness = df_replacement['effectiveness'].mean()
    avg_replacements = df_replacement['num_replacements'].mean()
    
    print(f"\n🎯 RECOMMENDATIONS:")
    if avg_effectiveness > 0.5:
        print("✅ High replacement effectiveness - temporal replacement is working well")
    elif avg_effectiveness > 0.3:
        print("⚠️ Moderate effectiveness - consider tuning parameters (alpha, tau_tc, omega)")
    else:
        print("❌ Low effectiveness - review attention patterns and segment selection")
    
    if avg_replacements > len(spurious_idx) * 0.7:
        print("✅ Most spurious patterns successfully replaced")
    else:
        print("⚠️ Many spurious patterns not replaced - consider lowering tau_tc threshold")
    
    print(f"\n🔧 PARAMETER TUNING SUGGESTIONS:")
    print(f"   • Current alpha (attention weight): 0.6")
    print(f"   • Current tau_tc (TC threshold): 0.7")  
    print(f"   • Current segment_size: 5")
    print(f"   • Try adjusting these based on effectiveness scores")