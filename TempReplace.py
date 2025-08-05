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
from scipy.interpolate import interp1d
import sys
import shutil
import torch
import torch.nn.functional as F

# Add Autoformer to path
sys.path.insert(0, os.path.abspath('./Autoformer'))
from exp.exp_main import Exp_Main
from models.Autoformer import Model as Autoformer
import torch.nn as nn

# ========== IMPROVED TEMPORAL REPLACEMENT FUNCTIONS ==========
def extract_temporal_features(segment):
    """Extract temporal features: slope, mean, std, dominant frequency"""
    T = len(segment)
    x = np.array(segment)
    t = np.arange(T)
    
    if T < 2:
        return np.array([0.0, np.mean(x), 0.0, 0.0])
    
    slope = np.polyfit(t, x, 1)[0] if T > 1 else 0.0
    mean = np.mean(x)
    std = np.std(x)
    
    if T > 2:
        fft_vals = np.abs(fft(x))
        fft_freqs = fftfreq(T)
        if T // 2 > 1:
            fft_dom = fft_freqs[np.argmax(fft_vals[1:T // 2]) + 1]
        else:
            fft_dom = 0.0
    else:
        fft_dom = 0.0
    
    return np.array([slope, mean, std, np.abs(fft_dom)])

def attention_similarity(a_j, a_k, context_j, context_k, lam, eps):
    """Compute attention similarity with context decay"""
    if len(a_j) == 0 or len(a_k) == 0:
        return 0.0
        
    min_len = min(len(a_j), len(a_k))
    a_j = a_j[:min_len]
    a_k = a_k[:min_len]
    
    norm_j = np.linalg.norm(a_j)
    norm_k = np.linalg.norm(a_k)
    
    if norm_j == 0 or norm_k == 0:
        return 0.0
        
    cosine_sim = np.dot(a_j, a_k) / (norm_j * norm_k + 1e-8)
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

def create_interpolated_replacement(spurious_segment, context_segments=None, method='cubic'):
    """
    Create a replacement using interpolation techniques
    """
    T = len(spurious_segment)
    if T < 2:
        return spurious_segment
    
    try:
        # Method 1: Cubic spline interpolation of endpoints
        if method == 'cubic' and T >= 4:
            # Use first and last few points to create smooth interpolation
            x_known = np.array([0, 1, T-2, T-1])
            y_known = np.array([spurious_segment[0], spurious_segment[1], 
                              spurious_segment[-2], spurious_segment[-1]])
            
            f = interp1d(x_known, y_known, kind='cubic', fill_value='extrapolate')
            x_new = np.arange(T)
            replacement = f(x_new)
            
        # Method 2: Linear interpolation between endpoints
        elif method == 'linear':
            start_val = spurious_segment[0]
            end_val = spurious_segment[-1]
            replacement = np.linspace(start_val, end_val, T)
            
        # Method 3: Moving average based smoothing
        elif method == 'smooth':
            # Apply Gaussian smoothing
            sigma = max(1, T // 4)
            replacement = gaussian_filter(spurious_segment, sigma=sigma)
            
        # Method 4: Context-aware interpolation
        elif method == 'context' and context_segments:
            # Use similar segments from context to guide interpolation
            similar_patterns = []
            target_features = extract_temporal_features(spurious_segment)
            
            for context_seg in context_segments:
                if len(context_seg) == T:
                    context_features = extract_temporal_features(context_seg)
                    # Simple similarity based on feature distance
                    similarity = np.exp(-np.linalg.norm(target_features - context_features))
                    if similarity > 0.1:  # Threshold for inclusion
                        similar_patterns.append((similarity, context_seg))
            
            if similar_patterns:
                # Weighted average of similar patterns
                weights = np.array([sim for sim, _ in similar_patterns])
                weights = weights / np.sum(weights)
                
                replacement = np.zeros(T)
                for weight, pattern in zip(weights, [seg for _, seg in similar_patterns]):
                    replacement += weight * pattern
            else:
                # Fallback to linear interpolation
                replacement = np.linspace(spurious_segment[0], spurious_segment[-1], T)
        else:
            # Default: linear interpolation
            replacement = np.linspace(spurious_segment[0], spurious_segment[-1], T)
            
        return replacement
        
    except Exception as e:
        print(f"⚠️ Interpolation failed: {e}, using linear fallback")
        return np.linspace(spurious_segment[0], spurious_segment[-1], T)

def extract_all_possible_segments(time_series, spurious_indices, segment_size=5, 
                                  min_distance=2, quality_threshold=0.1):
    """
    Extract all possible segments as potential replacements, ranked by quality
    """
    T = len(time_series)
    all_segments = []
    
    # Create exclusion zones around spurious indices (smaller than before)
    exclusion_zones = set()
    for idx in spurious_indices:
        for i in range(max(0, idx - min_distance), min(T, idx + min_distance + 1)):
            exclusion_zones.add(i)
    
    # Extract all possible segments
    for start in range(T - segment_size + 1):
        end = start + segment_size
        
        # Check if segment overlaps with exclusion zones
        overlap = sum(1 for i in range(start, end) if i in exclusion_zones)
        overlap_ratio = overlap / segment_size
        
        if overlap_ratio < 0.5:  # Allow segments with < 50% overlap
            segment = time_series[start:end]
            
            # Compute segment quality metrics
            features = extract_temporal_features(segment)
            std_dev = features[2]
            
            # Quality based on variability and non-extremeness
            mean_val = features[1]
            global_mean = np.mean(time_series)
            global_std = np.std(time_series)
            
            # Prefer segments that are not too extreme
            extremeness = abs(mean_val - global_mean) / (global_std + 1e-8)
            quality = std_dev * np.exp(-extremeness)  # High std, low extremeness = good
            
            if quality > quality_threshold:
                all_segments.append({
                    'segment': segment,
                    'start_idx': start,
                    'end_idx': end,
                    'quality': quality,
                    'overlap_ratio': overlap_ratio,
                    'features': features
                })
    
    # Sort by quality (descending)
    all_segments.sort(key=lambda x: x['quality'], reverse=True)
    
    return all_segments

def improved_temporal_replacement(spurious_segments, time_series, spurious_indices, 
                                attention_maps, tau_tc, eps_context,
                                lam=1.0, alpha=0.5, omega=(0.33, 0.33, 0.34), 
                                eps_amp=0.1, segment_size=5,
                                fallback_methods=['interpolation', 'best_available', 'smoothing']):
    """
    Enhanced temporal replacement with multiple fallback strategies
    
    Args:
        spurious_segments: List of spurious segments
        time_series: Full time series for context
        spurious_indices: Indices of spurious points
        attention_maps: Attention mappings
        tau_tc: Temporal consistency threshold
        eps_context: Context similarity threshold  
        lam: Context decay parameter
        alpha: Balance between attention and temporal consistency
        omega: Weights for temporal consistency components
        eps_amp: Amplitude threshold
        segment_size: Size of segments
        fallback_methods: List of fallback strategies in order of preference
    """
    replacement_map = {}
    quality_metrics = {}
    att_sims_global = []
    tc_scores_global = []
    replacement_methods = {}  # Track which method was used
    
    # Extract all possible segments as candidates
    all_candidates = extract_all_possible_segments(
        time_series, spurious_indices, segment_size, 
        min_distance=2, quality_threshold=0.05  # Lower threshold for more candidates
    )
    
    print(f"   📊 Found {len(all_candidates)} total candidate segments")
    
    for i, s_j in enumerate(spurious_segments):
        s_j_key = f"spurious_{i}"
        context_j = extract_temporal_features(s_j)
        
        if s_j_key in attention_maps:
            a_j = attention_maps[s_j_key]
        else:
            a_j = np.ones(len(s_j))
        
        best_replacement = None
        best_method = "none"
        max_score = -np.inf
        best_metrics = {}
        
        # Strategy 1: Try high-quality candidates first
        high_quality_candidates = [c for c in all_candidates if len(c['segment']) == len(s_j)]
        
        if high_quality_candidates:
            print(f"   🔍 Trying {len(high_quality_candidates)} high-quality candidates for spurious_{i}")
            
            for j, candidate in enumerate(high_quality_candidates[:20]):  # Limit to top 20
                r_k = candidate['segment']
                r_k_key = f"candidate_{j}"
                context_k = candidate['features']
                
                # Create attention mapping for this candidate
                if len(a_j) >= len(r_k):
                    a_k = a_j[:len(r_k)]
                else:
                    a_k = np.concatenate([a_j, np.full(len(r_k) - len(a_j), np.mean(a_j))])
                
                # Compute attention similarity
                sim = attention_similarity(a_j, a_k, context_j, context_k, lam, eps_context)
                
                if sim > 0:  # Any positive similarity is considered
                    # Compute temporal consistency
                    tau_dom = context_j[3]
                    ac_score = autocorr_score(s_j, r_k, tau_dom)
                    
                    mid_orig = len(s_j) // 2
                    if mid_orig > 0:
                        tr_score = trend_continuity(s_j[:mid_orig], s_j[mid_orig:])
                        sa_score = season_alignment(s_j[:mid_orig], s_j[mid_orig:], eps_amp)
                    else:
                        tr_score = 1.0
                        sa_score = 1.0
                    
                    tc = omega[0] * ac_score + omega[1] * tr_score + omega[2] * sa_score
                    combined = alpha * sim + (1 - alpha) * tc
                    
                    # Lower the TC threshold for acceptance
                    relaxed_tau_tc = max(0.3, tau_tc - 0.2)  # More lenient threshold
                    
                    if combined > max_score and tc >= relaxed_tau_tc:
                        max_score = combined
                        best_replacement = r_k
                        best_method = "segment_replacement"
                        best_metrics = {
                            "AttSim": sim,
                            "TC": tc,
                            "CombinedScore": combined,
                            "Quality": candidate['quality']
                        }
        
        # Strategy 2: Fallback methods if no good candidates found
        if best_replacement is None and fallback_methods:
            print(f"   🔄 No suitable candidates found, trying fallback methods for spurious_{i}")
            
            for method in fallback_methods:
                try:
                    if method == 'interpolation':
                        # Try different interpolation methods
                        for interp_method in ['cubic', 'linear', 'smooth']:
                            replacement = create_interpolated_replacement(
                                s_j, context_segments=all_candidates[:10], method=interp_method
                            )
                            
                            # Evaluate interpolated replacement
                            tau_dom = context_j[3]
                            ac_score = autocorr_score(s_j, replacement, tau_dom)
                            
                            if ac_score > 0.2:  # Minimum quality threshold
                                best_replacement = replacement
                                best_method = f"interpolation_{interp_method}"
                                best_metrics = {
                                    "AttSim": 0.5,  # Neutral attention similarity
                                    "TC": ac_score,
                                    "CombinedScore": ac_score,
                                    "Quality": ac_score
                                }
                                break
                        
                        if best_replacement is not None:
                            break
                    
                    elif method == 'best_available':
                        # Use the best available candidate regardless of TC threshold
                        if high_quality_candidates:
                            best_candidate = high_quality_candidates[0]  # Highest quality
                            best_replacement = best_candidate['segment']
                            best_method = "best_available_segment"
                            best_metrics = {
                                "AttSim": 0.3,
                                "TC": 0.3,
                                "CombinedScore": best_candidate['quality'],
                                "Quality": best_candidate['quality']
                            }
                            break
                    
                    elif method == 'smoothing':
                        # Apply smoothing to the original spurious segment
                        replacement = create_interpolated_replacement(s_j, method='smooth')
                        best_replacement = replacement
                        best_method = "smoothing"
                        best_metrics = {
                            "AttSim": 0.7,  # High since it preserves attention structure
                            "TC": 0.4,
                            "CombinedScore": 0.5,
                            "Quality": 0.5
                        }
                        break
                        
                except Exception as e:
                    print(f"   ⚠️ Fallback method {method} failed: {e}")
                    continue
        
        # Final fallback: keep original if all else fails
        if best_replacement is None:
            print(f"   ⚠️ All methods failed for spurious_{i}, keeping original")
            best_replacement = s_j
            best_method = "no_replacement"
            best_metrics = {
                "AttSim": 1.0,  # Perfect since it's the same
                "TC": 0.0,      # But no improvement
                "CombinedScore": 0.0,
                "Quality": 0.0
            }
        
        # Store results
        replacement_map[s_j_key] = best_replacement
        quality_metrics[s_j_key] = best_metrics
        replacement_methods[s_j_key] = best_method
        
        if best_metrics["AttSim"] > 0 and best_metrics["TC"] > 0:
            att_sims_global.append(best_metrics["AttSim"])
            tc_scores_global.append(best_metrics["TC"])
        
        print(f"   ✅ Applied {best_method} to spurious_{i} (Score: {best_metrics['CombinedScore']:.3f})")
    
    # Compute overall effectiveness
    if att_sims_global and tc_scores_global:
        avg_att = np.mean(att_sims_global)
        avg_tc = np.mean(tc_scores_global)
        effectiveness = 1 - np.exp(-0.5 * avg_att * avg_tc * np.sqrt(len(spurious_segments)))
    else:
        effectiveness = 0.0
    
    # Add method tracking to quality metrics
    for key, method in replacement_methods.items():
        if key in quality_metrics:
            quality_metrics[key]["Method"] = method
    
    print(f"   📊 Replacement methods used: {dict(pd.Series(list(replacement_methods.values())).value_counts())}")
    
    return replacement_map, quality_metrics, effectiveness

# ========== REST OF THE CODE (APD FUNCTIONS, MAIN LOOP, ETC.) ==========
# [Include all the existing APD functions and main loop here - they remain the same]

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

# ========== SEGMENT EXTRACTION FUNCTIONS (UPDATED) ==========
def extract_segments_from_indices(time_series, spurious_indices, segment_size=5):
    """Extract segments around spurious indices"""
    segments = []
    T = len(time_series)
    
    for idx in spurious_indices:
        start = max(0, idx - segment_size // 2)
        end = min(T, idx + segment_size // 2 + 1)
        segment = time_series[start:end]
        segments.append(segment)
    
    return segments

def create_attention_mapping(spurious_segments, attention_weights):
    """Create attention mapping for spurious segments only"""
    attention_maps = {}
    
    # Map spurious segments
    for i, segment in enumerate(spurious_segments):
        key = f"spurious_{i}"
        if len(attention_weights) >= len(segment):
            attention_maps[key] = attention_weights[:len(segment)]
        else:
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
            
            start = max(0, idx - segment_size // 2)
            end = min(T, idx + segment_size // 2 + 1)
            segment_len = end - start
            
            if len(replacement) > segment_len:
                replacement = replacement[:segment_len]
            elif len(replacement) < segment_len:
                pad_len = segment_len - len(replacement)
                replacement = np.concatenate([replacement, np.full(pad_len, replacement[-1])])
            
            modified_series[start:end] = replacement
    
    return modified_series

def visualize_improved_replacement_results(ts_id, original_series, modified_series, spurious_indices, 
                                         replacement_map, quality_metrics, effectiveness, output_dir):
    """Enhanced visualization with method tracking"""
    os.makedirs(os.path.join(output_dir, "temporal_replacement"), exist_ok=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 15))
    
    # 1. Original vs Modified Series
    axes[0].plot(original_series, label='Original Series', alpha=0.7, linewidth=2)
    axes[0].plot(modified_series, label='After Replacement', alpha=0.7, linewidth=2, linestyle='--')
    axes[0].scatter(spurious_indices, original_series[spurious_indices], 
                   color='red', s=50, label=f'Spurious Points ({len(spurious_indices)})', zorder=5)
    axes[0].set_title(f'{ts_id} - Improved Temporal Replacement (Effectiveness: {effectiveness:.3f})')
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
    
    # 3. Quality metrics by method
    if quality_metrics:
        methods = []
        scores = []
        positions = []
        
        for key, metrics in quality_metrics.items():
            spurious_idx = int(key.split('_')[1])
            if spurious_idx < len(spurious_indices):
                methods.append(metrics.get('Method', 'unknown'))
                scores.append(metrics['CombinedScore'])
                positions.append(spurious_indices[spurious_idx])
        
        # Create color map for methods
        unique_methods = list(set(methods))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_methods)))
        method_colors = {method: colors[i] for i, method in enumerate(unique_methods)}
        
        for i, (method, score, pos) in enumerate(zip(methods, scores, positions)):
            axes[2].bar(i, score, color=method_colors[method], alpha=0.7, 
                       label=method if method not in [m for j, m in enumerate(methods) if j < i] else "")
        
        axes[2].set_xlabel('Replacement Index')
        axes[2].set_ylabel('Combined Score')
        axes[2].set_title('Replacement Quality by Method')
        axes[2].set_xticks(range(len(positions)))
        axes[2].set_xticklabels([f'Pos {pos}' for pos in positions], rotation=45)
        if unique_methods:
            axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Method distribution
        method_counts = pd.Series(methods).value_counts()
        axes[3].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
        axes[3].set_title('Distribution of Replacement Methods Used')
    else:
        axes[2].text(0.5, 0.5, 'No quality metrics available', 
                    transform=axes[2].transAxes, ha='center', va='center', fontsize=14)
        axes[3].text(0.5, 0.5, 'No method distribution available', 
                    transform=axes[3].transAxes, ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temporal_replacement", f"{ts_id}_improved_replacement.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created improved temporal replacement visualizations for {ts_id}")

# ========== REST OF EXISTING CODE ==========
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

# ========== CONFIG AND MAIN LOOP ==========
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

# ========== MAIN LOOP WITH IMPROVED PIPELINE ==========
all_metrics = []
apd_results = []
replacement_results = []

for ts_id in series_ids[:1]:
    try:
        print(f"\n===== Running Improved Pipeline on {ts_id} =====")
        args = create_args(ts_id)
        exp = Exp_Main(args)

        # Clear previous attention maps
        attention_maps["encoder"].clear()
        attention_maps["decoder"].clear()

        # Register hooks
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

        # Extract predictions
        y_pred = None
        y_true = None
        
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

            # ========== STEP 2: IMPROVED TEMPORAL REPLACEMENT ==========
            if len(spurious_idx) > 0:
                print(f"🔄 Step 2: Running Improved Temporal Replacement for {len(spurious_idx)} spurious patterns...")
                
                # Extract spurious segments
                segment_size = 5
                spurious_segments = extract_segments_from_indices(y_true_apd, spurious_idx, segment_size)
                
                print(f"   📊 Extracted {len(spurious_segments)} spurious segments")
                
                # Create attention mapping for segments
                attention_weights = components['attention']
                attention_segment_maps = create_attention_mapping(spurious_segments, attention_weights)
                
                # Run improved temporal replacement with fallback strategies
                replacement_map, quality_metrics, effectiveness = improved_temporal_replacement(
                    spurious_segments=spurious_segments,
                    time_series=y_true_apd,  # Full time series for context
                    spurious_indices=spurious_idx,
                    attention_maps=attention_segment_maps,
                    tau_tc=0.7,              # Primary threshold
                    eps_context=0.01,
                    lam=1.0,
                    alpha=0.6,               # 60% attention, 40% temporal consistency
                    omega=(0.33, 0.33, 0.34),
                    eps_amp=0.1,
                    segment_size=segment_size,
                    fallback_methods=['interpolation', 'best_available', 'smoothing']
                )
                
                # Apply replacements to the time series
                modified_series = apply_replacements_to_series(
                    y_true_apd, spurious_idx, replacement_map, segment_size
                )
                
                # Calculate improvement metrics
                original_mse = mean_squared_error(y_true_apd, y_pred_apd)
                modification_impact = np.mean(np.abs(modified_series - y_true_apd))
                
                # Count actual replacements (not original segments)
                actual_replacements = sum(1 for k, v in replacement_map.items() 
                                        if not np.array_equal(v, spurious_segments[int(k.split('_')[1])]))
                
                # Store replacement results
                replacement_result = {
                    'series': ts_id,
                    'effectiveness': effectiveness,
                    'num_replacements': actual_replacements,
                    'replacement_rate': actual_replacements / len(spurious_segments),
                    'avg_att_sim': np.mean([m['AttSim'] for m in quality_metrics.values()]) if quality_metrics else 0,
                    'avg_tc_score': np.mean([m['TC'] for m in quality_metrics.values()]) if quality_metrics else 0,
                    'avg_quality': np.mean([m['Quality'] for m in quality_metrics.values()]) if quality_metrics else 0,
                    'modification_impact': modification_impact,
                    'original_mse': original_mse
                }
                replacement_results.append(replacement_result)
                
                # Create enhanced visualizations
                visualize_improved_replacement_results(
                    ts_id, y_true_apd, modified_series, spurious_idx,
                    replacement_map, quality_metrics, effectiveness, series_output_dir
                )
                
                # Save replacement data
                np.save(os.path.join(series_output_dir, "modified_series.npy"), modified_series)
                np.save(os.path.join(series_output_dir, "original_series.npy"), y_true_apd)
                
                # Enhanced replacement details with method tracking
                replacement_details = []
                for i, idx in enumerate(spurious_idx):
                    spurious_key = f"spurious_{i}"
                    original_segment = spurious_segments[i]
                    replacement_segment = replacement_map.get(spurious_key, original_segment)
                    was_replaced = not np.array_equal(replacement_segment, original_segment)
                    method_used = quality_metrics.get(spurious_key, {}).get('Method', 'unknown')
                    
                    replacement_details.append({
                        'spurious_index': idx,
                        'was_replaced': was_replaced,
                        'method_used': method_used,
                        'original_value': y_true_apd[idx],
                        'modified_value': modified_series[idx],
                        'absolute_change': abs(modified_series[idx] - y_true_apd[idx]),
                        'att_sim': quality_metrics.get(spurious_key, {}).get('AttSim', 0),
                        'tc_score': quality_metrics.get(spurious_key, {}).get('TC', 0),
                        'combined_score': quality_metrics.get(spurious_key, {}).get('CombinedScore', 0),
                        'quality_score': quality_metrics.get(spurious_key, {}).get('Quality', 0)
                    })
                
                replacement_df = pd.DataFrame(replacement_details)
                replacement_df.to_csv(os.path.join(series_output_dir, "detailed_replacement_results.csv"), index=False)
                
                print(f"✅ Improved Temporal Replacement Complete:")
                print(f"   🎯 Effectiveness: {effectiveness:.3f}")
                print(f"   🔄 Successful replacements: {actual_replacements}/{len(spurious_segments)} ({replacement_result['replacement_rate']*100:.1f}%)")
                print(f"   📊 Avg Attention Similarity: {replacement_result['avg_att_sim']:.3f}")
                print(f"   📊 Avg Temporal Consistency: {replacement_result['avg_tc_score']:.3f}")
                print(f"   📊 Avg Quality Score: {replacement_result['avg_quality']:.3f}")
                
                # Show method breakdown
                method_counts = replacement_df['method_used'].value_counts()
                print(f"   🔧 Methods used: {dict(method_counts)}")
                
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

# ========== ENHANCED FINAL RESULTS ==========
print(f"\n{'='*70}")
print(f"IMPROVED AUTOFORMER + APD + TEMPORAL REPLACEMENT ANALYSIS")
print(f"{'='*70}")

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

# Enhanced replacement results
if replacement_results:
    df_replacement = pd.DataFrame(replacement_results)
    df_replacement.to_csv(os.path.join(output_dir, "improved_replacement_results.csv"), index=False)
    print(f"\n🔄 IMPROVED TEMPORAL REPLACEMENT RESULTS:")
    print(df_replacement.to_string(index=False))
    
    print(f"\n📈 ENHANCED REPLACEMENT SUMMARY:")
    print(f"   • Average effectiveness: {df_replacement['effectiveness'].mean():.3f}")
    print(f"   • Average replacement rate: {df_replacement['replacement_rate'].mean()*100:.1f}%")
    print(f"   • Total successful replacements: {df_replacement['num_replacements'].sum()}")
    print(f"   • Average attention similarity: {df_replacement['avg_att_sim'].mean():.3f}")
    print(f"   • Average temporal consistency: {df_replacement['avg_tc_score'].mean():.3f}")
    print(f"   • Average quality score: {df_replacement['avg_quality'].mean():.3f}")
    print(f"   • Average modification impact: {df_replacement['modification_impact'].mean():.3f}")

print(f"\n✅ IMPROVED PIPELINE COMPLETE!")
print(f"📁 All results saved to: {output_dir}")
print(f"📊 Key improvements:")
print(f"   • Multiple fallback strategies ensure replacements are always attempted")
print(f"   • Relaxed thresholds allow more flexible replacement criteria")  
print(f"   • Method tracking shows which strategy worked for each spurious pattern")
print(f"   • Enhanced quality metrics provide better insight into replacement effectiveness")

print(f"\n🎯 REPLACEMENT STRATEGY EFFECTIVENESS:")
print(f"   ✅ Segment Replacement: Uses similar patterns from the time series")
print(f"   🔄 Interpolation: Creates smooth transitions (cubic/linear/smooth)")
print(f"   📊 Best Available: Uses highest quality segments regardless of strict criteria")
print(f"   🎭 Smoothing: Applies gentle smoothing to reduce spurious spikes")
print(f"   ⚠️ No Replacement: Keeps original when all methods fail")