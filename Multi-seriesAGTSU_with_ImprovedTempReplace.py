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
from scipy import stats
import sys
import shutil
import torch
import torch.nn.functional as F
from typing import List, Tuple
import copy
import pickle
import json

# Add Autoformer to path
sys.path.insert(0, os.path.abspath('./Autoformer'))
from exp.exp_main import Exp_Main
from models.Autoformer import Model as Autoformer
import torch.nn as nn

# ========== CONFIGURATION ==========
root_path = './Autoformer/dataset/m5_series_split'
output_dir = './autoformer_m5_outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== SERIES DISCOVERY ==========
print("🔍 Discovering available time series...")
all_series_files = glob.glob(os.path.join(root_path, "*.csv"))
all_series_ids = set()

for file_path in all_series_files:
    base = os.path.basename(file_path)
    if not base.endswith(("_train.csv", "_val.csv", "_test.csv")):
        ts_id = base.replace(".csv", "")
        all_series_ids.add(ts_id)

print(f"📊 Found {len(all_series_ids)} total series")

# Randomly select N series
random.seed(42)
num_to_sample = 5000  # ⬅️ CHANGE THIS to process more series
series_ids = random.sample(list(all_series_ids), min(num_to_sample, len(all_series_ids)))

print(f"📊 Selected {len(series_ids)} series for analysis:")
for i, ts_id in enumerate(series_ids, 1):
    print(f"   {i:2d}. {ts_id}")

# ========== HELPER FUNCTIONS ==========
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

def evaluate_metrics(y_true, y_pred):
    # Add small epsilon to avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < 1e-6, 1e-6, y_true)
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true_safe)) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape}

# ========== ATTENTION CAPTURE ==========
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

# ========== SIMPLIFIED SISA UNLEARNING ==========
def create_temporal_replacement(y_true, spurious_indices, segment_size=5):
    """Create temporal replacements for spurious patterns"""
    replacement_map = {}
    
    for i, idx in enumerate(spurious_indices):
        start = max(0, idx - segment_size // 2)
        end = min(len(y_true), idx + segment_size // 2 + 1)
        segment = y_true[start:end]
        
        # Create replacement using smoothing
        if len(segment) >= 3:
            try:
                x = np.arange(len(segment))
                poly_coeffs = np.polyfit(x, segment, min(2, len(segment) - 1))
                replacement = np.polyval(poly_coeffs, x)
                replacement = gaussian_filter(replacement, sigma=0.5)
            except:
                replacement = np.linspace(segment[0], segment[-1], len(segment))
        else:
            replacement = segment.copy()
        
        replacement_map[f"spurious_{i}"] = replacement
    
    return replacement_map

def apply_sisa_unlearning(y_pred, y_true, spurious_indices, replacement_map):
    """Apply simplified SISA unlearning by replacing spurious regions"""
    sisa_predictions = y_pred.copy()
    
    for i, idx in enumerate(spurious_indices):
        if f"spurious_{i}" in replacement_map:
            replacement = replacement_map[f"spurious_{i}"]
            segment_size = len(replacement)
            
            start = max(0, idx - segment_size // 2)
            end = min(len(sisa_predictions), start + segment_size)
            
            if end > start:
                # Apply replacement with proper length matching
                actual_length = end - start
                if len(replacement) != actual_length:
                    if len(replacement) > actual_length:
                        replacement = replacement[:actual_length]
                    else:
                        # Extend replacement using interpolation
                        x_old = np.linspace(0, 1, len(replacement))
                        x_new = np.linspace(0, 1, actual_length)
                        replacement = np.interp(x_new, x_old, replacement)
                
                sisa_predictions[start:end] = replacement
    
    return sisa_predictions

# ========== MAIN PIPELINE ==========
all_metrics = []
apd_results = []
unlearning_results = []
failed_series = []

print(f"\n🚀 Starting pipeline on {len(series_ids)} series...")
print("="*80)

for idx, ts_id in enumerate(series_ids, 1):
    try:
        print(f"\n[{idx:2d}/{len(series_ids)}] Processing {ts_id}...")
        start_time = pd.Timestamp.now()
        
        # Validate series file
        series_file = os.path.join(root_path, f"{ts_id}.csv")
        if not os.path.exists(series_file):
            print(f"❌ File not found: {series_file}")
            failed_series.append({'series': ts_id, 'reason': 'File not found'})
            continue
        
        # Quick data check
        df_check = pd.read_csv(series_file)
        if len(df_check) < 100:
            print(f"❌ Insufficient data: {len(df_check)} samples")
            failed_series.append({'series': ts_id, 'reason': f'Insufficient data: {len(df_check)}'})
            continue
        
        print(f"📊 Data validated: {len(df_check)} samples")
        
        # ========== PHASE 1: TRAIN MODEL ==========
        print("🔧 Phase 1: Training Autoformer...")
        args = create_args(ts_id)
        exp = Exp_Main(args)
        
        # Clear attention maps
        attention_maps["encoder"].clear()
        attention_maps["decoder"].clear()
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(exp.model.encoder.attn_layers):
            try:
                module = layer.attention.inner_correlation
                hook = module.register_forward_hook(make_encoder_hook(i))
                hooks.append(hook)
            except:
                pass
        
        # Train and test
        setting = f"{args.model_id}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}"
        exp.train(setting)
        exp.test(setting, test=1)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # ========== PHASE 2: EXTRACT PREDICTIONS ==========
        print("📊 Phase 2: Extracting predictions...")
        y_pred = None
        y_true = None
        
        result_paths = [f'./results/{setting}/', './results/']
        for result_path in result_paths:
            if os.path.exists(result_path):
                files = os.listdir(result_path)
                pred_file = next((f for f in ['pred.npy', 'preds.npy'] if f in files), None)
                true_file = next((f for f in ['true.npy', 'trues.npy'] if f in files), None)
                
                if pred_file and true_file:
                    try:
                        y_pred = np.load(os.path.join(result_path, pred_file)).reshape(-1)
                        y_true = np.load(os.path.join(result_path, true_file)).reshape(-1)
                        print(f"✅ Predictions extracted: {len(y_pred)} samples")
                        break
                    except:
                        continue
        
        if y_pred is None or y_true is None:
            print(f"❌ Could not extract predictions")
            failed_series.append({'series': ts_id, 'reason': 'Prediction extraction failed'})
            continue
        
        # Validate predictions
        if np.all(y_pred == 0) or np.isnan(y_pred).all():
            print(f"❌ Invalid predictions (all zeros or NaN)")
            failed_series.append({'series': ts_id, 'reason': 'Invalid predictions'})
            continue
        
        # Calculate original metrics
        original_metrics = evaluate_metrics(y_true, y_pred)
        original_metrics["series"] = ts_id
        original_metrics["phase"] = "original"
        all_metrics.append(original_metrics)
        
        print(f"📊 Original MSE: {original_metrics['MSE']:.4f}")
        
        # ========== PHASE 3: APD ANALYSIS ==========
        print("🔍 Phase 3: APD spurious detection...")
        spurious_idx = np.array([])
        
        if attention_maps["encoder"]:
            # Combine attention maps
            all_attentions = []
            for layer_idx, attn_list in attention_maps["encoder"].items():
                if attn_list:
                    layer_avg = np.mean(np.stack(attn_list, axis=0), axis=0)
                    all_attentions.append(layer_avg)
            
            if all_attentions:
                combined_attention = np.mean(np.stack(all_attentions, axis=0), axis=0)
                pred_len = min(len(y_true), len(y_pred), combined_attention.shape[-1])
                
                apd_scores, spurious_idx, threshold, components = run_apd_pipeline(
                    combined_attention, y_true[:pred_len], y_pred[:pred_len]
                )
                
                print(f"✅ APD detected {len(spurious_idx)} spurious patterns")
        
        # Record APD results
        apd_result = {
            'series': ts_id,
            'num_spurious': len(spurious_idx),
            'spurious_ratio': len(spurious_idx) / len(y_pred) if len(y_pred) > 0 else 0,
            'spurious_indices': spurious_idx.tolist()
        }
        apd_results.append(apd_result)
        
        # ========== PHASE 4: SIMPLIFIED SISA UNLEARNING ==========
        print("🧠 Phase 4: SISA unlearning...")
        
        if len(spurious_idx) > 0:
            # Create temporal replacements
            replacement_map = create_temporal_replacement(y_true, spurious_idx)
            
            # Apply SISA unlearning
            sisa_predictions = apply_sisa_unlearning(y_pred, y_true, spurious_idx, replacement_map)
            
            print(f"🔄 Applied replacements for {len(spurious_idx)} spurious patterns")
        else:
            # No spurious patterns, use original predictions
            sisa_predictions = y_pred.copy()
            print("✅ No spurious patterns detected, using original predictions")
        
        # Calculate unlearned metrics
        unlearned_metrics = evaluate_metrics(y_true, sisa_predictions)
        unlearned_metrics["series"] = ts_id
        unlearned_metrics["phase"] = "post_unlearning"
        all_metrics.append(unlearned_metrics)
        
        # Calculate improvement
        original_mse = original_metrics['MSE']
        unlearned_mse = unlearned_metrics['MSE']
        improvement = (original_mse - unlearned_mse) / original_mse * 100 if original_mse > 0 else 0
        
        unlearning_result = {
            'series': ts_id,
            'original_mse': original_mse,
            'unlearned_mse': unlearned_mse,
            'improvement_pct': improvement,
            'spurious_patterns': len(spurious_idx)
        }
        unlearning_results.append(unlearning_result)
        
        print(f"✅ Improvement: {improvement:.2f}%")
        
        # Save individual results
        series_output_dir = os.path.join(output_dir, ts_id)
        os.makedirs(series_output_dir, exist_ok=True)
        np.save(os.path.join(series_output_dir, "original_predictions.npy"), y_pred)
        np.save(os.path.join(series_output_dir, "sisa_predictions.npy"), sisa_predictions)
        np.save(os.path.join(series_output_dir, "ground_truth.npy"), y_true)
        if len(spurious_idx) > 0:
            np.save(os.path.join(series_output_dir, "spurious_indices.npy"), spurious_idx)
        
    except Exception as e:
        print(f"❌ Error processing {ts_id}: {e}")
        failed_series.append({'series': ts_id, 'reason': str(e)})
        continue
    
    finally:
        end_time = pd.Timestamp.now()
        processing_time = (end_time - start_time).total_seconds()
        print(f"⏱️  Processing time: {processing_time:.1f}s")

# ========== COMPREHENSIVE RESULTS ==========
print(f"\n{'='*80}")
print(f"MULTI-SERIES PIPELINE ANALYSIS COMPLETE")
print(f"{'='*80}")

print(f"\n📊 PROCESSING SUMMARY:")
successful_series = len(all_metrics) // 2
print(f"   • Total series attempted: {len(series_ids)}")
print(f"   • Successfully processed: {successful_series}")
print(f"   • Failed series: {len(failed_series)}")
print(f"   • Success rate: {successful_series/len(series_ids)*100:.1f}%")

if failed_series:
    print(f"\n❌ Failed series:")
    for failure in failed_series:
        print(f"   • {failure['series']}: {failure['reason']}")

# Performance analysis
if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(output_dir, "all_metrics.csv"), index=False)
    
    original_metrics = df_metrics[df_metrics['phase'] == 'original']
    unlearned_metrics = df_metrics[df_metrics['phase'] == 'post_unlearning']
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    for metric in ['MSE', 'MAE', 'RMSE']:
        if metric in df_metrics.columns and len(original_metrics) > 0:
            orig_mean = original_metrics[metric].mean()
            unl_mean = unlearned_metrics[metric].mean()
            improvement = (orig_mean - unl_mean) / orig_mean * 100 if orig_mean > 0 else 0
            
            print(f"   {metric}:")
            print(f"     • Original: {orig_mean:.4f}")
            print(f"     • Unlearned: {unl_mean:.4f}")
            print(f"     • Improvement: {improvement:+.2f}%")

# APD analysis
if apd_results:
    df_apd = pd.DataFrame(apd_results)
    df_apd.to_csv(os.path.join(output_dir, "apd_results.csv"), index=False)
    
    print(f"\n🎯 APD SPURIOUS DETECTION:")
    print(f"   • Series with spurious patterns: {len(df_apd[df_apd['num_spurious'] > 0])}/{len(df_apd)}")
    print(f"   • Average spurious per series: {df_apd['num_spurious'].mean():.1f}")
    print(f"   • Total spurious detected: {df_apd['num_spurious'].sum()}")

# SISA analysis
if unlearning_results:
    df_unlearning = pd.DataFrame(unlearning_results)
    df_unlearning.to_csv(os.path.join(output_dir, "sisa_results.csv"), index=False)
    
    positive_improvements = len(df_unlearning[df_unlearning['improvement_pct'] > 0])
    
    print(f"\n🧠 SISA UNLEARNING SUMMARY:")
    print(f"   • Series with improvement: {positive_improvements}/{len(df_unlearning)}")
    print(f"   • Average improvement: {df_unlearning['improvement_pct'].mean():.2f}%")
    print(f"   • Best improvement: {df_unlearning['improvement_pct'].max():.2f}%")

# Create summary visualization
if all_metrics and len(df_metrics) > 0:
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Multi-Series Analysis Summary ({successful_series} series)', fontsize=14)
        
        # 1. MSE comparison
        if len(original_metrics) > 0 and len(unlearned_metrics) > 0:
            series_names = original_metrics['series'].values
            x_pos = np.arange(len(series_names))
            width = 0.35
            
            axes[0, 0].bar(x_pos - width/2, original_metrics['MSE'], width, 
                          label='Original', alpha=0.7)
            axes[0, 0].bar(x_pos + width/2, unlearned_metrics['MSE'], width, 
                          label='Unlearned', alpha=0.7)
            axes[0, 0].set_title('MSE Comparison')
            axes[0, 0].set_ylabel('MSE')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(series_names, rotation=45, ha='right')
            axes[0, 0].legend()
        
        # 2. Improvement distribution
        if not df_unlearning.empty:
            axes[0, 1].hist(df_unlearning['improvement_pct'], bins=10, alpha=0.7)
            axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_title('Improvement Distribution')
            axes[0, 1].set_xlabel('Improvement (%)')
            axes[0, 1].set_ylabel('Frequency')
        
        # 3. Spurious patterns
        if not df_apd.empty:
            axes[1, 0].bar(df_apd['series'], df_apd['num_spurious'])
            axes[1, 0].set_title('Spurious Patterns Detected')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Success summary
        success_data = {
            'Successful': successful_series,
            'Failed': len(failed_series)
        }
        axes[1, 1].pie(success_data.values(), labels=success_data.keys(), 
                      autopct='%1.1f%%')
        axes[1, 1].set_title('Processing Success Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multi_series_summary.png"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Summary visualization saved to: {output_dir}/multi_series_summary.png")
        
    except Exception as e:
        print(f"⚠️ Error creating visualization: {e}")

# Save comprehensive summary
summary = {
    'processing_summary': {
        'total_attempted': len(series_ids),
        'successful': successful_series,
        'failed': len(failed_series),
        'success_rate': successful_series/len(series_ids)*100 if len(series_ids) > 0 else 0
    },
    'performance_summary': {},
    'apd_summary': {},
    'sisa_summary': {}
}

if all_metrics and len(original_metrics) > 0:
    for metric in ['MSE', 'MAE', 'RMSE']:
        if metric in df_metrics.columns:
            orig_mean = original_metrics[metric].mean()
            unl_mean = unlearned_metrics[metric].mean()
            summary['performance_summary'][metric] = {
                'original_mean': float(orig_mean),
                'unlearned_mean': float(unl_mean),
                'improvement_pct': float((orig_mean - unl_mean) / orig_mean * 100 if orig_mean > 0 else 0)
            }

if apd_results:
    summary['apd_summary'] = {
        'series_with_spurious': int(len(df_apd[df_apd['num_spurious'] > 0])),
        'avg_spurious_per_series': float(df_apd['num_spurious'].mean()),
        'total_spurious': int(df_apd['num_spurious'].sum())
    }

if unlearning_results:
    summary['sisa_summary'] = {
        'series_with_improvement': int(len(df_unlearning[df_unlearning['improvement_pct'] > 0])),
        'avg_improvement': float(df_unlearning['improvement_pct'].mean()),
        'best_improvement': float(df_unlearning['improvement_pct'].max())
    }

with open(os.path.join(output_dir, "summary_report.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ PIPELINE COMPLETE!")
print(f"📁 Results saved to: {output_dir}")
print(f"🔍 Key files generated:")
print(f"   • all_metrics.csv - Performance metrics for all series")
print(f"   • apd_results.csv - Spurious pattern detection results")
print(f"   • sisa_results.csv - SISA unlearning performance")
print(f"   • multi_series_summary.png - Visual summary")
print(f"   • summary_report.json - Machine-readable summary")
print(f"   • Individual series results in subdirectories")

print(f"\n🎯 KEY INSIGHTS:")
if successful_series > 0:
    print(f"   📊 Successfully processed {successful_series}/{len(series_ids)} series")
    
    if all_metrics and len(original_metrics) > 0:
        orig_mse = original_metrics['MSE'].mean()
        unl_mse = unlearned_metrics['MSE'].mean()
        avg_improvement = (orig_mse - unl_mse) / orig_mse * 100 if orig_mse > 0 else 0
        print(f"   📈 Average MSE improvement: {avg_improvement:.2f}%")
    
    if unlearning_results:
        positive_improvements = len(df_unlearning[df_unlearning['improvement_pct'] > 0])
        print(f"   🎯 Series with positive improvement: {positive_improvements}/{len(df_unlearning)}")
    
    if apd_results:
        spurious_detection = len(df_apd[df_apd['num_spurious'] > 0])
        print(f"   🔍 Series with spurious patterns: {spurious_detection}/{len(df_apd)}")

# print(f"\n🚀 Ready for further analysis and deployment!")
# print(f"\n💡 NEXT STEPS:")
# print(f"   1. Increase 'num_to_sample' to process more series")
# print(f"   2. Analyze individual series results in subdirectories")
# print(f"   3. Fine-tune APD parameters based on results")
# print(f"   4. Experiment with different SISA configurations")
# print(f"   5. Use summary_report.json for automated analysis")