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
import sys
import shutil
import torch
import torch.nn.functional as F

# Add Autoformer to path
sys.path.insert(0, os.path.abspath('./Autoformer'))
from exp.exp_main import Exp_Main
from models.Autoformer import Model as Autoformer
import torch.nn as nn

# ========== APD SCORING FUNCTIONS ==========
def standardize(x):
    """Standardize array to zero mean and unit variance"""
    mu = np.mean(x)
    sigma = np.std(x) + 1e-6
    return (x - mu) / sigma

def compute_avg_attention(attention_map):
    """
    Compute average attention received per time step
    attention_map: np.ndarray of shape [H, T, T] (heads, time, time)
    Returns: np.array of shape [T] with average attention received per time step
    """
    if attention_map.ndim != 3:
        print(f"Warning: Expected 3D attention map, got shape {attention_map.shape}")
        if attention_map.ndim == 4:
            # Take mean over batch dimension
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 2:
            # Add head dimension
            attention_map = attention_map[np.newaxis, :, :]
    
    # Sum over query positions (axis=1), then average over heads (axis=0)
    return np.mean(np.sum(attention_map, axis=1), axis=0)

def compute_deviation(x, window=25):
    """
    Compute L2 deviation from local mean
    x: np.ndarray of shape [T, D] or [T]
    window: window size for local mean
    Returns: np.ndarray of shape [T] with L2 deviation from local mean
    """
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
    """
    Simplified marginal loss computation using prediction errors
    y_true: ground truth values
    y_pred: predicted values
    Returns: absolute prediction errors as proxy for marginal loss
    """
    return np.abs(y_true - y_pred)

def compute_apd_score(bar_A, L_t, Dev_t, beta1=1.0, beta2=1.0, beta3=1.0):
    """
    Compute APD (Attention Pattern Detection) score
    bar_A: average attention per position
    L_t: marginal loss per position
    Dev_t: deviation from local mean per position
    """
    A_std = standardize(bar_A)
    L_std = standardize(L_t)
    D_std = standardize(Dev_t)
    return beta1 * A_std + beta2 * L_std + beta3 * D_std

def detect_spurious(apd_scores, adaptive=True, multiplier=2.0):
    """
    Detect spurious/anomalous positions based on APD scores
    Returns indices of positions where APD exceeds threshold
    """
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
    """
    Complete pipeline to compute APD score and detect spurious patterns
    
    Args:
        attention_map: np.ndarray of shape [H, T, T] - attention weights
        y_true: ground truth time series values
        y_pred: predicted time series values
        beta1, beta2, beta3: weighting factors for attention, loss, and deviation
        window: window size for local deviation computation
    
    Returns:
        apd_scores: APD scores for each time step
        spurious_idx: indices of detected spurious positions
        threshold: detection threshold used
        components: dict with individual components (attention, loss, deviation)
    """
    # Compute individual components
    bar_A = compute_avg_attention(attention_map)
    Dev = compute_deviation(y_true, window=window)
    L = compute_marginal_loss_simple(y_true, y_pred)
    
    # Ensure all components have the same length
    min_len = min(len(bar_A), len(Dev), len(L))
    bar_A = bar_A[:min_len]
    Dev = Dev[:min_len]
    L = L[:min_len]
    
    # Compute APD scores
    apd_scores = compute_apd_score(bar_A, L, Dev, beta1, beta2, beta3)
    
    # Detect spurious patterns
    spurious_idx, threshold = detect_spurious(apd_scores)
    
    components = {
        'attention': bar_A,
        'loss': L,
        'deviation': Dev
    }
    
    return apd_scores, spurious_idx, threshold, components

def visualize_apd_analysis(ts_id, y_true, y_pred, apd_scores, spurious_idx, threshold, components, output_dir):
    """
    Create comprehensive visualization of APD analysis results
    """
    os.makedirs(os.path.join(output_dir, "apd_analysis"), exist_ok=True)
    
    # 1. Main APD visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Time series with spurious points highlighted
    axes[0].plot(y_true, label='Ground Truth', alpha=0.7, linewidth=2)
    axes[0].plot(y_pred, label='Prediction', alpha=0.7, linewidth=2, linestyle='--')
    if len(spurious_idx) > 0:
        axes[0].scatter(spurious_idx, y_true[spurious_idx], color='red', s=50, 
                       label=f'Spurious Points ({len(spurious_idx)})', zorder=5)
    axes[0].set_title(f'{ts_id} - Time Series with Spurious Pattern Detection')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Individual components
    axes[1].plot(components['attention'], label='Attention Score', color='blue')
    axes[1].set_title('Average Attention per Position')
    axes[1].set_ylabel('Attention')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    axes[2].plot(components['loss'], label='Prediction Error', color='orange')
    axes[2].set_title('Prediction Error per Position')
    axes[2].set_ylabel('Error')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # APD scores with threshold
    axes[3].plot(apd_scores, label='APD Score', color='purple', linewidth=2)
    axes[3].axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Threshold ({threshold:.2f})')
    if len(spurious_idx) > 0:
        axes[3].scatter(spurious_idx, apd_scores[spurious_idx], color='red', s=50, zorder=5)
    axes[3].set_title('APD Scores and Spurious Detection')
    axes[3].set_xlabel('Time Step')
    axes[3].set_ylabel('APD Score')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "apd_analysis", f"{ts_id}_apd_analysis.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Component correlation analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plots between components
    axes[0,0].scatter(components['attention'], components['loss'], alpha=0.6)
    axes[0,0].set_xlabel('Attention Score')
    axes[0,0].set_ylabel('Prediction Error')
    axes[0,0].set_title('Attention vs Prediction Error')
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].scatter(components['attention'], components['deviation'], alpha=0.6)
    axes[0,1].set_xlabel('Attention Score')
    axes[0,1].set_ylabel('Local Deviation')
    axes[0,1].set_title('Attention vs Local Deviation')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].scatter(components['loss'], components['deviation'], alpha=0.6)
    axes[1,0].set_xlabel('Prediction Error')
    axes[1,0].set_ylabel('Local Deviation')
    axes[1,0].set_title('Prediction Error vs Local Deviation')
    axes[1,0].grid(True, alpha=0.3)
    
    # APD score distribution
    axes[1,1].hist(apd_scores, bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(threshold, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('APD Score')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('APD Score Distribution')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "apd_analysis", f"{ts_id}_component_analysis.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created APD analysis visualizations for {ts_id}")

# ========== CONFIG ==========
root_path = './Autoformer/dataset/m5_series_split'
output_dir = './autoformer_m5_outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== GETTING THE SERIES ==========
series_ids = [f"FOODS_1_0{str(i).zfill(2)}_CA_1" for i in range(1, 2)]

# ========== ATTENTION CAPTURE (EXISTING CODE) ==========
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

# ========== STRUCT FOR ARGS ==========
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

# ========== MAIN LOOP WITH APD INTEGRATION ==========
all_metrics = []
apd_results = []

for ts_id in series_ids[:1]:  # Process first series as example
    try:
        print(f"\n===== Running Autoformer with APD Analysis on {ts_id} =====")
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

        # Extract predictions (your existing prediction extraction code)
        possible_result_paths = [
            f'./results/{setting}/',
            f'./results/',
            f'./checkpoints/{setting}/',
            './outputs/',
            './exp_results/'
        ]
        
        y_pred = None
        y_true = None
        
        # Try to find predictions from files first
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

        # If still no predictions, extract from data loader
        if y_pred is None:
            print("🔍 Extracting predictions from data loader...")
            test_data, test_loader = exp._get_data(flag='test')
            exp.model.eval()
            preds, trues = [], []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                    batch_x = batch_x.float()
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float()
                    batch_y_mark = batch_y_mark.float()
                    
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
                    
                    outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    if isinstance(outputs, tuple):
                        pred = outputs[0].detach().cpu().numpy()
                    else:
                        pred = outputs.detach().cpu().numpy()
                    
                    true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
            
            if preds and trues:
                y_pred = np.concatenate(preds, axis=0).reshape(-1)
                y_true = np.concatenate(trues, axis=0).reshape(-1)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if y_pred is None or y_true is None:
            print(f"❌ Could not extract predictions for {ts_id}")
            continue

        # Calculate standard metrics
        metrics = evaluate_metrics(y_true, y_pred)
        metrics["series"] = ts_id
        all_metrics.append(metrics)

        # ========== APD ANALYSIS ==========
        print(f"🔍 Running APD analysis for {ts_id}...")
        
        # Get attention maps - use encoder attention (combine all layers)
        combined_attention = None
        if attention_maps["encoder"]:
            all_attentions = []
            for layer_idx, attn_list in attention_maps["encoder"].items():
                if attn_list:
                    # Average across batches for this layer
                    layer_avg = np.mean(np.stack(attn_list, axis=0), axis=0)  # [H, T, T]
                    all_attentions.append(layer_avg)
            
            if all_attentions:
                # Average across layers: [L, H, T, T] -> [H, T, T]
                combined_attention = np.mean(np.stack(all_attentions, axis=0), axis=0)
                print(f"✅ Combined attention shape: {combined_attention.shape}")
        
        # Run APD analysis if we have attention maps
        if combined_attention is not None:
            try:
                # Ensure we have the right dimensions for time series length
                pred_len = min(len(y_true), len(y_pred), combined_attention.shape[-1])
                y_true_apd = y_true[:pred_len]
                y_pred_apd = y_pred[:pred_len]
                
                # Run APD pipeline
                apd_scores, spurious_idx, threshold, components = run_apd_pipeline(
                    attention_map=combined_attention,
                    y_true=y_true_apd,
                    y_pred=y_pred_apd,
                    beta1=1.0,  # attention weight
                    beta2=1.0,  # prediction error weight  
                    beta3=1.0,  # deviation weight
                    window=25
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
                
                # Create visualizations
                visualize_apd_analysis(ts_id, y_true_apd, y_pred_apd, apd_scores, 
                                     spurious_idx, threshold, components, series_output_dir)
                
                # Save APD data
                np.save(os.path.join(series_output_dir, "apd_scores.npy"), apd_scores)
                np.save(os.path.join(series_output_dir, "spurious_indices.npy"), spurious_idx)
                
                # Save detailed results
                apd_df = pd.DataFrame({
                    'time_step': range(len(apd_scores)),
                    'apd_score': apd_scores,
                    'attention': components['attention'],
                    'prediction_error': components['loss'],
                    'local_deviation': components['deviation'],
                    'is_spurious': [i in spurious_idx for i in range(len(apd_scores))]
                })
                apd_df.to_csv(os.path.join(series_output_dir, "apd_detailed_results.csv"), index=False)
                
                print(f"✅ APD Analysis Complete for {ts_id}:")
                print(f"   🎯 Detected {len(spurious_idx)} spurious patterns ({len(spurious_idx)/len(apd_scores)*100:.1f}%)")
                print(f"   📊 Average APD Score: {np.mean(apd_scores):.3f}")
                print(f"   🚨 Threshold: {threshold:.3f}")
                
                if len(spurious_idx) > 0:
                    print(f"   🔍 Spurious indices: {spurious_idx[:10]}{'...' if len(spurious_idx) > 10 else ''}")
                
            except Exception as apd_error:
                print(f"❌ Error in APD analysis for {ts_id}: {apd_error}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ No attention maps available for APD analysis on {ts_id}")

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

# ========== FINAL RESULTS ==========
print(f"\n{'='*50}")
print(f"AUTOFORMER + APD ANALYSIS COMPLETE")
print(f"{'='*50}")

# Standard metrics
if all_metrics:
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(os.path.join(output_dir, "autoformer_evaluation_summary.csv"), index=False)
    print(f"\n📊 STANDARD METRICS:")
    print(df_metrics.to_string(index=False))

# APD results
if apd_results:
    df_apd = pd.DataFrame(apd_results)
    df_apd.to_csv(os.path.join(output_dir, "apd_analysis_summary.csv"), index=False)
    print(f"\n🎯 APD ANALYSIS RESULTS:")
    print(df_apd.to_string(index=False))
    
    print(f"\n📈 APD SUMMARY STATISTICS:")
    print(f"   • Average spurious patterns detected: {df_apd['num_spurious'].mean():.1f}")
    print(f"   • Average spurious ratio: {df_apd['spurious_ratio'].mean()*100:.1f}%")
    print(f"   • Average APD score: {df_apd['avg_apd_score'].mean():.3f}")

print(f"\n✅ All results saved to: {output_dir}")
print(f"🔍 Check the 'apd_analysis' subdirectories for detailed visualizations!")