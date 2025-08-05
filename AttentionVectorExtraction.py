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

# ========== CONFIG ==========
root_path = './Autoformer/dataset/m5_series_split'
output_dir = './autoformer_m5_outputs'
os.makedirs(output_dir, exist_ok=True)

# ========== GETTING THE SERIES ==========

# series_ids = [f"FOODS_1_0{str(i).zfill(2)}_CA_1" for i in range(1, 2)]\

#RANDOM taking a series from the M5 dataset
# Step 1: Find all full time series (not *_train/val/test)
all_series_files = glob.glob(os.path.join(root_path, "*.csv"))
all_series_ids = set()

for file_path in all_series_files:
    base = os.path.basename(file_path)
    if not base.endswith(("_train.csv", "_val.csv", "_test.csv")):
        ts_id = base.replace(".csv", "")
        all_series_ids.add(ts_id)

# # Step 2: Randomly select N series
random.seed(42)
num_to_sample = 10000  # ⬅️ You can change this
series_ids = random.sample(list(all_series_ids), min(num_to_sample, len(all_series_ids)))

print(f"📊 Randomly selected {len(series_ids)} series:\n{series_ids}")

# ========== Ploting the series in whole ==========

def visualize_full_series(series_ids, root_path, output_dir, max_plots=5):
    os.makedirs(os.path.join(output_dir, "timeseries_visuals"), exist_ok=True)
    
    for i, ts_id in enumerate(series_ids):
        if i >= max_plots:
            break

        full_path = os.path.join(root_path, f"{ts_id}.csv")
        
        if not os.path.exists(full_path):
            print(f"⚠️ Missing: {full_path}")
            continue
        
        df = pd.read_csv(full_path)
        
        if 'value' not in df.columns:
            print(f"⚠️ 'value' column missing in {ts_id}")
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(df['value'], label='Value', linewidth=2)
        plt.title(f"Original Time Series: {ts_id}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "timeseries_visuals", f"{ts_id}_raw.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"📈 Saved plot for {ts_id} → {plot_path}")

visualize_full_series(series_ids, root_path, output_dir)

# ========== LAYER-WISE ATTENTION VISUALIZATION FUNCTIONS ==========

def visualize_encoder_layers(attention_dict, ts_id, output_dir):
    """Create separate PNG files for each encoder layer"""
    encoder_dir = os.path.join(output_dir, "encoder_layers")
    os.makedirs(encoder_dir, exist_ok=True)
    
    for layer_idx, attn_list in attention_dict.items():
        if not attn_list:
            print(f"⚠️ No attention data for encoder layer {layer_idx}")
            continue
        
        try:

            # Ensure all attention arrays have the same shape
            base_shape = attn_list[0].shape
            attn_list = [a for a in attn_list if a.shape == base_shape]

            if not attn_list:
                print(f"⚠️ Skipping visualization for encoder layer {layer_idx} due to inconsistent shapes.")
                continue

            # Stack and average over batches
            attn_stack = np.stack(attn_list, axis=0)  # [batch, heads, seq, seq]
            avg_attn = np.mean(attn_stack, axis=0)    # [heads, seq, seq]
            
            num_heads = avg_attn.shape[0]
            seq_len = avg_attn.shape[1]
            
            print(f"📊 Encoder Layer {layer_idx}: {num_heads} heads, {seq_len}x{seq_len} attention matrix")
            
            # Create subplot grid for all heads in this layer
            num_cols = 4
            num_rows = int(np.ceil(num_heads / num_cols))
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
            if num_heads == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for h in range(num_heads):
                sns.heatmap(avg_attn[h], ax=axes[h], cmap="viridis", cbar=True,
                           square=True, xticklabels=False, yticklabels=False)
                axes[h].set_title(f"Head {h+1}", fontsize=12)
                axes[h].set_xlabel("Key Position")
                axes[h].set_ylabel("Query Position")
            
            # Hide unused subplots
            for h in range(num_heads, len(axes)):
                axes[h].axis('off')
            
            plt.suptitle(f"{ts_id} - Encoder Layer {layer_idx} ({num_heads} heads, {seq_len}x{seq_len})", fontsize=16)
            plt.tight_layout()
            
            # Save with layer-specific filename
            fig_path = os.path.join(encoder_dir, f"encoder_layer_{layer_idx}_heads.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Save mean attention for this layer
            create_encoder_mean_attention(avg_attn, layer_idx, ts_id, encoder_dir)  
            
            print(f"✅ Saved encoder layer {layer_idx} visualization → {fig_path}")
            
            # Also save the attention data for this layer
            layer_attn_path = os.path.join(encoder_dir, f"encoder_layer_{layer_idx}_attn.npy")
            np.save(layer_attn_path, avg_attn)
            
        except Exception as e:
            print(f"❌ Error visualizing encoder layer {layer_idx}: {e}")
            continue

# ========== IMPROVED ATTENTION HOOKS ==========
attention_maps = {"encoder": {}, "decoder": []}

def make_encoder_hook(layer_idx):
    def hook_encoder_attention(module, input, output):
        try:
            if hasattr(module, 'last_attn'):
                attn = module.last_attn  # shape: (B, H, L, L) or (B, H, L, K)
                if isinstance(attn, torch.Tensor):
                    attn_np = attn.detach().cpu().numpy()

                    if attn_np.ndim == 4:
                        # (B, H, L, L) → self-attention
                        avg_attn = np.mean(attn_np, axis=0)  # avg over batch → (H, L, L)
                        # print(f"[Hook ✅] Layer {layer_idx}: Attention shape {avg_attn.shape}")

                        if layer_idx not in attention_maps["encoder"]:
                            attention_maps["encoder"][layer_idx] = []
                        attention_maps["encoder"][layer_idx].append(avg_attn)
                    else:
                        print(f"[Hook ⚠️] Layer {layer_idx} has non-square attention: {attn_np.shape}")
        except Exception as e:
            print(f"[Hook ❌] Failed at layer {layer_idx}: {e}")
            import traceback; traceback.print_exc()
    return hook_encoder_attention

def hook_decoder_attention(module, input, output):
    """
    Hook for capturing decoder cross-attention weights
    """
    try:
        # print(f"[Hook] Decoder module type: {type(module)}")
        
        attn = None
        
        # Method 1: Cross attention
        if hasattr(module, 'cross_attention'):
            if hasattr(module.cross_attention, 'inner_correlation'):
                if hasattr(module.cross_attention.inner_correlation, 'last_attn'):
                    attn = module.cross_attention.inner_correlation.last_attn
                elif hasattr(module.cross_attention.inner_correlation, 'attn_weights'):
                    attn = module.cross_attention.inner_correlation.attn_weights
            elif hasattr(module.cross_attention, 'attn_weights'):
                attn = module.cross_attention.attn_weights
        
        # Method 2: Self attention (fallback)
        elif hasattr(module, 'self_attention'):
            if hasattr(module.self_attention, 'inner_correlation'):
                if hasattr(module.self_attention.inner_correlation, 'last_attn'):
                    attn = module.self_attention.inner_correlation.last_attn
        
        # Method 3: Direct attention attribute
        elif hasattr(module, 'attention'):
            if hasattr(module.attention, 'inner_correlation'):
                if hasattr(module.attention.inner_correlation, 'last_attn'):
                    attn = module.attention.inner_correlation.last_attn
        
        if attn is not None:
            if isinstance(attn, torch.Tensor):
                attn_np = attn.detach().cpu().numpy()
                # print(f"[Hook ✅] Captured decoder attention: shape {attn_np.shape}")
                attention_maps["decoder"].append(attn_np)
            else:
                print(f"[Hook ⚠️] Attention is not a tensor: {type(attn)}")
        else:
            print(f"[Hook ⚠️] Could not find attention weights in decoder")
            print(f"[Hook] Available attributes: {[attr for attr in dir(module) if not attr.startswith('_')]}")
            
    except Exception as e:
        print(f"[Hook ⚠️] Decoder attention hook failed: {e}")
        import traceback
        traceback.print_exc()

# ========== MANUAL ATTENTION EXTRACTION ==========
def extract_attention_manually(model, data_loader, args):
    """
    Manually extract attention weights by running forward pass
    """
    model.eval()
    encoder_attns = []
    decoder_attns = []
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if i >= 5:  # Only process first 5 batches for attention
                break
                
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
            
            # Forward pass with attention extraction
            try:
                # Set model to return attention
                if hasattr(model, 'output_attention'):
                    model.output_attention = True
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Check if outputs contain attention weights
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    pred, attns = outputs[0], outputs[1:]
                    if attns:
                        encoder_attns.extend(attns[0] if attns[0] is not None else [])
                        if len(attns) > 1:
                            decoder_attns.extend(attns[1] if attns[1] is not None else [])
                
            except Exception as e:
                print(f"Manual attention extraction failed: {e}")
                break
    
    return encoder_attns, decoder_attns

# ========== STRUCT FOR ARGS FOR OBSERVING TIME-SERIES==========
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def create_args(ts_id, window=720, horizon=28):
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
        activation='gelu', output_attention=True,  # Enable attention output
        do_predict=True,
        moving_avg=25, use_gpu=False, gpu=0, use_multi_gpu=False,
        target='value', root_path=root_path,
        data_path=f'{ts_id}.csv',
        batch_size=32, learning_rate=0.0001, train_epochs=10,
        num_workers=0, des='Exp', itr=1,
        patience=3, lradj='type1', inverse=False,
        mix=True, cols=None,
        checkpoints='./checkpoints/', detail_freq='h',
        is_training=1, model_id=f'Autoformer_{ts_id}', loss='mse',
        use_amp=False, seasonal_patterns='Monthly'
    )

# ========== METRIC FUNCTION ==========
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse ** 0.5
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.maximum(y_true, 1)) * 100
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'WAPE': wape}

# ========== ATTENTION VISUALIZATION FUNCTIONS ==========
def create_attention_heatmaps(attn_path, ts_id, layer_type, output_dir):
    """
    Create comprehensive attention heatmaps
    """
    try:
        attn_data = np.load(attn_path)
        print(f"📊 Creating {layer_type} attention heatmaps for {ts_id}, shape: {attn_data.shape}")
        
        # Handle different tensor shapes
        if attn_data.ndim == 4:
            # [batch, heads, seq, seq] - take first batch
            attn_heads = attn_data[0]
        elif attn_data.ndim == 3:
            # [heads, seq, seq]
            attn_heads = attn_data
        else:
            print(f"⚠️ Unexpected attention shape: {attn_data.shape}")
            return
        
        num_heads = attn_heads.shape[0]
        
        # 1. Individual head plots ()
        num_cols = 4
        num_rows = math.ceil(num_heads / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        

        for h in range(num_heads):
            vmin, vmax = np.percentile(attn_heads[h], [1, 99])  # Avoids extreme outliers
            
            sns.heatmap(attn_heads[h], ax=axes[h], cmap="viridis", cbar=True, 
                       square=True, xticklabels=range(0, attn_heads[h].shape[-1], 4), yticklabels=range(0, attn_heads[h].shape[-2], 4), vmin=vmin, vmax=vmax)

            axes[h].set_title(f"{layer_type} Head {h+1}")
            axes[h].set_xlabel("Key Position")
            axes[h].set_ylabel("Query Position")
        
        # Hide unused subplots
        for ax in axes[num_heads:]:
            ax.axis('off')
        
        plt.suptitle(f"{ts_id} - {layer_type} Attention Heads", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_type.lower()}_heads.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Mean attention across all heads
        mean_attn = np.mean(attn_heads, axis=0)  # (L_dec, L_enc)
        q_len, k_len = mean_attn.shape
        tick_interval_q = max(1, q_len // 8)
        tick_interval_k = max(1, k_len // 8)

        plt.figure(figsize=(10, 8))

        ax = sns.heatmap(mean_attn, cmap="magma", square=True, cbar=True,
                 xticklabels=tick_interval_k, yticklabels=tick_interval_q,
                 linewidths=0.2, linecolor='gray')

        # Apply tick marks only at intervals
        ax.set_xticks(np.arange(0, k_len, tick_interval_k))
        ax.set_xticklabels(np.arange(0, k_len, tick_interval_k), rotation=0, fontsize=10)
        ax.set_yticks(np.arange(0, q_len, tick_interval_q))
        ax.set_yticklabels(np.arange(0, q_len, tick_interval_q), rotation=0, fontsize=10)

        plt.title(f"{ts_id} - {layer_type} Mean Attention\n(Averaged across {num_heads} heads)", fontsize=14)
        plt.xlabel("Key Position (Encoder)", fontsize=12)
        plt.ylabel("Query Position (Decoder)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_type.lower()}_mean.png"), dpi=150)
        plt.close()
        
        # 3. Attention pattern analysis
        # Average attention per position
        avg_attention_received = np.mean(mean_attn, axis=0)  # How much attention each position receives
        avg_attention_given = np.mean(mean_attn, axis=1)     # How much attention each position gives
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(avg_attention_received, linewidth=2, color='blue')
        ax1.set_title(f"{layer_type} - Attention Received by Position")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Average Attention Weight")
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(avg_attention_given, linewidth=2, color='red')
        ax2.set_title(f"{layer_type} - Attention Given by Position")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Average Attention Weight")
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"{ts_id} - {layer_type} Attention Patterns", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer_type.lower()}_patterns.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Created {layer_type} attention visualizations for {ts_id}")
        
    except Exception as e:
        print(f"❌ Error creating {layer_type} attention heatmaps for {ts_id}: {e}")
        import traceback
        traceback.print_exc()

def create_encoder_mean_attention(attn_heads, layer_idx, ts_id, output_dir):
    mean_attn = np.mean(attn_heads, axis=0)  # → (L, L)
    q_len, k_len = mean_attn.shape
    tick_interval_q = max(1, q_len // 8)
    tick_interval_k = max(1, k_len // 8)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mean_attn, cmap="magma", square=True, cbar=True,
                     xticklabels=tick_interval_k, yticklabels=tick_interval_q,
                     linewidths=0.2, linecolor='gray')

    ax.set_xticks(np.arange(0, k_len, tick_interval_k))
    ax.set_xticklabels(np.arange(0, k_len, tick_interval_k), rotation=0, fontsize=10)
    ax.set_yticks(np.arange(0, q_len, tick_interval_q))
    ax.set_yticklabels(np.arange(0, q_len, tick_interval_q), rotation=0, fontsize=10)

    plt.title(f"{ts_id} - Encoder Layer {layer_idx} Mean Attention", fontsize=14)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)
    plt.tight_layout()

    mean_path = os.path.join(output_dir, f"encoder_layer_{layer_idx}_mean.png")
    plt.savefig(mean_path, dpi=150)
    plt.close()
    print(f"✅ Saved mean attention heatmap for encoder layer {layer_idx} → {mean_path}")

# ========== IMPROVED ATTENTION MAPS SAVING ==========
def save_attention_map(attn_list, path, layer_type):
    if not attn_list:
        print(f"⚠️ No {layer_type} attention maps collected.")
        return False
    
    try:
        # Handle different attention map formats
        processed_attns = []
        for attn in attn_list:
            if isinstance(attn, torch.Tensor):
                attn = attn.detach().cpu().numpy()
            
            # Ensure 4D: [batch, heads, seq_len, seq_len]
            if attn.ndim == 2:
                # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                attn = attn[np.newaxis, np.newaxis, :, :]
            elif attn.ndim == 3:
                # [heads, seq_len, seq_len] -> [1, heads, seq_len, seq_len]
                attn = attn[np.newaxis, :, :, :]
            elif attn.ndim == 4:
                # Already correct format
                pass
            else:
                print(f"⚠️ Unexpected attention shape: {attn.shape}")
                continue
                
            processed_attns.append(attn)
        
        if processed_attns:
            # Stack and average over batches and layers
            attn_stack = np.concatenate(processed_attns, axis=0)
            avg_attn = np.mean(attn_stack, axis=0)  # Average over batches
            
            np.save(path, avg_attn)
            print(f"✅ Saved {layer_type} attention to {path}, shape: {avg_attn.shape}")
            return True
        else:
            print(f"⚠️ No valid {layer_type} attention maps to save.")
            return False
            
    except Exception as e:
        print(f"⚠️ Error saving {layer_type} attention: {e}")
        return False

# ========== MAIN LOOP TO RUN CODE ==========
all_metrics = []

for ts_id in series_ids:
    try:
        print(f"\n===== Running Autoformer on {ts_id} =====")
        args = create_args(ts_id)
        exp = Exp_Main(args)

        # print(f"[Debug] Model architecture:\n{exp.model}")
        # print(exp.model.encoder)  # This should list the layers and their modules

        # Clear previous attention maps
        attention_maps["encoder"].clear()
        attention_maps["decoder"].clear()

        # Register hooks with better error handling
        hooks = []
        for i, layer in enumerate(exp.model.encoder.attn_layers):
            try:
                # Target the AutoCorrelation module inside AutoCorrelationLayer
                module = layer.attention.inner_correlation  # 👈 this is key
                hook = module.register_forward_hook(make_encoder_hook(i))
                hooks.append(hook)
                print(f"✅ Registered encoder hook on layer {i}")
            except Exception as e:
                print(f"❌ Failed to hook layer {i}: {e}")

        if hasattr(exp.model, 'decoder') and hasattr(exp.model.decoder, 'layers'):
                print(f"✅ Registering hooks for {len(exp.model.decoder.layers)} decoder layers")

                num_decoder_layers = len(exp.model.decoder.layers)
                last_decoder_layer = exp.model.decoder.layers[-1]

                hook = last_decoder_layer.register_forward_hook(hook_decoder_attention)
                hooks.append(hook)

                print(f"✅ Registered hook only for LAST decoder layer (Layer {num_decoder_layers - 1})")


        # Train and test
        setting = f"{args.model_id}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}"
        exp.train(setting)
        
        # Get test data loader for attention extraction
        test_data, test_loader = exp._get_data(flag='test')
        
        # Try manual attention extraction if hooks didn't work
        if not attention_maps["encoder"] and not attention_maps["decoder"]:
            print("🔍 Hooks didn't capture attention, trying manual extraction...")
            enc_attns, dec_attns = extract_attention_manually(exp.model, test_loader, args)
            attention_maps["encoder"].extend(enc_attns)
            attention_maps["decoder"].extend(dec_attns)
        
        # Run test
        exp.test(setting, test=1)
        
        # Create output directory for this series
        series_output_dir = os.path.join(output_dir, ts_id, "results")
        os.makedirs(series_output_dir, exist_ok=True)
        
        # Extract predictions (your existing code)
        possible_result_paths = [
            f'./results/{setting}/',
            f'./results/',
            f'./checkpoints/{setting}/',
            './outputs/',
            './exp_results/'
        ]
        
        y_pred = None
        y_true = None
        
        for result_path in possible_result_paths:
            if os.path.exists(result_path):
                print(f"🔍 Checking {result_path}")
                files = os.listdir(result_path)
                
                pred_patterns = ['pred.npy', 'preds.npy', 'prediction.npy', 'test_pred.npy']
                true_patterns = ['true.npy', 'trues.npy', 'ground_truth.npy', 'test_true.npy']
                
                pred_patterns.extend([f'{setting}_pred.npy', f'pred_{setting}.npy'])
                true_patterns.extend([f'{setting}_true.npy', f'true_{setting}.npy'])
                
                pred_file = None
                true_file = None
                
                for pattern in pred_patterns:
                    if pattern in files:
                        pred_file = pattern
                        break
                
                for pattern in true_patterns:
                    if pattern in files:
                        true_file = pattern
                        break
                
                if pred_file and true_file:
                    try:
                        pred_data = np.load(os.path.join(result_path, pred_file))
                        true_data = np.load(os.path.join(result_path, true_file))
                        
                        y_pred = pred_data.reshape(-1)
                        y_true = true_data.reshape(-1)
                        
                        print(f"✅ Found predictions in {result_path}: {pred_file}, {true_file}")
                        break
                    except Exception as e:
                        print(f"⚠️ Error loading from {result_path}: {e}")
                        continue
        
        # If still no predictions, extract from data loader
        if y_pred is None or y_true is None:
            print("🔍 Trying to extract predictions from data loader...")
            try:
                exp.model.eval()
                preds = []
                trues = []
                
                with torch.no_grad():
                    for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                        batch_x = batch_x.float()
                        batch_y = batch_y.float()
                        batch_x_mark = batch_x_mark.float()
                        batch_y_mark = batch_y_mark.float()
                        
                        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float()
                        
                        outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        # Handle different output formats
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
                    print(f"✅ Extracted predictions from data loader")
                    
            except Exception as e:
                print(f"⚠️ Error extracting from data loader: {e}")
        
        # Remove hooks to prevent memory leaks
        for hook in hooks:
            hook.remove()
        
        if y_pred is None or y_true is None:
            print(f"❌ Could not find or extract predictions for {ts_id}")
            continue
            
        # Save predictions and ground truth
        np.save(os.path.join(series_output_dir, "pred.npy"), y_pred)
        np.save(os.path.join(series_output_dir, "true.npy"), y_true)
        
        # Save CSV
        df_series = pd.DataFrame({"GroundTruth": y_true, "Prediction": y_pred})
        df_series.to_csv(os.path.join(series_output_dir, "pred_vs_true.csv"), index=False)
        
        # # Forecast Plot
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_true, label="Ground Truth", linewidth=2, alpha=0.8)
        # plt.plot(y_pred, label="Autoformer Forecast", linestyle='--', linewidth=2, alpha=0.8)
        # plt.title(f"{ts_id} - Autoformer Forecast vs Ground Truth", fontsize=14)
        # plt.xlabel("Time Step")
        # plt.ylabel("Value")
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.savefig(os.path.join(series_output_dir, "forecast.png"), dpi=150)
        # plt.close()

        # Save attention maps
        encoder_path = os.path.join(series_output_dir, "encoder_attn.npy")
        decoder_path = os.path.join(series_output_dir, "decoder_attn.npy")
        
        # encoder_saved = save_attention_map(attention_maps["encoder"], encoder_path, "encoder")
        encoder_saved = False
        for layer_idx, attn_list in attention_maps["encoder"].items():
            layer_path = os.path.join(series_output_dir, f"encoder_attn_layer{layer_idx}.npy")
            saved = save_attention_map(attn_list, layer_path, f"encoder_layer_{layer_idx}")
            encoder_saved = encoder_saved or saved  # mark as saved if at least one layer succeeded
            
        decoder_saved = save_attention_map(attention_maps["decoder"], decoder_path, "decoder")
        
        # Evaluate metrics FIRST (before any potential errors)
        metrics = evaluate_metrics(y_true, y_pred)
        metrics["series"] = ts_id
        all_metrics.append(metrics)
        print(f"✅ Metrics calculated for {ts_id}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")

        # Create attention visualizations
        try:
            if attention_maps["encoder"]:
                visualize_encoder_layers(attention_maps["encoder"], ts_id, series_output_dir)
            else:
                print("⚠️ No encoder attention maps collected")

            if decoder_saved:
                create_attention_heatmaps(decoder_path, ts_id, "Decoder", series_output_dir)
            else:
                print("⚠️ No decoder attention maps collected")

        except Exception as viz_error:
            print(f"⚠️ Error creating attention visualizations for {ts_id}: {viz_error}")

        # ======= GLOBAL MEAN ACROSS ENCODER LAYERS =======
        if attention_maps["encoder"]:
            all_means = []
            for layer_idx, attn_list in attention_maps["encoder"].items():
                if attn_list:
                    avg = np.mean(np.stack(attn_list, axis=0), axis=0)  # (H, L, L)
                    layer_mean = np.mean(avg, axis=0)  # (L, L)
                    all_means.append(layer_mean)

            if all_means:
                global_mean = np.mean(np.stack(all_means, axis=0), axis=0)
                plt.figure(figsize=(6, 5))
                sns.heatmap(global_mean, cmap="magma", square=True,
                            xticklabels=range(0, global_mean.shape[1], 4),
                            yticklabels=range(0, global_mean.shape[0], 4))
                plt.title(f"{ts_id} - Mean Attention Across All Encoder Layers", fontsize=12)
                plt.xlabel("Key Position")
                plt.ylabel("Query Position")
                plt.tight_layout()
                global_mean_path = os.path.join(series_output_dir, "encoder_mean_heads.png")
                plt.savefig(global_mean_path, dpi=150)
                plt.close()
                print(f"✅ Saved global encoder mean attention → {global_mean_path}")

        # Remove hooks
        for hook in hooks:
            hook.remove()

    except Exception as e:
        print(f"❌ Error processing {ts_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Final metrics analysis (your existing code)
df_metrics = pd.DataFrame(all_metrics)

if not df_metrics.empty:
    df_metrics.to_csv(os.path.join(output_dir, "autoformer_m5_evaluation_summary.csv"), index=False)
    
    print(f"\n=== FINAL RESULTS FOR {len(df_metrics)} SERIES ===")
    print(df_metrics.to_string(index=False))

    print("\n=== OVERALL AVERAGE METRICS ===")
    avg_metrics = df_metrics[["MSE", "MAE", "RMSE", "MAPE", "WAPE"]].mean()
    print(avg_metrics.to_string())

    # Save metrics visualization
    plt.figure(figsize=(12, 6))
    metrics_to_plot = ["MSE", "MAE", "RMSE", "MAPE", "WAPE"]
    sns.boxplot(data=df_metrics[metrics_to_plot])
    plt.title("Autoformer Performance - Metric Distribution Across Series")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_boxplot.png"), dpi=150)
    plt.close()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🎯 Successfully processed {len(df_metrics)} series with attention maps!")
else:
    print("⚠️ No metrics to display - check for errors above")

print(f"\n🎉 Autoformer attention analysis finished!")
print(f"📊 Check '{output_dir}' for all results including attention heatmaps")