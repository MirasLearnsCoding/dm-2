"""
Hyperparameter importance analysis via sensitivity analysis.
Identifies which hyperparameters matter most for fake news detection.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import json
from datetime import datetime

from models import LSTMClassifier, TransformerClassifier
from training import train_model


# Conservative default hyperparameters
LSTM_DEFAULTS = {
    "embed_dim": 64,
    "hidden_size": 128,
    "num_layers": 1,
    "dropout": 0.1
}

TRANSFORMER_DEFAULTS = {
    "d_model": 128,
    "nhead": 2,
    "num_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.1
}

# Hyperparameter candidates for sensitivity analysis
LSTM_HP_CANDIDATES = {
    "embed_dim": [64, 128, 256],
    "hidden_size": [128, 256, 512],
    "num_layers": [1, 2, 3]
}

TRANSFORMER_HP_CANDIDATES = {
    "d_model": [128, 256],  # Must be divisible by nhead
    "nhead": [2, 4, 8],
    "num_layers": [2, 4],
    "dim_feedforward": [512, 1024]
}


def analyze_hyperparameter_sensitivity(
    model_type: str,
    hp_name: str,
    hp_values: List,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    max_len: int,
    device: torch.device,
    defaults: Dict
) -> Dict:
    """
    Analyze sensitivity of a single hyperparameter.
    
    Args:
        model_type: "lstm" or "transformer"
        hp_name: Name of the hyperparameter to analyze
        hp_values: List of values to test for this hyperparameter
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        device: torch.device
        defaults: Dictionary of default hyperparameter values
        
    Returns:
        Dictionary with results:
        {
            "values": [...],
            "f1_scores": [...],
            "delta_f1": float
        }
    """
    print(f"\n{'='*60}")
    print(f"Analyzing {hp_name} for {model_type.upper()}")
    print(f"Values to test: {hp_values}")
    print(f"{'='*60}")
    
    f1_scores = []
    tested_values = []
    
    for hp_value in hp_values:
        print(f"\nTesting {hp_name}={hp_value}...")
        
        # Create hyperparameter dict with defaults, then override the current HP
        hp_dict = defaults.copy()
        hp_dict[hp_name] = hp_value
        
        # For Transformer, ensure d_model is divisible by nhead
        skip_value = False
        if model_type == "transformer":
            if hp_name == "d_model":
                # If we're varying d_model, ensure nhead divides it
                nhead = hp_dict["nhead"]
                if hp_value % nhead != 0:
                    # Adjust nhead to be compatible
                    # Find largest nhead that divides hp_value
                    compatible_nhead = None
                    for n in [8, 4, 2]:
                        if hp_value % n == 0:
                            compatible_nhead = n
                            break
                    if compatible_nhead:
                        hp_dict["nhead"] = compatible_nhead
                        print(f"  Adjusted nhead to {compatible_nhead} for compatibility")
                    else:
                        print(f"  Skipping: d_model={hp_value} not divisible by any valid nhead")
                        skip_value = True
            elif hp_name == "nhead":
                # If we're varying nhead, ensure it divides d_model
                d_model = hp_dict["d_model"]
                if d_model % hp_value != 0:
                    # Skip incompatible combinations
                    print(f"  Skipping: d_model={d_model} not divisible by nhead={hp_value}")
                    skip_value = True
        
        if skip_value:
            continue
        
        # Create model
        if model_type == "lstm":
            model = LSTMClassifier(
                vocab_size=vocab_size,
                embed_dim=hp_dict["embed_dim"],
                hidden_size=hp_dict["hidden_size"],
                num_layers=hp_dict["num_layers"],
                dropout=hp_dict["dropout"]
            )
        else:  # transformer
            model = TransformerClassifier(
                vocab_size=vocab_size,
                d_model=hp_dict["d_model"],
                nhead=hp_dict["nhead"],
                num_layers=hp_dict["num_layers"],
                dim_feedforward=hp_dict["dim_feedforward"],
                max_len=max_len,
                dropout=hp_dict["dropout"]
            )
        
        # Train model with early stopping
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=10,
            patience=2,
            model_type=model_type
        )
        
        # Get best validation F1 score
        best_val_f1 = max(history["val_f1"])
        f1_scores.append(best_val_f1)
        tested_values.append(hp_value)
        print(f"  Best validation F1: {best_val_f1:.4f}")
        
        # Clean up model to free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute delta F1
    if len(f1_scores) > 0:
        delta_f1 = max(f1_scores) - min(f1_scores)
    else:
        delta_f1 = 0.0
    
    return {
        "values": tested_values,  # Only include values that were actually tested
        "f1_scores": f1_scores,
        "delta_f1": delta_f1
    }


def run_sensitivity_analysis(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    max_len: int,
    device: torch.device
) -> Dict:
    """
    Run sensitivity analysis for all hyperparameters of a model type.
    
    Args:
        model_type: "lstm" or "transformer"
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        vocab_size: Vocabulary size
        max_len: Maximum sequence length
        device: torch.device
        
    Returns:
        Dictionary with results for all hyperparameters
    """
    assert model_type in ["lstm", "transformer"], \
        f"model_type must be 'lstm' or 'transformer', got '{model_type}'"
    
    # Get defaults and candidates based on model type
    if model_type == "lstm":
        defaults = LSTM_DEFAULTS
        candidates = LSTM_HP_CANDIDATES
    else:
        defaults = TRANSFORMER_DEFAULTS
        candidates = TRANSFORMER_HP_CANDIDATES
    
    results = {}
    
    # Analyze each hyperparameter
    for hp_name, hp_values in candidates.items():
        result = analyze_hyperparameter_sensitivity(
            model_type=model_type,
            hp_name=hp_name,
            hp_values=hp_values,
            train_loader=train_loader,
            val_loader=val_loader,
            vocab_size=vocab_size,
            max_len=max_len,
            device=device,
            defaults=defaults
        )
        results[hp_name] = result
    
    return results


def save_results(
    model_type: str,
    results: Dict,
    output_file: str = None
) -> str:
    """
    Save sensitivity analysis results to JSON file.
    
    Args:
        model_type: "lstm" or "transformer"
        results: Results dictionary from run_sensitivity_analysis
        output_file: Output file path (default: {model_type}_importance.json)
        
    Returns:
        Path to saved file
    """
    if output_file is None:
        output_file = f"{model_type}_importance.json"
    
    output_data = {
        "model_type": model_type,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    return output_file


if __name__ == '__main__':
    """
    Example usage and validation.
    """
    import sys
    import os
    
    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from datasets import TransformerDataset, LSTMDataset
    from data_preprocessing import get_data_splits, encode_texts_for_transformer, encode_texts_for_lstm
    
    print("=" * 60)
    print("Hyperparameter Importance Analysis")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load data and build vocabulary
    print("Loading data and building vocabulary...")
    train_df, val_df, test_df, vocab_dict = get_data_splits(max_len=256)
    vocab_size = len(vocab_dict)
    max_len = 256
    
    print(f"Vocab size: {vocab_size}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}\n")
    
    # Encode texts for both model types
    print("Encoding texts...")
    
    # For Transformer
    train_input_ids_trans, train_masks_trans = encode_texts_for_transformer(
        train_df['content'].tolist(), vocab_dict, max_len
    )
    val_input_ids_trans, val_masks_trans = encode_texts_for_transformer(
        val_df['content'].tolist(), vocab_dict, max_len
    )
    
    # For LSTM
    train_input_ids_lstm = encode_texts_for_lstm(
        train_df['content'].tolist(), vocab_dict, max_len
    )
    val_input_ids_lstm = encode_texts_for_lstm(
        val_df['content'].tolist(), vocab_dict, max_len
    )
    
    # Create datasets and loaders
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    
    # Use a subset for faster testing (optional - remove for full analysis)
    # For demonstration, use first 1000 samples
    subset_size = min(1000, len(train_df))
    print(f"Using subset of {subset_size} samples for faster analysis...\n")
    
    train_dataset_trans = TransformerDataset(
        train_input_ids_trans[:subset_size],
        train_masks_trans[:subset_size],
        train_labels[:subset_size]
    )
    val_dataset_trans = TransformerDataset(
        val_input_ids_trans[:subset_size],
        val_masks_trans[:subset_size],
        val_labels[:subset_size]
    )
    
    train_dataset_lstm = LSTMDataset(
        train_input_ids_lstm[:subset_size],
        train_labels[:subset_size]
    )
    val_dataset_lstm = LSTMDataset(
        val_input_ids_lstm[:subset_size],
        val_labels[:subset_size]
    )
    
    train_loader_trans = DataLoader(train_dataset_trans, batch_size=32, shuffle=True)
    val_loader_trans = DataLoader(val_dataset_trans, batch_size=32, shuffle=False)
    
    train_loader_lstm = DataLoader(train_dataset_lstm, batch_size=32, shuffle=True)
    val_loader_lstm = DataLoader(val_dataset_lstm, batch_size=32, shuffle=False)
    
    # Run sensitivity analysis for LSTM
    print("\n" + "=" * 60)
    print("LSTM Sensitivity Analysis")
    print("=" * 60)
    
    lstm_results = run_sensitivity_analysis(
        model_type="lstm",
        train_loader=train_loader_lstm,
        val_loader=val_loader_lstm,
        vocab_size=vocab_size,
        max_len=max_len,
        device=device
    )
    
    save_results("lstm", lstm_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("LSTM Hyperparameter Importance Summary")
    print("=" * 60)
    for hp_name, result in lstm_results.items():
        print(f"{hp_name}:")
        print(f"  Delta F1: {result['delta_f1']:.4f}")
        print(f"  F1 scores: {[f'{f:.4f}' for f in result['f1_scores']]}")
    
    # Run sensitivity analysis for Transformer
    print("\n" + "=" * 60)
    print("Transformer Sensitivity Analysis")
    print("=" * 60)
    
    transformer_results = run_sensitivity_analysis(
        model_type="transformer",
        train_loader=train_loader_trans,
        val_loader=val_loader_trans,
        vocab_size=vocab_size,
        max_len=max_len,
        device=device
    )
    
    save_results("transformer", transformer_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Transformer Hyperparameter Importance Summary")
    print("=" * 60)
    for hp_name, result in transformer_results.items():
        print(f"{hp_name}:")
        print(f"  Delta F1: {result['delta_f1']:.4f}")
        print(f"  F1 scores: {[f'{f:.4f}' for f in result['f1_scores']]}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

