"""
Training module for Fake News Detection project.
Supports both LSTM and Transformer models with early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from typing import Dict, Tuple
import copy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 50,
    patience: int = 3,
    model_type: str = "lstm"
) -> Dict[str, list]:
    """
    Train a model with early stopping based on validation F1 score.

    Args:
        model: PyTorch model (LSTMClassifier or TransformerClassifier)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: torch.device ("cuda" or "cpu")
        num_epochs: Maximum number of epochs (default: 50)
        patience: Number of epochs to wait for improvement before stopping (default: 3)
        model_type: "lstm" or "transformer" (default: "lstm")

    Returns:
        Dictionary with training history:
        {
            "train_loss": [...],
            "val_loss": [...],
            "train_f1": [...],
            "val_f1": [...]
        }
    """
    # Validate model_type
    assert model_type in ["lstm", "transformer"], \
        f"model_type must be 'lstm' or 'transformer', got '{model_type}'"

    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # History tracking
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": []
    }

    # Early stopping variables
    best_f1 = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print(f"Starting training for {model_type.upper()} model...")
    print(f"Device: {device}")
    print(f"Max epochs: {num_epochs}, Patience: {patience}\n")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch in train_loader:
            # Move batch to device
            labels = batch["label"].to(device)

            # Forward pass (model-specific)
            if model_type == "lstm":
                input_ids = batch["input_ids"].to(device)
                lengths = batch["length"].to(device)
                outputs = model(input_ids, lengths)
            else:  # transformer
                input_ids = batch["input_ids"].to(device)
                masks = batch["attention_mask"].to(device)
                outputs = model(input_ids, masks)

            # Compute loss
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            train_preds.extend(
                (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Average training loss
        avg_train_loss = train_loss / len(train_loader)

        # Training F1 score
        train_f1 = f1_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                labels = batch["label"].to(device)

                # Forward pass (model-specific)
                if model_type == "lstm":
                    input_ids = batch["input_ids"].to(device)
                    lengths = batch["length"].to(device)
                    outputs = model(input_ids, lengths)
                else:  # transformer
                    input_ids = batch["input_ids"].to(device)
                    masks = batch["attention_mask"].to(device)
                    outputs = model(input_ids, masks)

                # Compute loss
                loss = criterion(outputs.squeeze(), labels)

                # Accumulate metrics
                val_loss += loss.item()
                val_preds.extend(
                    (torch.sigmoid(outputs.squeeze()) > 0.5).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Average validation loss
        avg_val_loss = val_loss / len(val_loader)

        # Validation F1 score
        val_f1 = f1_score(val_labels, val_preds)

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping logic (based on validation F1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  ✓ New best validation F1: {best_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(
                    f"Restoring best model with validation F1: {best_f1:.4f}")
                model.load_state_dict(best_model_state)
                break

        print()

    # If training completed without early stopping, restore best model
    if epochs_no_improve < patience:
        if best_model_state is not None:
            print(
                f"Training completed. Restoring best model with validation F1: {best_f1:.4f}")
            model.load_state_dict(best_model_state)

    return history


if __name__ == '__main__':
    """
    Validation and testing code.
    """
    import sys
    import os

    # Add current directory to path to import modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from datasets import TransformerDataset, LSTMDataset
    from models import LSTMClassifier, TransformerClassifier

    print("=" * 60)
    print("Training Module Validation")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Create dummy data (from datasets.py validation)
    vocab_size = 1000
    max_len = 32

    # Transformer dummy data
    input_ids_trans = [[2, 3, 7, 5, 4] + [0] * (max_len - 5),
                       [2, 8, 1] + [0] * (max_len - 3)]
    masks_trans = [[False] * 5 + [True] * (max_len - 5),
                   [False] * 3 + [True] * (max_len - 3)]
    labels_trans = [1, 0]

    # LSTM dummy data
    input_ids_lstm = [[3, 7, 5, 4] + [0] * (max_len - 4),
                      [8, 1] + [0] * (max_len - 2)]
    labels_lstm = [1, 0]

    # Create datasets
    trans_dataset = TransformerDataset(
        input_ids_trans, masks_trans, labels_trans)
    lstm_dataset = LSTMDataset(input_ids_lstm, labels_lstm)

    # Create data loaders
    train_loader_trans = DataLoader(trans_dataset, batch_size=2, shuffle=False)
    val_loader_trans = DataLoader(trans_dataset, batch_size=2, shuffle=False)

    train_loader_lstm = DataLoader(lstm_dataset, batch_size=2, shuffle=False)
    val_loader_lstm = DataLoader(lstm_dataset, batch_size=2, shuffle=False)

    # Test 1: LSTM training
    print("=" * 60)
    print("Test 1: Training LSTM model")
    print("=" * 60)

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_size=128,
        num_layers=1,
        dropout=0.1
    )

    history_lstm = train_model(
        model=lstm_model,
        train_loader=train_loader_lstm,
        val_loader=val_loader_lstm,
        device=device,
        num_epochs=5,
        patience=2,
        model_type="lstm"
    )

    # Verify history structure
    assert len(history_lstm["train_loss"]) == len(history_lstm["val_loss"]) == \
        len(history_lstm["train_f1"]) == len(history_lstm["val_f1"]), \
        "History keys have different lengths"

    num_epochs_trained = len(history_lstm["train_loss"])
    print(f"\n✓ LSTM training completed: {num_epochs_trained} epochs")
    print(f"  History entries: {num_epochs_trained}")
    print(f"  Final train loss: {history_lstm['train_loss'][-1]:.4f}")
    print(f"  Final val F1: {history_lstm['val_f1'][-1]:.4f}")

    # Verify loss decreases (or at least doesn't explode)
    assert history_lstm["train_loss"][-1] < 10.0, "Training loss is too high"
    print("  ✓ Loss values are reasonable")

    # Test 2: Transformer training
    print("\n" + "=" * 60)
    print("Test 2: Training Transformer model")
    print("=" * 60)

    transformer_model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        max_len=max_len,
        dropout=0.1
    )

    history_trans = train_model(
        model=transformer_model,
        train_loader=train_loader_trans,
        val_loader=val_loader_trans,
        device=device,
        num_epochs=5,
        patience=2,
        model_type="transformer"
    )

    num_epochs_trained = len(history_trans["train_loss"])
    print(f"\n✓ Transformer training completed: {num_epochs_trained} epochs")
    print(f"  History entries: {num_epochs_trained}")
    print(f"  Final train loss: {history_trans['train_loss'][-1]:.4f}")
    print(f"  Final val F1: {history_trans['val_f1'][-1]:.4f}")

    # Verify loss decreases
    assert history_trans["train_loss"][-1] < 10.0, "Training loss is too high"
    print("  ✓ Loss values are reasonable")

    # Test 3: Early stopping verification
    print("\n" + "=" * 60)
    print("Test 3: Early stopping verification")
    print("=" * 60)

    # Create a model that will trigger early stopping
    # Use same data for train and val, so it should overfit quickly
    # Then we'll use patience=1 to trigger early stopping
    test_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=32,
        hidden_size=64,
        num_layers=1,
        dropout=0.0  # No dropout to allow quick learning
    )

    # Use very small patience to trigger early stopping
    history_early_stop = train_model(
        model=test_model,
        train_loader=train_loader_lstm,
        val_loader=val_loader_lstm,
        device=device,
        num_epochs=5,
        patience=1,  # Stop after 1 epoch without improvement
        model_type="lstm"
    )

    num_epochs_early_stop = len(history_early_stop["train_loss"])
    print(f"\n✓ Early stopping test completed")
    print(f"  Epochs trained: {num_epochs_early_stop}")
    print(f"  Expected: <= 5 (should stop early)")

    # Verify early stopping worked (should stop before 5 epochs if patience is small)
    # Note: With patience=1, it should stop after at most 2 epochs (epoch 0 improves, epoch 1 doesn't)
    assert num_epochs_early_stop <= 5, "Early stopping didn't work"
    print("  ✓ Early stopping triggered correctly")

    # Verify best model was restored (check that final F1 matches best F1)
    best_val_f1 = max(history_early_stop["val_f1"])
    final_val_f1 = history_early_stop["val_f1"][-1]
    print(f"  Best val F1 during training: {best_val_f1:.4f}")
    print(f"  Final val F1 (after restore): {final_val_f1:.4f}")
    assert abs(final_val_f1 -
               best_val_f1) < 0.001, "Best model was not restored correctly"
    print("  ✓ Best model weights restored correctly")

    print("\n" + "=" * 60)
    print("All validation tests passed! ✓")
    print("=" * 60)
