"""
PyTorch Dataset classes for Fake News Detection project.
Pure PyTorch implementation (no TensorFlow/Keras).
"""

import torch
from torch.utils.data import Dataset
from typing import List, Optional
import json


class TransformerDataset(Dataset):
    """
    Dataset class for Transformer models.

    Expects:
    - input_ids: List[List[int]] of shape [num_samples, max_len]
    - attention_masks: List[List[bool]] of shape [num_samples, max_len]
      (True = padding position, False = actual token)
    - labels: List[int] of shape [num_samples] (0 or 1)
    """

    def __init__(self, input_ids: List[List[int]], attention_masks: List[List[bool]], labels: List[int]):
        """
        Initialize TransformerDataset.

        Args:
            input_ids: List of token index sequences
            attention_masks: List of attention masks (True where padding)
            labels: List of labels (0 or 1)
        """
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

        # Validate that all lists have the same length
        assert len(input_ids) == len(attention_masks) == len(labels), \
            f"Mismatched lengths: input_ids={len(input_ids)}, attention_masks={len(attention_masks)}, labels={len(labels)}"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
            - "input_ids": torch.LongTensor of shape [max_len]
            - "attention_mask": torch.BoolTensor of shape [max_len] (True = padding)
            - "label": torch.FloatTensor (scalar, 0.0 or 1.0)
        """
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.bool),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class LSTMDataset(Dataset):
    """
    Dataset class for LSTM models.

    Expects:
    - input_ids: List[List[int]] of shape [num_samples, max_len]
    - labels: List[int] of shape [num_samples] (0 or 1)
    - lengths: Optional[List[int]] of original sequence lengths before padding
               If None, computed on-the-fly from input_ids (counting non-PAD tokens)
    """

    def __init__(self, input_ids: List[List[int]], labels: List[int], lengths: Optional[List[int]] = None):
        """
        Initialize LSTMDataset.

        Args:
            input_ids: List of token index sequences
            labels: List of labels (0 or 1)
            lengths: Optional list of original sequence lengths. If None, computed from input_ids.
        """
        self.input_ids = input_ids
        self.labels = labels
        self.lengths = lengths

        # Validate that input_ids and labels have the same length
        assert len(input_ids) == len(labels), \
            f"Mismatched lengths: input_ids={len(input_ids)}, labels={len(labels)}"

        # If lengths provided, validate they match
        if self.lengths is not None:
            assert len(self.lengths) == len(input_ids), \
                f"Mismatched lengths: input_ids={len(input_ids)}, lengths={len(self.lengths)}"

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)

    def _compute_length(self, idx: int) -> int:
        """
        Compute the original sequence length for a given sample.
        Length is the number of non-PAD tokens (where token != 0).

        Args:
            idx: Sample index

        Returns:
            Original sequence length (number of non-PAD tokens)
        """
        seq = self.input_ids[idx]
        # Find indices where token != 0 (non-PAD)
        non_pad_indices = [i for i, token in enumerate(seq) if token != 0]

        if len(non_pad_indices) == 0:
            # All tokens are PAD, length is 0
            return 0

        # Length is the index of the last non-PAD token + 1
        # Or equivalently, the number of non-PAD tokens
        return len(non_pad_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
            - "input_ids": torch.LongTensor of shape [max_len]
            - "length": torch.LongTensor (scalar, original sequence length)
            - "label": torch.FloatTensor (scalar, 0.0 or 1.0)
        """
        # Get or compute length
        if self.lengths is not None:
            length = self.lengths[idx]
        else:
            length = self._compute_length(idx)

        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


if __name__ == '__main__':
    # 1. Load vocab.json to verify <PAD>=0
    print("=" * 60)
    print("Loading vocab.json to verify <PAD> token...")
    print("=" * 60)
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    pad_token_id = vocab.get('<PAD>', None)
    print(f"<PAD> token ID: {pad_token_id}")
    assert pad_token_id == 0, f"Expected <PAD>=0, got {pad_token_id}"
    print("✓ <PAD> token verified as 0\n")

    # 2. Create dummy data
    print("=" * 60)
    print("Creating dummy data...")
    print("=" * 60)

    # Transformer data
    # input_ids: [[2, 3, 7, 5, 4], [2, 8, 1, 0, 0]]
    # 2 = <CLS>, 0 = <PAD>
    input_ids_trans = [[2, 3, 7, 5, 4], [2, 8, 1, 0, 0]]

    # masks: [[False, False, False, False, False], [False, False, False, True, True]]
    # True = padding position
    masks_trans = [[False, False, False, False, False],
                   [False, False, False, True, True]]

    # Labels
    labels = [1, 0]

    # LSTM data (no <CLS> token)
    # input_ids: [[3, 7, 5, 4, 0], [8, 1, 0, 0, 0]]
    input_ids_lstm = [[3, 7, 5, 4, 0], [8, 1, 0, 0, 0]]

    print(f"Transformer input_ids: {input_ids_trans}")
    print(f"Transformer attention_masks: {masks_trans}")
    print(f"LSTM input_ids: {input_ids_lstm}")
    print(f"Labels: {labels}\n")

    # 3. Instantiate both datasets
    print("=" * 60)
    print("Instantiating datasets...")
    print("=" * 60)
    transformer_dataset = TransformerDataset(
        input_ids_trans, masks_trans, labels)
    lstm_dataset = LSTMDataset(input_ids_lstm, labels)

    print(f"TransformerDataset size: {len(transformer_dataset)}")
    print(f"LSTMDataset size: {len(lstm_dataset)}\n")

    # 4. Test __getitem__(0) for each
    print("=" * 60)
    print("Testing TransformerDataset[0]:")
    print("=" * 60)
    trans_sample = transformer_dataset[0]
    print(f"input_ids shape: {trans_sample['input_ids'].shape}")
    print(f"input_ids dtype: {trans_sample['input_ids'].dtype}")
    print(f"input_ids values: {trans_sample['input_ids'].tolist()}")
    print(f"\nattention_mask shape: {trans_sample['attention_mask'].shape}")
    print(f"attention_mask dtype: {trans_sample['attention_mask'].dtype}")
    print(f"attention_mask values: {trans_sample['attention_mask'].tolist()}")
    print(f"\nlabel shape: {trans_sample['label'].shape}")
    print(f"label dtype: {trans_sample['label'].dtype}")
    print(f"label value: {trans_sample['label'].item()}")

    # Verify Transformer attention mask has True only where token=0
    print(f"\nVerifying attention mask:")
    input_ids_vals = trans_sample['input_ids'].tolist()
    mask_vals = trans_sample['attention_mask'].tolist()
    for i, (token_id, mask_val) in enumerate(zip(input_ids_vals, mask_vals)):
        expected_mask = (token_id == 0)
        assert mask_val == expected_mask, \
            f"Position {i}: token={token_id}, mask={mask_val}, expected={expected_mask}"
    print("✓ Attention mask correctly marks padding positions (True where token=0)\n")

    print("=" * 60)
    print("Testing LSTMDataset[0]:")
    print("=" * 60)
    lstm_sample = lstm_dataset[0]
    print(f"input_ids shape: {lstm_sample['input_ids'].shape}")
    print(f"input_ids dtype: {lstm_sample['input_ids'].dtype}")
    print(f"input_ids values: {lstm_sample['input_ids'].tolist()}")
    print(f"\nlength shape: {lstm_sample['length'].shape}")
    print(f"length dtype: {lstm_sample['length'].dtype}")
    print(f"length value: {lstm_sample['length'].item()}")
    print(f"\nlabel shape: {lstm_sample['label'].shape}")
    print(f"label dtype: {lstm_sample['label'].dtype}")
    print(f"label value: {lstm_sample['label'].item()}")

    # Verify LSTM length for first sample = 4 (stops at first 0)
    expected_length = 4
    actual_length = lstm_sample['length'].item()
    print(f"\nVerifying length:")
    print(
        f"Expected length: {expected_length} (4 non-PAD tokens: [3, 7, 5, 4])")
    print(f"Actual length: {actual_length}")
    assert actual_length == expected_length, \
        f"Expected length={expected_length}, got {actual_length}"
    print("✓ Length correctly computed as 4\n")

    # Test second sample
    print("=" * 60)
    print("Testing LSTMDataset[1]:")
    print("=" * 60)
    lstm_sample_1 = lstm_dataset[1]
    print(f"input_ids: {lstm_sample_1['input_ids'].tolist()}")
    print(f"length: {lstm_sample_1['length'].item()}")
    expected_length_1 = 2  # [8, 1] are non-PAD
    actual_length_1 = lstm_sample_1['length'].item()
    assert actual_length_1 == expected_length_1, \
        f"Expected length={expected_length_1}, got {actual_length_1}"
    print(f"✓ Length correctly computed as {expected_length_1}\n")

    # Test with provided lengths
    print("=" * 60)
    print("Testing LSTMDataset with provided lengths:")
    print("=" * 60)
    provided_lengths = [4, 2]
    lstm_dataset_with_lengths = LSTMDataset(
        input_ids_lstm, labels, lengths=provided_lengths)
    lstm_sample_provided = lstm_dataset_with_lengths[0]
    print(f"Provided length: {provided_lengths[0]}")
    print(f"Retrieved length: {lstm_sample_provided['length'].item()}")
    assert lstm_sample_provided['length'].item() == provided_lengths[0]
    print("✓ Provided lengths work correctly\n")

    print("=" * 60)
    print("All validation tests passed! ✓")
    print("=" * 60)
