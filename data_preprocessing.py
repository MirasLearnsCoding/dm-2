"""
Data preprocessing module for Fake News Detection project.
Pure PyTorch-compatible code (no TensorFlow/Keras).
"""

import pandas as pd
import numpy as np
import re
import json
from typing import List, Tuple, Dict
from collections import Counter
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """
    Clean text while preserving stylistic fake news cues.

    Args:
        text: Input text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs - replace with " <URL> "
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' <URL> ', text)
    text = re.sub(
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' <URL> ', text)

    # 2. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 3. Preserve stylistic fake news cues

    # Replace ALL CAPS words (len > 2) with " <ALLCAPS> "
    def replace_allcaps(match):
        word = match.group(0)
        if len(word) > 2 and word.isupper() and word.isalpha():
            return ' <ALLCAPS> '
        return word

    # Find all words (sequences of letters)
    text = re.sub(r'\b[A-Z]{3,}\b', replace_allcaps, text)

    # Replace repeated exclamation marks (!!!) with " <MULTIEXCL> "
    text = re.sub(r'!{2,}', ' <MULTIEXCL> ', text)

    # Replace repeated question marks (???) with " <MULTIQ> "
    text = re.sub(r'\?{2,}', ' <MULTIQ> ', text)

    # Replace ellipsis (...) with " <ELLIPSIS> "
    text = re.sub(r'\.{3,}', ' <ELLIPSIS> ', text)

    # Keep single !, ?, . with spaces around them
    # Add space before punctuation if not already present
    text = re.sub(r'([^\s])([!?.])', r'\1 \2', text)
    # Add space after punctuation if not already present
    text = re.sub(r'([!?.])([^\s])', r'\1 \2', text)

    # 4. Remove non-alphanumeric except spaces, !, ?, ., and digits
    # Keep: alphanumeric, spaces, !, ?, ., digits
    text = re.sub(r'[^\w\s!?.]', '', text)

    # 5. Do NOT lowercase text (case is a feature) - skip this step

    # 6. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def get_data_splits(max_len: int = 256) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load data, combine files, split into train/val/test, and build vocabulary.

    Args:
        max_len: Maximum sequence length (for reference, not used in this function)

    Returns:
        Tuple of (train_df, val_df, test_df, vocab_dict)
    """
    # Load CSV files
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')

    # Add labels: Fake=1, True=0
    fake_df['label'] = 1
    true_df['label'] = 0

    # Combine title + text into content column
    fake_df['content'] = fake_df['title'].astype(
        str) + ' ' + fake_df['text'].astype(str)
    true_df['content'] = true_df['title'].astype(
        str) + ' ' + true_df['text'].astype(str)

    # Combine dataframes
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Stratified split: first split train from temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        stratify=df['label'],
        random_state=42
    )

    # Split temp into val and test (50/50 of temp = 15% each of total)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )

    # Build vocabulary from TRAIN SET ONLY
    # Clean all texts in train set
    train_texts = train_df['content'].apply(clean_text).tolist()

    # Tokenize (split on whitespace)
    all_tokens = []
    for text in train_texts:
        tokens = text.split()
        all_tokens.extend(tokens)

    # Count token frequencies
    token_counts = Counter(all_tokens)

    # Build vocabulary with min_frequency=2
    vocab_dict = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<CLS>': 2
    }

    # Add tokens with frequency >= 2, starting from index 3
    idx = 3
    for token, count in token_counts.items():
        if count >= 2:
            vocab_dict[token] = idx
            idx += 1

    return train_df, val_df, test_df, vocab_dict


def encode_texts_for_transformer(
    texts: List[str],
    vocab: Dict[str, int],
    max_len: int
) -> Tuple[List[List[int]], List[List[bool]]]:
    """
    Encode texts for Transformer model.

    Args:
        texts: List of text strings
        vocab: Vocabulary dictionary mapping tokens to indices
        max_len: Maximum sequence length

    Returns:
        Tuple of (input_ids, attention_masks)
        - input_ids: List of token index sequences
        - attention_masks: List of boolean masks (True where token is NOT <PAD>)
    """
    input_ids = []
    attention_masks = []

    unk_idx = vocab.get('<UNK>', 1)
    cls_idx = vocab.get('<CLS>', 2)
    pad_idx = vocab.get('<PAD>', 0)

    for text in texts:
        # Clean text
        cleaned = clean_text(text)

        # Split into tokens
        tokens = cleaned.split()

        # Map tokens to indices (use <UNK> for OOV)
        token_indices = [vocab.get(token, unk_idx) for token in tokens]

        # Truncate to max_len-1 (before adding <CLS>)
        if len(token_indices) > max_len - 1:
            token_indices = token_indices[:max_len - 1]

        # Prepend <CLS> token
        token_indices = [cls_idx] + token_indices

        # Pad to max_len with <PAD>
        while len(token_indices) < max_len:
            token_indices.append(pad_idx)

        # Create attention mask: True where token is NOT <PAD>, False where it is <PAD>
        attention_mask = [token_idx != pad_idx for token_idx in token_indices]

        input_ids.append(token_indices)
        attention_masks.append(attention_mask)

    return input_ids, attention_masks


def encode_texts_for_lstm(
    texts: List[str],
    vocab: Dict[str, int],
    max_len: int
) -> List[List[int]]:
    """
    Encode texts for LSTM model.

    Args:
        texts: List of text strings
        vocab: Vocabulary dictionary mapping tokens to indices
        max_len: Maximum sequence length

    Returns:
        List of token index sequences (input_ids only, no masks)
    """
    input_ids = []

    unk_idx = vocab.get('<UNK>', 1)
    pad_idx = vocab.get('<PAD>', 0)

    for text in texts:
        # Clean text
        cleaned = clean_text(text)

        # Split into tokens
        tokens = cleaned.split()

        # Map tokens to indices (use <UNK> for OOV)
        token_indices = [vocab.get(token, unk_idx) for token in tokens]

        # Truncate to max_len (no <CLS> token)
        if len(token_indices) > max_len:
            token_indices = token_indices[:max_len]

        # Pad to max_len with <PAD>
        while len(token_indices) < max_len:
            token_indices.append(pad_idx)

        input_ids.append(token_indices)

    return input_ids


if __name__ == '__main__':
    # 1. Get data splits and vocabulary
    print("Loading data and building vocabulary...")
    train_df, val_df, test_df, vocab_dict = get_data_splits(max_len=256)

    print(f"\nData splits:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Vocab size: {len(vocab_dict)}")

    # 2. Encode 5 samples from train set for both models
    sample_texts = train_df['content'].head(5).tolist()

    print(f"\n{'='*60}")
    print("Example cleaned text:")
    print(f"{'='*60}")
    cleaned_sample = clean_text(sample_texts[0])
    print(cleaned_sample[:500] +
          "..." if len(cleaned_sample) > 500 else cleaned_sample)

    # Encode for Transformer
    print(f"\n{'='*60}")
    print("Transformer encoding (first sample):")
    print(f"{'='*60}")
    transformer_ids, transformer_masks = encode_texts_for_transformer(
        sample_texts[:1], vocab_dict, max_len=256
    )
    print(f"Input IDs (first 50): {transformer_ids[0][:50]}")
    print(f"Attention mask (first 50): {transformer_masks[0][:50]}")
    print(f"Sequence length: {len(transformer_ids[0])}")
    print(f"Number of non-padding tokens: {sum(transformer_masks[0])}")

    # Encode for LSTM
    print(f"\n{'='*60}")
    print("LSTM encoding (first sample):")
    print(f"{'='*60}")
    lstm_ids = encode_texts_for_lstm(sample_texts[:1], vocab_dict, max_len=256)
    print(f"Input IDs (first 50): {lstm_ids[0][:50]}")
    print(f"Sequence length: {len(lstm_ids[0])}")
    print(
        f"Number of non-padding tokens: {sum(1 for x in lstm_ids[0] if x != 0)}")

    # Encode all 5 samples
    print(f"\n{'='*60}")
    print("Encoding 5 samples for both models:")
    print(f"{'='*60}")
    transformer_ids_all, transformer_masks_all = encode_texts_for_transformer(
        sample_texts, vocab_dict, max_len=256
    )
    lstm_ids_all = encode_texts_for_lstm(sample_texts, vocab_dict, max_len=256)

    print(f"Transformer: {len(transformer_ids_all)} sequences encoded")
    print(f"LSTM: {len(lstm_ids_all)} sequences encoded")

    # 4. Save vocab to vocab.json
    print(f"\n{'='*60}")
    print("Saving vocabulary to vocab.json...")
    print(f"{'='*60}")
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
    print("Vocabulary saved successfully!")

    print(f"\n{'='*60}")
    print("Validation complete!")
    print(f"{'='*60}")
