"""
Fake News Detection - Complete Preprocessing Pipeline
=====================================================
This script implements a robust preprocessing pipeline for fake news classification
using LSTM models with advanced text cleaning to prevent data leakage.
"""

import pandas as pd
import numpy as np
import re
import pickle
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_and_merge_data(fake_path='Fake.csv', true_path='True.csv'):
    """
    Load both CSV files, create labels, combine title and text, and merge.

    Args:
        fake_path: Path to Fake.csv
        true_path: Path to True.csv

    Returns:
        Merged and shuffled DataFrame with 'combined_text' and 'label' columns
    """
    print("Loading data...")

    # Load datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Create labels: 1 for Fake, 0 for True
    fake_df['label'] = 1
    true_df['label'] = 0

    # Combine title and text columns
    # Handle NaN values by filling with empty string
    fake_df['combined_text'] = (
        fake_df['title'].fillna('') + ' ' + fake_df['text'].fillna('')
    )
    true_df['combined_text'] = (
        true_df['title'].fillna('') + ' ' + true_df['text'].fillna('')
    )

    # Merge both dataframes
    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Fake news (label=1): {df['label'].sum()}")
    print(f"True news (label=0): {(df['label'] == 0).sum()}")

    return df


def clean_text(text):
    """
    Advanced text cleaning with data leakage prevention and noise removal.

    Cleaning steps:
    1. Remove data leakage patterns (location/source like "WASHINGTON (Reuters) - ...")
    2. Remove "Reuters" mentions to prevent model from using source as feature
    3. Handle excessive punctuation by adding spaces (e.g., "What???" -> "What ? ? ?")
    4. Remove URLs, Twitter handles, and Twitter image links
    5. Convert to lowercase

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Step 1: Remove data leakage patterns
    # Pattern: "LOCATION (Source) - " or "LOCATION (Source) -" at the beginning
    # Examples: "WASHINGTON (Reuters) - ", "SEATTLE/WASHINGTON (Reuters) - "
    # This regex matches:
    # - One or more words (location) optionally separated by /
    # - Followed by (Source) where Source can be any word
    # - Followed by " - " or " -"
    text = re.sub(r'^[A-Z][A-Z\s/]+\([^)]+\)\s*-\s*',
                  '', text, flags=re.MULTILINE)

    # Step 2: Remove standalone "Reuters" mentions (case-insensitive)
    # This prevents the model from using "Reuters" as a feature
    # We replace it with space to avoid concatenating words
    text = re.sub(r'\bReuters\b', '', text, flags=re.IGNORECASE)

    # Step 3: Handle excessive punctuation by adding spaces
    # Convert multiple consecutive ! or ? into spaced versions
    # Example: "What???" -> "What ? ? ?"
    # This preserves the punctuation as distinct tokens while making them separate

    # Handle multiple exclamation marks
    text = re.sub(r'!{2,}', lambda m: ' ' + ' '.join(m.group()) + ' ', text)

    # Handle multiple question marks
    text = re.sub(r'\?{2,}', lambda m: ' ' + ' '.join(m.group()) + ' ', text)

    # Handle mixed punctuation like "!?" or "?!"
    text = re.sub(r'[!?]{2,}', lambda m: ' ' + ' '.join(m.group()) + ' ', text)

    # Step 4: Remove URLs (http, https, www)
    text = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Step 5: Remove Twitter handles (@username)
    text = re.sub(r'@\w+', '', text)

    # Step 6: Remove Twitter image links (pic.twitter.com/...)
    text = re.sub(r'pic\.twitter\.com/\w+', '', text)

    # Step 7: Convert to lowercase
    text = text.lower()

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def preprocess_data(df, vocab_size=10000, max_length=256, oov_token="<OOV>"):
    """
    Tokenize and pad sequences.

    Args:
        df: DataFrame with 'combined_text' column
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length for padding
        oov_token: Token for out-of-vocabulary words

    Returns:
        X: Padded sequences
        y: Labels
        tokenizer: Fitted tokenizer object
        df: DataFrame with cleaned_text column
    """
    print("\nPreprocessing text data...")

    # Clean all texts
    print("Cleaning texts...")
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

    # Store custom_filter for saving config later
    default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    custom_filter = default_filter.replace('!', '').replace('?', '')

    # Check for empty texts after cleaning
    empty_mask = df['cleaned_text'].str.len() == 0
    empty_count = empty_mask.sum()
    if empty_count > 0:
        print(
            f"Warning: {empty_count} texts became empty after cleaning.")
        # Optionally filter them out or keep them (they'll be all zeros after padding)
        # For now, we'll keep them but warn the user

    # Initialize tokenizer with custom filters
    # Default Keras Tokenizer filters remove: '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    # We want to KEEP '!' and '?' so we exclude them from filters
    # This allows the model to use excessive punctuation as features (important for fake news detection)

    tokenizer = Tokenizer(
        num_words=vocab_size,
        oov_token=oov_token,
        filters=custom_filter  # Custom filter excluding ! and ?
    )

    # Fit tokenizer on cleaned texts
    print("Fitting tokenizer...")
    tokenizer.fit_on_texts(df['cleaned_text'].values)

    # Convert texts to sequences
    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)

    # Pad sequences
    print(f"Padding sequences to max_length={max_length}...")
    X = pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    # Extract labels and ensure proper data types for TensorFlow
    y = df['label'].values.astype(np.int32)

    # Convert X to float32 for TensorFlow (though it's already int, this ensures consistency)
    X = X.astype(np.int32)

    # Calculate sequence length statistics before padding
    seq_lengths = [len(seq) for seq in sequences]
    print(f"\nSequence length statistics (before padding):")
    print(f"  Min: {min(seq_lengths)}")
    print(f"  Max: {max(seq_lengths)}")
    print(f"  Mean: {np.mean(seq_lengths):.1f}")
    print(f"  Median: {np.median(seq_lengths):.1f}")
    print(
        f"  Percentiles - 75th: {np.percentile(seq_lengths, 75):.1f}, 90th: {np.percentile(seq_lengths, 90):.1f}, 95th: {np.percentile(seq_lengths, 95):.1f}")

    print(f"\nVocabulary size: {len(tokenizer.word_index)}")
    print(f"X shape: {X.shape} (dtype: {X.dtype})")
    print(f"y shape: {y.shape} (dtype: {y.dtype})")

    return X, y, tokenizer, df


def main():
    """Main preprocessing pipeline."""

    # Step 1: Load and merge data
    df = load_and_merge_data('Fake.csv', 'True.csv')

    # Step 2: Preprocess (clean, tokenize, pad)
    vocab_size = 10000
    max_length = 256
    oov_token = "<OOV>"

    X, y, tokenizer, df = preprocess_data(
        df,
        vocab_size=vocab_size,
        max_length=max_length,
        oov_token=oov_token
    )

    # Step 3: Train/Validation/Test split (80/10/10)
    print("\nPerforming train/validation/test split (80/10/10)...")

    # First split: 80% train, 20% temp (which will become validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )

    # Second split: Split temp 50/50 to get 10% validation and 10% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp  # Maintain class distribution
    )

    # Step 4: Output shapes
    print("\n" + "="*60)
    print("FINAL DATA SHAPES")
    print("="*60)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(
        f"\nTraining set   - Fake news: {y_train.sum()}, True news: {(y_train == 0).sum()}")
    print(
        f"Validation set - Fake news: {y_val.sum()}, True news: {(y_val == 0).sum()}")
    print(
        f"Test set       - Fake news: {y_test.sum()}, True news: {(y_test == 0).sum()}")

    # Step 5: Print sample comparisons
    print("\n" + "="*60)
    print("SAMPLE TEXT COMPARISONS (Original vs Cleaned)")
    print("="*60)

    # Show 5 samples from each class
    for label in [0, 1]:
        label_name = "TRUE" if label == 0 else "FAKE"
        print(f"\n--- {label_name} NEWS SAMPLES ---")

        # Get indices for this label
        label_indices = df[df['label'] == label].index[:5]

        for idx in label_indices:
            original = df.loc[idx, 'combined_text']
            cleaned = df.loc[idx, 'cleaned_text']

            # Truncate for display
            orig_display = original[:200] + \
                "..." if len(original) > 200 else original
            clean_display = cleaned[:200] + \
                "..." if len(cleaned) > 200 else cleaned

            print(f"\nSample {idx} (Label: {label}):")
            print(f"Original: {orig_display}")
            print(f"Cleaned:  {clean_display}")
            print("-" * 60)

    # Step 6: Save preprocessed data and tokenizer
    print("\n" + "="*60)
    print("Saving preprocessed data and tokenizer...")
    print("="*60)

    try:
        # Get custom_filter for config (recreate it)
        default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        custom_filter = default_filter.replace('!', '').replace('?', '')

        # Save tokenizer (CRITICAL for inference)
        tokenizer_path = 'tokenizer.pkl'
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"✓ Tokenizer saved to: {tokenizer_path}")

        # Save tokenizer config as JSON for reference
        tokenizer_config = {
            'vocab_size': vocab_size,
            'max_length': max_length,
            'oov_token': oov_token,
            'word_index_size': len(tokenizer.word_index),
            'filters': custom_filter
        }
        with open('tokenizer_config.json', 'w') as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"✓ Tokenizer config saved to: tokenizer_config.json")

        # Save preprocessed arrays (optional but useful to avoid reprocessing)
        np.savez_compressed(
            'preprocessed_data.npz',
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test
        )
        print(f"✓ Preprocessed arrays saved to: preprocessed_data.npz")
        print("  (You can load with: data = np.load('preprocessed_data.npz'))")
    except Exception as e:
        print(f"⚠ Warning: Error saving files: {e}")
        print("  Data is still available in memory, but files were not saved.")

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print("\nTo use the preprocessed data in your model:")
    print("  - X_train, X_val, X_test: Input sequences")
    print("  - y_train, y_val, y_test: Labels")
    print("  - tokenizer: Tokenizer object for future text preprocessing")
    print("\nUsage:")
    print("  - Training set: Used to train the model")
    print("  - Validation set: Used for hyperparameter tuning and early stopping")
    print("  - Test set: Used ONLY for final evaluation (do not use for tuning)")
    print("\nSaved files:")
    print("  - tokenizer.pkl: Load with pickle for preprocessing new texts")
    print("  - tokenizer_config.json: Tokenizer hyperparameters")
    print("  - preprocessed_data.npz: All preprocessed arrays (optional reload)")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = main()
