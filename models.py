"""
PyTorch models for Fake News Detection project.
No pretrained weights - everything randomly initialized.
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for fake news detection.

    Architecture:
    - Embedding layer
    - Multi-layer LSTM
    - Packed sequences for efficient processing
    - Final hidden state → classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize LSTMClassifier.

        Args:
            vocab_size: Size of vocabulary (from vocab.json)
            embed_dim: Embedding dimension (tunable HP: 64, 128, 256)
            hidden_size: LSTM hidden size (tunable HP: 128, 256, 512)
            num_layers: Number of LSTM layers (tunable HP: 1, 2, 3)
            dropout: Dropout rate (default: 0.1)
        """
        super(LSTMClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer with padding_idx=0 (<PAD> token)
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        # LSTM layer
        # dropout is only applied between layers if num_layers > 1
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch_size, max_len]
            lengths: Original sequence lengths [batch_size]

        Returns:
            Logits [batch_size, 1] (raw logits, no sigmoid)
        """
        # 1. Embed tokens
        embedded = self.embedding(input_ids)  # [B, L, embed_dim]

        # 2. Clamp lengths to minimum 1 to avoid errors
        lengths = torch.clamp(lengths, min=1)

        # 3. Pack padded sequences (must use .cpu() for lengths)
        # enforce_sorted=False allows unsorted sequences
        packed = rnn_utils.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # 4. Pass through LSTM
        # hn: [num_layers, B, hidden_size] - hidden states of last layer
        # cn: [num_layers, B, hidden_size] - cell states (not used)
        _, (hn, _) = self.lstm(packed)

        # 5. Extract last hidden state from final layer
        last_hidden = hn[-1]  # [B, hidden_size]

        # 6. Classify
        return self.classifier(last_hidden)  # [B, 1]


class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for fake news detection.

    Architecture:
    - Embedding layer
    - Learned positional encoding
    - Multi-layer Transformer encoder
    - [CLS] token representation → classifier
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_len: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize TransformerClassifier.

        Args:
            vocab_size: Size of vocabulary (from vocab.json)
            d_model: Model dimension (tunable HP: 128, 256, 512; must be divisible by nhead)
            nhead: Number of attention heads (tunable HP: 2, 4, 8)
            num_layers: Number of encoder layers (tunable HP: 2, 4, 6)
            dim_feedforward: Feedforward dimension (tunable HP: 512, 1024, 2048)
            max_len: Maximum sequence length (default: 256)
            dropout: Dropout rate (default: 0.1)
        """
        super(TransformerClassifier, self).__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.dropout = dropout

        # Validate d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        # Embedding layer with padding_idx=0 (<PAD> token)
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=0
        )

        # Learned positional encoding
        # Shape: [1, max_len, d_model] - batch dimension for broadcasting
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, d_model)
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input: [B, L, d_model]
            norm_first=False   # Apply norm after attention/FFN
        )

        # Transformer encoder (stack of encoder layers)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [batch_size, max_len] (starts with <CLS>=2 at position 0)
            attention_mask: Padding mask [batch_size, max_len] (True=padding position to ignore)

        Returns:
            Logits [batch_size, 1] (raw logits, no sigmoid)
        """
        # 1. Embed tokens (minimal scaling to prevent NaN)
        # No scaling - let the model learn naturally
        x = self.embedding(input_ids)

        # 2. Add positional encoding (truncate to input length)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]

        # 3. Apply transformer encoder
        # src_key_padding_mask: True for positions to ignore (padding)
        x = self.transformer_encoder(x, src_key_padding_mask=attention_mask)

        # 4. Use [CLS] token representation (always at position 0)
        cls_repr = x[:, 0, :]  # [B, d_model]

        # 5. Classify
        return self.classifier(cls_repr)  # [B, 1]


if __name__ == '__main__':
    """
    Validation and testing code.
    """
    print("=" * 60)
    print("Testing LSTMClassifier")
    print("=" * 60)

    # Test LSTM model
    vocab_size = 10000
    batch_size = 4
    max_len = 256

    lstm_model = LSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.1
    )

    # Create dummy input
    input_ids_lstm = torch.randint(0, vocab_size, (batch_size, max_len))
    lengths_lstm = torch.tensor([256, 128, 64, 32], dtype=torch.long)

    print(f"Input shape: {input_ids_lstm.shape}")
    print(f"Lengths: {lengths_lstm.tolist()}")

    # Forward pass
    with torch.no_grad():
        output_lstm = lstm_model(input_ids_lstm, lengths_lstm)

    print(f"Output shape: {output_lstm.shape}")
    print(f"Output dtype: {output_lstm.dtype}")
    print(f"Output sample: {output_lstm[0].item():.4f}")
    print("✓ LSTMClassifier test passed\n")

    print("=" * 60)
    print("Testing TransformerClassifier")
    print("=" * 60)

    # Test Transformer model
    transformer_model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        max_len=max_len,
        dropout=0.1
    )

    # Create dummy input
    # First token should be <CLS>=2, rest can be random
    input_ids_trans = torch.randint(0, vocab_size, (batch_size, max_len))
    input_ids_trans[:, 0] = 2  # Set first token to <CLS>

    # Create attention mask (True = padding)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    attention_mask[0, 200:] = True  # Some padding in first sample
    attention_mask[1, 150:] = True  # Some padding in second sample
    attention_mask[2, 100:] = True  # Some padding in third sample
    attention_mask[3, 50:] = True   # Some padding in fourth sample

    print(f"Input shape: {input_ids_trans.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print(f"First token (should be <CLS>=2): {input_ids_trans[0, 0].item()}")
    print(
        f"Padding positions in first sample: {attention_mask[0].sum().item()}")

    # Forward pass
    with torch.no_grad():
        output_trans = transformer_model(input_ids_trans, attention_mask)

    print(f"Output shape: {output_trans.shape}")
    print(f"Output dtype: {output_trans.dtype}")
    print(f"Output sample: {output_trans[0].item():.4f}")
    print("✓ TransformerClassifier test passed\n")

    # Test GPU compatibility (if available)
    if torch.cuda.is_available():
        print("=" * 60)
        print("Testing GPU compatibility")
        print("=" * 60)

        device = torch.device('cuda')
        lstm_model_gpu = lstm_model.to(device)
        transformer_model_gpu = transformer_model.to(device)

        input_ids_lstm_gpu = input_ids_lstm.to(device)
        lengths_lstm_gpu = lengths_lstm.to(device)
        input_ids_trans_gpu = input_ids_trans.to(device)
        attention_mask_gpu = attention_mask.to(device)

        with torch.no_grad():
            output_lstm_gpu = lstm_model_gpu(
                input_ids_lstm_gpu, lengths_lstm_gpu)
            output_trans_gpu = transformer_model_gpu(
                input_ids_trans_gpu, attention_mask_gpu)

        print(f"LSTM GPU output shape: {output_lstm_gpu.shape}")
        print(f"Transformer GPU output shape: {output_trans_gpu.shape}")
        print("✓ GPU compatibility test passed\n")
    else:
        print("=" * 60)
        print("GPU not available, skipping GPU tests")
        print("=" * 60)

    # Test edge cases
    print("=" * 60)
    print("Testing edge cases")
    print("=" * 60)

    # LSTM: Test with length=1
    lengths_edge = torch.tensor([1, 1], dtype=torch.long)
    input_ids_edge = torch.randint(0, vocab_size, (2, max_len))
    with torch.no_grad():
        output_edge = lstm_model(input_ids_edge, lengths_edge)
    print(f"LSTM with length=1: output shape {output_edge.shape} ✓")

    # Transformer: Test with all padding in one sample
    attention_mask_edge = torch.ones(2, max_len, dtype=torch.bool)
    attention_mask_edge[0, :10] = False  # First sample has 10 real tokens
    input_ids_trans_edge = torch.randint(0, vocab_size, (2, max_len))
    input_ids_trans_edge[:, 0] = 2  # <CLS> token
    with torch.no_grad():
        output_trans_edge = transformer_model(
            input_ids_trans_edge, attention_mask_edge)
    print(
        f"Transformer with heavy padding: output shape {output_trans_edge.shape} ✓")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
