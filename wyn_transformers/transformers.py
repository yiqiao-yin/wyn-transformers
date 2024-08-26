import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization


def scaled_dot_product_attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor = None
) -> tuple:
    """
    Calculate the attention weights and outputs using scaled dot-product attention.

    Args:
        query (tf.Tensor): The query tensor of shape (..., seq_len_q, depth).
        key (tf.Tensor): The key tensor of shape (..., seq_len_k, depth).
        value (tf.Tensor): The value tensor of shape (..., seq_len_v, depth_v).
        mask (tf.Tensor, optional): A mask tensor that prevents attention to certain positions. Default is None.

    Returns:
        output (tf.Tensor): The output of the attention mechanism, a weighted sum of values.
        attention_weights (tf.Tensor): The attention weights for each query-key pair.
    """
    # Compute the dot product between query and key
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # Scale the dot products by the square root of the depth of the key
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # Apply the mask (if provided) by adding a large negative number to the logits where the mask is 1
    if mask is not None:
        logits += mask * -1e9

    # Apply softmax to compute the attention weights
    attention_weights = tf.nn.softmax(logits, axis=-1)

    # Compute the output by applying the attention weights to the value tensor
    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-Head Attention layer that computes attention for multiple heads in parallel.

    Args:
        d_model (int): The dimensionality of the model.
        num_heads (int): The number of attention heads.

    Attributes:
        num_heads (int): The number of attention heads.
        d_model (int): The dimensionality of the model.
        depth (int): The dimensionality of each attention head.
        wq (Dense): Dense layer for projecting queries.
        wk (Dense): Dense layer for projecting keys.
        wv (Dense): Dense layer for projecting values.
        dense (Dense): Dense layer for output projection.
    """

    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        # Depth of each attention head
        self.depth = d_model // self.num_heads

        # Dense layers to project queries, keys, and values
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        # Final dense layer after concatenation of all heads
        self.dense = Dense(d_model)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Split the last dimension of a tensor into (num_heads, depth), and transpose the result.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            batch_size (int): The batch size.

        Returns:
            tf.Tensor: Transposed tensor of shape (batch_size, num_heads, seq_len, depth).
        """
        # Reshape the input tensor to separate heads
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # Transpose to bring the num_heads dimension forward
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Perform the forward pass for the multi-head attention layer.

        Args:
            v (tf.Tensor): Value tensor of shape (batch_size, seq_len_v, d_model).
            k (tf.Tensor): Key tensor of shape (batch_size, seq_len_k, d_model).
            q (tf.Tensor): Query tensor of shape (batch_size, seq_len_q, d_model).
            mask (tf.Tensor, optional): A mask tensor that prevents attention to certain positions. Default is None.

        Returns:
            tf.Tensor: The output tensor after applying multi-head attention.
        """
        batch_size = tf.shape(q)[0]

        # Project queries, keys, and values using dense layers
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Split the projected queries, keys, and values into multiple heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Compute scaled dot-product attention for each head
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)

        # Transpose and reshape to concatenate all heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # Apply final dense layer to concatenated attention outputs
        output = self.dense(concat_attention)

        return output


def point_wise_feed_forward_network(d_model: int, dff: int) -> tf.keras.Sequential:
    """
    Create a point-wise feed-forward network.

    Args:
        d_model (int): The dimensionality of the model.
        dff (int): The dimensionality of the feed-forward network hidden layer.

    Returns:
        tf.keras.Sequential: A sequential model consisting of two dense layers.
    """
    # Sequential model with two dense layers
    return tf.keras.Sequential(
        [
            Dense(dff, activation="relu"),  # First dense layer with ReLU activation
            Dense(d_model),  # Second dense layer to project back to d_model
        ]
    )


def positional_encoding(position: int, d_model: int) -> tf.Tensor:
    """
    Generate positional encoding for input sequences.

    Args:
        position (int): The maximum length of the input sequences.
        d_model (int): The dimensionality of the model.

    Returns:
        tf.Tensor: The positional encoding tensor of shape (1, position, d_model).
    """
    # Calculate the angles for positional encoding
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # Apply sine to even indices in the array and cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Add batch dimension and convert to tensor
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> np.ndarray:
    """
    Compute the angle rates for positional encoding.

    Args:
        pos (np.ndarray): The positions.
        i (np.ndarray): The dimensions.
        d_model (int): The dimensionality of the model.

    Returns:
        np.ndarray: The computed angle rates for positional encoding.
    """
    # Compute angle rates using the formula for positional encoding
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization

# Assuming the following functions and classes are defined earlier in your code:
# MultiHeadAttention, point_wise_feed_forward_network, positional_encoding


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer for the Transformer model.

    This layer consists of a multi-head self-attention mechanism followed by a point-wise feed-forward network.
    Each of these components is followed by a layer normalization step and a dropout layer for regularization.

    Args:
        d_model (int): The dimensionality of the output space.
        num_heads (int): The number of attention heads.
        dff (int): The dimensionality of the feed-forward network's hidden layer.
        rate (float, optional): Dropout rate. Default is 0.1.

    Attributes:
        mha (MultiHeadAttention): Multi-head attention layer.
        ffn (tf.keras.Sequential): Point-wise feed-forward network.
        layernorm1 (LayerNormalization): Layer normalization applied after attention.
        layernorm2 (LayerNormalization): Layer normalization applied after feed-forward network.
        dropout1 (Dropout): Dropout layer applied after attention.
        dropout2 (Dropout): Dropout layer applied after feed-forward network.
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super(EncoderLayer, self).__init__()

        # Initialize multi-head attention and feed-forward network
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        # Layer normalization and dropout layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(
        self, x: tf.Tensor, training: bool = False, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Forward pass for the encoder layer.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_seq_len, d_model).
            training (bool, optional): Boolean to indicate if the model is in training mode. Default is False.
            mask (tf.Tensor, optional): Mask tensor to avoid paying attention to certain positions. Default is None.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, input_seq_len, d_model).
        """
        # Apply multi-head attention to the input
        attn_output = self.mha(x, x, x, mask)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)  # Apply dropout
        out1 = self.layernorm1(
            x + attn_output
        )  # Residual connection followed by layer normalization

        # Apply point-wise feed-forward network to the output of attention
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # Apply dropout
        out2 = self.layernorm2(
            out1 + ffn_output
        )  # Residual connection followed by layer normalization

        return out2


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder consisting of a stack of EncoderLayers.

    The encoder is responsible for processing the input sequence to produce contextualized representations.
    It consists of an embedding layer to convert token indices to dense vectors, a positional encoding to inject
    position information, and multiple encoder layers.

    Args:
        num_layers (int): Number of encoder layers.
        d_model (int): The dimensionality of the output space.
        num_heads (int): The number of attention heads.
        dff (int): The dimensionality of the feed-forward network's hidden layer.
        input_vocab_size (int): The size of the input vocabulary.
        maximum_position_encoding (int): The maximum length of input sequences.
        rate (float, optional): Dropout rate. Default is 0.1.

    Attributes:
        d_model (int): The dimensionality of the output space.
        num_layers (int): Number of encoder layers.
        embedding (Embedding): Embedding layer to convert token indices to dense vectors.
        pos_encoding (tf.Tensor): Positional encoding for input sequences.
        enc_layers (List[EncoderLayer]): List of encoder layers.
        dropout (Dropout): Dropout layer applied after positional encoding.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        maximum_position_encoding: int,
        rate: float = 0.1,
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding layer for input token indices and positional encoding for sequence information
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        # Stack of encoder layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = Dropout(rate)

    def call(
        self, x: tf.Tensor, training: bool = False, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Forward pass for the Transformer encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_seq_len).
            training (bool, optional): Boolean to indicate if the model is in training mode. Default is False.
            mask (tf.Tensor, optional): Mask tensor to avoid paying attention to certain positions. Default is None.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, input_seq_len, d_model).
        """
        seq_len = tf.shape(x)[1]

        # Apply embedding and scale by square root of d_model
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # Add positional encoding to the embeddings
        x += self.pos_encoding[:, :seq_len, :]

        # Apply dropout to the input embeddings
        x = self.dropout(x, training=training)

        # Pass the input through each encoder layer
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)

        return x


import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class TransformerModel(tf.keras.Model):
    """
    Transformer Model for sequence-to-sequence tasks.

    This model is built using an encoder architecture with multiple layers of self-attention and feed-forward networks.
    It is designed to handle natural language processing tasks such as translation, text generation, and more.

    Args:
        num_layers (int): The number of encoder layers in the Transformer.
        d_model (int): The dimensionality of the embedding space.
        num_heads (int): The number of attention heads in the multi-head attention mechanism.
        dff (int): The dimensionality of the feed-forward network hidden layer.
        input_vocab_size (int): The size of the input vocabulary.
        maximum_position_encoding (int): The maximum length of the input sequences.
        rate (float, optional): Dropout rate for regularization. Default is 0.1.
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        encoder (Encoder): The Transformer encoder composed of stacked layers.
        final_layer (Dense): Final dense layer to project the encoder outputs to the input vocabulary size.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dff: int,
        input_vocab_size: int,
        maximum_position_encoding: int,
        rate: float = 0.1,
        **kwargs
    ):
        # Pass any additional keyword arguments to the parent class
        super(TransformerModel, self).__init__(**kwargs)

        # Store the model parameters
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate

        # Define the encoder and final dense layer
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
            rate,
        )
        self.final_layer = tf.keras.layers.Dense(input_vocab_size)

    def call(
        self, x: tf.Tensor, training: bool = False, mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            x (tf.Tensor): Input tensor of shape (batch_size, input_seq_len).
            training (bool, optional): Boolean to indicate if the model is in training mode. Default is False.
            mask (tf.Tensor, optional): Mask tensor to avoid paying attention to certain positions. Default is None.

        Returns:
            tf.Tensor: Output tensor of shape (batch_size, input_seq_len, input_vocab_size).
        """
        # Pass the input through the encoder
        enc_output = self.encoder(x, training=training, mask=mask)

        # Apply the final dense layer to get the output logits
        final_output = self.final_layer(enc_output)
        return final_output

    def get_config(self) -> dict:
        """
        Returns the configuration of the model for serialization.

        Returns:
            dict: Configuration dictionary containing model parameters.
        """
        # Get the base configuration from the parent class
        config = super(TransformerModel, self).get_config()
        # Update the configuration with model-specific parameters
        config.update(
            {
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "input_vocab_size": self.input_vocab_size,
                "maximum_position_encoding": self.maximum_position_encoding,
                "rate": self.rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> "TransformerModel":
        """
        Creates a TransformerModel instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            TransformerModel: A new instance of TransformerModel initialized from the configuration.
        """
        return cls(**config)
