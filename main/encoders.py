import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from helpers import freeze_params
from transformer_layers import CA_TransformerEncoderLayer
from position_encoding import PositionalEncoding


# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """

    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        hidden_size: int = 512,
        ff_size: int = 2048,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        freeze: bool = False,
        cope=False,
        **kwargs
    ):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                CA_TransformerEncoderLayer(
                    encoder_dim=hidden_size,
                    num_attention_heads=num_heads,
                    feed_forward_expansion_factor=ff_size//hidden_size,
                    conv_expansion_factor=2,
                    attention_dropout_p=dropout,
                    conv_dropout_p=dropout,
                    half_step_residual= True,
                    cope=cope
                )
                for _ in range(num_layers)
            ])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.pe = PositionalEncoding(hidden_size)
        self._output_size = hidden_size
        if freeze:
            freeze_params(self)

    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
        """
        Forward pass of the Transformer encoder.
        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, hidden_size),
            - hidden_concat: last hidden state with
                shape (batch_size, hidden_size)
        """
        # add positional encoding to the input
        embed_src = self.pe(embed_src)
        # apply dropout to the input
        embed_src = self.emb_dropout(embed_src)

        # apply the Transformer layers
        for layer in self.layers:
            embed_src = layer(embed_src, mask)

        # apply layer normalization
        embed_src = self.layer_norm(embed_src)

        # only feed the final state of the top-most layer to the decoder
        hidden_concat = embed_src[:, 0, :]
        return embed_src, hidden_concat