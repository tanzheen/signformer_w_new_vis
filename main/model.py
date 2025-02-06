# coding: utf-8
from torchvision.models import resnet18
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from itertools import groupby
from initialization import initialize_model
from embeddings import Embeddings, SpatialEmbeddings, V_encoder
from encoders import Encoder,  TransformerEncoder
from decoders import Decoder,  TransformerDecoder
from search import beam_search, greedy
from vocabulary import (
    TextVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)

from helpers import freeze_params
from torch import Tensor
from typing import Union
import torch 


class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        vis_extractor: nn.Module,
        encoder: Encoder,
        decoder: Decoder,
        sgn_embed: V_encoder,
        txt_embed: Embeddings,
        txt_vocab: TextVocabulary,
        do_recognition: bool = True,
        do_translation: bool = True,
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()
        self.vis_extractor = vis_extractor
        self.encoder = encoder
        self.decoder = decoder

        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]
        self.do_recognition = do_recognition
        self.do_translation = do_translation

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        # Print tokenized sequence
        #tokenized = [self.txt_vocab.itos[idx.item()] for idx in txt_input[0]]
        # #print("\nTokenized sequence:")
        # #print(tokenized)
        #print("============================================")
        #print("sgn shape", sgn.shape)
        #print("sgn mask shape", sgn_mask.shape)
        #print("text mask shape", txt_mask.shape)
        #print(f"txt input shape: {txt_input.shape}")
        
        # Print tokenized sequence
        tokenized = [self.txt_vocab.itos[idx.item()] for idx in txt_input[0]]
        #print("\nTokenized sequence:")
        #print(tokenized)
        
        # Print token indices
        #print("\nToken indices:")
        #print(txt_input[0].tolist())
        #print("============================================")
        # Print token indices
        #print("\nToken indices:")
        #print(txt_input[0].tolist())
        #print("============================================")

        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths
        )
        # #print("encoder_output: ", encoder_output.shape)
        # #print("encoder_hidden: ", encoder_hidden.shape)
        # #print("txt_mask : ", txt_mask.shape)
        # #print("src_mask: ", sgn_mask.shape)
        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs


    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor
    ) -> (Tensor, Tensor):
        """
        Encodes the source visuals.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        sgn = self.vis_extractor(sgn, sgn_length)
        #print("sgn after rearranging from resnet: ", sgn.shape)
        return self.encoder(
            embed_src=self.sgn_embed(src = sgn),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch,
        translation_loss_function: nn.Module,
        translation_loss_weight: float,
    ) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param translation_loss_weight: Weight for translation loss
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable
        # #print ('================================')
        # #print ('video shape: ', batch['video'].shape)
        # #print ('attention_mask shape: ', batch['attention_mask'].shape)
        # #print ('src_length shape: ', batch['src_length'].shape)
        # #print ('txt_input shape: ', batch['txt_input'].shape)
        # #print ('txt_mask shape: ', batch['txt_mask'].shape)
        # #print ('================================')
        # Do a forward pass
        decoder_outputs, last_hidden_state = self.forward(
            sgn=batch['video'].cuda(),
            sgn_mask=batch['attention_mask'].cuda(),
            sgn_lengths=batch['src_length'].cuda(),
            txt_input=batch['txt_input'].cuda(),
            txt_mask=batch['txt_mask'].cuda(),
        )
        ##print("decoder_outputs shape: ", decoder_outputs.shape)
       
        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs = decoder_outputs
            ##print("word_outputs shape: ", word_outputs.shape)
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            ##print("txt_log_probs shape: ", txt_log_probs.shape)
            ##print("batch['txt_input'] shape: ", batch['txt_input'].shape)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch['labels'].cuda())
                * translation_loss_weight
            )


        return  translation_loss

    def run_batch(
        self,
        batch,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
    
        
        encoder_output, encoder_hidden = self.encode(
            sgn=batch['video'].cuda(), sgn_mask=batch['attention_mask'].cuda(), sgn_length=batch['src_length'].cuda()
        )


        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch['attention_mask'].cuda(),
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch['attention_mask'].cuda(),
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )

def make_resnet(name='resnet18'):
    if name == 'resnet18':
        model = resnet18(pretrained= False )
    else:
        raise Exception('There are no supported resnet model {}.'.format(('resnet')))
    weights_path = "resnet18-f37072fd.pth"
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model


class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=0,batch_first=True)
        return x


def build_model(
    cfg: dict,
    sgn_dim: int,
    txt_vocab: TextVocabulary,
    multimodal: bool = False,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """
    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    # build visual encoder 
    vis_extractor = resnet()

    # Multimodal MLP for sign embeddings to match hidden size of text transforrmer
    sgn_embed: V_encoder = V_encoder(
        embedding_dim=cfg["encoder"]["embeddings"]["embedding_dim"],
        input_size=sgn_dim,
        config=cfg
    )

    # build text encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    cope= cfg.get("cope")
    #print('================================')
    #print('COPE: ', cope)
    #print('================================')
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
            cope=cope
        )

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                vocab_size=len(txt_vocab),
                emb_dropout=dec_emb_dropout,
                cope=cope
            )
    else:
        txt_embed = None
        decoder = None

    model: SignModel = SignModel(
        vis_extractor=vis_extractor, 
        encoder=encoder,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        txt_vocab=txt_vocab,
        do_translation=do_translation,
    )
    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    return model