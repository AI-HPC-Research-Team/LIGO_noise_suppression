# coding=utf-8
# Copyright (c) 2022, PengCheng Laboratory.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Most of the code here has been copied from:
#   https://github.com/NVIDIA/Megatron-LM/blob/v2.5/megatron/model/language_model.py
# with some modifications.

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.model.enums import LayerType, AttnMaskType
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model import LayerNorm

def parallel_gw_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.scatter_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    word_embeddings_weight = word_embeddings_weight.transpose(0, 1) 
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.reduce_from_tensor_model_parallel_region(logits_parallel)

def get_waveform_model(num_tokentypes, add_pooler,
                       encoder_attn_mask_type, init_method=None,
                       scaled_init_method=None, add_decoder=False,
                       decoder_attn_mask_type=AttnMaskType.causal,
                       pre_process=True, post_process=True, get_atten_value=False):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

    # GW model.
    gw_model = TransformerWaveformModel(
        init_method,
        scaled_init_method,
        encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        get_atten_value=get_atten_value
    )
    # key used for checkpoints.
    gw_model_key = 'gw_model'

    return gw_model, gw_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.kernel_size = 3

        args = get_args()
        self.dets = len(args.dets.split(','))
        assert max_sequence_length % self.dets == 0, 'Error in data composition of multiple detectors.'
        self.segs_per_det = max_sequence_length // self.dets

        # Token/Word Embedding (TE/WE).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (PE).
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size)
            # self.segs_per_det, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)

        # Conv1D embedding (CE), for local feature extraction
        self.conv_embeddings = torch.nn.Conv1d(max_sequence_length, max_sequence_length, self.kernel_size,
                            padding='same', bias=False, dtype=args.params_dtype)

        self._conv_embeddings_key = 'conv_embeddings'
        # Initialize the position embeddings.
        self.init_method(self.conv_embeddings.weight)

        self.conv_proj = torch.nn.Linear(args.segment_length, self.hidden_size, bias=False, dtype=args.params_dtype)
        self._conv_proj_key = 'conv_proj'
        # Initialize the position embeddings.
        self.init_method(self.conv_proj.weight)

        # Conv2d layer (serial), for 2D feature extraction/residual module
        self.conv_layer = torch.nn.Conv2d(self.dets, self.dets, 7, stride=1, padding=3, bias=False, dtype=args.params_dtype)
        self._conv_layer_key = 'conv_layer'
        # Initialize the position embeddings.
        self.init_method(self.conv_layer.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)
        # Embeddings activation
        self.embedding_activation = torch.nn.GELU()


    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        conv_embeddings = self.conv_embeddings(input_ids.to(self.conv_embeddings.weight.dtype))

        conv_embeddings = self.embedding_activation(conv_embeddings)

        conv_embeddings = self.conv_proj(conv_embeddings)

        embeddings = words_embeddings + position_embeddings + conv_embeddings

        conv_output = self.conv_layer(embeddings.reshape(-1, self.dets, self.segs_per_det, self.hidden_size)).reshape(embeddings.shape)
        conv_output = self.embedding_activation(conv_output)

        # residual block
        embeddings_output = embeddings + conv_output

        # Dropout.
        embeddings_output = self.embedding_dropout(embeddings_output)


        return embeddings_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] \
           = self.position_embeddings.state_dict(
               destination, prefix, keep_vars)
        state_dict_[self._conv_embeddings_key] \
           = self.conv_embeddings.state_dict(
               destination, prefix, keep_vars)
        state_dict_[self._conv_proj_key] \
           = self.conv_proj.state_dict(
               destination, prefix, keep_vars)
        state_dict_[self._conv_layer_key] \
           = self.conv_layer.state_dict(
               destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Token embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
           state_dict_ = state_dict[self._position_embeddings_key]
        else:
           # for backward compatibility.
           state_dict_ = {}
           for key in state_dict.keys():
               if 'position_embeddings' in key:
                   state_dict_[key.split('position_embeddings.')[1]] \
                       = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Conv1D embedding.
        if self._conv_embeddings_key in state_dict:
           state_dict_ = state_dict[self._conv_embeddings_key]
        else:
           # for backward compatibility.
           state_dict_ = {}
           for key in state_dict.keys():
               if 'conv_embeddings' in key:
                   state_dict_[key.split('conv_embeddings.')[1]] \
                       = state_dict[key]
        self.conv_embeddings.load_state_dict(state_dict_, strict=strict)

        if self._conv_proj_key in state_dict:
           state_dict_ = state_dict[self._conv_proj_key]
        else:
           # for backward compatibility.
           state_dict_ = {}
           for key in state_dict.keys():
               if 'conv_proj' in key:
                   state_dict_[key.split('conv_proj.')[1]] \
                       = state_dict[key]
        self.conv_proj.load_state_dict(state_dict_, strict=strict)
        
        # Conv2D layer.
        if self._conv_layer_key in state_dict:
           state_dict_ = state_dict[self._conv_layer_key]
        else:
           # for backward compatibility.
           state_dict_ = {}
           for key in state_dict.keys():
               if 'conv_layer' in key:
                   state_dict_[key.split('conv_layer.')[1]] \
                       = state_dict[key]
        self.conv_layer.load_state_dict(state_dict_, strict=strict)


class TransformerWaveformModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 encoder_attn_mask_type,
                 num_tokentypes=0,
                 add_decoder=False,
                 decoder_attn_mask_type=AttnMaskType.causal,
                 add_pooler=False,
                 pre_process=True,
                 post_process=True,
                 get_atten_value=False):
        super(TransformerWaveformModel, self).__init__()
        args = get_args()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.get_atten_value=get_atten_value

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       self.init_method,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Transformer.
        self.encoder = ParallelTransformer(
            self.init_method,
            output_layer_init_method,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            get_atten_value=self.get_atten_value
        )
        self._encoder_key = 'encoder'

        # Decoder
        if self.add_decoder:
            assert args.pipeline_model_parallel_size == 1, \
                'pipeline parallelism is not supported in the presence of decoder'
            self.decoder = ParallelTransformer(
                self.init_method,
                output_layer_init_method,
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type)
            self._decoder_key = 'decoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.encoder.set_input_tensor(input_tensor)

    def forward(self, enc_input_ids, enc_position_ids, enc_attn_mask,
                dec_input_ids=None, dec_position_ids=None, dec_attn_mask=None,
                enc_dec_attn_mask=None, tokentype_ids=None, layer_past=None,
                get_key_value=False, pooling_sequence_index=0,
                enc_hidden_states=None, output_enc_hidden=False):

        # Embeddings.
        if self.pre_process:
            embedding_output = self.embedding(enc_input_ids, enc_position_ids,
                                              tokentype_ids=tokentype_ids)
            encoder_input = embedding_output
        else:
            encoder_input = None

        # encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder(encoder_input,
                                          enc_attn_mask,
                                          layer_past=layer_past,
                                          get_key_value=get_key_value)
            # encoder_output = encoder_output + encoder_input
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output,
                                            pooling_sequence_index)

        # output_enc_hidden refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        state_dict_[self._encoder_key] \
            = self.encoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] \
                    = self.pooler.state_dict_for_save_checkpoint(
                        destination, prefix, keep_vars)
        if self.add_decoder:
            state_dict_[self._decoder_key] \
                = self.decoder.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]
        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.",
                    ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, \
                    'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)
        # decoder
        if self.add_decoder:
            assert 'decoder' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key],
                                         strict=strict)
                                         
