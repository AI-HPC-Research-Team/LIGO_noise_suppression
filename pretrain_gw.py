# coding=utf-8
# Copyright (c) 2022, PCL.  All rights reserved.
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
#   https://github.com/NVIDIA/Megatron-LM/blob/v2.5/megatron/pretrain_bert.py
# with some modifications.

"""Pretrain WaveFormer"""
from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.dataset_utils import build_train_valid_test_datasets
from megatron.model import WaveFormerModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building WaveFormer model ...')

    args = get_args()
    num_tokentypes = 2 if args.binary_head else 0

    model = WaveFormerModel(
        num_tokentypes=num_tokentypes,
        add_binary_head=args.binary_head,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process)

    return model


def get_batch(data_iterator):
    """Build the batch."""

    keys = ['noisy_signal', 'clean_signal', 'mask', 'params']
    datatype = torch.float64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    noisy_signal = data_b['noisy_signal'] #.long()
    clean_signal = data_b['clean_signal'] #.long()
    loss_mask = data_b['mask'] #.long()
    params = data_b['params'] #.long()

    return noisy_signal, clean_signal, loss_mask, params

def loss_func(loss_mask, clean_signal, output_tensor):

    denoised_signal, sop_logits = output_tensor

    loss_fn = torch.nn.MSELoss()
    lm_loss = loss_fn(denoised_signal.to(torch.float32) * loss_mask.to(torch.float32), clean_signal.to(torch.float32) * loss_mask.to(torch.float32))

    loss = lm_loss  # + lm_loss_add
    averaged_losses = average_losses_across_data_parallel_group(
        [loss])
    return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    noisy_signal, clean_signal, loss_mask, params = get_batch(data_iterator)

    # loss_mask is used to calculate loss
    padding_mask = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device)   # device='cuda:0'
    gw_labels = torch.ones(noisy_signal.shape[:2],device=noisy_signal.device) * -1
    types = torch.zeros(noisy_signal.shape[:2],device=noisy_signal.device)

    timers('batch-generator').stop()

    if not args.binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(noisy_signal, padding_mask, tokentype_ids=types,
                          gw_labels=gw_labels)

    return output_tensor, partial(loss_func, loss_mask, clean_signal)


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for BERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        seq_length=args.seq_length,
        segment_length=args.segment_length)
    
    print_rank_0("> finished creating BERT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step)


