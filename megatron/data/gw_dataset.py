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

"""gravitational waveform dataset."""

import numpy as np
import torch
import h5py
import os
from megatron import (
    mpu,
)

class GwDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, seq_length, segment_length, seed=1234):

        self.name = name
        self.seed = seed

        self.samples = get_samples(data_prefix, self.name)
        self.noisy = self.samples['noisy']
        self.clean = self.samples['clean']
        self.params = self.samples['params']
        assert self.noisy.shape == self.clean.shape
        self.step = segment_length
        self.patches = seq_length

    def __len__(self):
        return self.noisy.shape[0]

    def __getitem__(self, idx):
        tmp_idx = idx % self.noisy.shape[0]
        noisy_np = self.noisy[tmp_idx]
        clean_np = self.clean[tmp_idx]
        param_np = np.real(self.params[tmp_idx].reshape(1, -1))
        noisy_input = np.zeros([self.patches, self.step], dtype=self.noisy.dtype)
        clean_input = np.zeros([self.patches, self.step], dtype=self.clean.dtype)
        for ind in range(self.patches):
            noisy_input[ind] = noisy_np[0, ind * self.step : (ind+1) * self.step]
            clean_input[ind] = clean_np[0, ind * self.step : (ind+1) * self.step]
        
        mask_input = np.ones(noisy_input.shape, dtype=noisy_input.dtype)
        train_sample = {
            'name' : self.name,
            'noisy_signal': noisy_input,
            'clean_signal': clean_input,
            'mask': mask_input,
            'params': param_np}

        return train_sample

# modify this function based on your own dataset storage
def get_samples(data_prefix, name):
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    if name in ["valid", "test"]:
        data_path = os.path.join(data_prefix, name + '.hdf5')
        f_data = h5py.File(data_path, 'r')
        dataset = {}
        for data_name in ['noisy', 'clean']:
            dataset[data_name] = f_data[data_name][:, :, :]
        dataset['params'] = f_data['params'][:, :]
        f_data.close()
    else:
        dataset = {}
        for i in range(1, 11):
            data_path = os.path.join(data_prefix, "{}_{}.hdf5".format(name, i))
            f_data = h5py.File(data_path, 'r')
            if i == 1:
                for data_name in ['noisy', 'clean']:
                    dataset[data_name] = f_data[data_name][:, :, :]
                dataset['params'] = f_data['params'][:, :]
            else:
                for data_name in ['noisy', 'clean']:
                    dataset[data_name] = np.append(dataset[data_name], f_data[data_name][:, :, :], axis=0)
                dataset['params'] = np.append(dataset['params'], f_data['params'][:, :], axis=0)
            f_data.close()

    return dataset

