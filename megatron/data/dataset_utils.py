# coding=utf-8
# Copyright (c) 2022, PengCheng Laboratory.  All rights reserved.
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

import os
from megatron.data.gw_dataset import GwDataset

def build_train_valid_test_datasets(data_prefix, seq_length, segment_length):

    def build_dataset(name, folder):
        dataset = GwDataset(name=name,
                    data_prefix=folder,
                    seq_length=seq_length,
                    segment_length=segment_length)
        return dataset

    train_dataset = build_dataset('train', os.path.join(data_prefix[0], 'train_data'))
    valid_dataset = [build_dataset('valid', os.path.join(data_prefix[0], 'valid_data'))]
    test_dataset = [build_dataset('test', os.path.join(data_prefix[0], 'test_data'))]
    
    return (train_dataset, valid_dataset, test_dataset)
