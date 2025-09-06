
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

"""
| Tools for data.
"""

import numpy as np
import random
from paddle.io import IterableDataset


class DynaBatchDataset(IterableDataset):
    """
    DynaDataset that yields flexible batch data depending on the length
    """
    def __init__(self, dataset, collate_fn=None, boundaries=None, batch_sizes=None):
        self.dataset = dataset
        
        if collate_fn is None:
            self.collate_fn = dataset.collate_fn
        else:
            self.collate_fn = collate_fn
        if boundaries is None:
            self.boundaries = dataset.get_boundaries()
        else:
            self.boundaries = boundaries
        if batch_sizes is None:
            self.batch_sizes = dataset.get_batch_sizes()
        else:
            self.batch_sizes = batch_sizes
    
    def __iter__(self):
        buckets = [[] for i in range(len(self.batch_sizes))]
        for length, data in self.dataset:
            cur_bucket = np.digitize(length, self.boundaries)
            buckets[cur_bucket].append(data)
            if len(buckets[cur_bucket]) == self.batch_sizes[cur_bucket]:
                bucket = self.collate_fn(buckets[cur_bucket])
                yield bucket
                buckets[cur_bucket] = []

        for bucket in buckets:
            if bucket:
                bucket = self.collate_fn(bucket)
                yield bucket


class DynaDataset(object):
    """tbd"""
    def get_boundaries(self):
        """tbd"""
        raise NotImplementedError()

    def get_batch_sizes(self):
        """tbd"""
        raise NotImplementedError()

    def __iter__(self):
        """tbd"""
        raise NotImplementedError()
    
    def collate_fn(self, data_list):
        """tbd"""
        raise NotImplementedError()
    