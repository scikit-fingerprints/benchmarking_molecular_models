#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
Featurizer for the retro-syn model.
"""
import numpy as np 
import pgl
import random
from pahelix.utils.compound_tools import lite_gem_mol2graph


class RetroSynTransformFn(object):
    """tbd"""
    def __init__(self):
        pass

    def __call__(self, raw_data):
        """
        Gen features according to raw data and return a single graph data.
        Args:
            raw_data: It contains smiles and label,we convert smiles 
            to mol by rdkit,then convert mol to graph data.
        Returns:
            data: It contains reshape label and smiles.
        """
        smiles1 = raw_data["reactant"]
        smiles3 = raw_data["production"]
        data = {}
        data['reactant'] = lite_gem_mol2graph(smiles1)
        # print(f"reactant data: {data}")
        data['reactant']['smiles'] = smiles1
        data['production'] = lite_gem_mol2graph(smiles3)
        data['production']['smiles'] = smiles3
        data['label'] = raw_data["label"]
        data['smiles'] = smiles1
        return data


class RetroSynCollateFn(object):
    """tbd"""
    def __init__(self, atom_names, bond_names):
        self.atom_names = atom_names
        self.bond_names = bond_names

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def construct_graph(self, data):
        """tbd"""
        n = len(data[self.atom_names[0]])
        E = len(data['edges'])
        g = pgl.graph.Graph(num_nodes=n,
                    edges = data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names})
        return g

    def __call__(self, batch_data_list):
        """tbd"""
        rectant_graph_list = []
        production_graph_list = []
        #  full_graph_list = []
        labels = []
        for data in batch_data_list:
            reactant_data = data['reactant']
            production_data = data['production']
            reactant_g = self.construct_graph(reactant_data)
            production_g = self.construct_graph(production_data)

            # graph_list.append(g)
            rectant_graph_list.append(reactant_g)
            production_graph_list.append(production_g)

            labels.append(data["label"])
            
        labels = np.array(labels, dtype="float32")
        # g = pgl.Graph.batch(graph_list)
        batced_rectant_g = pgl.Graph.batch(rectant_graph_list)
        self._flat_shapes(batced_rectant_g.node_feat)
        self._flat_shapes(batced_rectant_g.edge_feat)
        batched_production_g = pgl.Graph.batch(production_graph_list)
        self._flat_shapes(batched_production_g.node_feat)
        self._flat_shapes(batched_production_g.edge_feat)
        graph_dict = {'reactant_g':batced_rectant_g,
            'product_g':batched_production_g}
        # others = {'smiles': smiles_list}
        return graph_dict, labels


class RetroSynReshuffleCollateFn(object):
    """tbd"""
    def __init__(self, atom_names, bond_names):
        self.atom_names = atom_names
        self.bond_names = bond_names

    def _flat_shapes(self, d):
        """TODO: reshape due to pgl limitations on the shape"""
        for name in d:
            d[name] = d[name].reshape([-1])

    def gen_shuffled_list(self, l1, l2):
        """
        Generate a shuffled version of list l2
        Args:
            l1: first list of data
            l2: second list of data, having the same shape as l1
        
        Returns:
            l1: the original l1
            l2: shuffeld l2, such that each element of l1 is paired with a different element
        """
        used_index = []
        shuffled_l1 = l1
        shuffled_l2 = []
        for i in range(len(l1)):
            n = random.randint(0, len(l1)-1)
            while n in used_index:
                n = random.randint(0, len(l1)-1)
                if n == i:
                    continue
            shuffled_l2.append(l2[n])
        return shuffled_l1, shuffled_l2

    def construct_graph(self, data):
        """tbd"""
        n = len(data[self.atom_names[0]])
        E = len(data['edges'])
        g = pgl.graph.Graph(num_nodes=n,
                    edges = data['edges'],
                    node_feat={name: data[name].reshape([-1, 1]) for name in self.atom_names},
                    edge_feat={name: data[name].reshape([-1, 1]) for name in self.bond_names})
        return g

    def __call__(self, batch_data_list):
        """tbd"""
        rectant_graph_list = []
        production_graph_list = []
        labels = []
        neg_labels = []
        for data in batch_data_list:
            reactant_data = data['reactant']
            production_data = data['production']
            reactant_g = self.construct_graph(reactant_data)
            production_g = self.construct_graph(production_data)

            rectant_graph_list.append(reactant_g)
            production_graph_list.append(production_g)

            labels.append(data["label"])
            neg_labels.append(0)            
        labels = np.array(labels + neg_labels, dtype="float32")
        neg_rectant_graph_list, neg_production_graph_list = self.gen_shuffled_list(rectant_graph_list,
                                                                                production_graph_list)
        batced_rectant_g = pgl.Graph.batch(rectant_graph_list + neg_rectant_graph_list)
        self._flat_shapes(batced_rectant_g.node_feat)
        self._flat_shapes(batced_rectant_g.edge_feat)
        batched_production_g = pgl.Graph.batch(production_graph_list + neg_production_graph_list)
        self._flat_shapes(batched_production_g.node_feat)
        self._flat_shapes(batched_production_g.edge_feat)
        graph_dict = {'reactant_g':batced_rectant_g,
            'product_g':batched_production_g}

        return graph_dict, labels