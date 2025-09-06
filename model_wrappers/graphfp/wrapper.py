import torch
import graphfp_repo.mol.downstream_old as base
import graphfp_repo.mol.downstream_frag_old as frag

from hydra.utils import get_original_cwd
from os.path import join
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.utils import to_networkx
from networkx import weisfeiler_lehman_graph_hash
from graphfp_repo.mol.loader import mol_to_graph_data_obj_simple
from graphfp_repo.mol.mol_bpe import Tokenizer

from src.common.types import SmilesEmbedder

organic_major_ish = {'[C]', '[O]', '[N]', '[F]', '[Cl]', '[Br]', '[I]', '[S]', '[P]', '[B]', '[H]'}


class GraphFPWrapper(SmilesEmbedder):
    def __init__(self, path, mode):
        self._mode = mode
        model = base.GNN_graphpred(1, emb_dim=300, num_gnn_layers=5, dropout=0, graph_pooling='mean')
        check_point = torch.load(path, map_location=torch.device('cpu'))
        model.gnn.load_state_dict(
            check_point['gnn']
            if 'gnn' in check_point
            else check_point['mol_gnn']
        )
        self._model = model.gnn
        self._emb_dim = 300

    def _forward_step(self, smile):
        pyg_graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smile))
        batch = Batch.from_data_list([pyg_graph])

        node_representation = self._model(batch)
        out = global_mean_pool(node_representation, batch=batch.batch).squeeze()
        return out
        
    def safe_forward_step(self, smile):
        try:
            return self._forward_step(smile)
        except Exception as e:
            print(f"Error processing SMILES '{smile}': {e}")
            return torch.tensor([float('nan')] * self._emb_dim)

    def forward(self, smiles):
        emb = torch.stack([self.safe_forward_step(smile) for smile in smiles])
        return emb.detach().cpu().numpy()

    @property
    def name(self):
        return "GraphFP-" + self._mode
    
    @property
    def device_used(self):
        return "cpu"


class GraphFPWrapperFrag(SmilesEmbedder):
    def __init__(self, path, tokenizer_path):
        self._tokenizer = Tokenizer(tokenizer_path)
        self._vocab_dict = {smiles: i for i, smiles in enumerate(self._tokenizer.vocab_dict.keys())}

        check_points = torch.load(path, map_location=torch.device('cpu'))
        mol_model = frag.GNN_graphpred(1, emb_dim=300,
                                       num_gnn_layers=5, dropout=0, graph_pooling = 'mean')
        mol_model.gnn.load_state_dict(check_points['mol_gnn'])

        frag_model = frag.GNN_graphpred(1, emb_dim=300,
                                        num_gnn_layers=2, dropout=0, graph_pooling = 'mean', atom=False)
        frag_model.gnn.load_state_dict(check_points['frag_gnn'])
        self._frag = frag_model.gnn
        self._mol = mol_model.gnn

    def process(self, smiles_list):
        data_list = []
        for i in tqdm(range(len(smiles_list))):

            data = Data()

            smiles = smiles_list[i]
            mol = Chem.MolFromSmiles(smiles)
            data = mol_to_graph_data_obj_simple(mol)
            try:
                tree = self._tokenizer(smiles)
            except:
                print("Unable to process SMILES:", smiles)
                continue

            # Manually consructing the fragment graph
            map = [0] * data.num_nodes
            frag = [[0] for _ in range(len(tree.nodes))]
            frag_edge_index = [[], []]

            try:
                for node_i in tree.nodes:
                    node = tree.get_node(node_i)
                    # for atom in node, set map
                    for atom_i in node.atom_mapping.keys():
                        map[atom_i] = node_i
                        # extend frag
                        frag[node_i][0] = self._vocab_dict[node.smiles]
                for src, dst in tree.edges:
                    # extend edge index
                    frag_edge_index[0].extend([src, dst])
                    frag_edge_index[1].extend([dst, src])
            except KeyError as e:
                print("Error in matching subgraphs", e)
                continue

            unique_frag = torch.LongTensor(list(set([frag[i][0] for i in range(len(frag))])))
            frag_unique = torch.zeros(3200).index_fill_(0, unique_frag, 1).type(torch.LongTensor)

            data.map = torch.LongTensor(map)
            data.frag = torch.LongTensor(frag)
            data.frag_edge_index = torch.LongTensor(frag_edge_index)
            data.frag_unique = frag_unique

            data_list.append(data)

        tree_dict = {}
        hash_str_list = []
        for data in data_list:
            tree = Data()
            tree.x = data.frag
            tree.edge_index = data.frag_edge_index
            nx_graph = to_networkx(tree, to_undirected=True)
            hash_str = weisfeiler_lehman_graph_hash(nx_graph)
            if hash_str not in tree_dict:
                tree_dict[hash_str] = len(tree_dict)
            hash_str_list.append(hash_str)

        tree = []
        for hash_str in hash_str_list:
            tree.append(tree_dict[hash_str])

        for i, data in enumerate(data_list):
            data.tree = tree[i]

        return data_list

    def _forward_step(self, frg, smile):
        pyg_graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smile))
        batch = Batch.from_data_list([pyg_graph])
        frg = Batch.from_data_list([frg])

        mol_embedding = global_mean_pool(self._mol(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
        ), batch=batch.batch).squeeze()

        print(f"Smile: {smile}, frag {frg}")

        try:
            frag_node_embedding = self._frag(
                frg.frag.squeeze(),
                frg.frag_edge_index,
                None
            )
        except Exception as e:
            print("Error in fragment embedding, SMILES", smile)
            # print("Cause: ", e)
            # print("Frag: ", frg)
            return None

        frag_batch = torch.zeros(frag_node_embedding.size(0), dtype=torch.long)

        frag_embedding = global_mean_pool(frag_node_embedding, batch=frag_batch).squeeze()
        return torch.cat([mol_embedding, frag_embedding])

    def forward(self, smiles):
        smiles = smiles[:500]
        frags = self.process(smiles)
        emb = [self._forward_step(frg, smile) for frg, smile in zip(frags, smiles)]
        emb = torch.stack([x for x in emb if x is not None])
        # emb = torch.stack([
        #     self._forward_step(data, smile)
        #     for data, smile in zip(frags, smiles)])
        return emb.detach().cpu().numpy()

    @property
    def name(self):
        return "GraphFP-CPF"
    
    @property
    def device_used(self):
        return "cpu"  # GraphFP does not use GPU, so we return "cpu"


def get_embedder(name, task, **_kwargs):
    if 'CPF' in name:
        model_path = join(get_original_cwd(), "model_wrappers/graphfp/graphfp_repo/mol/pretrain/GIN_CPF_01.pth")
        tokenizer_path = join(get_original_cwd(), "model_wrappers/graphfp/graphfp_repo/mol/vocab.txt")
        return GraphFPWrapperFrag(model_path, tokenizer_path)
    elif 'graphfp' in name.lower():
        mode = name.split('-')[1]
        if mode == 'C':
            model_filename = 'GIN_C.pth'
        else:
            model_filename = 'GIN_CP_03.pth'
        model_path = join(get_original_cwd(), "model_wrappers/graphfp/graphfp_repo/mol/pretrain", model_filename)
        return GraphFPWrapper(model_path, mode)
    else:
        raise ValueError(f"Unknown model name: {name}")
