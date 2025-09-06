
#!/usr/bin/env python3                                                                                       
#-*-coding:utf-8-*- 
#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
mol utils
"""
import numpy as np
import copy
import gzip
import pickle
import logging
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import SanitizeFlags
from rdkit.Geometry import Point3D
import os


def load_protein(path):
    """tbd"""
    mol = None
    if path.endswith('.pdb'):
        try:
            mol = Chem.MolFromPDBFile(path, sanitize=False)
            mol = Chem.RemoveHs(mol, sanitize=False)
        except Exception as e:
            logging.info(f'Failed to load protein {path}: {e}')
            print(f'Failed to load protein {path}: {e}')
    elif path.endswith('rdmol.pkl.gz'):
        try:
            with gzip.open(path, 'rb') as f:
                mol = pickle.load(f)
        except Exception as e:
            logging.info(f'Failed to load protein {path}: {e}')
    else:
        raise ValueError(f'invalid suffix: {path}')
    return mol


def load_ligand(path):
    """tbd"""
    mol = None
    if path.endswith(".mol2_obabel") or path.endswith(".mol2"):
        try:
            mol = Chem.MolFromMol2File(path, sanitize=False)
            mol = Chem.RemoveHs(mol, sanitize=False)
            try:
                Chem.SanitizeMol(mol, \
                        SanitizeFlags.SANITIZE_SETCONJUGATION | \
                        SanitizeFlags.SANITIZE_SETHYBRIDIZATION | \
                        SanitizeFlags.SANITIZE_FINDRADICALS)
                Chem.AssignStereochemistryFrom3D(mol)
            except:
                print('Sanitize fail on ', path)
                pass
        except Exception as e:
            logging.info(f'Failed to load ligand {path}: {e}')
    elif path.endswith(".pdb"):
        try:
            mol = Chem.MolFromPDBFile(path, sanitize=False)
            mol = Chem.RemoveHs(mol, sanitize=False)
            try:
                Chem.SanitizeMol(mol, \
                        SanitizeFlags.SANITIZE_SETCONJUGATION | \
                        SanitizeFlags.SANITIZE_SETHYBRIDIZATION | \
                        SanitizeFlags.SANITIZE_FINDRADICALS)
                Chem.AssignStereochemistryFrom3D(mol)
            except:
                pass
        except Exception as e:
            print(f'Failed to load pdb ligand {path}: {e}')
    elif path.endswith(".sdf"):
        try:
            sdf_supplier = Chem.SDMolSupplier(path) 
            mol = sdf_supplier[0]
            mol = Chem.RemoveHs(mol, sanitize=False)
            try:
                Chem.SanitizeMol(mol, \
                        SanitizeFlags.SANITIZE_SETCONJUGATION | \
                        SanitizeFlags.SANITIZE_SETHYBRIDIZATION | \
                        SanitizeFlags.SANITIZE_FINDRADICALS)
                Chem.AssignStereochemistryFrom3D(mol)
            except:
                pass
        except Exception as e:
            print(f'Failed to load pdb ligand {path}: {e}')
    elif path.endswith('rdmol.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            mol = pickle.load(f)
    else:
        raise ValueError(f'invalid suffix: {path}')
    return mol


def extract_pocket(lig_mol, pro_mol, theta=8):
    """tbd"""
    # find distant protein atoms to be removed
    dis_matrix = distance_matrix(lig_mol.GetConformers()[0].GetPositions(), 
                                    pro_mol.GetConformers()[0].GetPositions())
    pro_list = dis_matrix.min(axis=0) > theta
    pro_set_toberemoved = np.where(pro_list)[0]
    
    # remove atoms
    mw = Chem.RWMol(pro_mol)
    mw.BeginBatchEdit()
    for i in pro_set_toberemoved[::-1]:     # from the end
        mw.RemoveAtom(int(i))
    mw.CommitBatchEdit()
    pro_mol = Chem.Mol(mw)

    poc_dis_matrix = dis_matrix[:, ~pro_list]
    return lig_mol, pro_mol, poc_dis_matrix


def get_bond_length(node_pos, edge_index):
    """
    node_pos: (N, 3)
    edge_index: (E, 2)
    """
    pos1 = node_pos[edge_index[:, 0]]
    pos2 = node_pos[edge_index[:, 1]]
    return np.sqrt(np.sum(np.square(pos1 - pos2), 1)).astype('float32')




def save_atom_pos_list_to_sdf(mol, atom_pos_list, sdf_file):
    """
    save a list of mol with specified atom_pos to `sdf_file`
    """
    try:
        with Chem.SDWriter(sdf_file) as w:
            for atom_pos in atom_pos_list:
                # create new atom position
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    x, y, z = atom_pos[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                w.write(mol)
        return True
    except Exception as e:
        logging.warning(f'Failed to write sdf_file {sdf_file}: {e}')
        return False

def get_obrms_rmsd(gt_file, pred_file):
    """
    tbd
    """
    obrms_bin = 'obrms'
    result = os.popen(f'{obrms_bin} {gt_file} {pred_file}').readline()
    print(f'obrms command : {obrms_bin} {gt_file} {pred_file}, ', 'result : ', result.split(' ')[-1][:-2])
    try:
        rmsd = float(result.split(' ')[-1][:-2])
    except Exception as e:
        print(f'{gt_file} {pred_file} {e}')
        rmsd = None
    return rmsd
