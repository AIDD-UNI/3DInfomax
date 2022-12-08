import copy
import os

import torch
import dgl
import torch_geometric
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector, get_atom_feature_dims, \
    get_bond_feature_dims
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from scipy.constants import physical_constants

hartree2eV = physical_constants['hartree-electron volt relationship'][0]

def process(chembl_id):
    # print('processing data from ({}) and saving it to ({})'.format(root,
                                                                    # os.path.join(root, 'processed')))
    # chembl_ids = os.listdir(os.path.join(root, 'structures'))
    root='/sharefs/healthshare/yuancheng/datasets/QMugs'
    processed_file='processed.pt'

    targets = {'DFT:ATOMIC_ENERGY': [], 'DFT:TOTAL_ENERGY': [], 'DFT:HOMO_ENERGY': []}
    atom_slices = [0]
    edge_slices = [0]
    all_atom_features = []
    all_edge_features = []
    edge_indices = []  # edges of each molecule in coo format
    total_atoms = 0
    total_edges = 0
    n_atoms_list = []
    coordinates = torch.tensor([])
    avg_degree = 0  # average degree in the dataset
    # for mol_idx, chembl_id in tqdm(enumerate(chembl_ids)):

    mol_path = os.path.join(root, 'structures', chembl_id)
    sdf_names = os.listdir(mol_path)
    conformers = []
    for conf_idx, sdf_name in enumerate(sdf_names):
        sdf_path = os.path.join(mol_path, sdf_name)
        suppl = Chem.SDMolSupplier(sdf_path)
        mol = next(iter(suppl))
        c = next(iter(mol.GetConformers()))
        conformers.append(torch.tensor(c.GetPositions()))
        if conf_idx == 0:
            n_atoms = len(mol.GetAtoms())
            n_atoms_list.append(n_atoms)
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            all_atom_features.append(torch.tensor(atom_features_list, dtype=torch.long))

            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_feature = bond_to_feature_vector(bond)

                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)
            # Graph connectivity in COO format with shape [2, num_edges]
            edge_index = torch.tensor(edges_list, dtype=torch.long).T
            edge_features = torch.tensor(edge_features_list, dtype=torch.long)

            avg_degree += (len(edges_list) / 2) / n_atoms

            # get all 19 attributes that should be predicted, so we drop the first two entries (name and smiles)
            targets['DFT:HOMO_ENERGY'].append(float(mol.GetProp('DFT:HOMO_ENERGY')))
            targets['DFT:TOTAL_ENERGY'].append(float(mol.GetProp('DFT:TOTAL_ENERGY')))
            targets['DFT:ATOMIC_ENERGY'].append(float(mol.GetProp('DFT:ATOMIC_ENERGY')))
            edge_indices.append(edge_index)
            all_edge_features.append(edge_features)

            total_edges += len(edges_list)
            total_atoms += n_atoms
            edge_slices.append(total_edges)
            atom_slices.append(total_atoms)
    if len(conformers) < 3:  # if there are less than 10 conformers we add the first one a few times
        conformers.extend([conformers[0]] * (3 - len(conformers)))

    coordinates = torch.cat([coordinates, torch.cat(conformers, dim=1)], dim=0)

    return (n_atoms_list, atom_slices, edge_slices, edge_indices, all_atom_features, all_edge_features, coordinates, targets, avg_degree)
    # data_dict = {'chembl_ids': chembl_ids,
    #                 'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
    #                 'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
    #                 'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
    #                 'edge_indices': torch.cat(edge_indices, dim=1),
    #                 'atom_features': torch.cat(all_atom_features, dim=0),
    #                 'edge_features': torch.cat(all_edge_features, dim=0),
    #                 'coordinates': coordinates,
    #                 'targets': targets,
    #                 'avg_degree': avg_degree / len(chembl_ids)
    #                 }
    # for key, value in targets.items():
    #     targets[key] = torch.tensor(value)[:, None]
    # data_dict.update(targets)

    if not os.path.exists(os.path.join(root, 'processed')):
        os.mkdir(os.path.join(root, 'processed'))
    torch.save(data_dict, os.path.join(root, 'processed', processed_file))


def listdir(root='/sharefs/healthshare/yuancheng/datasets/QMugs'):
    chembl_ids = os.listdir(os.path.join(root, 'structures'))
    for id in chembl_ids:
        yield id



if __name__ == "__main__":
    import multiprocessing
    pool = multiprocessing.Pool(16)

    root = '/sharefs/healthshare/yuancheng/datasets/QMugs'
    processed_file='processed.pt'

    chembl_ids = os.listdir(os.path.join(root, 'structures'))
    atom_slices = [0]
    edge_slices = [0]
    all_atom_features = []
    all_edge_features = []
    edge_indices = []  # edges of each molecule in coo format
    total_atoms = 0
    total_edges = 0
    coordinates = torch.tensor([])
    n_atoms_list = []
    avg_degree = 0
    targets = {'DFT:ATOMIC_ENERGY': [], 'DFT:TOTAL_ENERGY': [], 'DFT:HOMO_ENERGY': []}
    cnt = 0
    for res in tqdm(pool.imap(process,listdir(), chunksize=100000), total=len(chembl_ids)):
        n_atoms_list_, atom_slices_, edge_slices_, edge_indices_, all_atom_features_, all_edge_features_, conformers_, targets_, avg_degree_ = res
        n_atoms_list.append(n_atoms_list_[-1])
        atom_slices.append(atom_slices_[-1])
        edge_slices.append(edge_slices_[-1])
        all_atom_features.append(all_atom_features_[-1])
        all_edge_features.append(all_edge_features_[-1])
        edge_indices.append(edge_indices_[-1])
        coordinates = torch.cat([coordinates, torch.cat(conformers_, dim=1)], dim=0)
        avg_degree += avg_degree_

        for k, v in targets_.items():
            targets[k] += v

        cnt += 1 
        if cnt >= 10000:
            break

    data_dict = {'chembl_ids': chembl_ids,
                'n_atoms': torch.tensor(n_atoms_list, dtype=torch.long),
                'atom_slices': torch.tensor(atom_slices, dtype=torch.long),
                'edge_slices': torch.tensor(edge_slices, dtype=torch.long),
                'edge_indices': torch.cat(edge_indices, dim=1),
                'atom_features': torch.cat(all_atom_features, dim=0),
                'edge_features': torch.cat(all_edge_features, dim=0),
                'coordinates': coordinates,
                'targets': targets,
                'avg_degree': avg_degree / len(chembl_ids)
                } 
    for key, value in targets.items():
        targets[key] = torch.tensor(value)[:, None]
    data_dict.update(targets)

    if not os.path.exists(os.path.join(root, 'processed')):
        os.mkdir(os.path.join(root, 'processed'))
    torch.save(data_dict, os.path.join(root, 'processed', processed_file))