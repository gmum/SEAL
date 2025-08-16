from tdc.single_pred import HTS, ADME, QM
from tdc.single_pred import Tox

from rdkit import Chem
from torch_geometric.data import Data
import numpy as np
import torch
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc.utils import retrieve_label_name_list
import pandas as pd
from tdc.utils import retrieve_label_name_list

DEFAULT_ATOM_TYPE_SET = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "Other"]
# DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "SP3D","SP3D2" ,"Other"]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "Other"]

SYNTHETIC_DATASET=[
        "rings-count",
        "rings-max",
        "X",
        "P",
        "B",
        "indole",
        "PAINS",
    ]
REAL_DATASET=[
    "sol",
    "cyp",
    "herg",
    "herg_K",
]      

CLASSIFICATION_DATASET = {
    "rings-count",
    "rings-max",
    "X",
    "P",
    "B",
    "indole",
    "PAINS",
    "cyp",
    "herg",
    "herg_K",
}
REGRESSION_DATASET = {
    "sol",
}
STATS_DATASET={
    "sol": {"mean": -2.86, "std": 2.38},
}

def build_dataset(kwargs:dict) -> tuple:
    """
    Build the dataset based on the provided arguments.

    Args:
        kwargs (dict): Arguments to configure the dataset.
    Returns:
        tuple: Training, validation, and test splits of the dataset.
    """

    if kwargs["data_set"] == "sol":
        data = ADME(name='Solubility_AqSolDB')
        split = data.get_split(seed=kwargs['split'])
    elif kwargs["data_set"] == "cyp":
        data = ADME(name='CYP2C9_Veith')
        split = data.get_split(seed=kwargs['split'])
    elif kwargs["data_set"] == "herg":
        data= Tox(name = 'hERG')
        split = data.get_split(seed=kwargs['split'])
    elif kwargs["data_set"] == "herg_K":
        data= Tox(name = 'hERG_Karim')
        split = data.get_split(seed=kwargs['split'])
    elif kwargs["data_set"] in [
        "rings-count",
        "rings-max",
        "X",
        "P",
        "B",
        "indole",
        "PAINS",
    ]:
        data=pd.read_csv('data/data.csv')
        data=data.rename(columns={"smiles": "Drug", kwargs['data_set']: "Y"})
        
        train = data[data[f"split_{kwargs['split']}"] == 'train']
        valid = data[data[f"split_{kwargs['split']}"] == 'valid']
        test = data[data[f"split_{kwargs['split']}"] == 'test']

        split = {}
        split['train'] = train
        split['valid'] = valid
        split['test'] = test
    else:
        raise ValueError("Invalid dataset name")
    return split['train'], split['valid'], split['test']


class Featurizer:
    """
    Base class for featurizing molecular graphs.
    """
    def __init__(self, y_column, smiles_col='Drug', **kwargs):
        self.y_column = y_column
        self.smiles_col = smiles_col
        self.__dict__.update(kwargs)

    def __call__(self, df: pd.DataFrame, kwargs: dict) -> list:
        raise NotImplementedError()


class GraphFeaturizer(Featurizer):
    """
    Featurizer for graph-based molecular representations.
    """
    def __call__(self, df: pd.DataFrame, kwargs: dict) -> list:
        """
        Converts a DataFrame into a list of PyTorch Geometric Data objects.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            kwargs (dict): Additional keyword arguments.

        Returns:
            list: List of PyTorch Geometric Data objects.
        """
        self.mean = kwargs.get("mean", 0)
        self.std = kwargs.get("std", 1)
        graphs = []
        labels = []
        assignments = []
        assignments_nodes =[]

        number_clusters = []

        for i, row in df.iterrows():
            # Extract label and SMILES
            y = float(row[self.y_column])
            y = (y - self.mean) / self.std

            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue
            # Add edges and nodes
            edges = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edges.append((start, end))
                edges.append((end, start))

            if len(edges) == 0:
                edges = np.empty((0, 2)).T
                # mask=np.empty())
            else:
                edges = np.array(edges).T

            nodes = []
            # Extract features for each atom
            for atom in mol.GetAtoms():
                res = self.get_features(atom)
                # res.extend(self.get_atom_degree(atom))
                # res.extend(self.get_formal_charge(atom))
                # res.extend(self.get_is_aromatic(atom))
                # res.extend(self.get_hybridization(atom))
                # res.extend(self.get_valence(atom))
                # res.extend(self.get_is_inring(atom))
                # res.extend(self.get_num_hs(atom))
                
                nodes.append(res)
            nodes = np.array(nodes)
            num_atoms = mol.GetNumAtoms()
            # create BRICS decomposition structure
            cliques, breaks = self.brics_decomp_extra(mol)
            s = self.create_s(cliques, num_atoms)
            s_node = self.create_s_atom_wise(cliques, num_atoms)

            # Create a mask for broken edges and assignments
            assignments.append(s)
            assignments_nodes.append(s_node)
            mask = self.mask_broken_edges(torch.LongTensor(edges), breaks)
            graphs.append((nodes, edges, mask))
            labels.append(y)
            number_clusters.append(s.shape[1])
        max_num_clusters = max([s.shape[1] for s in assignments])
        max_num_nodes = max([s_node.shape[1] for s_node in assignments_nodes])
        for i in range(len(assignments)):
            num_clusters = assignments[i].shape[1]
            num_atoms = assignments[i].shape[0]
            padding = torch.zeros((num_atoms, max_num_clusters-num_clusters))
            assing_new = torch.cat((assignments[i], padding), dim=1)
            assignments[i] = np.array(assing_new)
            
        for i in range(len(assignments_nodes)):
            num_clusters = assignments_nodes[i].shape[1]
            num_atoms = assignments_nodes[i].shape[0]
            padding = torch.zeros((num_atoms, max_num_nodes-num_clusters))
            assing_new = torch.cat((assignments_nodes[i], padding), dim=1)
            assignments_nodes[i] = np.array(assing_new)

        labels = np.array(labels)
        return [Data(
            x=torch.FloatTensor(x),
            edge_index=torch.LongTensor(edge_index),
            s=torch.FloatTensor(s),
            s_node=torch.FloatTensor(s_node),
            y=torch.FloatTensor([y]),
            num_cluster=torch.LongTensor([num_cluster]),
            mask=torch.FloatTensor(mask)
        ) for ((x, edge_index, mask), s, s_node ,num_cluster, y) in zip(graphs, assignments, assignments_nodes, number_clusters, labels)]

    def mask_broken_edges(self, edge_index, breaks, apply_sigmoid=False):
        """
        Creates a mask for broken edges in the graph.
        Args:
            edge_index (torch.Tensor): The edge index tensor.
            breaks (list): A list of broken edges.
            apply_sigmoid (bool, optional): Whether to apply sigmoid to the mask. Defaults to False.
        Returns:
            torch.Tensor: The mask for broken edges.
        """
        num_edges = edge_index.size(1)
        mask = torch.ones(num_edges, dtype=torch.float32)
        broken_edges = set(frozenset([a, b]) for a, b in breaks)

        for i in range(num_edges):
            edge = frozenset(
                [edge_index[0, i].item(), edge_index[1, i].item()])
            if edge in broken_edges:
                mask[i] = 0.0
        return mask

    """
    Functions to create features for atoms and molecules
    """
    @staticmethod
    def one_of_k_encoding_unk(x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: float(x == s), allowable_set))

    def get_features(self, x):
        return GraphFeaturizer.one_of_k_encoding_unk(x.GetSymbol(), DEFAULT_ATOM_TYPE_SET)

    def get_atom_degree(self, x):
        return [float(x.GetDegree())]

    # def get_bond_type(self,x):
    #     return GraphFeaturizer.one_of_k_encoding_unk(x.GetBondType(), DEFAULT_BOND_TYPE_SET)

    def get_formal_charge(self, x):
        return [float(x.GetFormalCharge())]

    def get_is_aromatic(self, x):
        return [float(x.GetIsAromatic())]

    def get_hybridization(self, x):
        return GraphFeaturizer.one_of_k_encoding_unk(x.GetHybridization(), DEFAULT_HYBRIDIZATION_SET)

    def get_valence(self, x):
        return [float(x.GetImplicitValence())]

    def get_is_inring(self, x):
        return [float(x.IsInRing())]

    def get_num_hs(self, x):
        return [float(x.GetTotalNumHs())]
    
    def find_murcko_link_bond(self, mol):
        core = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_index = mol.GetSubstructMatch(core)
        link_bond_list = []
        num_bonds = mol.GetNumBonds()
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            link_score = 0
            if u in scaffold_index:
                link_score += 1
            if v in scaffold_index:
                link_score += 1
            if link_score == 1:
                link_bond_list.append([u, v])
        return link_bond_list
    
    """
    Functions for decomposition and substructure extraction, adopted from
    https://github.com/wzxxxx/Substructure-Mask-Explanation
    """
    def murcko_decomp(self, mol):
        d = self.return_murcko_leaf_structure(mol)
        cliques = list(d['substructure'].values())
        return cliques, d['substructure_bond']

    def return_murcko_leaf_structure(self, m):

        # return murcko_link_bond
        all_murcko_bond = self.find_murcko_link_bond(m)

        all_murcko_substructure_subset = dict()
        # return atom in all_murcko_bond
        all_murcko_atom = []
        for murcko_bond in all_murcko_bond:
            all_murcko_atom = list(set(all_murcko_atom + murcko_bond))

        if len(all_murcko_atom) > 0:

            all_break_atom = dict()
            for murcko_atom in all_murcko_atom:
                murcko_break_atom = []
                for murcko_bond in all_murcko_bond:
                    if murcko_atom in murcko_bond:
                        murcko_break_atom += list(set(murcko_bond))
                murcko_break_atom = [
                    x for x in murcko_break_atom if x != murcko_atom]
                all_break_atom[murcko_atom] = murcko_break_atom

            substrate_idx = dict()
            used_atom = []
            for initial_atom_idx, break_atoms_idx in all_break_atom.items():
                if initial_atom_idx not in used_atom:
                    neighbor_idx = [initial_atom_idx]
                    substrate_idx_i = neighbor_idx
                    begin_atom_idx_list = [initial_atom_idx]
                    while len(neighbor_idx) != 0:
                        for idx in begin_atom_idx_list:
                            initial_atom = m.GetAtomWithIdx(idx)
                            neighbor_idx = neighbor_idx + [neighbor_atom.GetIdx() for neighbor_atom in
                                                           initial_atom.GetNeighbors()]
                            exlude_idx = all_break_atom[initial_atom_idx] + \
                                substrate_idx_i
                            if idx in all_break_atom.keys():
                                exlude_idx = all_break_atom[initial_atom_idx] + \
                                    substrate_idx_i + all_break_atom[idx]
                            neighbor_idx = [
                                x for x in neighbor_idx if x not in exlude_idx]
                            substrate_idx_i += neighbor_idx
                            begin_atom_idx_list += neighbor_idx
                        begin_atom_idx_list = [
                            x for x in begin_atom_idx_list if x not in substrate_idx_i]
                    substrate_idx[initial_atom_idx] = substrate_idx_i
                    used_atom += substrate_idx_i
                else:
                    pass
        else:
            substrate_idx = dict()
            substrate_idx[0] = [x for x in range(m.GetNumAtoms())]
        all_murcko_substructure_subset['substructure'] = substrate_idx
        all_murcko_substructure_subset['substructure_bond'] = all_murcko_bond
        return all_murcko_substructure_subset

    def brics_decomp_extra(self, mol):
        # Atomic numbers of F, Cl, Br, I, At, Ts
        halogens = {9, 17, 35, 53, 85, 117}

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])

        res = list(BRICS.FindBRICSBonds(mol))

        for bond in res:
            bond_atoms = [bond[0][0], bond[0][1]]
            if bond_atoms in cliques:
                cliques.remove(bond_atoms)
                breaks.append(bond_atoms)
            else:
                reverse_bond = [bond[0][1], bond[0][0]]
                cliques.remove(reverse_bond)
                breaks.append(reverse_bond)
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()

        # print(ring_info)
        # print(atom_rings)
        atom_to_rings = {i: set() for i in range(mol.GetNumAtoms())}

        for id_ring, ring in enumerate(atom_rings):
            for atom in ring:
                atom_to_rings[atom].add(id_ring)
        # print(atom_to_rings)
        cliques_to_remove = []
        for c in cliques:
            if len(c) > 1:
                atom0, atom1 = c[0], c[1]
                atom0_in_ring = mol.GetAtomWithIdx(atom0).IsInRing()
                atom1_in_ring = mol.GetAtomWithIdx(atom1).IsInRing()

                if atom0_in_ring and not atom1_in_ring:
                    cliques_to_remove.append(c)
                    cliques.append([atom1])
                    breaks.append(c)
                elif atom1_in_ring and not atom0_in_ring:
                    cliques_to_remove.append(c)
                    cliques.append([atom0])
                    breaks.append(c)
                elif (atom0_in_ring and atom1_in_ring) and not (atom_to_rings[atom0] & atom_to_rings[atom1]):
                    # Both atoms are in rings but in different rings
                    cliques_to_remove.append(c)
                    cliques.append([atom0])
                    cliques.append([atom1])
                    breaks.append(c)

        for c in cliques_to_remove:
            cliques.remove(c)

        halogen_bonds_to_remove = []
        for c in cliques:
            if len(c) == 2:
                atom0, atom1 = c[0], c[1]
                if mol.GetAtomWithIdx(atom0).GetAtomicNum() in halogens or mol.GetAtomWithIdx(atom1).GetAtomicNum() in halogens:
                    halogen_bonds_to_remove.append(c)
                    cliques.append([atom0])
                    cliques.append([atom1])
                    breaks.append(c)

        for c in halogen_bonds_to_remove:
            cliques.remove(c)

        # Select atoms at intersections as motifs
        for atom in mol.GetAtoms():
            if len(atom.GetNeighbors()) > 3 and not atom.IsInRing():
                atom_idx = atom.GetIdx()
                cliques.append([atom_idx])
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    bond_forward = [nei_idx, atom_idx]
                    bond_reverse = [atom_idx, nei_idx]
                    if bond_forward in cliques:
                        cliques.remove(bond_forward)
                        breaks.append(bond_forward)
                    elif bond_reverse in cliques:
                        cliques.remove(bond_reverse)
                        breaks.append(bond_reverse)
                    cliques.append([nei_idx])

        # Merge cliques
        for c in range(len(cliques) - 1):
            if c >= len(cliques):
                break
            for k in range(c + 1, len(cliques)):
                if k >= len(cliques):
                    break
                if len(set(cliques[c]) & set(cliques[k])) > 0:
                    cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                    cliques[k] = []
            cliques = [c for c in cliques if len(c) > 0]
        cliques = [c for c in cliques if len(c) > 0]

        return cliques, breaks

    def brics_decomp(self, mol):
        n_atoms = mol.GetNumAtoms()

        if n_atoms == 1:
            return [[0]], []

        cliques = []
        breaks = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            cliques.append([a1, a2])

        res = list(BRICS.FindBRICSBonds(mol))
        # if len(res) == 0:
        #     return [list(range(n_atoms))], []
        # else:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
                breaks.append([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
                breaks.append([bond[0][1], bond[0][0]])

            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

        # merge cliques
        for c in range(len(cliques) - 1):
            if c >= len(cliques):
                break
            for k in range(c + 1, len(cliques)):
                if k >= len(cliques):
                    break
                if len(set(cliques[c]) & set(cliques[k])) > 0:
                    cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                    cliques[k] = []
            cliques = [c for c in cliques if len(c) > 0]
        cliques = [c for c in cliques if len(c) > 0]

        return cliques, breaks
    
    def create_s_atom_wise(self, cliques, num_atoms):
        s = torch.zeros((num_atoms, num_atoms))
        for id_atom in range(num_atoms):
            s[id_atom, id_atom] = 1
        return s

    def create_s(self, cliques, num_atoms):
        s = torch.zeros((num_atoms, len(cliques)))
        for clique_idx, clique in enumerate(cliques):
            for atom_idx in clique:
                s[atom_idx, clique_idx] = 1
        return s


class SyntheticGraphFeaturizer(GraphFeaturizer):
    """
    Featurizer for synthetic graph-based molecular representations.
    """
    def __call__(self, df: pd.DataFrame, kwargs: dict) -> list:
        """
            Converts a DataFrame from synthetic data into a list of PyTorch Geometric Data objects.
            Args:
                df (pd.DataFrame): DataFrame containing the data.
                kwargs (dict): Additional keyword arguments.
            Returns:
                list: List of PyTorch Geometric Data objects.
        """
        self.mean = kwargs.get("mean", 0)
        self.std = kwargs.get("std", 1)
        graphs = []
        labels = []
        assignments = []
        assignments_nodes =[]
        number_clusters = []
        explain_labels=[]
        data_explain = Chem.SDMolSupplier("data/explanations.sdf")
        if data_explain is None:
            raise ValueError("Invalid data_explain_path")
        exp=self.explaination(kwargs)
        for (i, row) in df.iterrows():
            y = float(row[self.y_column])
            y = (y - self.mean) / self.std

            smiles = row[self.smiles_col]
            id_ = row['Unnamed: 0']
            mol_exp = data_explain[int(id_)]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES: {smiles}")
                continue

            edges = []
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                edges.append((start, end))
                edges.append((end, start))

            if len(edges) == 0:
                edges = np.empty((0, 2)).T

            else:
                edges = np.array(edges).T

            nodes = []
            for atom in mol.GetAtoms():
                res = self.get_features(atom)
                # res.extend(self.get_atom_degree(atom))
                # res.extend(self.get_formal_charge(atom))
                # res.extend(self.get_is_aromatic(atom))
                # res.extend(self.get_hybridization(atom))
                # res.extend(self.get_valence(atom))
                # res.extend(self.get_is_inring(atom))
                # res.extend(self.get_num_hs(atom))

                nodes.append(res)
            nodes = np.array(nodes)
            num_atoms = mol.GetNumAtoms()
            cliques, breaks = self.brics_decomp_extra(mol)
            s = self.create_s(cliques, num_atoms)
            s_node = self.create_s_atom_wise(cliques, num_atoms)
            assignments.append(s)
            assignments_nodes.append(s_node)
            # Store nodes, edges.T, and s together
            mask = self.mask_broken_edges(torch.LongTensor(edges), breaks)
            graphs.append((nodes, edges, mask))
            labels.append(y)
            number_clusters.append(s.shape[1])

            explaination_list = exp(mol,mol_exp)
            explaination_list=np.array(explaination_list)
            explaination_list = np.expand_dims(explaination_list, axis=1)
            explain_labels.append(explaination_list)

            # print(explaination_list.shape)
            # print(explaination_list)
        max_num_clusters = max([s.shape[1] for s in assignments])
        max_num_nodes = max([s_node.shape[1] for s_node in assignments_nodes])

        for i in range(len(assignments)):
            num_clusters = assignments[i].shape[1]
            num_atoms = assignments[i].shape[0]
            padding = torch.zeros((num_atoms, max_num_clusters-num_clusters))
            assing_new = torch.cat((assignments[i], padding), dim=1)
            assignments[i] = np.array(assing_new)

        for i in range(len(assignments_nodes)):
            num_clusters = assignments_nodes[i].shape[1]
            num_atoms = assignments_nodes[i].shape[0]
            padding = torch.zeros((num_atoms, max_num_nodes-num_clusters))
            assing_new = torch.cat((assignments_nodes[i], padding), dim=1)
            assignments_nodes[i] = np.array(assing_new)

        labels = np.array(labels)
        return [Data(
            x=torch.FloatTensor(x),
            edge_index=torch.LongTensor(edge_index),
            s=torch.FloatTensor(s),
            s_node=torch.FloatTensor(s_node),
            y=torch.FloatTensor([y]),
            num_cluster=torch.LongTensor([num_cluster]),
            mask=torch.FloatTensor(mask),
            explanation=torch.FloatTensor(explain_label)
        ) for ((x, edge_index, mask), s, s_node, num_cluster, y, explain_label) in zip(graphs, assignments, assignments_nodes, number_clusters, labels, explain_labels)]


    """
    Functions for extracting explanations based on the dataset task.
    """
    def explaination(self, kwargs):
        if kwargs["data_set"] == "rings-count":
            return self.check_if_node_in_ring
        if kwargs["data_set"] == "rings-max":
            return self.check_if_node_in_ring_size
        if kwargs["data_set"] == "X":
            return self.check_halogen
        if kwargs["data_set"] == "P":
            return self.check_phosphorus
        if kwargs["data_set"] == "B":
            return self.check_boron
        if kwargs["data_set"] == "indole":
            return self.check_indole
        if kwargs["data_set"] == "PAINS":
            return self.check_pains

    def check_if_node_in_ring(self, mol, mol_exp):
        e=[]
        rings = set(map(int, filter(None, mol_exp.GetProp("rings").split(","))))

        for atom in mol.GetAtoms():
            if atom.GetIdx() in rings:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_if_node_in_ring_size(self, mol, mol_exp):
        e=[]
        rings = set(map(int, filter(None, mol_exp.GetProp("largest_rings").split(","))))

        for atom in mol.GetAtoms():
            if atom.GetIdx() in rings:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_halogen(self, mol, mol_exp):

        halogens = set(map(int, filter(None, mol_exp.GetProp("X").split(","))))
        e=[]

        for atom in mol.GetAtoms():
            if atom.GetIdx() in halogens:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_indole(self, mol, mol_exp):
        indole = set(map(int, filter(None, mol_exp.GetProp("indole").split(","))))

        e=[]
        for atom in mol.GetAtoms():
            if atom.GetIdx() in indole:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_pains(self, mol, mol_exp):
        pains = set(map(int, filter(None, mol_exp.GetProp("pains").split(","))))

        e=[]
        for atom in mol.GetAtoms():
            if atom.GetIdx() in pains:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_phosphorus(self, mol, mol_exp):
        e=[]
        phosphorus = set(map(int, filter(None, mol_exp.GetProp("P").split(","))))

        for atom in mol.GetAtoms():
            if atom.GetIdx() in phosphorus:
                e.append(1)
            else:
                e.append(0)
        return e

    def check_boron(self, mol, mol_exp):
        e=[]
        boron = set(map(int, filter(None, mol_exp.GetProp("B").split(","))))

        for atom in mol.GetAtoms():
            if atom.GetIdx() in boron:
                e.append(1)
            else:
                e.append(0)
        return e