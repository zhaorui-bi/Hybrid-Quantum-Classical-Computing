# Step 0: Import Packages
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import pandas as pd


# Step 1: Atom Featurisation
def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]

    return binary_encoding


def get_atom_features(atom,
                      use_chirality=True,
                      hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


# Step 2: Bond Featurisation
def get_bond_features(bond,
                      use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


# Step 3: Generating labeled Pytorch Geometric Graph Objects
def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """

    data_list = []

    for (smiles, y_val) in zip(x_smiles, y):

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))

    return data_list

from rdkit import Chem
# from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import ast
IPythonConsole.ipython_useSVG = True
from torch_geometric.data import InMemoryDataset
import pandas as pd

# property 6
class AMPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AMPDataset, self).__init__(root, transform, pre_transform)  # transform
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['AMPDataset.dataset']

    def download(self):
        pass

    def process(self):
        df = pd.read_csv('./data.csv', header=None)
        smiles = df[0].tolist()
        del smiles[0]
        for i in range(len(smiles)):
            peptide_smiles = Chem.MolToSmiles(Chem.MolFromFASTA(smiles[i]))
            smiles[i] = peptide_smiles
        y = df[19].tolist()
        del y[0]
        arr = np.array(y, dtype=float)
        y = arr.tolist()
        # for i in range(len(y)):
        #     y[i] = ast.literal_eval(y[i])
        # print(smiles)
        # print(y)
        data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(smiles, y)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

dataset = AMPDataset(root='data/')
print(dataset)
print(dataset.num_features)
print(dataset.y)

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGEConv,GATConv,GATv2Conv,GCNConv
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# dataset = MoleculeNet(root='.', name='ESOL')
dataset = AMPDataset(root='data/')
data_size = len(dataset)
BATCH = 64
hidden_channels = 64
dropout = 0.5
num_heads = 4
num_layers = 2

# define the GCN model
class AMPNet(torch.nn.Module):
    def __init__(self):
        # Initialization
        super().__init__()
        torch.manual_seed(42)
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads = 1)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads = 1)
        # self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        # depends on cat pooling number
        self.out = Linear(hidden_channels, 1)

    def forward(self, x , edge_index, batch_index):
        # 1st Conv layer
        hidden = self.conv1(x, edge_index)
        hidden = torch.relu(hidden)
        hidden = self.dropout_layer(hidden)
        # Other layers
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.relu(hidden)
        # hidden = self.dropout_layer(hidden)
        # hidden = self.conv3(hidden, edge_index)
        # hidden = torch.relu(hidden)
        # hidden = self.dropout_layer(hidden)
        # Global pooling
        hidden = gmp(hidden, batch_index)
        # hidden = torch.cat((gmp(hidden, batch_index), gap(hidden, batch_index)), dim=1)
        # hidden = torch.cat((gmp(hidden, batch_index), gap(hidden, batch_index), gap(hidden, batch_index)), dim=1)
        out = self.out(hidden)
        return out

    def train_model(self, loss_fn, optimizer, loader, num_epochs):
        self.train()
        device = next(self.parameters()).device
        losses = []
        for epoch in range (num_epochs):
            total_loss = 0
            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                pred = self(data.x.float(), data.edge_index, data.batch)
                loss = loss_fn(pred, data.y.view(-1, 1).float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(loader)
            losses.append(epoch_loss)
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {epoch_loss}")
        return losses

model = AMPNet()
print(model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,weight_decay=5e-4)

loader = DataLoader(dataset[:int(data_size * 0.8)],
                    batch_size=BATCH, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.6):],
                         batch_size=BATCH, shuffle=True)

losses = model.train_model(loss_fn, optimizer, loader, 300)

# Visualize learning (training loss)
losses_float = [float(loss) for loss in losses]
loss_indices = [i for i, _ in enumerate(losses_float)]
sns.lineplot(x=loss_indices, y=losses_float)
plt.show()

test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch)
    df = pd.DataFrame()
    df["y_real"] = test_batch.y.tolist()
    df["y_pred"] = pred.tolist()
df["y_real"] = df["y_real"].apply(lambda row: row[0] if isinstance(row, list) else row)
df["y_pred"] = df["y_pred"].apply(lambda row: row[0] if isinstance(row, list) else row)
print(df)
df.to_csv('predication.csv', index=False)

# Visualize predication (real)
COMPARE = sns.scatterplot(data=df, x="y_real", y="y_pred", color='blue', label='Comparison')
COMPARE.set(xlim=(-7, 2))
COMPARE.set(ylim=(-7, 2))
plt.show()