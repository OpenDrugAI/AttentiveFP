import matplotlib.pyplot as plt
plt.switch_backend('agg')

from rdkit.Chem import MolFromSmiles
from rdkit.Chem import Draw
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import os
from AttentiveFP.Featurizer_aromaticity_rm import *
import pickle
import time
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from rdkit.Chem.Draw import SimilarityMaps
from io import StringIO

smilesList = ['CC']
degrees = [0, 1, 2, 3, 4, 5]


class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    for atom in mol.GetAtoms():
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph

def array_rep_from_smiles(molgraph):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    #molgraph = graph_from_smiles_tuple(tuple(smiles))
    degrees = [0,1,2,3,4,5]
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix'      : molgraph.rdkit_ix_array()}

    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def gen_descriptor_data(smilesList):

    smiles_to_fingerprint_array = {}

    for i,smiles in enumerate(smilesList):
#         if i > 5:
#             print("Due to the limited computational resource, submission with more than 5 molecules will not be processed")
#             break
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            molgraph = graph_from_smiles(smiles)
            molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_smiles(molgraph)

            smiles_to_fingerprint_array[smiles] = arrayrep
            
        except:
            print(smiles)
            time.sleep(3)
    return smiles_to_fingerprint_array



def get_smiles_dicts(smilesList):
    #first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}

    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len,num_atom_features = atom_features.shape
        bond_len,num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    #then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}

    smiles_to_atom_mask = {}

    degrees = [0,1,2,3,4,5]
    #then run through our numpy array again
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))

        #get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len,num_atom_features))
        bonds = np.zeros((max_bond_len,num_bond_features))

        #then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len,len(degrees)))
        bond_neighbors = np.zeros((max_atom_len,len(degrees)))

        #now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        for i,feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j,feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i,degree_array in enumerate(atom_neighbors_list):
                    for j,value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count,j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i,degree_array in enumerate(bond_neighbors_list):
                    for j,value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count,j] = value
                    bond_neighbor_count += 1

        #then add everything to my arrays
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        
        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    feature_dicts = {}
#     feature_dicts['smiles_to_atom_mask'] = smiles_to_atom_mask
#     feature_dicts['smiles_to_atom_info']= smiles_to_atom_info
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }
    return feature_dicts
def save_smiles_dicts(smilesList,filename):
    #first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    smiles_to_rdkit_list = {}

    smiles_to_fingerprint_features = gen_descriptor_data(smilesList)

    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        smiles_to_rdkit_list[smiles] = rdkit_list

        atom_len,num_atom_features = atom_features.shape
        bond_len,num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    #then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    smiles_to_atom_info = {}
    smiles_to_bond_info = {}

    smiles_to_atom_neighbors = {}
    smiles_to_bond_neighbors = {}

    smiles_to_atom_mask = {}

    degrees = [0,1,2,3,4,5]
    #then run through our numpy array again
    for smiles,arrayrep in smiles_to_fingerprint_features.items():
        mask = np.zeros((max_atom_len))

        #get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len,num_atom_features))
        bonds = np.zeros((max_bond_len,num_bond_features))

        #then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len,len(degrees)))
        bond_neighbors = np.zeros((max_atom_len,len(degrees)))

        #now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        for i,feature in enumerate(atom_features):
            mask[i] = 1.0
            atoms[i] = feature

        for j,feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []
        for degree in degrees:
            atom_neighbors_list = arrayrep[('atom_neighbors', degree)]
            bond_neighbors_list = arrayrep[('bond_neighbors', degree)]

            if len(atom_neighbors_list) > 0:

                for i,degree_array in enumerate(atom_neighbors_list):
                    for j,value in enumerate(degree_array):
                        atom_neighbors[atom_neighbor_count,j] = value
                    atom_neighbor_count += 1

            if len(bond_neighbors_list) > 0:
                for i,degree_array in enumerate(bond_neighbors_list):
                    for j,value in enumerate(degree_array):
                        bond_neighbors[bond_neighbor_count,j] = value
                    bond_neighbor_count += 1

        #then add everything to my arrays
        smiles_to_atom_info[smiles] = atoms
        smiles_to_bond_info[smiles] = bonds

        smiles_to_atom_neighbors[smiles] = atom_neighbors
        smiles_to_bond_neighbors[smiles] = bond_neighbors
        
        smiles_to_atom_mask[smiles] = mask

    del smiles_to_fingerprint_features
    feature_dicts = {}
#     feature_dicts['smiles_to_atom_mask'] = smiles_to_atom_mask
#     feature_dicts['smiles_to_atom_info']= smiles_to_atom_info
    feature_dicts = {
        'smiles_to_atom_mask': smiles_to_atom_mask,
        'smiles_to_atom_info': smiles_to_atom_info,
        'smiles_to_bond_info': smiles_to_bond_info,
        'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
        'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
        'smiles_to_rdkit_list': smiles_to_rdkit_list
    }
    pickle.dump(feature_dicts,open(filename+'.pickle',"wb"))
    print('feature dicts file saved as '+ filename+'.pickle')
    return feature_dicts

def get_smiles_array(smilesList, feature_dicts):
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    for smiles in smilesList:
        x_mask.append(feature_dicts['smiles_to_atom_mask'][smiles])
        x_atom.append(feature_dicts['smiles_to_atom_info'][smiles])
        x_bonds.append(feature_dicts['smiles_to_bond_info'][smiles])
        x_atom_index.append(feature_dicts['smiles_to_atom_neighbors'][smiles])
        x_bond_index.append(feature_dicts['smiles_to_bond_neighbors'][smiles])
    return np.asarray(x_atom),np.asarray(x_bonds),np.asarray(x_atom_index),\
        np.asarray(x_bond_index),np.asarray(x_mask),feature_dicts['smiles_to_rdkit_list']


def moltosvg(mol,molSize=(280,218),kekulize=False):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
#    return svg
    return svg.replace('svg:','')

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def moltosvg_highlight(smiles, atom_list, atom_predictions, molecule_prediction,\
    molSize=(280,218),kekulize=False):
    
    mol = Chem.MolFromSmiles(smiles)
    min_pred = 0.05
    max_pred = 0.8
    # min_pred = np.amax([np.amin(all_atom_predictions),0])
    # min_pred = np.amin(all_atom_predictions)
    # min_pred = min(filter(lambda x:x>0,all_atom_predictions))
    # max_pred = np.amax(all_atom_predictions)
    note = 'y_pred: '+ str(molecule_prediction)

    norm = matplotlib.colors.Normalize(vmin=np.exp(0.068),vmax=np.exp(max_pred))
    cmap = cm.get_cmap('gray_r')


    plt_colors = cm.ScalarMappable(norm=norm,cmap=cmap)

    atom_colors = {}
    for i,atom in enumerate(atom_list):
        color_rgba = plt_colors.to_rgba(atom_predictions[i])
        atom_rgb = color_rgba #(color_rgba[0],color_rgba[1],color_rgba[2])
        atom_colors[atom] = atom_rgb

    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    #drawer.DrawMolecule(mc)
    drawer.DrawMolecule(mol,highlightAtoms=atom_list,highlightBonds=[],
        highlightAtomColors=atom_colors,legend=note)
    drawer.SetFontSize(68)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # svg = svg.replace('size:15px','size:18px')
    # svg = svg.replace('size:12px','size:28px')
    # svg = rreplace(svg,'size:11px','size:28px',1)
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')

def moltosvg_highlight_known(smiles, atom_list, atom_predictions, molecule_prediction, molecule_experiment, Number,\
        molSize=(280,218),kekulize=False):
    
    mol = Chem.MolFromSmiles(smiles)
    min_pred = 0.05
    max_pred = 0.8
    # min_pred = np.amax([np.amin(all_atom_predictions),0])
    # min_pred = np.amin(all_atom_predictions)
    # min_pred = min(filter(lambda x:x>0,all_atom_predictions))
    # max_pred = np.amax(all_atom_predictions)
    note = '('+ str(Number) +') y-y\' : '+ str(round(molecule_experiment,2)) + '-' + str(round(molecule_prediction,2))

    norm = matplotlib.colors.Normalize(vmin=0.01*5,vmax=max_pred*6)
    cmap = cm.get_cmap('gray_r')


    plt_colors = cm.ScalarMappable(norm=norm,cmap=cmap)

    atom_colors = {}
    for i,atom in enumerate(atom_list):
        color_rgba = plt_colors.to_rgba(atom_predictions[i])
        atom_rgb = color_rgba #(color_rgba[0],color_rgba[1],color_rgba[2])
        atom_colors[atom] = atom_rgb

    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    #drawer.DrawMolecule(mc)
    drawer.DrawMolecule(mol,highlightAtoms=atom_list,highlightBonds=[],
        highlightAtomColors=atom_colors,legend=note)
    drawer.SetFontSize(68)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # svg = svg.replace('size:15px','size:18px')
    # svg = svg.replace('size:12px','size:28px')
    # svg = rreplace(svg,'size:11px','size:28px',1)
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:','')

def weighted_highlight_known(smiles, atom_list, atom_predictions, molecule_prediction, molecule_experiment, Number,\
        molSize=(128,128)):
    
    mol = Chem.MolFromSmiles(smiles)
    note = '('+ str(Number) +') y-y\' : '+ str(round(molecule_experiment,2)) + '-' + str(round(molecule_prediction,2))

    contribs = [atom_predictions[m] for m in np.argsort(atom_list)]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, colorMap='bwr', contourLines=5, size=molSize)
    fig.axes[0].set_title(note)
    sio = StringIO()
    fig.savefig(sio, format="svg", bbox_inches='tight')
    svg = sio.getvalue()   
    return svg

def moltosvg_interaction_known(mol, atom_list, atom_predictions, molecule_prediction, molecule_experiment, max_atom_pred, min_atom_pred, Number):
    
    note = '('+ str(Number) +') y-y\' : '+ str(round(molecule_experiment,2)) + '-' + str(round(molecule_prediction,2))
    norm = matplotlib.colors.Normalize(vmin=min_atom_pred*0.9,vmax=max_atom_pred*1.1)
    cmap = cm.get_cmap('gray_r')

    plt_colors = cm.ScalarMappable(norm=norm,cmap=cmap)

    atom_colors = {}
    for i,atom in enumerate(atom_list):
        atom_colors[atom] = plt_colors.to_rgba(atom_predictions[i])          
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(280,218)
    op = drawer.drawOptions()
    for i in range(mol.GetNumAtoms()):
        op.atomLabels[i]=mol.GetAtomWithIdx(i).GetSymbol() + str(i)

    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol,highlightAtoms=atom_list,highlightBonds=[],
        highlightAtomColors=atom_colors,legend=note)
    drawer.SetFontSize(68)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')
