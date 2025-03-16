"""Make database from pdb files

This module provides functionality to process protein structure files (PDB/CIF) and create
a dataset of residue neighbors for machine learning applications. It handles:
- Parsing protein structure files
- Extracting residue information and coordinates
- Calculating neighbor relationships
- Caching processed data for efficient reuse
- Creating PyTorch datasets for training
"""

import os
import pickle

import torch
from torch.utils.data import Dataset, ConcatDataset
from einops import rearrange  # For tensor reshaping operations
from Bio.PDB import PDBParser, FastMMCIFParser  # For parsing protein structure files


class NeighborRecord:
    """Class to process and store information about residues and their neighbors in a protein structure.
    
    This class handles:
    - Parsing PDB/CIF files
    - Extracting residue information (coordinates, B-factors, etc.)
    - Calculating spatial relationships between residues
    - Managing neighbor relationships
    """
    
    def __init__(self, file_path: str) -> None:
        """Initialize the NeighborRecord by parsing a protein structure file.
        
        Args:
            file_path: Path to the PDB or CIF file to process
        """
        # Check file type
        file_type = os.path.splitext(file_path)[-1]
        if file_type not in ('.pdb', '.cif'):
            raise TypeError(f'Only .pdb and .cif type are supported, get {file_type}')

        # Initialize appropriate parser based on file type
        parser = PDBParser(QUIET=True) if file_type == '.pdb'\
            else FastMMCIFParser(QUIET=True)
        
        # Parse the structure file
        structure = parser.get_structure('none', file_path)
        models = list(structure.get_models())
        
        # Validate single model structure
        if len(models) != 1:
            raise ValueError(
                f'Only single model PDBs are supported. Found {len(models)} models.')
        model = models[0]

        # Initialize lists to store residue information
        residue_infos = []  # (chain_id, residue_id, residue_name)
        b_factors = []      # B-factors for main chain atoms
        main_chain_coords = []  # Coordinates of N, CA, C atoms
        gloabl_res_id = []  # Unique identifier for each residue

        # Process each chain and residue in the structure
        for chain in model.get_chains():
            for res in chain:
                # Create unique identifier for residue
                chain_id_hash = hash(chain.id)

                # Check for insertion codes (not supported)
                if res.id[2] != ' ':
                    raise ValueError(
                        f'PDB contains an insertion code at chain {chain.id} and residue '
                        f'index {res.id[1]}. These are not supported.')

                # Skip non-standard residues (e.g., HETATM)
                if res.id[0] != ' ':
                    continue

                # Define main chain atoms we're interested in
                main_chain_atom_names = ['N', 'CA', 'C']
                res_info = (chain.id, res.id[1], res.resname)
                
                # Initialize tensors for storing atom information
                b_factor = torch.zeros(3)  # For N, CA, C B-factors
                main_chain_coord = torch.zeros(3, 3)  # For N, CA, C coordinates
                atom_fill_flag = torch.zeros(3, dtype=torch.bool)  # Track which atoms we've found

                # Process each atom in the residue
                for i, atom in enumerate(res.get_atoms()):
                    if atom.name not in main_chain_atom_names:
                        continue
                    # Get index of current atom in our main chain list
                    index = main_chain_atom_names.index(atom.name)
                    # Store atom coordinates and B-factor
                    main_chain_coord[index] = torch.tensor(atom.get_coord())
                    b_factor[index] = atom.get_bfactor()
                    atom_fill_flag[index] = True
                    
                    # If we've found all main chain atoms, stop searching
                    if atom_fill_flag.all():
                        break

                # Validate we found all main chain atoms
                if not atom_fill_flag.all():
                    print(chain.id, res.id[1],
                        list(zip(main_chain_atom_names, atom_fill_flag)))
                    raise ValueError('Imcomplete main chain atom set')

                # Store residue information
                residue_infos.append(res_info)
                b_factors.append(b_factor)
                main_chain_coords.append(main_chain_coord)
                gloabl_res_id.append(chain_id_hash + res.id[1])

        # Convert lists to tensors for efficient computation
        gloabl_res_id = torch.tensor(gloabl_res_id)
        b_factors = torch.stack(b_factors)
        main_chain_coords = torch.stack(main_chain_coords)
        
        # Calculate CA atom distances between all residues
        ca_coords = main_chain_coords[:, 1, :]  # Get CA coordinates
        ca_coords = rearrange(ca_coords, 'L d -> () L d', d=3)
        ca_mutual_distance = torch.cdist(ca_coords, ca_coords)[0]
        
        # Calculate sequence distance between residues
        res_id_distance = gloabl_res_id.unsqueeze(-1) - gloabl_res_id.unsqueeze(0)

        # Store processed data as instance variables
        self.residue_infos = residue_infos
        self.b_factors = b_factors
        self.main_chain_coords = main_chain_coords
        self.ca_mutual_distance = ca_mutual_distance
        self.res_id_distance = res_id_distance
        self.structure = structure

    def get_residue_neighbors(self, radius=3, max_index_diff=3):
        """Calculate neighboring residues for each residue in the structure.
        
        Args:
            radius: Spatial distance threshold (in Angstroms) for considering neighbors
            max_index_diff: Maximum sequence separation to consider as neighbors
            
        Returns:
            Dictionary mapping residue indices to their neighbor indices
        """
        # Find spatially close residues
        is_neighbor = self.ca_mutual_distance < (radius ** 2)
        # Find sequentially close residues
        is_neighbor_2 = torch.abs(self.res_id_distance) <= max_index_diff
        # Combine spatial and sequence proximity criteria
        is_neighbor = is_neighbor | is_neighbor_2
        
        result = {}
        # For each residue, find and sort its neighbors
        for i in range(len(self.residue_infos)):
            neighbor_indexs = is_neighbor[i].nonzero(as_tuple=True)[0]
            neighbor_distances = self.ca_mutual_distance[i][neighbor_indexs]
            # Sort neighbors by distance
            neighbor_indexs = neighbor_indexs[neighbor_distances.argsort()]
            result[i] = neighbor_indexs
        return result

    def store_cache_to(self, file_name='', radius=3, to_return=False):
        """Cache processed data to disk or return as dictionary.
        
        Args:
            file_name: Path to save cache file (optional)
            radius: Neighbor radius used for calculations
            to_return: Whether to return the cache dictionary
            
        Returns:
            If to_return is True, returns cache dictionary
        """
        cache = {
            'residue_info': self.residue_infos,
            'neighbor_index': self.get_residue_neighbors(radius),
            'b_factor': self.b_factors,
            'main_chain_coord': self.main_chain_coords,
            'radius': radius
        }
        # Save to file if specified
        if file_name:
            with open(file_name, 'wb') as fout:
                pickle.dump(cache, fout)
        # Return cache if requested
        if to_return:
            return cache


class NeighborDataset(Dataset):
    """PyTorch Dataset class for accessing processed protein structure data.
    
    This class handles:
    - Loading cached data
    - Providing access to residue neighbor information
    - Formatting data for machine learning tasks
    """
    
    def __init__(self, file_path, radius=3, cache_suffix=''):
        """Initialize the dataset.
        
        Args:
            file_path: Path to protein structure file
            radius: Neighbor radius for calculations
            cache_suffix: Optional suffix for cache file naming
        """
        super().__init__()

        # Determine cache file path
        cache_path =\
            f'{os.path.splitext(file_path)[0]}_{cache_suffix}.pkl' if cache_suffix\
                else ''
        
        # Load from cache or process new data
        if not os.path.exists(cache_path):
            db = NeighborRecord(file_path)
            cache = db.store_cache_to(cache_path, to_return=True, radius=radius)
        else:
            with open(cache_path, 'rb') as fin:
                cache = pickle.load(fin)
                
        self.radius = radius
        self.data = cache
        # Only keep samples with at least one neighbor
        self.valid_sample_index = [i for i in cache['neighbor_index'].keys()
            if len(cache['neighbor_index'][i]) > 1]

    def __len__(self):
        """Return number of valid samples in dataset."""
        return len(self.valid_sample_index)

    def __getitem__(self, index):
        """Get a single sample from the dataset.
        
        Returns a dictionary with:
        - Feature information about neighboring residues
        - Label information about the target residue
        """
        index = self.valid_sample_index[index]
        # Get target residue information
        label_index = self.data['neighbor_index'][index][0]
        target_chain, target_id, target_name =\
            self.data['residue_info'][label_index]
        # Get neighbor residue indices
        neighbor_indexs = self.data['neighbor_index'][index][1:]
        main_chain_coord = self.data['main_chain_coord']
        
        # Extract information about neighbors
        chains, ids, residue_names =\
            zip(*[self.data['residue_info'][index] for index in neighbor_indexs])
            
        return {
            'feature': {
                'neighbor_name': residue_names,
                'neighbor_id': ids,
                'neighbor_chain': chains,
                'neighbor_main_chain_coord': main_chain_coord[neighbor_indexs, ...],
                'neighbor_b_factor': self.data['b_factor'][neighbor_indexs, ...],
                'target_main_chain_coord': main_chain_coord[label_index],
                'target_id': target_id,
                'target_chain': target_chain
            },
            'label': {
                'target_name': target_name,
                'target_b_factor': self.data['b_factor'][label_index]
            }
        }


def make_dataset(path, store_ds_cache=True, cache_name=None,
    pid_black_list=set(), radius=3, cache_suffix=''):
    """Create a dataset from protein structure files.
    
    Handles:
    - Single files or directories of files
    - Caching of processed datasets
    - Blacklisting specific proteins
    - Combining multiple files into a single dataset
    
    Args:
        path: Path to file, directory, or list of paths
        store_ds_cache: Whether to cache the dataset
        cache_name: Name for cache file
        pid_black_list: Set of protein IDs to exclude
        radius: Neighbor radius for calculations
        cache_suffix: Suffix for cache file naming
        
    Returns:
        PyTorch Dataset object containing processed protein data
    """
    # Handle list of paths
    if isinstance(path, list):
        return ConcatDataset([make_dataset(
            ele, store_ds_cache=store_ds_cache, cache_name=cache_name, radius=radius,
            pid_black_list=pid_black_list, cache_suffix=cache_suffix)
            for ele in path])
            
    # Handle directory of files
    if os.path.isdir(path):
        if cache_name:
            ds_pkl_path = os.path.join(path, f'{cache_name}.ds_pkl')
            # Load cached dataset if available
            if os.path.exists(ds_pkl_path):
                with open(ds_pkl_path, 'rb') as fin:
                    return pickle.load(fin)
                    
        datasets = []
        counter = 1
        print('Constructing datasets...')
        # Process each file in directory
        for file in filter(
                lambda x: os.path.splitext(x)[-1] in ('.pdb', '.cif'),
                os.listdir(path)):
            try:
                print(counter, file)
                pid = os.path.splitext(file)[0]
                # Skip blacklisted proteins
                if pid in pid_black_list:
                    print(f'Skip black list item {pid}.')
                else:
                    datasets.append(
                        NeighborDataset(os.path.join(path, file), radius=radius,
                        cache_suffix=cache_suffix))
            except ValueError:
                pass
            counter += 1
            
        # Combine all datasets
        db = ConcatDataset(datasets)
        # Save combined dataset if requested
        if cache_name:
            with open(ds_pkl_path, 'wb') as fout:
                pickle.dump(db, fout)
        return db
        
    # Handle pre-cached dataset file
    elif os.path.splitext(path)[-1] == '.ds_pkl':
        with open(path, 'rb') as fin:
            return pickle.load(fin)
            
    # Handle single file
    else:
        return NeighborDataset(path, radius=radius, cache_suffix=cache_suffix)
