"""Design a protein sequence given the backbone using a transformer-based model.

This script implements a protein sequence design pipeline that:
1. Takes a protein backbone structure as input (PDB or mmCIF format)
2. Uses a pre-trained transformer model to predict optimal amino acid sequences
3. Iteratively refines the sequence based on model predictions
4. Outputs designed sequences in FASTA format
"""
import argparse
import os
from random import sample
import torch
from Bio.PDB import PDBParser, FastMMCIFParser, PDBIO
from pe.common import residue_constants as rc  # Amino acid constants and mappings
from pe.common import convert  # Structure conversion utilities
from pe.model import modules  # Transformer model implementation
import evaluator  # Model evaluation utilities


def identity(str1, str2):
    """Calculate sequence identity between two strings.
    
    Args:
        str1: First sequence string
        str2: Second sequence string
        
    Returns:
        Fraction of matching positions between the two sequences
    """
    c = 0
    for a, b in zip(str1, str2):
        if a == b:
            c += 1
    return c / len(str1)


def read_constrain_file(constrain):
    """Read constraints file specifying fixed residue positions.
    
    Args:
        constrain: Path to constraints file
        
    Returns:
        Dictionary mapping residue indices (0-based) to constrained amino acids
    """
    result = {}
    with open(constrain) as fin:
        for line in map(str.rstrip, fin):
            tokens = line.split()
            index = int(tokens[0]) - 1  # Convert to 0-based index
            resi = tokens[1]
            result[index] = resi
    return result


def design(body, file_dir, constrain=None, log_file='', interval=1, args={}):
    """Main protein design function.
    
    Args:
        body: Pre-trained transformer model
        file_dir: Path to input structure file
        constrain: Dictionary of residue constraints
        log_file: Path to log file for saving intermediate results
        interval: Interval for logging intermediate results
        args: Additional arguments
        
    Returns:
        Dictionary containing designed sequence and other results
    """
    # Set up logging if specified
    if log_file:
        log_fin = open(log_file, 'w', encoding='utf-8')

    # Parse input file path and type
    file_dir_root, file_type = os.path.splitext(file_dir)
    pid = os.path.basename(file_dir_root)
    if file_type not in ('.pdb', '.cif'):
        raise TypeError(f'Only .pdb and .cif type are supported, get {file_type}')

    # Initialize appropriate structure parser
    protein_parser = PDBParser(QUIET=True) if file_type == '.pdb'\
        else FastMMCIFParser(QUIET=True)
    structure = protein_parser.get_structure('none', file_dir)
    
    # Validate single model structure
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]

    # Prepare main chain structure file
    main_chain_dir = os.path.join(args.cache_dir, f'{pid}_m.pdb')
    convert.reorder_model(model, 1)

    # Initialize and randomize starting sequence
    original_residues = []
    init_residues = []
    chains = list(model.get_chains())
    for chain in chains:
        for res in chain.get_residues():
            # Store original residue type
            original_residues.append(rc.restype_3to1.get(res.resname, 'X'))
            # Randomize residue type for initial sequence
            res.resname = sample(list(rc.restype_1to3.values()), k=1)[0]
            init_residues.append(rc.restype_3to1.get(res.resname, 'X'))
    original_residues = ''.join(original_residues)
    
    # Save main chain structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(main_chain_dir, convert.CustomSelect(only_main=True))

    # Log initial sequence if specified
    if log_file:
        init_residues = ''.join(init_residues)
        log_fin.write(init_residues)
        log_fin.write('\n')

    # Initialize sequence and accuracy tracking
    seqs = []
    accs = []
    
    # Set up optimization schedule
    if args.strict:
        len_stage_2 = len(original_residues) * 3
        ks = [1] * len_stage_2  # Single residue changes
    else:
        len_stage_1 = int(len(original_residues)/ 1)
        len_stage_2 = int(len(original_residues) / 5)
        ks = [5] * len_stage_1 + [1] * len_stage_2  # Start with multiple changes

    # Main optimization loop
    for i, k in enumerate(ks):
        # Evaluate current structure
        eval_result = evaluator.evaluate(body, main_chain_dir, return_logits=True,
            radius=3.5, return_is_optimal=True)
        acc = eval_result['accuracy']

        # Track accuracy after first iteration
        if i > 0:
            accs.append(acc)

        # Get model predictions
        logits = eval_result['logits']
        new_index = logits.argmax(dim=-1)
        
        # Apply constraints if specified
        if constrain is not None:
            for i, r in constrain.items():
                new_index[i] = rc.resname_to_idx[rc.restype_1to3[r]]
        
        # Convert predicted indices to sequence
        new_res = ''.join(rc.restypes_with_x[index] for index in new_index.tolist())
        
        # Reload structure for modification
        structure = protein_parser.get_structure('none', main_chain_dir)
        model = list(structure.get_models())[0]
        chains = list(model.get_chains())

        # Identify non-optimal positions to change
        non_optimal_list =\
            (~ eval_result['is_optimal']).nonzero().squeeze(dim=1).tolist()
        index_to_change = sample(non_optimal_list, min(len(non_optimal_list), k))
        
        # Add constrained positions to change list
        if constrain is not None:
            index_to_change.extend(list(constrain.keys()))

        # Apply changes to structure
        for j in index_to_change:
            old_residues = list(chains[0].get_residues())
            old_residues[j].resname = rc.restype_1to3.get(new_res[j], 'UNK')

        # Generate new sequence string
        new_res = ''.join(
            rc.restype_3to1.get(res.resname, 'X') for res in old_residues)
        identity_ = identity(new_res, original_residues)
        seqs.append(new_res)

        # Save modified structure
        io.set_structure(structure)
        io.save(main_chain_dir)

    # Select best sequence based on accuracy
    seqs.pop()
    accs = torch.tensor(accs)
    index_chosen = accs.argmax().item()

    result = {}
    result['seq'] = seqs[index_chosen]

    # Clean up logging
    if log_file:
        log_fin.close()
    return result


def main():
    """Main execution function."""
    # Initialize transformer model
    num_class = len(rc.restypes_with_x)
    body = modules.Transformer(46, num_class, 256, nhead=16, nlayer=3,
        device=args.device)
    
    # Load pre-trained weights
    stored = torch.load(args.store_dir, map_location='cpu')
    body.load_state_dict(stored['model_state_dict'])

    # Process input files
    input_files = []
    if os.path.isdir(args.input_path):
        for file in os.listdir(args.input_path):
            input_files.append(os.path.join(args.input_path, file))
    else:
        input_files.append(args.input_path)
    input_files = sorted(input_files)

    # Filter for valid structure files
    files_to_process = []
    for file in input_files:
        if os.path.splitext(file)[-1] in ('.cif', '.pdb'):
            files_to_process.append(file)
    print('Files to process:', len(files_to_process))

    # Read constraints if specified
    constrain=None
    if args.constrain:
        constrain=read_constrain_file(args.constrain)

    # Process each input file
    for file in files_to_process:
        pid = os.path.splitext(os.path.basename(file))[0]
        for i in range(args.index, args.index + args.num_seq):
            print(f'Designing {pid}_{i}')
            log_file_path = ''
            if args.log:
                log_file_path = os.path.join(args.cache_dir, f'{pid}_{i}.log')
            result = design(body, file, constrain, log_file=log_file_path,
                interval=args.interval, args=args)
            # Save results in FASTA format
            with open(os.path.join(
                    args.output_path, f'{pid}_{i}.fasta'), 'w', encoding='utf-8') as fout:
                fout.write(f'>{pid}\n')
                fout.write(f'{result["seq"]}\n')


if __name__ == '__main__':
    # Set up command line interface
    parser = argparse.ArgumentParser(description='Protein sequence design using transformer models')
    parser.add_argument('store_dir', type=str, help='Path to stored model weights')
    parser.add_argument('input_path', type=str, help='Input directory or structure file')
    parser.add_argument('-c', '--cache_dir', type=str, default='.', help='Cache directory for intermediate files')
    parser.add_argument('-o', '--output_path', type=str, default='.', help='Output directory for results')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run model on (cpu/cuda)')
    parser.add_argument('-C', '--constrain', type=str, help='Path to constraints file (1-indexed)')
    parser.add_argument('-S', '--strict', action='store_true', help='Use strict optimization schedule')
    parser.add_argument('-L', '--log', action='store_true', help='Enable logging of intermediate results')
    parser.add_argument('-I', '--interval', type=int, default=1,
        help='Interval for logging intermediate results')
    parser.add_argument('-n', '--num_seq', type=int, default=1,
        help='Number of sequences to generate per structure')
    parser.add_argument('-i', '--index', type=int, default=1, help='Starting index for output files')
    args = parser.parse_args()
    main()
