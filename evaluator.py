import argparse
from statistics import mean
import torch
from torch import nn
from Bio.PDB import PDBIO, StructureBuilder
from pe.data import loader, make_database
from pe.model import preprocess
from pe.common import util
from pe.common import residue_constants as rc
from pe.model import modules


def get_confidence(logits):
    """Calculate confidence scores from model logits using entropy.
    
    Args:
        logits: Raw model outputs of shape (B, C) where:
            B = batch size
            C = number of classes (amino acid types)
            
    Returns:
        Tensor of shape (B,) containing confidence scores for each prediction
        Lower values indicate higher confidence (lower entropy)
    """
    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=-1).detach()  # (B, C)
    
    # Calculate entropy: -sum(p * log(p)) for each prediction
    confidencies = (probs * torch.log(probs)).sum(-1)  # (B, )
    return confidencies


def get_acc(output, target, topk=1, reduction='mean'):
    """Calculate prediction accuracy.
    
    Args:
        output: Model predictions of shape (B, C)
        target: Ground truth labels of shape (B,)
        topk: Consider prediction correct if true label is in top k predictions
        reduction: 'mean' returns accuracy as fraction, 'sum' returns total count
        
    Returns:
        Accuracy score (float or int depending on reduction)
    """
    assert reduction in ('mean', 'sum')
    batch_size = len(target)
    
    # Get indices of top k predictions for each sample
    max_index = torch.topk(output, topk)[1]  # (B, k) <- (B, C)
    
    # Check if true label matches any of top k predictions
    bingo = (max_index == target.unsqueeze(-1))  # (B, k)
    
    # Count number of correct predictions
    num_correct = bingo.any(dim=1).sum()  # () <- (B, k)
    
    if reduction == 'sum':
        return num_correct
    else:
        return num_correct / batch_size


def evaluate(model, validate_file, topk=1, return_logits=False, by_chain=False,
            radius=3, cache_suffix='', cache_name=None, return_is_optimal=False, 
            top_k=5, mask_type=False):
    """Evaluate model performance on validation data.
    
    Args:
        model: Trained model to evaluate
        validate_file: Path to validation data file
        topk: Number of top predictions to consider for accuracy
        return_logits: Whether to return raw model outputs
        by_chain: Whether to calculate loss statistics per protein chain
        radius: Cutoff distance for neighborhood features
        cache_suffix: Suffix for cached data files
        cache_name: Name for cached data files
        return_is_optimal: Whether to return per-position optimality
        top_k: Number of top predictions to consider (unused in current implementation)
        mask_type: Whether to mask amino acid type features
        
    Returns:
        Dictionary containing evaluation metrics:
        - loss: Average loss across all samples
        - accuracy: Prediction accuracy
        - logits: Raw model outputs (if return_logits=True)
        - is_optimal: Per-position optimality (if return_is_optimal=True)
        - total_loss_by_chain: Per-chain total loss (if by_chain=True)
        - mean_loss_by_chain: Per-chain average loss (if by_chain=True)
    """
    result = {}
    model.eval()  # Set model to evaluation mode
    
    # Initialize containers for results
    if return_logits:
        logits = []
    
    # Create data loader for validation set
    validate = loader.get_loader(
        make_database.make_dataset(validate_file, radius=radius,
                                 cache_suffix=cache_suffix, cache_name=cache_name),
        device=util.get_model_device(model), batch_size=1000)
    
    # Initialize metrics
    total_loss = 0
    losses = []
    is_optimal = []
    total_loss_by_chain = {}
    total_acc = 0
    n_sample = 0

    # Initialize preprocessing and loss function
    process = preprocess.PreProcess()
    loss = nn.CrossEntropyLoss(reduction='none')

    # Process validation data in batches
    for batch in validate:
        # Preprocess features
        processed = process(batch['feature'])
        if mask_type:
            processed[..., :21] = 0  # Mask amino acid type features if specified
            
        # Get model predictions
        output = model(processed, mask=batch['mask'])
        
        # Calculate accuracy
        acc = get_acc(output, batch['label']['target_name'], topk=topk,
                     reduction='sum')
        
        # Track optimal positions if requested
        if return_is_optimal:
            is_optimal.append(output.argmax(dim=1) == batch['label']['target_name'])
            
        # Calculate loss
        l = loss(output, batch['label']['target_name'])

        # Update metrics
        total_loss += l.sum().detach().item()
        total_acc += acc.detach().item()
        
        # Track per-chain loss if requested
        if by_chain:
            for i, chain in enumerate(batch['feature']['target_chain_list']):
                if chain in total_loss_by_chain:
                    total_loss_by_chain[chain].append(l[i].item())
                else:
                    total_loss_by_chain[chain] = [l[i].item()]

        n_sample += processed.size(0)
        
        # Store logits if requested
        if return_logits:
            logits.append(output)

    # Calculate per-chain statistics if requested
    if by_chain:
        mean_loss_by_chain = {}
        for chain, losses in total_loss_by_chain.items():
            mean_loss_by_chain[chain] = mean(losses)
            total_loss_by_chain[chain] = sum(losses)

    # Compile final results
    result['loss'] = total_loss / n_sample
    if by_chain:
        result['total_loss_by_chain'] = total_loss_by_chain
        result['mean_loss_by_chain'] = mean_loss_by_chain
    result['accuracy'] = total_acc / n_sample
    if return_is_optimal:
        result['is_optimal'] = torch.cat(is_optimal)
    if return_logits:
        result['logits'] = torch.cat(logits)
    return result
