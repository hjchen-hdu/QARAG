import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F
import wandb

from collections import defaultdict
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_yaml
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

# QARAG NOTE:
# This training loop is intentionally kept close to SubgraphRAG. The first
# extension should be local and easy to ablate:
#   baseline BCE
#   + query-aware soft labels
#   + confidence weighting
#   + bottleneck/sparsity regularization
# Avoid adding LLM supervision or path decoding in this file for the first
# QARAG iteration.

@torch.no_grad()
def eval_epoch(config, device, data_loader, model):
    model.eval()
    
    metric_dict = defaultdict(list)
    
    for sample in tqdm(data_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, target_triple_confidence,\
        a_entity_id_list = prepare_sample(device, sample)

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot).reshape(-1)
        
        # Triple ranking.
        #
        # QARAG TODO 1 - new retrieval metrics:
        # Keep this recall calculation, then add metrics for:
        #   - soft-label weighted recall@K
        #   - selected subgraph size
        #   - answer entity recall under adaptive budget
        #   - optional path coverage after connector/path closure
        sorted_triple_ids_pred = torch.argsort(
            pred_triple_logits, descending=True).cpu()
        triple_ranks_pred = torch.empty_like(sorted_triple_ids_pred)
        triple_ranks_pred[sorted_triple_ids_pred] = torch.arange(
            len(triple_ranks_pred))
        
        target_triple_ids = target_triple_probs.nonzero().squeeze(-1)
        num_target_triples = len(target_triple_ids)
        
        if num_target_triples == 0:
            continue

        num_total_entities = len(entity_embs) + num_non_text_entities
        for k in config['eval']['k_list']:
            recall_k_sample = (
                triple_ranks_pred[target_triple_ids] < k).sum().item()
            metric_dict[f'triple_recall@{k}'].append(
                recall_k_sample / num_target_triples)
            
            triple_mask_k = triple_ranks_pred < k
            entity_mask_k = torch.zeros(num_total_entities)
            entity_mask_k[h_id_tensor[triple_mask_k]] = 1.
            entity_mask_k[t_id_tensor[triple_mask_k]] = 1.
            recall_k_sample_ans = entity_mask_k[a_entity_id_list].sum().item()
            metric_dict[f'ans_recall@{k}'].append(
                recall_k_sample_ans / len(a_entity_id_list))

    for key, val in metric_dict.items():
        metric_dict[key] = np.mean(val)
    
    return metric_dict

def train_epoch(device, train_loader, model, optimizer):
    model.train()
    epoch_loss = 0
    for sample in tqdm(train_loader):
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, target_triple_confidence,\
        a_entity_id_list = prepare_sample(device, sample)
            
        if len(h_id_tensor) == 0:
            continue

        pred_triple_logits = model(
            h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
            num_non_text_entities, relation_embs, topic_entity_one_hot)
        target_triple_probs = target_triple_probs.to(device).unsqueeze(-1)

        # Baseline objective: binary BCE against shortest-path labels.
        #
        # FIXED(QARAG): `target_triple_probs` is now a dense soft label and
        # `target_triple_confidence` downweights weak negatives outside selected
        # query-matched answer paths.
        #
        # TODO(QARAG): Add ablations for:
        #   1. hard labels with confidence weighting;
        #   2. noisy-or triple aggregation;
        #   3. ranking loss over high-confidence positives vs weak negatives.
        target_triple_confidence = target_triple_confidence.to(device).unsqueeze(-1)
        loss_raw = F.binary_cross_entropy_with_logits(
            pred_triple_logits, target_triple_probs, reduction='none')
        loss = (target_triple_confidence * loss_raw).sum() /\
            target_triple_confidence.sum().clamp_min(1e-8)

        # QARAG TODO 3 - evidence bottleneck:
        # Add a sparse Bernoulli prior over the selected edge mask:
        #   pi = sigmoid(pred_triple_logits)
        #   rho = config['bottleneck']['prior_rate']  # e.g. 0.05 or 0.1
        #   kl = KL(Bernoulli(pi) || Bernoulli(rho))
        #   size = pi.mean()
        #   loss = weak_loss + beta * kl + lambda_size * size
        # Keep beta/lambda configurable so this is easy to ablate.

        # QARAG TODO 4 - ranking loss:
        # For high-confidence positives and reliable negatives, add a margin loss:
        #   max(0, margin - s_pos + s_neg)
        # This is useful when non-path triples are treated as unlabeled rather than
        # hard negatives.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        epoch_loss += loss
    
    epoch_loss /= len(train_loader)
    
    log_dict = {'loss': epoch_loss}
    return log_dict

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/retriever/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    device = torch.device('cuda:0')
    torch.set_num_threads(config['env']['num_threads'])
    set_seed(config['env']['seed'])

    ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    config_df = pd.json_normalize(config, sep='/')
    exp_prefix = config['train']['save_prefix']
    exp_name = f'{exp_prefix}_{ts}'
    wandb.init(
        project=f'{args.dataset}',
        name=exp_name,
        config=config_df.to_dict(orient='records')[0]
    )
    os.makedirs(exp_name, exist_ok=True)

    train_set = RetrieverDataset(config=config, split='train')
    val_set = RetrieverDataset(config=config, split='val')

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever)
    val_loader = DataLoader(
        val_set, batch_size=1, collate_fn=collate_retriever)
    
    emb_size = train_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    num_patient_epochs = 0
    best_val_metric = 0
    for epoch in range(config['train']['num_epochs']):
        num_patient_epochs += 1
        
        val_eval_dict = eval_epoch(config, device, val_loader, model)
        target_val_metric = val_eval_dict['triple_recall@100']
        
        if target_val_metric > best_val_metric:
            num_patient_epochs = 0
            best_val_metric = target_val_metric
            best_state_dict = {
                'config': config,
                'model_state_dict': model.state_dict()
            }
            torch.save(best_state_dict, os.path.join(exp_name, f'cpt.pth'))

            val_log = {'val/epoch': epoch}
            for key, val in val_eval_dict.items():
                val_log[f'val/{key}'] = val
            wandb.log(val_log)

        train_log_dict = train_epoch(device, train_loader, model, optimizer)
        
        train_log_dict.update({
            'num_patient_epochs': num_patient_epochs,
            'epoch': epoch
        })
        wandb.log(train_log_dict)
        if num_patient_epochs == config['train']['patience']:
            break

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
