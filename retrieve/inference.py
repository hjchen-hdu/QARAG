import os
import torch

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

# QARAG NOTE:
# Inference currently emits a fixed top-K list of scored triples. QARAG should
# evolve this into compact evidence subgraph selection, but keep the output
# schema backward compatible with SubgraphRAG's reasoner at first.

@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0')
    
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, target_triple_confidence,\
            a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)

            # Baseline selection: keep the highest `max_K` triples.
            #
            # QARAG TODO 1 - adaptive evidence budget:
            # Replace or augment top-K with:
            #   - score threshold;
            #   - expected mask budget from bottleneck prior;
            #   - min/max selected triples;
            #   - query-type-aware budget if available.
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            # QARAG TODO 2 - connectivity repair:
            # After selecting high-score triples, check whether the evidence graph
            # is connected to at least one topic entity. If not, add a small number
            # of connector triples using shortest paths or flow-guided paths.
            #
            # QARAG TODO 3 - path serialization support:
            # Add optional fields such as:
            #   'evidence_paths': [...]
            #   'connector_triples': [...]
            # so the reasoner can receive both a compact subgraph and readable
            # topic-centered paths.

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            # QARAG TODO 4 - keep retriever diagnostics:
            # Store selected budget, threshold, average flow prior, and number of
            # connected components. These diagnostics will be useful when tuning
            # bottleneck strength and prompt size.
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }
        
        pred_dict[raw_sample['id']] = sample_dict

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()
    
    main(args)
