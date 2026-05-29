import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F

from tqdm import tqdm

# QARAG NOTE:
# This file is the data-side control point of the retriever. In the original
# SubgraphRAG code, it builds binary shortest-path labels:
#   y_tau = 1 if triple tau lies on a shortest path from a topic entity to an
#   answer entity; otherwise y_tau = 0.
#
# For QARAG, keep the public sample format stable at first, but replace or
# augment `target_triple_probs` with query-aware soft evidence labels later.
# The intended extensions are documented inline with `QARAG TODO`.

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Load pre-computed embeddings.
        emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # FIXED(QARAG): Query-aware weak supervision is implemented here. Answer
        # entities define candidate paths, while query-relation similarity selects
        # which paths produce stronger soft triple labels.
        #
        # TODO(QARAG): Move this into a dedicated QueryAwareWeakLabelBuilder once
        # additional variants such as flow prior, noisy-or aggregation, or path
        # diversity are added.
        weak_config = config.get('weak_supervision', {})
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list, emb_dict, weak_config)

        # Put everything together.
        self._assembly(
            processed_dict_list, triple_score_dict, emb_dict, skip_no_path)

    def _load_processed(
        self,
        dataset_name,
        split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list,
        emb_dict,
        weak_config
    ):
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = self._get_triple_score_file(save_dir, split, weak_config)

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, triple_confidence_i, max_path_length_i =\
                self._extract_paths_and_score(
                    sample_i, emb_dict[sample_i_id], weak_config)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'triple_confidence': triple_confidence_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    def _get_triple_score_file(
        self,
        save_dir,
        split,
        weak_config
    ):
        strategy = weak_config.get('strategy', 'shortest_hard')
        if strategy == 'shortest_hard':
            return os.path.join(save_dir, f'{split}.pth')

        # FIXED(QARAG): Soft-label caches are separated from SubgraphRAG hard
        # shortest-path caches to avoid accidentally reusing stale labels.
        # TODO(QARAG): Move cache naming into the future label-builder module
        # when more scoring variants are added.
        delta = weak_config.get('near_shortest_delta', 0)
        top_m = weak_config.get('top_m_paths', 10)
        max_paths = weak_config.get('max_paths_per_pair', 50)
        temperature = str(weak_config.get('score_temperature', 1.0)).replace('.', 'p')
        neg_conf = str(weak_config.get('negative_confidence', 0.05)).replace('.', 'p')
        return os.path.join(
            save_dir,
            f'{split}_{strategy}_delta{delta}_top{top_m}_max{max_paths}_'
            f'temp{temperature}_neg{neg_conf}.pth')

    def _extract_paths_and_score(
        self,
        sample,
        emb_sample,
        weak_config
    ):
        # QARAG TODO 2 - candidate path expansion:
        # This function currently only asks for all shortest paths. To improve
        # weak labels without using an LLM, add a near-shortest path collector:
        #   paths = shortest_paths union simple_paths(cutoff=shortest_len + delta)
        # Keep delta small, e.g. 1 or 2, to avoid noisy path explosion.
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs. These paths are structural only:
        # they do not know whether the relation sequence matches the query.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._candidate_paths(
                    nx_g, q_entity_id, a_entity_id, weak_config)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Each processed path is represented by its triple IDs and a query-aware
        # confidence score. Path structure is only used in Dataset construction;
        # the training loop receives dense triple-level soft labels.
        #
        # FIXED(QARAG): Path score is intentionally simple:
        #   cosine(q_emb, mean relation_embs along the path).
        # TODO(QARAG): Try relation-sequence text encoding, flow_prior, or length
        # penalty only after this minimal version is ablated.
        path_records = []

        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i = nx_g[h_id_i][t_id_i]['triple_id']
                triples_path.append(triple_id_i)

            if len(triples_path) > 0:
                path_records.append({
                    'triple_ids': triples_path,
                    'score': self._score_path_by_query_relation(
                        triples_path, sample, emb_sample)
                })

        num_triples = len(sample['h_id_list'])
        triple_scores, triple_confidence = self._score_triples(
            path_records, num_triples, weak_config)
        
        return triple_scores, triple_confidence, max_path_length

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths

    def _candidate_paths(
        self,
        nx_g,
        q_entity_id,
        a_entity_id,
        weak_config
    ):
        shortest_paths = self._shortest_path(nx_g, q_entity_id, a_entity_id)
        delta = weak_config.get('near_shortest_delta', 0)
        if (delta <= 0) or (len(shortest_paths) == 0):
            return shortest_paths

        # FIXED(QARAG): Optional near-shortest expansion is bounded by
        # `shortest_len + delta` and `max_paths_per_pair`.
        # TODO(QARAG): Replace this simple all_simple_paths expansion with a
        # k-shortest path collector if path explosion appears on CWQ.
        shortest_edge_len = min(len(path) - 1 for path in shortest_paths)
        cutoff = shortest_edge_len + delta
        max_paths = weak_config.get('max_paths_per_pair', 50)
        path_set = {tuple(path) for path in shortest_paths}

        for source, target in [
            (q_entity_id, a_entity_id),
            (a_entity_id, q_entity_id)
        ]:
            try:
                for path in nx.all_simple_paths(
                    nx_g, source, target, cutoff=cutoff):
                    path_set.add(tuple(path))
                    if len(path_set) >= max_paths:
                        break
            except:
                continue

        return [list(path) for path in path_set]

    def _score_path_by_query_relation(
        self,
        triple_ids,
        sample,
        emb_sample
    ):
        if len(triple_ids) == 0:
            return 0.

        # FIXED(QARAG): Minimal query-aware path score. It uses existing
        # precomputed embeddings, so no extra encoder call or LLM scoring is
        # introduced during Dataset construction.
        # TODO(QARAG): Add a mode that embeds the literal relation sequence text
        # if mean relation embeddings are too weak for compositional queries.
        relation_ids = torch.tensor([
            sample['r_id_list'][triple_id] for triple_id in triple_ids
        ])
        q_emb = emb_sample['q_emb'].reshape(-1)
        relation_embs = emb_sample['relation_embs'][relation_ids]
        path_emb = relation_embs.mean(dim=0)
        score = F.cosine_similarity(
            q_emb.unsqueeze(0), path_emb.unsqueeze(0)).item()

        return score

    def _score_triples(
        self,
        path_records,
        num_triples,
        weak_config
    ):
        # Baseline label aggregation: any triple appearing on a selected path is
        # marked as positive with label 1.
        #
        # QARAG TODO 4 - soft label aggregation:
        # Change `path_list` from `List[List[List[int]]]` to include path weights:
        #   weighted_paths = [(path_triple_ids, path_weight), ...]
        # Then aggregate into soft triple confidence:
        #   y_tau = max_P w_P                 if tau in P
        # or:
        #   y_tau = 1 - product_P(1 - w_P)    if tau in P
        # Non-path triples should not automatically become strong negatives.
        # Prefer confidence-weighted BCE or PU-style negative sampling in train.py.
        triple_scores = torch.zeros(num_triples)

        negative_confidence = weak_config.get('negative_confidence', 0.05)
        triple_confidence = torch.full((num_triples,), negative_confidence)
        strategy = weak_config.get('strategy', 'shortest_hard')

        if strategy == 'shortest_hard':
            triple_confidence = torch.ones(num_triples)
            for path_record in path_records:
                for triple_id in path_record['triple_ids']:
                    triple_scores[triple_id] = 1.
            return triple_scores, triple_confidence

        # FIXED(QARAG): Keep only the top-M query-matched paths, then aggregate
        # path confidence into dense triple-level labels with max aggregation.
        # Unselected triples still get y=0, but only weak negative confidence.
        #
        # TODO(QARAG): Compare max aggregation with noisy-or:
        #   y_tau = 1 - product_P(1 - w_P)
        top_m = weak_config.get('top_m_paths', 10)
        if top_m > 0:
            path_records = sorted(
                path_records, key=lambda item: item['score'], reverse=True)
            path_records = path_records[:top_m]

        temperature = weak_config.get('score_temperature', 1.0)
        temperature = max(temperature, 1e-6)
        for path_record in path_records:
            path_confidence = torch.sigmoid(
                torch.tensor(path_record['score'] / temperature)).item()
            for triple_id in path_record['triple_ids']:
                triple_scores[triple_id] = max(
                    triple_scores[triple_id].item(), path_confidence)
                triple_confidence[triple_id] = max(
                    triple_confidence[triple_id].item(), path_confidence)

        return triple_scores, triple_confidence

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        file_path = f'data_files/{dataset_name}/emb/{text_encoder_name}/{split}.pth'
        dict_file = torch.load(file_path)
        
        return dict_file

    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            triple_confidence_i = triple_score_dict[sample_i_id].get(
                'triple_confidence', torch.ones_like(triple_score_i))
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            # FIXED(QARAG): Carry per-triple label confidence into the training
            # loop. Triples outside selected paths are weak negatives, not hard
            # negatives.
            #
            # TODO(QARAG): Add a separate confidence source for near-path triples
            # if later we want auxiliary evidence instead of pure negatives.
            sample_i['target_triple_confidence'] = triple_confidence_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i.update(emb_dict[sample_i_id])

            # QARAG TODO 6 - flow prior features:
            # This is the best place to attach precomputed per-triple features
            # because the graph, text embeddings, and question embedding are all
            # available per sample. A minimal first version can store:
            #   sample_i['flow_prior'] = Tensor[num_triples]
            #   sample_i['relation_query_sim'] = Tensor[num_triples]
            #   sample_i['triple_query_sim'] = Tensor[num_triples]
            # `collate_retriever` and `Retriever.forward` then need to pass these
            # tensors through as optional model inputs.

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')

    def __len__(self):
        return len(self.processed_dict_list)
    
    def __getitem__(self, i):
        return self.processed_dict_list[i]

def collate_retriever(data):
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    # FIXED(QARAG): Collate now passes label confidence to train.py.
    # TODO(QARAG): Append flow/query feature tensors here when model-side feature
    # expansion is implemented.
    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['topic_entity_one_hot'], sample['target_triple_probs'],\
        sample['target_triple_confidence'], sample['a_entity_id_list']
