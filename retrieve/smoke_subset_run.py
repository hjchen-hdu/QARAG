import argparse
import logging
import math
import os
import pickle
import tempfile
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import prepare_sample, set_seed
from train import train_epoch


LOGGER = logging.getLogger(__name__)


def _build_config(
    use_structure_encoder: bool = True,
    use_semantic_encoder: bool = True
) -> dict:
    return {
        'env': {
            'num_threads': 1,
            'seed': 42,
        },
        'dataset': {
            'name': 'smoke',
            'text_encoder_name': 'gte-large-en-v1.5',
        },
        'retriever': {
            'topic_pe': True,
            'use_structure_encoder': use_structure_encoder,
            'use_semantic_encoder': use_semantic_encoder,
            'use_relation_query_sim': True,
            'use_triple_query_sim': True,
            'DDE_kwargs': {
                'num_rounds': 1,
                'num_reverse_rounds': 1,
            },
        },
        'optimizer': {
            'lr': 1e-3,
        },
        'eval': {
            'k_list': [1, 2],
        },
        'weak_supervision': {
            'strategy': 'query_path_soft',
            'near_shortest_delta': 0,
            'top_m_paths': 2,
            'max_paths_per_pair': 8,
            'negative_confidence': 0.05,
            'score_temperature': 1.0,
        },
        'train': {
            'num_epochs': 1,
            'patience': 1,
            'save_prefix': 'smoke',
        },
    }


def _build_processed_sample(sample_id: str) -> dict:
    return {
        'id': sample_id,
        'question': 'Which entity is reached by relation one then relation two?',
        'q_entity': ['topic_entity'],
        'q_entity_id_list': [0],
        'text_entity_list': [
            'topic_entity',
            'bridge_entity',
            'answer_entity',
            'distractor_entity',
        ],
        'non_text_entity_list': [],
        'relation_list': [
            'relation_one',
            'relation_two',
            'distractor_relation',
        ],
        'h_id_list': [0, 1, 0],
        'r_id_list': [0, 1, 2],
        't_id_list': [1, 2, 3],
        'a_entity': ['answer_entity'],
        'a_entity_id_list': [2],
    }


def _build_embeddings(sample_id: str, emb_size: int = 8) -> dict:
    q_emb = torch.zeros(emb_size)
    q_emb[0] = 1.

    entity_embs = torch.eye(emb_size)[:4].clone()

    relation_embs = torch.zeros(3, emb_size)
    relation_embs[0, 0] = 1.
    relation_embs[1, 0] = 1.
    relation_embs[2, 1] = 1.

    return {
        sample_id: {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs,
        }
    }


def _write_smoke_data(work_dir: Path, config: dict) -> None:
    dataset_name = config['dataset']['name']
    text_encoder_name = config['dataset']['text_encoder_name']
    processed_dir = work_dir / 'data_files' / dataset_name / 'processed'
    emb_dir = work_dir / 'data_files' / dataset_name / 'emb' / text_encoder_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val', 'test']:
        sample_id = f'smoke_{split}_0'
        processed_path = processed_dir / f'{split}.pkl'
        with processed_path.open('wb') as file_handle:
            pickle.dump([_build_processed_sample(sample_id)], file_handle)
        torch.save(_build_embeddings(sample_id), emb_dir / f'{split}.pth')


def _run_training(config: dict, device: torch.device) -> tuple[dict, Retriever]:
    train_set = RetrieverDataset(config=config, split='train')
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=False, collate_fn=collate_retriever)

    emb_size = train_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    optimizer = Adam(model.parameters(), **config['optimizer'])

    train_log = train_epoch(device, train_loader, model, optimizer)
    loss = train_log['loss']
    assert math.isfinite(loss), f'Non-finite smoke train loss: {loss}'

    return train_log, model


def _assert_dataset_labels(config: dict) -> None:
    train_set = RetrieverDataset(config=config, split='train')
    assert len(train_set) == 1
    sample = train_set[0]
    assert sample['target_triple_probs'].sum().item() > 0.
    assert sample['target_triple_confidence'].shape == sample[
        'target_triple_probs'].shape


@torch.no_grad()
def _run_inference(
    config: dict,
    checkpoint_path: Path,
    result_path: Path,
    max_k: int,
    device: torch.device
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    infer_set = RetrieverDataset(
        config=checkpoint['config'], split='test', skip_no_path=False)

    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pred_dict = {}
    for raw_sample in infer_set:
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, target_triple_confidence,\
            a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample[
            'non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_k_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_k_results = torch.topk(
                pred_triple_scores, min(max_k, len(pred_triple_scores)))

            for score, triple_id in zip(
                top_k_results.values.cpu().tolist(),
                top_k_results.indices.cpu().tolist()
            ):
                top_k_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    score,
                ))

            target_relevant_triple_ids = raw_sample[
                'target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        pred_dict[raw_sample['id']] = {
            'question': raw_sample['question'],
            'scored_triples': top_k_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [
                entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [
                entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples,
        }

    torch.save(pred_dict, result_path)


def _assert_retrieval_result(result_path: Path) -> None:
    assert result_path.exists(), f'Missing retrieval result: {result_path}'
    pred_dict = torch.load(result_path, map_location='cpu')
    assert len(pred_dict) == 1
    sample = next(iter(pred_dict.values()))
    assert len(sample['scored_triples']) > 0
    assert len(sample['target_relevant_triples']) > 0
    assert sample['a_entity_in_graph'] == ['answer_entity']


def run_smoke(
    work_dir: Path,
    keep_work_dir: bool = False,
    use_structure_encoder: bool = True,
    use_semantic_encoder: bool = True
) -> Path:
    config = _build_config(
        use_structure_encoder=use_structure_encoder,
        use_semantic_encoder=use_semantic_encoder)
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    device = torch.device('cpu')

    _write_smoke_data(work_dir, config)

    original_cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        _assert_dataset_labels(config)
        train_log, model = _run_training(config, device)

        run_dir = work_dir / 'smoke_run'
        run_dir.mkdir(exist_ok=True)
        checkpoint_path = run_dir / 'cpt.pth'
        torch.save({
            'config': config,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)

        result_path = run_dir / 'retrieval_result.pth'
        _run_inference(config, checkpoint_path, result_path, 2, device)
        _assert_retrieval_result(result_path)
    finally:
        os.chdir(original_cwd)

    LOGGER.info('Smoke train loss: %.6f', train_log['loss'])
    LOGGER.info('Smoke artifacts: %s', run_dir)
    if not keep_work_dir:
        LOGGER.info('Temporary work dir will be removed by the caller.')

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a tiny RetrieverDataset -> train_epoch -> inference smoke test.')
    parser.add_argument(
        '--work-dir',
        type=Path,
        default=None,
        help='Optional work directory for synthetic smoke data and artifacts.')
    parser.add_argument(
        '--keep-work-dir',
        action='store_true',
        help='Keep the temporary work directory when --work-dir is not provided.')
    parser.add_argument(
        '--no-structure-encoder',
        action='store_true',
        help='Disable DDE/topic structure features for ablation smoke testing.')
    parser.add_argument(
        '--no-semantic-encoder',
        action='store_true',
        help='Disable query-aware semantic features for ablation smoke testing.')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    use_structure_encoder = not args.no_structure_encoder
    use_semantic_encoder = not args.no_semantic_encoder

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_smoke(
            args.work_dir,
            keep_work_dir=True,
            use_structure_encoder=use_structure_encoder,
            use_semantic_encoder=use_semantic_encoder)
        return

    if args.keep_work_dir:
        work_dir = Path(tempfile.mkdtemp(prefix='qarag_smoke_'))
        run_smoke(
            work_dir,
            keep_work_dir=True,
            use_structure_encoder=use_structure_encoder,
            use_semantic_encoder=use_semantic_encoder)
        LOGGER.info('Kept temporary work dir: %s', work_dir)
        return

    with tempfile.TemporaryDirectory(prefix='qarag_smoke_') as temp_dir:
        run_smoke(
            Path(temp_dir),
            keep_work_dir=False,
            use_structure_encoder=use_structure_encoder,
            use_semantic_encoder=use_semantic_encoder)


if __name__ == '__main__':
    main()
