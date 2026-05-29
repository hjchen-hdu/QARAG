import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

# QARAG NOTE:
# This model is still the SubgraphRAG triple scorer. It scores every triple
# independently after adding topic-entity structural encodings. QARAG should
# keep this lightweight shape initially and add new signals as features rather
# than replacing it with a heavy path decoder.

class PEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, edge_index, x):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(PEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(PEConv())
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index
    ):
        # DDE propagates topic-entity indicators in both directions. This gives
        # each entity a cheap structural feature: how it is positioned relative
        # to the question/topic entities.
        result_list = []
        
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev)
            result_list.append(h_pe_rev)
        
        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        use_structure_encoder=True,
        use_semantic_encoder=False,
        use_relation_query_sim=True,
        use_triple_query_sim=True
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.use_structure_encoder = use_structure_encoder
        self.use_semantic_encoder = use_semantic_encoder
        self.use_relation_query_sim = use_relation_query_sim
        self.use_triple_query_sim = use_triple_query_sim
        self.dde = DDE(**DDE_kwargs)
        
        pred_in_size = 4 * emb_size
        if use_structure_encoder:
            if topic_pe:
                pred_in_size += 2 * 2
            pred_in_size += 2 * 2 * (
                DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        if use_semantic_encoder:
            semantic_in_size = 5 * emb_size
            if use_relation_query_sim:
                semantic_in_size += 1
            if use_triple_query_sim:
                semantic_in_size += 1

            self.semantic_encoder = nn.Sequential(
                nn.Linear(semantic_in_size, emb_size),
                nn.ReLU(),
                nn.Linear(emb_size, emb_size),
                nn.ReLU()
            )
            pred_in_size += emb_size
        else:
            self.semantic_encoder = None

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

        # QARAG TODO 2 - bottleneck parameterization:
        # The simplest bottleneck can reuse this logit as the Bernoulli mask logit:
        #   pi_tau = sigmoid(pred_logit_tau)
        # A later version can add a separate `mask_head` if you want to decouple
        # relevance score and selection probability.

    def _get_base_entity_embs(
        self,
        entity_embs,
        num_non_text_entities,
        device
    ):
        return torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(
                        num_non_text_entities, -1)
            ],
            dim=0
        )

    def _encode_structure(
        self,
        base_entity_embs,
        topic_entity_one_hot,
        h_id_tensor,
        t_id_tensor
    ):
        if not self.use_structure_encoder:
            return base_entity_embs

        h_e_list = [base_entity_embs]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        dde_list = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index)
        h_e_list.extend(dde_list)

        return torch.cat(h_e_list, dim=1)

    def _encode_semantics(
        self,
        q_emb,
        h_head,
        h_relation,
        h_tail
    ):
        q_expand = q_emb.expand(len(h_relation), -1)
        semantic_feature_list = [
            q_expand,
            h_head,
            h_relation,
            h_tail,
            q_expand * h_relation
        ]

        if self.use_relation_query_sim:
            relation_query_sim = F.cosine_similarity(
                q_expand, h_relation, dim=1).unsqueeze(-1)
            semantic_feature_list.append(relation_query_sim)

        if self.use_triple_query_sim:
            triple_emb = (h_head + h_relation + h_tail) / 3.
            triple_query_sim = F.cosine_similarity(
                q_expand, triple_emb, dim=1).unsqueeze(-1)
            semantic_feature_list.append(triple_query_sim)

        semantic_input = torch.cat(semantic_feature_list, dim=1)
        return self.semantic_encoder(semantic_input)

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        base_entity_embs = self._get_base_entity_embs(
            entity_embs, num_non_text_entities, device)
        h_e = self._encode_structure(
            base_entity_embs, topic_entity_one_hot, h_id_tensor, t_id_tensor)

        h_q = q_emb.reshape(1, -1)
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]

        # Baseline triple representation:
        #   [question, head entity, relation, tail entity]
        # plus DDE-expanded entity states for head/tail.
        #
        h_triple_list = [
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor]
        ]

        if self.use_semantic_encoder:
            h_sem = self._encode_semantics(
                h_q,
                base_entity_embs[h_id_tensor],
                h_r,
                base_entity_embs[t_id_tensor]
            )
            h_triple_list.append(h_sem)

        h_triple = torch.cat(h_triple_list, dim=1)
        
        return self.pred(h_triple)
