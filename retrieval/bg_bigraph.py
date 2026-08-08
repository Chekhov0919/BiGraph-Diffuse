from src.bg_emb_store import EmbeddingStore
from src.bg_utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.bg_ner import SpacyNER
import igraph as ig
import re
import logging
import torch

logger = logging.getLogger(__name__)


class BiGraphRAG:
    def __init__(self, global_config):
        self.bg_cfg = global_config
        logger.info(f"Initializing BiGraphRAG with config: {self.bg_cfg}")
        retrieval_method = "Vectorized Matrix-based" if self.bg_cfg.use_vectorized_retrieval else "BFS Iteration"
        logger.info(f"Using retrieval method: {retrieval_method}")

        self.bg_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.bg_cfg.use_vectorized_retrieval:
            logger.info(f"Using device: {self.bg_device} for vectorized retrieval")

        self.bg_dname = global_config.dataset_name
        self.load_embedding_store()
        self.bg_llm = self.bg_cfg.llm_model
        self.bg_ner = SpacyNER(self.bg_cfg.spacy_model)
        self.bg_graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        self.bg_pas_store = EmbeddingStore(self.bg_cfg.embedding_model,
                                                      db_filename=os.path.join(self.bg_cfg.working_dir,
                                                                               self.bg_dname,
                                                                               "passage_embedding.parquet"),
                                                      batch_size=self.bg_cfg.batch_size, namespace="passage")
        self.bg_ent_store = EmbeddingStore(self.bg_cfg.embedding_model,
                                                     db_filename=os.path.join(self.bg_cfg.working_dir,
                                                                              self.bg_dname,
                                                                              "entity_embedding.parquet"),
                                                     batch_size=self.bg_cfg.batch_size, namespace="entity")
        self.bg_sent_store = EmbeddingStore(self.bg_cfg.embedding_model,
                                                       db_filename=os.path.join(self.bg_cfg.working_dir,
                                                                                self.bg_dname,
                                                                                "sentence_embedding.parquet"),
                                                       batch_size=self.bg_cfg.batch_size, namespace="sentence")

    def load_existing_data(self, passage_hash_ids):
        self.bg_ner_path = os.path.join(self.bg_cfg.working_dir, self.bg_dname, "ner_results.json")
        if os.path.exists(self.bg_ner_path):
            existing_ner_reuslts = json.load(open(self.bg_ner_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations."""
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
        with ThreadPoolExecutor(max_workers=self.bg_cfg.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.bg_llm.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results

    def retrieve(self, questions):
        self.bg_ent_hids = list(self.bg_ent_store.bg_hid2text.keys())
        self.bg_ent_embs = np.array(self.bg_ent_store.bg_embs)
        self.bg_pas_hids = list(self.bg_pas_store.bg_hid2text.keys())
        self.bg_pas_embs = np.array(self.bg_pas_store.bg_embs)
        self.bg_sent_hids = list(self.bg_sent_store.bg_hid2text.keys())
        self.bg_sent_embs = np.array(self.bg_sent_store.bg_embs)
        self.bg_name2vidx = {v["name"]: v.index for v in self.bg_graph.vs if "name" in v.attributes()}
        self.bg_vidx2name = {v.index: v["name"] for v in self.bg_graph.vs if "name" in v.attributes()}

        if self.bg_cfg.use_vectorized_retrieval:
            logger.info("Precomputing sparse adjacency matrices for vectorized retrieval...")
            self._precompute_sparse_matrices()
            e2s_shape = self.bg_e2s_mat.shape
            s2e_shape = self.bg_s2e_mat.shape
            e2s_nnz = self.bg_e2s_mat._nnz()
            s2e_nnz = self.bg_s2e_mat._nnz()
            logger.info(f"Matrices built: Entity-Sentence {e2s_shape}, Sentence-Entity {s2e_shape}")
            logger.info(f"E2S Sparsity: {(1 - e2s_nnz / (e2s_shape[0] * e2s_shape[1])) * 100:.2f}% (nnz={e2s_nnz})")
            logger.info(f"S2E Sparsity: {(1 - s2e_nnz / (s2e_shape[0] * s2e_shape[1])) * 100:.2f}% (nnz={s2e_nnz})")
            logger.info(f"Device: {self.bg_device}")

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            question = question_info["question"]
            question_embedding = self.bg_cfg.embedding_model.encode(question, normalize_embeddings=True,
                                                                    show_progress_bar=False,
                                                                    batch_size=self.bg_cfg.batch_size)
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(
                question)
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = self.graph_search_with_seed_entities(question,
                                                                                                      question_embedding,
                                                                                                      seed_entity_indices,
                                                                                                      seed_entities,
                                                                                                      seed_entity_hash_ids,
                                                                                                      seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.bg_cfg.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.bg_cfg.retrieval_top_k]
                final_passages = [self.bg_pas_store.bg_hid2text[passage_hash_id] for passage_hash_id in
                                  final_passage_hash_ids]
            else:
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.bg_cfg.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.bg_cfg.retrieval_top_k]
                final_passages = [self.bg_pas_store.bg_texts[idx] for idx in final_passage_indices]
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        return retrieval_results

    def _precompute_sparse_matrices(self):
        num_entities = len(self.bg_ent_hids)
        num_sentences = len(self.bg_sent_hids)

        entity_to_sentence_indices = []
        entity_to_sentence_values = []

        for entity_hash_id, sentence_hash_ids in self.bg_ent2sent.items():
            entity_idx = self.bg_ent_store.bg_hid2idx[entity_hash_id]
            for sentence_hash_id in sentence_hash_ids:
                sentence_idx = self.bg_sent_store.bg_hid2idx[sentence_hash_id]
                entity_to_sentence_indices.append([entity_idx, sentence_idx])
                entity_to_sentence_values.append(1.0)

        sentence_to_entity_indices = []
        sentence_to_entity_values = []

        for sentence_hash_id, entity_hash_ids in self.bg_sent2ent.items():
            sentence_idx = self.bg_sent_store.bg_hid2idx[sentence_hash_id]
            for entity_hash_id in entity_hash_ids:
                entity_idx = self.bg_ent_store.bg_hid2idx[entity_hash_id]
                sentence_to_entity_indices.append([sentence_idx, entity_idx])
                sentence_to_entity_values.append(1.0)

        if len(entity_to_sentence_indices) > 0:
            e2s_indices = torch.tensor(entity_to_sentence_indices, dtype=torch.long).t()
            e2s_values = torch.tensor(entity_to_sentence_values, dtype=torch.float32)
            self.bg_e2s_mat = torch.sparse_coo_tensor(
                e2s_indices, e2s_values, (num_entities, num_sentences), device=self.bg_device
            ).coalesce()
        else:
            self.bg_e2s_mat = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_entities, num_sentences), device=self.bg_device
            )

        if len(sentence_to_entity_indices) > 0:
            s2e_indices = torch.tensor(sentence_to_entity_indices, dtype=torch.long).t()
            s2e_values = torch.tensor(sentence_to_entity_values, dtype=torch.float32)
            self.bg_s2e_mat = torch.sparse_coo_tensor(
                s2e_indices, s2e_values, (num_sentences, num_entities), device=self.bg_device
            ).coalesce()
        else:
            self.bg_s2e_mat = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32),
                (num_sentences, num_entities), device=self.bg_device
            )

    def graph_search_with_seed_entities(self, question, question_embedding, seed_entity_indices, seed_entities,
                                        seed_entity_hash_ids, seed_entity_scores):
        if self.bg_cfg.use_vectorized_retrieval:
            entity_weights, actived_entities = self.calculate_entity_scores_vectorized(question_embedding,
                                                                                       seed_entity_indices,
                                                                                       seed_entities,
                                                                                       seed_entity_hash_ids,
                                                                                       seed_entity_scores)
        else:
            entity_weights, actived_entities = self.calculate_entity_scores(question_embedding, seed_entity_indices,
                                                                            seed_entities, seed_entity_hash_ids,
                                                                            seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question, question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights):
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        pagerank_scores = self.bg_graph.personalized_pagerank(
            vertices=range(len(self.bg_name2vidx)),
            damping=self.bg_cfg.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.bg_pas_nidxs])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]

        sorted_passage_hash_ids = [
            self.bg_vidx2name[self.bg_pas_nidxs[i]]
            for i in sorted_indices_in_doc_scores
        ]

        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids,
                                seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.bg_graph.vs["name"]))
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(seed_entity_indices,
                                                                                        seed_entities,
                                                                                        seed_entity_hash_ids,
                                                                                        seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.bg_name2vidx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        iteration = 1
        while len(current_entities) > 0 and iteration < self.bg_cfg.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.bg_cfg.iteration_threshold:
                    continue
                sentence_hash_ids = [sid for sid in list(self.bg_ent2sent[entity_hash_id]) if
                                     sid not in used_sentence_hash_ids]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [self.bg_sent_store.bg_hid2idx[sid] for sid in sentence_hash_ids]
                sentence_embeddings = self.bg_sent_embs[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(
                    question_embedding.shape) == 1 else question_embedding
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                top_sentence_indices = np.argsort(sentence_similarities)[::-1][:self.bg_cfg.top_k_sentence]
                for top_sentence_index in top_sentence_indices:
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    entity_hash_ids_in_sentence = self.bg_sent2ent[top_sentence_hash_id]
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        if next_entity_score < self.bg_cfg.iteration_threshold:
                            continue
                        next_enitity_node_idx = self.bg_name2vidx[next_entity_hash_id]
                        entity_weights[next_enitity_node_idx] += next_entity_score
                        new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration + 1)
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, actived_entities

    def calculate_entity_scores_vectorized(self, question_embedding, seed_entity_indices, seed_entities,
                                           seed_entity_hash_ids, seed_entity_scores):
        entity_weights = np.zeros(len(self.bg_graph.vs["name"]))
        num_entities = len(self.bg_ent_hids)
        num_sentences = len(self.bg_sent_hids)

        question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
        sentence_similarities_np = np.dot(self.bg_sent_embs, question_emb).flatten()

        sentence_similarities = torch.from_numpy(sentence_similarities_np).float().to(self.bg_device)

        used_sentence_mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.bg_device)

        seed_indices = torch.tensor([[idx] for idx in seed_entity_indices], dtype=torch.long).t()
        seed_values = torch.tensor(seed_entity_scores, dtype=torch.float32)
        entity_scores_sparse = torch.sparse_coo_tensor(
            seed_indices, seed_values, (num_entities,), device=self.bg_device
        ).coalesce()

        entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.bg_device)
        entity_scores_dense.scatter_(0, torch.tensor(seed_entity_indices, device=self.bg_device),
                                     torch.tensor(seed_entity_scores, dtype=torch.float32, device=self.bg_device))

        actived_entities = {}
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(
                seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores
        ):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 0)
            seed_entity_node_idx = self.bg_name2vidx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score

        current_entity_scores_sparse = entity_scores_sparse

        for iteration in range(1, self.bg_cfg.max_iterations):
            current_entity_scores_dense = current_entity_scores_sparse.to_dense()

            current_entity_scores_dense = torch.where(
                current_entity_scores_dense >= self.bg_cfg.iteration_threshold,
                current_entity_scores_dense,
                torch.zeros_like(current_entity_scores_dense)
            )

            nonzero_mask = current_entity_scores_dense > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=False).squeeze(-1)

            if len(nonzero_indices) == 0:
                break

            nonzero_values = current_entity_scores_dense[nonzero_indices]
            current_entity_scores_sparse = torch.sparse_coo_tensor(
                nonzero_indices.unsqueeze(0), nonzero_values, (num_entities,), device=self.bg_device
            ).coalesce()

            current_scores_2d = torch.sparse_coo_tensor(
                torch.stack([nonzero_indices, torch.zeros_like(nonzero_indices)]),
                nonzero_values,
                (num_entities, 1),
                device=self.bg_device
            ).coalesce()

            sentence_activation = torch.sparse.mm(
                self.bg_e2s_mat.t(),
                current_scores_2d
            )
            if sentence_activation.is_sparse:
                sentence_activation = sentence_activation.to_dense()
            sentence_activation = sentence_activation.squeeze()

            sentence_activation = torch.where(
                used_sentence_mask,
                torch.zeros_like(sentence_activation),
                sentence_activation
            )

            selected_sentence_indices_list = []

            if len(nonzero_indices) > 0 and self.bg_cfg.top_k_sentence > 0:
                for i, entity_idx in enumerate(nonzero_indices):
                    entity_score = nonzero_values[i]

                    entity_row = self.bg_e2s_mat[entity_idx].coalesce()
                    entity_sentence_indices = entity_row.indices()[0]

                    if len(entity_sentence_indices) == 0:
                        continue

                    sentence_mask = ~used_sentence_mask[entity_sentence_indices]
                    available_sentence_indices = entity_sentence_indices[sentence_mask]

                    if len(available_sentence_indices) == 0:
                        continue

                    sentence_sims = sentence_similarities[available_sentence_indices]

                    k = min(self.bg_cfg.top_k_sentence, len(sentence_sims))
                    if k > 0:
                        top_k_values, top_k_local_indices = torch.topk(sentence_sims, k)
                        top_k_sentence_indices = available_sentence_indices[top_k_local_indices]
                        selected_sentence_indices_list.append(top_k_sentence_indices)

                if len(selected_sentence_indices_list) > 0:
                    all_selected_sentences = torch.cat(selected_sentence_indices_list)
                    unique_selected_sentences = torch.unique(all_selected_sentences)

                    used_sentence_mask[unique_selected_sentences] = True

                    weighted_sentence_scores = sentence_activation * sentence_similarities

                    mask = torch.zeros(num_sentences, dtype=torch.bool, device=self.bg_device)
                    mask[unique_selected_sentences] = True
                    weighted_sentence_scores = torch.where(
                        mask,
                        weighted_sentence_scores,
                        torch.zeros_like(weighted_sentence_scores)
                    )
                else:
                    weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.bg_device)
            else:
                weighted_sentence_scores = torch.zeros(num_sentences, dtype=torch.float32, device=self.bg_device)

            weighted_nonzero_mask = weighted_sentence_scores > 0
            weighted_nonzero_indices = torch.nonzero(weighted_nonzero_mask, as_tuple=False).squeeze(-1)

            if len(weighted_nonzero_indices) > 0:
                weighted_nonzero_values = weighted_sentence_scores[weighted_nonzero_indices]
                weighted_scores_2d = torch.sparse_coo_tensor(
                    torch.stack([weighted_nonzero_indices, torch.zeros_like(weighted_nonzero_indices)]),
                    weighted_nonzero_values,
                    (num_sentences, 1),
                    device=self.bg_device
                ).coalesce()

                next_entity_scores_result = torch.sparse.mm(
                    self.bg_s2e_mat.t(),
                    weighted_scores_2d
                )
                if next_entity_scores_result.is_sparse:
                    next_entity_scores_result = next_entity_scores_result.to_dense()
                next_entity_scores_dense = next_entity_scores_result.squeeze()
            else:
                next_entity_scores_dense = torch.zeros(num_entities, dtype=torch.float32, device=self.bg_device)
            entity_scores_dense += next_entity_scores_dense

            next_entity_scores_np = next_entity_scores_dense.cpu().numpy()
            active_indices = np.where(next_entity_scores_np >= self.bg_cfg.iteration_threshold)[0]
            for entity_idx in active_indices:
                score = next_entity_scores_np[entity_idx]
                entity_hash_id = self.bg_ent_hids[entity_idx]
                actived_entities[entity_hash_id] = (entity_idx, float(score), iteration)

            next_nonzero_mask = next_entity_scores_dense > 0
            next_nonzero_indices = torch.nonzero(next_nonzero_mask, as_tuple=False).squeeze(-1)
            if len(next_nonzero_indices) > 0:
                next_nonzero_values = next_entity_scores_dense[next_nonzero_indices]
                current_entity_scores_sparse = torch.sparse_coo_tensor(
                    next_nonzero_indices.unsqueeze(0), next_nonzero_values,
                    (num_entities,), device=self.bg_device
                ).coalesce()
            else:
                break

        entity_scores_final = entity_scores_dense.cpu().numpy()
        nonzero_indices = np.where(entity_scores_final > 0)[0]
        for entity_idx in nonzero_indices:
            score = entity_scores_final[entity_idx]
            entity_hash_id = self.bg_ent_hids[entity_idx]
            entity_node_idx = self.bg_name2vidx[entity_hash_id]
            entity_weights[entity_node_idx] = float(score)

        return entity_weights, actived_entities

    def calculate_passage_scores(self, question, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.bg_graph.vs["name"]))
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)
        apply_attribute_boost = (
                self.bg_cfg.enable_hybrid_attribute_fallback
                and self._is_attribute_query(question)
        )
        question_lower = question.lower()

        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0
            passage_hash_id = self.bg_pas_store.bg_hids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.bg_pas_store.bg_hid2text[passage_hash_id].lower()
            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.bg_ent_store.bg_hid2text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                if entity_occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

            passage_score = self.bg_cfg.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)

            if apply_attribute_boost:
                overlap = self._attribute_keyword_overlap(question_lower, passage_text_lower)
                if overlap > 0:
                    passage_score += self.bg_cfg.attribute_keyword_boost * math.log(1 + overlap)

            passage_node_idx = self.bg_name2vidx[passage_hash_id]
            passage_weights[passage_node_idx] = passage_score * self.bg_cfg.passage_node_weight
        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.bg_pas_embs, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores

    def _is_attribute_query(self, question):
        tokens = set(re.findall(r"\w+", question.lower()))
        return any(keyword in tokens for keyword in self.bg_cfg.attribute_query_keywords)

    def _attribute_keyword_overlap(self, question_lower, passage_text_lower):
        overlap = 0
        for keyword in self.bg_cfg.attribute_query_keywords:
            if keyword in question_lower and keyword in passage_text_lower:
                overlap += 1
        return overlap

    def get_seed_entities(self, question):
        question_entities = list(self.bg_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], []
        question_entity_embeddings = self.bg_cfg.embedding_model.encode(question_entities, normalize_embeddings=True,
                                                                        show_progress_bar=False,
                                                                        batch_size=self.bg_cfg.batch_size)
        similarities = np.dot(self.bg_ent_embs, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idx = np.argmax(entity_scores)
            best_entity_score = entity_scores[best_entity_idx]
            best_entity_hash_id = self.bg_ent_hids[best_entity_idx]
            best_entity_text = self.bg_ent_store.bg_hid2text[best_entity_hash_id]
            seed_entity_indices.append(best_entity_idx)
            seed_entity_texts.append(best_entity_text)
            seed_entity_hash_ids.append(best_entity_hash_id)
            seed_entity_scores.append(best_entity_score)
        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self.bg_n2n_stats = defaultdict(dict)
        self.bg_e2s_stats = defaultdict(dict)
        self.bg_pas_store.insert_text(passages)
        hash_id_to_passage = self.bg_pas_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(
            hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k: hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.bg_ner.batch_ner(new_hash_id_to_passage,
                                                                                                 self.bg_cfg.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities,
                                   new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.bg_e2s, self.bg_s2e = self.extract_nodes_and_edges(
            existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.bg_sent_store.insert_text(list(sentence_nodes))
        self.bg_ent_store.insert_text(list(entity_nodes))
        self.bg_ent2sent = {}
        for entity, sentence in self.bg_e2s.items():
            entity_hash_id = self.bg_ent_store.bg_text2hid[entity]
            self.bg_ent2sent[entity_hash_id] = [self.bg_sent_store.bg_text2hid[s]
                                                                        for s in sentence]
        self.bg_sent2ent = {}
        for sentence, entities in self.bg_s2e.items():
            sentence_hash_id = self.bg_sent_store.bg_text2hid[sentence]
            self.bg_sent2ent[sentence_hash_id] = [self.bg_ent_store.bg_text2hid[e]
                                                                          for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(self.bg_cfg.working_dir, self.bg_dname, "BiGraphRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)
        self.bg_graph.write_graphml(output_graphml_path)

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.bg_pas_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.bg_n2n_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.bg_graph.vs if "name" in v.attributes()}
        entity_hash_id_to_text = self.bg_ent_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.bg_pas_store.get_hash_id_to_text()
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        passage_hash_ids = set(passage_hash_id_to_text.keys())

        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.bg_graph.add_vertex(name=hash_id, content=text)

        self.bg_name2vidx = {v["name"]: v.index for v in self.bg_graph.vs if "name" in v.attributes()}
        self.bg_pas_nidxs = [
            self.bg_name2vidx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.bg_name2vidx
        ]

    def add_edges(self):
        edges = []
        weights = []

        for node_hash_id, node_to_node_stats in self.bg_n2n_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.bg_graph.add_edges(edges)
        self.bg_graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.bg_pas_store.bg_hid2text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.bg_ent_store.bg_text2hid[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.bg_n2n_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence = defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities,
                          new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.bg_ner_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities,
                       "sentence_to_entities": existing_sentence_to_entities}, f)
