import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
import os
import json
from sentence_transformers import SentenceTransformer

from src.ragconfig import BiGraphRAGConfig
from src.bigraph_rag import BiGraphRAG


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def llada_generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=64, temperature=0,
                   cfg_scale=0, remasking='low_confidence', mask_id=126336, logits_eos_inf=False,
                   confidence_eos_eot_inf=False):
    prompt_seq_len = prompt.shape[1]
    x = torch.full((prompt.shape[0], prompt_seq_len + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt_seq_len] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_seq_len + num_block * block_length
        block_end = prompt_seq_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


class BiGraphDiffuseRAG:
    def __init__(self, base_model_path, lora_path,
                 rag_corpus_path,
                 rag_dataset_name="llada_mental_corpus",
                 use_rag=True,
                 rag_top_k=1,
                 score_threshold=0.1):
        print("Loading base model & LoRA weights...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        assert self.tokenizer.pad_token_id != 126336, "pad_token_id conflicts with mask_id!"
        self.mask_id = 126336
        self.use_rag = use_rag
        self.rag_top_k = rag_top_k
        self.score_threshold = score_threshold
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model.eval()

        if self.use_rag:
            print(f"Initializing BiGraph-RAG, loading corpus: {rag_corpus_path}")
            self._init_bigraph_rag(rag_corpus_path, rag_dataset_name)
            print("BiGraph-RAG initialized, graph index built!")

    def _init_bigraph_rag(self, corpus_path, dataset_name):
        """Initialize BiGraph-RAG"""
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_text = f.read().replace("\n\n", "\n").strip()

        self.rag_passages = [p.strip() for p in raw_text.split("\n") if p.strip()]
        print(f"Corpus split into {len(self.rag_passages)} passages (BiGraph-RAG will auto-split sentences)")

        self.rag_embedding_model = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.rag_config = BiGraphRAGConfig(
            dataset_name=dataset_name,
            embedding_model=self.rag_embedding_model,
            spacy_model="en_core_web_trf",
            max_workers=8,
            max_iterations=3,
            retrieval_top_k=self.rag_top_k,
            use_vectorized_retrieval=False,
            iteration_threshold=0.4,
            passage_ratio=2.0,
            chunk_token_size=500,
            chunk_overlap_token_size=50,
            llm_model=None
        )

        self.bigraph_rag = BiGraphRAG(global_config=self.rag_config)
        self.bigraph_rag.index(self.rag_passages)

    def _bigraph_rag_retrieve(self, query):
        rag_query = [{"question": query, "answer": ""}]
        retrieval_result = self.bigraph_rag.retrieve(rag_query)[0]

        filtered_passages = []
        filtered_scores = []
        for passage, score in zip(retrieval_result["sorted_passage"], retrieval_result["sorted_passage_scores"]):
            if score >= self.score_threshold:
                filtered_passages.append(passage)
                filtered_scores.append(score)

        if not filtered_passages:
            filtered_passages = retrieval_result["sorted_passage"][:1]
            filtered_scores = retrieval_result["sorted_passage_scores"][:1]

        filtered_passages = filtered_passages[:1]
        filtered_scores = filtered_scores[:1]

        core_entities = []
        if hasattr(self.bigraph_rag, "actived_entities") and self.bigraph_rag.actived_entities:
            for entity_hash_id, (_, score, _) in self.bigraph_rag.actived_entities.items():
                if score >= self.score_threshold:
                    entity_text = self.bigraph_rag.entity_embedding_store.hash_id_to_text.get(entity_hash_id, "")
                    if entity_text:
                        core_entities.append(entity_text)
        core_entities = list(set(core_entities))

        entity_prompt = ""
        if core_entities:
            entity_prompt = "Relevant concepts: " + ", ".join(core_entities)
        max_passage_chars = 500
        passage_prompt = "\n".join([
            f"- {p.strip()[:max_passage_chars]}"
            for p in filtered_passages
        ])

        rag_text = "\n".join([entity_prompt, passage_prompt]).strip()
        if rag_text:
            rag_text = "You are a supportive counselor.The following knowledge may help understand the client's experience.Use it if it feels helpful.\n" + rag_text
        return rag_text

    def _build_multi_round_prompt(self, dialogue_history, current_user_input):
        history_text = ""
        for round_idx, (patient_utter, counselor_utter) in enumerate(dialogue_history):
            history_text += f"### Round {round_idx + 1} Patient: {patient_utter}\n### Round {round_idx + 1} Counselor: {counselor_utter}\n"

        rag_reference = self._bigraph_rag_retrieve(current_user_input) if self.use_rag else ""

        prompt_text = f"### Reference: {rag_reference}\n### History: {history_text}###\nPatient: {current_user_input}\n### thought:"
        return prompt_text

    @torch.no_grad()
    def generate(self, user_input, steps=128, gen_length=256, block_length=32, temperature=0.3, cfg_scale=1):
        return self.generate_with_history([], user_input, steps, gen_length, block_length, temperature, cfg_scale)

    @torch.no_grad()
    def generate_with_history(self, dialogue_history, current_user_input, steps=128, gen_length=256, block_length=32,
                              temperature=0.3, cfg_scale=1):
        prompt_text = self._build_multi_round_prompt(dialogue_history, current_user_input)

        encoded = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded['attention_mask'].to(self.model.device)

        output_ids = llada_generate(
            model=self.model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking='low_confidence',
            mask_id=self.mask_id
        )

        full_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return self._parse_and_clean(full_text)
    
    def _parse_and_clean(self, text):
        text = re.sub(r'<mask>|■|\[\[MASK\]\]||||●|◆', '', text)
        text = re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        
        thought = ""
        counselor = ""

        thought_match = re.search(
            r'^(.*?)(?=(?:Counselor|Response)\s*[:])',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()
            thought = re.sub(r'^##\s*thought:\s*', '', thought, flags=re.IGNORECASE)

        counselor_match = re.search(
            r'(?:Counselor|Response)\s*[:]\s*(.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if counselor_match:
            counselor = counselor_match.group(1).strip()
        
        thought = re.sub(r'#+', '', thought).strip()
        counselor = re.sub(r'#+', '', counselor).strip()
        return thought.strip(), counselor.strip()


if __name__ == "__main__":
    BASE_MODEL = "/path/to/LLaDA_model"
    LORA_CHECKPOINT = "/path/to/lora_checkpoint"
    RAG_CORPUS_PATH = "./data/RAGCorpus.txt"
    RAG_DATASET_NAME = "Psychological_Corpus_BiGraphRAG"

    infer_engine = BiGraphDiffuseRAG(
        base_model_path=BASE_MODEL,
        lora_path=LORA_CHECKPOINT,
        rag_corpus_path=RAG_CORPUS_PATH,
        rag_dataset_name=RAG_DATASET_NAME,
        use_rag=True,
        rag_top_k=3,
        score_threshold=0.1
    )

    print("===== Single-turn test (BiGraph-Diffuse-RAG) =====")
    question = "Everything is changing and I don't know who I am anymore. I'm scared of the future."
    thought, response = infer_engine.generate(question, gen_length=256)
    print("-" * 50)
    print(f"[Thought]:\n{thought}")
    print("-" * 50)
    print(f"[Response]:\n{response}")

    print("\n===== Multi-turn test (BiGraph-Diffuse-RAG) =====")
    dialogue_history = [
        ("I feel lonely even when I'm with my friends.",
         "It sounds like you're feeling disconnected even in company—like you're physically there but emotionally apart. That must be really hard to navigate."),
        ("Yes, exactly. I try to talk to them but it feels like no one gets it.",
         "It takes so much courage to reach out, and it’s understandable to feel let down when that connection doesn’t land. Have you noticed if there’s a specific moment when this feeling gets stronger?")
    ]
    current_input = "I just want to feel seen, but I don't know how to ask for it."
    thought_multi, response_multi = infer_engine.generate_with_history(dialogue_history, current_input, gen_length=256)
    print("-" * 50)
    print(f"[Multi-turn Thought]:\n{thought_multi}")
    print("-" * 50)
    print(f"[Multi-turn Response]:\n{response_multi}")