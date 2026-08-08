from copy import deepcopy
from src.bg_utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.bg_emb_model = embedding_model
        self.bg_db_path = db_filename
        self.bg_bsize = batch_size
        self.bg_ns = namespace
        
        self.bg_hids = []
        self.bg_texts = []
        self.bg_embs = []
        self.bg_hid2text = {}
        self.bg_hid2idx = {}
        self.bg_text2hid = {}
        
        self._load_data()
    
    def _load_data(self):
        if os.path.exists(self.bg_db_path):
            df = pd.read_parquet(self.bg_db_path)
            self.bg_hids = df["hash_id"].values.tolist()
            self.bg_texts = df["text"].values.tolist()
            self.bg_embs = df["embedding"].values.tolist()
            
            self.bg_hid2idx = {h: idx for idx, h in enumerate(self.bg_hids)}
            self.bg_hid2text = {h: t for h, t in zip(self.bg_hids, self.bg_texts)}
            self.bg_text2hid = {t: h for t, h in zip(self.bg_texts, self.bg_hids)}
            print(f"[{self.bg_ns}] Loaded {len(self.bg_hids)} records from {self.bg_db_path}")
        
    def insert_text(self, text_list):
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.bg_ns + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        
        existing = set(self.bg_hids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        all_embeddings = self.bg_emb_model.encode(texts_to_encode,normalize_embeddings=True, show_progress_bar=False,batch_size=self.bg_bsize)
        
        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        self.bg_hids.extend(hash_ids)
        self.bg_texts.extend(texts)
        self.bg_embs.extend(embeddings)
        
        self.bg_hid2idx = {h: idx for idx, h in enumerate(self.bg_hids)}
        self.bg_hid2text = {h: t for h, t in zip(self.bg_hids, self.bg_texts)}
        self.bg_text2hid = {t: h for t, h in zip(self.bg_texts, self.bg_hids)}
        
        self._save_data()

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.bg_hids,
            "text": self.bg_texts,
            "embedding": self.bg_embs
        })
        os.makedirs(os.path.dirname(self.bg_db_path), exist_ok=True)
        data_to_save.to_parquet(self.bg_db_path, index=False)
      
    def get_hash_id_to_text(self):
        return deepcopy(self.bg_hid2text)
    
    def encode_texts(self, texts):
        return self.bg_emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.bg_bsize)
    
    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])
        indices = np.array([self.bg_hid2idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.bg_embs)[indices]
        return embeddings