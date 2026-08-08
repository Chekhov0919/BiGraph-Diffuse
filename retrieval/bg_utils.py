from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()

class LLM_Model:
    def __init__(self, llm_model):
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.bg_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )
        self.bg_llm_cfg = {
            "model": llm_model,
            "max_tokens": 2000,
            "temperature": 0,
        }
    def infer(self, messages):
        response = self.bg_client.chat.completions.create(**self.bg_llm_cfg,messages=messages)
        return response.choices[0].message.content



def normalize_answer(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val
