# BiGraph-Diffuse: Bipartite Graph Retrieval-Augmented Diffusion Models for Empathetic Counseling

This repository contains the core implementation of **BiGraph-Diffuse**, a retrieval-augmented diffusion language model for mental health counseling dialogues. Our method combines diffusion-based text generation with a novel bipartite graph retrieval mechanism (**BiGraph-RAG**) to produce empathetic, context-aware counselor responses.

## Overview

BiGraph-Diffuse integrates two key components:

- **BiGraph-Diffuse-Base**: A diffusion language model (LLaDA) fine-tuned with LoRA on counseling dialogues.
- **BiGraph-Diffuse-RAG**: The base model augmented with **BiGraph-RAG**, a graph-based retrieval system that constructs an entity-sentence bipartite graph and retrieves relevant psychological knowledge via Personalized PageRank.

## Model Weights

LoRA adapter weights are available on Hugging Face: [Chekhov0919/BiGraph-Diffuse](https://huggingface.co/Chekhov0919/BiGraph-Diffuse)

## Project Structure

```
├── inference/
│   └── inference.py          # BiGraph-Diffuse-RAG inference engine
├── retrieval/
│   ├── bg_bigraph.py          # BiGraph-RAG: bipartite graph retrieval
│   ├── bg_config.py           # BiGraphRAGConfig dataclass
│   ├── bg_emb_store.py        # Vector embedding storage
│   ├── bg_ner.py              # spaCy-based named entity recognition
│   └── bg_utils.py            # Utility functions
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

Core dependencies:
- `torch >= 2.1`
- `transformers >= 4.40`
- `peft >= 0.5`
- `sentence-transformers >= 2.6`
- `python-igraph >= 0.11`
- `spacy >= 3.7`

## Usage

### BiGraph-Diffuse-Base (without RAG)

```python
from inference.inference import BiGraphDiffuseRAG

engine = BiGraphDiffuseRAG(
    base_model_path="/path/to/LLaDA_model",
    lora_path="/path/to/lora_checkpoint",
    rag_corpus_path="/path/to/knowledge_corpus.txt",
    use_rag=False
)
thought, response = engine.generate("I feel anxious about the future.")
```

### BiGraph-Diffuse-RAG (with BiGraph-RAG)

```python
engine = BiGraphDiffuseRAG(
    base_model_path="/path/to/LLaDA_model",
    lora_path="/path/to/lora_checkpoint",
    rag_corpus_path="/path/to/knowledge_corpus.txt",
    use_rag=True,
    rag_top_k=3
)
thought, response = engine.generate("I feel anxious about the future.")
```

### Multi-turn Dialogue

```python
history = [
    ("I feel lonely even with friends.", "It sounds like you're feeling disconnected..."),
]
thought, response = engine.generate_with_history(history, "I just want to feel seen.")
```

## BiGraph-RAG Retrieval

1. **Entity Extraction**: Named entities extracted from both the query and knowledge corpus using spaCy.
2. **Bipartite Graph Construction**: Entities and sentences form a bipartite graph, connected by co-occurrence edges.
3. **Iterative Graph Search**: Seed entities from the query activate connected sentences; top-ranked sentences in turn activate new entities.
4. **Personalized PageRank**: Node weights are propagated through the graph via PPR to rank final passages.
5. **Augmented Generation**: Retrieved knowledge is prepended to the diffusion model's prompt.

## Configuration

### BiGraphRAGConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieval_top_k` | 5 | Number of passages to retrieve |
| `max_iterations` | 3 | Graph search iterations |
| `passage_ratio` | 1.5 | Dense retrieval weight in passage scoring |
| `damping` | 0.5 | PPR damping factor |
| `iteration_threshold` | 0.5 | Minimum score for entity activation |

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 128 | Total denoising steps |
| `gen_length` | 256 | Maximum generation length |
| `block_length` | 32 | Tokens per generation block |
| `temperature` | 0.3 | Gumbel noise temperature |
| `cfg_scale` | 1.0 | Classifier-free guidance scale |

## Citation

If you find this work useful, please stay tuned — citation information will be added upon publication.
