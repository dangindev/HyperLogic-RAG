# Data Directory

This directory should contain the preprocessed datasets and index files required for training and evaluation.

## Required Files

### Datasets
| File | Description | How to Obtain |
|------|-------------|---------------|
| `mimic_cxr_clean.jsonl` | Preprocessed MIMIC-CXR reports | Run `scripts/preprocess_mimic.py` |
| `iu_cxr_clean.jsonl` | Preprocessed IU X-Ray reports | Run `scripts/preprocess_iu.py` |

### Image Directories (symlinks)
| Symlink | Target |
|---------|--------|
| `mimic_cxr/` | Path to MIMIC-CXR image directory |
| `iu_xray/` | Path to IU X-Ray image directory |

### RAG Index Files
| File | Description | How to Obtain |
|------|-------------|---------------|
| `rag_index_mimic.pt` | BiomedCLIP embeddings for MIMIC-CXR | Run `scripts/build_rag_index.py` |
| `rag_index.json` | Report text index for MIMIC-CXR | Run `scripts/build_rag_index.py` |

### Hypergraph Files
| File | Description | How to Obtain |
|------|-------------|---------------|
| `hypergraph.pkl` | Clinical entity hypergraph | Run `scripts/build_hypergraph.py` |
| `radgraph/` | RadGraph entity annotations | Run `scripts/extract_radgraph.py` |

## Setup Steps

```bash
# 1. Create symlinks to your image directories
ln -s /path/to/mimic-cxr/images data/mimic_cxr
ln -s /path/to/iu-xray/images data/iu_xray

# 2. Preprocess reports
python scripts/preprocess_mimic.py
python scripts/preprocess_iu.py

# 3. Extract RadGraph entities and build hypergraph
python scripts/extract_radgraph.py
python scripts/build_hypergraph.py

# 4. Build RAG index
python scripts/build_rag_index.py
```
