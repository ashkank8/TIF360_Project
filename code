from __future__ import annotations
import json
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=UserWarning)

# Optional dependencies, the script degrades gracefully if missing.
try:
    import requests
    HAVE_REQUESTS = True
except ImportError:
    HAVE_REQUESTS = False

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False
    def tqdm(x, **kw): return x  # no-op fallback

try:
    from Bio import PDB
    from Bio.PDB.Polypeptide import is_aa
    HAVE_BIOPYTHON = True
except ImportError:
    HAVE_BIOPYTHON = False


# Configuration

@dataclass
class Config:
    # data
    data_dir: str = "pdb_data"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    max_proteins: int = 400        # PDB entries to keep
    min_seq_len: int = 30          # discard shorter chains
    max_seq_len: int = 96          # truncate / pad to this length
    max_resolution: float = 2.5    # X-ray resolution cutoff (Å)
    train_fraction: float = 0.80
    val_fraction: float = 0.10

    # model
    d_model: int = 96              # embedding / transformer width
    n_heads: int = 4
    n_transformer_layers: int = 4
    n_gnn_layers: int = 3
    dropout: float = 0.10

    # loss
    contact_cutoff: float = 8.0    # Å, "in contact" if true distance < this
    contact_weight: float = 2.0    # multiplier on contact pairs in the loss
    bond_weight: float = 0.10      # weight on the consecutive-Cα bond term
    init_head_weight: float = 0.30  # weight on the auxiliary pre-GNN loss
    contact_bce_weight: float = 0.50  # weight on the contact-classification head

    # evaluation: primary contact separation (chains are short, 30-96 res).
    # We use 12 (medium+long range); 24 (strict CASP long-range) is also reported.
    contact_separation: int = 12

    # training
    batch_size: int = 16
    epochs: int = 60
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 12             # early-stopping patience (epochs)
    seed: int = 42

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Aminosyrorna
# 20 + padding (0) + unknown

AA_VOCAB = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9,
    "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17,
    "V": 18, "W": 19, "Y": 20, "<PAD>": 0, "<UNK>": 21,
}
VOCAB_SIZE = 22

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G",
    "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
    "TRP": "W", "TYR": "Y",
}


# PDB data
# Query the RCSB PDB Search API for reproducible, criteria based entries
# X-ray, resolution <= cutoff, single protein chain, length in range

PDB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{}.pdb"

# Fallback list used only if the Search API is unreachable.
PDB_FALLBACK = [
    "1CRN", "1UBQ", "1L2Y", "1VII", "1PGB", "2CI2", "1ENH", "1SHG",
    "1SHF", "1WHO", "2OVO", "1BDD", "2GB1", "1FSD", "2HBA", "1MBN",
    "3LZT", "1HRC", "1BPI", "1LMB", "1AHO", "1E0L", "2PTL", "1BX7",
    "1WIT", "3GB1", "1IGD", "2JOF", "1QYS", "2LZM",
]


def query_pdb_search_api(min_len: int, max_len: int, max_results: int,
                         max_resolution: float = 2.5) -> List[str]:
    """Return PDB IDs matching our criteria, sorted by resolution (best first).
    Falls back to PDB_FALLBACK on network errors."""
    def term(attribute, operator, value):
        return {"type": "terminal", "service": "text", "parameters": {
            "attribute": attribute, "operator": operator, "value": value}}

    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            term("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
            term("rcsb_entry_info.resolution_combined", "less_or_equal",
                 max_resolution),
            term("rcsb_entry_info.selected_polymer_entity_types",
                 "exact_match", "Protein (only)"),
            term("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
            term("entity_poly.rcsb_sample_sequence_length",
                 "greater_or_equal", min_len),
            term("entity_poly.rcsb_sample_sequence_length",
                 "less_or_equal", max_len),
        ]},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined",
                      "direction": "asc"}],
        },
    }

    try:
        if HAVE_REQUESTS:
            r = requests.post(PDB_SEARCH_URL, json=query, timeout=30)
            r.raise_for_status()
            payload = r.json()
        else:
            import urllib.request
            req = urllib.request.Request(
                PDB_SEARCH_URL, data=json.dumps(query).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[DATA]  RCSB API call failed ({type(e).__name__}: {e}). "
              f"Using fallback list of {len(PDB_FALLBACK)} entries.")
        return list(PDB_FALLBACK)

    ids = [hit["identifier"] for hit in payload.get("result_set", [])]
    print(f"[DATA]  RCSB API returned {len(ids)} entries "
          f"(X-ray, <= {max_resolution} A, single chain, "
          f"{min_len} <= length <= {max_len}).")
    return ids


def download_pdb_file(pdb_id: str, data_dir: str) -> Optional[Path]: # returns None on failure
    out_path = Path(data_dir) / f"{pdb_id.lower()}.pdb"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = PDB_DOWNLOAD_URL.format(pdb_id.upper())
    try:
        if HAVE_REQUESTS:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            out_path.write_text(r.text)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, out_path)
        return out_path
    except Exception:
        if out_path.exists():
            out_path.unlink()
        return None


def parse_ca_from_pdb(pdb_path: Path, max_len: int 
                      ) -> Optional[Tuple[str, np.ndarray]]:
    if HAVE_BIOPYTHON:
        return _parse_ca_biopython(pdb_path, max_len)
    return _parse_ca_manual(pdb_path, max_len)


def _parse_ca_biopython(pdb_path: Path, max_len: int
                        ) -> Optional[Tuple[str, np.ndarray]]:
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", str(pdb_path))
    except Exception:
        return None

    for model in structure:
        for chain in model:
            seq_chars, ca_coords = [], []
            for residue in chain.get_residues():
                if not is_aa(residue, standard=True) or "CA" not in residue:
                    continue
                aa1 = THREE_TO_ONE.get(residue.get_resname().strip())
                if aa1 is None:
                    continue
                seq_chars.append(aa1)
                ca_coords.append(residue["CA"].get_vector().get_array())
            L = len(seq_chars)
            if L < 10:
                continue
            if L > max_len:
                seq_chars, ca_coords = seq_chars[:max_len], ca_coords[:max_len]
            return "".join(seq_chars), np.asarray(ca_coords, dtype=np.float32)
        break  # only first model
    return None


def _parse_ca_manual(pdb_path: Path, max_len: int # lightweight fallback if Biopython not available
                     ) -> Optional[Tuple[str, np.ndarray]]:
    seq_chars, ca_coords, seen = [], [], set()
    try:
        for line in pdb_path.read_text().splitlines():
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA" or line[21:22] != "A":
                continue
            res_key = (line[22:26].strip(), line[26:27].strip())
            if res_key in seen:
                continue
            seen.add(res_key)
            aa1 = THREE_TO_ONE.get(line[17:20].strip())
            if aa1 is None:
                continue
            seq_chars.append(aa1)
            ca_coords.append((float(line[30:38]), float(line[38:46]),
                              float(line[46:54])))
    except Exception:
        return None

    L = len(seq_chars)
    if L < 10:
        return None
    if L > max_len:
        seq_chars, ca_coords = seq_chars[:max_len], ca_coords[:max_len]
    return "".join(seq_chars), np.asarray(ca_coords, dtype=np.float32)


def build_dataset(cfg: Config) -> List[Tuple[str, np.ndarray]]: # builds (seq, coords) dataset, uses on disk caching to avoid repeated downloads/parsing
    cache_path = Path(cfg.data_dir) / "dataset_cache.json"
    if cache_path.exists():
        print(f"[DATA]  Loading cached dataset from {cache_path} ...")
        with open(cache_path) as fh:
            raw = json.load(fh)
        data = [(d["seq"], np.asarray(d["coords"], dtype=np.float32))
                for d in raw]
        print(f"[DATA]  Loaded {len(data)} proteins from cache.")
        if len(data) > cfg.max_proteins:  # honour current cap even if cache larger
            data = data[: cfg.max_proteins]
        return data

    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    # Over-query to leave headroom for parse failures.
    pdb_ids = query_pdb_search_api(
        min_len=cfg.min_seq_len, max_len=cfg.max_seq_len,
        max_results=int(cfg.max_proteins * 1.5),
        max_resolution=cfg.max_resolution,
    )

    dataset: List[Tuple[str, np.ndarray]] = []
    failed = 0
    iterator = tqdm(pdb_ids, desc="Downloading PDB") if HAVE_TQDM else pdb_ids
    for pid in iterator:
        path = download_pdb_file(pid, cfg.data_dir)
        if path is None:
            failed += 1
            continue
        result = parse_ca_from_pdb(path, cfg.max_seq_len)
        if result is None:
            failed += 1
            continue
        seq, coords = result
        if len(seq) < cfg.min_seq_len:
            continue
        dataset.append((seq, coords))
        if len(dataset) >= cfg.max_proteins:
            break
        time.sleep(0.02)  # be polite to RCSB

    print(f"[DATA]  Final dataset: {len(dataset)} proteins "
          f"({failed} download / parse failures).")

    cache_payload = [{"seq": s, "coords": c.tolist()} for s, c in dataset]
    with open(cache_path, "w") as fh:
        json.dump(cache_payload, fh)
    print(f"[DATA]  Cached to {cache_path}")
    return dataset


# 3. Dataset 

class ProteinDataset(Dataset): # wrapper around the raw (seq, coords) list, handles tokenization, distance matrix computation, and padding/truncation to max_len
    def __init__(self, data: List[Tuple[str, np.ndarray]], max_len: int):
        self.data = data
        self.N = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        seq, coords = self.data[idx]
        L = min(len(seq), self.N)
        seq, coords = seq[:L], coords[:L]

        tokens = np.zeros(self.N, dtype=np.int64)
        for i, aa in enumerate(seq):
            tokens[i] = AA_VOCAB.get(aa, AA_VOCAB["<UNK>"])

        diff = coords[:, None, :] - coords[None, :, :]
        d_real = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)
        dmat = np.zeros((self.N, self.N), dtype=np.float32)
        dmat[:L, :L] = d_real

        mask = np.zeros((self.N, self.N), dtype=np.float32)
        mask[:L, :L] = 1.0

        return (torch.from_numpy(tokens), torch.from_numpy(dmat),
                torch.from_numpy(mask), torch.tensor(L, dtype=torch.long))


# 4. Model: Transformer encoder + GNN refine
# tokens -> embed + sinusoidal PE -> Transformer encoder -> pair features
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class EdgeRefinementBlock(nn.Module):
    """One round of edge-feature message passing on the dense pair graph:
    transform (message from h_i,h_j,e_ij) -> propagate (mean over neighbours)
    -> update (residual add, symmetrise, layer-norm)."""
    def __init__(self, d_node: int, d_edge: int, dropout: float):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_edge), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_edge, d_edge))
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * d_edge, d_edge), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_edge, d_edge))
        self.norm = nn.LayerNorm(d_edge)

    def forward(self, node_feats: torch.Tensor, edge_feats: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # node_feats (B,L,d_node); edge_feats (B,L,L,d_edge); mask (B,L,L)
        B, L, _ = node_feats.shape

        h_i = node_feats.unsqueeze(2).expand(B, L, L, -1)
        h_j = node_feats.unsqueeze(1).expand(B, L, L, -1)
        msg = self.msg_mlp(torch.cat([h_i, h_j, edge_feats], dim=-1))

        # Aggregate over k: mean of (i,k) edges and mean of (k,j) edges.
        mask4 = mask.unsqueeze(-1)
        msg_m = msg * mask4
        denom_r = mask.sum(dim=2, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        agg_rows = (msg_m.sum(dim=2, keepdim=True) / denom_r).expand(-1, -1, L, -1)
        denom_c = mask.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
        agg_cols = (msg_m.sum(dim=1, keepdim=True) / denom_c).expand(-1, L, -1, -1)
        neighbours = 0.5 * (agg_rows + agg_cols)

        out = self.update_mlp(torch.cat([msg, neighbours], dim=-1))
        out = self.norm(edge_feats + out)
        return 0.5 * (out + out.transpose(1, 2))  # distance matrices are symmetric


class HybridTransformerGNN(nn.Module): # Transformer encoder produces initial pairwise distance map, GNN refines it, contact head predicts contact probabilities for evaluation and ranking
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model

        # Embedding + positional encoding.
        self.embed = nn.Embedding(VOCAB_SIZE, D, padding_idx=0)
        self.pos_enc = PositionalEncoding(D, max_len=cfg.max_seq_len + 8)
        self.embed_norm = nn.LayerNorm(D)

        # Pre-norm Transformer encoder stack (more stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=cfg.n_heads, dim_feedforward=4 * D,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_transformer_layers)

        # Pair-feature builder: [h_i, h_j, h_i o h_j] -> d_edge
        self.d_edge = D
        self.pair_proj = nn.Linear(3 * D, self.d_edge)

        # Initial distance head (Softplus -> non-negative)
        self.init_dist_head = nn.Sequential(
            nn.Linear(self.d_edge, self.d_edge), nn.GELU(),
            nn.Linear(self.d_edge, 1), nn.Softplus())
        self.dist_to_edge = nn.Linear(1, self.d_edge)  # feed d back into edges

        # GNN refinement stack
        self.gnn_blocks = nn.ModuleList([
            EdgeRefinementBlock(D, self.d_edge, cfg.dropout)
            for _ in range(cfg.n_gnn_layers)])
        self.final_dist_head = nn.Sequential(
            nn.Linear(self.d_edge, self.d_edge), nn.GELU(),
            nn.Linear(self.d_edge, 1), nn.Softplus())
        # Contact classification head: per-pair in contact logit, trained with
        # class-balanced BCE, used to RANK pairs for the top-L/k metric
        self.contact_head = nn.Sequential(
            nn.Linear(self.d_edge, self.d_edge), nn.GELU(),
            nn.Linear(self.d_edge, 1))

    def forward(self, tokens: torch.Tensor, pair_mask: torch.Tensor):
        """tokens (B,L); pair_mask (B,L,L). Returns (d_init, d_final, c_logits),
        all (B,L,L); distance maps have zeroed diagonals, c_logits symmetric."""
        B, L = tokens.shape
        device = tokens.device

        # Sequence encoder
        x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
        x = self.pos_enc(x)
        x = self.embed_norm(x)
        key_pad_mask = (tokens == 0)  # True = ignore (padding)
        h = self.transformer(x, src_key_padding_mask=key_pad_mask)  # (B,L,D)

        # Build pair features
        h_i = h.unsqueeze(2).expand(B, L, L, -1)
        h_j = h.unsqueeze(1).expand(B, L, L, -1)
        pair = torch.cat([h_i, h_j, h_i * h_j], dim=-1)
        edge = self.pair_proj(pair)
        edge = 0.5 * (edge + edge.transpose(1, 2))  # symmetric init

        d_init = self.init_dist_head(edge).squeeze(-1)
        d_init = 0.5 * (d_init + d_init.transpose(1, 2))
        eye = torch.eye(L, device=device).unsqueeze(0)
        d_init = d_init * (1.0 - eye)

        # GNN refinement
        edge = edge + self.dist_to_edge(d_init.unsqueeze(-1))
        for blk in self.gnn_blocks:
            edge = blk(h, edge, pair_mask)
        d_final = self.final_dist_head(edge).squeeze(-1)
        d_final = 0.5 * (d_final + d_final.transpose(1, 2))
        d_final = d_final * (1.0 - eye)

        c_logits = self.contact_head(edge).squeeze(-1)
        c_logits = 0.5 * (c_logits + c_logits.transpose(1, 2))
        return d_init, d_final, c_logits


# 5. Loss functions

CA_CA_BOND_TARGET = 3.8  # Å, canonical consecutive Cα distance


def contact_weighted_l1(pred: torch.Tensor, target: torch.Tensor, # Masked, contact-weighted L1 loss for the distance head
                        mask: torch.Tensor, cutoff: float, weight: float
                        ) -> torch.Tensor:
    diff = (pred - target).abs()
    contact = (target < cutoff).float()
    w = 1.0 + (weight - 1.0) * contact
    num = (diff * w * mask).sum()
    den = (w * mask).sum().clamp_min(1.0)
    return num / den


def neighbour_bond_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: # Soft L1 penalty on consecutive-Cα distances deviating from the canonical bond length, averaged over all real pairs (masked)
    B, L, _ = pred.shape
    if L < 2:
        return pred.new_tensor(0.0)
    idx = torch.arange(L - 1, device=pred.device)
    nb_pred = pred[:, idx, idx + 1]
    nb_mask = mask[:, idx, idx + 1]
    diff = (nb_pred - CA_CA_BOND_TARGET).abs() * nb_mask
    return diff.sum() / nb_mask.sum().clamp_min(1.0)


def contact_bce_loss(logits: torch.Tensor, target_d: torch.Tensor, # Class-balanced binary cross-entropy loss for the contact classification head
                     mask: torch.Tensor, cutoff: float,
                     min_sep: int = 4) -> torch.Tensor:
    B, L, _ = logits.shape
    device = logits.device

    ii = torch.arange(L, device=device).view(1, L, 1)
    jj = torch.arange(L, device=device).view(1, 1, L)
    sep_mask = (torch.abs(ii - jj) > min_sep).float()
    elig = mask * sep_mask  # (B,L,L)

    target = (target_d < cutoff).float() * elig  # 1 = true contact
    n_pos = target.sum().clamp_min(1.0)
    n_neg = (elig.sum() - target.sum()).clamp_min(1.0)
    pos_weight = (n_neg / n_pos).detach()

    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, weight=elig, pos_weight=pos_weight, reduction="sum")
    return bce / elig.sum().clamp_min(1.0)


# 6. Evaluation metrics (alignment-free, SE(3)-inv)

def drmsd(pred_d: np.ndarray, true_d: np.ndarray) -> float: # Distance-RMSD over unique pairs (Å, lower is better). No alignment
    iu = np.triu_indices(pred_d.shape[0], k=1)
    p, t = pred_d[iu], true_d[iu]
    return float(np.sqrt(np.mean((p - t) ** 2)))


def contact_precision(pred_d: np.ndarray, true_d: np.ndarray, # Precision of predicted contacts (predicted distance < cutoff) against true contacts (true distance < cutoff) 
                      cutoff: float = 8.0) -> float:
    L = pred_d.shape[0]
    i, j = np.triu_indices(L, k=5)  # skip local pairs
    p, t = pred_d[i, j], true_d[i, j]
    pred_c = p < cutoff
    if pred_c.sum() == 0:
        return 0.0
    true_c = t < cutoff
    return float((pred_c & true_c).sum() / pred_c.sum())


def topk_contact_precision(pred_d: np.ndarray, true_d: np.ndarray, # Precision of the top L/k predicted contacts (ranked by confidence or distance) against true contacts
                           k_div: int = 5, sep: int = 24,
                           cutoff: float = 8.0,
                           score: Optional[np.ndarray] = None) -> float:
    L = pred_d.shape[0]
    i, j = np.triu_indices(L, k=sep)  # |i - j| >= sep
    if i.size == 0:
        return float("nan")  # chain too short for long-range

    true_pair = true_d[i, j]
    n_top = max(1, L // k_div)
    n_top = min(n_top, i.size)

    if score is not None:
        order = np.argsort(score[i, j])[::-1][:n_top]  # descending confidence
    else:
        order = np.argsort(pred_d[i, j])[:n_top]       # ascending distance

    true_is_contact = (true_pair[order] < cutoff)
    return float(true_is_contact.mean())


# 7. 3D reconstruction (classical MDS)

def mds_reconstruct(dist: np.ndarray) -> np.ndarray: # classical MDS to reconstruct 3D coordinates from a distance matrix, for visualisation only
    n = dist.shape[0]
    d2 = dist.astype(np.float64) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ d2 @ J
    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:3]
    L = np.diag(np.sqrt(np.clip(eigvals[idx], 0.0, None)))
    return eigvecs[:, idx] @ L


def kabsch_align(P: np.ndarray, Q: np.ndarray) -> Tuple[float, np.ndarray]: # Kabsch algorithm to align two sets of 3D points P and Q, returns (RMSD, aligned_P)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    best = (np.inf, Pc)
    for sign in (+1.0, -1.0):
        D = np.diag([1.0, 1.0, sign * np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        Pa = Pc @ R.T
        r = float(np.sqrt(((Pa - Qc) ** 2).sum() / P.shape[0]))
        if r < best[0]:
            best = (r, Pa)
    return best[0], best[1] + Q.mean(0)


# 8. Training/evaluation loop

def run_epoch(model: HybridTransformerGNN, loader: DataLoader, # one epoch of training or evaluation
              optimizer: Optional[torch.optim.Optimizer], cfg: Config,
              training: bool) -> Tuple[float, float, float]:
    model.train(training)
    total_loss = 0.0
    drmsds: List[float] = []
    cprecs: List[float] = []

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for tokens, dmat, mask, lengths in loader:
            tokens = tokens.to(cfg.device)
            dmat = dmat.to(cfg.device)
            mask = mask.to(cfg.device)

            d_init, d_final, c_logits = model(tokens, mask)
            l_init = contact_weighted_l1(
                d_init, dmat, mask, cfg.contact_cutoff, cfg.contact_weight)
            l_final = contact_weighted_l1(
                d_final, dmat, mask, cfg.contact_cutoff, cfg.contact_weight)
            l_bond = neighbour_bond_loss(d_final, mask)
            l_cbce = contact_bce_loss(c_logits, dmat, mask, cfg.contact_cutoff)
            loss = (cfg.init_head_weight * l_init + 1.0 * l_final
                    + cfg.bond_weight * l_bond + cfg.contact_bce_weight * l_cbce)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            total_loss += loss.item()

            # per-sample metrics on CPU
            d_pred_np = d_final.detach().cpu().numpy()
            d_true_np = dmat.cpu().numpy()
            for b, L in enumerate(lengths.tolist()):
                pred = d_pred_np[b, :L, :L]
                true = d_true_np[b, :L, :L]
                drmsds.append(drmsd(pred, true))
                cprecs.append(contact_precision(pred, true, cfg.contact_cutoff))

    return (total_loss / max(1, len(loader)),
            float(np.mean(drmsds)) if drmsds else float("nan"),
            float(np.mean(cprecs)) if cprecs else float("nan"))


# 8b Sequence distance baseline
# Null model using no amino-acid identity: predict the mean Cα–Cα distance for
# each sequence separation k = |i - j|. Beating it shows the network learns

def fit_sequence_distance_baseline( # returns mean distance as a function of sequence separation k, fitted on the training set
    train_data: List[Tuple[str, np.ndarray]], max_len: int) -> np.ndarray:
    sums = np.zeros(max_len, dtype=np.float64)
    counts = np.zeros(max_len, dtype=np.int64)
    for _, coords in train_data:
        L = coords.shape[0]
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        for k in range(1, L):
            sums[k] += d[np.arange(L - k), np.arange(k, L)].sum()
            counts[k] += (L - k)
    mean_d = np.zeros(max_len, dtype=np.float32)
    mask = counts > 0
    mean_d[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    last_seen = mean_d[mask][-1] if mask.any() else 0.0
    mean_d[~mask] = last_seen
    return mean_d


def baseline_predict(L: int, mean_d: np.ndarray) -> np.ndarray: # Build a distance matrix of size LxL where the distance depends only on the sequence separation k = |i - j|
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    k = np.abs(i - j)
    return mean_d[k].astype(np.float32)


# 8c. Per-test-protein eval (baseline / tfm-only / +GNN)
def evaluate_test_set(model: HybridTransformerGNN, # evaluate on the test set, returning a dictionary of metrics for each protein and overall. For the first few proteins, also save the raw data for visualisation.
                      test_data: List[Tuple[str, np.ndarray]],
                      baseline_mean_d: np.ndarray, cfg: Config) -> dict:
    model.eval()
    results = {
        "length": [], "drmsd_baseline": [], "drmsd_init": [], "drmsd_final": [],
        "cprec_baseline": [], "cprec_init": [], "cprec_final": [],
        "topL_baseline": [], "topL_init": [], "topL_final": [],
        "topL2_baseline": [], "topL2_init": [], "topL2_final": [],
        "topL5_baseline": [], "topL5_init": [], "topL5_final": [],
        "topL5lr_baseline": [], "topL5lr_init": [], "topL5lr_final": [],
        "examples": [],
    }

    with torch.no_grad():
        for k, (seq, true_xyz) in enumerate(test_data):
            L = len(true_xyz)
            diff = true_xyz[:, None, :] - true_xyz[None, :, :]
            true_d = np.sqrt((diff ** 2).sum(-1)).astype(np.float32)

            d_baseline = baseline_predict(L, baseline_mean_d)
            np.fill_diagonal(d_baseline, 0.0)

            # model predictions (both heads)
            N = cfg.max_seq_len
            tokens = torch.zeros(1, N, dtype=torch.long, device=cfg.device)
            for i, aa in enumerate(seq[:N]):
                tokens[0, i] = AA_VOCAB.get(aa, AA_VOCAB["<UNK>"])
            pair_mask = torch.zeros(1, N, N, device=cfg.device)
            pair_mask[0, :L, :L] = 1.0
            d_init_t, d_final_t, c_logits_t = model(tokens, pair_mask)
            d_init = d_init_t[0, :L, :L].cpu().numpy()
            d_final = d_final_t[0, :L, :L].cpu().numpy()
            c_prob = torch.sigmoid(c_logits_t)[0, :L, :L].cpu().numpy()

            results["length"].append(L)
            results["drmsd_baseline"].append(drmsd(d_baseline, true_d))
            results["drmsd_init"].append(drmsd(d_init, true_d))
            results["drmsd_final"].append(drmsd(d_final, true_d))
            results["cprec_baseline"].append(
                contact_precision(d_baseline, true_d, cfg.contact_cutoff))
            results["cprec_init"].append(
                contact_precision(d_init, true_d, cfg.contact_cutoff))
            results["cprec_final"].append(
                contact_precision(d_final, true_d, cfg.contact_cutoff))

            # top-L/k at the primary separation; baseline/tfm ranked by
            # distance, full model ranked by contact-head probability
            for kdiv, tag in ((1, "topL"), (2, "topL2"), (5, "topL5")):
                results[f"{tag}_baseline"].append(
                    topk_contact_precision(d_baseline, true_d, k_div=kdiv,
                                           sep=cfg.contact_separation))
                results[f"{tag}_init"].append(
                    topk_contact_precision(d_init, true_d, k_div=kdiv,
                                           sep=cfg.contact_separation))
                results[f"{tag}_final"].append(
                    topk_contact_precision(d_final, true_d, k_div=kdiv,
                                           sep=cfg.contact_separation,
                                           score=c_prob))
            # CASP long-range (sep=24), top-L/5 only
            results["topL5lr_baseline"].append(
                topk_contact_precision(d_baseline, true_d, k_div=5, sep=24))
            results["topL5lr_init"].append(
                topk_contact_precision(d_init, true_d, k_div=5, sep=24))
            results["topL5lr_final"].append(
                topk_contact_precision(d_final, true_d, k_div=5, sep=24,
                                       score=c_prob))

            if k < 3:
                results["examples"].append({
                    "seq": seq, "true_xyz": true_xyz, "true_d": true_d,
                    "d_baseline": d_baseline, "d_init": d_init,
                    "d_final": d_final,
                })

    array_keys = ("length", "drmsd_baseline", "drmsd_init", "drmsd_final",
                  "cprec_baseline", "cprec_init", "cprec_final",
                  "topL_baseline", "topL_init", "topL_final",
                  "topL2_baseline", "topL2_init", "topL2_final",
                  "topL5_baseline", "topL5_init", "topL5_final",
                  "topL5lr_baseline", "topL5lr_init", "topL5lr_final")
    for key in array_keys:
        results[key] = np.asarray(results[key], dtype=float)
    return results




def plot_learning_curves(train_losses, val_losses, val_drmsds, val_cprec, # one plot of training/validation loss curves
                         save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ep = range(1, len(train_losses) + 1)
    ax.plot(ep, train_losses, label="train")
    ax.plot(ep, val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (A, masked L1 on distances)")
    ax.set_title("Learning curves")
    ax.legend()
    fig.tight_layout()
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(save_dir) / "learning_curves.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


def plot_distance_matrices(true_d: np.ndarray, pred_d: np.ndarray, # side-by-side heatmaps of the true and predicted distance matrices
                           label: str, save_dir: str) -> None:
    L = true_d.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vmax = float(max(true_d.max(), pred_d.max()))
    axes[0].imshow(true_d, vmin=0, vmax=vmax)
    axes[0].set_title("ground truth")
    axes[1].imshow(pred_d, vmin=0, vmax=vmax)
    axes[1].set_title("prediction")
    for ax in axes:
        ax.set_xlabel("residue j")
        ax.set_ylabel("residue i")
    fig.suptitle(f"{label}: L={L}, dRMSD={drmsd(pred_d, true_d):.2f} A")
    fig.tight_layout()
    out = Path(save_dir) / f"distmat_{label}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


def plot_3d_structure(true_xyz: np.ndarray, pred_xyz: np.ndarray, # side-by-side 3D scatter plots of the true and predicted Cα coordinates, after Kabsch alignment
                      label: str, save_dir: str) -> None:
    rmsd_3d, pred_xyz_a = kabsch_align(pred_xyz, true_xyz)
    L = true_xyz.shape[0]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    for ax, xyz, ttl in [(ax1, true_xyz, "ground truth"),
                         (ax2, pred_xyz_a, "prediction")]:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], "-o", ms=3, lw=1)
        ax.set_title(ttl)
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")
    fig.suptitle(f"{label}: L={L}, 3D RMSD={rmsd_3d:.2f} A")
    fig.tight_layout()
    out = Path(save_dir) / f"structure_{label}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


def plot_drmsd_vs_length(results: dict, save_dir: str) -> None: # scatter plot of per-protein dRMSD vs sequence length, with a linear fit line and mean dRMSD in the legend for each method
    L = results["length"]
    b = results["drmsd_baseline"]
    di = results["drmsd_init"]
    df = results["drmsd_final"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(L, b, c="gray", marker="x", s=40, alpha=0.7,
               label=f"baseline  (mean {b.mean():.2f} A)")
    ax.scatter(L, di, c="C0", marker="s", s=40, alpha=0.7,
               label=f"Transformer-only  (mean {di.mean():.2f} A)")
    ax.scatter(L, df, c="C3", marker="o", s=45, alpha=0.85,
               label=f"Transformer + GNN  (mean {df.mean():.2f} A)")

    if len(L) >= 2:
        m, c = np.polyfit(L, df, 1)
        xs = np.linspace(L.min(), L.max(), 50)
        ax.plot(xs, m * xs + c, "C3--", lw=1, alpha=0.7,
                label=f"linear fit: dRMSD = {m:.3f} * L + {c:.2f}")

    ax.set_xlabel("sequence length L (residues)")
    ax.set_ylabel("dRMSD (A)")
    ax.set_title("Per-protein test-set performance vs chain length")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = Path(save_dir) / "drmsd_vs_length.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


def plot_method_comparison_bars(results: dict, save_dir: str) -> None: # bar plots comparing the mean dRMSD and contact precision for all 3 cases
    methods = ["baseline", "Transformer\nonly", "Transformer\n+ GNN"]
    drmsd_means = [results["drmsd_baseline"].mean(),
                   results["drmsd_init"].mean(),
                   results["drmsd_final"].mean()]
    drmsd_stds = [results["drmsd_baseline"].std(),
                  results["drmsd_init"].std(),
                  results["drmsd_final"].std()]
    cprec_means = [results["cprec_baseline"].mean(),
                   results["cprec_init"].mean(),
                   results["cprec_final"].mean()]
    colors = ["gray", "C0", "C3"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(methods, drmsd_means, yerr=drmsd_stds, capsize=4, color=colors)
    axes[0].set_ylabel("dRMSD (A)  -- lower is better")
    axes[0].set_title("Distance-RMSD on the test set")
    for i, v in enumerate(drmsd_means):
        axes[0].text(i, v + drmsd_stds[i] * 0.5, f"{v:.2f}",
                     ha="center", fontsize=10)

    axes[1].bar(methods, cprec_means, color=colors)
    axes[1].set_ylabel("contact precision  -- higher is better")
    axes[1].set_title("Long-range contact precision (|i-j| > 4)")
    axes[1].set_ylim(0, max(0.3, max(cprec_means) * 1.3))
    for i, v in enumerate(cprec_means):
        axes[1].text(i, v + 0.005, f"{v:.2%}", ha="center", fontsize=10)

    fig.tight_layout()
    out = Path(save_dir) / "method_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


def plot_distance_matrices_3way(example: dict, label: str, # 4-way comparison of true distance matrix, baseline, transformer-only, and full model
                                save_dir: str) -> None:
    true_d = example["true_d"]
    d_base = example["d_baseline"]
    d_init = example["d_init"]
    d_final = example["d_final"]
    L = true_d.shape[0]

    vmax = float(max(true_d.max(), d_base.max(), d_init.max(), d_final.max()))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (true_d, "ground truth", None),
        (d_base, "baseline (|i-j| lookup)", drmsd(d_base, true_d)),
        (d_init, "Transformer only", drmsd(d_init, true_d)),
        (d_final, "Transformer + GNN", drmsd(d_final, true_d)),
    ]
    for ax, (mat, ttl, dr) in zip(axes, panels):
        ax.imshow(mat, vmin=0, vmax=vmax)
        sub = ttl if dr is None else f"{ttl}\ndRMSD = {dr:.2f} A"
        ax.set_title(sub, fontsize=10)
        ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
    fig.suptitle(f"{label}: L = {L} residues", fontsize=11)
    fig.tight_layout()
    out = Path(save_dir) / f"distmat_3way_{label}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"[PLOT]  Saved {out}")


@torch.no_grad()
def predict_structure(model: HybridTransformerGNN, sequence: str, # predict the distance matrix and 3D coordinates for a single sequence, for visualisation
                      cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    N = cfg.max_seq_len
    L = min(len(sequence), N)
    tokens = torch.zeros(1, N, dtype=torch.long, device=cfg.device)
    for i, aa in enumerate(sequence[:L]):
        tokens[0, i] = AA_VOCAB.get(aa, AA_VOCAB["<UNK>"])
    pair_mask = torch.zeros(1, N, N, device=cfg.device)
    pair_mask[0, :L, :L] = 1.0

    _, d_final, _ = model(tokens, pair_mask)
    d = d_final[0, :L, :L].cpu().numpy()
    xyz = mds_reconstruct(d)
    return d, xyz



def main() -> None:
    set_seed(CFG.seed)
    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)
    Path(CFG.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Hybrid Transformer-GNN - coarse-grained Ca structure prediction")
    print("=" * 70)
    print(f"[INFO]  Device          : {CFG.device}")
    print(f"[INFO]  Max seq length  : {CFG.max_seq_len}")
    print(f"[INFO]  Biopython       : {'yes' if HAVE_BIOPYTHON else 'NO (manual parser)'}")
    print(f"[INFO]  Requests        : {'yes' if HAVE_REQUESTS else 'NO (urllib fallback)'}")
    print(f"[INFO]  tqdm            : {'yes' if HAVE_TQDM else 'NO'}")

    # Step 1: dataset
    print("\n" + "-" * 70)
    print(" Step 1  -  Dataset")
    print("-" * 70)
    dataset = build_dataset(CFG)
    if len(dataset) < 30:
        raise RuntimeError(
            f"Only {len(dataset)} usable proteins. Need at least 30. "
            f"Check internet access to https://files.rcsb.org or relax the "
            f"length / resolution filters in Config."
        )

    rng = np.random.default_rng(CFG.seed)
    idx = rng.permutation(len(dataset))
    n_train = int(CFG.train_fraction * len(dataset))
    n_val = int(CFG.val_fraction * len(dataset))
    train_data = [dataset[i] for i in idx[:n_train]]
    val_data = [dataset[i] for i in idx[n_train:n_train + n_val]]
    test_data = [dataset[i] for i in idx[n_train + n_val:]]
    print(f"[DATA]  split  train={len(train_data)}  "
          f"val={len(val_data)}  test={len(test_data)}")

    train_loader = DataLoader(ProteinDataset(train_data, CFG.max_seq_len),
                              batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(ProteinDataset(val_data, CFG.max_seq_len),
                            batch_size=CFG.batch_size, shuffle=False)
    test_loader = DataLoader(ProteinDataset(test_data, CFG.max_seq_len),
                             batch_size=CFG.batch_size, shuffle=False)

    # Step 2: model
    print("\n" + "-" * 70)
    print(" Step 2  -  Model")
    print("-" * 70)
    model = HybridTransformerGNN(CFG).to(CFG.device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters: {n_params:,}  ({n_params/1e6:.2f} M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr,
                                  weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=1e-6)

    # Step 3: training
    print("\n" + "-" * 70)
    print(" Step 3  -  Training")
    print("-" * 70)
    train_losses, val_losses, val_drmsds, val_cprecs = [], [], [], []
    best_val_loss = float("inf")
    patience = 0
    best_ckpt = Path(CFG.checkpoint_dir) / "best_model.pt"

    t0 = time.time()
    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_drmsd, tr_cp = run_epoch(
            model, train_loader, optimizer, CFG, training=True)
        vl_loss, vl_drmsd, vl_cp = run_epoch(
            model, val_loader, None, CFG, training=False)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        val_drmsds.append(vl_drmsd); val_cprecs.append(vl_cp)
        lr_now = scheduler.get_last_lr()[0]

        marker = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience = 0
            marker = "  *new best*"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "val_loss": vl_loss,
                "val_drmsd": vl_drmsd,
                "config": asdict(CFG),
            }, best_ckpt)
        else:
            patience += 1

        print(f"epoch {epoch:3d}/{CFG.epochs}  "
              f"| train loss {tr_loss:.3f}  dRMSD {tr_drmsd:5.2f} A  "
              f"| val loss {vl_loss:.3f}  dRMSD {vl_drmsd:5.2f} A  "
              f"cp {vl_cp:.2%}  | lr {lr_now:.1e}{marker}")

        if patience >= CFG.patience:
            print(f"[TRAIN] Early stopping at epoch {epoch} "
                  f"(no improvement for {CFG.patience} epochs).")
            break

    print(f"[TRAIN] Training took {(time.time()-t0)/60:.1f} min")

    # Step 4: test-set evaluation
    print("\n" + "-" * 70)
    print(" Step 4  -  Test Evaluation")
    print("-" * 70)
    ckpt = torch.load(best_ckpt, map_location=CFG.device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    te_loss, te_drmsd, te_cp = run_epoch(
        model, test_loader, None, CFG, training=False)
    print(f"[RESULT]  Test contact-weighted L1 loss : {te_loss:.4f}")
    print(f"[RESULT]  Test dRMSD (Transformer + GNN): {te_drmsd:.2f} A")
    print(f"[RESULT]  Test contact precision        : {te_cp:.2%}")

    # Fit baseline on TRAINING data only, then evaluate every test protein
    print("[EVAL]   Fitting sequence-distance baseline on training set ...")
    baseline_mean_d = fit_sequence_distance_baseline(train_data, CFG.max_seq_len)
    print("[EVAL]   Evaluating baseline / Transformer-only / Transformer+GNN "
          "on every test protein ...")
    results = evaluate_test_set(model, test_data, baseline_mean_d, CFG)

    print(f"[RESULT]  Mean test dRMSD by method (n = {len(results['length'])}):")
    print(f"          baseline           : {results['drmsd_baseline'].mean():.2f} A  "
          f"(std {results['drmsd_baseline'].std():.2f})")
    print(f"          Transformer only   : {results['drmsd_init'].mean():.2f} A  "
          f"(std {results['drmsd_init'].std():.2f})")
    print(f"          Transformer + GNN  : {results['drmsd_final'].mean():.2f} A  "
          f"(std {results['drmsd_final'].std():.2f})")
    drop_b_to_f = (results['drmsd_baseline'].mean()
                   - results['drmsd_final'].mean())
    drop_i_to_f = (results['drmsd_init'].mean()
                   - results['drmsd_final'].mean())
    print(f"[RESULT]  Improvement over baseline      : {drop_b_to_f:+.2f} A")
    print(f"[RESULT]  Improvement from GNN refinement: {drop_i_to_f:+.2f} A")

    # Standard CASP-style top-L/k contact precision (nanmean over eligible chains)
    n_lr = int(np.sum(~np.isnan(results["topL_final"])))
    print(f"\n[RESULT]  Standard top-L/k contact precision "
          f"(separation |i-j|>={CFG.contact_separation}, "
          f"n={n_lr} eligible chains):")
    print(f"          {'method':<18}{'top-L':>10}{'top-L/2':>10}{'top-L/5':>10}")
    for tag, label in (("baseline", "baseline"),
                       ("init", "Transformer"),
                       ("final", "Transf.+GNN")):
        tl = np.nanmean(results[f"topL_{tag}"]) * 100
        tl2 = np.nanmean(results[f"topL2_{tag}"]) * 100
        tl5 = np.nanmean(results[f"topL5_{tag}"]) * 100
        print(f"          {label:<18}{tl:>9.1f}%{tl2:>9.1f}%{tl5:>9.1f}%")
    n_strict = int(np.sum(~np.isnan(results["topL5lr_final"])))
    print(f"[RESULT]  Strict CASP long-range top-L/5 (|i-j|>=24, "
          f"n={n_strict}):  "
          f"base {np.nanmean(results['topL5lr_baseline'])*100:.1f}%  "
          f"tfm {np.nanmean(results['topL5lr_init'])*100:.1f}%  "
          f"+gnn {np.nanmean(results['topL5lr_final'])*100:.1f}%")

    # Step 5: plots
    print("\n" + "-" * 70)
    print(" Step 5  -  Visualisation")
    print("-" * 70)
    plot_learning_curves(train_losses, val_losses, val_drmsds, val_cprecs,
                         CFG.output_dir)
    plot_drmsd_vs_length(results, CFG.output_dir)
    plot_method_comparison_bars(results, CFG.output_dir)

    model.eval()
    for k, ex in enumerate(results["examples"]):
        seq = ex["seq"]
        true_xyz = ex["true_xyz"]
        L = len(seq)
        label = f"test_{k+1}_L{L}"

        pred_xyz = mds_reconstruct(ex["d_final"].astype(np.float64))
        plot_distance_matrices(ex["true_d"], ex["d_final"], label, CFG.output_dir)
        plot_3d_structure(true_xyz, pred_xyz, label, CFG.output_dir)
        plot_distance_matrices_3way(ex, label, CFG.output_dir)

    # Step 6: write summary
    summary_path = Path(CFG.output_dir) / "summary.txt"
    with open(summary_path, "w") as fh:
        fh.write("Hybrid Transformer-GNN -- coarse-grained Ca structure prediction\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"Dataset      : {len(train_data)} train / "
                 f"{len(val_data)} val / {len(test_data)} test\n")
        fh.write(f"Max length   : {CFG.max_seq_len}\n")
        fh.write(f"Model        : d_model={CFG.d_model}  heads={CFG.n_heads}  "
                 f"tfm={CFG.n_transformer_layers}  gnn={CFG.n_gnn_layers}\n")
        fh.write(f"Training     : epochs={CFG.epochs}  batch={CFG.batch_size}  "
                 f"lr={CFG.lr}\n")
        fh.write(f"Trainable parameters: {n_params:,}\n")
        fh.write("-" * 70 + "\n")
        fh.write("Test set comparison (n = {} proteins)\n".format(
            len(results["length"])))
        fh.write("-" * 70 + "\n")
        fh.write("                            mean dRMSD (A)   mean contact prec.\n")
        fh.write(f"  baseline                 : "
                 f"{results['drmsd_baseline'].mean():8.3f}      "
                 f"{results['cprec_baseline'].mean():8.3f}\n")
        fh.write(f"  Transformer only         : "
                 f"{results['drmsd_init'].mean():8.3f}      "
                 f"{results['cprec_init'].mean():8.3f}\n")
        fh.write(f"  Transformer + GNN        : "
                 f"{results['drmsd_final'].mean():8.3f}      "
                 f"{results['cprec_final'].mean():8.3f}\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"  Improvement over baseline       : {drop_b_to_f:+.3f} A\n")
        fh.write(f"  Improvement from GNN refinement : {drop_i_to_f:+.3f} A\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"Standard top-L/k contact precision "
                 f"(separation |i-j|>={CFG.contact_separation}, n={n_lr} chains)\n")
        fh.write("                       top-L     top-L/2    top-L/5\n")
        for tag, label in (("baseline", "baseline"),
                           ("init", "Transformer only"),
                           ("final", "Transformer + GNN")):
            tl = np.nanmean(results[f"topL_{tag}"]) * 100
            tl2 = np.nanmean(results[f"topL2_{tag}"]) * 100
            tl5 = np.nanmean(results[f"topL5_{tag}"]) * 100
            fh.write(f"  {label:<20}{tl:7.1f}%   {tl2:7.1f}%   {tl5:7.1f}%\n")
        fh.write("  Strict CASP long-range (|i-j|>=24), top-L/5:\n")
        fh.write(f"    baseline {np.nanmean(results['topL5lr_baseline'])*100:5.1f}%   "
                 f"Transformer {np.nanmean(results['topL5lr_init'])*100:5.1f}%   "
                 f"+GNN {np.nanmean(results['topL5lr_final'])*100:5.1f}%\n")
        fh.write("-" * 70 + "\n")
        fh.write(f"Test contact-weighted L1 (A) : {te_loss:.4f}\n")
        fh.write(f"Test dRMSD              (A) : {te_drmsd:.4f}\n")
        fh.write(f"Test contact precision (lenient, all-contacts) : {te_cp:.4f}\n")
    print(f"[OUT]  Wrote {summary_path}")

    print("\n" + "=" * 70)
    print(" DONE")
    print("=" * 70)
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"  Plots           : {CFG.output_dir}/")
    print(f"  Test dRMSD      : {te_drmsd:.2f} A   "
          f"(contact precision {te_cp:.2%})")


if __name__ == "__main__":
    main()
