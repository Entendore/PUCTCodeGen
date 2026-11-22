"""
MuZero for Program Synthesis — Batched & Checkpointed Edition
Features:
1. Batched MuZero Training Step (parallel sequence unrolls).
2. Model Checkpointing (Auto-load/Save).
3. Thread-Safe Pause/Resume mechanism.
4. Export Code to File.
5. Transformer Encoder + Action Masking + Tanh Value scaling.
"""
import os
import re
import ast
import sys
import math
import random
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# ---- CRITICAL: set Qt binding BEFORE importing matplotlib backends ----
os.environ.setdefault('QT_API', 'pyside6')

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import Levenshtein

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QTreeWidget, QTreeWidgetItem, QSplitter,
    QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog
)
from PySide6.QtCore import QThread, Signal, QObject, Qt
from PySide6.QtGui import QFont, QFontDatabase
from torch.utils.tensorboard import SummaryWriter

# ============================================================
# 1. AST NORMALIZER & STABLE HASHER
# ============================================================

class ASTNormalizer(ast.NodeTransformer):
    """Normalizes AST to catch semantically identical programs."""
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Constant) and node.right.value == 0: return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 0: return node.right
        if isinstance(node.op, ast.Sub):
            if isinstance(node.right, ast.Constant) and node.right.value == 0: return node.left
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1: return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 1: return node.right
        if isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 1: return node.left
        if isinstance(node.op, (ast.Add, ast.Mult)):
            left_dump = ast.dump(node.left)
            right_dump = ast.dump(node.right)
            if left_dump > right_dump:
                node.left, node.right = node.right, node.left
        return node

def normalize_and_hash(code: str) -> Tuple[str, str]:
    try:
        tree = ast.parse(code)
        norm_tree = ASTNormalizer().visit(tree)
        ast.fix_missing_locations(norm_tree)
        norm_code = ast.unparse(norm_tree)
        ast_hash = hashlib.sha1(ast.dump(norm_tree).encode()).hexdigest()[:16]
        return norm_code, ast_hash
    except Exception:
        return code, hashlib.sha1(code.encode()).hexdigest()[:16]

_EDIT_DISTANCE_CACHE: Dict[Tuple[str, str], float] = {}

def ast_edit_distance(generated_code: str, reference_code: str) -> float:
    key = (generated_code, reference_code)
    if key in _EDIT_DISTANCE_CACHE: return _EDIT_DISTANCE_CACHE[key]
    try:
        gen_tokens = ast.dump(ast.parse(generated_code)).split()
        ref_tokens = ast.dump(ast.parse(reference_code)).split()
        ratio = Levenshtein.ratio(" ".join(gen_tokens), " ".join(ref_tokens))
    except Exception:
        ratio = 0.0
    if len(_EDIT_DISTANCE_CACHE) < 5000: _EDIT_DISTANCE_CACHE[key] = ratio
    return ratio

# ============================================================
# 2. PEG GRAMMAR
# ============================================================

@dataclass(frozen=True)
class Production:
    name: str
    expansion: Tuple[str, ...]

class PEGGrammar:
    def __init__(self, peg_text: str):
        self.productions: List[Production] = []
        self.terminals: Dict[str, str] = {}
        self.start: Optional[str] = None
        self.by_nt: Dict[str, List[int]] = {}
        self._parse_peg(peg_text)
        vocab_tokens = ["<PAD>"] + sorted(set(self.terminals.keys()) | {p.name for p in self.productions})
        self.vocab = {s: i for i, s in enumerate(vocab_tokens)}
        self.pad_id = 0
        self.unk_id = 0

    def _parse_peg(self, peg_text: str):
        lines = [l.strip() for l in peg_text.split('\n') if l.strip() and not l.strip().startswith('#')]
        for line in lines:
            if '<-' not in line: continue
            nt_name, rules = line.split('<-', 1)
            nt_name = nt_name.strip()
            if self.start is None: self.start = nt_name
            alternatives = re.split(r'\s*/\s*|\s*\|\s*', rules.strip())
            for alt in alternatives:
                tokens = []
                for part in re.findall(r'"[^"]*"|\'[^\']*\'|\S+', alt):
                    if (part.startswith('"') and part.endswith('"')) or (part.startswith("'") and part.endswith("'")):
                        literal = part[1:-1]
                        key = f"T_{literal}"
                        tokens.append(key)
                        self.terminals[key] = literal
                    else:
                        tokens.append(part)
                prod = Production(name=nt_name, expansion=tuple(tokens))
                self.by_nt.setdefault(nt_name, []).append(len(self.productions))
                self.productions.append(prod)

    def actions_for(self, nt: str) -> List[int]: return self.by_nt.get(nt, [])
    def is_terminal(self, sym: str) -> bool: return sym in self.terminals

    def action_str(self, action: int) -> str:
        if action == -1: return "Root"
        prod = self.productions[action]
        exp_str = [f'"{self.terminals[t]}"' if t in self.terminals else t for t in prod.expansion]
        return f"{prod.name} <- {' '.join(exp_str)}"

# ============================================================
# 3. TEST RUNNER
# ============================================================

@dataclass
class TestCase:
    func_name: str
    func_args: Tuple
    expected_result: Any

    def evaluate_python(self, full_code: str, reference_code: str) -> float:
        norm_code, ast_hash = normalize_and_hash(full_code)
        local_vars: Dict[str, Any] = {}
        safe_builtins = {
            'range': range, 'len': len, 'int': int, 'float': float, 'min': min, 'max': max,
            'abs': abs, 'sorted': sorted, 'sum': sum, 'list': list, 'tuple': tuple, 'bool': bool,
            'all': all, 'any': any
        }

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(100)
        try:
            exec(norm_code, {"__builtins__": safe_builtins, "__name__": "__test__"}, local_vars)
            func = local_vars.get(self.func_name)
            if func is None: return 0.0
            result = func(*self.func_args)
            if result == self.expected_result: return 1.0
            
            # If it runs but gives wrong answer, give partial reward based on AST similarity
            edit_sim = ast_edit_distance(norm_code, reference_code)
            return 0.1 + edit_sim * 0.3
        except Exception:
            # If it crashes, give a tiny reward if it at least parsed successfully
            try:
                ast.parse(norm_code)
                edit_sim = ast_edit_distance(norm_code, reference_code)
                return 0.05 * edit_sim
            except Exception:
                return 0.0
        finally:
            sys.setrecursionlimit(old_limit)

@dataclass
class TestSuite:
    cases: List[TestCase]
    language: str = "python"
    boilerplate: str = ""
    reference_code: str = ""
    active_cases: List[TestCase] = field(default_factory=list)
    solved_count: int = 0

    def __post_init__(self):
        def complexity(c: TestCase) -> float:
            arg_c = sum(abs(a) if isinstance(a, (int, float)) else 
                        (sum(abs(x) if isinstance(x, (int, float)) else 1 for x in a) if isinstance(a, (list, tuple)) else 1)
                        for a in c.func_args)
            res_c = abs(c.expected_result) if isinstance(c.expected_result, (int, float)) else 1
            return arg_c + res_c
        self.cases.sort(key=complexity)
        if self.cases and not self.active_cases:
            self.active_cases = [self.cases[0]]

    def promote(self) -> bool:
        self.solved_count += 1
        if self.solved_count < len(self.cases):
            self.active_cases.append(self.cases[self.solved_count])
            return True
        return False

    def evaluate(self, generated_code: str) -> float:
        full_code = self.boilerplate.replace("__HOLE__", generated_code)
        total_score = 0.0
        eval_cases = list(self.active_cases)
        
        if self.solved_count > 1 and random.random() < 0.15:
            eval_cases.append(random.choice(self.cases[:self.solved_count]))
            
        for case in eval_cases:
            if self.language == "python":
                total_score += case.evaluate_python(full_code, self.reference_code)
        
        normalized = total_score / max(1, len(eval_cases))
        if normalized >= 0.999: return 1.0
        if normalized > 0: return normalized * 0.5
        return -0.05

# ============================================================
# 4. ENVIRONMENT
# ============================================================

class ProgramEnv:
    def __init__(self, grammar: PEGGrammar, test_suite: TestSuite, max_steps: int = 15):
        self.g = grammar
        self.suite = test_suite
        self.max_steps = max_steps
        self.step_penalty = -0.01
        self.cache: Dict[str, float] = {}

    def clear_cache(self): self.cache.clear()
    def reset(self) -> List[str]: return [self.g.start]

    def get_legal_actions(self, state: List[str]) -> List[int]:
        for s in state:
            if not self.g.is_terminal(s): return self.g.actions_for(s)
        return []

    def _first_nonterminal_idx(self, state: List[str]) -> int:
        for i, s in enumerate(state):
            if not self.g.is_terminal(s): return i
        return -1

    def is_terminal(self, state: List[str]) -> bool:
        if len(state) > self.max_steps: return True
        return all(self.g.is_terminal(s) for s in state)

    def step(self, state: List[str], action: int) -> Tuple[List[str], float, bool]:
        prod = self.g.productions[action]
        new_state = list(state)
        idx = self._first_nonterminal_idx(new_state)
        if idx == -1 or new_state[idx] != prod.name:
            return new_state, -0.1, True
        new_state = new_state[:idx] + list(prod.expansion) + new_state[idx + 1:]
        done = self.is_terminal(new_state)
        if done: return new_state, self._terminal_reward(new_state), True
        return new_state, self.step_penalty, False

    def _terminal_reward(self, state: List[str]) -> float:
        if len(state) > self.max_steps: return -0.5
        # Safely build code string ignoring any unresolved non-terminals
        code_body = "".join(self.g.terminals[s] for s in state if s in self.g.terminals)
        full_code = self.suite.boilerplate.replace("__HOLE__", code_body)
        _, ast_hash = normalize_and_hash(full_code)
        if ast_hash in self.cache: return self.cache[ast_hash]
        score = self.suite.evaluate(code_body)
        if len(self.cache) < 20000: self.cache[ast_hash] = score
        return score

    def encode_state(self, state: List[str]) -> torch.Tensor:
        MAX_LEN = 32
        ids = [self.g.vocab.get(s, self.g.unk_id) for s in state[:MAX_LEN]]
        while len(ids) < MAX_LEN: ids.append(self.g.pad_id)
        return torch.tensor([ids], dtype=torch.long)

# ============================================================
# 5. MuZero NEURAL NETWORK (Transformer + Masking)
# ============================================================

class MuZeroNet(nn.Module):
    def __init__(self, vocab_size: int, n_actions: int, hidden_dim: int = 128, n_heads: int = 4, max_len: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*2, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.act_emb = nn.Embedding(n_actions, hidden_dim)
        self.dyn = nn.GRUCell(hidden_dim * 2, hidden_dim)
        
        self.r_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.v_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.pi_head = nn.Linear(hidden_dim, n_actions)

    def represent(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids == 0)
        e = self.embed(token_ids)
        pos = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        e = e + self.pos_emb(pos)
        h = self.enc(e, src_key_padding_mask=mask)
        h = h * (~mask).unsqueeze(-1).float()
        h = h.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        return F.normalize(h, dim=-1)

    def dynamics(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, self.act_emb(a)], dim=-1)
        s2 = self.dyn(x, s)
        s2 = F.normalize(s2, dim=-1)
        return s2, self.r_head(s2)

    def predict(self, s: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        pi = self.pi_head(s)
        if mask is not None:
            pi = pi.masked_fill(~mask, float('-inf'))
        return pi, self.v_head(s)

# ============================================================
# 6. MCTS ALGORITHMS
# ============================================================

class MCTSNode:
    __slots__ = ("state", "prior", "reward", "terminal", "action_taken",
                 "children", "visit_count", "total_value", "amaf_visits", "amaf_value")

    def __init__(self, state: List[str], prior: float, reward: float = 0.0, terminal: bool = False, action_taken: int = -1):
        self.state = state; self.prior = prior; self.reward = reward; self.terminal = terminal; self.action_taken = action_taken
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0; self.total_value = 0.0
        self.amaf_visits: Dict[int, int] = {}; self.amaf_value: Dict[int, float] = {}

def serialize_node(node: MCTSNode, grammar: PEGGrammar, depth: int = 0, max_depth: int = 3, max_children: int = 5) -> Optional[dict]:
    if depth > max_depth: return None
    q_val = node.total_value / max(1, node.visit_count)
    action_str = grammar.action_str(node.action_taken) if node.action_taken != -1 else "Root"
    data = {"id": id(node), "visits": node.visit_count, "q_val": q_val, "action": action_str, "depth": depth, "children": []}
    children = sorted(node.children.values(), key=lambda c: c.visit_count, reverse=True)
    for child in children[:max_children]:
        child_data = serialize_node(child, grammar, depth + 1, max_depth, max_children)
        if child_data: data["children"].append(child_data)
    return data

class AcademicMCTS:
    def __init__(self, net: MuZeroNet, env: ProgramEnv, c_puct: float = 2.0, use_rave: bool = True, use_evolved_puct: bool = True, num_simulations: int = 50, discount: float = 0.997):
        self.net = net; self.env = env; self.c_puct = c_puct; self.use_rave = use_rave
        self.use_evolved_puct = use_evolved_puct; self.evolved_beta = 1.2; self.num_simulations = num_simulations; self.discount = discount

    @torch.no_grad()
    def run_simulation(self, root_state: List[str], legal_actions: List[int]) -> MCTSNode:
        tok_ids = self.env.encode_state(root_state)
        s = self.net.represent(tok_ids)
        
        mask = torch.zeros(len(self.env.g.productions), dtype=torch.bool)
        for a in legal_actions: mask[a] = True
        mask = mask.unsqueeze(0)
        
        pi_logits, _ = self.net.predict(s, mask)
        priors = torch.softmax(pi_logits[0], dim=-1).cpu().numpy()

        root = MCTSNode(root_state, prior=0.0)
        for a in legal_actions:
            next_state, r, done = self.env.step(root_state, a)
            root.children[a] = MCTSNode(next_state, prior=float(priors[a]), reward=r, terminal=done, action_taken=a)

        if legal_actions:
            noise = np.random.dirichlet([0.3] * len(legal_actions))
            for i, a in enumerate(legal_actions):
                child = root.children[a]
                child.prior = 0.75 * child.prior + 0.25 * float(noise[i])

        for _ in range(self.num_simulations):
            node = root; search_path = [node]; s_cur = s; actions_taken = []
            while True:
                legal = self.env.get_legal_actions(node.state)
                if not legal or node.terminal: break
                a = self._select_action(node, legal)
                actions_taken.append(a)
                if a in node.children:
                    node = node.children[a]; search_path.append(node)
                    s_cur, _ = self.net.dynamics(s_cur, torch.tensor([a]))
                else:
                    next_state, r, done = self.env.step(node.state, a)
                    s_cur, _ = self.net.dynamics(s_cur, torch.tensor([a]))
                    
                    child_mask = torch.zeros(len(self.env.g.productions), dtype=torch.bool)
                    for la in self.env.get_legal_actions(next_state): child_mask[la] = True
                    child_mask = child_mask.unsqueeze(0)
                    
                    pi_logits, _ = self.net.predict(s_cur, child_mask)
                    priors = torch.softmax(pi_logits[0], dim=-1).cpu().numpy()
                    child = MCTSNode(next_state, prior=float(priors[a]), reward=r, terminal=done, action_taken=a)
                    node.children[a] = child; search_path.append(child); break
            
            _, v_leaf = self.net.predict(s_cur, None)
            value = float(v_leaf.item())
            g = value
            for n in reversed(search_path):
                g = n.reward + self.discount * g
                n.visit_count += 1
                n.total_value += (g - n.total_value) / n.visit_count

            if self.use_rave:
                for i, n in enumerate(search_path):
                    for a in actions_taken[i:]:
                        legal_at_n = self.env.get_legal_actions(n.state)
                        if a in legal_at_n:
                            n.amaf_visits[a] = n.amaf_visits.get(a, 0) + 1
                            curr = n.amaf_value.get(a, 0.0)
                            n.amaf_value[a] = curr + (g - curr) / n.amaf_visits[a]
        return root

    def _select_action(self, node: MCTSNode, legal_actions: List[int]) -> int:
        best_a, best_score = -1, -1e18
        parent_visits = max(1, node.visit_count)
        for a in legal_actions:
            if a in node.children:
                child = node.children[a]
                q = child.total_value / max(1, child.visit_count)
                n = child.visit_count
                prior = child.prior
                if self.use_evolved_puct:
                    u = self.c_puct * (prior ** 1.1) * (math.sqrt(parent_visits ** self.evolved_beta) / (1 + n))
                else:
                    u = self.c_puct * prior * (math.sqrt(parent_visits) / (1 + n))
                if self.use_rave and a in node.amaf_visits:
                    amaf_n = node.amaf_visits[a]
                    if amaf_n > 0 and n > 0:
                        beta = amaf_n / (amaf_n + n + 1e-5)
                        q = (1 - beta) * q + beta * (node.amaf_value[a] / amaf_n)
            else:
                q = 0.0; u = self.c_puct * 0.5 * math.sqrt(parent_visits + 1)
            score = q + u
            if score > best_score: best_score = score; best_a = a
        return best_a

# ============================================================
# 7. SELF-PLAY & BATCHED TRAINING
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 5000): self.buffer = deque(maxlen=capacity)
    def push(self, trajectory: List[Tuple]): self.buffer.append(trajectory)
    def sample(self, batch_size: int): return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self): return len(self.buffer)
    def clear(self): self.buffer.clear()

def self_play_episode(net: MuZeroNet, env: ProgramEnv, mcts: AcademicMCTS, temperature: float = 1.0) -> Tuple[List, float, str, Optional[MCTSNode]]:
    state = env.reset(); trajectory: List[Tuple] = []; last_root: Optional[MCTSNode] = None
    while not env.is_terminal(state):
        legal = env.get_legal_actions(state)
        if not legal: break
        root = mcts.run_simulation(state, legal); last_root = root
        visits = np.zeros(len(env.g.productions), dtype=np.float32)
        for a, child in root.children.items(): visits[a] = child.visit_count
        
        # Safe Fallback for random choice if all root children resulted in 0 visits
        if visits.sum() == 0:
            pi = np.zeros_like(visits)
            pi[legal] = 1.0 / len(legal)
        else:
            if temperature > 0:
                pi = visits ** (1.0 / temperature)
                pi = pi / max(1e-8, pi.sum())
            else:
                pi = np.zeros_like(visits)
                pi[int(np.argmax(visits))] = 1.0
                
        a = int(np.random.choice(len(pi), p=pi))
        next_state, r, done = env.step(state, a)
        trajectory.append((state, pi, a, r, done)); state = next_state

    # Safely build code string ignoring any unresolved non-terminals
    code = "".join(env.g.terminals[s] for s in state if s in env.g.terminals)
    z = env._terminal_reward(state)
    returns = []; R = z
    for (_, _, _, r, _) in reversed(trajectory):
        R = r + 0.997 * R; returns.insert(0, R)
    full_traj = [(s, pi, a, r, ret) for (s, pi, a, r, _), ret in zip(trajectory, returns)]
    return full_traj, z, code, last_root

def train_step(net: MuZeroNet, opt: torch.optim.Optimizer, buffer: ReplayBuffer, env: ProgramEnv) -> Tuple[float, float, float]:
    if len(buffer) < 8: return 0.0, 0.0, 0.0
    batch = buffer.sample(4)
    B = len(batch)
    K = min(5, min(len(t) for t in batch))
    if K == 0: return 0.0, 0.0, 0.0

    states_0 = torch.cat([env.encode_state(t[0][0]) for t in batch], dim=0)
    actions = torch.tensor([[t[k][2] for k in range(K)] for t in batch], dtype=torch.long)
    pi_targets = torch.tensor([[t[k][1] for k in range(K)] for t in batch], dtype=torch.float32)
    v_targets = torch.tensor([[t[k][4] for k in range(K)] for t in batch], dtype=torch.float32)
    r_targets = torch.tensor([[t[k][3] for k in range(K)] for t in batch], dtype=torch.float32)
    
    masks = torch.zeros(B, K, len(env.g.productions), dtype=torch.bool)
    for b, t in enumerate(batch):
        for k in range(K):
            for la in env.get_legal_actions(t[k][0]): masks[b, k, la] = True

    s = net.represent(states_0)
    opt.zero_grad()
    total_loss = torch.zeros(1, requires_grad=False)
    total_policy, total_value, total_reward = 0.0, 0.0, 0.0

    for k in range(K):
        mask_k = masks[:, k, :]
        pi_logits, v = net.predict(s, mask_k)
        p_loss = F.cross_entropy(pi_logits, pi_targets[:, k].argmax(dim=1))
        v_loss = F.mse_loss(v.squeeze(-1), v_targets[:, k])
        
        s_next, r_pred = net.dynamics(s, actions[:, k])
        r_loss = F.mse_loss(r_pred.squeeze(-1), r_targets[:, k])
        
        total_loss = total_loss + p_loss + v_loss + r_loss
        total_policy += p_loss.item(); total_value += v_loss.item(); total_reward += r_loss.item()
        s = s_next 

    total_loss = total_loss / B
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    opt.step()
    return (total_loss.item(), total_policy / K, total_value / K)

# ============================================================
# 8. WORKER & GUI (Checkpointing & Pause)
# ============================================================

DARK_STYLE = """
QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: 'Segoe UI', sans-serif; }
QPushButton { background-color: #007acc; color: white; border: none; padding: 6px 12px; border-radius: 4px; min-width: 80px; }
QPushButton:hover { background-color: #0098ff; }
QPushButton:pressed { background-color: #005a9e; }
QPushButton:checked { background-color: #005a9e; }
QTextEdit, QLineEdit, QTreeWidget, QSpinBox, QDoubleSpinBox {
    background-color: #252526; border: 1px solid #3c3c3c; border-radius: 4px; padding: 4px;
}
QTextEdit:focus, QLineEdit:focus { border: 1px solid #007acc; }
QHeaderView::section { background-color: #333333; padding: 4px; border: 0px; }
QSplitter::handle { background-color: #3c3c3c; }
QLabel { background: transparent; }
"""

class SynthesisWorker(QObject):
    stats_signal = Signal(dict); code_signal = Signal(str); mcts_signal = Signal(list)
    graph_signal = Signal(dict); log_signal = Signal(str); finished_signal = Signal()

    def __init__(self, target_lang: str = "python", max_episodes: int = 100000):
        super().__init__(); self.is_running = True; self.target_lang = target_lang
        self.user_prefix = ""; self.max_episodes = max_episodes
        self.writer = SummaryWriter('runs/muzero_synth_dev')
        self.pause_event = threading.Event(); self.pause_event.set()

    def setup(self):
        # Removed Left Recursion (Expr_Int <- Expr_Int...) to prevent state blowups
        peg_content = """
Start <- "return " Expr_Int
Expr_Int <- Term_Int "+" Expr_Int | Term_Int "-" Expr_Int | Term_Int
Term_Int <- "len(" Expr_List ")" | "sum(" Expr_List ")" | "max(" Expr_List ")" | "min(" Expr_List ")" | Atom_Int
Atom_Int <- "x" | "y" | "10" | "0"
Expr_List <- "x" | "y" | "sort(" Expr_List ")"
"""
        self.grammar = PEGGrammar(peg_content)
        test_cases = [
            TestCase(func_name="f", func_args=([3, 1, 2],), expected_result=3),
            TestCase(func_name="f", func_args=([0, 0, 0, 0],), expected_result=4),
            TestCase(func_name="f", func_args=([],), expected_result=0),
            TestCase(func_name="f", func_args=([5, 5],), expected_result=2),
        ]
        reference_code = "def f(x):\n    return len(x)"
        boilerplate = f"def f(x):\n    {self.user_prefix}\n    __HOLE__"
        self.suite = TestSuite(cases=test_cases, language=self.target_lang, boilerplate=boilerplate, reference_code=reference_code)
        self.env = ProgramEnv(self.grammar, self.suite, max_steps=15)
        self.net = MuZeroNet(len(self.grammar.vocab), len(self.grammar.productions), hidden_dim=128)
        
        if os.path.exists("muzero_synth.pth"):
            try:
                self.net.load_state_dict(torch.load("muzero_synth.pth"))
                self.log_signal.emit("Loaded existing model weights.")
            except Exception as e:
                self.log_signal.emit(f"Failed to load weights: {e}")
                
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.mcts = AcademicMCTS(self.net, self.env, c_puct=2.0, use_rave=True, use_evolved_puct=True, num_simulations=50)
        self.buffer = ReplayBuffer()

    def run(self):
        self.setup()
        self.log_signal.emit("Starting MuZero Program Synthesis [Batched & Checkpointed]...")
        self.log_signal.emit(f"Curriculum: Test {self.suite.solved_count + 1}/{len(self.suite.cases)}")
        for ep in range(self.max_episodes):
            if not self.is_running: break
            self.pause_event.wait()
            
            temperature = 1.0 if ep < 50 else (0.5 if ep < 150 else 0.1)
            trajectory, reward, code, root = self_play_episode(self.net, self.env, self.mcts, temperature=temperature)
            self.buffer.push(trajectory)
            loss, p_loss, v_loss = train_step(self.net, self.opt, self.buffer, self.env)
            self.writer.add_scalar('Loss/Total', loss, ep); self.writer.add_scalar('Stats/Reward', reward, ep)
            
            mcts_depth = 0
            if root:
                def get_depth(node):
                    if not node.children: return 0
                    return 1 + max(get_depth(c) for c in node.children.values())
                mcts_depth = get_depth(root)
            
            stats = {"episode": ep, "reward": reward, "loss": loss, "buf_size": len(self.buffer), 
                     "test_level": self.suite.solved_count + 1, "total_tests": len(self.suite.cases), 
                     "mcts_depth": mcts_depth, "temperature": temperature}
            self.stats_signal.emit(stats)
            
            full_code = self.suite.boilerplate.replace("__HOLE__", code)
            try: self.code_signal.emit(ast.unparse(ast.parse(full_code)))
            except Exception: self.code_signal.emit(full_code)
                
            mcts_data = []
            if root:
                for a, child in sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True):
                    q_val = child.total_value / max(1, child.visit_count)
                    mcts_data.append({"action_str": self.grammar.action_str(a), "visits": child.visit_count, "q_val": q_val, "prior": child.prior})
            self.mcts_signal.emit(mcts_data)
            
            if root: self.graph_signal.emit(serialize_node(root, self.grammar, max_depth=3))
            
            if reward >= 1.0:
                self.log_signal.emit(f"[Ep {ep}] SOLVED TEST {self.suite.solved_count + 1}!")
                if self.suite.promote():
                    self.log_signal.emit(f">>> CURRICULUM ADVANCING TO TEST {self.suite.solved_count + 1}/{len(self.suite.cases)} <<<")
                    self.env.clear_cache(); self.buffer.clear()
                else:
                    self.log_signal.emit("*** ALL TESTS PASSED! SYNTHESIS COMPLETE. ***")
                    self.save_model()
                    self.finished_signal.emit(); return
                    
        self.log_signal.emit("Synthesis stopped or reached episode limit.")
        self.finished_signal.emit()

    def stop(self): self.is_running = False
    def toggle_pause(self, paused: bool):
        if paused: self.pause_event.clear()
        else: self.pause_event.set()
        
    def update_prefix(self, new_prefix: str):
        self.user_prefix = new_prefix
        self.log_signal.emit("User updated prefix. Restarting search...")
        self.env.clear_cache(); self.buffer.clear(); self.setup()
        
    def update_config(self, num_sims: int, lr: float):
        if hasattr(self, 'mcts'): self.mcts.num_simulations = num_sims
        if hasattr(self, 'opt'):
            for g in self.opt.param_groups: g['lr'] = lr
        self.log_signal.emit(f"Config updated -> Sims: {num_sims}, LR: {lr}")

    def save_model(self):
        try:
            torch.save(self.net.state_dict(), "muzero_synth.pth")
            self.log_signal.emit("Model saved to muzero_synth.pth")
        except Exception as e:
            self.log_signal.emit(f"Error saving model: {e}")

class MCTSGraphCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#1e1e1e'); super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111); self.G = nx.DiGraph()
    def update_graph(self, root_data: Optional[dict]):
        self.ax.clear(); self.G.clear(); new_G = nx.DiGraph()
        def add_nodes_edges(node_data, parent_id=None):
            node_id = node_data["id"]
            label = f"{node_data['action'][:15]}\nV:{node_data['visits']}\nQ:{node_data['q_val']:.2f}"
            color = "#90ee90" if node_data["q_val"] > 0.5 else ("#fa8072" if node_data["q_val"] < 0 else "#add8e6")
            new_G.add_node(node_id, label=label, color=color)
            if parent_id is not None: new_G.add_edge(parent_id, node_id)
            for child in node_data["children"]: add_nodes_edges(child, node_id)
        if root_data: add_nodes_edges(root_data)
        self.G = new_G
        if len(self.G.nodes) > 0:
            colors = [d['color'] for _, d in self.G.nodes(data=True)]
            labels = {n: d['label'] for n, d in self.G.nodes(data=True)}
            try: pos = nx.nx_agraph.graphviz_layout(self.G, prog='dot')
            except Exception: pos = nx.spring_layout(self.G, seed=42)
            nx.draw(self.G, pos, ax=self.ax, with_labels=True, labels=labels, node_color=colors, 
                    node_size=1500, font_size=7, font_family="monospace", font_color='#1e1e1e', arrows=True)
        else:
            self.ax.text(0.5, 0.5, "Waiting for MCTS data...", ha='center', va='center', color='#d4d4d4', transform=self.ax.transAxes)
        self.ax.set_facecolor('#1e1e1e')
        for spine in self.ax.spines.values(): spine.set_color('#3c3c3c')
        self.ax.set_title("Live MCTS Tree", color='#d4d4d4')
        self.draw_idle()

class SynthesisGUI(QMainWindow):
    def __init__(self, target_lang: str = "python"):
        super().__init__(); self.target_lang = target_lang
        self.setWindowTitle("MuZero Synthesis [Batched & Checkpointed]")
        self.resize(1400, 900)
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central); splitter = QSplitter(Qt.Horizontal); main_layout.addWidget(splitter)

        left = QWidget(); left_layout = QVBoxLayout(left)
        self.stats_label = QLabel("Ep: 0 | R: 0.00 | L: 0.0000 | Buf: 0 | Test: 0/0 | D: 0 | T: 1.0")
        self.stats_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 5px;"); left_layout.addWidget(self.stats_label)
        
        config_layout = QHBoxLayout()
        config_layout.addWidget(QLabel("Sims:"))
        self.sim_spin = QSpinBox(); self.sim_spin.setRange(10, 200); self.sim_spin.setValue(50); config_layout.addWidget(self.sim_spin)
        config_layout.addWidget(QLabel("LR:"))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.0001, 0.1); self.lr_spin.setValue(0.001); self.lr_spin.setSingleStep(0.0005); self.lr_spin.setDecimals(4); config_layout.addWidget(self.lr_spin)
        self.config_btn = QPushButton("Apply"); config_layout.addWidget(self.config_btn)
        left_layout.addLayout(config_layout)
        
        left_layout.addWidget(QLabel("<b>MCTS Root Children:</b>"))
        self.tree_widget = QTreeWidget(); self.tree_widget.setColumnCount(4)
        self.tree_widget.setHeaderLabels(["Action", "Visits", "Q-Value", "Prior"])
        left_layout.addWidget(self.tree_widget)
        splitter.addWidget(left)

        center = QWidget(); center_layout = QVBoxLayout(center)
        center_layout.addWidget(QLabel("<b>Human Pair-Programming Prefix (Optional):</b>"))
        prefix_layout = QHBoxLayout()
        self.prefix_input = QLineEdit(); self.prefix_input.setPlaceholderText("e.g., z = len(x); return z")
        prefix_layout.addWidget(self.prefix_input)
        self.set_btn = QPushButton("Solve From Here"); prefix_layout.addWidget(self.set_btn)
        center_layout.addLayout(prefix_layout)

        center_layout.addWidget(QLabel("<b>Generated Code (Live):</b>"))
        self.code_text = QTextEdit(); self.code_text.setReadOnly(True)
        self.code_text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        center_layout.addWidget(self.code_text, 2)
        
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause"); self.pause_btn.setCheckable(True); btn_layout.addWidget(self.pause_btn)
        self.save_btn = QPushButton("Save Model"); btn_layout.addWidget(self.save_btn)
        self.export_btn = QPushButton("Export Code"); btn_layout.addWidget(self.export_btn)
        center_layout.addLayout(btn_layout)
        
        center_layout.addWidget(QLabel("<b>Event Log:</b>"))
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(150)
        center_layout.addWidget(self.log_text, 1)
        splitter.addWidget(center)

        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Live MCTS Graph Visualization:</b>"))
        self.graph_canvas = MCTSGraphCanvas(self); right_layout.addWidget(self.graph_canvas)
        splitter.addWidget(right); splitter.setSizes([300, 600, 500])

        self.thread = QThread(); self.worker = SynthesisWorker(target_lang)
        self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
        self.worker.stats_signal.connect(self.update_stats); self.worker.code_signal.connect(self.update_code)
        self.worker.mcts_signal.connect(self.update_mcts_tree); self.worker.graph_signal.connect(self.update_graph)
        self.worker.log_signal.connect(self.update_log); self.worker.finished_signal.connect(self.thread.quit)
        self.set_btn.clicked.connect(self.trigger_prefix_update)
        self.config_btn.clicked.connect(lambda: self.worker.update_config(self.sim_spin.value(), self.lr_spin.value()))
        self.pause_btn.clicked.connect(self.worker.toggle_pause)
        self.save_btn.clicked.connect(self.worker.save_model)
        self.export_btn.clicked.connect(self.export_code)

    def start(self): self.thread.start()
    def trigger_prefix_update(self): self.worker.update_prefix(self.prefix_input.text())
    def export_code(self):
        code = self.code_text.toPlainText()
        if not code: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Synthesized Code", "synthesized.py", "Python Files (*.py)")
        if file_path:
            try:
                with open(file_path, "w") as f: f.write(code)
                self.log_text.append(f"[System] Code exported to {file_path}")
            except Exception as e: self.log_text.append(f"[Error] Failed to save: {e}")
                
    def update_stats(self, stats):
        self.stats_label.setText(f"Ep: {stats['episode']} | R: {stats['reward']:.2f} | L: {stats['loss']:.4f} | Buf: {stats['buf_size']} | Test: {stats['test_level']}/{stats['total_tests']} | D: {stats['mcts_depth']} | T: {stats.get('temperature', 1.0):.2f}")
    def update_code(self, code): self.code_text.setPlainText(code)
    def update_graph(self, graph_data): self.graph_canvas.update_graph(graph_data)
    def update_mcts_tree(self, mcts_data):
        self.tree_widget.clear()
        for item_data in mcts_data:
            item = QTreeWidgetItem([item_data["action_str"], str(item_data["visits"]), f"{item_data['q_val']:.3f}", f"{item_data['prior']:.3f}"])
            if item_data["q_val"] > 0.5: item.setBackground(0, Qt.darkGreen)
            elif item_data["q_val"] < 0: item.setBackground(0, Qt.darkRed)
            self.tree_widget.addTopLevelItem(item)
    def update_log(self, msg): self.log_text.append(msg)
    def closeEvent(self, event):
        self.worker.stop(); self.thread.quit(); self.thread.wait(3000); event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    gui = SynthesisGUI(target_lang="python")
    gui.show(); gui.start()
    sys.exit(app.exec())