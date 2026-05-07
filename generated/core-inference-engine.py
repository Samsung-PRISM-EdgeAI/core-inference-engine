import torch
import torch.nn as nn
import numpy as np
import random
import math
from typing import Callable, List, Optional

def evaluate_program(program: str, x: float) -> float:
    """
    Evaluate a string expression containing the variable 'x' and numeric constants.
    
    Parameters
    ----------
    program : str
        Arithmetic expression, e.g. "(x+3)*2".
    x : float
        Value to substitute for the variable 'x'.
    
    Returns
    -------
    float
        Result of the evaluated expression. Returns ``float('inf')`` if evaluation fails.
    """
    try:
        return float(eval(program, {"x": x}))
    except Exception:
        return float('inf')


def build_random_program(max_depth: int) -> str:
    """
    Recursively build a random arithmetic expression string.
    
    The expression is fully parenthesized to avoid precedence issues.
    
    Parameters
    ----------
    max_depth : int
        Maximum recursion depth; controls expression size.
    
    Returns
    -------
    str
        Random arithmetic expression.
    """
    def _build(depth: int) -> str:
        if depth == 0:
            # Leaf: either the variable 'x' or a numeric constant
            if random.random() < 0.5:
                return 'x'
            else:
                return f"{random.uniform(-10, 10):.2f}"
        left = _build(depth - 1)
        right = _build(depth - 1)
        op = random.choice(['+', '-', '*', '/'])
        return f"({left}{op}{right})"
    return _build(max_depth)


def get_state(program: str, max_len: int) -> torch.Tensor:
    """
    Create a simple state representation for the policy network.
    
    Parameters
    ----------
    program : str
        The current program (expression string).
    max_len : int
        Maximum allowed program length for normalization.
    
    Returns
    -------
    torch.Tensor
        Tensor of shape (1,) containing the normalized length.
    """
    return torch.tensor([len(program)], dtype=torch.float32) / max_len


class PolicyNet(nn.Module):
    """
    Simple policy network used to guide mutation.
    
    Input: normalized program length (scalar).
    Output: logits over 4 actions: [mutate, const, x, op].
    """
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
    
    def forward(self, x):
        return self.net(x)


def core_inference_engine(
    target_func: Callable[[float], float],
    max_iter: int = 100,
    pop_size: int = 10,
    mutation_rate: float = 0.2,
    elite_frac: float = 0.2,
    max_program_length: int = 10,
    device: Optional[torch.device] = None,
) -> str:
    """
    Core inference engine implementing the evolutionary search described in
    AlphaEvolve (see https://www.semanticscholar.org/paper/ed32e5bb11a9e5e2f03ea805782e8670a6e10efd).

    The algorithm maintains a population of programs (arithmetic expressions)
    that are evolved to approximate ``target_func``. At each generation:
      1. All programs are evaluated on a set of sampled inputs.
      2. The elite fraction of the population is kept.
      3. With probability ``mutation_rate`` a program is replaced by a new
         randomly generated program (guided by a tiny policy network).
      4. The new population is formed from elites plus mutated/offspring,
         preserving ``pop_size``.

    Hyperparameters (default values):
        - max_iter: int, maximum number of generations (default 100)
        - pop_size: int, number of programs in the population (default 10)
        - mutation_rate: float, probability of generating a new random program
          instead of mutating an existing one (default 0.2)
        - elite_frac: float, fraction of the population kept as elite
          (default 0.2)
        - max_program_length: int, maximum depth of the expression tree
          (default 10)

    Returns:
        The best program (as a string) found during the search.
    """
    if device is None:
        device = torch.device('cpu')
    
    # Sample input points for evaluation
    xs = np.linspace(-10, 10, 50)
    ys = np.array([target_func(x) for x in xs], dtype=np.float32)

    # Maximum program length for state normalization
    max_len = max_program_length

    # Initialize policy network
    policy = PolicyNet().to(device)

    # Initialize population
    population: List[str] = [build_random_program(max_depth=max_program_length) for _ in range(pop_size)]

    best_program: str = None
    best_loss: float = float('inf')

    for gen in range(max_iter):
        # Evaluate all programs
        losses: List[float] = []
        programs: List[str] = []
        for prog in population:
            program_loss = 0.0
            for x_val, y_val in zip(xs, ys):
                pred = evaluate_program(prog, x_val)
                program_loss += (pred - y_val) ** 2
            program_loss /= len(xs)
            losses.append(program_loss)
            programs.append(prog)
            if program_loss < best_loss:
                best_loss = program_loss
                best_program = prog

        # Sort programs by loss
        sorted_idx = np.argsort(losses)
        sorted_programs = [programs[i] for i in sorted_idx]
        sorted_losses = [losses[i] for i in sorted_idx]

        # Select elite individuals
        elite_size = int(elite_frac * pop_size)
        elite_programs = sorted_programs[:elite_size]

        # Build new population
        new_population = elite_programs.copy()

        # Generate additional programs
        for i in range(pop_size - elite_size):
            if random.random() < mutation_rate:
                # Use policy network to decide mutation
                base_prog = elite_programs[i % elite_size]
                state = get_state(base_prog, max_len).to(device)
                logits = policy(state).squeeze(0)  # shape (4,)
                probs = torch.softmax(logits, dim=0)
                action = torch.multinomial(probs, num_samples=1).item()
                if action == 0:  # mutate -> replace with new random program
                    new_prog = build_random_program(max_depth=max_program_length)
                else:
                    new_prog = base_prog
            else:
                # Simple mutation: replace a random token
                base_prog = elite_programs[i % elite_size]
                tokens = base_prog.replace('(', ' ').replace(')', ' ').split()
                if tokens:
                    idx = random.randrange(len(tokens))
                    token_type = random.choice(['+', '-', '*', '/', 'x', 'const'])
                    if token_type == 'const':
                        tokens[idx] = f"{random.uniform(-10, 10):.2f}"
                    else:
                        tokens[idx] = token_type
                new_prog = ' '.join(tokens)

            new_population.append(new_prog)

        population = new_population

    return best_program


if __name__ == "__main__":
    def target(x):
        return 2 * x + 3

    best_prog = core_inference_engine(
        target_func=target,
        max_iter=200,
        pop_size=15,
        mutation_rate=0.3,
        elite_frac=0.25,
        max_program_length=12,
    )
    print("Best program discovered:", best_prog)

    test_x = np.array([-5, -2, 0, 1, 4])
    predictions = [evaluate_program(best_prog, x) for x in test_x]
    print("Target values :", target(test_x))
    print("Program predictions :", predictions)