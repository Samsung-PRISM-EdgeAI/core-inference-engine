import random
import torch
from typing import List, Dict, Any, Tuple, Set, Optional
import logging
import json # For LLM output parsing and stable hashing

# Setup basic logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestInput:
    """
    Represents a single test case in the fuzzing corpus.

    Attributes
    ----------
    data : Dict[str, Any]
        The actual input data for the NPU kernel (e.g., a dictionary describing
        an operation and its parameters).
    source_generation : int
        The generation number in which this test input was created or first added
        to the corpus. 0 for initial seeds.
    coverage_achieved : Set[str]
        A set of unique coverage points (e.g., basic block IDs, edge IDs)
        that this specific test input achieved during its execution.
    is_bug_trigger : bool
        True if this test input was observed to trigger a bug or crash.
    id : int
        A unique hash ID for the test input, based on its data.
    """
    def __init__(self,
                 data: Dict[str, Any],
                 source_generation: int = 0,
                 coverage_achieved: Optional[Set[str]] = None,
                 is_bug_trigger: bool = False):
        self.data = data