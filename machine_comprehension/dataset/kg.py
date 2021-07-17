from .utils import aho_create_statemachine, aho_find_all
import json


class KGEmbeddings:
    def __init__(self, patterns2id_path):
        with open(patterns2id_path) as f:
            self.patterns2id = json.load(f)

        self.forest = aho_create_statemachine(self.patterns2id.keys())
        self.len_patterns = {p: len(p.split(' ')) for p in self.patterns2id.leys()}

    def process(self, text):
        positions = aho_find_all(text.split(' '), self.forest)

        # Check overlap
        # Keep the one that starts first
        next = -1
        validated_positions = []

        for start, pattern in positions:
            if start > next:
                next = start + self.len_patterns[pattern]
                validated_positions.append((start, pattern))

        # Get ids
        # We repeat the KG embedding when it appears multiple times
        ids = [0] * len(text)
        for start, pattern in validated_positions:
            size = self.len_patterns[pattern]
            ids[start:start + size] = [self.patterns2id[pattern]] * size

        return ids
