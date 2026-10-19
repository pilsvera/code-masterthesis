import heapq
from collections import defaultdict
from typing import Dict, List, Tuple

class TopKSelector:
    """Keep top-K correct overall and top-K mistakes for each (true,pred) pair."""
    def __init__(self, k_correct: int = 5, k_error: int = 3):
        self.k_correct = k_correct
        self.k_error = k_error
        self._correct_heap: List[Tuple[float, int, dict]] = []  # (conf, ds_idx, rec)
        self._error_heaps: Dict[Tuple[int, int], List[Tuple[float, int, dict]]] = defaultdict(list)

    @staticmethod
    def _push_bounded(heap: List[Tuple[float, int, dict]], item: Tuple[float, int, dict], k: int):
        if k <= 0:
            return
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            # compare by confidence only; ds_idx just breaks ties deterministically
            if item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] < heap[0][1]):
                heapq.heapreplace(heap, item)

    def add(self, rec: dict):
        """
        rec must have:
          - 'conf'  (float)    confidence/score
          - 'true'  (int)
          - 'pred'  (int)
          - 'ds_idx' (int)     dataset index (used as tiebreaker)
        you can include 'name' etc. too.
        """
        conf = float(rec["conf"])
        ds_idx = int(rec["ds_idx"])
        item = (conf, ds_idx, rec)

        if rec["true"] == rec["pred"]:
            self._push_bounded(self._correct_heap, item, self.k_correct)
        else:
            pair = (int(rec["true"]), int(rec["pred"]))
            self._push_bounded(self._error_heaps[pair], item, self.k_error)

    def selected(self) -> List[dict]:
        """Return selected records sorted by descending confidence (stable on ties)."""
        out: List[dict] = []
        out.extend([rec for (c, i, rec) in sorted(self._correct_heap, key=lambda x: (x[0], -x[1]), reverse=True)])
        for heap in self._error_heaps.values():
            out.extend([rec for (c, i, rec) in sorted(heap, key=lambda x: (x[0], -x[1]), reverse=True)])
        return out
