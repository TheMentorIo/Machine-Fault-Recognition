from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MicroBatcher:
    max_workers: int = 2

    def map_ordered(self, items: list[Path], function):
        ordered = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(function, item): index for index, item in enumerate(items)}
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        return ordered
