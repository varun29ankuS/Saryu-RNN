"""Append-only metric logging for local runs, readable by scripts/dashboard.py.

One JSON object per line, so a run can be tailed while it trains and nothing is lost if it dies.
Files land in runs/<name>.jsonl. Keep the metric names stable across arms -- the dashboard groups by
name and overlays runs, which is the whole point of having one.

    from saryu.metrics import Run
    run = Run('base', config=dict(pairs=4, gap=4, lr=3e-4))
    run.log(step, loss=..., top1=..., top5=...)
    run.done()

Names may use a '/' to group: 'eval/top1', 'grad/norm'. The dashboard uses the prefix as a section.
"""
from __future__ import annotations

import json
import math
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'runs')


class Run:
    def __init__(self, name, config=None, root=RUNS):
        os.makedirs(root, exist_ok=True)
        self.name = name
        self.path = os.path.join(root, f'{name}.jsonl')
        self.t0 = time.time()
        # Truncate: a re-run of the same name replaces it, so the dashboard never mixes two runs.
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({'kind': 'meta', 'run': name, 'started': self.t0,
                                'config': config or {}}) + '\n')

    def log(self, step, **metrics):
        def clean(v):
            # NaN/Infinity are valid Python json output but NOT valid JSON, and JSON.parse in the
            # dashboard rejects the whole line. Send null instead; a missing point is drawn as a
            # gap, which is what an undefined metric should look like.
            if v is None:
                return None
            v = float(v)
            return v if math.isfinite(v) else None

        rec = {'kind': 'point', 'step': int(step), 't': round(time.time() - self.t0, 3),
               'metrics': {k: clean(v) for k, v in metrics.items()}}
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')

    def done(self, **summary):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'kind': 'done', 'elapsed': round(time.time() - self.t0, 2),
                                'summary': summary}) + '\n')
