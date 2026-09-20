"""Live dashboard for local runs: charts plus a diagnosis panel that says when it is going wrong.

    python scripts/dashboard.py            # http://127.0.0.1:8472
    python scripts/dashboard.py --port 9000 --runs runs

Any script that logs through saryu.metrics.Run shows up automatically and updates while it trains:

    from saryu.metrics import Run
    run = Run('base', config=dict(pairs=4, gap=4, chance_loss=4.159))
    run.log(step, **{'loss': l, 'eval/top1': a1, 'eval/top5': a5, 'grad/norm': gn})

The page polls every 3 seconds, so a run that is still writing is shown live and a half-written
final line is ignored rather than breaking the view.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))


def make_handler(runs_dir):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=HERE, **kw)

        def _send(self, body, ctype='application/json'):
            body = body.encode('utf-8') if isinstance(body, str) else body
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split('?')[0]
            if path in ('/', '/index.html'):
                with open(os.path.join(HERE, 'dashboard.html'), 'rb') as f:
                    return self._send(f.read(), 'text/html; charset=utf-8')
            if path == '/runs.json':
                names = []
                if os.path.isdir(runs_dir):
                    names = sorted(f[:-6] for f in os.listdir(runs_dir) if f.endswith('.jsonl'))
                return self._send(json.dumps({'runs': names}))
            if path.startswith('/runs/'):
                name = os.path.basename(path[len('/runs/'):])
                fp = os.path.join(runs_dir, name)
                if os.path.isfile(fp) and name.endswith('.jsonl'):
                    with open(fp, 'rb') as f:
                        return self._send(f.read(), 'text/plain; charset=utf-8')
                return self.send_error(404)
            return self.send_error(404)

        def log_message(self, *a):
            pass                                   # the poll loop would drown the console
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8472)
    ap.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    a = ap.parse_args()
    os.makedirs(a.runs, exist_ok=True)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(('127.0.0.1', a.port), make_handler(a.runs)) as httpd:
        print(f'dashboard  http://127.0.0.1:{a.port}   watching {a.runs}')
        print('Ctrl-C to stop')
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == '__main__':
    main()
