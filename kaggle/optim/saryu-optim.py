"""AdamW against Muon on Saryu itself, stage A: learning-rate sweep at 5M, one seed.

WHY. Cerruti et al. 2026 (arXiv 2607.07953) report that Muon consistently beats AdamW across
DeltaNet, Gated DeltaNet, KDA and Gated DeltaNet-2 at 350M-3B. Saryu's trained checkpoints used
AdamW, and its transport (products of reflections, orthogonal at beta = 2) is not one of the
architectures they measured, so the result does not transfer for free: Muon's update is itself
orthogonalised, and what that does to an already-orthogonal transport is the open question.

WHAT IS CONTROLLED. Same model, data, batch, steps, schedule (WSD), gradient clipping, z-loss,
initialisation and evaluation for every arm; the only difference is the optimiser and its learning
rate. In muon mode the embeddings, the head and every 1-D parameter stay on AdamW at 1e-3, as Muon
requires; the swept rate is Muon's own. Both optimisers get a rate sweep, so neither wins by being
the only one tuned.

WHAT IS MEASURED. bits per character at four context lengths, scored at IDENTICAL positions across
lengths (the 2026-09-16 evaluation fix), plus seconds per step: Muon's Newton-Schulz costs time, so
the honest comparison is loss against wall clock, not loss against steps.

STAGE B (separate run, after this one): the best rate of each optimiser at three seeds, then the
winner at 25M. Do not read a single-seed sweep as a result; it selects the rate, nothing more.

Runtime: 6 arms x 5,000 steps at ~0.2 s/step plus evaluation is about 2 hours on one T4.
"""
import os
import subprocess
import sys
import time

REPO = 'https://github.com/varun29ankuS/Saryu-RNN'
WORK = '/kaggle/working/saryu'
ARMS = 'adamw@5e-4,adamw@1e-3,adamw@2e-3,muon@0.01,muon@0.02,muon@0.04'

t0 = time.time()
subprocess.run(['git', 'clone', '--depth', '1', REPO, WORK], check=True)
os.chdir(WORK)
print('repo at', subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True).stdout.strip(), flush=True)

corpus = next((os.path.join(r, f) for r, _, fs in os.walk('/kaggle/input') for f in fs
               if f == 'enwik8'), None)
if corpus is None:                      # no dataset attached: fetch it (kernel needs internet on)
    os.makedirs('corpus', exist_ok=True)
    subprocess.run('curl -sL http://mattmahoney.net/dc/enwik8.zip -o /tmp/enwik8.zip '
                   '&& unzip -o -q /tmp/enwik8.zip -d corpus', shell=True, check=True)
    corpus = 'corpus/enwik8'
print('corpus', corpus, os.path.getsize(corpus), 'bytes', flush=True)

env = dict(os.environ, CORPUS=corpus, ARMS=ARMS, SEEDS='0',
           TARGET_PARAMS='5000000', STEPS='5000', CTX='128', BS='48', NL='4',
           EVAL_LENS='128,512,2048,8192', EVAL_ANCHOR='8192', SAVE='', WALLCAP_S='36000')
subprocess.run([sys.executable, 'scripts/train.py'], env=env, check=True)
print('total {:.0f}s'.format(time.time() - t0), flush=True)
