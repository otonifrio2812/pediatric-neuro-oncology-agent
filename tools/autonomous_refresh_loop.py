#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistent loop: update PubMed/ClinicalTrials evidence, then run watcher.py."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from literature_trial_updater import refresh_evidence_sources


def run_once(pubmed_days: int = 30, pubmed_retmax: int = 50, offline: bool = False) -> None:
    """One refresh + watcher cycle. Demo-resilient: never raises.

    - offline=True: skip the PubMed/CT.gov calls entirely; the watcher still runs.
    - offline=False: try refresh; if it raises (extremely unlikely — internal calls
      are already guarded), log and continue to the watcher so the demo always
      produces a fresh report.
    """
    if offline:
        print('[refresh-loop] offline mode: skipping PubMed/ClinicalTrials.gov refresh.')
        print('[refresh-loop] existing rag_sources/ cache preserved; SimpleRAG will serve it.')
    else:
        try:
            print(refresh_evidence_sources(pubmed_days=pubmed_days, pubmed_retmax=pubmed_retmax))
        except Exception as exc:
            # Defensive: refresh_evidence_sources already wraps each API call, but if
            # an unexpected error escapes, we MUST NOT block the watcher / demo.
            print(f'[refresh-loop] refresh failed (graceful skip): {exc}')

    watcher_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agent', 'watcher.py'))
    if Path(watcher_path).exists():
        subprocess.run([sys.executable, watcher_path, '--once'], check=False)
    else:
        print(f'[refresh-loop] watcher.py not found at {watcher_path}; skipped.')


def main() -> None:
    p = argparse.ArgumentParser(
        description='Persistent autonomous loop: refresh PubMed/CT.gov, then run the agent watcher.'
    )
    p.add_argument('--once', action='store_true', help='Run one cycle and exit.')
    p.add_argument('--interval-hours', type=float, default=6.0, help='Loop interval (default 6h).')
    p.add_argument('--pubmed-days', type=int, default=30)
    p.add_argument('--pubmed-retmax', type=int, default=50)
    p.add_argument('--offline', action='store_true',
                   help='Skip network refresh; preserve existing rag_sources/ cache. '
                        'Watcher and run_demo still execute (demo-friendly).')
    args = p.parse_args()
    if args.once:
        run_once(args.pubmed_days, args.pubmed_retmax, offline=args.offline)
        return
    while True:
        try:
            run_once(args.pubmed_days, args.pubmed_retmax, offline=args.offline)
        except Exception as exc:
            print('loop error:', exc)
        time.sleep(int(args.interval_hours * 3600))


if __name__ == '__main__':
    main()
