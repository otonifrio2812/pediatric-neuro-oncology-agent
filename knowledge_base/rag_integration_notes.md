# RAG integration notes

The agent retrieves from `knowledge_base/*.md` and `rag_sources/*.jsonl`.

Recommended autonomous refresh:

```bash
python autonomous_refresh_loop.py --once --pubmed-days 30 --pubmed-retmax 50
```

Persistent deployment example:

```bash
nohup python autonomous_refresh_loop.py --interval-hours 6 > logs/refresh_loop_stdout.log 2>&1 &
```

ClinicalTrials.gov and PubMed records are stored as JSONL and then used as evidence snippets for Nemotron reasoning.
