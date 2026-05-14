"""Memory profiler v2 — with precomputed embeddings."""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, io, time, gc
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import psutil

def get_mem_mb():
    p = psutil.Process(os.getpid())
    return p.memory_info().rss / (1024 * 1024)

def fmt(mb):
    return f"{mb:.1f} MB"

print("=" * 70)
print("  MEMORY PROFILER v2 — Precomputed Embeddings")
print("=" * 70)

gc.collect()
baseline = get_mem_mb()
print(f"\n[0] Python baseline:                  {fmt(baseline)}")

import json, requests, numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
gc.collect()
after_core = get_mem_mb()
print(f"[1] After core imports:               {fmt(after_core)} (+{fmt(after_core - baseline)})")

from app.catalog import catalog
catalog.load()
gc.collect()
after_catalog = get_mem_mb()
print(f"[2] After catalog load:               {fmt(after_catalog)} (+{fmt(after_catalog - after_core)})")

t3 = time.time()
from app.retriever import retriever
retriever.build()
gc.collect()
after_retriever = get_mem_mb()
t_ret = time.time() - t3
print(f"[3] After retriever build:            {fmt(after_retriever)} (+{fmt(after_retriever - after_catalog)}) [{t_ret:.1f}s]")

from app.agent import get_agent
agent = get_agent()
gc.collect()
after_agent = get_mem_mb()
print(f"[4] After agent init:                 {fmt(after_agent)} (+{fmt(after_agent - after_retriever)})")

# Simulate query
query = "We need assessments for senior leadership, CXO level, selection benchmark"
results = retriever.search_hybrid(query, top_k=20)
gc.collect()
after_query = get_mem_mb()
print(f"[5] After retrieval query:            {fmt(after_query)} (+{fmt(after_query - after_agent)})")

peak = after_query
total_time = time.time() - t3

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  Startup time:        {total_time:.1f}s")
print(f"  Peak RSS:            {fmt(peak)}")
print(f"  Render free limit:   512.0 MB")
print(f"  Headroom:            {fmt(512.0 - peak)}")
print(f"  vs previous (988 MB): SAVED {fmt(988 - peak)}")
if peak > 512:
    print(f"  STATUS: *** STILL EXCEEDS RENDER FREE TIER ***")
elif peak > 400:
    print(f"  STATUS: *** TIGHT — may OOM under load ***")
else:
    print(f"  STATUS: FITS within Render free tier")
print(f"{'='*70}")
