"""Minimal HTTP wrapper around main.py's t9 decoder for Railway deploy.

The repo is a single-file T9 dictionary toy (main.py decodes a bz2 blob).
Railway services need something that listens on $PORT and answers /health.

This file exposes:
  GET /health           — always 200 ok
  GET /                 — service info
  GET /lookup?q=<digits> — t9 lookup for a digit string (e.g. ?q=43556 → ['hello'])
  GET /words            — full word list

Wraps main.py via subprocess (so we don't have to import the obfuscated blob).
"""
from __future__ import annotations
import os
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="teeny-tiny-t9 imt-api", version="1.0.0")

# Pre-build the dictionary once at startup by importing main.
sys.path.insert(0, os.path.dirname(__file__))
import main  # populates main.D

DICT: dict[str, list[str]] = main.D
WORDS: list[str] = main.W


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "word_count": len(WORDS), "dict_keys": len(DICT)}


@app.get("/")
def root() -> dict:
    return {
        "service": "teeny-tiny-t9 imt-api",
        "endpoints": ["/health", "/lookup?q=<digits>", "/words"],
        "word_count": len(WORDS),
    }


@app.get("/lookup")
def lookup(q: str) -> dict:
    if not q.isdigit():
        raise HTTPException(400, "q must be digits only")
    return {"q": q, "words": DICT.get(q, [])}


@app.get("/words")
def words() -> dict:
    return {"count": len(WORDS), "words": WORDS}
