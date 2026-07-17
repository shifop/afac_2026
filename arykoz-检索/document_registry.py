#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Document path discovery independent from shadow/baseline evaluation code."""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional


def build_doc_map(runtime_index=None, extra_dirs: Optional[Iterable[str]] = None) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if runtime_index is not None:
        for did, meta in runtime_index.meta_table.items():
            path = str(getattr(meta, "filepath", "") or "")
            domain = str(getattr(meta, "domain", "") or "")
            if path and os.path.exists(path):
                result[f"{domain}:{did}"] = path
    directories = list(extra_dirs or [])
    try:
        import b_config as cfg
        directories.extend([
            getattr(cfg, "EXTRACTED_DIR", ""),
            getattr(cfg, "REGULATORY_V2_DIR", ""),
        ])
    except Exception:
        pass
    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if not filename.endswith(".txt"):
                continue
            stem = filename[:-4]
            domain = ""
            doc_id = ""
            if stem.startswith("insurance_"):
                domain, doc_id = "insurance", stem.split("insurance_", 1)[1]
            elif stem.startswith("financial_contracts_"):
                domain, doc_id = "financial_contracts", stem.split("financial_contracts_", 1)[1]
            elif stem.startswith("financial_reports_"):
                domain, doc_id = "financial_reports", stem.split("financial_reports_", 1)[1]
            elif stem.startswith("research_"):
                domain, doc_id = "research", stem.split("research_", 1)[1]
            elif stem.startswith(("csrc_", "strict_v3_", "regulatory_")):
                domain, doc_id = "regulatory", stem
            if domain and doc_id:
                result.setdefault(f"{domain}:{doc_id}", os.path.join(directory, filename))
    return result
