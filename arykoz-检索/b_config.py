#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AFAC retrieval V2 lean configuration.

Only configuration consumed by the V2 production path is kept here.  The
retired V1/V2 scorer weights, legacy DeepSeek defaults, and baseline-specific
settings are intentionally excluded.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _project_root() -> Path:
    env = os.environ.get("AFAC_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    current = HERE
    for _ in range(9):
        if (current / "extracted_texts").is_dir() or (current / "03_PyMuPDF提取原文" / "extracted_texts").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    return HERE


PROJECT_ROOT = str(_project_root())


def _first_existing(candidates: list[Path]) -> str:
    for path in candidates:
        if path.is_dir():
            return str(path.resolve())
    return str(candidates[0].resolve())


_root = Path(PROJECT_ROOT)
EXTRACTED_DIR = _first_existing([
    _root / "extracted_texts",
    _root / "03_PyMuPDF提取原文" / "extracted_texts",
    HERE / "extracted_texts",
])
REGULATORY_V2_DIR = _first_existing([
    _root / "extracted_texts_v2",
    _root / "03_PyMuPDF提取原文" / "extracted_texts_v2",
    HERE / "extracted_texts_v2",
])
QUESTIONS_DIR = _first_existing([
    _root / "questions",
    _root / "02_官方题目",
    _root / "02_官方题目" / "questions",
    HERE / "questions",
])

INDEX_V2_PKL = os.environ.get("AFAC_INDEX_V2_PKL", str(HERE / "b_index_v2.pkl"))
INDEX_PKL = INDEX_V2_PKL
FACT_INDEX_PKL = os.environ.get("AFAC_FACT_INDEX_PKL", INDEX_V2_PKL + ".fact_index.pkl")

OUT_CSV = os.environ.get("AFAC_OUT_CSV", "answer.csv")
OUT_EVIDENCE = os.environ.get("AFAC_OUT_EVIDENCE", "evidence.json")
OUT_REPORT = os.environ.get("AFAC_OUT_REPORT", "retrieval_report.json")

QWEN_API_KEY = os.environ.get("DASHSCOPE_API_KEY", os.environ.get("QWEN_API_KEY", ""))
QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
INSURANCE_LLM_MODEL = os.environ.get("INSURANCE_LLM_MODEL", QWEN_MODEL)

THRESHOLDS = {
    "token_budget": int(os.environ.get("AFAC_TOKEN_BUDGET", "10000")),
}

DOMAINS = ["insurance", "regulatory", "financial_reports", "financial_contracts", "research"]

A_LIST_COMPANIES = {'中国人寿': ['中国人寿保险股份有限公司', '中国人寿', '国寿'],
 '中国建筑': ['中国建筑股份有限公司', '中国建筑', '601668'],
 '中国移动': ['中国移动有限公司', '中国移动', '600941'],
 '众安在线': ['众安在线财产保险股份有限公司', '众安保险', '众安在线', '众安'],
 '南京银行': ['南京银行股份有限公司', '南京银行'],
 '太平洋保险': ['中国太平洋人寿保险股份有限公司', '太平洋保险', '太保', '太平洋人寿'],
 '宁德时代': ['宁德时代新能源科技股份有限公司', '宁德时代', 'CATL', 'catl', '300750'],
 '平安人寿': ['中国平安人寿保险股份有限公司', '平安人寿', '平安'],
 '平安健康': ['平安健康保险股份有限公司', '平安健康'],
 '平安养老': ['平安养老保险股份有限公司', '平安养老'],
 '比亚迪': ['比亚迪股份有限公司', '比亚迪', 'BYD', 'byd'],
 '瑞丰高材': ['山东瑞丰高分子材料股份有限公司', '瑞丰高材'],
 '美的集团': ['美的集团股份有限公司', '美的集团', '000333'],
 '隆基绿能': ['隆基绿能科技股份有限公司', '隆基绿能', '隆基股份', '601012']}

A_LIST_INSURANCE_PRODUCTS = {'众安白血病医疗险': ['众安白血病医疗险', '白血病医疗险'],
 '国寿增益宝': ['国寿增益宝', '增益宝'],
 '国寿鑫享添盈': ['国寿鑫享添盈', '鑫享添盈'],
 '太保团体百万医疗': ['太保团体百万医疗', '太保百万医疗', '团体百万医疗'],
 '安享盈满金生养老年金保险': ['安享盈满金生', '盈满金生', '安享盈满'],
 '平安e生保': ['平安e生保', 'e生保'],
 '平安富鸿金生': ['平安富鸿金生', '富鸿金生'],
 '平安智盈金生': ['平安智盈金生', '智盈金生'],
 '祥宁终身寿险': ['祥宁终身寿险', '祥宁']}

A_LIST_BONDS = {'南银转债': ['南银转债'], '瑞丰转债': ['瑞丰转债'], '隆22转债': ['隆22转债']}

A_LIST_LAWS = {'上市公司信息披露管理办法': ['信披办法', '披露办法', '信息披露管理办法'],
 '上市公司收购管理办法': ['收购管理办法', '收购办法'],
 '上市公司章程指引': ['章程指引'],
 '上市公司股东、董监高减持股份的若干规定': ['减持规定', '减持若干规定'],
 '上市公司股权激励管理办法': ['股权激励管理办法', '股权激励办法'],
 '上市公司证券发行管理办法': ['证券发行管理办法'],
 '公司债券发行与交易管理办法': ['公司债管理办法', '债券管理办法'],
 '公司法': ['公司法', '中华人民共和国公司法'],
 '可转换公司债券管理办法': ['可转债管理办法'],
 '证券发行与承销管理办法': ['承销管理办法'],
 '证券法': ['证券法', '中华人民共和国证券法']}

