#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative field ontology shared by routing and evidence retrieval.

Only EXACT_FIELD_ALIASES may be treated as semantic equivalence.  Retrieval
aliases improve recall but never authorize a final verdict.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

EXACT_FIELD_ALIASES: Dict[str, List[str]] = {
    "营业收入": ["营业收入", "营业总收入", "营收"],
    "归母净利润": ["归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"],
    "经营现金流": ["经营现金流", "经营活动产生的现金流量净额", "经营活动现金流量净额"],
    "研发投入占营收比例": ["研发投入占营业收入比例", "研发投入占营业收入的比例", "研发投入占比"],
    "每股分红": ["每股分红", "每股派息", "每股现金红利", "每股股利"],
    "等待期": ["等待期", "观察期", "等待期间"],
    "犹豫期": ["犹豫期", "冷静期"],
    "免赔额": ["免赔额", "免赔金额", "年度免赔额"],
    "现金价值": ["现金价值", "保单现金价值", "退保现金价值"],
    "主体评级": ["主体评级", "主体信用评级", "发行人信用评级"],
    "债项评级": ["债项评级", "债券评级", "本期债券评级", "债券信用评级"],
    "发行金额": ["发行金额", "本期发行金额", "发行规模"],
    "注册金额": ["注册金额", "注册规模", "注册总额"],
    "主承销商": ["主承销商", "牵头主承销商", "联席主承销商"],
    "受托管理人": ["受托管理人", "债券受托管理人"],
    "保存期限": ["保存期限", "保管期限", "保存年限"],
    "提交期限": ["提交期限", "提交差异报告", "差异报告", "工作日内提交", "日内提交"],
    "施行日期": ["施行日期", "起施行", "施行之日起", "自发布之日起施行"],
    "市场规模": ["市场规模", "市场空间", "市场容量", "行业规模"],
    "复合增速": ["复合增速", "CAGR", "年复合增长率", "复合增长率"],
    "保费贡献率": ["保费贡献率", "银保渠道保费贡献", "银保渠道保费占比", "渠道保费占比", "银保渠道占比"],
    "营收同比增长": ["营收同比增长", "营业收入同比增长", "营收同比增速", "营收增速"],
}

RETRIEVAL_FIELD_ALIASES: Dict[str, List[str]] = {
    "净利润": ["净利润", "归母净利润", "扣非归母净利润", "利润总额"],
    "利润": ["净利润", "归母净利润", "扣非归母净利润", "利润总额"],
    "研发": ["研发投入", "研发费用", "研发支出", "研发占比"],
    "评级": ["主体评级", "债项评级", "信用评级", "评级展望"],
    "现金流": ["经营现金流", "投资活动现金流", "筹资活动现金流", "现金及现金等价物"],
    "分红": ["现金分红", "每股分红", "每10股分红", "股利支付率"],
    "保险责任": ["保险责任", "保障范围", "给付责任", "赔偿责任"],
    "责任免除": ["责任免除", "免责", "除外责任", "不保事项"],
}

NOT_EQUIVALENT_PAIRS: Set[Tuple[str, str]] = {
    tuple(sorted(pair)) for pair in [
        ("净利润", "归母净利润"),
        ("归母净利润", "扣非归母净利润"),
        ("主体评级", "债项评级"),
        ("发行金额", "注册金额"),
        ("研发费用", "研发投入"),
        ("营业收入", "主营业务收入"),
        ("增长率", "增长额"),
    ]
}

CLAUSE_TERMS = {
    "obligation": ["应当", "必须", "须", "有义务"],
    "prohibition": ["不得", "禁止", "严禁", "无权"],
    "permission": ["可以", "有权", "允许"],
    "exception": ["除外", "例外", "除非", "但书", "但是"],
    "consequence": ["责任", "处罚", "赔偿", "无效", "解除", "终止"],
    "time_limit": ["日内", "工作日内", "个月内", "期限", "届满"],
}


def aliases_for_field(field: str, *, adjudication: bool = False) -> List[str]:
    field = str(field or "").strip()
    if not field:
        return []
    if field in EXACT_FIELD_ALIASES:
        return list(dict.fromkeys(EXACT_FIELD_ALIASES[field]))
    for standard, aliases in EXACT_FIELD_ALIASES.items():
        if field == standard or field in aliases:
            return list(dict.fromkeys([standard] + aliases))
    if not adjudication:
        if field in RETRIEVAL_FIELD_ALIASES:
            return list(dict.fromkeys([field] + RETRIEVAL_FIELD_ALIASES[field]))
        for standard, aliases in RETRIEVAL_FIELD_ALIASES.items():
            if field == standard or field in aliases:
                return list(dict.fromkeys([standard] + aliases))
    return [field]


def fields_are_equivalent(left: str, right: str) -> bool:
    pair = tuple(sorted((str(left or ""), str(right or ""))))
    if pair in NOT_EQUIVALENT_PAIRS:
        return False
    return bool(set(aliases_for_field(left, adjudication=True)) & set(aliases_for_field(right, adjudication=True)))
