#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import shutil
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "marksix_local.db")
API_URL = "https://marksix6.net/index.php?api=1"
MINED_CONFIG_KEY = "mined_strategy_config_v1"
LAST_ML_TRAIN_KEY = "last_ml_train_issue"
ML_MODEL_KEY = "lightgbm_model"

ALL_NUMBERS = list(range(1, 50))

# -------------------- 策略标签（扩展后） --------------------
STRATEGY_LABELS = {
    "balanced_v1": "组合策略", "hot_v1": "热号策略", "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量", "ensemble_v2": "集成投票", "pattern_mined_v1": "规律挖掘",
    "ml_v1": "LightGBM机器学习"
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1", "ml_v1"]
SPECIAL_ANALYSIS_ORDER = ["pattern_mined_v1", "ensemble_v2", "momentum_v1", "cold_rebound_v1", "hot_v1", "balanced_v1", "ml_v1"]

# -------------------- 澳门优化常量 --------------------
FEATURE_WINDOW_DEFAULT = 10
STRATEGY_BASE_WINDOWS = {
    "hot_v1": 6, "momentum_v1": 7, "cold_rebound_v1": 13,
    "balanced_v1": 10, "pattern_mined_v1": 6, "ensemble_v2": 10, "ml_v1": 10
}
WEIGHT_WINDOW_DEFAULT = 30
HEALTH_WINDOW_DEFAULT = 18
BACKTEST_ISSUES_DEFAULT = 120
ENSEMBLE_DIVERSITY_BONUS = 0.18
BIAS_THRESHOLD = 0.65
BIAS_ADJUSTMENT = 0.40
FORCED_BIAS_COEFFICIENT = 0.75

# 生肖映射（1=马...）
ZODIAC_MAP = {
    "马": [1,13,25,37,49], "蛇": [2,14,26,38], "龙": [3,15,27,39], "兔": [4,16,28,40],
    "虎": [5,17,29,41], "牛": [6,18,30,42], "鼠": [7,19,31,43], "猪": [8,20,32,44],
    "狗": [9,21,33,45], "鸡": [10,22,34,46], "猴": [11,23,35,47], "羊": [12,24,36,48]
}

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "")
_WEIGHT_PROTECTION_PRINTED: set[str] = set()
_PROTECTION_PRINT_COUNTER = 0

@dataclass
class DrawRecord:
    issue_no: str
    draw_date: str
    numbers: List[int]
    special_number: int

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS draws (
            issue_no TEXT PRIMARY KEY, draw_date TEXT NOT NULL, numbers_json TEXT NOT NULL,
            special_number INTEGER NOT NULL, source TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, issue_no TEXT NOT NULL, strategy TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING', hit_count INTEGER, hit_rate REAL,
            hit_count_10 INTEGER, hit_rate_10 REAL, hit_count_14 INTEGER, hit_rate_14 REAL,
            hit_count_20 INTEGER, hit_rate_20 REAL, special_hit INTEGER,
            created_at TEXT NOT NULL, reviewed_at TEXT,
            UNIQUE(issue_no, strategy)
        );
        CREATE TABLE IF NOT EXISTS prediction_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, pick_type TEXT NOT NULL DEFAULT 'MAIN',
            number INTEGER NOT NULL, rank INTEGER NOT NULL, score REAL NOT NULL, reason TEXT NOT NULL,
            UNIQUE(run_id, number)
        );
        CREATE TABLE IF NOT EXISTS prediction_pools (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL, pool_size INTEGER NOT NULL,
            numbers_json TEXT NOT NULL, created_at TEXT NOT NULL, UNIQUE(run_id, pool_size)
        );
        CREATE TABLE IF NOT EXISTS model_state (
            key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            strategy TEXT NOT NULL,
            main_hit_count INTEGER NOT NULL,
            special_hit INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(issue_no, strategy)
        );
    """)
    cur = conn.execute("PRAGMA table_info(strategy_performance)")
    cols = [r[1] for r in cur.fetchall()]
    if "main_hit_count" not in cols:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN main_hit_count INTEGER DEFAULT 0")
    if "special_hit" not in cols:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN special_hit INTEGER DEFAULT 0")
    conn.commit()

def backup_database(db_path: str, max_backups: int = 5) -> str:
    db = Path(db_path)
    if not db.exists(): return ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db.with_name(f"{db.stem}_backup_{ts}{db.suffix}")
    shutil.copy2(db, backup)
    print(f"[backup] 已备份 → {backup.name}")
    for old in sorted(db.parent.glob(f"{db.stem}_backup_*{db.suffix}"), reverse=True)[max_backups:]:
        old.unlink()
    return str(backup)

def get_model_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM model_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None

def set_model_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = utc_now()
    conn.execute("INSERT INTO model_state(key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at", (key, value, now))

# -------------------- 香港数据获取（保持不变） --------------------
def _parse_date(d: str) -> Optional[str]:
    if not d: return None
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try: return datetime.strptime(d, f).strftime("%Y-%m-%d")
        except: pass
    return None

def _parse_numbers(v: str) -> List[int]:
    out = []
    for t in v.replace("，", ",").split(","):
        t = t.strip()
        if t.isdigit():
            n = int(t)
            if 1 <= n <= 49: out.append(n)
    return out

def _parse_marksix6_payload(payload: dict) -> List[DrawRecord]:
    records = []
    lottery_list = payload.get("lottery_data", [])
    hk_data = None
    for item in lottery_list:
        if isinstance(item, dict) and item.get("name") == "香港彩":
            hk_data = item
            break
    if not hk_data: return records
    history_list = hk_data.get("history", [])
    for line in history_list:
        match = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
        if not match: continue
        expect_raw = match.group(1)
        numbers_str = match.group(2)
        num_list = _parse_numbers(numbers_str)
        if len(num_list) < 7: continue
        main_numbers = num_list[:6]
        special = num_list[6]
        year = expect_raw[2:4]
        seq = str(int(expect_raw[4:]))
        issue_no = f"{year}/{seq.zfill(3)}"
        draw_date = _parse_date(hk_data.get("openTime", "").split()[0]) if hk_data.get("openTime") else "2026-01-01"
        records.append(DrawRecord(issue_no=issue_no, draw_date=draw_date, numbers=main_numbers, special_number=special))
    return records

def fetch_marksix6_records(retries: int = 3, timeout: int = 30) -> List[DrawRecord]:
    req = Request(API_URL, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    records = []
    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8-sig")
            payload = json.loads(raw)
            lottery_list = payload.get("lottery_data", [])
            for item in lottery_list:
                if item.get("name") == "香港彩":
                    print(f"[sync] 获取到 {len(item.get('history',[]))} 条香港记录")
                    records = _parse_marksix6_payload(payload)
                    break
            break
        except Exception as e:
            if attempt < retries:
                time.sleep(3)
                continue
            print(f"[sync] API获取失败: {e}")
    if len(records) < 30:
        print(f"[sync] 警告：当前只获取到 {len(records)} 条记录")
    return records

def upsert_draw(conn: sqlite3.Connection, record: DrawRecord, source: str) -> str:
    now = utc_now()
    exist = conn.execute("SELECT issue_no FROM draws WHERE issue_no=?", (record.issue_no,)).fetchone()
    if exist:
        conn.execute("UPDATE draws SET draw_date=?, numbers_json=?, special_number=?, source=?, updated_at=? WHERE issue_no=?", (record.draw_date, json.dumps(record.numbers), record.special_number, source, now, record.issue_no))
        return "updated"
    conn.execute("INSERT INTO draws(issue_no, draw_date, numbers_json, special_number, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?)", (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number, source, now, now))
    return "inserted"

def sync_from_records(conn: sqlite3.Connection, records: List[DrawRecord], source: str) -> Tuple[int, int, int]:
    ins, upd = 0, 0
    for r in records:
        res = upsert_draw(conn, r, source)
        if res == "inserted": ins += 1
        else: upd += 1
    conn.commit()
    return len(records), ins, upd

def load_recent_draws(conn: sqlite3.Connection, limit: int = 10) -> List[List[int]]:
    rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]

def _draws_ordered_asc(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date ASC, issue_no ASC").fetchall()

def next_issue(issue_no: str) -> str:
    p = issue_no.split("/")
    return f"{p[0]}/{int(p[1])+1:03d}"

# -------------------- 基础特征与策略（原有 + 增强） --------------------
def _normalize(m: Dict[int, float]) -> Dict[int, float]:
    vals = list(m.values())
    mn, mx = min(vals), max(vals)
    return {k: 0.0 for k in m} if mn == mx else {k: (v - mn) / (mx - mn) for k, v in m.items()}

def _freq_map(draws): freq = {n:0.0 for n in ALL_NUMBERS}; _ = [freq.update({n: freq[n]+1}) for d in draws for n in d]; return freq
def _omission_map(draws):
    om = {n: float(len(draws)+1) for n in ALL_NUMBERS}
    for i, d in enumerate(draws):
        for n in d: om[n] = min(om[n], float(i+1))
    return om
def _momentum_map(draws):
    m = {n:0.0 for n in ALL_NUMBERS}
    for i, d in enumerate(draws):
        w = 1.0/(1.0+i)
        for n in d: m[n] += w
    return m
def _pair_affinity_map(draws, w=6):
    cnt = {}
    for d in draws[:w]:
        s = sorted(d)
        for i in range(len(s)):
            for j in range(i+1, len(s)): cnt[(s[i], s[j])] = cnt.get((s[i], s[j]), 0) + 1
    social = {n:0.0 for n in ALL_NUMBERS}
    for (a,b), c in cnt.items(): social[a] += c; social[b] += c
    return social
def _zone_heat_map(draws, w=6):
    zc = [0.0]*5
    wd = draws[:w]
    if not wd: return {n:0.0 for n in ALL_NUMBERS}
    for d in wd:
        for n in d: zc[min(4, (n-1)//10)] += 1.0
    exp = 6.0*len(wd)/5.0
    zs = [exp - c for c in zc]
    return {n: zs[min(4, (n-1)//10)] for n in ALL_NUMBERS}

def _pick_top_six(scores: Dict[int, float], reason: str) -> List[Tuple[int, int, float, str]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = []
    for n, s in ranked:
        if len(picked) == 6: break
        prop = [pn for pn, _ in picked] + [n]
        odd = sum(1 for x in prop if x%2==1)
        if len(prop)>=4 and (odd==0 or odd==len(prop)): continue
        zc = {}
        for x in prop: z = min(4, (x-1)//10); zc[z] = zc.get(z,0)+1
        if any(c>=4 for c in zc.values()): continue
        picked.append((n,s))
    while len(picked)<6:
        for n,s in ranked:
            if n not in [pn for pn,_ in picked]:
                picked.append((n,s)); break
    top6 = [n for n,_ in picked[:6]]
    total = sum(top6)
    if not (95 <= total <= 205):
        for i in range(5,-1,-1):
            replaced = False
            for alt_n, alt_s in ranked:
                if alt_n in top6: continue
                cand = list(top6); cand[i] = alt_n
                if 95 <= sum(cand) <= 205:
                    picked[i] = (alt_n, alt_s); top6 = cand; replaced = True; break
            if replaced: break
    return [(n, i+1, s, f"{reason} score={s:.4f}") for i, (n, s) in enumerate(picked)]

def _default_mined_config():
    return {"window":6.0, "w_freq":0.30, "w_omit":0.50, "w_mom":0.20, "w_pair":0.00, "w_zone":0.10, "special_bonus":0.10}

def _apply_weight_config(draws, config, reason):
    win = int(config.get("window", 6))
    wd = draws[:max(6, win)]
    freq = _normalize(_freq_map(wd)); omission = _normalize(_omission_map(wd)); momentum = _normalize(_momentum_map(wd))
    pair = _normalize(_pair_affinity_map(wd, min(6, len(wd)))); zone = _normalize(_zone_heat_map(wd, min(6, len(wd))))
    wf, wo, wm, wp, wz = config.get("w_freq",0.45), config.get("w_omit",0.35), config.get("w_mom",0.20), config.get("w_pair",0.0), config.get("w_zone",0.0)
    scores = {n: freq[n]*wf + omission[n]*wo + momentum[n]*wm + pair[n]*wp + zone[n]*wz for n in ALL_NUMBERS}
    main_picks = _pick_top_six(scores, reason)
    main_set = {n for n,_,_,_ in main_picks}
    cand = [(n,s) for n,s in sorted(scores.items(), key=lambda x:x[1], reverse=True) if n not in main_set]
    if not cand: cand = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    sp, sp_score = cand[0]
    return main_picks, sp, sp_score, scores

# -------------------- 偏态检测（强制0.75） --------------------
def detect_bias(conn, window=10):
    return 0.75, {"forced":True, "zone_bias":0.75, "parity_bias":0.70, "hot_cold_bias":0.70, "zone_dist":[0]*5, "odd_ratio":0.5}

def adjust_weights_for_bias(weights, bias_score):
    if bias_score < BIAS_THRESHOLD: return weights
    adj = weights.copy()
    cold_boost = 1 + BIAS_ADJUSTMENT * bias_score
    adj["cold_rebound_v1"] = weights.get("cold_rebound_v1",0.15)*cold_boost
    adj["hot_v1"] = weights.get("hot_v1",0.15)*(1 - BIAS_ADJUSTMENT*bias_score*0.7)
    adj["momentum_v1"] = weights.get("momentum_v1",0.15)*(1 - BIAS_ADJUSTMENT*bias_score*0.5)
    total = sum(adj.values())
    return {k: v/total for k,v in adj.items()} if total>0 else adj

# -------------------- 生肖辅助函数（澳门优化） --------------------
def get_zodiac_by_number(n):
    for z, nums in ZODIAC_MAP.items():
        if n in nums: return z
    return "马"

def _zodiac_omission_map(rows):
    om = {z: len(rows)+1 for z in ZODIAC_MAP}
    for i, row in enumerate(rows):
        nums = json.loads(row["numbers_json"])
        sp = row["special_number"]
        appeared = set(get_zodiac_by_number(n) for n in nums)
        appeared.add(get_zodiac_by_number(sp))
        for z in appeared:
            if om[z] > i+1: om[z] = i+1
    return om

def _build_zodiac_scores_from_rows(rows, decay=0.08):
    scores = {z:0.0 for z in ZODIAC_MAP}
    om = _zodiac_omission_map(rows)
    for i, row in enumerate(rows):
        w = 1.0/(1.0 + i*decay)
        nums = json.loads(row["numbers_json"])
        for n in nums: scores[get_zodiac_by_number(n)] += 1.0*w
        scores[get_zodiac_by_number(row["special_number"])] += 1.8*w
    for z in scores:
        omit = om.get(z, len(rows))
        if omit >= 6: scores[z] += min(3.0, omit/4.0)
        elif omit >= 3: scores[z] += omit/6.0
    return scores

def _get_previous_issue(conn, cur):
    row = conn.execute("SELECT issue_no FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no=?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no=?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 1", (cur,cur,cur)).fetchone()
    return row["issue_no"] if row else None

def _check_two_zodiac_hit(conn, issue):
    draw = conn.execute("SELECT numbers_json, special_number FROM draws WHERE issue_no=?", (issue,)).fetchone()
    if not draw: return False
    win_zod = {get_zodiac_by_number(n) for n in json.loads(draw["numbers_json"])}
    win_zod.add(get_zodiac_by_number(draw["special_number"]))
    rows = conn.execute("SELECT numbers_json, special_number FROM draws WHERE draw_date < (SELECT draw_date FROM draws WHERE issue_no=?) OR (draw_date = (SELECT draw_date FROM draws WHERE issue_no=?) AND issue_no < ?) ORDER BY draw_date DESC, issue_no DESC LIMIT 16", (issue,issue,issue)).fetchall()
    if not rows: return False
    scores = _build_zodiac_scores_from_rows(rows, 0.08)
    ranked = sorted(scores.items(), key=lambda x:-x[1])
    picks = [ranked[0][0], ranked[1][0]] if len(ranked)>=2 else ["马","蛇"]
    return any(z in win_zod for z in picks)

def get_two_zodiac_picks(conn, issue_no, window=16):
    rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (window,)).fetchall()
    if not rows: return ["马","蛇"]
    scores = _build_zodiac_scores_from_rows(rows, 0.08)
    om = _zodiac_omission_map(rows)
    force_include = [z for z, omit in om.items() if omit >= 8]
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if pool20:
        for z, cnt in Counter(get_zodiac_by_number(n) for n in pool20).items():
            scores[z] += cnt*0.6
    top_sp = get_top_special_votes(conn, issue_no, 3)
    for sp in top_sp: scores[get_zodiac_by_number(sp)] += 1.5
    prev_issue = _get_previous_issue(conn, issue_no)
    prev_hit = _check_two_zodiac_hit(conn, prev_issue) if prev_issue else False
    if not prev_hit and prev_issue:
        prev_draw = conn.execute("SELECT numbers_json, special_number FROM draws WHERE issue_no=?", (prev_issue,)).fetchone()
        if prev_draw:
            prev_zod = [get_zodiac_by_number(n) for n in json.loads(prev_draw["numbers_json"])]
            prev_zod.append(get_zodiac_by_number(prev_draw["special_number"]))
            hot_two = [z for z,_ in Counter(prev_zod).most_common(2)]
            if len(hot_two)>=2: return hot_two[:2]
    ranked = sorted(scores.items(), key=lambda x:-x[1])
    picks = []
    for z in force_include:
        if z not in picks: picks.append(z)
    for z,_ in ranked:
        if len(picks)>=2: break
        if z not in picks: picks.append(z)
    return picks[:2]

def get_single_zodiac_pick(conn, issue_no, window=14):
    two = get_two_zodiac_picks(conn, issue_no, window)
    rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (window,)).fetchall()
    if not rows: return two[0] if two else "马"
    scores = _build_zodiac_scores_from_rows(rows, 0.05)
    om = _zodiac_omission_map(rows)
    for z in scores:
        omit = om.get(z, len(rows))
        scores[z] += min(5.0, omit*0.8)
    coldest = max(om.keys(), key=lambda z: om[z])
    scores[coldest] += 5.0
    _, _, _, pool20, _ = _weighted_consensus_pools(conn, issue_no)
    if pool20:
        for z, cnt in Counter(get_zodiac_by_number(n) for n in pool20).items():
            scores[z] += cnt*0.8
    top_sp = get_top_special_votes(conn, issue_no, 3)
    for sp in top_sp: scores[get_zodiac_by_number(sp)] += 2.5
    recent_sp_zod = [get_zodiac_by_number(int(r["special_number"])) for r in rows[:3]]
    for z in recent_sp_zod: scores[z] -= 0.1
    for z in two: scores[z] += 4.0
    ranked = sorted(scores.items(), key=lambda x:-x[1])
    for cand,_ in ranked:
        if cand in two: return cand
    return ranked[0][0]

# -------------------- 特别号 v4（含生肖缺失补偿） --------------------
def _generate_special_number_v4(conn, main_pool, issue_no):
    special_votes = []
    for s in STRATEGY_IDS:
        run = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'", (issue_no, s)).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp: special_votes.append(sp)
    vote_cnt = Counter(special_votes)
    recent_sps = [int(r["special_number"]) for r in conn.execute("SELECT special_number FROM draws ORDER BY draw_date DESC LIMIT 60").fetchall()]
    omission = {n:60 for n in ALL_NUMBERS}
    for i, num in enumerate(recent_sps): omission[num] = min(omission.get(num,60), i+1)
    zodiac_cycle = [get_zodiac_by_number(sp) for sp in recent_sps[:24]]
    if len(zodiac_cycle)>=6:
        least = min(Counter(zodiac_cycle[:6]).items(), key=lambda x:x[1])[0]
        pred_zodiac_nums = ZODIAC_MAP.get(least, [1,13,25,37,49])
    else: pred_zodiac_nums = ALL_NUMBERS
    tail_cnt = Counter(n%10 for n in recent_sps[:20])
    coldest_tail = min(tail_cnt.items(), key=lambda x:x[1])[0] if tail_cnt else 0
    main_zones = {(m-1)//10 for m in main_pool}
    main_set = set(main_pool)
    recent_main_zodiacs = []
    for row in conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT 2").fetchall():
        for n in json.loads(row["numbers_json"]): recent_main_zodiacs.append(get_zodiac_by_number(n))
    missing_zodiacs = set(ZODIAC_MAP.keys()) - set(recent_main_zodiacs[-12:]) if len(recent_main_zodiacs)>=12 else set()
    scores = {}
    for n in ALL_NUMBERS:
        if n in main_set: continue
        sc = 0.0
        sc += vote_cnt.get(n,0)*6.0
        om_score = (60 - omission.get(n,60))/60.0
        sc += om_score*(4.0 if omission.get(n,60)>12 else 1.0)
        if n in recent_sps[:2]: sc *= 0.2
        elif n in recent_sps[2:5]: sc *= 0.6
        if n in pred_zodiac_nums: sc += 1.8
        if n%10 == coldest_tail: sc += 1.2
        for mn in main_pool:
            diff = abs(n-mn)
            if diff==1: sc += 2.0
            elif diff==2: sc += 1.5
            elif diff==3: sc += 1.0
            elif n%10 == mn%10 and n!=mn: sc += 0.8
        if (n-1)//10 in main_zones: sc += 1.2
        recent_parity = [sp%2 for sp in recent_sps[:5]]
        if len(recent_parity)>=3:
            odd_ratio = sum(recent_parity)/len(recent_parity)
            if odd_ratio>0.6 and n%2==0: sc += 1.0
            elif odd_ratio<0.4 and n%2==1: sc += 1.0
        if missing_zodiacs and get_zodiac_by_number(n) in missing_zodiacs: sc += 3.0
        scores[n] = sc
    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    best = ranked[0][0]
    conf = min(1.0, ranked[0][1]/18)
    defenses = [n for n,_ in ranked[1:] if n not in main_set and n!=best][:3]
    return best, round(conf,3), defenses[:3]

# -------------------- 集成策略 v3.1 --------------------
def _ensemble_strategy_v3_1(draws, mined_config, strategy_weights, conn, issue_no):
    subs = ["hot_v1","cold_rebound_v1","momentum_v1","balanced_v1","pattern_mined_v1"]
    score_maps, sub_picks = [], {}
    bias_score, _ = detect_bias(conn, 10)
    adj_weights = adjust_weights_for_bias(strategy_weights, bias_score)
    print(f"[集成策略] 🔥 偏态系数={bias_score:.2f} 冷号权重={adj_weights.get('cold_rebound_v1',0):.3f}", flush=True)
    for sub in subs:
        win_size = STRATEGY_BASE_WINDOWS.get(sub, 10)
        sub_draws = draws[:win_size] if len(draws)>win_size else draws
        if sub == "pattern_mined_v1":
            cfg = mined_config or _default_mined_config()
            cfg["window"] = float(win_size)
            _, _, _, smap = _apply_weight_config(sub_draws, cfg, "规律挖掘")
        else:
            cfg = {"window": float(win_size)}
            if sub == "hot_v1": cfg.update({"w_freq":0.78, "w_omit":0.05, "w_mom":0.17})
            elif sub == "cold_rebound_v1": cfg.update({"w_freq":0.05, "w_omit":0.68, "w_mom":0.27})
            elif sub == "momentum_v1": cfg.update({"w_freq":0.12, "w_omit":0.05, "w_mom":0.83})
            else: cfg.update({"w_freq":0.40, "w_omit":0.30, "w_mom":0.20})
            _, _, _, smap = _apply_weight_config(sub_draws, cfg, STRATEGY_LABELS.get(sub,sub))
        score_maps.append(smap)
        sub_picks[sub] = [n for n,_ in sorted(smap.items(), key=lambda x:x[1], reverse=True)[:6]]
    votes = {n:0.0 for n in ALL_NUMBERS}
    for idx, sub in enumerate(subs):
        w = adj_weights.get(sub, 0.2)
        for rank, (n,_) in enumerate(sorted(score_maps[idx].items(), key=lambda x:x[1], reverse=True)):
            votes[n] += w * (49 - rank)
    cold_picks = sub_picks.get("cold_rebound_v1", [])
    for i, n in enumerate(cold_picks): votes[n] += 0.8 * (6 - i)
    for n in ALL_NUMBERS:
        appear = sum(1 for p in sub_picks.values() if n in p)
        votes[n] += (6 - appear) * ENSEMBLE_DIVERSITY_BONUS * 1.2
    voted = _normalize(votes)
    main_picked = _pick_top_six(voted, "集成投票v3.1")
    main6 = [n for n,_,_,_ in main_picked]
    sp, conf, _ = _generate_special_number_v4(conn, main6, issue_no)
    return main_picked, sp, conf, voted

# -------------------- ML 模型（保持原香港逻辑） --------------------
def extract_features_for_number(draws, target):
    recent = draws[:12]
    feats = [1 if any(target in d for d in recent[:lag]) else 0 for lag in [1,2,3,5,8]]
    all_rec = [n for d in recent for n in d]
    freq = all_rec.count(target)/max(len(all_rec),1)
    omission = next((i+1 for i,d in enumerate(recent) if target in d), len(recent)+1)
    feats.extend([freq, omission, 1.0/(omission+1), sum(1 for d in recent if target in d)])
    feats.append(next((i for i,d in enumerate(recent) if target in d), -1))
    feats.extend([target%2, 1 if target<=24 else 0, target//10, target%10])
    return np.array(feats, dtype=np.float32)

def train_ml_model(conn):
    print("[ML] 训练 LightGBM...")
    draws = []
    for row in conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date ASC, issue_no ASC").fetchall():
        draws.append(json.loads(row["numbers_json"]))
    if len(draws) < 50: return None
    X, y = [], []
    for i in range(20, len(draws)-1):
        hist = draws[i-20:i]
        for num in ALL_NUMBERS:
            X.append(extract_features_for_number(hist, num))
            y.append(1 if num in draws[i] else 0)
    X, y = np.array(X), np.array(y)
    if len(np.unique(y)) < 2: return None
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {'objective':'binary','metric':'auc','boosting_type':'gbdt','num_leaves':31,'learning_rate':0.05,
              'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,'random_state':42}
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200,
                      callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    print(f"[ML] AUC: {model.best_score['valid_0']['auc']:.4f}")
    set_model_state(conn, ML_MODEL_KEY, pickle.dumps(model).hex())
    return model

def load_ml_model(conn):
    hex_str = get_model_state(conn, ML_MODEL_KEY)
    return pickle.loads(bytes.fromhex(hex_str)) if hex_str else None

def ml_strategy(draws, model):
    if model is None:
        return _apply_weight_config(draws, {"window":6.0, "w_freq":0.55, "w_omit":0.25, "w_mom":0.20}, "ML回退")
    X = np.array([extract_features_for_number(draws, n) for n in ALL_NUMBERS])
    probs = model.predict(X)
    scores = {n: float(probs[i]) for i, n in enumerate(ALL_NUMBERS)}
    return _apply_weight_config(draws, {"window":6.0, "w_freq":1.0, "w_omit":0.0, "w_mom":0.0}, "LightGBM")

# -------------------- 策略调度（增强版） --------------------
def generate_strategy(draws, strategy, mined_config=None, strategy_weights=None, conn=None, issue_no=None):
    if strategy == "hot_v1": return _apply_weight_config(draws, {"window":6.0, "w_freq":0.78, "w_omit":0.05, "w_mom":0.17}, "热号策略")
    if strategy == "cold_rebound_v1": return _apply_weight_config(draws, {"window":6.0, "w_freq":0.05, "w_omit":0.68, "w_mom":0.27}, "冷号回补")
    if strategy == "momentum_v1": return _apply_weight_config(draws, {"window":6.0, "w_freq":0.12, "w_omit":0.05, "w_mom":0.83}, "近期动量")
    if strategy == "balanced_v1": return _apply_weight_config(draws, {"window":6.0, "w_freq":0.40, "w_omit":0.30, "w_mom":0.20, "w_pair":0.05, "w_zone":0.05}, "组合策略")
    if strategy == "pattern_mined_v1":
        cfg = mined_config or _default_mined_config()
        return _apply_weight_config(draws, cfg, "规律挖掘")
    if strategy == "ensemble_v2":
        if strategy_weights is None: strategy_weights = {s:1.0/len(STRATEGY_IDS) for s in STRATEGY_IDS}
        if conn is None or issue_no is None: raise ValueError("集成投票需要数据库连接和期号")
        return _ensemble_strategy_v3_1(draws, mined_config, strategy_weights, conn, issue_no)
    if strategy == "ml_v1":
        model = load_ml_model(conn) if conn else None
        return ml_strategy(draws, model)
    return _apply_weight_config(draws, {"window":6.0, "w_freq":0.40, "w_omit":0.30, "w_mom":0.20}, "组合策略")

# -------------------- 动态权重与健康度 --------------------
def get_strategy_weights(conn, window=WEIGHT_WINDOW_DEFAULT):
    rows = conn.execute("""
        SELECT strategy, AVG(main_hit_count) as avg_hit
        FROM strategy_performance
        WHERE issue_no IN (SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?)
        GROUP BY strategy
    """, (window,)).fetchall()
    baseline = 0.6
    weights = {s: baseline for s in STRATEGY_IDS}
    prot_msgs = []
    for r in rows:
        s = r["strategy"]
        if s in weights: weights[s] = max(float(r["avg_hit"] or 0.0), baseline)
    health = get_strategy_health(conn, HEALTH_WINDOW_DEFAULT)
    for s, h in health.items():
        if s not in weights: continue
        recent = h.get("recent_avg_hit", 0.0)
        hit1 = h.get("hit1_rate", 0.0)
        cold = h.get("cold_streak", 0)
        shrink = 1.0
        if recent < 0.7: shrink *= 0.90 ** ((0.7 - recent) * 8)
        if hit1 < 0.52: shrink *= 0.87
        if cold >= 3: shrink *= 0.72
        if s == "pattern_mined_v1" and (cold >= 2 or recent < 0.6):
            shrink *= 0.48
            prot_msgs.append(f"[保护] 规律挖掘连挂 {cold} 期，权重大幅下调")
        weights[s] = max(0.08, weights[s] * shrink)
    total = sum(weights.values())
    global _PROTECTION_PRINT_COUNTER
    for msg in prot_msgs:
        if msg not in _WEIGHT_PROTECTION_PRINTED:
            print(msg, flush=True)
            _WEIGHT_PROTECTION_PRINTED.add(msg)
    if prot_msgs:
        _PROTECTION_PRINT_COUNTER += 1
        if _PROTECTION_PRINT_COUNTER % 20 == 0:
            print(f"[保护] 当前规律挖掘/冷号回补仍处于权重保护中 (已持续{_PROTECTION_PRINT_COUNTER}期)", flush=True)
    return {k: round(v/total, 4) for k,v in weights.items()}

def get_strategy_health(conn, window=HEALTH_WINDOW_DEFAULT):
    health = {}
    for s in STRATEGY_IDS:
        rows = conn.execute("SELECT hit_count FROM prediction_runs WHERE strategy=? AND status='REVIEWED' ORDER BY reviewed_at DESC LIMIT ?", (s, window)).fetchall()
        if not rows:
            health[s] = {"samples":0.0, "recent_avg_hit":0.0, "hit1_rate":0.0, "hit2_rate":0.0, "cold_streak":0.0}
            continue
        hits = [int(r["hit_count"] or 0) for r in rows]
        samples = len(hits)
        hit1 = sum(1 for x in hits if x>=1)/samples
        hit2 = sum(1 for x in hits if x>=2)/samples
        avg = sum(hits)/samples
        cold = 0
        for x in hits:
            if x==0: cold+=1
            else: break
        health[s] = {"samples":float(samples), "recent_avg_hit":avg, "hit1_rate":hit1, "hit2_rate":hit2, "cold_streak":float(cold)}
    return health

def get_top_special_votes(conn, issue_no, top_n=3):
    all_sp = []
    for s in STRATEGY_IDS:
        run = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'", (issue_no, s)).fetchone()
        if run:
            _, sp = get_picks_for_run(conn, run["id"])
            if sp: all_sp.append(sp)
    return [n for n,_ in Counter(all_sp).most_common(top_n)] if all_sp else []

def _weighted_consensus_pools(conn, issue_no):
    weights = get_strategy_weights(conn, WEIGHT_WINDOW_DEFAULT)
    num_scores, sp_scores = {}, {}
    for s in STRATEGY_IDS:
        run = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'", (issue_no, s)).fetchone()
        if not run: continue
        w = weights.get(s, 1.0/len(STRATEGY_IDS))
        p20 = get_pool_numbers_for_run(conn, run["id"], 20)
        for i, n in enumerate(p20):
            if 1<=n<=49: num_scores[n] = num_scores.get(n,0) + w * ((20-i)/20.0)
        main6 = get_pool_numbers_for_run(conn, run["id"], 6)
        for n in main6:
            if 1<=n<=49: num_scores[n] = num_scores.get(n,0) + w*0.35
        _, sp = get_picks_for_run(conn, run["id"])
        if sp and 1<=sp<=49: sp_scores[sp] = sp_scores.get(sp,0) + w
    if not num_scores: return [], [], [], [], None
    ranked = [n for n,_ in sorted(num_scores.items(), key=lambda x:-x[1])]
    p20 = ranked[:20]
    p14, p10, p6 = p20[:14], p20[:10], p20[:6]
    sp = None
    if sp_scores: sp = sorted(sp_scores.items(), key=lambda x:-x[1])[0][0]
    else:
        for n in p20:
            if n not in p6: sp = n; break
    return p6, p10, p14, p20, sp

def get_trio_from_merged_pool20(conn, issue_no):
    _, _, _, p20, _ = _weighted_consensus_pools(conn, issue_no)
    if not p20 or len(p20)<3: return [1,2,3]
    all_pools = []
    for s in STRATEGY_IDS:
        run = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'", (issue_no, s)).fetchone()
        if run:
            p = get_pool_numbers_for_run(conn, run["id"], 20)
            all_pools.extend([n for n in p if n in p20])
    cnt = Counter(all_pools)
    cand = [n for n,c in cnt.items() if 1<=c<=2 and n in p20]
    if len(cand)<6: cand = [n for n,c in cnt.items() if c<=3 and n in p20]
    if len(cand)<3: cand = p20[:15]
    draws = load_recent_draws(conn, 10)
    if len(draws)<3: return cand[:3]
    mom = _normalize(_momentum_map(draws)); freq = _normalize(_freq_map(draws)); om = _normalize(_omission_map(draws))
    w_mom, w_hot, w_cold = 0.4, 0.35, 0.25
    scores = {n: w_mom*mom.get(n,0) + w_hot*freq.get(n,0) + w_cold*om.get(n,0) + (6-cnt.get(n,3))*0.15 for n in cand[:15]}
    sorted_nums = sorted(scores.items(), key=lambda x:-x[1])
    top10 = [n for n,_ in sorted_nums[:10]]
    def is_valid(trio): return 1 <= sum(1 for x in trio if x%2==1) <= 2 and 80 <= sum(trio) <= 130
    for i in range(len(top10)):
        for j in range(i+1, len(top10)):
            for k in range(j+1, len(top10)):
                t = (top10[i], top10[j], top10[k])
                if is_valid(t): return list(t)
    for i in range(len(top10)):
        for j in range(i+1, len(top10)):
            for k in range(j+1, len(top10)):
                t = (top10[i], top10[j], top10[k])
                if 1 <= sum(1 for x in t if x%2==1) <= 2: return list(t)
    return top10[:3] if len(top10)>=3 else p20[:3]

def get_pool_numbers_for_run(conn, run_id, pool_size=6):
    row = conn.execute("SELECT numbers_json FROM prediction_pools WHERE run_id=? AND pool_size=?", (run_id, pool_size)).fetchone()
    return json.loads(row["numbers_json"]) if row else []

def get_picks_for_run(conn, run_id):
    rows = conn.execute("SELECT pick_type, number FROM prediction_picks WHERE run_id=? ORDER BY rank ASC", (run_id,)).fetchall()
    mains = [r["number"] for r in rows if r["pick_type"] in (None,"MAIN")]
    sps = [r["number"] for r in rows if r["pick_type"]=="SPECIAL"]
    return mains, sps[0] if sps else None

# -------------------- 智能动态最终推荐（包含生肖） --------------------
_HAS_WARNED_DATA = False
def get_dynamic_final_recommendation(conn):
    global _HAS_WARNED_DATA
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row: return None
    cur_issue = row["issue_no"]
    next_iss = next_issue(cur_issue)
    rows = conn.execute("""
        SELECT strategy, AVG(hit_rate) as avg_rate
        FROM prediction_runs
        WHERE status='REVIEWED' AND issue_no IN (SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT 6)
        GROUP BY strategy
        ORDER BY avg_rate DESC LIMIT 3
    """).fetchall()
    if len(rows) >= 2:
        top_strats = [r["strategy"] for r in rows]
    else:
        if not _HAS_WARNED_DATA:
            print("[Final Rec] 历史数据不足，使用默认强策略组合")
            _HAS_WARNED_DATA = True
        top_strats = ["ensemble_v2", "ml_v1", "hot_v1"]
    main_pools, sp_list, weights_list = [], [], []
    for s in top_strats:
        run = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=? AND status='PENDING'", (next_iss, s)).fetchone()
        if not run: continue
        p6 = get_pool_numbers_for_run(conn, run["id"], 6)
        _, sp = get_picks_for_run(conn, run["id"])
        if p6 and len(p6)==6:
            main_pools.append(p6)
            if sp: sp_list.append(sp)
            weights_list.append(1.0)
    if len(main_pools) < 2:
        p6, p10, p14, p20, sp = _weighted_consensus_pools(conn, next_iss)
        trio = get_trio_from_merged_pool20(conn, next_iss)
        zodiac_two = get_two_zodiac_picks(conn, next_iss)
        zodiac_one = get_single_zodiac_pick(conn, next_iss)
        return (next_iss, p6, sp, p10, p14, p20, trio, 75, zodiac_one, zodiac_two)
    num_votes = Counter()
    total_w = sum(weights_list)
    for pool, w in zip(main_pools, weights_list):
        for n in pool: num_votes[n] += w/total_w
    final6 = [n for n,_ in num_votes.most_common(6)]
    sp_votes = Counter()
    for sp, w in zip(sp_list, weights_list): sp_votes[sp] += w
    final_sp = sp_votes.most_common(1)[0][0] if sp_votes else sp_list[0]
    all_nums = set()
    for p in main_pools: all_nums.update(p)
    sorted_all = sorted(all_nums, key=lambda x: num_votes.get(x,0), reverse=True)
    p10, p14, p20 = sorted_all[:10], sorted_all[:14], sorted_all[:20]
    trio = get_trio_from_merged_pool20(conn, next_iss)
    conf = min(98, max(60, int( sum(r["avg_rate"] for r in rows if r) / len(rows) * 135 )))
    zodiac_two = get_two_zodiac_picks(conn, next_iss)
    zodiac_one = get_single_zodiac_pick(conn, next_iss)
    return (next_iss, final6, final_sp, p10, p14, p20, trio, conf, zodiac_one, zodiac_two)

def print_final_recommendation(conn):
    rec = get_dynamic_final_recommendation(conn)
    if not rec:
        print("最终推荐: (暂无有效预测)")
        return
    iss, m6, sp, p10, p14, p20, trio, conf, z1, z2 = rec
    print("\n" + "="*70)
    print(f"【🔥 智能最终推荐 - {iss}期】")
    print(f"  6号池 : {' '.join(f'{n:02d}' for n in m6)} | 特别号: {sp:02d}")
    print(f"  10号池: {' '.join(f'{n:02d}' for n in p10)}")
    print(f"  14号池: {' '.join(f'{n:02d}' for n in p14)}")
    print(f"  20号池: {' '.join(f'{n:02d}' for n in p20)}")
    print(f"  三中三: {' '.join(f'{n:02d}' for n in trio)}")
    print(f"  🎯 1生肖: {z1}  2生肖: {'、'.join(z2)}")
    print(f"  置信度: {conf}/100")
    print("="*70)

def get_hot_cold_zodiacs(conn, window=3, top_n=3):
    rows = conn.execute("SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (window,)).fetchall()
    if len(rows) < window:
        default = ["马","蛇","龙","兔","虎","牛"]
        return default[:top_n], default[-top_n:]
    counter = Counter()
    for row in rows:
        nums = json.loads(row["numbers_json"])
        for n in nums: counter[get_zodiac_by_number(n)] += 1
        counter[get_zodiac_by_number(row["special_number"])] += 1
    sorted_freq = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    hot = [z for z,_ in sorted_freq[:top_n]]
    cold = [z for z,_ in sorted(counter.items(), key=lambda x:x[1])[:top_n]]
    return hot, cold

def get_latest_draw(conn):
    return conn.execute("SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()

def get_review_stats(conn):
    return conn.execute("""
        SELECT strategy, COUNT(*) c, AVG(hit_count) avg_hit, AVG(hit_rate) avg_rate
        FROM prediction_runs WHERE status='REVIEWED'
        GROUP BY strategy ORDER BY avg_rate DESC
    """).fetchall()

def print_dashboard(conn):
    latest = get_latest_draw(conn)
    if latest:
        nums = " ".join(f"{n:02d}" for n in json.loads(latest["numbers_json"]))
        print(f"最新开奖: {latest['issue_no']} {latest['draw_date']} | 主号: {nums} | 特别号: {latest['special_number']:02d}")
    hot, cold = get_hot_cold_zodiacs(conn, window=3, top_n=3)
    print(f"最近3期热门生肖: {', '.join(hot)}   |  冷门生肖: {', '.join(cold)}")
    print_final_recommendation(conn)
    print("\n📊 各策略历史表现（已复盘）：")
    stats = get_review_stats(conn)
    if stats:
        for s in stats[:7]:
            name = STRATEGY_LABELS.get(s["strategy"], s["strategy"])
            print(f"  {name:12s} : 次数={s['c']:3d}  平均命中={float(s['avg_hit']):.2f}  命中率={float(s['avg_rate'])*100:5.2f}%")
    else:
        print("  暂无已复盘数据，请先运行：python marksix_local.py fullbacktest")

def send_pushplus_notification(title, content):
    if not PUSHPLUS_TOKEN: return False
    import urllib.request, urllib.parse
    url = "https://www.pushplus.plus/send"
    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "txt"}
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode()).get("code") == 200
    except: return False

# -------------------- 主流程函数 --------------------
def generate_predictions(conn, issue_no=None):
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row: raise RuntimeError("没有开奖数据，请先 bootstrap")
    target = issue_no or next_issue(row["issue_no"])
    draws = load_recent_draws(conn, 20)
    if len(draws) < 6: raise RuntimeError("至少需要6期历史数据")
    mined_cfg = _default_mined_config()
    weights = get_strategy_weights(conn, WEIGHT_WINDOW_DEFAULT)
    last_train = get_model_state(conn, LAST_ML_TRAIN_KEY)
    if not last_train or (target > last_train):
        train_ml_model(conn)
        set_model_state(conn, LAST_ML_TRAIN_KEY, target)
    for s in STRATEGY_IDS:
        now = utc_now()
        exist = conn.execute("SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=?", (target, s)).fetchone()
        if exist:
            run_id = exist["id"]
            conn.execute("UPDATE prediction_runs SET status='PENDING', hit_count=NULL, hit_rate=NULL, hit_count_10=NULL, hit_rate_10=NULL, hit_count_14=NULL, hit_rate_14=NULL, hit_count_20=NULL, hit_rate_20=NULL, special_hit=NULL, reviewed_at=NULL, created_at=? WHERE id=?", (now, run_id))
            conn.execute("DELETE FROM prediction_picks WHERE run_id=?", (run_id,))
        else:
            cur = conn.execute("INSERT INTO prediction_runs(issue_no, strategy, status, created_at) VALUES (?,?,'PENDING',?)", (target, s, now))
            run_id = cur.lastrowid
        picks, sp, sp_score, score_map = generate_strategy(draws, s, mined_cfg, weights, conn, target)
        main_nums = [n for n,_,_,_ in picks]
        conn.executemany("INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason) VALUES (?,'MAIN',?,?,?,?)", [(run_id, n, r, sc, re) for n,r,sc,re in picks])
        conn.execute("INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason) VALUES (?,'SPECIAL',?,1,?,?)", (run_id, sp, sp_score, "特别号候选"))
        ranked = [n for n,_ in sorted(score_map.items(), key=lambda x:x[1], reverse=True)]
        main_uniq = []
        for n in main_nums:
            if n not in main_uniq: main_uniq.append(n)
        rest = [n for n in ranked if n not in main_uniq]
        pools = {6: main_uniq[:6], 10: main_uniq + rest[:max(0,10-len(main_uniq))], 14: main_uniq + rest[:max(0,14-len(main_uniq))], 20: main_uniq + rest[:max(0,20-len(main_uniq))]}
        conn.execute("DELETE FROM prediction_pools WHERE run_id=?", (run_id,))
        for sz, nums in pools.items():
            conn.execute("INSERT INTO prediction_pools(run_id, pool_size, numbers_json, created_at) VALUES (?,?,?,?)", (run_id, sz, json.dumps(nums), now))
    conn.commit()
    return target

def run_historical_backtest(conn, min_history=6, rebuild=False, progress_every=20, max_issues=120):
    draws = _draws_ordered_asc(conn)
    if len(draws) <= min_history: return 0,0
    if rebuild:
        conn.execute("DELETE FROM prediction_runs WHERE issue_no IN (SELECT issue_no FROM draws)")
        conn.execute("DELETE FROM strategy_performance WHERE issue_no IN (SELECT issue_no FROM draws)")
    total = min(max_issues, len(draws)-min_history)
    print(f"[backtest] 将处理 {total} 期...")
    processed = 0
    for i in range(min_history, len(draws)):
        if processed >= max_issues: break
        target = draws[i]
        issue = target["issue_no"]
        hist = [json.loads(draws[j]["numbers_json"]) for j in range(i-1, max(-1, i-20-1), -1)]
        win_main = set(json.loads(target["numbers_json"]))
        win_sp = target["special_number"]
        for s in STRATEGY_IDS:
            picks, sp, _, _ = generate_strategy(hist, s, _default_mined_config(), get_strategy_weights(conn), conn, issue)
            hit = len([n for n,_,_,_ in picks if n in win_main])
            sp_hit = 1 if sp == win_sp else 0
            now = utc_now()
            conn.execute("INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at) VALUES (?,?,?,?,?)", (issue, s, hit, sp_hit, now))
        processed += 1
        if processed % progress_every == 0: print(f"[backtest] {processed}/{total}")
    conn.commit()
    return processed, processed*len(STRATEGY_IDS)

def review_latest(conn):
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if row:
        runs = conn.execute("SELECT id, strategy FROM prediction_runs WHERE issue_no=? AND status='PENDING'", (row["issue_no"],)).fetchall()
        for run in runs:
            mains, sp = get_picks_for_run(conn, run["id"])
            draw = conn.execute("SELECT numbers_json, special_number FROM draws WHERE issue_no=?", (row["issue_no"],)).fetchone()
            if draw:
                win = set(json.loads(draw["numbers_json"]))
                hit = len([n for n in mains if n in win])
                sp_hit = 1 if sp == draw["special_number"] else 0
                conn.execute("UPDATE prediction_runs SET status='REVIEWED', hit_count=?, hit_rate=?, special_hit=?, reviewed_at=? WHERE id=?", (hit, hit/6.0, sp_hit, utc_now(), run["id"]))
                conn.execute("INSERT OR REPLACE INTO strategy_performance(issue_no, strategy, main_hit_count, special_hit, created_at) VALUES (?,?,?,?,?)", (row["issue_no"], run["strategy"], hit, sp_hit, utc_now()))
        conn.commit()
        return len(runs)
    return 0

# -------------------- 命令行 --------------------
def cmd_bootstrap(args):
    conn = connect_db(args.db); init_db(conn)
    records = fetch_marksix6_records()
    sync_from_records(conn, records, "api")
    generate_predictions(conn)
    print("Bootstrap done."); conn.close()

def cmd_sync(args):
    conn = connect_db(args.db); init_db(conn)
    records = fetch_marksix6_records()
    sync_from_records(conn, records, "api")
    generate_predictions(conn)
    conn.close()

def cmd_show(args):
    conn = connect_db(args.db); init_db(conn)
    print_dashboard(conn)
    if PUSHPLUS_TOKEN:
        rec = get_dynamic_final_recommendation(conn)
        if rec:
            iss, m6, sp, _, _, _, trio, conf, z1, z2 = rec
            content = f"【香港六合彩·{iss}期】\n6码: {' '.join(f'{n:02d}' for n in m6)}\n特别号: {sp:02d}\n三中三: {' '.join(f'{n:02d}' for n in trio)}\n1生肖: {z1} 2生肖: {'、'.join(z2)}\n置信度: {conf}/100"
            send_pushplus_notification(f"香港预测 {iss}", content)
    conn.close()

def cmd_fullbacktest(args):
    conn = connect_db(args.db); init_db(conn)
    backup_database(args.db)
    print("同步最新数据...")
    records = fetch_marksix6_records()
    sync_from_records(conn, records, "api")
    print(f"执行回测 (最多{args.max_issues}期)...")
    run_historical_backtest(conn, rebuild=True, max_issues=args.max_issues)
    review_latest(conn)
    generate_predictions(conn)
    print("全量回测完成。")
    conn.close()

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_PATH_DEFAULT)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap").set_defaults(func=cmd_bootstrap)
    sub.add_parser("sync").set_defaults(func=cmd_sync)
    sub.add_parser("show").set_defaults(func=cmd_show)
    fb = sub.add_parser("fullbacktest")
    fb.add_argument("--max-issues", type=int, default=60)
    fb.set_defaults(func=cmd_fullbacktest)
    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
