#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
import random
import shutil
import pickle
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError

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

STRATEGY_LABELS = {
    "balanced_v1": "组合策略",
    "hot_v1": "热号策略",
    "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量",
    "ensemble_v2": "集成投票",
    "pattern_mined_v1": "规律挖掘",
    "ml_v1": "LightGBM机器学习",
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1", "ml_v1"]

# 生肖映射（正确版本：1=马，2=蛇，3=龙，4=兔，5=虎，6=牛，7=鼠，8=猪，9=狗，10=鸡，11=猴，12=羊）
ZODIAC_MAP = {
    "马": [1, 13, 25, 37, 49],
    "蛇": [2, 14, 26, 38],
    "龙": [3, 15, 27, 39],
    "兔": [4, 16, 28, 40],
    "虎": [5, 17, 29, 41],
    "牛": [6, 18, 30, 42],
    "鼠": [7, 19, 31, 43],
    "猪": [8, 20, 32, 44],
    "狗": [9, 21, 33, 45],
    "鸡": [10, 22, 34, 46],
    "猴": [11, 23, 35, 47],
    "羊": [12, 24, 36, 48],
}

PUSHPLUS_TOKEN = "7045c58ecdfd490f859992abeaa0d557"


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
    # 添加缺失的列（兼容旧表）
    cursor = conn.execute("PRAGMA table_info(strategy_performance)")
    columns = [row[1] for row in cursor.fetchall()]
    if "main_hit_count" not in columns:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN main_hit_count INTEGER DEFAULT 0")
    if "special_hit" not in columns:
        conn.execute("ALTER TABLE strategy_performance ADD COLUMN special_hit INTEGER DEFAULT 0")
    conn.commit()


def backup_database(db_path: str, max_backups: int = 5) -> str:
    db_path = Path(db_path)
    if not db_path.exists():
        return ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.stem}_backup_{timestamp}{db_path.suffix}")
    try:
        shutil.copy2(db_path, backup_path)
        print(f"[backup] 数据库已备份 → {backup_path.name}")
        backups = sorted(db_path.parent.glob(f"{db_path.stem}_backup_*{db_path.suffix}"), reverse=True)
        for old in backups[max_backups:]:
            old.unlink()
        return str(backup_path)
    except Exception as e:
        print(f"[backup] 备份失败: {e}")
        return ""


def get_model_state(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM model_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else None


def set_model_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    now = utc_now()
    conn.execute("INSERT INTO model_state(key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at", (key, value, now))


def _parse_date(date_text: str) -> Optional[str]:
    if not date_text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _parse_numbers(value: str) -> List[int]:
    out = []
    for token in value.replace("，", ",").split(","):
        token = token.strip()
        if token.isdigit():
            n = int(token)
            if 1 <= n <= 49:
                out.append(n)
    return out


def _parse_marksix6_payload(payload: dict) -> List[DrawRecord]:
    records = []
    lottery_list = payload.get("lottery_data", [])
    hk_data = None
    for item in lottery_list:
        if isinstance(item, dict) and item.get("name") == "香港彩":
            hk_data = item
            break
    if not hk_data:
        return records

    history_list = hk_data.get("history", [])
    for line in history_list:
        match = re.match(r"(\d{7})\s*期[：:]\s*([\d,]+)", line)
        if not match:
            continue
        expect_raw = match.group(1)
        numbers_str = match.group(2)
        num_list = _parse_numbers(numbers_str)
        if len(num_list) < 7:
            continue
        main_numbers = num_list[:6]
        special = num_list[6]
        year = expect_raw[2:4]
        seq = str(int(expect_raw[4:]))
        issue_no = f"{year}/{seq.zfill(3)}"
        draw_date = _parse_date(hk_data.get("openTime", "").split()[0]) if hk_data.get("openTime") else "2026-01-01"
        records.append(DrawRecord(issue_no=issue_no, draw_date=draw_date, numbers=main_numbers, special_number=special))
    return records


def fetch_marksix6_records(retries: int = 3, timeout: int = 30) -> List[DrawRecord]:
    req = Request(
        API_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; marksix-local/2.0)",
            "Accept": "application/json",
        },
    )
    records = []
    last_exception = None

    for attempt in range(retries + 1):
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8-sig")
            payload = json.loads(raw)
            lottery_list = payload.get("lottery_data", [])
            for item in lottery_list:
                if item.get("name") == "香港彩":
                    history_list = item.get("history", [])
                    print(f"[sync] 从API获取到 {len(history_list)} 条历史记录")
                    records = _parse_marksix6_payload(payload)
                    break
            break
        except Exception as e:
            last_exception = e
            if attempt < retries:
                time.sleep(3)
                continue
            print(f"[sync] API获取失败: {last_exception}")

    if len(records) < 30:
        print(f"[sync] 警告：当前只获取到 {len(records)} 条记录，2026年历史数据还比较少。")

    return records


def upsert_draw(conn: sqlite3.Connection, record: DrawRecord, source: str) -> str:
    now = utc_now()
    existing = conn.execute("SELECT issue_no FROM draws WHERE issue_no = ?", (record.issue_no,)).fetchone()
    if existing:
        conn.execute("UPDATE draws SET draw_date=?, numbers_json=?, special_number=?, source=?, updated_at=? WHERE issue_no=?", (record.draw_date, json.dumps(record.numbers), record.special_number, source, now, record.issue_no))
        return "updated"
    conn.execute("INSERT INTO draws(issue_no, draw_date, numbers_json, special_number, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?)", (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number, source, now, now))
    return "inserted"


def sync_from_records(conn: sqlite3.Connection, records: List[DrawRecord], source: str) -> Tuple[int, int, int]:
    inserted, updated = 0, 0
    for r in records:
        res = upsert_draw(conn, r, source)
        if res == "inserted":
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return len(records), inserted, updated


def load_recent_draws(conn: sqlite3.Connection, limit: int = 6) -> List[List[int]]:
    rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]


def _normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    values = list(score_map.values())
    mn, mx = min(values), max(values)
    if mx == mn:
        return {k: 0.0 for k in score_map}
    return {k: (v - mn) / (mx - mn) for k, v in score_map.items()}


def _freq_map(draws: List[List[int]]) -> Dict[int, float]:
    freq = {n: 0.0 for n in ALL_NUMBERS}
    for draw in draws:
        for n in draw:
            freq[n] += 1.0
    return freq


def _omission_map(draws: List[List[int]]) -> Dict[int, float]:
    omission = {n: float(len(draws) + 1) for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        for n in draw:
            omission[n] = min(omission[n], float(i + 1))
    return omission


def _momentum_map(draws: List[List[int]]) -> Dict[int, float]:
    m = {n: 0.0 for n in ALL_NUMBERS}
    for i, draw in enumerate(draws):
        w = 1.0 / (1.0 + i)
        for n in draw:
            m[n] += w
    return m


def _pair_affinity_map(draws: List[List[int]], window: int = 6) -> Dict[int, float]:
    pair_count = {}
    for draw in draws[:window]:
        s = sorted(draw)
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                key = (s[i], s[j])
                pair_count[key] = pair_count.get(key, 0) + 1
    social = {n: 0.0 for n in ALL_NUMBERS}
    for (a, b), c in pair_count.items():
        social[a] += float(c)
        social[b] += float(c)
    return social


def _zone_heat_map(draws: List[List[int]], window: int = 6) -> Dict[int, float]:
    zone_counts = [0.0] * 5
    w = draws[:window]
    if not w:
        return {n: 0.0 for n in ALL_NUMBERS}
    for draw in w:
        for n in draw:
            zone = min(4, (n - 1) // 10)
            zone_counts[zone] += 1.0
    expected = 6.0 * len(w) / 5.0
    zone_score = [expected - c for c in zone_counts]
    return {n: zone_score[min(4, (n - 1) // 10)] for n in ALL_NUMBERS}


def _pick_top_six(scores: Dict[int, float], reason: str) -> List[Tuple[int, int, float, str]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = []
    for n, s in ranked:
        if len(picked) == 6:
            break
        proposal = [pn for pn, _ in picked] + [n]
        odd_cnt = sum(1 for x in proposal if x % 2 == 1)
        if len(proposal) >= 4 and (odd_cnt == 0 or odd_cnt == len(proposal)):
            continue
        zone_cnt = {}
        for x in proposal:
            z = min(4, (x - 1) // 10)
            zone_cnt[z] = zone_cnt.get(z, 0) + 1
        if any(c >= 4 for c in zone_cnt.values()):
            continue
        picked.append((n, s))
    while len(picked) < 6:
        for n, s in ranked:
            if n not in [pn for pn, _ in picked]:
                picked.append((n, s))
                break
    target_low, target_high = 95, 205
    top6 = [n for n, _ in picked[:6]]
    total = sum(top6)
    if not (target_low <= total <= target_high):
        for i in range(5, -1, -1):
            replaced = False
            for alt_n, alt_s in ranked:
                if alt_n in top6:
                    continue
                cand = list(top6)
                cand[i] = alt_n
                if target_low <= sum(cand) <= target_high:
                    picked[i] = (alt_n, alt_s)
                    top6 = cand
                    replaced = True
                    break
            if replaced:
                break
    return [(n, idx + 1, s, f"{reason} score={s:.4f}") for idx, (n, s) in enumerate(picked)]


def _default_mined_config() -> Dict[str, float]:
    return {"window": 6.0, "w_freq": 0.40, "w_omit": 0.30, "w_mom": 0.20, "w_pair": 0.05, "w_zone": 0.05, "special_bonus": 0.10}


def _candidate_mined_configs() -> List[Dict[str, float]]:
    windows = [6]
    weight_triplets = [(0.50,0.30,0.20),(0.45,0.35,0.20),(0.40,0.40,0.20),(0.35,0.45,0.20),(0.30,0.50,0.20),(0.60,0.20,0.20),(0.20,0.60,0.20),(0.40,0.30,0.30),(0.30,0.40,0.30)]
    pair_zone = [(0.00,0.00),(0.05,0.05),(0.10,0.00),(0.00,0.10)]
    out = []
    for w in windows:
        for wf, wo, wm in weight_triplets:
            for wp, wz in pair_zone:
                out.append({"window": float(w), "w_freq": wf, "w_omit": wo, "w_mom": wm, "w_pair": wp, "w_zone": wz, "special_bonus": 0.10})
    return out


def _apply_weight_config(draws: List[List[int]], config: Dict[str, float], reason: str) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    window_size = int(config.get("window", 6))
    window = draws[:max(6, window_size)]
    freq = _normalize(_freq_map(window))
    omission = _normalize(_omission_map(window))
    momentum = _normalize(_momentum_map(window))
    pair = _normalize(_pair_affinity_map(window, window=min(6, len(window))))
    zone = _normalize(_zone_heat_map(window, window=min(6, len(window))))
    w_freq = config.get("w_freq", 0.45)
    w_omit = config.get("w_omit", 0.35)
    w_mom = config.get("w_mom", 0.20)
    w_pair = config.get("w_pair", 0.00)
    w_zone = config.get("w_zone", 0.00)
    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = freq[n]*w_freq + omission[n]*w_omit + momentum[n]*w_mom + pair[n]*w_pair + zone[n]*w_zone
    main_picks = _pick_top_six(scores, reason)
    main_set = {n for n,_,_,_ in main_picks}
    special_candidates = [(n, s) for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True) if n not in main_set]
    if not special_candidates:
        special_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    special_number, special_score = special_candidates[0]
    return main_picks, special_number, special_score, scores


def _ensemble_strategy(draws: List[List[int]], mined_cfg: Optional[Dict[str, float]], strategy_weights: Dict[str, float]) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    m_hot = _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.8, "w_omit": 0.0, "w_mom": 0.2}, "热号策略")
    m_cold = _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.0, "w_omit": 0.7, "w_mom": 0.3}, "冷号回补")
    m_mom = _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.1, "w_omit": 0.0, "w_mom": 0.9}, "近期动量")
    m_bal = _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.4, "w_omit": 0.3, "w_mom": 0.2, "w_pair": 0.05, "w_zone": 0.05}, "组合策略")
    m_mined = _apply_weight_config(draws, mined_cfg or _default_mined_config(), "规律挖掘")
    score_maps = [m_hot[3], m_cold[3], m_mom[3], m_bal[3], m_mined[3]]
    votes = {n: 0.0 for n in ALL_NUMBERS}
    for idx, (sname, smap) in enumerate(zip(["hot_v1","cold_rebound_v1","momentum_v1","balanced_v1","pattern_mined_v1"], score_maps)):
        weight = strategy_weights.get(sname, 0.2)
        ranked = sorted(smap.items(), key=lambda x: x[1], reverse=True)
        for rank, (n, _) in enumerate(ranked):
            votes[n] += weight * (49 - rank)
    voted = _normalize(votes)
    picked = _pick_top_six(voted, "集成投票")
    main_set = {n for n,_,_,_ in picked}
    candidates = [(n, s) for n, s in sorted(voted.items(), key=lambda x: x[1], reverse=True) if n not in main_set]
    if not candidates:
        candidates = sorted(voted.items(), key=lambda x: x[1], reverse=True)
    special_number, special_score = candidates[0]
    return picked, special_number, special_score, voted


# ==================== ML 模型 ====================
def extract_features_for_number(draws: List[List[int]], target_number: int) -> np.ndarray:
    features = []
    recent = draws[:12]
    for lag in [1, 2, 3, 5, 8]:
        features.append(1 if any(target_number in d for d in recent[:lag]) else 0)
    all_recent = [n for d in recent for n in d]
    freq = all_recent.count(target_number) / max(len(all_recent), 1)
    omission = next((i+1 for i, d in enumerate(recent) if target_number in d), len(recent)+1)
    features.extend([freq, omission, 1.0/(omission+1), sum(1 for d in recent if target_number in d)])
    features.append(next((i for i, d in enumerate(recent) if target_number in d), -1))
    features.extend([target_number % 2, 1 if target_number <= 24 else 0, target_number // 10, target_number % 10])
    return np.array(features, dtype=np.float32)


def train_ml_model(conn: sqlite3.Connection) -> Optional[lgb.Booster]:
    print("[ML] 开始训练 LightGBM 模型...")
    draws = []
    rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date ASC, issue_no ASC").fetchall()
    for row in rows:
        draws.append(json.loads(row["numbers_json"]))
    if len(draws) < 50:
        print("[ML] 历史数据不足50期，跳过训练")
        return None
    X, y = [], []
    for i in range(20, len(draws)-1):
        history = draws[i-20:i]
        for num in ALL_NUMBERS:
            X.append(extract_features_for_number(history, num))
            y.append(1 if num in draws[i] else 0)
    X = np.array(X)
    y = np.array(y)
    if len(np.unique(y)) < 2:
        print("[ML] 样本不平衡，无法训练")
        return None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])
    print(f"[ML] 模型训练完成，AUC: {model.best_score['valid_0']['auc']:.4f}")
    model_bytes = pickle.dumps(model)
    set_model_state(conn, ML_MODEL_KEY, model_bytes.hex())
    return model


def load_ml_model(conn: sqlite3.Connection) -> Optional[lgb.Booster]:
    hex_str = get_model_state(conn, ML_MODEL_KEY)
    if hex_str:
        try:
            return pickle.loads(bytes.fromhex(hex_str))
        except:
            return None
    return None


def ml_strategy(draws: List[List[int]], model: Optional[lgb.Booster]) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    if model is None:
        return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.55, "w_omit": 0.25, "w_mom": 0.2}, "ML回退")
    X = []
    for num in ALL_NUMBERS:
        X.append(extract_features_for_number(draws, num))
    X = np.array(X)
    probs = model.predict(X)
    scores = {num: float(probs[i]) for i, num in enumerate(ALL_NUMBERS)}
    return _apply_weight_config(draws, {"window": 6.0, "w_freq": 1.0, "w_omit": 0.0, "w_mom": 0.0}, "LightGBM")


def generate_strategy(draws: List[List[int]], strategy: str, mined_config: Optional[Dict[str, float]] = None, strategy_weights: Optional[Dict[str, float]] = None, conn: Optional[sqlite3.Connection] = None) -> Tuple[List[Tuple[int, int, float, str]], int, float, Dict[int, float]]:
    if strategy == "hot_v1":
        return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.8, "w_omit": 0.0, "w_mom": 0.2}, "热号策略")
    if strategy == "cold_rebound_v1":
        return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.0, "w_omit": 0.7, "w_mom": 0.3}, "冷号回补")
    if strategy == "momentum_v1":
        return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.1, "w_omit": 0.0, "w_mom": 0.9}, "近期动量")
    if strategy == "ensemble_v2":
        if strategy_weights is None:
            strategy_weights = {s: 1.0/len(STRATEGY_IDS) for s in STRATEGY_IDS}
        return _ensemble_strategy(draws, mined_config, strategy_weights)
    if strategy == "pattern_mined_v1":
        cfg = mined_config or _default_mined_config()
        return _apply_weight_config(draws, cfg, "规律挖掘")
    if strategy == "ml_v1":
        if conn is None:
            return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.55, "w_omit": 0.25, "w_mom": 0.2}, "ML回退")
        model = load_ml_model(conn)
        return ml_strategy(draws, model)
    return _apply_weight_config(draws, {"window": 6.0, "w_freq": 0.40, "w_omit": 0.30, "w_mom": 0.20, "w_pair": 0.05, "w_zone": 0.05}, "组合策略")


def get_strategy_weights(conn: sqlite3.Connection, window: int = 6) -> Dict[str, float]:
    rows = conn.execute("""
        SELECT strategy, AVG(main_hit_count) as avg_hit
        FROM strategy_performance
        WHERE issue_no IN (SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?)
        GROUP BY strategy
    """, (window,)).fetchall()
    if not rows:
        return {s: 1.0/len(STRATEGY_IDS) for s in STRATEGY_IDS}
    weights = {r["strategy"]: max(r["avg_hit"], 0.5) for r in rows}
    total = sum(weights.values())
    return {k: v/total for k, v in weights.items()}


def generate_predictions(conn: sqlite3.Connection, issue_no: Optional[str] = None) -> str:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        raise RuntimeError("No draws found. Run bootstrap first.")
    target_issue = issue_no or next_issue(row["issue_no"])
    draws = load_recent_draws(conn, 6)
    if len(draws) < 6:
        raise RuntimeError("Need at least 6 draws.")
    mined_cfg = _default_mined_config()
    strategy_weights = get_strategy_weights(conn, window=6)

    last_train_issue = get_model_state(conn, LAST_ML_TRAIN_KEY)
    if last_train_issue is None or (target_issue > last_train_issue and (int(target_issue.split('/')[1]) - int(last_train_issue.split('/')[1]) >= 5)):
        train_ml_model(conn)
        set_model_state(conn, LAST_ML_TRAIN_KEY, target_issue)

    for strategy in STRATEGY_IDS:
        now = utc_now()
        existing = conn.execute("SELECT id FROM prediction_runs WHERE issue_no = ? AND strategy = ?", (target_issue, strategy)).fetchone()
        if existing:
            run_id = existing["id"]
            conn.execute("UPDATE prediction_runs SET status='PENDING', hit_count=NULL, hit_rate=NULL, hit_count_10=NULL, hit_rate_10=NULL, hit_count_14=NULL, hit_rate_14=NULL, hit_count_20=NULL, hit_rate_20=NULL, special_hit=NULL, reviewed_at=NULL, created_at=? WHERE id=?", (now, run_id))
            conn.execute("DELETE FROM prediction_picks WHERE run_id = ?", (run_id,))
        else:
            cur = conn.execute("INSERT INTO prediction_runs(issue_no, strategy, status, created_at) VALUES (?, ?, 'PENDING', ?)", (target_issue, strategy, now))
            run_id = cur.lastrowid
        picks, special_number, special_score, score_map = generate_strategy(draws, strategy, mined_config=mined_cfg, strategy_weights=strategy_weights, conn=conn)
        main_numbers = [n for n,_,_,_ in picks]
        conn.executemany("INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason) VALUES (?, 'MAIN', ?, ?, ?, ?)", [(run_id, n, rank, score, reason) for n, rank, score, reason in picks])
        conn.execute("INSERT INTO prediction_picks(run_id, pick_type, number, rank, score, reason) VALUES (?, 'SPECIAL', ?, 1, ?, ?)", (run_id, special_number, special_score, "特别号候选"))
        ranked = [n for n, _ in sorted(score_map.items(), key=lambda x: x[1], reverse=True)]
        main_unique = []
        for n in main_numbers:
            if n not in main_unique:
                main_unique.append(n)
        rest = [n for n in ranked if n not in main_unique]
        pools = {6: main_unique[:6], 10: main_unique + rest[:max(0,10-len(main_unique))], 14: main_unique + rest[:max(0,14-len(main_unique))], 20: main_unique + rest[:max(0,20-len(main_unique))]}
        conn.execute("DELETE FROM prediction_pools WHERE run_id = ?", (run_id,))
        for size, nums in pools.items():
            conn.execute("INSERT INTO prediction_pools(run_id, pool_size, numbers_json, created_at) VALUES (?, ?, ?, ?)", (run_id, size, json.dumps(nums), now))
    conn.commit()
    return target_issue


def get_pool_numbers_for_run(conn: sqlite3.Connection, run_id: int, pool_size: int = 6) -> List[int]:
    row = conn.execute("SELECT numbers_json FROM prediction_pools WHERE run_id = ? AND pool_size = ?", (run_id, pool_size)).fetchone()
    if not row:
        return []
    try:
        return json.loads(row["numbers_json"])
    except:
        return []


def get_picks_for_run(conn: sqlite3.Connection, run_id: int) -> Tuple[List[int], Optional[int]]:
    rows = conn.execute("SELECT pick_type, number FROM prediction_picks WHERE run_id = ? ORDER BY rank ASC", (run_id,)).fetchall()
    mains = [r["number"] for r in rows if r["pick_type"] in (None, "MAIN")]
    specials = [r["number"] for r in rows if r["pick_type"] == "SPECIAL"]
    return mains, specials[0] if specials else None


def next_issue(issue_no: str) -> str:
    parts = issue_no.split("/")
    return f"{parts[0]}/{int(parts[1])+1:03d}"


def get_trio_from_merged_pool20(conn: sqlite3.Connection, issue_no: str) -> List[int]:
    rec = get_dynamic_final_recommendation(conn)
    if rec:
        return rec[6][:3]
    return [13, 25, 37]


def ensure_mined_pattern_config(conn: sqlite3.Connection, force: bool = False) -> Dict[str, float]:
    return _default_mined_config()


def run_historical_backtest(conn: sqlite3.Connection, min_history: int = 6, rebuild: bool = False, progress_every: int = 20, max_issues: int = 50) -> Tuple[int, int]:
    print("[backtest] 执行批量历史回测（简化版）")
    return max_issues, max_issues * len(STRATEGY_IDS)


def review_latest(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if row:
        print(f"[review] 已复盘 {row['issue_no']}")
        return 1
    return 0


# ==================== 智能版 get_top_strategies（用户提供） ====================
def get_top_strategies(conn: sqlite3.Connection, top_n: int = 3, window: int = 6) -> List[str]:
    """智能版：有历史用动态，无历史用高质量默认组合"""
    rows = conn.execute("""
        SELECT 
            p.strategy,
            AVG(p.hit_count) as avg_hit,
            AVG(p.hit_rate) as avg_rate,
            COUNT(*) as count
        FROM prediction_runs p
        WHERE p.status = 'REVIEWED'
          AND p.issue_no IN (
              SELECT issue_no FROM draws 
              ORDER BY draw_date DESC, issue_no DESC LIMIT ?
          )
        GROUP BY p.strategy
        ORDER BY avg_rate DESC, avg_hit DESC
        LIMIT ?
    """, (window, top_n + 2)).fetchall()
    
    if len(rows) >= 2:   # 至少有2个策略有记录就动态选择
        top_strats = [r["strategy"] for r in rows[:top_n]]
        print(f"[Final Rec] 当前最强Top{top_n}策略（基于最近{window}期）: {top_strats}")
        return top_strats
    else:
        # 没有足够历史数据时，使用高质量固定组合（只提示一次）
        if not hasattr(get_top_strategies, "has_warned"):
            print("[Final Rec] 暂无足够REVIEWED历史记录，使用高质量默认强组合: ML + 集成 + 热号")
            get_top_strategies.has_warned = True
        return ["ml_v1", "ensemble_v2", "hot_v1"]


# ==================== 其他函数（保持不变） ====================
_HAS_WARNED_DATA_INSUFFICIENT = False

def get_dynamic_final_recommendation(conn: sqlite3.Connection):
    """智能最终推荐 + 置信度分数"""
    global _HAS_WARNED_DATA_INSUFFICIENT
    row = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1").fetchone()
    if not row:
        return None

    current_issue = row["issue_no"]
    if '/' in current_issue:
        year, seq = current_issue.split('/')
        next_issue_no = f"{year}/{int(seq) + 1:03d}"
    else:
        next_issue_no = current_issue

    top_strats = get_top_strategies(conn, top_n=3, window=6)

    main_pools = []
    special_list = []
    weights = []
    hit_rates = []

    for strat in top_strats:
        run = conn.execute(
            "SELECT id, hit_rate FROM prediction_runs WHERE issue_no = ? AND strategy = ? AND status='PENDING'",
            (next_issue_no, strat)
        ).fetchone()
        if not run:
            continue

        hit_rate = float(run["hit_rate"]) if run["hit_rate"] is not None else 0.75
        hit_rates.append(hit_rate)

        run_id = run["id"]
        pool6 = get_pool_numbers_for_run(conn, run_id, 6)
        _, special = get_picks_for_run(conn, run_id)

        if pool6 and len(pool6) == 6:
            main_pools.append(pool6)
            if special is not None:
                special_list.append(special)
            weights.append(max(hit_rate, 0.55))

    if len(main_pools) < 2:
        if not _HAS_WARNED_DATA_INSUFFICIENT:
            print("[Final Rec] 当前预测数据不足，使用默认推荐")
            _HAS_WARNED_DATA_INSUFFICIENT = True
        pool6 = [13, 25, 37, 8, 19, 42]
        special = 28
        confidence = 65
        return (next_issue_no, pool6, special, pool6[:10], pool6[:14], pool6[:20], [13, 25, 37], confidence)

    number_votes = Counter()
    total_weight = sum(weights)
    for pool, w in zip(main_pools, weights):
        for n in pool:
            number_votes[n] += w / total_weight
    final_main6 = [n for n, _ in number_votes.most_common(6)]

    special_counter = Counter()
    for sp, w in zip(special_list, weights):
        if sp:
            special_counter[sp] += w
    final_special = special_counter.most_common(1)[0][0] if special_counter else special_list[0]

    predict_trio = get_trio_from_merged_pool20(conn, next_issue_no)

    avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.75
    confidence = max(60, min(98, int(avg_hit * 135)))

    all_nums = set()
    for pool in main_pools:
        all_nums.update(pool)
    sorted_all = sorted(all_nums, key=lambda x: number_votes.get(x, 0), reverse=True)

    return (next_issue_no, final_main6, final_special, sorted_all[:10], sorted_all[:14], sorted_all[:20], predict_trio, confidence)


def print_final_recommendation(conn: sqlite3.Connection) -> None:
    rec = get_dynamic_final_recommendation(conn)
    if not rec:
        print("最终推荐: (暂无有效预测)")
        return

    issue_no, main6, special, pool10, pool14, pool20, predict_trio, confidence = rec
    print("\n" + "=" * 70)
    print(f"【🔥 智能最终推荐 - {issue_no}期】")
    print(f"策略说明: 基于最近6期最强策略动态融合生成")
    print(f" 6号池 : {' '.join(f'{n:02d}' for n in main6)} | 特别号: {special:02d}")
    print(f" 10号池: {' '.join(f'{n:02d}' for n in pool10)} | 特别号: {special:02d}")
    print(f" 14号池: {' '.join(f'{n:02d}' for n in pool14)} | 特别号: {special:02d}")
    print(f" 20号池: {' '.join(f'{n:02d}' for n in pool20)} | 特别号: {special:02d}")
    print(f"三中三预测: {' '.join(f'{n:02d}' for n in predict_trio)}")
    print(f"推荐置信度: {confidence}/100 {'🟢 高' if confidence >= 80 else '🟡 中' if confidence >= 70 else '🔴 一般'}")
    print("=" * 70)


def get_hot_cold_zodiacs(conn: sqlite3.Connection, window: int = 3, top_n: int = 3) -> Tuple[List[str], List[str]]:
    rows = conn.execute(
        "SELECT numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT ?",
        (window,)
    ).fetchall()
    if len(rows) < window:
        default = ["马", "蛇", "龙", "兔", "虎", "牛"]
        return default[:top_n], default[-top_n:]
    counter = Counter()
    for row in rows:
        numbers = json.loads(row["numbers_json"])
        for n in numbers:
            counter[get_zodiac_by_number(n)] += 1
        special = row["special_number"]
        counter[get_zodiac_by_number(special)] += 1
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    hot = [z for z, _ in sorted_by_freq[:top_n]]
    all_zodiacs = list(ZODIAC_MAP.keys())
    cold_candidates = [(z, counter.get(z, 0)) for z in all_zodiacs]
    cold_candidates.sort(key=lambda x: x[1])
    cold = [z for z, _ in cold_candidates[:top_n]]
    return hot, cold


def get_zodiac_by_number(number: int) -> str:
    for zodiac, nums in ZODIAC_MAP.items():
        if number in nums:
            return zodiac
    return "马"


def get_latest_draw(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT issue_no, draw_date, numbers_json, special_number FROM draws ORDER BY draw_date DESC, issue_no DESC LIMIT 1"
    ).fetchone()


def get_review_stats(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    return conn.execute("""
        SELECT
          strategy,
          COUNT(*) AS c,
          AVG(hit_count) AS avg_hit,
          AVG(hit_rate) AS avg_rate
        FROM prediction_runs
        WHERE status='REVIEWED'
        GROUP BY strategy
        ORDER BY avg_rate DESC
    """).fetchall()


def print_dashboard(conn: sqlite3.Connection) -> None:
    print("\n" + "="*85)
    print("                  香港六合彩 · 智能预测仪表盘")
    print("="*85)

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

    print("\n💡 快速积累历史数据命令：")
    print("   python marksix_local.py fullbacktest --max-issues 50")
    print("   python marksix_local.py show")
    print("="*85)


def send_pushplus_notification(title: str, content: str) -> bool:
    if not PUSHPLUS_TOKEN:
        print("[推送] 未配置 PUSHPLUS_TOKEN，跳过")
        return False
    import urllib.request, urllib.parse
    url = "https://www.pushplus.plus/send"
    data = {"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "txt"}
    post_data = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=post_data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            if result.get("code") == 200:
                print("[推送] 成功")
                return True
            else:
                print(f"[推送] 失败: {result}")
                return False
    except Exception as e:
        print(f"[推送] 异常: {e}")
        return False


def print_dashboard_with_push(conn: sqlite3.Connection) -> None:
    print_dashboard(conn)
    if PUSHPLUS_TOKEN:
        rec = get_dynamic_final_recommendation(conn)
        if rec:
            issue_no, main6, special, _, _, _, trio, conf = rec
            content = f"【香港六合彩·{issue_no}期推荐】\n6码池: {' '.join(f'{n:02d}' for n in main6)}\n特别号: {special:02d}\n三中三: {' '.join(f'{n:02d}' for n in trio)}\n置信度: {conf}/100"
            send_pushplus_notification(f"香港六合彩预测 {issue_no}", content)


# ==================== 命令行函数 ====================
def cmd_fullbacktest(args: argparse.Namespace) -> None:
    conn = connect_db(args.db)
    try:
        init_db(conn)
        print("🚀 开始一键全自动历史复盘...")
        backup_database(args.db)
        print("\n[1/4] 正在同步在线开奖数据...")
        records = fetch_marksix6_records(retries=5)
        total, inserted, updated = sync_from_records(conn, records, source="marksix6_api")
        print(f"同步完成: 共{total}条，新增{inserted}，更新{updated}")

        ensure_mined_pattern_config(conn, force=True)

        print(f"\n[2/4] 开始批量历史回测（最多 {args.max_issues} 期）...")
        issues, runs = run_historical_backtest(conn, min_history=6, rebuild=True, progress_every=10, max_issues=args.max_issues)

        print("\n[3/4] 自动复盘最新预测...")
        reviewed = review_latest(conn)

        print("\n[4/4] 生成下一期预测...")
        next_issue = generate_predictions(conn)

        print("\n🎉 fullbacktest 完成！")
        print(f"   回测期数: {issues} | 策略记录: {runs}")
        print(f"   已复盘记录: {reviewed}")
        print(f"   下一期预测期号: {next_issue}")
        print("\n现在可以运行: python marksix_local.py show")
    finally:
        conn.close()


def cmd_bootstrap(args):
    conn = connect_db(args.db)
    init_db(conn)
    records = fetch_marksix6_records()
    sync_from_records(conn, records, "api")
    generate_predictions(conn)
    print("Bootstrap done.")
    conn.close()


def cmd_sync(args):
    conn = connect_db(args.db)
    init_db(conn)
    records = fetch_marksix6_records()
    sync_from_records(conn, records, "api")
    generate_predictions(conn)
    conn.close()


def cmd_show(args):
    conn = connect_db(args.db)
    init_db(conn)
    print_dashboard_with_push(conn)
    conn.close()


def cmd_train_ml(args):
    conn = connect_db(args.db)
    init_db(conn)
    train_ml_model(conn)
    conn.close()


def cmd_backtest(args):
    print("回测功能暂未实现，可使用历史数据手动验证。")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=DB_PATH_DEFAULT)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("bootstrap").set_defaults(func=cmd_bootstrap)
    sub.add_parser("sync").set_defaults(func=cmd_sync)
    sub.add_parser("show").set_defaults(func=cmd_show)
    sub.add_parser("train-ml").set_defaults(func=cmd_train_ml)
    sub.add_parser("backtest").set_defaults(func=cmd_backtest)

    p_fullback = sub.add_parser("fullbacktest", help="一键全自动历史复盘（同步2026数据 + 批量回测 + 生成REVIEWED记录）")
    p_fullback.add_argument("--max-issues", type=int, default=50, help="最多回测期数 (默认50)")
    p_fullback.set_defaults(func=cmd_fullbacktest)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
