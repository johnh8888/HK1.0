#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH_DEFAULT = str(SCRIPT_DIR / "marksix_local.db")

# 数据源优先级：marksix6.net → 香港赛马会官网
HK_API_URL = "https://marksix6.net/index.php?api=1"
OFFICIAL_URL = "https://bet.hkjc.com/contentserver/jcbw/cmc/last30draw.json"

STRATEGY_LABELS = {
    "balanced_v1": "组合策略",
    "hot_v1": "热号策略",
    "cold_rebound_v1": "冷号回补",
    "momentum_v1": "近期动量",
    "ensemble_v2": "集成投票",
    "pattern_mined_v1": "规律挖掘",
}
STRATEGY_IDS = ["balanced_v1", "hot_v1", "cold_rebound_v1", "momentum_v1", "ensemble_v2", "pattern_mined_v1"]
ALL_NUMBERS = list(range(1, 50))
MINED_CONFIG_KEY = "mined_strategy_config_v1"


@dataclass
class DrawRecord:
    issue_no: str
    draw_date: str      # 日期字符串 YYYY-MM-DD
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
            issue_no TEXT PRIMARY KEY,
            draw_date TEXT NOT NULL,
            numbers_json TEXT NOT NULL,
            special_number INTEGER NOT NULL,
            source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS prediction_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_no TEXT NOT NULL,
            strategy TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            hit_count INTEGER, hit_rate REAL,
            hit_count_10 INTEGER, hit_rate_10 REAL,
            hit_count_14 INTEGER, hit_rate_14 REAL,
            hit_count_20 INTEGER, hit_rate_20 REAL,
            special_hit INTEGER,
            created_at TEXT NOT NULL, reviewed_at TEXT,
            UNIQUE(issue_no, strategy)
        );
        CREATE TABLE IF NOT EXISTS prediction_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pick_type TEXT NOT NULL DEFAULT 'MAIN',
            number INTEGER NOT NULL,
            rank INTEGER NOT NULL,
            score REAL NOT NULL,
            reason TEXT NOT NULL,
            UNIQUE(run_id, number)
        );
        CREATE TABLE IF NOT EXISTS prediction_pools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            pool_size INTEGER NOT NULL,
            numbers_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(run_id, pool_size)
        );
        CREATE TABLE IF NOT EXISTS model_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.commit()


# ====================== 数据获取（多源适配） ======================
def _parse_marksix6_response(payload: dict) -> List[DrawRecord]:
    """解析 marksix6.net 的返回数据"""
    records = []
    hk_data = None
    for lottery in payload.get("lottery_data", []):
        if lottery.get("name") == "香港彩":
            hk_data = lottery
            break
    if not hk_data:
        return records

    # 获取最新开奖时间，用于倒推历史日期
    latest_open_time_str = hk_data.get("openTime", "")
    try:
        latest_open_time = datetime.strptime(latest_open_time_str, "%Y-%m-%d %H:%M:%S")
    except:
        latest_open_time = datetime.now()

    history = hk_data.get("history", [])
    # 历史记录倒序（最新在前）
    for idx, item in enumerate(history):
        try:
            # 格式: "2026054 期：07,13,15,05,31,27,16"
            parts = item.split("期：")
            if len(parts) != 2:
                continue
            issue_no = parts[0].strip()
            all_numbers = [int(n.strip()) for n in parts[1].split(",")]
            if len(all_numbers) != 7:
                continue
            numbers = all_numbers[:6]
            special = all_numbers[6]

            # 日期推算：最新一期使用 openTime，历史期按 2 天间隔递减
            # 六合彩每周二、四、六开奖，平均间隔 2 天
            draw_date = (latest_open_time - timedelta(days=idx * 2)).strftime("%Y-%m-%d")

            records.append(DrawRecord(issue_no, draw_date, numbers, special))
        except:
            continue
    return records


def _parse_official_response(payload: list) -> List[DrawRecord]:
    """解析香港赛马会官网的返回数据（旧版）"""
    records = []
    for item in payload:
        try:
            issue_no = str(item.get("drawNo") or item.get("issueNo"))
            draw_date = str(item.get("drawDate", ""))[:10]
            numbers = [int(item[f"no{i}"]) for i in range(1, 7)]
            special = int(item.get("specialNumber") or item.get("no7"))

            if issue_no and draw_date and len(numbers) == 6:
                records.append(DrawRecord(issue_no, draw_date, numbers, special))
        except:
            continue
    return records


def fetch_official_records() -> List[DrawRecord]:
    """尝试从多个数据源获取六合彩记录"""
    # 1. 尝试 marksix6.net
    print(f"正在从 marksix6.net 获取数据...")
    try:
        req = Request(HK_API_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; marksix-local/2.0)"})
        with urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            records = _parse_marksix6_response(payload)
            if records:
                print(f"✅ marksix6 获取成功，共 {len(records)} 期")
                return sorted(records, key=lambda x: x.issue_no, reverse=True)
    except Exception as e:
        print(f"⚠️ marksix6 获取失败: {e}")

    # 2. 回退到赛马会官网
    print(f"正在从香港赛马会官网获取数据...")
    try:
        req = Request(OFFICIAL_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; marksix-local/1.0)"})
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8-sig"))
            records = _parse_official_response(payload)
            if records:
                print(f"✅ 官网获取成功，共 {len(records)} 期")
                return sorted(records, key=lambda x: x.issue_no)
    except Exception as e:
        print(f"❌ 官网获取失败: {e}")

    print("❌ 所有数据源均无法获取数据")
    return []


# ====================== 数据库操作 ======================
def upsert_draw(conn: sqlite3.Connection, record: DrawRecord) -> str:
    now = utc_now()
    if conn.execute("SELECT 1 FROM draws WHERE issue_no=?", (record.issue_no,)).fetchone():
        conn.execute(
            "UPDATE draws SET draw_date=?, numbers_json=?, special_number=?, updated_at=? WHERE issue_no=?",
            (record.draw_date, json.dumps(record.numbers), record.special_number, now, record.issue_no)
        )
        return "updated"
    else:
        conn.execute(
            "INSERT INTO draws VALUES (?, ?, ?, ?, ?, ?, ?)",
            (record.issue_no, record.draw_date, json.dumps(record.numbers), record.special_number, "official", now, now)
        )
        return "inserted"


def sync_online(conn: sqlite3.Connection) -> Tuple[int, int, int]:
    records = fetch_official_records()
    if not records:
        # 没有新数据，检查数据库是否有历史数据
        existing = conn.execute("SELECT COUNT(*) FROM draws").fetchone()[0]
        if existing == 0:
            raise RuntimeError("无网络且本地无数据，请检查网络或手动导入数据")
        print("⚠️ 未获取到新数据，使用本地历史数据继续。")
        return 0, 0, 0

    inserted = updated = 0
    for r in records:
        res = upsert_draw(conn, r)
        if res == "inserted":
            inserted += 1
        else:
            updated += 1
    conn.commit()
    return len(records), inserted, updated


# ====================== 核心预测逻辑（保持不变） ======================
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
            freq[n] += 1
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


def _pick_top_six(scores: Dict[int, float], reason: str) -> List[Tuple[int, int, float, str]]:
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = []
    for n, s in ranked:
        if len(picked) == 6:
            break
        picked.append((n, s))
    return [(n, idx + 1, s, f"{reason} score={s:.4f}") for idx, (n, s) in enumerate(picked)]


def _default_mined_config() -> Dict[str, float]:
    return {"window": 80.0, "w_freq": 0.40, "w_omit": 0.30, "w_mom": 0.20, "w_pair": 0.05, "w_zone": 0.05, "special_bonus": 0.10}


def _apply_weight_config(draws: List[List[int]], config: Dict[str, float], reason: str):
    window = draws[:int(config.get("window", 80))]
    freq = _normalize(_freq_map(window))
    omission = _normalize(_omission_map(window))
    momentum = _normalize(_momentum_map(window))

    scores = {}
    for n in ALL_NUMBERS:
        scores[n] = (
            freq[n] * config.get("w_freq", 0.4) +
            omission[n] * config.get("w_omit", 0.3) +
            momentum[n] * config.get("w_mom", 0.2)
        )
    main_picks = _pick_top_six(scores, reason)
    main_set = {n for n, _, _, _ in main_picks}
    special_candidates = sorted([(n, s) for n, s in scores.items() if n not in main_set], key=lambda x: x[1], reverse=True)
    special = special_candidates[0][0] if special_candidates else 1
    return main_picks, special, special_candidates[0][1] if special_candidates else 0.0, scores


def generate_strategy(draws: List[List[int]], strategy: str, mined_config=None):
    if strategy == "hot_v1":
        return _apply_weight_config(draws, {"window": 80, "w_freq": 0.8, "w_omit": 0.0, "w_mom": 0.2}, "热号")
    if strategy == "cold_rebound_v1":
        return _apply_weight_config(draws, {"window": 80, "w_freq": 0.0, "w_omit": 0.7, "w_mom": 0.3}, "冷号")
    if strategy == "momentum_v1":
        return _apply_weight_config(draws, {"window": 80, "w_freq": 0.1, "w_omit": 0.0, "w_mom": 0.9}, "动量")
    if strategy == "pattern_mined_v1":
        cfg = mined_config or _default_mined_config()
        return _apply_weight_config(draws, cfg, "规律挖掘")
    # 默认 balanced_v1 / ensemble_v2
    return _apply_weight_config(draws, {"window": 80, "w_freq": 0.4, "w_omit": 0.3, "w_mom": 0.2}, "平衡")


def load_recent_draws(conn: sqlite3.Connection, limit: int = 200) -> List[List[int]]:
    rows = conn.execute("SELECT numbers_json FROM draws ORDER BY draw_date DESC LIMIT ?", (limit,)).fetchall()
    return [json.loads(r["numbers_json"]) for r in rows]


def get_model_state(conn, key):
    row = conn.execute("SELECT value FROM model_state WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def set_model_state(conn, key, value):
    now = utc_now()
    conn.execute("INSERT OR REPLACE INTO model_state VALUES (?, ?, ?)", (key, json.dumps(value), now))


def ensure_mined_pattern_config(conn):
    cfg = get_model_state(conn, MINED_CONFIG_KEY)
    if cfg:
        return cfg
    cfg = _default_mined_config()
    set_model_state(conn, MINED_CONFIG_KEY, cfg)
    return cfg


def next_issue_number(current_issue: str) -> str:
    """从当期期号推算下一期，兼容 2026054 或 24/032 格式"""
    digits = ''.join(ch for ch in current_issue if ch.isdigit())
    if not digits:
        return f"{current_issue}-NEXT"
    num = int(digits) + 1
    if '/' in current_issue:
        parts = current_issue.rsplit('/', 1)
        return f"{parts[0]}/{num:0{len(digits)}d}"
    return f"{num:0{len(digits)}d}"


def generate_and_store_predictions(conn: sqlite3.Connection) -> str:
    """生成下一期所有策略的预测，并持久化到数据库"""
    latest = conn.execute("SELECT issue_no FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    if not latest:
        raise RuntimeError("No draw data yet")
    target_issue = next_issue_number(latest["issue_no"])

    draws = load_recent_draws(conn)
    mined_cfg = ensure_mined_pattern_config(conn)
    now = utc_now()

    for strategy in STRATEGY_IDS:
        # 跳过已存在的预测
        exist = conn.execute(
            "SELECT id FROM prediction_runs WHERE issue_no=? AND strategy=?",
            (target_issue, strategy)
        ).fetchone()
        if exist:
            continue

        picks, special, special_score, _ = generate_strategy(draws, strategy, mined_cfg)

        # 插入 prediction_run
        cur = conn.execute(
            "INSERT INTO prediction_runs (issue_no, strategy, status, created_at) VALUES (?, ?, 'PENDING', ?)",
            (target_issue, strategy, now)
        )
        run_id = cur.lastrowid

        # 插入主号 (rank 1-6)
        for num, rank, score, reason in picks:
            conn.execute(
                "INSERT INTO prediction_picks (run_id, pick_type, number, rank, score, reason) VALUES (?, 'MAIN', ?, ?, ?, ?)",
                (run_id, num, rank, score, reason)
            )

        # 插入特别号 (rank 7)
        conn.execute(
            "INSERT INTO prediction_picks (run_id, pick_type, number, rank, score, reason) VALUES (?, 'SPECIAL', ?, 7, ?, ?)",
            (run_id, special, special_score, f"{STRATEGY_LABELS.get(strategy, strategy)} 特别号")
        )

        # 插入预测池（主号6个）
        main_numbers = sorted([n for n, _, _, _ in picks])
        conn.execute(
            "INSERT INTO prediction_pools (run_id, pool_size, numbers_json, created_at) VALUES (?, 6, ?, ?)",
            (run_id, json.dumps(main_numbers), now)
        )

    conn.commit()
    return target_issue


def print_latest_result(conn: sqlite3.Connection):
    """打印最新开奖信息"""
    latest = conn.execute("SELECT * FROM draws ORDER BY draw_date DESC LIMIT 1").fetchone()
    if latest:
        nums = " ".join(f"{n:02d}" for n in json.loads(latest["numbers_json"]))
        print(f"\n最新开奖: {latest['issue_no']} | {nums} + {latest['special_number']:02d}")
    else:
        print("\n暂无开奖数据")


def print_predictions(conn: sqlite3.Connection):
    """打印下一期的所有预测"""
    latest_issue = conn.execute(
        "SELECT issue_no FROM prediction_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not latest_issue:
        print("暂无预测，请先执行 sync 同步数据")
        return

    runs = conn.execute(
        "SELECT * FROM prediction_runs WHERE issue_no=? ORDER BY strategy",
        (latest_issue["issue_no"],)
    ).fetchall()

    print(f"\n预测期号: {latest_issue['issue_no']}")
    for run in runs:
        picks = conn.execute(
            "SELECT number, pick_type, rank FROM prediction_picks WHERE run_id=? ORDER BY rank",
            (run["id"],)
        ).fetchall()
        main_nums = [str(p["number"]).zfill(2) for p in picks if p["pick_type"] == "MAIN"]
        special = next((str(p["number"]).zfill(2) for p in picks if p["pick_type"] == "SPECIAL"), "--")
        label = STRATEGY_LABELS.get(run["strategy"], run["strategy"])
        print(f"  {label:　<8s}: {' '.join(main_nums)} + {special}")


# ====================== 命令行处理 ======================
def cmd_sync(args):
    conn = connect_db(args.db)
    try:
        init_db(conn)
        total, ins, upd = sync_online(conn)
        if total > 0:
            print(f"\n✅ 同步完成！共 {total} 期 (新增{ins} / 更新{upd})")
        print_latest_result(conn)

        print("\n正在生成新一期预测...")
        target = generate_and_store_predictions(conn)
        print_predictions(conn)
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    finally:
        conn.close()


def cmd_show(args):
    conn = connect_db(args.db)
    try:
        print_latest_result(conn)
        print_predictions(conn)
    finally:
        conn.close()


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--db", default=DB_PATH_DEFAULT, help="数据库路径")

    p = argparse.ArgumentParser(description="香港六合彩线上预测工具", parents=[parent_parser])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sync", parents=[parent_parser], help="同步最新开奖数据并预测").set_defaults(func=cmd_sync)
    sub.add_parser("show", parents=[parent_parser], help="显示最新开奖和已有预测").set_defaults(func=cmd_show)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()