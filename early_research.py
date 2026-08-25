# -*- coding: utf-8 -*-
"""
Ver7.1 EARLY 研究版 改良版
============================================================
2026-08-25 修正
1. J-Quants 429 Rate limit対策
   - APIリクエスト間隔を自動制御
   - 429時はRetry-After / 指数バックオフで自動再試行
   - 一部銘柄が429になっても後で再試行
2. JPX公開信用残の取得改善
   - ページ内のPDFを「最後の1個」に決め打ちしない
   - PDF候補を順番に解析し、100銘柄以上抽出できる信用残PDFを採用
   - 日付をPDF本文から取得
   - 取得失敗時には診断CSVを保存
3. 信用データは列仕様を目視確認するまで正式ENTRY条件に使用しない
4. 現行Ver7.0は変更しない
============================================================
"""

from __future__ import annotations

import os
import re
import io
import time
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pdfplumber

MODEL_VERSION = "7.1-EARLY-RESEARCH-R2"

API = "https://api.jquants.com/v2"
JPX_MARGIN = "https://www.jpx.co.jp/markets/statistics-equities/margin/05.html"
KEY = os.getenv("JQUANTS_API_KEY", "").strip()

BASE = Path(__file__).resolve().parent
DATA = BASE / "data_early"
REPORT = BASE / "reports_early"
DATA.mkdir(exist_ok=True)
REPORT.mkdir(exist_ok=True)

SIGNALS = DATA / "early_signals.csv"
OUTCOMES = DATA / "early_outcomes.csv"
MARGINS = DATA / "jpx_margin_weekly.csv"
MARGIN_DIAG = DATA / "jpx_margin_diagnostics.csv"
LATEST = REPORT / "early_latest.csv"

# Lightプランの429回避用。1分あたりの上限が変わっても
# 429 Retryで吸収できるようにする。
JQ_MIN_INTERVAL_SEC = float(os.getenv("JQ_MIN_INTERVAL_SEC", "1.25"))
JQ_MAX_RETRIES = 6
JQ_BACKOFF_BASE_SEC = 12

CODES = [
"1332","1605","1721","1801","1802","1803","1808","1812","1925","1928","1963","2002",
"2269","2282","2501","2502","2503","2801","2802","2871","2914","3101","3103","3401",
"3402","3405","3407","3861","4004","4005","4021","4042","4043","4061","4063","4183",
"4188","4208","4452","4631","4901","4911","6988","4151","4502","4503","4506","4507",
"4519","4523","4568","4578","5019","5020","5101","5108","5201","5214","5232","5233",
"5301","5332","5333","5401","5406","5411","3436","5706","5711","5713","5714","5801",
"5802","5803","6103","6113","6301","6302","6305","6326","6361","6367","6471","6472",
"6473","7004","7011","7012","7013","6501","6503","6504","6506","6526","6594","6645",
"6701","6702","6723","6724","6752","6753","6758","6762","6770","6841","6857","6861",
"6902","6920","6954","6971","6976","6981","7735","7751","7752","8035","285A","7201",
"7202","7203","7205","7211","7261","7267","7269","7270","4543","7731","7733","7741",
"7762","7832","7911","7912","7951","7974","8001","8002","8015","8031","8053","8058",
"3086","3092","3099","3382","7453","7532","8233","8252","8267","9843","9983","8306",
"8308","8309","8316","8331","8354","8411","7186","8253","8591","8601","8604","8628",
"8697","8725","8750","8766","8795","3289","8801","8802","8804","8830","9001","9005",
"9007","9008","9009","9020","9021","9022","9023","9064","9147","9101","9104","9107",
"9201","9202","9301","9432","9433","9434","9613","9984","9501","9502","9503","9531",
"9532","2413","2432","3659","4324","4689","4704","4751","4755","6098","6178","9602",
"9735","9766"]


def sf(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def pct(a, b):
    a, b = sf(a), sf(b)
    return (a / b - 1) * 100 if not pd.isna(a) and not pd.isna(b) and b else np.nan


def save(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def load(path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame()


def ncode(x):
    s = re.sub(r"\.0$", "", str(x).strip())
    return s[:4] if len(s) == 5 and s.endswith("0") else s


class JQ:
    def __init__(self, key):
        if not key:
            raise RuntimeError("GitHub Secret JQUANTS_API_KEY が未設定です")
        self.s = requests.Session()
        self.s.headers["x-api-key"] = key
        self.last_request_at = 0.0
        self.rate_limit_waits = 0

    def _pace(self):
        elapsed = time.monotonic() - self.last_request_at
        if elapsed < JQ_MIN_INTERVAL_SEC:
            time.sleep(JQ_MIN_INTERVAL_SEC - elapsed)

    def _request(self, ep, params):
        last_error = None

        for attempt in range(JQ_MAX_RETRIES):
            self._pace()

            r = self.s.get(API + ep, params=params, timeout=40)
            self.last_request_at = time.monotonic()

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                self.rate_limit_waits += 1

                retry_after = r.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else None
                except Exception:
                    wait = None

                if wait is None:
                    wait = min(
                        JQ_BACKOFF_BASE_SEC * (2 ** attempt),
                        120
                    )

                print(
                    f"[J-Quants] 429 Rate limit。"
                    f" {wait:.0f}秒待機して再試行 "
                    f"({attempt+1}/{JQ_MAX_RETRIES})"
                )
                time.sleep(wait)
                last_error = f"429: {r.text[:300]}"
                continue

            if 500 <= r.status_code < 600:
                wait = min(5 * (2 ** attempt), 60)
                print(
                    f"[J-Quants] {r.status_code}。"
                    f" {wait}秒後に再試行"
                )
                time.sleep(wait)
                last_error = f"{r.status_code}: {r.text[:300]}"
                continue

            raise RuntimeError(
                f"J-Quants {r.status_code}: {r.text[:300]}"
            )

        raise RuntimeError(
            "J-Quants再試行上限に到達: " + str(last_error)
        )

    def get(self, ep, params=None):
        p = dict(params or {})
        out = []

        while True:
            r = self._request(ep, p)
            d = r.json()
            out += d.get("data", [])

            k = d.get("pagination_key")
            if not k:
                break

            p["pagination_key"] = k

        return pd.DataFrame(out)

    def master(self):
        return self.get("/equities/master")

    def bars(self, code, fr, to):
        return self.get(
            "/equities/bars/daily",
            {"code": code, "from": fr, "to": to}
        )


def col(df, names, req=True):
    for n in names:
        if n in df.columns:
            return n
    if req:
        raise KeyError(
            f"列が見つかりません {names}; actual={list(df.columns)}"
        )
    return None


def std_master(df):
    c = col(df, ["Code", "code"])
    n = col(df, ["CompanyName", "CoName", "Name"], False)

    return pd.DataFrame({
        "code": df[c].map(ncode),
        "name": df[n].astype(str) if n else df[c].map(ncode)
    }).drop_duplicates("code")


def std_bars(df):
    if df.empty:
        return df

    d = col(df, ["Date", "date"])
    c = col(df, ["Code", "code"])
    C = col(df, ["AdjustmentClose", "AdjClose", "Close", "C"])
    H = col(df, ["AdjustmentHigh", "AdjHigh", "High", "H"])
    L = col(df, ["AdjustmentLow", "AdjLow", "Low", "L"])
    V = col(df, ["AdjustmentVolume", "AdjVolume", "Volume", "Vo"])
    T = col(df, ["TurnoverValue", "Turnover", "Value", "Va"], False)

    z = pd.DataFrame({
        "date": pd.to_datetime(df[d]),
        "code": df[c].map(ncode),
        "close": pd.to_numeric(df[C], errors="coerce"),
        "high": pd.to_numeric(df[H], errors="coerce"),
        "low": pd.to_numeric(df[L], errors="coerce"),
        "volume": pd.to_numeric(df[V], errors="coerce"),
    })

    z["turnover"] = (
        pd.to_numeric(df[T], errors="coerce")
        if T else np.nan
    )

    return z.dropna(subset=["date", "close"]).sort_values("date")


# ============================================================
# JPX信用残
# ============================================================

def _extract_attachment_candidates(html_text):
    """
    <a>タグだけでなく、生HTML中の -att/ ファイルも拾う。
    """
    soup = BeautifulSoup(html_text, "html.parser")
    candidates = []

    # 通常リンク
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        full = urljoin(JPX_MARGIN, href)

        if re.search(r"\.(pdf|xlsx?|csv)(?:\?|$)", full, re.I):
            parent_text = ""
            try:
                parent_text = a.parent.get_text(" ", strip=True)
            except Exception:
                pass

            candidates.append({
                "url": full,
                "label": a.get_text(" ", strip=True),
                "context": parent_text
            })

    # JSや特殊HTMLでリンクが通常のaタグに乗らない場合
    raw_links = re.findall(
        r'["\']([^"\']*(?:-att/|/att/)[^"\']+?\.(?:pdf|xlsx?|csv)(?:\?[^"\']*)?)["\']',
        html_text,
        flags=re.I
    )

    for href in raw_links:
        candidates.append({
            "url": urljoin(JPX_MARGIN, href),
            "label": "",
            "context": ""
        })

    # 重複除去
    seen = set()
    out = []

    for x in candidates:
        if x["url"] not in seen:
            seen.add(x["url"])
            out.append(x)

    return out


def _extract_asof_date(text):
    patterns = [
        r"(20\d{2})[/.-](\d{1,2})[/.-](\d{1,2})",
        r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日",
    ]

    found = []

    for ptn in patterns:
        for m in re.finditer(ptn, text):
            try:
                d = pd.Timestamp(
                    year=int(m.group(1)),
                    month=int(m.group(2)),
                    day=int(m.group(3))
                )
                found.append(d)
            except Exception:
                pass

    return max(found) if found else pd.NaT


def _parse_margin_pdf(pdf_bytes, url, context=""):
    text = ""

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join(
            (p.extract_text() or "")
            for p in pdf.pages
        )

    asof = _extract_asof_date(text + "\n" + context)

    rows = []

    for line in text.splitlines():
        # 行内のどこかに4桁コードがあるケースにも対応
        code_match = re.search(r"(?<!\d)(\d{4})(?!\d)", line)
        if not code_match:
            continue

        code = code_match.group(1)

        # コード以降を優先的に数値化
        rest = line[code_match.end():]
        nums = re.findall(
            r"(?:▲|-)?[\d,]+(?:\.\d+)?",
            rest
        )

        clean = []
        for x in nums:
            try:
                sign = -1 if x.startswith("▲") else 1
                x = x.lstrip("▲")
                clean.append(sign * float(x.replace(",", "")))
            except Exception:
                pass

        # 信用残PDFなら複数の数値列があるはず。
        if len(clean) < 2:
            continue

        rows.append({
            "margin_date": (
                asof.strftime("%Y-%m-%d")
                if not pd.isna(asof) else ""
            ),
            "code": code,

            # ここは列仕様の目視確認まで candidate 扱い
            "margin_sell_candidate": clean[-2],
            "margin_buy_candidate": clean[-1],

            "source_url": url,
            "raw_line": line[:700],
        })

    return pd.DataFrame(rows), text


def margin_latest():
    """
    JPXページ内のPDF候補を複数試し、
    100銘柄以上抽出できたPDFのうち日付が新しいものを採用。
    """
    diagnostics = []

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        }

        page = requests.get(
            JPX_MARGIN,
            headers=headers,
            timeout=40
        )
        page.raise_for_status()

        candidates = _extract_attachment_candidates(page.text)

        print(
            f"[JPX] 添付ファイル候補: {len(candidates)}件"
        )

        parsed_options = []

        # ページ中の全部を無制限に取らない
        # PDFを優先し、最大20件まで。
        pdf_candidates = [
            x for x in candidates
            if ".pdf" in x["url"].lower()
        ][:20]

        for i, x in enumerate(pdf_candidates, 1):
            url = x["url"]

            try:
                r = requests.get(
                    url,
                    headers=headers,
                    timeout=45
                )
                r.raise_for_status()

                parsed, text = _parse_margin_pdf(
                    r.content,
                    url,
                    x.get("context", "")
                )

                n_codes = parsed["code"].nunique() if not parsed.empty else 0
                asof = (
                    parsed["margin_date"].iloc[0]
                    if not parsed.empty else ""
                )

                diagnostics.append({
                    "url": url,
                    "context": x.get("context", "")[:200],
                    "parsed_rows": len(parsed),
                    "unique_codes": n_codes,
                    "margin_date": asof,
                    "status": "parsed",
                })

                print(
                    f"[JPX] PDF {i}: "
                    f"{n_codes}銘柄 / date={asof}"
                )

                if n_codes >= 100:
                    parsed_options.append(parsed)

            except Exception as e:
                diagnostics.append({
                    "url": url,
                    "context": x.get("context", "")[:200],
                    "parsed_rows": 0,
                    "unique_codes": 0,
                    "margin_date": "",
                    "status": "error: " + str(e)[:200],
                })

        # 診断は必ず保存
        diag_df = pd.DataFrame(diagnostics)

        if diag_df.empty:
            diag_df = pd.DataFrame([{ 
                "url": "",
                "context": "",
                "parsed_rows": 0,
                "unique_codes": 0,
                "margin_date": "",
                "status": (
                    "JPXページからPDF候補を発見できませんでした。"
                    f" candidates={len(candidates)}"
                )
            }])

        save(diag_df, MARGIN_DIAG)

        if not parsed_options:
            print(
                "[JPX] 信用残として認識できるPDFがありません。"
                " jpx_margin_diagnostics.csv を確認してください。"
            )
            return pd.DataFrame()

        # 日付が取れたものは最新を採用
        def rank_df(df):
            try:
                d = pd.to_datetime(df["margin_date"].iloc[0])
                return d
            except Exception:
                return pd.Timestamp("1900-01-01")

        best = max(parsed_options, key=rank_df).copy()
        best["fetched_at"] = pd.Timestamp.now().isoformat()

        print(
            "[JPX] 信用残PDF採用:",
            best["margin_date"].iloc[0],
            f"{best['code'].nunique()}銘柄"
        )

        return best

    except Exception as e:
        save(pd.DataFrame([{
            "url": JPX_MARGIN,
            "context": "",
            "parsed_rows": 0,
            "unique_codes": 0,
            "margin_date": "",
            "status": "page error: " + str(e)[:300],
        }]), MARGIN_DIAG)

        print(
            "JPX信用残取得失敗（株価分析は継続）:",
            e
        )
        return pd.DataFrame()


def merge_margin_history(new):
    old = load(MARGINS)

    if new.empty:
        return old

    x = (
        pd.concat([old, new], ignore_index=True)
        if not old.empty else new
    )

    # 同じ週・同じ銘柄なら最新取得を残す
    dedup = [
        c for c in ["margin_date", "code"]
        if c in x.columns
    ]

    if dedup:
        x = x.drop_duplicates(dedup, keep="last")

    save(x, MARGINS)
    return x


# ============================================================
# EARLY特徴量
# ============================================================

def features(b):
    if len(b) < 260:
        return None

    b = b.reset_index(drop=True)
    c, h, l, v = b.close, b.high, b.low, b.volume

    cur = float(c.iloc[-1])
    ma5 = c.rolling(5).mean()
    ma25 = c.rolling(25).mean()

    r5 = pct(cur, c.iloc[-6])
    r20 = pct(cur, c.iloc[-21])
    accel = r5 - r20 / 4
    md = pct(cur, ma25.iloc[-1])

    vmean = v.tail(20).mean()
    vr = (
        float(v.iloc[-1] / vmean)
        if vmean and vmean > 0 else np.nan
    )

    hh = float(h.tail(252).max())
    dd = pct(cur, hh)

    peak_idx = int(h.tail(252).idxmax())
    peak_age = len(b) - 1 - peak_idx

    higher_low = (
        float(l.iloc[-5:].min())
        > float(l.iloc[-10:-5].min())
    )

    ma5up = ma5.iloc[-1] > ma5.iloc[-3]

    r10 = (
        (h.iloc[-10:].max() - l.iloc[-10:].min())
        / cur * 100
    )

    prev = float(c.iloc[-30:-10].mean())
    r20prev = (
        (h.iloc[-30:-10].max() - l.iloc[-30:-10].min())
        / prev * 100
    )

    contract = r10 < r20prev

    long_adj = dd <= -15 and peak_age >= 60
    high_base = -15 <= dd <= -3 and peak_age >= 20

    not_hot = (
        r5 <= 6
        and r20 <= 10
        and abs(md) <= 5
    )

    early = (
        not_hot
        and r5 > 0
        and accel > 0
        and ma5up
        and higher_low
        and vr >= 1.10
        and (long_adj or high_base)
    )

    confirmed = early and md >= 0

    evidence = sum([
        bool(r5 > 0),
        bool(accel > 0),
        bool(ma5up),
        bool(higher_low),
        bool(vr >= 1.10),
        bool(abs(md) <= 3),
        bool(contract),
        bool(long_adj or high_base),
    ])

    grade = (
        "A" if evidence >= 7
        else "B" if evidence >= 6
        else "C" if evidence >= 5
        else "D" if evidence >= 4
        else "E"
    )

    return {
        "date": b.date.iloc[-1],
        "close": cur,
        "return_5d": r5,
        "return_20d": r20,
        "momentum_acceleration": accel,
        "ma25_distance": md,
        "volume_ratio_20d": vr,
        "drawdown_252d_high": dd,
        "peak_age": peak_age,
        "higher_low": higher_low,
        "ma5_rising": ma5up,
        "volatility_contracting": contract,
        "long_adjustment": long_adj,
        "high_base": high_base,
        "early_price_setup": early,
        "confirm_setup": confirmed,
        "early_grade": grade,
    }


def margin_features(row, mh):
    row.update({
        "margin_data_available": False,
        "margin_date": "",
        "margin_buy_latest": np.nan,
        "margin_sell_latest": np.nan,
        "margin_buy_change_4obs_pct": np.nan,
        "margin_buy_change_13obs_pct": np.nan,
    })

    if mh.empty:
        return row

    s = mh[
        mh.code.astype(str) == str(row["code"])
    ].copy()

    if s.empty:
        return row

    if "margin_date" in s.columns:
        s["margin_date_dt"] = pd.to_datetime(
            s["margin_date"],
            errors="coerce"
        )
        s = s.sort_values("margin_date_dt")
    elif "fetched_at" in s.columns:
        s["fetched_at"] = pd.to_datetime(
            s["fetched_at"],
            errors="coerce"
        )
        s = s.sort_values("fetched_at")

    buys = pd.to_numeric(
        s.margin_buy_candidate,
        errors="coerce"
    ).dropna()

    if buys.empty:
        return row

    row["margin_data_available"] = True
    row["margin_buy_latest"] = float(buys.iloc[-1])

    if "margin_date" in s.columns:
        row["margin_date"] = str(
            s["margin_date"].iloc[-1]
        )

    sells = pd.to_numeric(
        s.margin_sell_candidate,
        errors="coerce"
    ).dropna()

    if not sells.empty:
        row["margin_sell_latest"] = float(sells.iloc[-1])

    # 4観測前 = 約4週
    if len(buys) >= 4:
        row["margin_buy_change_4obs_pct"] = pct(
            buys.iloc[-1],
            buys.iloc[-4]
        )

    # 13観測前 = 約13週
    if len(buys) >= 13:
        row["margin_buy_change_13obs_pct"] = pct(
            buys.iloc[-1],
            buys.iloc[-13]
        )

    return row


def strategy(r):
    if r["confirm_setup"]:
        return "B_EARLY_CONFIRMED"
    if r["early_price_setup"]:
        return "A_EARLY"
    return "NONE"


def reason(r):
    a = []

    if r["long_adjustment"]:
        a.append("長期調整後")
    if r["high_base"]:
        a.append("高値持ち合い後")
    if r["higher_low"]:
        a.append("安値切り上げ")
    if r["ma5_rising"]:
        a.append("5日線上向き")
    if r["volume_ratio_20d"] >= 1.10:
        a.append(f"出来高{r['volume_ratio_20d']:.1f}倍")
    if r["momentum_acceleration"] > 0:
        a.append("勢い改善")
    if abs(r["ma25_distance"]) <= 3:
        a.append("25日線付近")

    # 信用データは目視検証済みになってから
    # 正式理由に格上げする予定。
    if (
        r["margin_data_available"]
        and not pd.isna(r["margin_buy_change_13obs_pct"])
        and r["margin_buy_change_13obs_pct"] < 0
    ):
        a.append("信用買い残減少(研究値)")

    return " / ".join(a)


def append_signals(today):
    old = load(SIGNALS)
    x = (
        pd.concat([old, today], ignore_index=True)
        if not old.empty else today
    )

    x["date"] = pd.to_datetime(x.date, errors="coerce")
    x = (
        x.sort_values("date")
        .drop_duplicates(
            ["date", "code", "model_version"],
            keep="last"
        )
    )
    x["date"] = x.date.dt.strftime("%Y-%m-%d")

    save(x, SIGNALS)
    return x


def outcomes(signals, cache):
    old = load(OUTCOMES)
    rows = [] if old.empty else old.to_dict("records")

    sig = signals[
        signals.strategy.isin(
            ["A_EARLY", "B_EARLY_CONFIRMED"]
        )
    ].copy()

    sig["date"] = pd.to_datetime(
        sig.date,
        errors="coerce"
    )

    for _, s in sig.iterrows():
        key = (
            s.date.strftime("%Y-%m-%d"),
            str(s.code),
            str(s.strategy)
        )

        tar = next((
            r for r in rows
            if (
                str(r.get("signal_date", ""))[:10],
                str(r.get("code", "")),
                str(r.get("strategy", ""))
            ) == key
        ), None)

        if tar is None:
            tar = {
                "signal_date": key[0],
                "code": key[1],
                "name": s["name"],
                "strategy": key[2],
                "entry_price": s["close"],
                "model_version": MODEL_VERSION
            }
            rows.append(tar)

        b = cache.get(str(s.code))
        if b is None:
            continue

        f = b[b.date > s.date].sort_values("date")
        entry = sf(tar["entry_price"])

        for n in (5, 10, 20, 60):
            if len(f) < n:
                continue

            p = f.iloc[:n]
            tar[f"return_{n}d"] = pct(
                p.close.iloc[-1], entry
            )
            tar[f"mfe_{n}d"] = pct(
                p.high.max(), entry
            )
            tar[f"mae_{n}d"] = pct(
                p.low.min(), entry
            )
            tar[f"exit_date_{n}d"] = (
                p.date.iloc[-1].strftime("%Y-%m-%d")
            )

    z = pd.DataFrame(rows)
    if not z.empty:
        save(z, OUTCOMES)

    return z


def main():
    print("=" * 72)
    print("Ver7.1 EARLY研究版 R2")
    print("429対策 + JPX信用残取得改善")
    print("=" * 72)

    jq = JQ(KEY)

    print("[1] J-Quants銘柄マスタ")
    m = std_master(jq.master())
    names = dict(zip(m.code, m.name))

    print("[2] JPX公開信用残")
    latest_margin = margin_latest()
    mh = merge_margin_history(latest_margin)

    print(
        "[JPX] 保存済信用履歴:",
        0 if mh.empty else len(mh),
        "行"
    )

    today = (
        pd.Timestamp.now(tz="Asia/Tokyo")
        .tz_localize(None)
        .normalize()
    )

    fr = (
        today - pd.Timedelta(days=550)
    ).strftime("%Y-%m-%d")
    to = today.strftime("%Y-%m-%d")

    rows = []
    cache = {}
    errs = []

    print("[3] 日経225 EARLY分析")

    for i, code in enumerate(CODES, 1):
        print(f"{i}/{len(CODES)} {code}")

        try:
            b = std_bars(
                jq.bars(code, fr, to)
            )

            f = features(b)
            if f is None:
                raise RuntimeError("履歴不足")

            r = {
                "date": pd.Timestamp(
                    f["date"]
                ).strftime("%Y-%m-%d"),
                "model_version": MODEL_VERSION,
                "code": code,
                "name": names.get(code, code),
                **f
            }

            r = margin_features(r, mh)
            r["strategy"] = strategy(r)
            r["reason"] = reason(r)

            rows.append(r)
            cache[code] = b

        except Exception as e:
            errs.append(f"{code}: {e}")

    z = pd.DataFrame(rows)

    if z.empty:
        raise RuntimeError("分析結果なし")

    pri = {
        "B_EARLY_CONFIRMED": 2,
        "A_EARLY": 1,
        "NONE": 0
    }
    gr = {
        "A": 5,
        "B": 4,
        "C": 3,
        "D": 2,
        "E": 1
    }

    z["_p"] = z.strategy.map(pri)
    z["_g"] = z.early_grade.map(gr)

    z = (
        z.sort_values(
            [
                "_p", "_g",
                "return_5d",
                "volume_ratio_20d"
            ],
            ascending=[
                False, False,
                True, False
            ]
        )
        .drop(columns=["_p", "_g"])
    )

    save(z, LATEST)
    allsig = append_signals(z)

    print("[4] Outcome更新")
    outcomes(allsig, cache)

    cand = z[z.strategy != "NONE"]

    print("\n=== 🌱 EARLY候補 ===")

    if cand.empty:
        print("本日は該当なし")
    else:
        c = [
            "strategy",
            "early_grade",
            "name",
            "code",
            "close",
            "return_5d",
            "return_20d",
            "momentum_acceleration",
            "ma25_distance",
            "volume_ratio_20d",
            "drawdown_252d_high",
            "peak_age",
            "margin_data_available",
            "margin_date",
            "margin_buy_change_13obs_pct",
            "reason"
        ]
        print(cand[c].head(30).to_string(index=False))

    print("\n=== 実行品質 ===")
    print(
        f"分析成功: {len(z)} / {len(CODES)}銘柄"
    )
    print(
        f"J-Quants 429待機回数: {jq.rate_limit_waits}"
    )
    print(
        "信用データ取得銘柄:",
        int(z["margin_data_available"].sum())
    )

    print("\n保存:")
    print(LATEST)
    print(SIGNALS)
    print(OUTCOMES)
    print(MARGINS)
    print(MARGIN_DIAG)

    if errs:
        print("\n一部エラー:")
        for x in errs[:40]:
            print("-", x)


if __name__ == "__main__":
    main()
