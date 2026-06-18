
import os
import re
import time
import glob
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

# ============================================================
# Market Radar Core v1.0
# 東証プライム全銘柄版
# 目的：
# 市場平均を上回り始めた「挑戦者」を検知する
# ============================================================

JST = timezone(timedelta(hours=9))

SEND_LINE = True
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
NOTIFY_MODE = os.environ.get("NOTIFY_MODE", "MORNING").upper()

DATA_DIR = "market_radar_data"
HISTORY_DIR = os.path.join(DATA_DIR, "history")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

JPX_LISTED_COMPANY_XLS = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

BENCHMARK_TICKER = "^N225"
YF_CHUNK_SIZE = 80
FUNDAMENTAL_LIMIT = int(os.environ.get("FUNDAMENTAL_LIMIT", "450"))

# スコア配分
# 相対強度 35%
# 相対強度加速 25%
# 利益成長 20%
# 売上成長 10%
# 出来高増加 10%


def now_jst():
    return datetime.now(JST)


def today_str():
    return now_jst().strftime("%Y%m%d")


def safe_float(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def send_line_message(message):
    if not SEND_LINE:
        print(message)
        return

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
        print(message)
        return

    url = "https://api.line.me/v2/bot/message/broadcast"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
    }

    payload = {
        "messages": [
            {
                "type": "text",
                "text": message[:4900],
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload, timeout=20)
    print("LINE送信ステータス:", response.status_code)
    print(response.text)


def normalize_code(code):
    code = str(code).strip()
    code = code.replace(".0", "")
    code = code.upper()
    return code


def is_stock_code(code):
    return bool(re.fullmatch(r"[0-9A-Z]{4}", str(code)))


def get_prime_universe_from_jpx():
    print("JPXから東証プライム銘柄リストを取得します。")

    df = pd.read_excel(JPX_LISTED_COMPANY_XLS)

    code_col = None
    name_col = None
    market_col = None
    industry_col = None

    for col in df.columns:
        c = str(col)
        if "コード" in c:
            code_col = col
        if "銘柄名" in c:
            name_col = col
        if "市場・商品区分" in c:
            market_col = col
        if "33業種区分" in c:
            industry_col = col

    if code_col is None or name_col is None or market_col is None:
        raise Exception("JPXファイルの列名を認識できませんでした。")

    prime = df[df[market_col].astype(str).str.contains("プライム", na=False)].copy()

    prime["コード"] = prime[code_col].apply(normalize_code)
    prime["銘柄名"] = prime[name_col].astype(str)

    if industry_col is not None:
        prime["業種"] = prime[industry_col].astype(str)
    else:
        prime["業種"] = ""

    prime = prime[prime["コード"].apply(is_stock_code)]
    prime = prime[["コード", "銘柄名", "業種"]].drop_duplicates("コード")

    if len(prime) < 1000:
        raise Exception(f"プライム銘柄数が少なすぎます：{len(prime)}")

    print(f"東証プライム銘柄数：{len(prime)}")
    return prime.reset_index(drop=True)


def load_universe():
    local_file = "universe_prime.csv"

    if os.path.exists(local_file):
        print("ローカル universe_prime.csv を読み込みます。")
        df = pd.read_csv(local_file, dtype=str)
        df["コード"] = df["コード"].apply(normalize_code)
        if "銘柄名" not in df.columns:
            df["銘柄名"] = df["コード"]
        if "業種" not in df.columns:
            df["業種"] = ""
        df = df[df["コード"].apply(is_stock_code)]
        return df[["コード", "銘柄名", "業種"]].drop_duplicates("コード").reset_index(drop=True)

    return get_prime_universe_from_jpx()


def yf_download_batch(tickers, period="2y"):
    print(f"価格データ取得：{len(tickers)}銘柄")

    all_data = {}

    for start in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[start:start + YF_CHUNK_SIZE]
        print(f"yfinance chunk {start + 1} - {start + len(chunk)}")

        try:
            data = yf.download(
                chunk,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )

            if data.empty:
                continue

            if len(chunk) == 1:
                all_data[chunk[0]] = data.copy()
            else:
                for ticker in chunk:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            all_data[ticker] = data[ticker].dropna(how="all").copy()
                    except Exception:
                        pass

            time.sleep(1.0)

        except Exception as e:
            print("chunk取得エラー:", e)

    return all_data


def score_relative_strength(rs_1m, rs_3m, rs_6m):
    score = 0

    if not pd.isna(rs_1m):
        if rs_1m >= 10:
            score += 10
        elif rs_1m >= 5:
            score += 7
        elif rs_1m >= 0:
            score += 4
        elif rs_1m <= -10:
            score -= 5

    if not pd.isna(rs_3m):
        if rs_3m >= 25:
            score += 15
        elif rs_3m >= 15:
            score += 12
        elif rs_3m >= 8:
            score += 9
        elif rs_3m >= 0:
            score += 5
        elif rs_3m <= -15:
            score -= 8

    if not pd.isna(rs_6m):
        if rs_6m >= 35:
            score += 10
        elif rs_6m >= 20:
            score += 8
        elif rs_6m >= 10:
            score += 6
        elif rs_6m >= 0:
            score += 3
        elif rs_6m <= -20:
            score -= 5

    return max(0, min(35, score))


def score_relative_strength_acceleration(rs_1m, rs_3m, rs_6m):
    score = 0

    # 直近の強さが中期より強い＝加速
    if not pd.isna(rs_1m) and not pd.isna(rs_3m):
        accel_short = rs_1m - (rs_3m / 3)
        if accel_short >= 10:
            score += 12
        elif accel_short >= 5:
            score += 9
        elif accel_short >= 2:
            score += 6
        elif accel_short < -5:
            score -= 5

    if not pd.isna(rs_3m) and not pd.isna(rs_6m):
        accel_mid = rs_3m - (rs_6m / 2)
        if accel_mid >= 15:
            score += 13
        elif accel_mid >= 8:
            score += 10
        elif accel_mid >= 3:
            score += 6
        elif accel_mid < -8:
            score -= 5

    return max(0, min(25, score))


def score_growth(growth, max_score):
    if pd.isna(growth):
        return 0

    g = growth * 100

    if g >= 80:
        return max_score
    if g >= 50:
        return max_score * 0.85
    if g >= 30:
        return max_score * 0.7
    if g >= 20:
        return max_score * 0.55
    if g >= 10:
        return max_score * 0.4
    if g >= 0:
        return max_score * 0.15
    return -max_score * 0.3


def score_volume(volume_ratio):
    if pd.isna(volume_ratio):
        return 0

    if volume_ratio >= 2.0:
        return 10
    if volume_ratio >= 1.5:
        return 8
    if volume_ratio >= 1.2:
        return 5
    if volume_ratio >= 1.0:
        return 3
    return 0


def classify_stock(total_score, rank_change, rs_accel_score, growth_score, volume_score):
    if total_score >= 80 and rs_accel_score >= 15:
        return "挑戦者"
    if total_score >= 70 and rank_change >= 30:
        return "急浮上"
    if total_score >= 65 and growth_score >= 18:
        return "予兆"
    if total_score >= 60:
        return "監視"
    return "対象外"


def judge_market_temperature(benchmark_close):
    if benchmark_close is None or len(benchmark_close) < 200:
        return 50, "中立"

    current = float(benchmark_close.iloc[-1])
    ma25 = float(benchmark_close.rolling(25).mean().iloc[-1])
    ma75 = float(benchmark_close.rolling(75).mean().iloc[-1])
    ma200 = float(benchmark_close.rolling(200).mean().iloc[-1])

    ret_1m = float((current / benchmark_close.iloc[-21] - 1) * 100)
    ret_3m = float((current / benchmark_close.iloc[-63] - 1) * 100)

    score = 50

    if current > ma25:
        score += 10
    if current > ma75:
        score += 10
    if current > ma200:
        score += 10
    if ma75 > ma200:
        score += 10
    if ret_1m > 0:
        score += 5
    if ret_3m > 0:
        score += 5

    score = max(0, min(100, score))

    if score >= 75:
        label = "強気"
    elif score >= 55:
        label = "中立"
    else:
        label = "弱気"

    return round(score, 1), label


def get_previous_ranking_file():
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "market_radar_*.csv")))

    if not files:
        return None

    today = today_str()
    candidates = []

    for f in files:
        base = os.path.basename(f)
        if today not in base:
            candidates.append(f)

    if not candidates:
        return None

    return candidates[-1]


def add_rank_change(ranking):
    prev_file = get_previous_ranking_file()

    ranking["前回順位"] = np.nan
    ranking["順位変化"] = 0

    if prev_file is None:
        return ranking

    try:
        prev = pd.read_csv(prev_file, dtype={"コード": str})
        prev["コード"] = prev["コード"].apply(normalize_code)

        rank_map = dict(zip(prev["コード"], prev["順位"]))

        ranking["前回順位"] = ranking["コード"].map(rank_map)
        ranking["順位変化"] = ranking.apply(
            lambda r: int(r["前回順位"] - r["順位"])
            if not pd.isna(r["前回順位"])
            else 0,
            axis=1,
        )

    except Exception as e:
        print("順位変化計算エラー:", e)

    return ranking


def get_fundamentals(ticker):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    return {
        "売上成長率": safe_float(info.get("revenueGrowth")),
        "利益成長率": safe_float(info.get("earningsGrowth")),
        "PER": safe_float(info.get("trailingPE")),
        "予想PER": safe_float(info.get("forwardPE")),
        "時価総額": safe_float(info.get("marketCap")),
    }


def update_old_returns(current_price_map):
    files = sorted(glob.glob(os.path.join(HISTORY_DIR, "market_radar_*.csv")))

    if not files:
        return

    today_dt = now_jst().date()
    horizons = [7, 30, 90, 180]

    for file in files:
        try:
            base = os.path.basename(file)
            m = re.search(r"market_radar_(\d{8})_", base)
            if not m:
                continue

            file_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            days_passed = (today_dt - file_date).days

            if days_passed <= 0:
                continue

            df = pd.read_csv(file, dtype={"コード": str})
            changed = False

            for h in horizons:
                col = f"{h}日後リターン%"
                if col not in df.columns:
                    df[col] = np.nan

                if days_passed >= h:
                    mask = df[col].isna()
                    if mask.any():
                        def calc_return(row):
                            code = normalize_code(row["コード"])
                            entry = safe_float(row.get("株価"))
                            now_price = current_price_map.get(code, np.nan)
                            if pd.isna(entry) or entry <= 0 or pd.isna(now_price):
                                return np.nan
                            return round((now_price / entry - 1) * 100, 2)

                        df.loc[mask, col] = df.loc[mask].apply(calc_return, axis=1)
                        changed = True

            if changed:
                df.to_csv(file, index=False, encoding="utf-8-sig")
                print("過去ファイルの成績更新:", file)

        except Exception as e:
            print("過去リターン更新エラー:", file, e)


def analyze():
    universe = load_universe()
    tickers = [code + ".T" for code in universe["コード"].tolist()]

    benchmark = yf.download(
        BENCHMARK_TICKER,
        period="2y",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if benchmark.empty:
        raise Exception("日経平均のデータ取得に失敗しました。")

    benchmark_close = benchmark["Close"].squeeze()

    market_score, market_label = judge_market_temperature(benchmark_close)

    bench_current = float(benchmark_close.iloc[-1])
    bench_ret_1m = float((bench_current / benchmark_close.iloc[-21] - 1) * 100)
    bench_ret_3m = float((bench_current / benchmark_close.iloc[-63] - 1) * 100)
    bench_ret_6m = float((bench_current / benchmark_close.iloc[-126] - 1) * 100)

    price_data = yf_download_batch(tickers, period="2y")

    rows = []
    current_price_map = {}

    for _, urow in universe.iterrows():
        code = normalize_code(urow["コード"])
        ticker = code + ".T"
        name = urow["銘柄名"]
        industry = urow.get("業種", "")

        df = price_data.get(ticker)

        if df is None or df.empty or len(df) < 130:
            continue

        try:
            close = df["Close"].dropna()
            volume = df["Volume"].dropna()

            if len(close) < 130 or len(volume) < 60:
                continue

            current = float(close.iloc[-1])
            current_price_map[code] = current

            ret_1m = float((current / close.iloc[-21] - 1) * 100)
            ret_3m = float((current / close.iloc[-63] - 1) * 100)
            ret_6m = float((current / close.iloc[-126] - 1) * 100)

            rs_1m = ret_1m - bench_ret_1m
            rs_3m = ret_3m - bench_ret_3m
            rs_6m = ret_6m - bench_ret_6m

            avg_vol_20 = float(volume.tail(20).mean())
            avg_vol_60 = float(volume.tail(60).mean())
            volume_ratio = avg_vol_20 / avg_vol_60 if avg_vol_60 > 0 else np.nan

            high_52w = float(close.tail(252).max()) if len(close) >= 252 else float(close.max())
            distance_52w_high = (current / high_52w - 1) * 100 if high_52w > 0 else np.nan

            ma25 = float(close.rolling(25).mean().iloc[-1])
            ma75 = float(close.rolling(75).mean().iloc[-1])
            deviation_25 = (current / ma25 - 1) * 100 if ma25 > 0 else np.nan

            rs_score = score_relative_strength(rs_1m, rs_3m, rs_6m)
            rs_accel_score = score_relative_strength_acceleration(rs_1m, rs_3m, rs_6m)
            vol_score = score_volume(volume_ratio)

            price_base_score = rs_score + rs_accel_score + vol_score

            rows.append({
                "コード": code,
                "銘柄名": name,
                "業種": industry,
                "株価": round(current, 1),
                "1か月%": round(ret_1m, 2),
                "3か月%": round(ret_3m, 2),
                "6か月%": round(ret_6m, 2),
                "日経比1か月%": round(rs_1m, 2),
                "日経比3か月%": round(rs_3m, 2),
                "日経比6か月%": round(rs_6m, 2),
                "52週高値乖離%": round(distance_52w_high, 2),
                "25日乖離%": round(deviation_25, 2),
                "出来高倍率": round(volume_ratio, 2) if not pd.isna(volume_ratio) else np.nan,
                "相対強度点": round(rs_score, 1),
                "相対強度加速点": round(rs_accel_score, 1),
                "出来高点": round(vol_score, 1),
                "価格ベース点": round(price_base_score, 1),
            })

        except Exception as e:
            print(f"{ticker} 分析エラー:", e)

    if not rows:
        raise Exception("ランキング作成に必要なデータがありません。")

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values("価格ベース点", ascending=False).reset_index(drop=True)

    # 価格・相対強度ベースで上位だけファンダ取得
    shortlist = ranking.head(FUNDAMENTAL_LIMIT)["コード"].tolist()
    fundamentals_map = {}

    print(f"ファンダメンタル取得：上位 {len(shortlist)} 銘柄")

    for i, code in enumerate(shortlist, start=1):
        ticker = code + ".T"
        print(f"fundamental {i}/{len(shortlist)} {ticker}")
        fundamentals_map[code] = get_fundamentals(ticker)
        time.sleep(0.2)

    ranking["売上成長率%"] = np.nan
    ranking["利益成長率%"] = np.nan
    ranking["PER"] = np.nan
    ranking["予想PER"] = np.nan
    ranking["時価総額"] = np.nan
    ranking["売上成長点"] = 0.0
    ranking["利益成長点"] = 0.0

    for idx, row in ranking.iterrows():
        code = row["コード"]
        f = fundamentals_map.get(code, {})

        revenue_growth = safe_float(f.get("売上成長率"))
        earnings_growth = safe_float(f.get("利益成長率"))

        revenue_score = score_growth(revenue_growth, 10)
        earnings_score = score_growth(earnings_growth, 20)

        ranking.at[idx, "売上成長率%"] = round(revenue_growth * 100, 2) if not pd.isna(revenue_growth) else np.nan
        ranking.at[idx, "利益成長率%"] = round(earnings_growth * 100, 2) if not pd.isna(earnings_growth) else np.nan
        ranking.at[idx, "PER"] = round(safe_float(f.get("PER")), 1) if not pd.isna(safe_float(f.get("PER"))) else np.nan
        ranking.at[idx, "予想PER"] = round(safe_float(f.get("予想PER")), 1) if not pd.isna(safe_float(f.get("予想PER"))) else np.nan
        ranking.at[idx, "時価総額"] = safe_float(f.get("時価総額"))
        ranking.at[idx, "売上成長点"] = round(revenue_score, 1)
        ranking.at[idx, "利益成長点"] = round(earnings_score, 1)

    ranking["総合点"] = (
        ranking["相対強度点"]
        + ranking["相対強度加速点"]
        + ranking["出来高点"]
        + ranking["売上成長点"]
        + ranking["利益成長点"]
    ).round(1)

    ranking = ranking.sort_values("総合点", ascending=False).reset_index(drop=True)
    ranking["順位"] = ranking.index + 1

    ranking = add_rank_change(ranking)

    ranking["分類"] = ranking.apply(
        lambda r: classify_stock(
            r["総合点"],
            r["順位変化"],
            r["相対強度加速点"],
            r["売上成長点"] + r["利益成長点"],
            r["出来高点"],
        ),
        axis=1,
    )

    ranking["買いゾーン下限"] = (ranking["株価"] * 0.96).round(1)
    ranking["買いゾーン上限"] = (ranking["株価"] * 1.03).round(1)
    ranking["損切り目安"] = (ranking["株価"] * 0.90).round(1)

    ranking["利確目安1"] = (ranking["株価"] * 1.20).round(1)
    ranking["利確目安2"] = (ranking["株価"] * 1.50).round(1)

    ranking["市場温度点"] = market_score
    ranking["市場温度"] = market_label

    for h in [7, 30, 90, 180]:
        col = f"{h}日後リターン%"
        if col not in ranking.columns:
            ranking[col] = np.nan

    mode = NOTIFY_MODE
    date = today_str()

    history_file = os.path.join(HISTORY_DIR, f"market_radar_{date}_{mode}.csv")
    latest_file = os.path.join(OUTPUT_DIR, "market_radar_latest.csv")
    excel_file = os.path.join(OUTPUT_DIR, f"market_radar_{date}_{mode}.xlsx")

    ranking.to_csv(history_file, index=False, encoding="utf-8-sig")
    ranking.to_csv(latest_file, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="総合ランキング", index=False)
        ranking[ranking["分類"] == "挑戦者"].to_excel(writer, sheet_name="挑戦者", index=False)
        ranking[ranking["分類"] == "急浮上"].to_excel(writer, sheet_name="急浮上", index=False)
        ranking[ranking["分類"] == "予兆"].to_excel(writer, sheet_name="予兆", index=False)
        ranking[ranking["分類"] == "監視"].to_excel(writer, sheet_name="監視", index=False)

    update_old_returns(current_price_map)

    print(ranking.head(30).to_string(index=False))
    return ranking, market_score, market_label


def make_morning_message(ranking, market_score, market_label):
    now = now_jst().strftime("%Y-%m-%d %H:%M")

    challengers = ranking[ranking["分類"].isin(["挑戦者", "急浮上", "予兆"])].head(5)
    top = ranking.head(5)
    rising = ranking.sort_values(["順位変化", "総合点"], ascending=[False, False]).head(5)

    msg = f"【Market Radar 7:30 戦略通知 v1.0】\n{now}\n\n"
    msg += f"市場温度：{market_score}点（{market_label}）\n"
    msg += "対象：東証プライム全銘柄\n\n"

    msg += "◆ 挑戦者TOP5\n"
    if challengers.empty:
        msg += "該当なし\n"
    else:
        for _, row in challengers.iterrows():
            msg += f"{int(row['順位'])}位 {row['銘柄名']}（{row['コード']}） {row['分類']}\n"
            msg += f"総合:{row['総合点']} / 日経比3M:{row['日経比3か月%']}% / 加速点:{row['相対強度加速点']}\n"
            msg += f"利益成長:{row['利益成長率%']}% / 出来高:{row['出来高倍率']}倍\n\n"

    msg += "◆ 急浮上TOP5\n"
    for _, row in rising.iterrows():
        if row["順位変化"] <= 0:
            continue
        msg += f"・{row['銘柄名']}（{row['コード']}） 前回比 +{row['順位変化']}位 / 現在{int(row['順位'])}位\n"

    msg += "\n◆ 総合TOP5\n"
    for _, row in top.iterrows():
        msg += f"・{row['銘柄名']}（{row['コード']}）{row['総合点']}点 / {row['分類']}\n"

    msg += "\n※目的は『今日上がる銘柄』ではなく『主役化し始めた挑戦者』の検知。"
    return msg


def make_preopen_message(ranking, market_score, market_label):
    now = now_jst().strftime("%Y-%m-%d %H:%M")

    candidates = ranking[ranking["分類"].isin(["挑戦者", "急浮上", "予兆", "監視"])].head(7)

    msg = f"【Market Radar 8:55 寄り前確認 v1.0】\n{now}\n\n"
    msg += f"市場温度：{market_score}点（{market_label}）\n"
    msg += "※無料データでは板・気配値は未取得。寄り付き前の最終確認用。\n\n"

    for _, row in candidates.iterrows():
        caution = ""

        if row["25日乖離%"] >= 15:
            caution = " / 短期過熱注意"
        elif row["52週高値乖離%"] >= -2:
            caution = " / 高値圏"
        elif row["日経比3か月%"] < 0:
            caution = " / 相対弱め"

        msg += f"・{row['銘柄名']}（{row['コード']}）{row['分類']}{caution}\n"
        msg += f"総合:{row['総合点']} / 順位:{int(row['順位'])}位 / 変化:{row['順位変化']}位\n"
        msg += f"株価:{row['株価']}円 / 買いゾーン:{row['買いゾーン下限']}〜{row['買いゾーン上限']}円\n"
        msg += f"損切り:{row['損切り目安']}円 / 利確:{row['利確目安1']}〜{row['利確目安2']}円\n\n"

    msg += "寄り付きで大幅GUなら飛びつき注意。押し目・出来高・板を確認。"
    return msg


if __name__ == "__main__":
    try:
        ranking, market_score, market_label = analyze()

        if ranking.empty:
            send_line_message("Market Radar分析に失敗しました。ランキングが空です。")
        else:
            if NOTIFY_MODE == "PREOPEN":
                message = make_preopen_message(ranking, market_score, market_label)
            else:
                message = make_morning_message(ranking, market_score, market_label)

            print("=" * 80)
            print("LINE通知メッセージ")
            print("=" * 80)
            print(message)

            send_line_message(message)

    except Exception as e:
        error_msg = f"Market Radarでエラーが発生しました。\n\n{e}"
        print(error_msg)
        send_line_message(error_msg)
