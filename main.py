import os
import re
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta, timezone

SEND_LINE = True
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("NpKOKCACNv/kV1a3zE+KX4j1PpVq2kEMlzBrnfUq4tqS9uAYsWs+KL/dzHYWUVNMX5QHmAml7eoGX7xzXG9Bxmjcb8plQJ8drRR3IhtK0jxZSspn0Sf4CNk3mWe0OJ/hjB8m81HSUo8YVdNss39mqgdB04t89/1O/w1cDnyilFU=", "")
NOTIFY_MODE = os.environ.get("NOTIFY_MODE", "MORNING")

JST = timezone(timedelta(hours=9))

fallback_nikkei225_codes = [
    "1332","1605","1721","1801","1802","1803","1808","1812","1925","1928",
    "1963","2002","2269","2282","2501","2502","2503","2801","2802","2871",
    "2914","3101","3103","3401","3402","3405","3407","3861","4004","4005",
    "4021","4042","4043","4061","4063","4183","4188","4208","4452","4631",
    "4901","4911","6988","4151","4502","4503","4506","4507","4519","4523",
    "4568","4578","5019","5020","5101","5108","5201","5214","5232","5233",
    "5301","5332","5333","5401","5406","5411","3436","5706","5711","5713",
    "5714","5801","5802","5803","6103","6113","6301","6302","6305","6326",
    "6361","6367","6471","6472","6473","7004","7011","7012","7013","6501",
    "6503","6504","6506","6526","6594","6645","6701","6702","6723","6724",
    "6752","6753","6758","6762","6770","6841","6857","6861","6902","6920",
    "6954","6971","6976","6981","7735","7751","7752","8035","285A","7201",
    "7202","7203","7205","7211","7261","7267","7269","7270","4543","7731",
    "7733","7741","7762","7832","7911","7912","7951","7974","8001","8002",
    "8015","8031","8053","8058","3086","3092","3099","3382","7453","7532",
    "8233","8252","8267","9843","9983","8306","8308","8309","8316","8331",
    "8354","8411","7186","8253","8591","8601","8604","8628","8697","8725",
    "8750","8766","8795","3289","8801","8802","8804","8830","9001","9005",
    "9007","9008","9009","9020","9021","9022","9023","9064","9147","9101",
    "9104","9107","9201","9202","9301","9432","9433","9434","9613","9984",
    "9501","9502","9503","9531","9532","2413","2432","3659","4324","4689",
    "4704","4751","4755","6098","6178","9602","9735","9766"
]

semiconductor_ai_codes = [
    "8035","6857","6723","6724","6981","6861","6758","6701",
    "6702","6501","6503","6504","6594","4063","6988","9984","285A"
]

high_dividend_codes = [
    "8306","8316","8308","8309","8411","8591","8058","8001",
    "8002","8031","2914","9432","9433","9434","8766","8750"
]


def get_nikkei225_codes_auto():
    url = "https://ja.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "日経平均株価",
        "prop": "text",
        "format": "json"
    }
    headers = {"User-Agent": "Mozilla/5.0 nikkei225-analysis-tool/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()

    html = r.json()["parse"]["text"]["*"]
    tables = pd.read_html(html)

    codes = []
    for table in tables:
        for col in table.columns:
            for value in table[col].astype(str):
                found = re.findall(r"\b\d{4}\b|\b\d{3}[A-Z]\b", value)
                codes.extend(found)

    codes = list(dict.fromkeys(codes))

    if len(codes) > 225:
        codes = codes[:225]

    if len(codes) < 200:
        raise Exception(f"自動取得数が少なすぎます：{len(codes)}件")

    return codes


def safe_float(v):
    try:
        if v is None:
            return np.nan
        return float(v)
    except:
        return np.nan


def score_per(per):
    if pd.isna(per) or per <= 0:
        return 0
    if per <= 10:
        return 10
    if per <= 15:
        return 8
    if per <= 20:
        return 6
    if per <= 30:
        return 3
    return 0


def score_pbr(pbr):
    if pd.isna(pbr) or pbr <= 0:
        return 0
    if pbr <= 1:
        return 5
    if pbr <= 1.5:
        return 4
    if pbr <= 2.5:
        return 2
    return 0


def score_roe(roe):
    if pd.isna(roe):
        return 0
    roe_p = roe * 100
    if roe_p >= 15:
        return 15
    if roe_p >= 10:
        return 12
    if roe_p >= 8:
        return 8
    if roe_p >= 5:
        return 4
    return 0


def score_dividend(dividend_yield):
    if pd.isna(dividend_yield):
        return 0
    d = dividend_yield * 100
    if d >= 4:
        return 8
    if d >= 3:
        return 6
    if d >= 2:
        return 4
    if d >= 1:
        return 2
    return 0


def score_growth(growth):
    if pd.isna(growth):
        return 0
    g = growth * 100
    if g >= 20:
        return 10
    if g >= 10:
        return 8
    if g >= 5:
        return 5
    if g >= 0:
        return 2
    return -5


def judge_signal(score):
    if score >= 120:
        return "強気買い"
    if score >= 105:
        return "買い候補"
    if score >= 90:
        return "監視候補"
    return "対象外"


def judge_rank(score):
    if score >= 130:
        return "S"
    if score >= 120:
        return "A"
    if score >= 110:
        return "B"
    if score >= 100:
        return "C"
    if score >= 90:
        return "D"
    return "E"


def send_line_message(message):
    if not SEND_LINE:
        print(message)
        return

    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("LINE_CHANNEL_ACCESS_TOKENが未設定です。")
        print(message)
        return

    url = "https://api.line.me/v2/bot/message/broadcast"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }

    payload = {
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    print("LINE送信ステータス:", response.status_code)
    print(response.text)


def analyze():
    try:
        codes = get_nikkei225_codes_auto()
        source = "Wikipedia API"
    except Exception as e:
        print("日経225自動取得失敗:", e)
        codes = fallback_nikkei225_codes
        source = "固定リスト"

    codes = list(dict.fromkeys(codes))
    tickers = [c + ".T" for c in codes]

    print(f"銘柄取得方法：{source}")
    print(f"分析対象：{len(tickers)}銘柄")

    results = []
    errors = []

    for i, ticker in enumerate(tickers, start=1):
        code = ticker.replace(".T", "")
        print(f"{i}/{len(tickers)} {ticker}")

        try:
            df = yf.download(
                ticker,
                period="2y",
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if df.empty or len(df) < 250:
                errors.append(f"{ticker}: データ不足")
                continue

            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze()

            current = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])

            ma25 = float(close.rolling(25).mean().iloc[-1])
            ma75 = float(close.rolling(75).mean().iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1])

            rsi = float(RSIIndicator(close).rsi().iloc[-1])

            ret_1m = float((current / close.iloc[-21] - 1) * 100)
            ret_3m = float((current / close.iloc[-63] - 1) * 100)
            ret_6m = float((current / close.iloc[-126] - 1) * 100)

            deviation_25 = float((current / ma25 - 1) * 100)
            volatility = float(close.pct_change().rolling(20).std().iloc[-1] * 100)

            high_52w = float(close.tail(252).max())
            low_52w = float(close.tail(252).min())

            distance_from_52w_high = float((current / high_52w - 1) * 100)
            rebound_from_52w_low = float((current / low_52w - 1) * 100)

            recent_high_6m = float(close.tail(126).max())
            drawdown_6m = float((current / recent_high_6m - 1) * 100)

            high_6m_idx = close.tail(126).idxmax()
            days_since_6m_high = int((close.index[-1] - high_6m_idx).days)

            avg_volume_20 = float(volume.tail(20).mean())
            avg_volume_60 = float(volume.tail(60).mean())
            volume_ratio = float(avg_volume_20 / avg_volume_60) if avg_volume_60 > 0 else np.nan

            volume_spike = volume_ratio >= 1.5
            price_not_following = ret_1m < 3
            volume_bad_signal = volume_spike and price_not_following

            try:
                info = yf.Ticker(ticker).info
            except:
                info = {}

            name = info.get("shortName", code)

            per = safe_float(info.get("trailingPE"))
            forward_per = safe_float(info.get("forwardPE"))
            pbr = safe_float(info.get("priceToBook"))
            roe = safe_float(info.get("returnOnEquity"))
            dividend_yield = safe_float(info.get("dividendYield"))
            revenue_growth = safe_float(info.get("revenueGrowth"))
            earnings_growth = safe_float(info.get("earningsGrowth"))

            technical_score = 0
            fundamental_score = 0
            supply_score = 0
            theme_score = 0
            reasons = []

            if current > ma200:
                technical_score += 15
                reasons.append("200日線上")

            if current > ma75:
                technical_score += 10
                reasons.append("75日線上")

            if 45 <= rsi <= 65:
                technical_score += 10
                reasons.append("RSI適正")
            elif 65 < rsi <= 70:
                technical_score += 5
                reasons.append("RSIやや高め")
            elif rsi > 75:
                technical_score -= 15
                reasons.append("RSI過熱")

            if ret_3m > 0:
                technical_score += min(ret_3m / 2, 10)
                reasons.append("3か月上昇")

            if ret_6m > 0:
                technical_score += min(ret_6m / 3, 10)
                reasons.append("6か月上昇")

            if -5 <= deviation_25 <= 5:
                technical_score += 8
                reasons.append("25日線付近")
            elif deviation_25 > 10:
                technical_score -= 10
                reasons.append("25日線高乖離")

            if volatility <= 3:
                technical_score += 5
                reasons.append("値動き安定")

            fundamental_score += score_per(per)
            fundamental_score += score_pbr(pbr)
            fundamental_score += score_roe(roe)
            fundamental_score += score_dividend(dividend_yield)
            fundamental_score += score_growth(revenue_growth)
            fundamental_score += score_growth(earnings_growth)

            if score_per(per) > 0:
                reasons.append("PER良好")
            if score_pbr(pbr) > 0:
                reasons.append("PBR良好")
            if score_roe(roe) > 0:
                reasons.append("ROE良好")
            if score_dividend(dividend_yield) > 0:
                reasons.append("配当あり")
            if score_growth(revenue_growth) > 0:
                reasons.append("売上成長")
            if score_growth(earnings_growth) > 0:
                reasons.append("利益成長")
            if score_growth(revenue_growth) < 0:
                reasons.append("売上成長マイナス")
            if score_growth(earnings_growth) < 0:
                reasons.append("利益成長マイナス")

            if -25 <= distance_from_52w_high <= -8:
                supply_score += 12
                reasons.append("高値から適度に調整")
            elif -8 < distance_from_52w_high <= -2:
                supply_score += 5
                reasons.append("高値接近")
            elif distance_from_52w_high > -2:
                supply_score -= 15
                reasons.append("52週高値圏")
            elif distance_from_52w_high < -40:
                supply_score -= 10
                reasons.append("高値から大幅下落")

            if 120 <= days_since_6m_high <= 190 and drawdown_6m < -8:
                supply_score -= 15
                reasons.append("信用期日警戒")
            elif days_since_6m_high >= 90 and -20 <= drawdown_6m <= -5:
                supply_score += 8
                reasons.append("整理進行")

            if volume_ratio >= 1.3 and ret_1m > 5:
                supply_score += 8
                reasons.append("出来高増で上昇")
            elif volume_bad_signal:
                supply_score -= 12
                reasons.append("出来高増でも伸びず")

            if rebound_from_52w_low >= 80:
                supply_score -= 8
                reasons.append("急反発済み")
            elif 20 <= rebound_from_52w_low <= 60:
                supply_score += 7
                reasons.append("適度に反発")

            if drawdown_6m > -15:
                supply_score += 5
                reasons.append("下落浅い")

            if code in semiconductor_ai_codes:
                theme_score += 10
                reasons.append("半導体AI関連")

            if code in high_dividend_codes:
                theme_score += 5
                reasons.append("高配当候補")

            total_score = round(max(0, technical_score + fundamental_score + supply_score + theme_score), 1)

            buy_zone_low = round(current * 0.97, 0)
            buy_zone_high = round(current * 1.02, 0)
            stop_loss = round(current * 0.92, 0)
            take_profit = round(current * 1.15, 0)

            preopen_judgement = "中立"
            if distance_from_52w_high > -3 or rsi >= 70 or deviation_25 > 8:
                preopen_judgement = "飛びつき注意"
            elif -25 <= distance_from_52w_high <= -8 and 45 <= rsi <= 65:
                preopen_judgement = "押し目候補"
            elif volume_bad_signal:
                preopen_judgement = "需給警戒"

            results.append({
                "銘柄名": name,
                "コード": code,
                "株価": round(current, 0),
                "前日終値": round(prev_close, 0),
                "総合点": total_score,
                "ランク": judge_rank(total_score),
                "判定": judge_signal(total_score),
                "最終確認": preopen_judgement,
                "テクニカル点": round(technical_score, 1),
                "ファンダ点": round(fundamental_score, 1),
                "需給点": round(supply_score, 1),
                "テーマ点": round(theme_score, 1),
                "PER": round(per, 1) if not pd.isna(per) else np.nan,
                "予想PER": round(forward_per, 1) if not pd.isna(forward_per) else np.nan,
                "PBR": round(pbr, 1) if not pd.isna(pbr) else np.nan,
                "ROE%": round(roe * 100, 1) if not pd.isna(roe) else np.nan,
                "配当利回り%": round(dividend_yield * 100, 2) if not pd.isna(dividend_yield) else np.nan,
                "売上成長率%": round(revenue_growth * 100, 1) if not pd.isna(revenue_growth) else np.nan,
                "利益成長率%": round(earnings_growth * 100, 1) if not pd.isna(earnings_growth) else np.nan,
                "RSI": round(rsi, 1),
                "1か月%": round(ret_1m, 1),
                "3か月%": round(ret_3m, 1),
                "6か月%": round(ret_6m, 1),
                "25日乖離%": round(deviation_25, 1),
                "52週高値乖離%": round(distance_from_52w_high, 1),
                "52週安値反発%": round(rebound_from_52w_low, 1),
                "6か月高値からの日数": days_since_6m_high,
                "6か月高値下落率%": round(drawdown_6m, 1),
                "出来高倍率": round(volume_ratio, 2),
                "買いゾーン下限": buy_zone_low,
                "買いゾーン上限": buy_zone_high,
                "損切り目安": stop_loss,
                "利確目安": take_profit,
                "半導体AI関連": "該当" if code in semiconductor_ai_codes else "",
                "高配当候補": "該当" if code in high_dividend_codes else "",
                "理由": "、".join(reasons)
            })

            time.sleep(0.2)

        except Exception as e:
            errors.append(f"{ticker}: {e}")

    ranking = pd.DataFrame(results)

    if ranking.empty:
        return ranking, errors

    ranking = ranking.sort_values("総合点", ascending=False).reset_index(drop=True)

    today = datetime.now(JST).strftime("%Y%m%d")
    ranking.to_csv(f"nikkei225_ranking_ver8_{today}.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(f"nikkei225_ranking_ver8_{today}.xlsx", engine="openpyxl") as writer:
        ranking.to_excel(writer, sheet_name="総合ランキング", index=False)
        ranking[ranking["半導体AI関連"] == "該当"].to_excel(writer, sheet_name="半導体AI", index=False)
        ranking[ranking["高配当候補"] == "該当"].to_excel(writer, sheet_name="高配当", index=False)

    print(ranking.head(20).to_string(index=False))

    if errors:
        print("エラー銘柄")
        for e in errors[:30]:
            print(e)
    else:
        print("エラーなし")

    return ranking, errors


def make_morning_message(ranking):
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    top5 = ranking.head(5)
    semi = ranking[ranking["半導体AI関連"] == "該当"].head(3)
    high_div = ranking[ranking["高配当候補"] == "該当"].sort_values(
        ["配当利回り%", "総合点"],
        ascending=False
    ).head(3)

    msg = f"【7:30 朝の有望銘柄 Ver8】\n{now}\n\n"
    msg += "◆ 総合TOP5\n"

    for i, row in top5.iterrows():
        msg += f"{i+1}位 {row['銘柄名']}（{row['コード']}）\n"
        msg += f"総合:{row['総合点']} / {row['ランク']} / {row['判定']}\n"
        msg += f"内訳 テク:{row['テクニカル点']} ファンダ:{row['ファンダ点']} 需給:{row['需給点']}\n"
        msg += f"52週高値乖離:{row['52週高値乖離%']}% / RSI:{row['RSI']}\n"
        msg += f"買いゾーン:{row['買いゾーン下限']}〜{row['買いゾーン上限']}円\n\n"

    msg += "◆ 半導体AI TOP3\n"
    if semi.empty:
        msg += "該当なし\n"
    else:
        for _, row in semi.iterrows():
            msg += f"・{row['銘柄名']}（{row['コード']}）{row['総合点']}点 需給:{row['需給点']}\n"

    msg += "\n◆ 高配当 TOP3\n"
    if high_div.empty:
        msg += "該当なし\n"
    else:
        for _, row in high_div.iterrows():
            msg += f"・{row['銘柄名']}（{row['コード']}）利回り:{row['配当利回り%']}% 総合:{row['総合点']}\n"

    msg += "\n※7:30は候補発掘用。実際の買い判断は8:55通知と板確認後。"
    return msg


def make_preopen_message(ranking):
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")

    candidates = ranking[
        ranking["判定"].isin(["強気買い", "買い候補", "監視候補"])
    ].copy()

    candidates = candidates.sort_values(
        ["最終確認", "総合点"],
        ascending=[True, False]
    ).head(7)

    msg = f"【8:55 寄り前最終確認 Ver8】\n{now}\n\n"
    msg += "※無料データでは板・気配値は未取得。\n"
    msg += "高値掴み・需給悪化・押し目度で最終確認。\n\n"

    for i, row in candidates.iterrows():
        mark = "○"
        if row["最終確認"] == "飛びつき注意":
            mark = "⚠️"
        elif row["最終確認"] == "需給警戒":
            mark = "△"
        elif row["最終確認"] == "押し目候補":
            mark = "◎"

        msg += f"{mark} {row['銘柄名']}（{row['コード']}）\n"
        msg += f"判断:{row['最終確認']} / 総合:{row['総合点']} / 需給:{row['需給点']}\n"
        msg += f"株価:{row['株価']}円 / 買いゾーン:{row['買いゾーン下限']}〜{row['買いゾーン上限']}円\n"
        msg += f"52週高値乖離:{row['52週高値乖離%']}% / 6か月高値から:{row['6か月高値からの日数']}日\n"
        msg += f"RSI:{row['RSI']} / 25日乖離:{row['25日乖離%']}% / 出来高倍率:{row['出来高倍率']}\n\n"

    msg += "寄り付きで大幅GUなら飛びつき注意。押し目・出来高・板を見て判断。"
    return msg


ranking, errors = analyze()

if ranking.empty:
    send_line_message("日経225分析に失敗しました。ランキングが空です。")
else:
    if NOTIFY_MODE == "PREOPEN":
        message = make_preopen_message(ranking)
    else:
        message = make_morning_message(ranking)

    print("=" * 80)
    print("LINE通知メッセージ")
    print("=" * 80)
    print(message)

    send_line_message(message)
