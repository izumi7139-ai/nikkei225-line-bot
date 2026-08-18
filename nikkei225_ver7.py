# -*- coding: utf-8 -*-
"""日経225分析ツール Ver7.0
Ver6の銘柄選定を残し、Entry/履歴/Outcomeを追加。
"""
from __future__ import annotations
import os,re,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

MODEL_VERSION='7.0'
BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/'data'; REPORT_DIR=BASE_DIR/'reports'; DAILY_DIR=REPORT_DIR/'daily'; MONTHLY_DIR=REPORT_DIR/'monthly'
for d in (DATA_DIR,REPORT_DIR,DAILY_DIR,MONTHLY_DIR): d.mkdir(parents=True,exist_ok=True)
DAILY_SIGNALS_FILE=DATA_DIR/'daily_signals.csv'; SIGNAL_MASTER_FILE=DATA_DIR/'signal_master.csv'; OUTCOMES_FILE=DATA_DIR/'outcomes.csv'; PERFORMANCE_FILE=DATA_DIR/'performance_summary.csv'
SIGNAL_BREAK_DAYS=2; MAX_ENTRY_NOTIFY=3; MAX_WATCH_NOTIFY=3; MAX_HOLD_NOTIFY=3
MAX_ACCEL_MA25_DISTANCE_PCT=8.0; MAX_ACCEL_FIRST_SIGNAL_RETURN_PCT=12.0
REENTRY_MAX_MA25_DISTANCE_ABS_PCT=4.0; REENTRY_MIN_PULLBACK_FROM_60D_HIGH_PCT=-5.0
ATR_STOP_MULTIPLIER=1.5; SWING_LOOKBACK=20; SWING_BUFFER_PCT=0.005
VER6_STOP_PCT=0.08; VER6_TAKE_PROFIT_PCT=0.15
SEND_LINE=os.getenv('SEND_LINE','false').lower() in ('1','true','yes','on')
LINE_SEND_MODE=os.getenv('LINE_SEND_MODE','broadcast').lower(); LINE_CHANNEL_ACCESS_TOKEN=os.getenv('LINE_CHANNEL_ACCESS_TOKEN',''); LINE_USER_ID=os.getenv('LINE_USER_ID','')

fallback_nikkei225_codes=[
'1332','1605','1721','1801','1802','1803','1808','1812','1925','1928','1963','2002','2269','2282','2501','2502','2503','2801','2802','2871','2914','3101','3103','3401','3402','3405','3407','3861','4004','4005','4021','4042','4043','4061','4063','4183','4188','4208','4452','4631','4901','4911','6988','4151','4502','4503','4506','4507','4519','4523','4568','4578','5019','5020','5101','5108','5201','5214','5232','5233','5301','5332','5333','5401','5406','5411','3436','5706','5711','5713','5714','5801','5802','5803','6103','6113','6301','6302','6305','6326','6361','6367','6471','6472','6473','7004','7011','7012','7013','6501','6503','6504','6506','6526','6594','6645','6701','6702','6723','6724','6752','6753','6758','6762','6770','6841','6857','6861','6902','6920','6954','6971','6976','6981','7735','7751','7752','8035','285A','7201','7202','7203','7205','7211','7261','7267','7269','7270','4543','7731','7733','7741','7762','7832','7911','7912','7951','7974','8001','8002','8015','8031','8053','8058','3086','3092','3099','3382','7453','7532','8233','8252','8267','9843','9983','8306','8308','8309','8316','8331','8354','8411','7186','8253','8591','8601','8604','8628','8697','8725','8750','8766','8795','3289','8801','8802','8804','8830','9001','9005','9007','9008','9009','9020','9021','9022','9023','9064','9147','9101','9104','9107','9201','9202','9301','9432','9433','9434','9613','9984','9501','9502','9503','9531','9532','2413','2432','3659','4324','4689','4704','4751','4755','6098','6178','9602','9735','9766']
semiconductor_ai_codes=['8035','6857','6723','6724','6981','6861','6758','6701','6702','6501','6503','6504','6594','4063','6988','9984','285A']
high_dividend_candidate_codes=['8306','8316','8308','8309','8411','8591','8058','8001','8002','8031','2914','9432','9433','9434','8766','8750']

def safe_float(v):
    try:
        if v is None:return np.nan
        x=float(v); return x if np.isfinite(x) else np.nan
    except:return np.nan

def pct_change(cur,base):
    try:
        cur=float(cur); base=float(base)
        return (cur/base-1)*100 if base and np.isfinite(cur) and np.isfinite(base) else np.nan
    except:return np.nan

def rn(v,d=1): return np.nan if pd.isna(v) else round(float(v),d)
def read_csv_safe(path,date_cols=()):
    if not path.exists(): return pd.DataFrame()
    df=pd.read_csv(path,encoding='utf-8-sig')
    for c in date_cols:
        if c in df.columns: df[c]=pd.to_datetime(df[c],errors='coerce')
    return df

def atomic_to_csv(df,path):
    tmp=path.with_suffix(path.suffix+'.tmp'); df.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(path)
def to_bool(v): return v if isinstance(v,bool) else str(v).lower() in ('true','1','yes','on')
def fmt(v,d=1,s=''): return '-' if pd.isna(v) else f'{float(v):.{d}f}{s}'

def get_nikkei225_codes_auto():
    url='https://ja.wikipedia.org/w/api.php'; params={'action':'parse','page':'日経平均株価','prop':'text','format':'json'}
    r=requests.get(url,params=params,headers={'User-Agent':'Mozilla/5.0 nikkei225-ver7'},timeout=20); r.raise_for_status()
    codes=[]
    for t in pd.read_html(r.json()['parse']['text']['*']):
        for col in t.columns:
            for val in t[col].astype(str): codes += re.findall(r'\b\d{4}\b|\b\d{3}[A-Z]\b',val)
    codes=list(dict.fromkeys(codes))[:225]
    if len(codes)<200: raise RuntimeError(f'自動取得数不足:{len(codes)}')
    return codes

def score_per(x):
    if pd.isna(x) or x<=0:return 0
    return 10 if x<=10 else 8 if x<=15 else 6 if x<=20 else 3 if x<=30 else 0
def score_pbr(x):
    if pd.isna(x) or x<=0:return 0
    return 5 if x<=1 else 4 if x<=1.5 else 2 if x<=2.5 else 0
def score_roe(x):
    if pd.isna(x):return 0
    p=x*100; return 15 if p>=15 else 12 if p>=10 else 8 if p>=8 else 4 if p>=5 else 0
def score_dividend(x):
    if pd.isna(x):return 0
    p=x*100; return 8 if p>=4 else 6 if p>=3 else 4 if p>=2 else 2 if p>=1 else 0
def score_growth(x):
    if pd.isna(x):return 0
    p=x*100; return 10 if p>=20 else 8 if p>=10 else 5 if p>=5 else 2 if p>=0 else 0
def judge_signal(s): return '強気買い' if s>=120 else '買い候補' if s>=105 else '監視候補' if s>=90 else '対象外'
def judge_rank(s): return 'S' if s>=130 else 'A' if s>=120 else 'B' if s>=110 else 'C' if s>=100 else 'D' if s>=90 else 'E'

def send_line_message(message):
    if not SEND_LINE:
        print('LINE送信OFF'); return False
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print('LINE_CHANNEL_ACCESS_TOKEN未設定'); return False
    headers={'Content-Type':'application/json','Authorization':f'Bearer {LINE_CHANNEL_ACCESS_TOKEN}'}
    if LINE_SEND_MODE=='broadcast':
        url='https://api.line.me/v2/bot/message/broadcast'; payload={'messages':[{'type':'text','text':message}]}
    elif LINE_SEND_MODE=='push':
        if not LINE_USER_ID: print('LINE_USER_ID未設定'); return False
        url='https://api.line.me/v2/bot/message/push'; payload={'to':LINE_USER_ID,'messages':[{'type':'text','text':message}]}
    else:
        print('LINE_SEND_MODE不正'); return False
    try:
        r=requests.post(url,headers=headers,json=payload,timeout=30); print('LINE:',r.status_code,r.text); return r.ok
    except Exception as e: print('LINE送信エラー',e); return False

def download_price_history(ticker,period='1y'):
    df=yf.download(ticker,period=period,progress=False,auto_adjust=True,threads=False)
    if df is None or df.empty:return pd.DataFrame()
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    need=['Open','High','Low','Close','Volume']
    if not set(need).issubset(df.columns):return pd.DataFrame()
    df=df[need].copy(); df.index=pd.to_datetime(df.index).tz_localize(None); return df.sort_index().dropna(subset=['Close'])

def analyze_one_stock(ticker):
    code=ticker.replace('.T',''); df=download_price_history(ticker)
    if df.empty or len(df)<220: raise RuntimeError(f'株価データ不足:{len(df)}')
    close=df['Close'].astype(float); high=df['High'].astype(float); low=df['Low'].astype(float); volume=df['Volume'].astype(float)
    current=float(close.iloc[-1]); market_date=pd.Timestamp(close.index[-1]).normalize()
    ma25=float(close.rolling(25).mean().iloc[-1]); ma75=float(close.rolling(75).mean().iloc[-1]); ma200=float(close.rolling(200).mean().iloc[-1])
    rsi=float(RSIIndicator(close).rsi().iloc[-1])
    r1=pct_change(current,close.iloc[-21]); r3=pct_change(current,close.iloc[-63]); r6=pct_change(current,close.iloc[-126])
    r5=pct_change(current,close.iloc[-6]); r20=pct_change(current,close.iloc[-21]); accel=r5-r20*0.25 if not pd.isna(r5) and not pd.isna(r20) else np.nan
    dev25=pct_change(current,ma25); vol=float(close.pct_change().rolling(20).std().iloc[-1]*100); high60=float(close.tail(60).max()); dd60=pct_change(current,high60)
    avgv=float(volume.tail(20).mean()); d1=pct_change(current,close.iloc[-2])
    try: atr=float(AverageTrueRange(high,low,close,window=14).average_true_range().iloc[-1])
    except: atr=np.nan
    swing_low=float(low.tail(SWING_LOOKBACK).min()); atr_stop=round(current-ATR_STOP_MULTIPLIER*atr,1) if not pd.isna(atr) else np.nan; swing_stop=round(swing_low*(1-SWING_BUFFER_PCT),1)
    try: info=yf.Ticker(ticker).info or {}
    except: info={}
    name=info.get('shortName',code); per=safe_float(info.get('trailingPE')); fper=safe_float(info.get('forwardPE')); pbr=safe_float(info.get('priceToBook')); roe=safe_float(info.get('returnOnEquity')); dy=safe_float(info.get('dividendYield')); rg=safe_float(info.get('revenueGrowth')); eg=safe_float(info.get('earningsGrowth'))
    score=0.0; reasons=[]
    if current>ma200:score+=15;reasons.append('200日線上')
    if current>ma75:score+=10;reasons.append('75日線上')
    if 45<=rsi<=70:score+=10;reasons.append('RSI適正')
    if r3>0:score+=min(r3/2,10);reasons.append('3か月上昇')
    if r6>0:score+=min(r6/3,10);reasons.append('6か月上昇')
    if -8<=dev25<=8:score+=5;reasons.append('25日線付近')
    for val,fn,label in [(per,score_per,'PER良好'),(pbr,score_pbr,'PBR良好'),(roe,score_roe,'ROE良好'),(dy,score_dividend,'配当あり'),(rg,score_growth,'売上成長'),(eg,score_growth,'利益成長')]:
        add=fn(val)
        if add>0: score+=add; reasons.append(label)
    if vol<=3:score+=5;reasons.append('値動き安定')
    if dd60>-15:score+=5;reasons.append('下落浅い')
    if rsi>=75:score-=10;reasons.append('RSI過熱')
    if dd60<=-25:score-=10;reasons.append('下落大')
    score=round(max(0,score),1); signal=judge_signal(score); rank=judge_rank(score)
    buy_low=round(current*0.97,0); buy_high=round(current*1.02,0); sl=round(current*(1-VER6_STOP_PCT),0); tp=round(current*(1+VER6_TAKE_PROFIT_PCT),0); er=pct_change(tp,current); el=pct_change(sl,current); rr=abs(er/el) if el else np.nan
    result={'市場日付':market_date.strftime('%Y-%m-%d'),'銘柄名':name,'コード':code,'Ticker':ticker,'株価':round(current,1),'総合点':score,'ランク':rank,'判定':signal,
    'PER':rn(per,1),'予想PER':rn(fper,1),'PBR':rn(pbr,1),'ROE%':rn(roe*100 if not pd.isna(roe) else np.nan,1),'配当利回り%':rn(dy*100 if not pd.isna(dy) else np.nan,2),'売上成長率%':rn(rg*100 if not pd.isna(rg) else np.nan,1),'利益成長率%':rn(eg*100 if not pd.isna(eg) else np.nan,1),
    'RSI':round(rsi,1),'前日比%':rn(d1,2),'5日%':rn(r5,2),'20日%':rn(r20,2),'MomentumAcceleration%':rn(accel,2),'1か月%':rn(r1,1),'3か月%':rn(r3,1),'6か月%':rn(r6,1),'25日乖離%':rn(dev25,2),'60日高値乖離%':rn(dd60,2),'20日平均出来高':round(avgv,0),
    'MA25':round(ma25,1),'MA75':round(ma75,1),'MA200':round(ma200,1),'ATR14':rn(atr,2),'20日安値':round(swing_low,1),'買いゾーン下限':buy_low,'買いゾーン上限':buy_high,'損切り目安':sl,'利確目安':tp,'ATR損切り':atr_stop,'Swing損切り':swing_stop,'期待利益%':rn(er,1),'想定損失%':rn(el,1),'RR比':rn(rr,2),'半導体AI関連':'該当' if code in semiconductor_ai_codes else '','高配当候補':'該当' if code in high_dividend_candidate_codes else '','理由':'、'.join(reasons)}
    return result

def validate_history_files():
    daily=read_csv_safe(DAILY_SIGNALS_FILE,['date','signal_start_date']); master=read_csv_safe(SIGNAL_MASTER_FILE,['signal_start_date','last_gate_date','end_date'])
    if DAILY_SIGNALS_FILE.exists() and not daily.empty:
        req={'date','code','ver6_score','ver6_rank_number','quality_gate'}; miss=req-set(daily.columns)
        if miss: raise RuntimeError(f'daily_signals必須列不足:{sorted(miss)}')
    if SIGNAL_MASTER_FILE.exists() and not master.empty:
        req={'signal_id','code','signal_start_date','signal_start_price','signal_age','days_outside_gate','status'}; miss=req-set(master.columns)
        if miss: raise RuntimeError(f'signal_master必須列不足:{sorted(miss)}')
    return daily,master

def get_prev(daily,code):
    if daily.empty:return None
    x=daily[daily['code'].astype(str)==str(code)].copy()
    if x.empty:return None
    x['date']=pd.to_datetime(x['date'],errors='coerce'); x=x.dropna(subset=['date']).sort_values('date'); return None if x.empty else x.iloc[-1]
def get_active(master,code):
    if master.empty:return None
    x=master[(master['code'].astype(str)==str(code))&(master['status'].astype(str)=='ACTIVE')]; return None if x.empty else x.iloc[-1]
def next_num(master,code): return 1 if master.empty else len(master[master['code'].astype(str)==str(code)])+1
def make_signal_id(code,date,n): return f"{code}_{pd.Timestamp(date).strftime('%Y%m%d')}_{n:02d}"

def update_signal_state(ranking,daily,master,market_date):
    ranking=ranking.copy()
    if master.empty:
        master=pd.DataFrame(columns=['signal_id','code','name','signal_start_date','signal_start_price','signal_age','days_outside_gate','last_gate_date','end_date','status','start_ver6_score','start_ver6_rank_number','model_version'])
    out={k:[] for k in ['signal_id','signal_start_date','signal_start_price','SignalAge','days_outside_gate','FirstSignalReturn%','前回Ver6順位','RankVelocity','ScoreVelocity']}
    for _,row in ranking.iterrows():
        code=str(row['コード']); q=row['判定'] in ('強気買い','買い候補'); cur=float(row['株価']); vrank=int(row['Ver6順位']); score=float(row['総合点']); prev=get_prev(daily,code)
        prev_rank=rv=sv=np.nan
        if prev is not None:
            if not pd.isna(prev.get('ver6_rank_number',np.nan)): prev_rank=float(prev['ver6_rank_number']); rv=prev_rank-vrank
            if not pd.isna(prev.get('ver6_score',np.nan)): sv=score-float(prev['ver6_score'])
        active=get_active(master,code); sid=''; sdate=pd.NaT; sprice=np.nan; age=np.nan; outside=0
        if active is None:
            if q:
                n=next_num(master,code); sid=make_signal_id(code,market_date,n); sdate=market_date; sprice=cur; age=0; outside=0
                master=pd.concat([master,pd.DataFrame([{'signal_id':sid,'code':code,'name':row['銘柄名'],'signal_start_date':market_date,'signal_start_price':cur,'signal_age':0,'days_outside_gate':0,'last_gate_date':market_date,'end_date':pd.NaT,'status':'ACTIVE','start_ver6_score':score,'start_ver6_rank_number':vrank,'model_version':MODEL_VERSION}])],ignore_index=True)
        else:
            idx=master.index[master['signal_id']==active['signal_id']][-1]; sid=str(active['signal_id']); sdate=pd.to_datetime(active['signal_start_date']); sprice=float(active['signal_start_price']); age=int(float(active.get('signal_age',0))); outside=int(float(active.get('days_outside_gate',0)))
            if q:
                age+=1; outside=0; master.loc[idx,'signal_age']=age; master.loc[idx,'days_outside_gate']=0; master.loc[idx,'last_gate_date']=market_date
            else:
                outside+=1; master.loc[idx,'days_outside_gate']=outside
                if outside>=SIGNAL_BREAK_DAYS: master.loc[idx,'status']='ENDED'; master.loc[idx,'end_date']=market_date
        first_ret=pct_change(cur,sprice) if sid else np.nan
        vals={'signal_id':sid,'signal_start_date':pd.Timestamp(sdate).strftime('%Y-%m-%d') if not pd.isna(sdate) else '','signal_start_price':sprice,'SignalAge':age,'days_outside_gate':outside,'FirstSignalReturn%':first_ret,'前回Ver6順位':prev_rank,'RankVelocity':rv,'ScoreVelocity':sv}
        for k,v in vals.items(): out[k].append(v)
    for k,v in out.items(): ranking[k]=v
    return ranking,master

def classify_phase(row):
    q=row['判定'] in ('強気買い','買い候補')
    if not q:return 'MATURE' if row.get('signal_id','') else ''
    r5=safe_float(row.get('5日%')); r20=safe_float(row.get('20日%')); acc=safe_float(row.get('MomentumAcceleration%')); ma25=safe_float(row.get('25日乖離%')); first=safe_float(row.get('FirstSignalReturn%')); age=safe_float(row.get('SignalAge')); rv=safe_float(row.get('RankVelocity')); dd=safe_float(row.get('60日高値乖離%'))
    if any(pd.isna(x) for x in [r5,r20,acc,ma25]):return 'DATA_INSUFFICIENT'
    if not pd.isna(age) and age>=5 and not pd.isna(first) and first>0 and not pd.isna(dd) and dd<=REENTRY_MIN_PULLBACK_FROM_60D_HIGH_PCT and abs(ma25)<=REENTRY_MAX_MA25_DISTANCE_ABS_PCT and r5>0 and acc>0:return 'RE-ENTRY'
    if (not pd.isna(first) and first>=MAX_ACCEL_FIRST_SIGNAL_RETURN_PCT) or ma25>MAX_ACCEL_MA25_DISTANCE_PCT or (r20>0 and r5<0 and acc<0):return 'MATURE'
    fresh=(not pd.isna(age) and age<=5) or (not pd.isna(rv) and rv>0)
    if r5>0 and r20>0 and acc>0 and ma25<=MAX_ACCEL_MA25_DISTANCE_PCT and (pd.isna(first) or first<MAX_ACCEL_FIRST_SIGNAL_RETURN_PCT) and fresh:return 'ACCELERATION'
    if not pd.isna(age) and age<=2:return 'NEW'
    if r20>0:return 'HOLD'
    return 'MATURE'
def decide_action(p): return 'ENTRY' if p=='ACCELERATION' else 'WATCH' if p in ('NEW','RE-ENTRY') else 'HOLD' if p=='HOLD' else 'SKIP'
def sort_key(row):
    first=safe_float(row.get('FirstSignalReturn%')); acc=safe_float(row.get('MomentumAcceleration%')); ma25=safe_float(row.get('25日乖離%')); rv=safe_float(row.get('RankVelocity'))
    return (0 if pd.isna(acc) else acc*10)+(0 if pd.isna(first) else -first)+(0 if pd.isna(ma25) else -abs(ma25)*0.5)+(0 if pd.isna(rv) else rv*0.5)

def make_daily_rows(ranking,market_date):
    rows=[]
    for _,r in ranking.iterrows():
        rows.append({'date':market_date.strftime('%Y-%m-%d'),'model_version':MODEL_VERSION,'code':str(r['コード']),'name':r['銘柄名'],'ticker':r['Ticker'],'close':r['株価'],'ver6_score':r['総合点'],'ver6_rank_number':r['Ver6順位'],'ver6_rank_letter':r['ランク'],'ver6_signal':r['判定'],'ver6_selected':r['Ver6順位']<=5,'quality_gate':r['判定'] in ('強気買い','買い候補'),'rsi':r['RSI'],'return_1d':r['前日比%'],'return_5d_past':r['5日%'],'return_20d_past':r['20日%'],'momentum_acceleration':r['MomentumAcceleration%'],'ma25_distance':r['25日乖離%'],'drawdown_60d_high':r['60日高値乖離%'],'atr14':r['ATR14'],'signal_id':r.get('signal_id',''),'signal_start_date':r.get('signal_start_date',''),'signal_start_price':r.get('signal_start_price',np.nan),'signal_age':r.get('SignalAge',np.nan),'days_outside_gate':r.get('days_outside_gate',0),'first_signal_return':r.get('FirstSignalReturn%',np.nan),'previous_ver6_rank':r.get('前回Ver6順位',np.nan),'rank_velocity':r.get('RankVelocity',np.nan),'score_velocity':r.get('ScoreVelocity',np.nan),'phase':r.get('Phase',''),'action':r.get('Action',''),'ver7_entry':r.get('Action','')=='ENTRY','buy_zone_low':r['買いゾーン下限'],'buy_zone_high':r['買いゾーン上限'],'ver6_stop':r['損切り目安'],'ver6_take_profit':r['利確目安'],'atr_stop':r['ATR損切り'],'swing_stop':r['Swing損切り']})
    return pd.DataFrame(rows)

def append_daily(history,today):
    c=today.copy() if history.empty else pd.concat([history,today],ignore_index=True)
    c['date']=pd.to_datetime(c['date'],errors='coerce'); c=c.sort_values('date').drop_duplicates(['date','code','model_version'],keep='last'); c['date']=c['date'].dt.strftime('%Y-%m-%d'); return c

def calc_outcome(ticker,signal_date,entry_price,h):
    sd=pd.Timestamp(signal_date).normalize(); start=(sd-pd.Timedelta(days=10)).strftime('%Y-%m-%d'); end=(sd+pd.Timedelta(days=h*3+20)).strftime('%Y-%m-%d')
    df=yf.download(ticker,start=start,end=end,progress=False,auto_adjust=True,threads=False)
    if df is None or df.empty:return None
    if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index).tz_localize(None); fut=df[df.index>sd].sort_index()
    if len(fut)<h:return None
    p=fut.iloc[:h]; ret=pct_change(float(p['Close'].iloc[-1]),entry_price); mfe=pct_change(float(p['High'].max()),entry_price); mae=pct_change(float(p['Low'].min()),entry_price); eff=mfe/abs(mae) if not pd.isna(mae) and abs(mae)>1e-9 else np.nan
    return {f'return_{h}d':ret,f'mfe_{h}d':mfe,f'mae_{h}d':mae,f'efficiency_{h}d':eff,f'exit_date_{h}d':pd.Timestamp(p.index[-1]).strftime('%Y-%m-%d')}

def update_outcomes(daily):
    existing=read_csv_safe(OUTCOMES_FILE,['signal_date'])
    if daily.empty:return existing
    base=daily.copy(); base['date']=pd.to_datetime(base['date'],errors='coerce'); base=base.dropna(subset=['date'])
    cand=base[base['ver6_selected'].apply(to_bool)|base['ver7_entry'].apply(to_bool)].copy()
    if existing.empty: existing=pd.DataFrame(columns=['signal_date','code','ticker','signal_id','model_version','ver6_selected','ver7_entry','entry_price'])
    for _,r in cand.iterrows():
        sd=pd.Timestamp(r['date']).normalize(); code=str(r['code']); ver=str(r.get('model_version',MODEL_VERSION)); keymask=(existing.get('signal_date',pd.Series(dtype=str)).astype(str).str[:10]==sd.strftime('%Y-%m-%d'))&(existing.get('code',pd.Series(dtype=str)).astype(str)==code)&(existing.get('model_version',pd.Series(dtype=str)).astype(str)==ver)
        if keymask.any(): idx=existing.index[keymask][-1]
        else:
            existing=pd.concat([existing,pd.DataFrame([{'signal_date':sd.strftime('%Y-%m-%d'),'code':code,'ticker':r['ticker'],'signal_id':r.get('signal_id',''),'model_version':ver,'ver6_selected':to_bool(r.get('ver6_selected',False)),'ver7_entry':to_bool(r.get('ver7_entry',False)),'entry_price':r['close']}])],ignore_index=True); idx=existing.index[-1]
        for h in (5,10,20):
            col=f'return_{h}d'
            if col in existing.columns and not pd.isna(existing.loc[idx,col]):continue
            try:o=calc_outcome(str(r['ticker']),sd,float(r['close']),h)
            except Exception as e: print('Outcome error',r['ticker'],sd,h,e); o=None
            if o:
                for k,v in o.items(): existing.loc[idx,k]=v
            time.sleep(0.03)
    return existing

def performance_summary(outcomes):
    if outcomes.empty:return pd.DataFrame()
    x=outcomes.copy(); x['signal_date']=pd.to_datetime(x['signal_date'],errors='coerce'); x=x.dropna(subset=['signal_date']); x['month']=x['signal_date'].dt.to_period('M').astype(str); rows=[]
    for month,m in x.groupby('month'):
        for strategy,flag in [('Ver6_TOP5','ver6_selected'),('Ver7_ENTRY','ver7_entry')]:
            s=m[m[flag].apply(to_bool)].copy()
            if 'return_10d' not in s.columns:continue
            s=s.dropna(subset=['return_10d'])
            if s.empty:continue
            r=pd.to_numeric(s['return_10d'],errors='coerce'); mae=pd.to_numeric(s.get('mae_10d'),errors='coerce'); mfe=pd.to_numeric(s.get('mfe_10d'),errors='coerce')
            rows.append({'month':month,'strategy':strategy,'signals':len(s),'win_rate_10d_pct':round((r>0).mean()*100,1),'mean_return_10d_pct':round(r.mean(),2),'median_return_10d_pct':round(r.median(),2),'mean_mae_10d_pct':round(mae.mean(),2),'mean_mfe_10d_pct':round(mfe.mean(),2)})
    return pd.DataFrame(rows)

def phase_label(phase):
    labels={
        'ACCELERATION':'上昇加速中',
        'NEW':'新規シグナル',
        'HOLD':'保有継続',
        'MATURE':'上昇後半・高値注意',
        'RE-ENTRY':'押し目から再上昇',
        'DATA_INSUFFICIENT':'データ不足'
    }
    return labels.get(str(phase), str(phase))

def action_label(action):
    labels={
        'ENTRY':'新規買い候補',
        'WATCH':'監視',
        'HOLD':'保有継続',
        'SKIP':'新規見送り'
    }
    return labels.get(str(action), str(action))

def build_ver7_message(ranking,date):
    entry=ranking[ranking['Action']=='ENTRY'].copy()
    watch=ranking[ranking['Action']=='WATCH'].copy()
    hold=ranking[ranking['Action']=='HOLD'].copy()

    if not entry.empty:
        entry['SortKey']=entry.apply(sort_key,axis=1)
        entry=entry.sort_values(['SortKey','総合点'],ascending=[False,False]).head(MAX_ENTRY_NOTIFY)

    if not watch.empty:
        watch['P']=watch['Phase'].map({'NEW':2,'RE-ENTRY':1}).fillna(0)
        watch=watch.sort_values(['P','総合点'],ascending=[False,False]).head(MAX_WATCH_NOTIFY)

    if not hold.empty:
        hold=hold.sort_values('総合点',ascending=False).head(MAX_HOLD_NOTIFY)

    msg=f"🚀【日本株 買いタイミング速報 Ver7.0】\n{date.strftime('%Y-%m-%d')}\n\n"

    if entry.empty:
        msg+='【今日の新規買い候補】\n本日は新規買い候補がありません。\n\n'
    else:
        msg+=f'【今日の新規買い候補：{len(entry)}銘柄】\n\n'
        for i,(_,r) in enumerate(entry.iterrows(),1):
            msg+=f"{i}位 {r['銘柄名']}（{r['コード']}）\n"
            msg+=f"判定：🚀 {phase_label(r['Phase'])}\n"
            msg+=f"Ver6評価：{r['判定']} / {r['総合点']}点\n"
            msg+=f"直近5日：{fmt(r['5日%'],1,'%')}\n"
            msg+=f"直近20日：{fmt(r['20日%'],1,'%')}\n"
            msg+=f"上昇の加速度：{fmt(r['MomentumAcceleration%'],1,'%')}\n"
            msg+=f"25日移動平均線からの乖離：{fmt(r['25日乖離%'],1,'%')}\n"

            age=r.get('SignalAge',np.nan)
            if pd.isna(age):
                msg+='シグナル発生：-\n'
            elif float(age)==0:
                msg+='シグナル発生：本日\n'
            else:
                msg+=f"シグナル発生から：{fmt(age,0)}営業日\n"

            msg+=f"初回通知からの上昇率：{fmt(r['FirstSignalReturn%'],1,'%')}\n"
            msg+=f"買い目安：{fmt(r['買いゾーン下限'],0)}〜{fmt(r['買いゾーン上限'],0)}円\n"
            msg+=f"値動き基準の損切り目安：{fmt(r['ATR損切り'],0)}円\n"
            msg+=f"直近安値基準の損切り目安：{fmt(r['Swing損切り'],0)}円\n\n"

    if not watch.empty:
        msg+='👀【監視銘柄】\n'
        for _,r in watch.iterrows():
            explanation='良い銘柄ですが、まだ買いタイミング待ち'
            if r['Phase']=='RE-ENTRY':
                explanation='押し目から再上昇の兆し。もう少し確認'
            msg+=f"・{r['銘柄名']}（{r['コード']}）\n"
            msg+=f"  → {phase_label(r['Phase'])}：{explanation}\n"
        msg+='\n'

    if not hold.empty:
        msg+='🔵【保有継続候補】\n'
        for _,r in hold.iterrows():
            msg+=f"・{r['銘柄名']}（{r['コード']}）｜初回比 {fmt(r['FirstSignalReturn%'],1,'%')}\n"
        msg+='\n'

    # 初めて使う人向けの用語説明
    msg+='📖【用語説明】\n'
    msg+='■ 上昇の加速度とは？\n'
    msg+='「最近5日間の上昇ペース」が「直近20日間の平均的な上昇ペース」と比べて、どれだけ強まっているかを表すVer7独自の指標です。\n'
    msg+='プラスが大きいほど、最近になって株価の上昇に勢いがついている状態です。\n'
    msg+='計算：直近5日騰落率 −（直近20日騰落率 ÷ 4）\n'
    msg+='例：5日 +4.1% / 20日 +2.7% → 4.1 − 0.7 ≒ +3.4%\n'
    msg+='※上昇の加速度が高くても、将来の株価上昇を保証するものではありません。\n\n'
    msg+='※Ver7.0は検証運用中です。最終的な投資判断はご自身で行ってください。'
    return msg

def build_ver6_fallback(ranking,date):
    msg=f"【中長期有望銘柄 Ver6 / Ver7フォールバック】\n{date.strftime('%Y-%m-%d')}\n\n"
    for i,(_,r) in enumerate(ranking.head(5).iterrows(),1): msg+=f"{i}位 {r['銘柄名']}（{r['コード']}）\nスコア：{r['総合点']}点 / {r['ランク']}\n判定：{r['判定']}\n株価：{r['株価']}円\n\n"
    return msg+'※Ver7履歴処理エラーのためVer6通知です。'

def main():
    print('='*80); print('日経225 Ver7.0'); print('='*80)
    try: codes=get_nikkei225_codes_auto(); source='Wikipedia API'
    except Exception as e: print('銘柄自動取得失敗:',e); codes=fallback_nikkei225_codes; source='固定リスト'
    codes=list(dict.fromkeys(codes)); tickers=[c+'.T' for c in codes]; print('対象',len(tickers),'銘柄 /',source)
    history_ok=True
    try: daily,master=validate_history_files()
    except Exception as e: history_ok=False; daily=pd.DataFrame(); master=pd.DataFrame(); print('Ver7履歴エラー:',e); traceback.print_exc()
    results=[]; errors=[]
    for i,t in enumerate(tickers,1):
        print(f'{i}/{len(tickers)} {t}')
        try: results.append(analyze_one_stock(t))
        except Exception as e: errors.append(f'{t}: {e}'); print('ERROR',e)
        time.sleep(0.12)
    ranking=pd.DataFrame(results)
    if ranking.empty: raise RuntimeError('分析結果が空です')
    market_date=pd.Timestamp(pd.to_datetime(ranking['市場日付']).mode().iloc[0]).normalize(); ranking=ranking.sort_values('総合点',ascending=False).reset_index(drop=True); ranking['Ver6順位']=np.arange(1,len(ranking)+1)
    if history_ok:
        try:
            ranking,master2=update_signal_state(ranking,daily,master,market_date); ranking['Phase']=ranking.apply(classify_phase,axis=1); ranking['Action']=ranking['Phase'].apply(decide_action)
        except Exception as e: history_ok=False; master2=master.copy(); print('Ver7判定エラー:',e); traceback.print_exc()
    if not history_ok:
        master2=master.copy()
        for c,v in [('signal_id',''),('signal_start_date',''),('signal_start_price',np.nan),('SignalAge',np.nan),('days_outside_gate',np.nan),('FirstSignalReturn%',np.nan),('前回Ver6順位',np.nan),('RankVelocity',np.nan),('ScoreVelocity',np.nan),('Phase',''),('Action','')]:
            if c not in ranking.columns:ranking[c]=v
    print(ranking[['Ver6順位','銘柄名','コード','総合点','判定','5日%','20日%','MomentumAcceleration%','25日乖離%','Phase','Action']].head(20).to_string(index=False))
    ds=market_date.strftime('%Y%m%d'); csvf=DAILY_DIR/f'nikkei225_ranking_ver7_{ds}.csv'; xlsf=DAILY_DIR/f'nikkei225_ranking_ver7_{ds}.xlsx'; ranking.to_csv(csvf,index=False,encoding='utf-8-sig')
    buys=ranking[ranking['判定'].isin(['強気買い','買い候補'])]; semi=ranking[ranking['半導体AI関連']=='該当']; div=ranking[ranking['高配当候補']=='該当'].sort_values(['配当利回り%','総合点'],ascending=False)
    with pd.ExcelWriter(xlsf,engine='openpyxl') as w:
        ranking.to_excel(w,sheet_name='総合ランキング',index=False); buys.to_excel(w,sheet_name='Ver6買い候補',index=False); semi.to_excel(w,sheet_name='半導体AI',index=False); div.to_excel(w,sheet_name='高配当',index=False)
        if history_ok:
            for action in ['ENTRY','WATCH','HOLD']:
                ranking[ranking['Action']==action].to_excel(w,sheet_name='Ver7_'+action,index=False)
    if history_ok:
        today=make_daily_rows(ranking,market_date); combined=append_daily(daily,today); atomic_to_csv(combined,DAILY_SIGNALS_FILE); atomic_to_csv(master2,SIGNAL_MASTER_FILE)
        try:
            outcomes=update_outcomes(combined)
            if not outcomes.empty:
                atomic_to_csv(outcomes,OUTCOMES_FILE); perf=performance_summary(outcomes)
                if not perf.empty: atomic_to_csv(perf,PERFORMANCE_FILE); atomic_to_csv(perf,MONTHLY_DIR/'performance_summary.csv')
        except Exception as e: print('Outcome Engineエラー:',e); traceback.print_exc()
    message=build_ver7_message(ranking,market_date) if history_ok else build_ver6_fallback(ranking,market_date); print('\n'+message); send_line_message(message)
    print('\nエラー件数:',len(errors)); [print(e) for e in errors[:30]]; print('保存:',csvf,xlsf)

if __name__=='__main__': main()
