#!/usr/bin/env python3
"""
BTC 15m 訊號機器人 v3（掃流動性 / 支撐壓力 / 突破 / 主動買賣差背離 / 資金費率極端，
                        MACD、RSI、成交量、1H/4H、主動買賣差、資金費率過濾）
數據：幣安公開 K 線 API（免費、不用帳號）    推播：Telegram Bot

指令（在檔案所在資料夾執行）：
  python btc_signal_bot.py levels     列出目前支撐壓力位 + 資金費率 + 主動買賣差
  python btc_signal_bot.py scan       參數掃描（預設 730 天）：前 50% 時間選參數、後 50% 驗證；通過才存成 best_config.json
  python btc_signal_bot.py backtest   回測目前設定（有 best_config.json 就用它）；加天數例：backtest 730
  python btc_signal_bot.py live       盯盤 + Telegram 推播
  python btc_signal_bot.py test       測試 Telegram

安裝：python -m pip install pandas numpy requests
Telegram：設定環境變數 TG_TOKEN、TG_CHAT_ID
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ───────────────────────── 基本設定 ─────────────────────────
BASE = dict(
    symbol="BTCUSDT", market="futures",        # "futures" U 本位合約 / "spot" 現貨
    tf="15m", htf1="1h", htf2="4h",
    macd_fast=12, macd_slow=26, macd_sig=9, rsi_len=14, atr_len=14, vol_len=20, htf_ema=50,
    pivot_k=5,               # 轉折點：左右各 k 根都比它低/高才算
    sr_lookback_days=14,     # 回測/盯盤時，只用最近幾天的轉折點當支撐壓力
    sr_merge_pct=0.3,        # 價格差在 0.3% 內的轉折點視為同一個價位（算觸碰次數）
    min_risk_pct=0.1,        # 止損距離小於 0.1% 的單不做（太容易被雜訊掃掉）
    fee_pct=0.05, slip_pct=0.02,
    backtest_days=730, live_days=30, train_ratio=0.5,
    fund_rank_n=90,          # 資金費率百分位：跟最近 90 次（約 30 天）比
    min_winrate=70.0, min_trades=30, min_pf=1.1,   # 推播門檻
    use_win_filter=True,
)

# 沒有 best_config.json 時使用的預設策略
DEFAULT_STRAT = dict(
    mode="sweep",            # sweep 掃流動性 / bounce 支撐壓力回踩 / break 突破 / cvd_div 主動買賣差背離 / funding 資金費率極端
    liq_len=20, sweep_window=5, vol_mult=1.5, near_atr=0.3, sr_tf="1h",
    flow="none",             # 主動買賣差(近 4H)過濾：none / with 同向 / against 反向
    fund="none",             # 資金費率過濾：none / contra 逆擁擠(費率低才多、高才空) / with 順擁擠
    cvd_n=16, cvd_th=0.05, fund_q=0.1,
    rsi_long_max=60, rsi_short_min=40, htf_mode="both",
    atr_buf=0.2, rr=1.5, side="both", reverse=False,
)

TG_TOKEN = os.environ.get("TG_TOKEN", "")
TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "")
MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
HERE = os.path.dirname(os.path.abspath(__file__))
BEST_PATH = os.path.join(HERE, "best_config.json")


# ───────────────────────── 數據 ─────────────────────────
def fetch_klines(interval, days):
    if BASE["market"] == "futures":
        url, max_limit = "https://fapi.binance.com/fapi/v1/klines", 1500
    else:
        url, max_limit = "https://api.binance.com/api/v3/klines", 1000
    now = int(time.time() * 1000)
    cur = now - int(days * 86_400_000)
    rows = []
    while cur < now:
        r = requests.get(url, params=dict(symbol=BASE["symbol"], interval=interval,
                                          startTime=cur, limit=max_limit), timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        cur = data[-1][0] + 1
        if len(data) < max_limit:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 6, 9]]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "taker_buy"]
    df = df.astype({"open_time": "int64", "close_time": "int64", "open": float, "high": float,
                    "low": float, "close": float, "volume": float, "taker_buy": float})
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df[df["close_time"] < now].reset_index(drop=True)      # 只留已收盤 K 棒


def fetch_funding(days):
    """資金費率歷史（只有合約有）"""
    if BASE["market"] != "futures":
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    now = int(time.time() * 1000)
    cur = now - int(days * 86_400_000)
    rows = []
    while cur < now:
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                         params=dict(symbol=BASE["symbol"], startTime=cur, limit=1000), timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data:
            break
        rows += data
        cur = int(data[-1]["fundingTime"]) + 1
        if len(data) < 1000:
            break
        time.sleep(0.3)
    f = pd.DataFrame(rows)[["fundingTime", "fundingRate"]]
    f = f.astype({"fundingTime": "int64", "fundingRate": float})
    return f.drop_duplicates("fundingTime").sort_values("fundingTime").reset_index(drop=True)


def load(days):
    df = fetch_klines(BASE["tf"], days)
    h1 = fetch_klines(BASE["htf1"], days + 30)
    h4 = fetch_klines(BASE["htf2"], days + 60)
    fr = fetch_funding(days + 40)
    return base_features(df, h1, h4, fr)


# ───────────────────────── 指標 ─────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def htf_dir(h):
    """高週期趨勢（收盤在 EMA 上=1、下=-1），只用上一根已收盤的高週期 K 棒"""
    d = np.sign(h["close"] - ema(h["close"], BASE["htf_ema"]))
    return pd.DataFrame({"open_time": h["open_time"], "d": d.shift(1)})


def pivots(h, ms):
    """轉折高低點。c = 確認時間（右邊 k 根收完才知道是轉折，避免偷看未來）"""
    k = BASE["pivot_k"]
    w = 2 * k + 1
    ph = h["high"] == h["high"].rolling(w, center=True).max()
    pl = h["low"] == h["low"].rolling(w, center=True).min()
    conf = h["open_time"].shift(-k) + ms
    p = pd.concat([
        pd.DataFrame({"t": h["open_time"][ph], "c": conf[ph], "p": h["high"][ph], "kind": "H"}),
        pd.DataFrame({"t": h["open_time"][pl], "c": conf[pl], "p": h["low"][pl], "kind": "L"}),
    ])
    return p.dropna().sort_values("c").reset_index(drop=True)


def attach_sr(df, piv, tag):
    """每根 15m K 棒：當下已確認、最近 N 天內的轉折點中，收盤價下方最近的=支撐、上方最近的=壓力"""
    look = BASE["sr_lookback_days"] * 86_400_000
    merge = BASE["sr_merge_pct"] / 100
    c_arr, p_arr = piv["c"].values.astype("int64"), piv["p"].values
    close = df["close"].values
    bar_end = df["open_time"].values + MS[BASE["tf"]]
    n = len(df)
    sup, res = np.full(n, np.nan), np.full(n, np.nan)
    supn, resn = np.zeros(n), np.zeros(n)
    lo = hi = 0
    for i in range(n):
        while hi < len(c_arr) and c_arr[hi] <= bar_end[i]:
            hi += 1
        while lo < hi and c_arr[lo] < bar_end[i] - look:
            lo += 1
        ps = p_arr[lo:hi]
        if ps.size == 0:
            continue
        below, above = ps[ps < close[i]], ps[ps > close[i]]
        if below.size:
            s = below.max()
            sup[i], supn[i] = s, np.sum(np.abs(ps - s) <= s * merge)
        if above.size:
            r = above.min()
            res[i], resn[i] = r, np.sum(np.abs(ps - r) <= r * merge)
    df[f"sup_{tag}"], df[f"res_{tag}"] = sup, res
    df[f"supn_{tag}"], df[f"resn_{tag}"] = supn, resn


def funding_rank(fr):
    """資金費率在最近 N 次中的百分位（0=最低、1=最高）"""
    n = BASE["fund_rank_n"]
    fr = fr.copy()
    fr["fr_rank"] = fr["fundingRate"].rolling(n).apply(lambda x: (x <= x[-1]).mean(), raw=True)
    return fr


def base_features(df, h1, h4, fr):
    df = df.copy()
    hi, lo, cl, v = df["high"], df["low"], df["close"], df["volume"]
    df["macd"] = ema(cl, BASE["macd_fast"]) - ema(cl, BASE["macd_slow"])
    df["sig"] = ema(df["macd"], BASE["macd_sig"])
    delta = cl.diff()
    rs = rma(delta.clip(lower=0), BASE["rsi_len"]) / rma(-delta.clip(upper=0), BASE["rsi_len"])
    df["rsi"] = 100 - 100 / (1 + rs)
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = rma(tr, BASE["atr_len"])
    df["vol_ratio"] = v / v.rolling(BASE["vol_len"]).mean()
    delta = 2 * df["taker_buy"] - v                       # 主動買 - 主動賣
    for n in (16, 48):                                    # 16 根 = 4 小時、48 根 = 12 小時
        df[f"cvd{n}"] = delta.rolling(n).sum() / v.rolling(n).sum()   # -1 ~ +1
    if len(fr):
        f = funding_rank(fr).rename(columns={"fundingTime": "bar_end", "fundingRate": "fr"})
        df["bar_end"] = df["open_time"] + MS[BASE["tf"]]
        df = pd.merge_asof(df, f, on="bar_end").drop(columns="bar_end")
    else:
        df["fr"], df["fr_rank"] = np.nan, np.nan
    df = pd.merge_asof(df, htf_dir(h1).rename(columns={"d": "dir1"}), on="open_time")
    df = pd.merge_asof(df, htf_dir(h4).rename(columns={"d": "dir2"}), on="open_time")
    attach_sr(df, pivots(h1, MS["1h"]), "1h")
    attach_sr(df, pivots(h4, MS["4h"]), "4h")
    return df


# ───────────────────────── 訊號 ─────────────────────────
def signals(b, S):
    op, hi, lo, cl, atr, rsi = b["open"], b["high"], b["low"], b["close"], b["atr"], b["rsi"]
    T = pd.Series(True, index=b.index)
    if S["htf_mode"] == "none":
        hl = hs = T
    elif S["htf_mode"] == "first":
        hl, hs = b["dir1"] == 1, b["dir1"] == -1
    else:
        hl = (b["dir1"] == 1) & (b["dir2"] == 1)
        hs = (b["dir1"] == -1) & (b["dir2"] == -1)

    macd_up = (b["macd"] > b["sig"]) & (b["macd"].shift() <= b["sig"].shift())
    macd_dn = (b["macd"] < b["sig"]) & (b["macd"].shift() >= b["sig"].shift())
    extra = {}
    if S["mode"] == "cvd_div":
        n, th = S["cvd_n"], S["cvd_th"]
        cv = b[f"cvd{n}"]
        cl_ = (cl < cl.shift(n)) & (cv > th) & hl          # 價格跌、主動買卻偏多
        cs_ = (cl > cl.shift(n)) & (cv < -th) & hs         # 價格漲、主動賣卻偏多
        L = cl_ & ~cl_.shift(1, fill_value=False)
        Sh = cs_ & ~cs_.shift(1, fill_value=False)
        lsl = lo.rolling(n).min() - atr * S["atr_buf"]
        ssl = hi.rolling(n).max() + atr * S["atr_buf"]
    elif S["mode"] == "funding":
        q = S["fund_q"]
        L = (b["fr_rank"] <= q) & macd_up & hl
        Sh = (b["fr_rank"] >= 1 - q) & macd_dn & hs
        lsl = lo.rolling(6).min() - atr * S["atr_buf"]
        ssl = hi.rolling(6).max() + atr * S["atr_buf"]
    elif S["mode"] == "sweep":
        sw = S["sweep_window"]
        vol_ok = b["vol_ratio"].rolling(sw + 1).max() >= S["vol_mult"] if S["vol_mult"] > 0 else T
        prev_low = lo.rolling(S["liq_len"]).min().shift(1)
        prev_high = hi.rolling(S["liq_len"]).max().shift(1)
        sweep_l = (lo < prev_low) & (cl > prev_low)
        sweep_s = (hi > prev_high) & (cl < prev_high)
        idx = pd.Series(np.arange(len(b)), index=b.index, dtype=float)
        recent_l = (idx - idx.where(sweep_l).ffill()) <= sw
        recent_s = (idx - idx.where(sweep_s).ffill()) <= sw
        L = recent_l & macd_up & vol_ok & (rsi < S["rsi_long_max"]) & (rsi > rsi.shift()) & hl
        Sh = recent_s & macd_dn & vol_ok & (rsi > S["rsi_short_min"]) & (rsi < rsi.shift()) & hs
        lsl = lo.rolling(sw + 1).min() - atr * S["atr_buf"]
        ssl = hi.rolling(sw + 1).max() + atr * S["atr_buf"]
        extra = dict(swept_low=prev_low.where(sweep_l).ffill().values,
                     swept_high=prev_high.where(sweep_s).ffill().values)
    else:
        tf = S["sr_tf"]
        sup, res = b[f"sup_{tf}"], b[f"res_{tf}"]
        vol_ok = b["vol_ratio"] >= S["vol_mult"] if S["vol_mult"] > 0 else T
        if S["mode"] == "bounce":
            tol = atr * S["near_atr"]
            L = (lo <= sup + tol) & (cl > sup) & (cl > op) & (rsi < S["rsi_long_max"]) & vol_ok & hl
            Sh = (hi >= res - tol) & (cl < res) & (cl < op) & (rsi > S["rsi_short_min"]) & vol_ok & hs
            lsl = np.minimum(lo, sup) - atr * S["atr_buf"]
            ssl = np.maximum(hi, res) + atr * S["atr_buf"]
        else:  # break
            rp, sp = res.shift(), sup.shift()
            L = (cl > rp) & (cl.shift() <= rp) & vol_ok & hl
            Sh = (cl < sp) & (cl.shift() >= sp) & vol_ok & hs
            lsl = np.minimum(lo, rp) - atr * S["atr_buf"]
            ssl = np.maximum(hi, sp) + atr * S["atr_buf"]
    cv = b["cvd16"]
    if S["flow"] == "with":
        L, Sh = L & (cv > 0), Sh & (cv < 0)
    elif S["flow"] == "against":
        L, Sh = L & (cv < 0), Sh & (cv > 0)
    fk = b["fr_rank"]
    if S["fund"] == "contra":
        L, Sh = L & (fk <= 0.3), Sh & (fk >= 0.7)
    elif S["fund"] == "with":
        L, Sh = L & (fk >= 0.7), Sh & (fk <= 0.3)
    return dict(L=L.fillna(False).values.astype(bool), S=Sh.fillna(False).values.astype(bool),
                lsl=np.asarray(lsl, dtype=float), ssl=np.asarray(ssl, dtype=float), **extra)


# ───────────────────────── 模擬交易 ─────────────────────────
def simulate(b, sig, S):
    """訊號 K 棒收盤 → 下一根開盤進場；同一根同時碰到止損止盈時保守算止損。
    空手時直接跳到下一個訊號（加速）。回傳 (已完成交易, 最後一根 K 棒的新訊號或 None)"""
    o, h, l, c = (b[x].values for x in ("open", "high", "low", "close"))
    t = b["open_time"].values
    fee, slip, min_risk = BASE["fee_pct"], BASE["slip_pct"] / 100, BASE["min_risk_pct"] / 100
    L, Sh, lsl, ssl = sig["L"], sig["S"], sig["lsl"], sig["ssl"]
    cand = np.flatnonzero(L != Sh)
    trades, last, n, i = [], None, len(b), 0
    while True:
        k = np.searchsorted(cand, i)
        if k >= len(cand):
            break
        i = cand[k]
        raw = 1 if L[i] else -1
        d = (c[i] - lsl[i]) if raw == 1 else (ssl[i] - c[i])
        side = -raw if S["reverse"] else raw
        if not (np.isfinite(d) and d >= c[i] * min_risk) or \
                not (S["side"] == "both" or (S["side"] == "long") == (side == 1)):
            i += 1
            continue
        pos = dict(side=side, raw=raw, ref=c[i], sl=c[i] - side * d, tp=c[i] + side * d * S["rr"],
                   i=int(i), time=int(t[i]))
        if i == n - 1:
            last = pos
            break
        j = i + 1
        entry = o[j] * (1 + side * slip)
        sl, tp, px = pos["sl"], pos["tp"], None
        while j < n:
            if side == 1:
                if o[j] <= sl: px, reason = o[j], "止損"
                elif o[j] >= tp: px, reason = o[j], "止盈"
                elif l[j] <= sl: px, reason = sl, "止損"
                elif h[j] >= tp: px, reason = tp, "止盈"
            else:
                if o[j] >= sl: px, reason = o[j], "止損"
                elif o[j] <= tp: px, reason = o[j], "止盈"
                elif h[j] >= sl: px, reason = sl, "止損"
                elif l[j] <= tp: px, reason = tp, "止盈"
            if px is not None:
                break
            j += 1
        if px is None:          # 到資料結尾還沒出場
            break
        px *= (1 - side * slip)
        trades.append(dict(pos, entry=entry, exit=px, reason=reason,
                           ret_pct=(px / entry - 1) * side * 100 - 2 * fee))
        i = j                   # 出場那根收盤就能接新訊號
    return trades, last


def stats(trades):
    if not trades:
        return dict(n=0, winrate=0.0, pf=0.0, total=0.0, mdd=0.0, avg=0.0)
    r = np.array([x["ret_pct"] for x in trades])
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    mdd = float((np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq).max())
    return dict(n=len(r), winrate=float((r > 0).mean() * 100), pf=float(gp / gl) if gl > 0 else 99.0,
                total=float(r.sum()), mdd=mdd, avg=float(r.mean()))


def fmt_stats(name, s):
    return (f"{name}: {s['n']} 筆 | 勝率 {s['winrate']:.1f}% | PF {s['pf']:.2f} | "
            f"平均 {s['avg']:+.3f}% | 累計 {s['total']:+.2f}% | 最大回撤 {s['mdd']:.2f}%")


def describe(S):
    m = {"sweep": f"掃流動性(回看{S['liq_len']})",
         "bounce": f"{S['sr_tf']}支撐壓力回踩(容差{S['near_atr']}ATR)",
         "break": f"{S['sr_tf']}支撐壓力突破",
         "cvd_div": f"主動買賣差背離({S['cvd_n'] // 4}H,門檻{S['cvd_th']})",
         "funding": f"資金費率極端(前後{S['fund_q']:.0%})+MACD"}[S["mode"]]
    side = {"both": "多空", "long": "只做多", "short": "只做空"}[S["side"]]
    htf = {"both": "1H+4H同向", "first": "1H同向", "none": "不看高週期"}[S["htf_mode"]]
    vol = f"｜量>{S['vol_mult']}x" if S["vol_mult"] > 0 and S["mode"] in ("sweep", "bounce", "break") else ""
    flow = {"none": "", "with": "｜買賣差同向", "against": "｜買賣差反向"}[S["flow"]]
    fund = {"none": "", "contra": "｜費率逆擁擠", "with": "｜費率順擁擠"}[S["fund"]]
    return f"{m}｜{side}{'｜反向' if S['reverse'] else ''}｜{htf}{vol}{flow}{fund}｜盈虧比1:{S['rr']}"


def split_stats(trades, split_t):
    return stats([x for x in trades if x["time"] < split_t]), stats([x for x in trades if x["time"] >= split_t])


def load_strat():
    if os.path.exists(BEST_PATH):
        with open(BEST_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        for k in list(cfg):
            if k in BASE:                      # 推播門檻等基本設定也可以寫在 best_config.json
                BASE[k] = cfg.pop(k)
        return dict(DEFAULT_STRAT, **cfg), "best_config.json"
    return dict(DEFAULT_STRAT), "預設設定"


# ───────────────────────── 指令：scan ─────────────────────────
def scan_grid():
    D = DEFAULT_STRAT
    sig_grid = []
    for htf in ["both", "none"]:
        for flow in ["none", "with", "against"]:
            for fund in ["none", "contra", "with"]:
                f = dict(htf_mode=htf, flow=flow, fund=fund)
                for liq in [20, 40]:
                    for vm in [0, 1.5]:
                        sig_grid.append(dict(D, mode="sweep", liq_len=liq, vol_mult=vm, **f))
                for tf in ["1h", "4h"]:
                    sig_grid.append(dict(D, mode="bounce", sr_tf=tf, near_atr=0.3, vol_mult=0, **f))
                    for vm in [0, 1.5]:
                        sig_grid.append(dict(D, mode="break", sr_tf=tf, vol_mult=vm, **f))
        for fund in ["none", "contra", "with"]:
            for n in [16, 48]:
                for th in [0.05, 0.1]:
                    sig_grid.append(dict(D, mode="cvd_div", cvd_n=n, cvd_th=th, htf_mode=htf, fund=fund))
        for flow in ["none", "with", "against"]:
            for q in [0.1, 0.2]:
                sig_grid.append(dict(D, mode="funding", fund_q=q, htf_mode=htf, flow=flow))
    sim_grid = [dict(rr=rr, side=sd, reverse=rv)
                for rr in [0.5, 1.0, 1.5, 2.0] for sd in ["both", "long", "short"] for rv in [False, True]]
    return sig_grid, sim_grid


def scan(b=None):
    if b is None:
        print("下載數據中…")
        b = load(BASE["backtest_days"])
    split_t = int(b["open_time"].quantile(BASE["train_ratio"]))
    sig_grid, sim_grid = scan_grid()
    total = len(sig_grid) * len(sim_grid)
    print(f"測試 {total} 種組合（前 {BASE['train_ratio']:.0%} 時間選參數，後段驗證）…")
    rows, k, t0 = [], 0, time.time()
    for sp in sig_grid:
        sig = signals(b, sp)
        for sm in sim_grid:
            S = dict(sp, **sm)
            trades, _ = simulate(b, sig, S)
            tr, te = split_stats(trades, split_t)
            rows.append(dict(S=S, tr=tr, te=te))
            k += 1
        eta = (time.time() - t0) / k * (total - k)
        print(f"  進度 {k}/{total}（約剩 {eta / 60:.1f} 分鐘）   ", end="\r", flush=True)
    print()

    out = pd.DataFrame([dict(策略=describe(r["S"]),
                             前段筆數=r["tr"]["n"], 前段勝率=round(r["tr"]["winrate"], 1), 前段PF=round(r["tr"]["pf"], 2),
                             後段筆數=r["te"]["n"], 後段勝率=round(r["te"]["winrate"], 1), 後段PF=round(r["te"]["pf"], 2),
                             後段累計=round(r["te"]["total"], 2)) for r in rows])
    out.to_csv(os.path.join(HERE, "scan_results.csv"), index=False, encoding="utf-8-sig")

    cand = [r for r in rows if r["tr"]["n"] >= BASE["min_trades"] and r["tr"]["pf"] >= 1.2]
    cand.sort(key=lambda r: r["tr"]["pf"], reverse=True)
    print(f"\n前段及格（≥{BASE['min_trades']} 筆、PF≥1.2）：{len(cand)} / {total} 組")
    for r in cand[:10]:
        print(f"- {describe(r['S'])}\n    前段 {r['tr']['n']} 筆 勝率 {r['tr']['winrate']:.1f}% PF {r['tr']['pf']:.2f}"
              f"  →  後段 {r['te']['n']} 筆 勝率 {r['te']['winrate']:.1f}% PF {r['te']['pf']:.2f}")

    both_ok = [r for r in rows if min(r["tr"]["n"], r["te"]["n"]) >= 20 and min(r["tr"]["pf"], r["te"]["pf"]) >= 1.1]
    print(f"\n前後段都 ≥20 筆且 PF≥1.1：{len(both_ok)} 組（純靠運氣也會冒出一些）")
    hi_wr = [r for r in rows if r["te"]["n"] >= 20 and r["te"]["winrate"] >= 70 and r["te"]["pf"] >= 1.1
             and r["tr"]["winrate"] >= 70 and r["tr"]["pf"] >= 1.1]
    print(f"\n前後段都「勝率≥70% 且 PF≥1.1」：{len(hi_wr)} 組")

    if cand:
        best = cand[0]
        ok = best["te"]["n"] >= 20 and best["te"]["pf"] >= 1.1
        print(f"\n前段第一名：{describe(best['S'])}")
        if ok:
            keep = {k: best["S"][k] for k in DEFAULT_STRAT}
            with open(BEST_PATH, "w", encoding="utf-8") as f:
                json.dump(keep, f, ensure_ascii=False, indent=2)
            print("✅ 後段也賺錢 → 已存成 best_config.json（live / backtest 會自動使用）")
        else:
            print("❌ 後段沒過（筆數<20 或 PF<1.1）→ 不存檔。這組在前段賺錢多半是運氣。")
    else:
        print("\n❌ 沒有任何組合在前段及格 → 這些進場邏輯在 15m 上看不到優勢")
    print(f"\n⚠ 測了 {total} 組，純靠運氣也會有幾組看起來不錯；就算通過，也建議先模擬單觀察 2–4 週。")
    print("完整結果：scan_results.csv（可用 Excel 開）")
    return rows


# ───────────────────────── 指令：backtest ─────────────────────────
def run_backtest(verbose=True, b=None):
    S, src = load_strat()
    if b is None:
        b = load(BASE["backtest_days"])
    trades, _ = simulate(b, signals(b, S), S)
    s = stats(trades)
    if verbose:
        t0 = pd.to_datetime(b["open_time"].iloc[0], unit="ms")
        t1 = pd.to_datetime(b["open_time"].iloc[-1], unit="ms")
        print(f"策略（{src}）：{describe(S)}")
        print(f"期間 {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d}（UTC），{len(b)} 根 {BASE['tf']} K 棒；已扣手續費與滑點")
        print(fmt_stats("全部", s))
        tr, te = split_stats(trades, int(b["open_time"].quantile(BASE["train_ratio"])))
        print(fmt_stats("前半段", tr))
        print(fmt_stats("後半段", te))
        if tr["pf"] < 1 and te["pf"] < 1:
            print("→ 前後段都虧：這個策略沒有優勢")
        elif tr["pf"] >= 1.1 and te["pf"] < 1:
            print("→ 前段賺、後段虧：可能是過度擬合，或行情型態變了")
        if s["n"] < BASE["min_trades"]:
            print(f"⚠ 只有 {s['n']} 筆，樣本太少，勝率不可信")
    return s


# ───────────────────────── 指令：levels ─────────────────────────
def cluster_levels(piv, merge):
    ps = piv.sort_values("p")
    groups, cur = [], []
    for _, row in ps.iterrows():
        if cur and row["p"] > cur[0]["p"] * (1 + merge):
            groups.append(cur)
            cur = []
        cur.append(row)
    if cur:
        groups.append(cur)
    return [dict(p=float(np.mean([g["p"] for g in grp])), n=len(grp),
                 last=max(int(g["t"]) for g in grp)) for grp in groups]


def levels():
    price = fetch_klines(BASE["tf"], 1)["close"].iloc[-1]
    now = int(time.time() * 1000)
    merge = BASE["sr_merge_pct"] / 100
    print(f"{BASE['symbol']} 現價約 {price:,.1f}（幣安{'合約' if BASE['market'] == 'futures' else '現貨'}，最近收盤的 15m K 棒）\n")
    for tf, days in [("1h", 14), ("4h", 60), ("1d", 365)]:
        h = fetch_klines(tf, days + 5)
        piv = pivots(h, MS[tf])
        piv = piv[(piv["t"] >= now - days * 86_400_000) & (piv["c"] <= now)]
        lv = cluster_levels(piv, merge)
        above = sorted([x for x in lv if x["p"] > price], key=lambda x: x["p"])[:5]
        below = sorted([x for x in lv if x["p"] < price], key=lambda x: -x["p"])[:5]
        print(f"【{tf}，最近 {days} 天轉折點】")
        for name, arr in [("壓力", above), ("支撐", below)]:
            for x in arr:
                d = (x["p"] / price - 1) * 100
                t = pd.to_datetime(x["last"], unit="ms", utc=True).tz_convert("Asia/Taipei")
                print(f"  {name} {x['p']:>11,.1f}  距離 {d:+.2f}%  觸碰 {x['n']} 次  最近一次 {t:%m/%d %H:%M}")
        print()
    print("說明：支撐壓力是用轉折高低點算的（左右各 5 根都比它低/高）；觸碰次數越多、越近期，參考價值越高。")
    print("      價位附近 0.3% 內都算同一區，不是精準的一條線。\n")
    k = fetch_klines(BASE["tf"], 2)
    delta = 2 * k["taker_buy"] - k["volume"]
    print("【資金流向】")
    for n, name in [(16, "近 4H"), (48, "近 12H")]:
        r = delta.tail(n).sum() / k["volume"].tail(n).sum() * 100
        print(f"  主動買賣差 {name}：{r:+.1f}%（正 = 主動買比較多；-100% ~ +100%）")
    fr = fetch_funding(40)
    if len(fr):
        fr = funding_rank(fr)
        last = fr.iloc[-1]
        t = pd.to_datetime(last["fundingTime"], unit="ms", utc=True).tz_convert("Asia/Taipei")
        print(f"  資金費率（{t:%m/%d %H:%M} 結算）：{last['fundingRate'] * 100:+.4f}%，"
              f"在最近 {BASE['fund_rank_n']} 次中排第 {last['fr_rank'] * 100:.0f} 百分位")
        print("  （百分位高 = 多頭付費給空頭、做多比較擁擠；低 = 做空比較擁擠）")


# ───────────────────────── 推播 / live ─────────────────────────
def notify(text):
    print(text, flush=True)
    if TG_TOKEN and TG_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT_ID, "text": text}, timeout=10)
        except Exception as e:
            print("Telegram 發送失敗：", e)


def dir_txt(d):
    return "多" if d == 1 else "空" if d == -1 else "盤整"


def f1(x):
    return "-" if x is None or not np.isfinite(x) else f"{x:,.1f}"


def fmt_signal(sig, row, S, s, extra):
    side = "多" if sig["side"] == 1 else "空"
    t = pd.to_datetime(sig["time"], unit="ms", utc=True).tz_convert("Asia/Taipei")
    risk = abs(sig["ref"] - sig["sl"])
    raw = sig["raw"]
    if S["mode"] == "sweep":
        lvl = extra["swept_low"][sig["i"]] if raw == 1 else extra["swept_high"][sig["i"]]
        why = (f"掃{'前低' if raw == 1 else '前高'} {f1(lvl)} 後收回；MACD {'金叉' if raw == 1 else '死叉'}；"
               f"RSI {row['rsi']:.1f}；量比 {row['vol_ratio']:.1f}x")
    elif S["mode"] == "bounce":
        tf = S["sr_tf"]
        why = (f"回踩 {tf} 支撐 {f1(row[f'sup_{tf}'])}（觸碰 {int(row[f'supn_{tf}'])} 次）收回" if raw == 1 else
               f"碰 {tf} 壓力 {f1(row[f'res_{tf}'])}（觸碰 {int(row[f'resn_{tf}'])} 次）被壓回") + f"；RSI {row['rsi']:.1f}"
    elif S["mode"] == "break":
        why = f"{'突破' if raw == 1 else '跌破'} {S['sr_tf']} {'壓力' if raw == 1 else '支撐'}；量比 {row['vol_ratio']:.1f}x"
    elif S["mode"] == "cvd_div":
        cv = row["cvd" + str(S["cvd_n"])] * 100
        why = f"近 {S['cvd_n'] // 4}H 價格{'下跌' if raw == 1 else '上漲'}，主動買賣差卻是 {cv:+.1f}%"
    else:
        why = f"資金費率在第 {row['fr_rank'] * 100:.0f} 百分位（{'做空擁擠' if raw == 1 else '做多擁擠'}）；MACD {'金叉' if raw == 1 else '死叉'}"
    if S["reverse"]:
        why = f"反向操作（原訊號偏{'多' if raw == 1 else '空'}，回測顯示反做較好）：" + why
    return (f"【{side}】{BASE['symbol']} {BASE['tf']}（K 棒 {t:%m/%d %H:%M} 台北時間）\n"
            f"入場：{f1(sig['ref'])}（參考收盤價，實際以下一根開盤為準）\n"
            f"止損：{f1(sig['sl'])}（風險 {risk / sig['ref'] * 100:.2f}%）\n"
            f"止盈：{f1(sig['tp'])}（盈虧比 1:{S['rr']}）\n"
            f"原因：{why}；1H {dir_txt(row['dir1'])} / 4H {dir_txt(row['dir2'])}\n"
            f"支撐壓力：1H {f1(row['sup_1h'])} / {f1(row['res_1h'])}；4H {f1(row['sup_4h'])} / {f1(row['res_4h'])}\n"
            f"資金流向：近 4H 主動買賣差 {row['cvd16'] * 100:+.1f}%；資金費率 {row['fr'] * 100:+.4f}%"
            f"（第 {row['fr_rank'] * 100:.0f} 百分位）\n"
            f"回測勝率：{s['winrate']:.1f}%（{s['n']} 筆，PF {s['pf']:.2f}，近 {BASE['backtest_days']} 天）\n"
            f"※ 回測勝率不代表這一單的勝率")


def live():
    S, src = load_strat()
    print(f"開始盯盤（{src}）：{describe(S)}\n每根 15m 收盤後檢查一次（Ctrl+C 停止）")
    s, s_time, last_alert = None, 0, None
    while True:
        try:
            if s is None or time.time() - s_time > 86_400:
                s = run_backtest(verbose=True)
                s_time = time.time()
                passed = s["n"] >= BASE["min_trades"] and s["winrate"] >= BASE["min_winrate"] and s["pf"] >= BASE["min_pf"]
                if BASE["use_win_filter"] and not passed:
                    print(f"⚠ 回測未達門檻（勝率≥{BASE['min_winrate']}%、PF≥{BASE['min_pf']}、≥{BASE['min_trades']} 筆），"
                          f"有訊號也不會推播")
            b = load(BASE["live_days"])
            sig_d = signals(b, S)
            _, sig = simulate(b, sig_d, S)
            if sig and sig["time"] != last_alert:
                last_alert = sig["time"]
                passed = s["n"] >= BASE["min_trades"] and s["winrate"] >= BASE["min_winrate"] and s["pf"] >= BASE["min_pf"]
                if passed or not BASE["use_win_filter"]:
                    notify(fmt_signal(sig, b.iloc[sig["i"]], S, s, sig_d))
                else:
                    print(time.strftime("%H:%M:%S"), "有訊號但回測未達門檻，不推播")
            else:
                print(time.strftime("%H:%M:%S"), "無訊號", flush=True)
        except Exception as e:
            print("錯誤：", e, flush=True)
        step = MS[BASE["tf"]] / 1000
        time.sleep(step - time.time() % step + 5)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if len(sys.argv) > 2:                      # 例：python btc_signal_bot.py backtest 730
        BASE["backtest_days"] = int(sys.argv[2])
    if cmd == "scan":
        scan()
    elif cmd == "backtest":
        run_backtest()
    elif cmd == "levels":
        levels()
    elif cmd == "live":
        live()
    elif cmd == "test":
        notify("測試訊息：BTC 盯盤機器人連線成功")
    else:
        print(__doc__)
