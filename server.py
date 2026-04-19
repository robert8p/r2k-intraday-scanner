import os, json, time, math, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import httpx
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("scanner")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
ALPACA_KEY    = os.environ.get("ALPACA_API_KEY","")
ALPACA_SECRET = os.environ.get("ALPACA_API_SECRET","")
ALPACA_URL    = os.environ.get("ALPACA_DATA_URL","https://data.alpaca.markets")
DATA_DIR      = Path("/data") if Path("/data").exists() else Path(__file__).parent / ".data"
MODEL_DIR     = DATA_DIR / "models"
OUTCOME_DIR   = DATA_DIR / "outcomes"
SCAN_DIR      = DATA_DIR / "scans"
PORT          = int(os.environ.get("PORT", 10000))

ET = ZoneInfo("America/New_York")
SCAN_HOURS = [11, 12, 13, 14, 15]  # 10:00 dropped: not enough bar history for RSI/trendStr
# v8: Volatility-adjusted barriers. TP/SL are expressed as multipliers of
# 14-day ATR (Average True Range, as a fraction of most recent close). So a
# stock with 2% ATR and TP_MULT=0.5 gets TP at +1.0%; SL_MULT=2.5 gets SL at -5.0%.
# This equalizes the statistical barrier probability across stocks with wildly
# different volatilities. TP_PCT/SL_PCT are kept for back-compat in places that
# still reference them, but first-passage labels use ATR-scaled barriers.
TP_MULT = 0.5    # TP = entry * (1 + TP_MULT * atr_pct)
SL_MULT = 2.5    # SL = entry * (1 - SL_MULT * atr_pct)
ATR_LOOKBACK_DAYS = 14  # daily ATR window
TP_PCT = 0.01    # legacy placeholder — 1% nominal, only used in atr_reach feature
SL_PCT = 0.05    # legacy placeholder
FORCED_CLOSE_MIN = 15*60+55  # 15:55 ET in minutes

# Universe: ~300 liquid Russell 2000 names across sectors.
# Names where inclusion is uncertain will simply be skipped at scan time if Alpaca
# returns insufficient data. Sector mapping is approximate/GICS-inspired.
TICKERS = [
    # Financials - Banks / Regionals
    "WAL","PACW","WTFC","WAFD","BANF","CVBF","ONB","TCBI","UCBI","TRMK",
    "HOMB","SFBS","FFIN","FULT","HWC","BPOP","SBNY","VLY","AUB","INDB",
    "CFR","PRK","GBCI","PNFP","SFNC","TOWN","WSBC","BANC","COLB","FIBK",
    # Financials - Insurance / Specialty
    "SIGI","KMPR","PLMR","JXN","HRTG","EG","PRA","UFCS","IGIC","BRO",
    "RLI","GSHD","AJG","WRB","ASB","PFG","GLRE","NMIH","ESGR","MKL",
    # Industrials
    "AAON","BLBD","BMI","SNDR","ARCB","HUBG","SAIA","MATX","WCC","ATKR",
    "TREX","AZZ","MTZ","PWR","DY","PRIM","TPC","MYRG","TNC","ESE",
    "GTES","KAI","ALG","LXFR","B","MOG.A","HEES","CIR","ROCK","UFPI",
    # Technology
    "CALX","DIOD","VIAV","ACLS","UCTT","ICHR","ONTO","PLXS","BHE","SANM",
    "TTMI","OSIS","MKSI","CRUS","SLAB","POWI","QLYS","NTCT","PDFS","CEVA",
    "PAYO","DOCN","ALRM","EVCM","NABL","YEXT","SMTC","RAMP","SUMO","KN",
    # Consumer Discretionary
    "BOOT","BJRI","CAKE","DNUT","DRVN","CROX","HBI","KTB","OXM","STMP",
    "SHOO","WWW","CAL","DECK","MODG","SCVL","HZO","MCFT","MBUU","PATK",
    "FOXF","THRM","LCII","KMT","WGO","LEG","TMHC","LGIH","MHO","MTH",
    # Consumer Staples
    "SMPL","CENT","CENTA","NAPA","HAIN","USFD","CHEF","SFM","JJSF","UVV",
    "VLGEA","WMK","NATR","EDBL","FARM","LWAY","LANC","PPC","THS","FLO",
    # Health Care - Biotech / Pharma
    "HALO","EXEL","UTHR","JAZZ","PTCT","ACAD","ARWR","BBIO","BCRX","BPMC",
    "FOLD","IONS","NBIX","PCVX","PRTA","RYTM","SAGE","TGTX","VRNA","INSM",
    "NRIX","VCYT","HRMY","ANIP","AMPH","PHAR","INVA","CORT","PACB","TWST",
    # Health Care - Equipment / Services
    "HAE","TFX","MMSI","NEOG","LMAT","ATRC","GKOS","AXGN","NVST","EMBC",
    "SHC","OMCL","PRCT","SILK","BFLY","CSTL","CERT","PAHC","GH","LIVN",
    "TNDM","INMD","IRMD","CPRX","PGNY","HIMS","LFST","EHC","RDNT","MD",
    # Energy
    "MTDR","MGY","CRC","PR","CHRD","HPK","SM","CIVI","CPE","GPOR",
    "REI","VTLE","BRY","NOG","TALO","KOS","MUR","WTI","VAL","HP",
    "PTEN","LBRT","RES","CLB","DRQ","NINE","OII","HLX","NE","RIG",
    # Materials
    "CMP","IOSP","SXT","KOP","HWKN","USLM","CMC","SCHN","WOR",
    "HAYN","ATI","BOOM","SMG","TROX","CENX","MTRN","UEC","MP","ASIX",
    # Real Estate
    "STAG","EPRT","PECO","IRT","NXRT","INN","APLE","XHR","CHH","PEB",
    "DRH","RLJ","SHO","BRX","ROIC","KRG","MAC","SKT","TNL","ESRT",
    "HPP","PDM","NHI","LTC","CTRE","SBRA","DOC","GMRE","GNL","OHI",
    # Utilities
    "PNW","IDA","POR","BKH","OGE","NWE","MGEE","ALE","AVA","NWN",
    "SR","SWX","SJI","UGI","CHX","ORA","WR","NJR","ES","NI",
    # Communication / Media
    "SBGI","TGNA","GCI","ATUS","CABO","SATS","LGF.A","AMCX","DIS","FUBO"
]

# Approximate sector map. Names not in the dict fall back to "?" and still work.
SECTORS = {
    "WAL":"Financial","PACW":"Financial","WTFC":"Financial","WAFD":"Financial","BANF":"Financial",
    "CVBF":"Financial","ONB":"Financial","TCBI":"Financial","UCBI":"Financial","TRMK":"Financial",
    "HOMB":"Financial","SFBS":"Financial","FFIN":"Financial","FULT":"Financial","HWC":"Financial",
    "BPOP":"Financial","SBNY":"Financial","VLY":"Financial","AUB":"Financial","INDB":"Financial",
    "CFR":"Financial","PRK":"Financial","GBCI":"Financial","PNFP":"Financial","SFNC":"Financial",
    "TOWN":"Financial","WSBC":"Financial","BANC":"Financial","COLB":"Financial","FIBK":"Financial",
    "SIGI":"Financial","KMPR":"Financial","PLMR":"Financial","JXN":"Financial","HRTG":"Financial",
    "EG":"Financial","PRA":"Financial","UFCS":"Financial","IGIC":"Financial","BRO":"Financial",
    "RLI":"Financial","GSHD":"Financial","AJG":"Financial","WRB":"Financial","ASB":"Financial",
    "PFG":"Financial","GLRE":"Financial","NMIH":"Financial","ESGR":"Financial","MKL":"Financial",
    "AAON":"Industrial","BLBD":"Industrial","BMI":"Industrial","SNDR":"Industrial","ARCB":"Industrial",
    "HUBG":"Industrial","SAIA":"Industrial","MATX":"Industrial","WCC":"Industrial","ATKR":"Industrial",
    "TREX":"Industrial","AZZ":"Industrial","MTZ":"Industrial","PWR":"Industrial","DY":"Industrial",
    "PRIM":"Industrial","TPC":"Industrial","MYRG":"Industrial","TNC":"Industrial","ESE":"Industrial",
    "GTES":"Industrial","KAI":"Industrial","ALG":"Industrial","LXFR":"Industrial","B":"Industrial",
    "MOG.A":"Industrial","HEES":"Industrial","CIR":"Industrial","ROCK":"Industrial","UFPI":"Industrial",
    "CALX":"Tech","DIOD":"Tech","VIAV":"Tech","ACLS":"Tech","UCTT":"Tech",
    "ICHR":"Tech","ONTO":"Tech","PLXS":"Tech","BHE":"Tech","SANM":"Tech",
    "TTMI":"Tech","OSIS":"Tech","MKSI":"Tech","CRUS":"Tech","SLAB":"Tech",
    "POWI":"Tech","QLYS":"Tech","NTCT":"Tech","PDFS":"Tech","CEVA":"Tech",
    "PAYO":"Tech","DOCN":"Tech","ALRM":"Tech","EVCM":"Tech","NABL":"Tech",
    "YEXT":"Tech","SMTC":"Tech","RAMP":"Tech","SUMO":"Tech","KN":"Tech",
    "BOOT":"Consumer","BJRI":"Consumer","CAKE":"Consumer","DNUT":"Consumer","DRVN":"Consumer",
    "CROX":"Consumer","HBI":"Consumer","KTB":"Consumer","OXM":"Consumer","STMP":"Consumer",
    "SHOO":"Consumer","WWW":"Consumer","CAL":"Consumer","DECK":"Consumer","MODG":"Consumer",
    "SCVL":"Consumer","HZO":"Consumer","MCFT":"Consumer","MBUU":"Consumer","PATK":"Consumer",
    "FOXF":"Consumer","THRM":"Consumer","LCII":"Consumer","KMT":"Consumer","WGO":"Consumer",
    "LEG":"Consumer","TMHC":"Consumer","LGIH":"Consumer","MHO":"Consumer","MTH":"Consumer",
    "SMPL":"Staples","CENT":"Staples","CENTA":"Staples","NAPA":"Staples","HAIN":"Staples",
    "USFD":"Staples","CHEF":"Staples","SFM":"Staples","JJSF":"Staples","UVV":"Staples",
    "VLGEA":"Staples","WMK":"Staples","NATR":"Staples","EDBL":"Staples","FARM":"Staples",
    "LWAY":"Staples","LANC":"Staples","PPC":"Staples","THS":"Staples","FLO":"Staples",
    "HALO":"Health","EXEL":"Health","UTHR":"Health","JAZZ":"Health","PTCT":"Health",
    "ACAD":"Health","ARWR":"Health","BBIO":"Health","BCRX":"Health","BPMC":"Health",
    "FOLD":"Health","IONS":"Health","NBIX":"Health","PCVX":"Health","PRTA":"Health",
    "RYTM":"Health","SAGE":"Health","TGTX":"Health","VRNA":"Health","INSM":"Health",
    "NRIX":"Health","VCYT":"Health","HRMY":"Health","ANIP":"Health","AMPH":"Health",
    "PHAR":"Health","INVA":"Health","CORT":"Health","PACB":"Health","TWST":"Health",
    "HAE":"Health","TFX":"Health","MMSI":"Health","NEOG":"Health","LMAT":"Health",
    "ATRC":"Health","GKOS":"Health","AXGN":"Health","NVST":"Health","EMBC":"Health",
    "SHC":"Health","OMCL":"Health","PRCT":"Health","SILK":"Health","BFLY":"Health",
    "CSTL":"Health","CERT":"Health","PAHC":"Health","GH":"Health","LIVN":"Health",
    "TNDM":"Health","INMD":"Health","IRMD":"Health","CPRX":"Health","PGNY":"Health",
    "HIMS":"Health","LFST":"Health","EHC":"Health","RDNT":"Health","MD":"Health",
    "MTDR":"Energy","MGY":"Energy","CRC":"Energy","PR":"Energy","CHRD":"Energy",
    "HPK":"Energy","SM":"Energy","CIVI":"Energy","CPE":"Energy","GPOR":"Energy",
    "REI":"Energy","VTLE":"Energy","BRY":"Energy","NOG":"Energy","TALO":"Energy",
    "KOS":"Energy","MUR":"Energy","WTI":"Energy","VAL":"Energy","HP":"Energy",
    "PTEN":"Energy","LBRT":"Energy","RES":"Energy","CLB":"Energy","DRQ":"Energy",
    "NINE":"Energy","OII":"Energy","HLX":"Energy","NE":"Energy","RIG":"Energy",
    "CMP":"Materials","IOSP":"Materials","SXT":"Materials","KOP":"Materials","HWKN":"Materials",
    "ORA":"Utilities","USLM":"Materials","CMC":"Materials","SCHN":"Materials","WOR":"Materials",
    "HAYN":"Materials","ATI":"Materials","BOOM":"Materials","SMG":"Materials","TROX":"Materials",
    "CENX":"Materials","MTRN":"Materials","UEC":"Materials","MP":"Materials","ASIX":"Materials",
    "STAG":"RealEstate","EPRT":"RealEstate","PECO":"RealEstate","IRT":"RealEstate","NXRT":"RealEstate",
    "INN":"RealEstate","APLE":"RealEstate","XHR":"RealEstate","CHH":"RealEstate","PEB":"RealEstate",
    "DRH":"RealEstate","RLJ":"RealEstate","SHO":"RealEstate","BRX":"RealEstate","ROIC":"RealEstate",
    "KRG":"RealEstate","MAC":"RealEstate","SKT":"RealEstate","TNL":"RealEstate","ESRT":"RealEstate",
    "HPP":"RealEstate","PDM":"RealEstate","NHI":"RealEstate","LTC":"RealEstate","CTRE":"RealEstate",
    "SBRA":"RealEstate","DOC":"RealEstate","GMRE":"RealEstate","GNL":"RealEstate","OHI":"RealEstate",
    "PNW":"Utilities","IDA":"Utilities","POR":"Utilities","BKH":"Utilities","OGE":"Utilities",
    "NWE":"Utilities","MGEE":"Utilities","ALE":"Utilities","AVA":"Utilities","NWN":"Utilities",
    "SR":"Utilities","SWX":"Utilities","SJI":"Utilities","UGI":"Utilities","CHX":"Utilities",
    "WR":"Utilities","NJR":"Utilities","ES":"Utilities","NI":"Utilities",
    "SBGI":"Comms","TGNA":"Comms","GCI":"Comms","ATUS":"Comms","CABO":"Comms",
    "SATS":"Comms","LGF.A":"Comms","AMCX":"Comms","DIS":"Comms","FUBO":"Comms"
}

FEATURE_NAMES = [
    # Bar-data features (12)
    "momentum","ret_from_open","rel_volume","vwap_dist","vwap_slope",
    "orb_strength","atr_reach","realized_vol","trend_str","rsi","range_expansion",
    "hours_left",
    # SPY-relative features (5) — how this stock moves vs the broad market
    "spy_ret","ret_vs_spy","spy_momentum","mom_vs_spy","spy_vol",
    # Sector-relative features (3) — how this stock moves vs its sector peers
    "ret_vs_sector","sector_breadth","mom_vs_sector",
    # Gap features (2) — overnight gap behavior
    "gap_pct","gap_filled",
    # Cross-sectional ranks (11)
    "rank_momentum","rank_ret","rank_volume","rank_vwap","rank_slope",
    "rank_orb","rank_atr_inv","rank_vol","rank_trend","rank_rsi","rank_range"
]

for d in [DATA_DIR, MODEL_DIR, OUTCOME_DIR, SCAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════
models = {}
calibrators = {}
model_meta = {}
last_scans = {}
training_in_progress = False
training_progress = {"phase":"idle","pct":0,"message":""}

# v17: extend history (fetch older bars to enrich cache without retraining)
extend_history_in_progress = False
extend_history_progress = {"phase":"idle","pct":0,"message":""}
sweep_in_progress = False
sweep_progress = {"phase":"idle","current":0,"total":0,"message":"","currentTP":None,"currentSL":None}

STATUS_PATH = DATA_DIR / "status.json"
SWEEP_RESULTS_PATH = DATA_DIR / "sweep_results.json"

def load_sweep_results():
    try: return json.loads(SWEEP_RESULTS_PATH.read_text())
    except: return {"grid":[],"startedAt":None,"completedAt":None}
def save_sweep_results(r): SWEEP_RESULTS_PATH.write_text(json.dumps(r,indent=2))
def load_status():
    try: return json.loads(STATUS_PATH.read_text())
    except: return {"trained":False,"trainDate":None,"outcomeDays":0,"daysSinceRetrain":0}
def save_status(s): STATUS_PATH.write_text(json.dumps(s,indent=2))
status = load_status()

def load_models():
    global models, calibrators, model_meta
    for h in SCAN_HOURS:
        mp = MODEL_DIR / f"model_{h}.txt"
        cp = MODEL_DIR / f"calibrator_{h}.pkl"
        mtp = MODEL_DIR / f"meta_{h}.json"
        if mp.exists():
            models[h] = lgb.Booster(model_file=str(mp))
            log.info(f"Loaded model {h}:00")
        if cp.exists():
            calibrators[h] = pickle.loads(cp.read_bytes())
        if mtp.exists():
            model_meta[h] = json.loads(mtp.read_text())
load_models()

# ─── v7 migration: clear analysis artifacts generated with old two-way split ──
# Threshold, patterns, and sensitivity results from v6 and earlier assumed an
# 80/20 train/val split. v7 uses 60/20/20 train/val/test. Old results are
# incompatible and must not be used for live gating.
_V7_MIGRATION_MARKER = DATA_DIR / ".v7_migrated"
if not _V7_MIGRATION_MARKER.exists():
    for stale in ["threshold_results.json", "patterns.json", "sensitivity_results.json"]:
        p = DATA_DIR / stale
        if p.exists():
            try: p.unlink(); log.info(f"v7 migration: cleared stale {stale}")
            except Exception as e: log.warning(f"Could not remove {stale}: {e}")
    try: _V7_MIGRATION_MARKER.write_text(datetime.now(ET).isoformat())
    except: pass

# ─── v8 migration: clear training_rows cache (incompatible labels) ────────────
# v7 and earlier produced labels via fixed TP/SL (e.g. ±1%/±5%).
# v8 produces labels via ATR-scaled barriers. The persisted rows cache
# contains old labels which cannot be used by v8. Force retraining from bars.
# Also clear model files since their labels assumption has changed.
# Note: CACHE_DIR is defined later in the file, so we construct the path inline here.
_V8_MIGRATION_MARKER = DATA_DIR / ".v8_migrated"
if not _V8_MIGRATION_MARKER.exists():
    # Clear training rows, sweep, threshold, and other model-output artifacts
    rows_cache = DATA_DIR / "cache" / "training_rows.pkl"
    if rows_cache.exists():
        try: rows_cache.unlink(); log.info("v8 migration: cleared training_rows.pkl (labels incompatible)")
        except Exception as e: log.warning(f"Could not remove training_rows.pkl: {e}")
    for stale in ["threshold_results.json", "sweep_results.json", "patterns.json", "sensitivity_results.json"]:
        p = DATA_DIR / stale
        if p.exists():
            try: p.unlink(); log.info(f"v8 migration: cleared {stale}")
            except Exception as e: log.warning(f"Could not remove {stale}: {e}")
    # Clear model files — labels assumption has changed
    if MODEL_DIR.exists():
        for mp in MODEL_DIR.glob("*"):
            try: mp.unlink(); log.info(f"v8 migration: cleared {mp.name}")
            except Exception as e: log.warning(f"Could not remove {mp.name}: {e}")
    # Clear loaded in-memory state
    models.clear(); calibrators.clear(); model_meta.clear()
    status["trained"] = False
    save_status(status)
    try: _V8_MIGRATION_MARKER.write_text(datetime.now(ET).isoformat())
    except: pass
    log.info("v8 migration complete. Retrain required.")
# ─────────────────────────────────────────────────────────────────────────────

LAST_SCAN_PATH = DATA_DIR / "last_scans.json"
try: last_scans = json.loads(LAST_SCAN_PATH.read_text())
except: last_scans = {}

# ═══════════════════════════════════════════════════════════════════
# TIME / ALPACA HELPERS
# ═══════════════════════════════════════════════════════════════════
def now_et(): return datetime.now(ET)
def today_et(): return now_et().strftime("%Y-%m-%d")
def hour_et(): return now_et().hour
def market_open():
    n = now_et()
    if n.weekday() >= 5: return False
    return 570 <= n.hour*60+n.minute <= 960
def has_creds():
    return bool(ALPACA_KEY and ALPACA_SECRET and ALPACA_KEY != "your_alpaca_api_key_here")

def sleep(ms): import time as t; t.sleep(ms)
def chunk(a,n):
    o=[]
    for i in range(0,len(a),n): o.append(a[i:i+n])
    return o

def alpaca_client():
    return httpx.Client(base_url=ALPACA_URL,
        headers={"APCA-API-KEY-ID":ALPACA_KEY,"APCA-API-SECRET-KEY":ALPACA_SECRET},
        timeout=30.0)

def fetch_bars(client, symbols, timeframe, start, end):
    all_bars = defaultdict(list)
    for batch in chunk(symbols, 50):
        syms = ",".join(batch)
        pt = None; pg = 0
        while True:
            params = {"symbols":syms,"timeframe":timeframe,"start":start,"end":end,
                      "limit":"10000","adjustment":"split","feed":"sip","sort":"asc"}
            if pt: params["page_token"] = pt
            r = client.get("/v2/stocks/bars", params=params)
            if r.status_code == 429: time.sleep(3); continue
            r.raise_for_status()
            data = r.json()
            for sym, bars in (data.get("bars") or {}).items():
                all_bars[sym].extend(bars)
            pt = data.get("next_page_token"); pg += 1
            if not pt or pg > 100: break
            time.sleep(0.25)
        time.sleep(0.3)
    return dict(all_bars)

def fetch_snapshots(client, symbols):
    snaps = {}
    for batch in chunk(symbols, 100):
        r = client.get("/v2/stocks/snapshots", params={"symbols":",".join(batch),"feed":"sip"})
        if r.status_code == 200: snaps.update(r.json())
        time.sleep(0.3)
    return snaps

def bar_to_et_minutes(b):
    """Convert bar timestamp to ET minutes-since-midnight."""
    try:
        dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(ET)
        return dt.hour * 60 + dt.minute
    except:
        return None

# ═══════════════════════════════════════════════════════════════════
# FIRST-PASSAGE LABEL: does price hit TP before SL?
# ═══════════════════════════════════════════════════════════════════
def compute_atr_pct(daily_bars, lookback=ATR_LOOKBACK_DAYS):
    """
    Compute Average True Range as a fraction of the most recent close.
    `daily_bars` is a list of daily bar dicts (keys: o, h, l, c, v, t).
    Needs lookback+1 bars minimum (for the prior close in true range calc).
    Returns None if insufficient data.

    True Range = max(high-low, |high - prior close|, |low - prior close|)
    ATR = mean of last `lookback` true ranges
    atr_pct = ATR / most_recent_close
    """
    if not daily_bars or len(daily_bars) < lookback + 1:
        return None
    trs = []
    for i in range(1, len(daily_bars)):
        b, pc = daily_bars[i], daily_bars[i-1]["c"]
        if pc <= 0: continue
        tr = max(b["h"] - b["l"], abs(b["h"] - pc), abs(b["l"] - pc))
        trs.append(tr)
    if len(trs) < lookback: return None
    atr = float(np.mean(trs[-lookback:]))
    last_close = daily_bars[-1]["c"]
    if last_close <= 0 or atr <= 0: return None
    return atr / last_close

def compute_trade_outcome(entry_price, bars_after_entry, tp_pct=None, sl_pct=None):
    """
    Walk bars in order. Check each bar against TP/SL barriers.
    tp_pct/sl_pct default to global TP_PCT/SL_PCT if not provided.

    Returns: (outcome, pnl_pct, exit_reason)
    """
    if tp_pct is None: tp_pct = TP_PCT
    if sl_pct is None: sl_pct = SL_PCT
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    for b in bars_after_entry:
        bmin = bar_to_et_minutes(b)

        # Force close at 15:55
        if bmin is not None and bmin >= FORCED_CLOSE_MIN:
            pnl = (b["c"] - entry_price) / entry_price
            return (1 if pnl > 0 else 0, round(pnl * 100, 3), "close_15:55")

        hit_tp = b["h"] >= tp_price
        hit_sl = b["l"] <= sl_price

        if hit_tp and hit_sl:
            # Both barriers in same bar — use open to disambiguate
            if b["o"] >= entry_price:
                return (1, round(tp_pct * 100, 3), "tp")
            else:
                return (0, round(-sl_pct * 100, 3), "sl")
        elif hit_tp:
            return (1, round(tp_pct * 100, 3), "tp")
        elif hit_sl:
            return (0, round(-sl_pct * 100, 3), "sl")

    # No barrier hit, no 15:55 bar — use last bar close
    if bars_after_entry:
        pnl = (bars_after_entry[-1]["c"] - entry_price) / entry_price
        return (1 if pnl > 0 else 0, round(pnl * 100, 3), "eod")
    return (0, 0.0, "no_data")

# ═══════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════
def compute_spy_context(spy_bars):
    """
    Compute SPY-level features from SPY bars up to scan time.
    These are the SAME for every stock at a given scan — they describe
    the market environment, not the individual stock.
    Returns dict or None if insufficient data.
    """
    if len(spy_bars) < 3:
        return {"spy_ret":0,"spy_momentum":0,"spy_vol":0}
    spy_open = spy_bars[0]["o"]
    spy_current = spy_bars[-1]["c"]
    spy_ret = (spy_current - spy_open) / spy_open if spy_open > 0 else 0

    tail = spy_bars[-3:]
    spy_momentum = (tail[-1]["c"] - tail[0]["o"]) / tail[0]["o"] if tail[0]["o"] > 0 else 0

    spy_rets = [math.log(spy_bars[i]["c"]/spy_bars[i-1]["c"])
                for i in range(1,len(spy_bars)) if spy_bars[i-1]["c"]>0]
    spy_vol = np.std(spy_rets)*math.sqrt(78) if len(spy_rets)>1 else 0

    return {"spy_ret":spy_ret, "spy_momentum":spy_momentum, "spy_vol":spy_vol}

def compute_features(bars, daily_bars, current_price, open_price, scan_hour,
                     spy_context=None, prev_close=None):
    """
    Compute per-stock features. Now includes SPY-relative and gap features.
    spy_context: dict from compute_spy_context (same for all stocks at this scan)
    prev_close: previous day's closing price for gap calculation
    """
    if len(bars) < 15: return None
    hours_left = 16 - scan_hour

    # ─── Original bar-data features ──────────────────────────────
    tail = bars[-3:]
    momentum = (tail[-1]["c"] - tail[0]["o"]) / tail[0]["o"] if tail[0]["o"] > 0 else 0
    ret_from_open = (current_price - open_price) / open_price if open_price > 0 else 0

    avg_bv = sum(b["v"] for b in bars) / len(bars)
    rel_volume = 1.0
    if daily_bars and len(daily_bars) >= 2:
        adv = sum(d["v"] for d in daily_bars[-5:]) / min(5, len(daily_bars))
        exp = adv / 78
        if exp > 0: rel_volume = avg_bv / exp

    vn = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars)
    vd = sum(b["v"] for b in bars)
    vwap = vn/vd if vd > 0 else current_price
    vwap_dist = (current_price - vwap) / vwap if vwap > 0 else 0

    vwap_slope = 0.0
    if len(bars) >= 6:
        t = len(bars)//3
        n1 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t])
        d1 = sum(b["v"] for b in bars[:t])
        n2 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t*2])
        d2 = sum(b["v"] for b in bars[:t*2])
        v1, v2 = (n1/d1 if d1>0 else current_price), (n2/d2 if d2>0 else current_price)
        vwap_slope = (v2-v1)/v1 if v1>0 else 0

    orb = bars[:min(6,len(bars))]
    orb_h, orb_l = max(b["h"] for b in orb), min(b["l"] for b in orb)
    orb_range = orb_h - orb_l
    orb_strength = (current_price - orb_h)/orb_range if orb_range > 0 else 0

    atr = current_price * 0.015
    if daily_bars and len(daily_bars) >= 5:
        trs = [max(daily_bars[i]["h"]-daily_bars[i]["l"],
                    abs(daily_bars[i]["h"]-daily_bars[i-1]["c"]),
                    abs(daily_bars[i]["l"]-daily_bars[i-1]["c"]))
               for i in range(1, len(daily_bars))]
        atr = np.mean(trs[-5:])
    target = current_price * TP_PCT
    atr_scaled = atr * math.sqrt(hours_left/6.5) if hours_left > 0 else atr*0.1
    atr_reach = target/atr_scaled if atr_scaled > 0 else 2.0

    rets = [math.log(bars[i]["c"]/bars[i-1]["c"]) for i in range(1,len(bars)) if bars[i-1]["c"]>0]
    realized_vol = np.std(rets)*math.sqrt(78) if len(rets)>1 else 0

    trend_str = 0.0
    if len(bars) >= 10:
        half = len(bars)//2
        trend_str = (np.mean([b["c"] for b in bars[-half:]]) / np.mean([b["c"] for b in bars[:half]]) - 1)

    rsi = 50.0
    if len(bars) >= 15:
        gains = [max(0, bars[i]["c"]-bars[i-1]["c"]) for i in range(len(bars)-14, len(bars))]
        losses = [max(0, bars[i-1]["c"]-bars[i]["c"]) for i in range(len(bars)-14, len(bars))]
        ag, al = np.mean(gains), np.mean(losses)
        rsi = 100 - (100/(1+ag/al)) if al > 0 else 100

    last_r = (bars[-1]["h"]-bars[-1]["l"])/bars[-1]["c"] if bars[-1]["c"]>0 else 0
    avg_r = np.mean([(b["h"]-b["l"])/b["c"] for b in bars[-10:] if b["c"]>0]) or 1
    range_expansion = last_r/avg_r if avg_r > 0 else 1

    # ─── SPY-relative features ───────────────────────────────────
    sc = spy_context or {"spy_ret":0,"spy_momentum":0,"spy_vol":0}
    spy_ret = sc["spy_ret"]
    ret_vs_spy = ret_from_open - spy_ret
    spy_momentum_val = sc["spy_momentum"]
    mom_vs_spy = momentum - spy_momentum_val
    spy_vol = sc["spy_vol"]

    # ─── Gap features ────────────────────────────────────────────
    gap_pct = 0.0
    gap_filled = 0
    if prev_close and prev_close > 0:
        gap_pct = (open_price - prev_close) / prev_close
        # Gap filled = price has returned to prev close level
        if gap_pct > 0:  # gapped up
            gap_filled = 1 if min(b["l"] for b in bars) <= prev_close else 0
        elif gap_pct < 0:  # gapped down
            gap_filled = 1 if max(b["h"] for b in bars) >= prev_close else 0

    # ─── Sector-relative features (placeholders — filled by add_sector_relative) ──
    # These get overwritten in the cross-sectional pass
    ret_vs_sector = 0.0
    sector_breadth = 0.5
    mom_vs_sector = 0.0

    return {
        "momentum":momentum,"ret_from_open":ret_from_open,"rel_volume":rel_volume,
        "vwap_dist":vwap_dist,"vwap_slope":vwap_slope,"orb_strength":orb_strength,
        "atr_reach":atr_reach,"realized_vol":realized_vol,"trend_str":trend_str,
        "rsi":rsi,"range_expansion":range_expansion,"hours_left":hours_left,
        # SPY-relative
        "spy_ret":spy_ret,"ret_vs_spy":ret_vs_spy,
        "spy_momentum":spy_momentum_val,"mom_vs_spy":mom_vs_spy,"spy_vol":spy_vol,
        # Sector-relative (placeholders)
        "ret_vs_sector":ret_vs_sector,"sector_breadth":sector_breadth,
        "mom_vs_sector":mom_vs_sector,
        # Gap
        "gap_pct":gap_pct,"gap_filled":gap_filled
    }

def add_ranks(features_list):
    n = len(features_list)
    if n < 2: return features_list
    def pr(vals):
        arr = np.array(vals); o = arr.argsort().argsort()
        return o / (n-1)
    ranks = {
        "rank_momentum": pr([f["momentum"] for f in features_list]),
        "rank_ret":      pr([f["ret_from_open"] for f in features_list]),
        "rank_volume":   pr([f["rel_volume"] for f in features_list]),
        "rank_vwap":     pr([f["vwap_dist"] for f in features_list]),
        "rank_slope":    pr([f["vwap_slope"] for f in features_list]),
        "rank_orb":      pr([f["orb_strength"] for f in features_list]),
        "rank_atr_inv":  pr([-f["atr_reach"] for f in features_list]),
        "rank_vol":      pr([f["realized_vol"] for f in features_list]),
        "rank_trend":    pr([f["trend_str"] for f in features_list]),
        "rank_rsi":      pr([50-abs(f["rsi"]-55) for f in features_list]),
        "rank_range":    pr([f["range_expansion"] for f in features_list]),
    }
    for i in range(n):
        for k, v in ranks.items(): features_list[i][k] = float(v[i])
    return features_list

def add_sector_relative(features_list, sector_list):
    """
    Compute sector-relative features: how each stock compares to its sector peers.
    features_list: list of feature dicts (one per stock)
    sector_list: list of sector strings (same length, same order)
    """
    n = len(features_list)
    if n < 2: return features_list

    # Group indices by sector
    sector_indices = defaultdict(list)
    for i, sec in enumerate(sector_list):
        sector_indices[sec].append(i)

    for i in range(n):
        sec = sector_list[i]
        peers = sector_indices[sec]
        if len(peers) < 2:
            # Not enough peers — leave defaults (0, 0.5, 0)
            continue

        # Sector average return from open (excluding self)
        peer_rets = [features_list[j]["ret_from_open"] for j in peers if j != i]
        if peer_rets:
            sector_avg_ret = np.mean(peer_rets)
            features_list[i]["ret_vs_sector"] = features_list[i]["ret_from_open"] - sector_avg_ret

        # Sector breadth: fraction of peers positive from open
        positive = sum(1 for j in peers if features_list[j]["ret_from_open"] > 0)
        features_list[i]["sector_breadth"] = positive / len(peers)

        # Sector average momentum (excluding self)
        peer_moms = [features_list[j]["momentum"] for j in peers if j != i]
        if peer_moms:
            sector_avg_mom = np.mean(peer_moms)
            features_list[i]["mom_vs_sector"] = features_list[i]["momentum"] - sector_avg_mom

    return features_list

def feat_to_arr(f):
    return np.array([f.get(n, 0) for n in FEATURE_NAMES])

# ═══════════════════════════════════════════════════════════════════
# TRAINING — FIRST-PASSAGE LABELS
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BARS_DAILY_CACHE = CACHE_DIR / "bars_daily.pkl"
BARS_INTRADAY_CACHE = CACHE_DIR / "bars_intraday.pkl"
TRAINING_ROWS_CACHE = CACHE_DIR / "training_rows.pkl"  # persisted for pattern search
PATTERNS_PATH = DATA_DIR / "patterns.json"  # discovered patterns
CACHE_MAX_AGE_HOURS = 24

def cache_age_hours(path):
    if not path.exists(): return 999
    age_sec = time.time() - path.stat().st_mtime
    return age_sec / 3600

def run_extend_history(months_back=12):
    """
    v17: Extend bar cache backward in time without retraining.
    v18: Add explicit SPY/IWM diagnostics — prior runs silently lost ETF bars.

    Finds the earliest date in the current cache, fetches `months_back` months
    of bars before that, merges them in. Does NOT overwrite existing bars.
    After running, the cache fingerprint changes and setup evaluation should
    be re-run to incorporate the extended data.

    Rate-limit friendly: uses the same fetch_bars which has built-in backoff.
    """
    global extend_history_in_progress, extend_history_progress
    if extend_history_in_progress:
        log.warning("extend_history already running")
        return
    extend_history_in_progress = True
    extend_history_progress = {"phase":"starting","pct":1,"message":"Preparing extended fetch..."}

    try:
        # Validate existing cache
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No existing cache. Run Training first to build initial cache.")

        extend_history_progress = {"phase":"loading","pct":3,"message":"Loading existing cache..."}
        existing_daily = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        existing_intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        # Find earliest date in existing intraday cache
        earliest_date = None
        for sym_bars in existing_intraday.values():
            for b in sym_bars:
                d = b.get("t", "")[:10]
                if d and (earliest_date is None or d < earliest_date):
                    earliest_date = d
        if not earliest_date:
            raise RuntimeError("Could not determine earliest date in existing cache")

        # Compute new fetch window: from (earliest - months_back) to earliest - 1 day
        earliest_dt = datetime.strptime(earliest_date, "%Y-%m-%d")
        new_end_dt = earliest_dt - timedelta(days=1)
        new_start_dt = earliest_dt - timedelta(days=int(months_back * 30.5))  # approximate month length
        new_start = new_start_dt.strftime("%Y-%m-%d")
        new_end = new_end_dt.strftime("%Y-%m-%d")

        log.info(f"Extending history: fetching {new_start} → {new_end} "
                 f"(current earliest: {earliest_date}, extending {months_back} months back)")
        extend_history_progress = {"phase":"fetching","pct":8,
            "message":f"Fetching {new_start} → {new_end} ({months_back}mo before {earliest_date})..."}

        fetch_symbols = list(set(TICKERS + ["SPY", "IWM"]))

        # Fetch daily first (fast)
        extend_history_progress = {"phase":"fetch_daily","pct":12,
            "message":f"Fetching daily bars for {months_back}mo extension..."}
        client = alpaca_client()
        try:
            new_daily = fetch_bars(client, fetch_symbols, "1Day", new_start, new_end)
        except Exception as e:
            client.close()
            raise RuntimeError(f"Daily fetch failed: {e}")

        extend_history_progress = {"phase":"fetch_intraday","pct":20,
            "message":f"Fetching intraday 5-min bars ({months_back}mo) — this takes time..."}
        try:
            new_intraday = fetch_bars(client, fetch_symbols, "5Min",
                                      f"{new_start}T09:30:00-04:00",
                                      f"{new_end}T16:00:00-04:00")
        finally:
            client.close()

        n_new_daily = sum(len(v) for v in new_daily.values())
        n_new_intra = sum(len(v) for v in new_intraday.values())
        log.info(f"Extended fetch complete: {n_new_daily} daily bars, {n_new_intra} intraday bars")

        # v18: Explicit ETF diagnostics. If SPY/IWM didn't come back, log loudly.
        spy_new_daily = len(new_daily.get("SPY", []))
        spy_new_intra = len(new_intraday.get("SPY", []))
        iwm_new_daily = len(new_daily.get("IWM", []))
        iwm_new_intra = len(new_intraday.get("IWM", []))
        log.info(f"v18 ETF fetch diagnostics: "
                 f"SPY daily={spy_new_daily}, SPY intraday={spy_new_intra}, "
                 f"IWM daily={iwm_new_daily}, IWM intraday={iwm_new_intra}")
        if spy_new_intra == 0:
            log.warning("v18: SPY returned 0 intraday bars. Either Alpaca failed the request or SPY was excluded.")
        if iwm_new_intra == 0:
            log.warning("v18: IWM returned 0 intraday bars.")
        # Retry SPY/IWM individually if either missing — this isolates the failure
        if spy_new_intra == 0 or iwm_new_intra == 0:
            log.info("v18: Retrying SPY/IWM individually to isolate failure...")
            extend_history_progress = {"phase":"etf_retry","pct":70,"message":"Retrying SPY/IWM individually..."}
            etf_client = alpaca_client()
            try:
                for etf in ["SPY", "IWM"]:
                    if len(new_intraday.get(etf, [])) > 0:
                        continue
                    log.info(f"v18: Individual fetch for {etf}")
                    try:
                        retry_daily = fetch_bars(etf_client, [etf], "1Day", new_start, new_end)
                        retry_intra = fetch_bars(etf_client, [etf], "5Min",
                                                 f"{new_start}T09:30:00-04:00",
                                                 f"{new_end}T16:00:00-04:00")
                        rd = len(retry_daily.get(etf, []))
                        ri = len(retry_intra.get(etf, []))
                        log.info(f"v18: {etf} individual retry — daily={rd}, intraday={ri}")
                        if rd > 0: new_daily[etf] = retry_daily[etf]
                        if ri > 0: new_intraday[etf] = retry_intra[etf]
                    except Exception as e:
                        log.error(f"v18: {etf} retry failed: {e}")
            finally:
                etf_client.close()
        extend_history_progress = {"phase":"merging","pct":75,
            "message":f"Merging {n_new_intra:,} new intraday bars into cache..."}

        # Merge: append new bars at the front of each ticker's list, dedup by timestamp
        def merge_symbol_bars(old_list, new_list):
            seen_ts = {b["t"] for b in old_list}
            # Add only bars not already present
            added = [b for b in new_list if b["t"] not in seen_ts]
            # Combine and sort by timestamp
            combined = added + old_list
            combined.sort(key=lambda b: b["t"])
            return combined

        merged_daily = {}
        merged_intraday = {}
        all_syms = set(existing_daily.keys()) | set(new_daily.keys())
        for sym in all_syms:
            merged_daily[sym] = merge_symbol_bars(
                existing_daily.get(sym, []), new_daily.get(sym, []))
        all_syms_i = set(existing_intraday.keys()) | set(new_intraday.keys())
        for sym in all_syms_i:
            merged_intraday[sym] = merge_symbol_bars(
                existing_intraday.get(sym, []), new_intraday.get(sym, []))

        extend_history_progress = {"phase":"writing","pct":92,"message":"Writing extended cache..."}
        BARS_DAILY_CACHE.write_bytes(pickle.dumps(merged_daily))
        BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(merged_intraday))

        total_intra = sum(len(v) for v in merged_intraday.values())
        total_daily = sum(len(v) for v in merged_daily.values())
        log.info(f"Cache extended: {total_daily} daily bars, {total_intra:,} intraday bars total")
        extend_history_progress = {"phase":"done","pct":100,
            "message":f"Extended history added. Cache now has {total_intra:,} intraday bars. Re-run Setup Evaluation to use."}

    except Exception as e:
        log.error(f"Extend history failed: {e}", exc_info=True)
        extend_history_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        extend_history_in_progress = False


# v18: repair ETF (SPY + IWM) coverage in existing cache.
# Prior versions lost SPY bars entirely; IWM remained flat across extend_history runs.
# This function fetches SPY + IWM for the FULL date range present in cache and merges.
# Does not affect stock bars.
repair_etf_in_progress = False
repair_etf_progress = {"phase":"idle","pct":0,"message":""}

def run_repair_etf():
    """v18: Fetch SPY + IWM across full cache date range and merge into cache."""
    global repair_etf_in_progress, repair_etf_progress
    if repair_etf_in_progress:
        log.warning("repair_etf already running")
        return
    repair_etf_in_progress = True
    repair_etf_progress = {"phase":"starting","pct":1,"message":"Starting ETF repair..."}
    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No existing cache.")

        existing_daily = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        existing_intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        # Find full date range in existing cache
        earliest_date = None
        latest_date = None
        for sym_bars in existing_intraday.values():
            for b in sym_bars:
                d = b.get("t", "")[:10]
                if not d: continue
                if earliest_date is None or d < earliest_date: earliest_date = d
                if latest_date is None or d > latest_date: latest_date = d
        if not earliest_date or not latest_date:
            raise RuntimeError("Could not determine date range in cache")

        log.info(f"ETF repair: fetching SPY + IWM from {earliest_date} to {latest_date}")
        repair_etf_progress = {"phase":"fetching","pct":10,
            "message":f"Fetching SPY + IWM from {earliest_date} to {latest_date}..."}

        client = alpaca_client()
        try:
            # Fetch SPY and IWM one at a time to isolate any per-symbol failure
            fresh = {}
            fresh_daily = {}
            for etf in ["SPY", "IWM"]:
                repair_etf_progress = {"phase":f"fetching_{etf}","pct":20 if etf=="SPY" else 55,
                    "message":f"Fetching {etf}..."}
                try:
                    etf_daily = fetch_bars(client, [etf], "1Day", earliest_date, latest_date)
                    etf_intra = fetch_bars(client, [etf], "5Min",
                                           f"{earliest_date}T09:30:00-04:00",
                                           f"{latest_date}T16:00:00-04:00")
                    d_count = len(etf_daily.get(etf, []))
                    i_count = len(etf_intra.get(etf, []))
                    log.info(f"v18 repair_etf: {etf} fetched {d_count} daily bars, {i_count} intraday bars")
                    if d_count > 0: fresh_daily[etf] = etf_daily[etf]
                    if i_count > 0: fresh[etf] = etf_intra[etf]
                except Exception as e:
                    log.error(f"v18 repair_etf: {etf} fetch failed: {e}")
                    raise
        finally:
            client.close()

        repair_etf_progress = {"phase":"merging","pct":85,"message":"Merging into cache..."}
        # Overwrite the SPY/IWM entries fully (since we fetched the full date range)
        for etf, bars in fresh_daily.items():
            existing_daily[etf] = sorted(bars, key=lambda b: b["t"])
        for etf, bars in fresh.items():
            existing_intraday[etf] = sorted(bars, key=lambda b: b["t"])

        BARS_DAILY_CACHE.write_bytes(pickle.dumps(existing_daily))
        BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(existing_intraday))

        n_spy = len(existing_intraday.get("SPY", []))
        n_iwm = len(existing_intraday.get("IWM", []))
        log.info(f"v18 repair_etf complete: SPY now {n_spy} bars, IWM now {n_iwm} bars")
        repair_etf_progress = {"phase":"done","pct":100,
            "message":f"ETF repair complete: SPY {n_spy} bars, IWM {n_iwm} bars. Re-run Setup Evaluation to use."}

    except Exception as e:
        log.error(f"v18 repair_etf failed: {e}", exc_info=True)
        repair_etf_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        repair_etf_in_progress = False


def run_training(tp_mult=None, sl_mult=None, tp_pct=None, sl_pct=None):
    """
    Train models with ATR-scaled barriers:
        TP = entry * (1 + tp_mult * atr_pct)
        SL = entry * (1 - sl_mult * atr_pct)
    where atr_pct is that stock's 14-day ATR as a fraction of its most recent close.

    tp_pct/sl_pct args are kept for API back-compat and are interpreted as multipliers
    when passed alongside tp_mult==None (legacy frontend behavior).
    """
    global models, calibrators, model_meta, training_in_progress, training_progress, status
    if training_in_progress: return
    training_in_progress = True
    training_progress = {"phase":"starting","pct":0,"message":"Starting..."}

    # Resolve multipliers. Priority: explicit tp_mult/sl_mult > legacy tp_pct/sl_pct as mults > globals.
    use_tp_mult = tp_mult if tp_mult is not None else (tp_pct if tp_pct is not None else TP_MULT)
    use_sl_mult = sl_mult if sl_mult is not None else (sl_pct if sl_pct is not None else SL_MULT)

    try:
        # ─── Load or fetch bar data ──────────────────────────────────
        daily_age = cache_age_hours(BARS_DAILY_CACHE)
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        cache_fresh = daily_age < CACHE_MAX_AGE_HOURS and intra_age < CACHE_MAX_AGE_HOURS

        # Fetch list includes SPY for market-relative features, IWM for rel_strength_iwm setup
        fetch_symbols = list(set(TICKERS + ["SPY", "IWM"]))

        # v11: if the cache is fresh but doesn't contain an expected symbol like IWM,
        # the cache was built by an older version and must be rebuilt.
        if cache_fresh:
            try:
                cached_intra = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
                required_syms = {"SPY", "IWM"}
                missing = [s for s in required_syms if s not in cached_intra or not cached_intra[s]]
                if missing:
                    log.info(f"Cache is fresh but missing required symbols {missing} — forcing refetch")
                    cache_fresh = False
                else:
                    # Cache is fresh AND has all required symbols
                    training_progress = {"phase":"loading_cache","pct":5,
                        "message":f"Loading cached bars (age {intra_age:.1f}h, {len(cached_intra)} symbols)..."}
                    log.info(f"Using cached bars (daily age {daily_age:.1f}h, intraday age {intra_age:.1f}h)")
                    daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
                    intraday = cached_intra
            except Exception as e:
                log.warning(f"Cache read failed, refetching: {e}")
                cache_fresh = False

        if not cache_fresh:
            client = alpaca_client()
            end_date = today_et()
            start_obj = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=380)
            start_date = start_obj.strftime("%Y-%m-%d")

            training_progress = {"phase":"fetch_daily","pct":3,"message":"Fetching daily bars (incl SPY + IWM)..."}
            daily_bars = fetch_bars(client, fetch_symbols, "1Day", start_date, end_date)

            training_progress = {"phase":"fetch_intraday","pct":8,"message":"Fetching 12 months of 5-min bars (incl SPY + IWM)..."}
            intraday = fetch_bars(client, fetch_symbols, "5Min",
                                  f"{start_date}T09:30:00-04:00", f"{end_date}T16:00:00-04:00")
            client.close()

            training_progress = {"phase":"caching","pct":44,"message":"Caching bars to disk..."}
            BARS_DAILY_CACHE.write_bytes(pickle.dumps(daily_bars))
            BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(intraday))
            log.info(f"Cached {sum(len(v) for v in intraday.values())} intraday bars to disk")

        # ─── Group bars by ticker+date ───────────────────────────────
        training_progress = {"phase":"grouping","pct":45,"message":"Grouping bars by date..."}
        by_td = defaultdict(lambda: defaultdict(list))
        for ticker in fetch_symbols:
            for b in intraday.get(ticker, []):
                by_td[ticker][b["t"][:10]].append(b)

        all_dates = sorted(set(d for t in by_td for d in by_td[t]))
        log.info(f"Training: {len(all_dates)} dates, TP={use_tp_mult}×ATR / SL={use_sl_mult}×ATR (volatility-adjusted)")

        # ─── Build training dataset with the specified ATR multipliers ───
        training_progress = {"phase":"features","pct":50,
            "message":f"Computing features + SPY/sector context (TP {use_tp_mult}×ATR / SL {use_sl_mult}×ATR)..."}
        rows_per_hour = defaultdict(list)

        for di, date in enumerate(all_dates):
            for scan_hour in SCAN_HOURS:
                scan_min = scan_hour * 60

                # ── Compute SPY context for this date+hour ──
                spy_day = by_td["SPY"].get(date, [])
                spy_before = [b for b in spy_day if (bar_to_et_minutes(b) or 0) < scan_min]
                spy_ctx = compute_spy_context(spy_before)

                date_features, date_meta, date_sectors = [], [], []

                for ticker in TICKERS:
                    day_bars = by_td[ticker].get(date, [])
                    if len(day_bars) < 12: continue

                    before, after = [], []
                    for b in day_bars:
                        bm = bar_to_et_minutes(b)
                        if bm is None: continue
                        if bm < scan_min: before.append(b)
                        else: after.append(b)

                    if len(before) < 15 or len(after) < 2: continue

                    entry_price = after[0]["o"]
                    feature_price = before[-1]["c"]
                    open_price = day_bars[0]["o"]
                    # Extended daily lookback to support 14-day ATR (need 15+ prior daily bars)
                    daily_up_to = [d for d in daily_bars.get(ticker,[]) if d["t"][:10] < date][-20:]

                    # Get previous close for gap calculation
                    prev_close = daily_up_to[-1]["c"] if daily_up_to else None

                    feat = compute_features(before, daily_up_to, feature_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue

                    # Volatility-adjusted barriers: per-stock per-date TP/SL
                    atr_pct = compute_atr_pct(daily_up_to, lookback=ATR_LOOKBACK_DAYS)
                    if atr_pct is None: continue  # need enough daily history
                    stock_tp = use_tp_mult * atr_pct
                    stock_sl = use_sl_mult * atr_pct

                    outcome, pnl, reason = compute_trade_outcome(entry_price, after[1:], tp_pct=stock_tp, sl_pct=stock_sl)

                    date_features.append(feat)
                    date_meta.append({"ticker":ticker,"label":outcome,"pnl":pnl,"reason":reason,"date":date})
                    date_sectors.append(SECTORS.get(ticker,"?"))

                if len(date_features) >= 10:
                    add_ranks(date_features)
                    add_sector_relative(date_features, date_sectors)
                    for j in range(len(date_features)):
                        date_features[j]["label"] = date_meta[j]["label"]
                        date_features[j]["date"] = date_meta[j]["date"]
                        date_features[j]["pnl"] = date_meta[j]["pnl"]
                        date_features[j]["reason"] = date_meta[j]["reason"]
                        rows_per_hour[scan_hour].append(date_features[j])

            if (di+1) % 10 == 0:
                training_progress = {"phase":"features","pct":50+int((di/len(all_dates))*35),
                    "message":f"Processed {di+1}/{len(all_dates)} days..."}

        # Persist enriched training rows for pattern search (after features + labels computed)
        try:
            TRAINING_ROWS_CACHE.write_bytes(pickle.dumps(dict(rows_per_hour)))
            total_rows = sum(len(v) for v in rows_per_hour.values())
            log.info(f"Saved {total_rows} enriched training rows to {TRAINING_ROWS_CACHE}")
        except Exception as e:
            log.warning(f"Could not persist training rows: {e}")

        training_progress = {"phase":"training","pct":87,"message":"Training LightGBM models..."}
        new_models, new_cals, new_meta = {}, {}, {}

        for h in SCAN_HOURS:
            rows = rows_per_hour[h]
            if len(rows) < 200:
                log.warning(f"{h}:00 only {len(rows)} samples, skip"); continue

            df = pd.DataFrame(rows)
            dates = sorted(df["date"].unique())
            # Three-way split: 60% train / 20% val / 20% test (temporal)
            # Train+val fit the model (val used for LightGBM early stopping).
            # Val is also used to choose the threshold.
            # Test is HELD OUT — never seen by model training or threshold selection.
            split_tr = int(len(dates) * 0.6)
            split_va = int(len(dates) * 0.8)
            train_dates = set(dates[:split_tr])
            val_dates = set(dates[split_tr:split_va])
            test_dates = set(dates[split_va:])

            train_df = df[df["date"].isin(train_dates)]
            val_df = df[df["date"].isin(val_dates)]
            test_df = df[df["date"].isin(test_dates)]

            X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["label"].values
            X_va, y_va = val_df[FEATURE_NAMES].values, val_df["label"].values
            X_te, y_te = test_df[FEATURE_NAMES].values, test_df["label"].values

            win_rate_train = y_tr.mean()
            win_rate_val = y_va.mean()
            win_rate_test = y_te.mean() if len(y_te) > 0 else 0

            log.info(f"{h}:00 — train {len(train_df)} (WR {win_rate_train:.3f}), val {len(val_df)} (WR {win_rate_val:.3f}), test {len(test_df)} (WR {win_rate_test:.3f})")

            ts = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES)
            vs = lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES, reference=ts)

            params = {
                "objective":"binary","metric":"binary_logloss",
                "boosting_type":"gbdt","num_leaves":31,"learning_rate":0.05,
                "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,
                "min_child_samples":20,"verbose":-1
            }
            model = lgb.train(params, ts, num_boost_round=500, valid_sets=[vs],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            val_probs = model.predict(X_va)
            test_probs = model.predict(X_te) if len(X_te) > 0 else np.array([])
            auc = roc_auc_score(y_va, val_probs) if len(set(y_va)) > 1 else 0
            test_auc = roc_auc_score(y_te, test_probs) if len(y_te) > 0 and len(set(y_te)) > 1 else 0

            # ─── Top-N precision and P&L per val day, for N in {1,3,5,10} ──
            val_df = val_df.copy()
            val_df["prob"] = val_probs
            test_df = test_df.copy()
            if len(test_probs) > 0:
                test_df["prob"] = test_probs

            topn_stats = {}
            for N in [1, 3, 5, 10]:
                wr_list, pnl_list = [], []
                for d in val_dates:
                    day = val_df[val_df["date"]==d].nlargest(N,"prob")
                    if len(day) >= N:
                        wr_list.append(day["label"].mean())
                        pnl_list.append(day["pnl"].mean())
                if wr_list:
                    topn_stats[N] = {
                        "avg_wr": round(float(np.mean(wr_list)), 4),
                        "avg_pnl": round(float(np.mean(pnl_list)), 3),
                        "std_pnl": round(float(np.std(pnl_list)), 3),
                        "min_pnl": round(float(np.min(pnl_list)), 3),
                        "max_pnl": round(float(np.max(pnl_list)), 3),
                        "n_days": int(len(pnl_list)),
                        "pnl_positive_days": int(sum(1 for p in pnl_list if p > 0)),
                    }
                else:
                    topn_stats[N] = {"avg_wr":0,"avg_pnl":0,"std_pnl":0,"min_pnl":0,"max_pnl":0,"n_days":0,"pnl_positive_days":0}

            # Legacy aliases (kept for diagnostic/UI backwards compat)
            avg_p10 = topn_stats[10]["avg_wr"]
            avg_pnl10 = topn_stats[10]["avg_pnl"]

            # EV at various thresholds
            # Notional break-even based on multipliers (TP_MULT × P = SL_MULT × (1-P)).
            # This is a universe-average approximation — true per-stock BE depends on ATR.
            # The realized_breakeven_wr in loss_distribution is the honest measure.
            breakeven_p = use_sl_mult / (use_sl_mult + use_tp_mult)

            # EV at various probability thresholds
            def ev_at(thresh):
                subset = val_df[val_df["prob"] >= thresh]
                if len(subset) == 0: return 0, 0
                return subset["pnl"].mean(), len(subset)
            ev_at_be, n_at_be = ev_at(breakeven_p)
            ev_at_be5, n_at_be5 = ev_at(breakeven_p + 0.05)
            ev_at_50 = val_df[val_df["prob"]>=0.5]["pnl"].mean() if len(val_df[val_df["prob"]>=0.5])>0 else 0
            n_above_50 = len(val_df[val_df["prob"]>=0.5])
            ev_at_55 = val_df[val_df["prob"]>=0.55]["pnl"].mean() if len(val_df[val_df["prob"]>=0.55])>0 else 0
            n_above_55 = len(val_df[val_df["prob"]>=0.55])

            # Exit reason breakdown (val)
            val_reasons = val_df["reason"].value_counts().to_dict() if "reason" in val_df.columns else {}

            # Loss distribution stats — critical for wide-stop strategies.
            # With SL set wide (e.g. 5%), most "losers" will close at 15:55 with
            # small/medium losses, not full -SL%. Realized break-even depends on
            # actual loser P&L distribution, not nominal SL.
            losers = val_df[val_df["label"] == 0]
            winners = val_df[val_df["label"] == 1]
            loss_dist = {}
            if len(losers) > 0:
                loss_pnls = losers["pnl"].values
                loss_dist = {
                    "n_losers": int(len(losers)),
                    "mean_loss_pnl": round(float(np.mean(loss_pnls)), 3),
                    "median_loss_pnl": round(float(np.median(loss_pnls)), 3),
                    "p10_loss_pnl": round(float(np.percentile(loss_pnls, 10)), 3),
                    "p90_loss_pnl": round(float(np.percentile(loss_pnls, 90)), 3),
                    "std_loss_pnl": round(float(np.std(loss_pnls)), 3),
                }
                # Per-exit-reason loss P&L
                for reason in ["sl", "close_15:55", "eod"]:
                    sub = losers[losers["reason"] == reason]
                    if len(sub) > 0:
                        loss_dist[f"mean_loss_{reason}"] = round(float(sub["pnl"].mean()), 3)
                        loss_dist[f"n_loss_{reason}"] = int(len(sub))
            # Winner P&L mean (should be close to +TP% for first-passage)
            if len(winners) > 0:
                loss_dist["mean_win_pnl"] = round(float(winners["pnl"].mean()), 3)
                loss_dist["n_winners"] = int(len(winners))
            # Realized break-even: what win rate makes E[PnL] = 0 given actual distributions?
            # E[PnL] = p*mean_win + (1-p)*mean_loss = 0 → p = -mean_loss / (mean_win - mean_loss)
            if loss_dist.get("mean_win_pnl") and loss_dist.get("mean_loss_pnl"):
                mw, ml = loss_dist["mean_win_pnl"], loss_dist["mean_loss_pnl"]
                if mw > ml:
                    realized_be = -ml / (mw - ml)
                    loss_dist["realized_breakeven_wr"] = round(float(realized_be) * 100, 2)

            # Isotonic calibration
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.95)
            cal.fit(val_probs, y_va)

            # Feature importance
            imp = dict(zip(FEATURE_NAMES, model.feature_importance("gain").tolist()))
            ti = sum(imp.values()) or 1
            imp = {k: round(v/ti, 4) for k,v in imp.items()}

            model.save_model(str(MODEL_DIR / f"model_{h}.txt"))
            (MODEL_DIR / f"calibrator_{h}.pkl").write_bytes(pickle.dumps(cal))

            meta = {
                "scan_hour":h,
                "train_samples":len(train_df),"val_samples":len(val_df),
                "train_dates":len(train_dates),"val_dates":len(val_dates),
                "train_win_rate":round(float(win_rate_train),4),
                "val_win_rate":round(float(win_rate_val),4),
                "auc":round(auc,4),
                "avg_win_rate_top10":round(float(avg_p10),4),
                "avg_pnl_top10":round(float(avg_pnl10),3),
                "topN": topn_stats,  # top-1, top-3, top-5, top-10 metrics per val day
                "ev_above_50pct":round(float(ev_at_50),3),
                "n_above_50pct":int(n_above_50),
                "ev_above_55pct":round(float(ev_at_55),3),
                "n_above_55pct":int(n_above_55),
                "breakeven_threshold":round(breakeven_p,3),
                "ev_above_breakeven":round(float(ev_at_be),3),
                "n_above_breakeven":int(n_at_be),
                "ev_above_breakeven_plus5":round(float(ev_at_be5),3),
                "n_above_breakeven_plus5":int(n_at_be5),
                "val_exit_reasons":val_reasons,
                "loss_distribution":loss_dist,
                "importance":imp,
                "trained_at":datetime.now(ET).isoformat(),
                "best_iteration":model.best_iteration,
                "tp_mult": use_tp_mult, "sl_mult": use_sl_mult,
                "atr_lookback_days": ATR_LOOKBACK_DAYS,
                # Legacy fields for back-compat (nominal values, meaningless in v8 but
                # kept so existing UI bits that read .tp_pct don't crash):
                "tp_pct": None, "sl_pct": None
            }
            (MODEL_DIR / f"meta_{h}.json").write_text(json.dumps(meta, indent=2))

            new_models[h] = model
            new_cals[h] = cal
            new_meta[h] = meta
            t1 = topn_stats[1]; t3 = topn_stats[3]; t5 = topn_stats[5]; t10 = topn_stats[10]
            log.info(f"{h}:00 — AUC {auc:.3f} | Top1 WR {t1['avg_wr']:.3f} PnL {t1['avg_pnl']:+.3f}% | "
                     f"Top3 WR {t3['avg_wr']:.3f} PnL {t3['avg_pnl']:+.3f}% | "
                     f"Top5 WR {t5['avg_wr']:.3f} PnL {t5['avg_pnl']:+.3f}% | "
                     f"Top10 WR {t10['avg_wr']:.3f} PnL {t10['avg_pnl']:+.3f}% | base {win_rate_val:.3f}")

        models.update(new_models)
        calibrators.update(new_cals)
        model_meta.update(new_meta)
        status["trained"] = True
        status["trainDate"] = datetime.now(ET).isoformat()
        status["daysSinceRetrain"] = 0
        status["activeTpMult"] = use_tp_mult
        status["activeSlMult"] = use_sl_mult
        status["atrLookback"] = ATR_LOOKBACK_DAYS
        save_status(status)

        training_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(new_models)} models trained (TP {use_tp_mult}×ATR / SL {use_sl_mult}×ATR)."}
        log.info(f"Training complete. Active barriers: TP {use_tp_mult}×ATR / SL {use_sl_mult}×ATR")

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        training_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        training_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# SWEEP: grid search over TP/SL combinations
# ═══════════════════════════════════════════════════════════════════
# Coarse grid: 3 TP × 5 SL = 15 combinations
# v8: Sweep values are now ATR multipliers (volatility-adjusted barriers)
# TP_MULT of 0.5 means TP at +0.5×ATR; SL_MULT of 2.5 means SL at -2.5×ATR.
# Realized barrier widths depend on each stock's individual ATR.
SWEEP_TP_VALUES = [0.25, 0.50, 1.00]
SWEEP_SL_VALUES = [2.5]  # wide relative to TP — catastrophe protection

def summarize_models_for_sweep(tp, sl):
    """After a training completes, extract sweep-relevant summary from model_meta.
    v8: `tp` and `sl` are ATR multipliers, not fixed percentages. Stored keys
    retain names tp_pct/sl_pct for backward compat with existing UI.
    """
    # Nominal breakeven is not meaningful with ATR-adjusted barriers (varies per stock).
    # We keep a notional value based on multipliers, but the real measure is realized BE from the loss dist.
    notional_be = round(sl / (sl + tp) * 100, 2) if (sl + tp) > 0 else None
    summary = {
        "tp_pct": tp, "sl_pct": sl,  # ATR multipliers, not percent
        "tp_mult": tp, "sl_mult": sl,
        "breakeven": notional_be,  # notional only — true measure is realized_be
        "hours": {},
        "avg_top10_wr": None, "avg_top10_pnl": None, "avg_auc": None,
        "avg_base_wr": None, "avg_edge": None,
        "avg_realized_be": None,
        # Top-N averages across scan hours
        "avg_top1_wr": None, "avg_top1_pnl": None,
        "avg_top3_wr": None, "avg_top3_pnl": None,
        "avg_top5_wr": None, "avg_top5_pnl": None,
        "completedAt": datetime.now(ET).isoformat()
    }
    top10_wrs, top10_pnls, aucs, base_wrs, realized_bes = [], [], [], [], []
    topN_agg = {1: {"wr":[], "pnl":[]}, 3: {"wr":[], "pnl":[]}, 5: {"wr":[], "pnl":[]}}

    for h in SCAN_HOURS:
        m = model_meta.get(h)
        if not m: continue
        wr10 = m.get("avg_win_rate_top10", 0) * 100
        pnl10 = m.get("avg_pnl_top10", 0)
        auc = m.get("auc", 0)
        base = m.get("val_win_rate", 0) * 100
        edge = wr10 - summary["breakeven"]
        ldist = m.get("loss_distribution", {})
        realized_be = ldist.get("realized_breakeven_wr")
        topN = m.get("topN", {})

        hour_summary = {
            "top10_wr": round(wr10, 2),
            "top10_pnl": round(pnl10, 3),
            "auc": round(auc, 4),
            "base_wr": round(base, 2),
            "edge": round(edge, 2),
            "realized_be": realized_be,
            "mean_loss_pnl": ldist.get("mean_loss_pnl"),
            "mean_win_pnl": ldist.get("mean_win_pnl"),
        }
        # Per-hour top-N details (wr, pnl, std, n_days)
        for N in [1, 3, 5, 10]:
            tn = topN.get(str(N), topN.get(N, {}))
            hour_summary[f"top{N}_wr"] = round(tn.get("avg_wr", 0) * 100, 2)
            hour_summary[f"top{N}_pnl"] = tn.get("avg_pnl", 0)
            hour_summary[f"top{N}_std_pnl"] = tn.get("std_pnl", 0)
            hour_summary[f"top{N}_pos_days"] = tn.get("pnl_positive_days", 0)
            hour_summary[f"top{N}_n_days"] = tn.get("n_days", 0)
            if N in topN_agg:
                topN_agg[N]["wr"].append(tn.get("avg_wr", 0) * 100)
                topN_agg[N]["pnl"].append(tn.get("avg_pnl", 0))

        # Edge per top-N vs realized BE (if available) or nominal BE
        comp_be = realized_be if realized_be is not None else summary["breakeven"]
        for N in [1, 3, 5, 10]:
            hour_summary[f"top{N}_edge"] = round(hour_summary[f"top{N}_wr"] - comp_be, 2)

        summary["hours"][str(h)] = hour_summary

        top10_wrs.append(wr10); top10_pnls.append(pnl10)
        aucs.append(auc); base_wrs.append(base)
        if realized_be: realized_bes.append(realized_be)

    if top10_wrs:
        summary["avg_top10_wr"] = round(float(np.mean(top10_wrs)), 2)
        summary["avg_top10_pnl"] = round(float(np.mean(top10_pnls)), 3)
        summary["avg_auc"] = round(float(np.mean(aucs)), 4)
        summary["avg_base_wr"] = round(float(np.mean(base_wrs)), 2)
        summary["avg_edge"] = round(summary["avg_top10_wr"] - summary["breakeven"], 2)
        if realized_bes:
            summary["avg_realized_be"] = round(float(np.mean(realized_bes)), 2)
            summary["realized_edge"] = round(summary["avg_top10_wr"] - summary["avg_realized_be"], 2)

        # Top-N averages across hours
        for N in [1, 3, 5]:
            if topN_agg[N]["wr"]:
                summary[f"avg_top{N}_wr"] = round(float(np.mean(topN_agg[N]["wr"])), 2)
                summary[f"avg_top{N}_pnl"] = round(float(np.mean(topN_agg[N]["pnl"])), 3)
                if realized_bes:
                    summary[f"avg_top{N}_edge"] = round(summary[f"avg_top{N}_wr"] - summary["avg_realized_be"], 2)

    return summary

def run_sweep(resume=True):
    """
    Grid search over TP × SL combinations. Trains a full model suite for each
    combination, records summary metrics, and saves progress to disk after
    each cell so it can resume after crashes.
    """
    global sweep_in_progress, sweep_progress, training_in_progress
    if sweep_in_progress: return
    if training_in_progress: return
    sweep_in_progress = True

    # Build full grid
    grid_cells = [(tp, sl) for tp in SWEEP_TP_VALUES for sl in SWEEP_SL_VALUES]
    total = len(grid_cells)

    # Load existing results for resume
    existing = load_sweep_results() if resume else {"grid":[],"startedAt":None,"completedAt":None}
    completed_keys = {f"{r['tp_pct']}_{r['sl_pct']}" for r in existing.get("grid",[])}
    if not existing.get("startedAt") or not resume:
        existing = {"grid":[], "startedAt":datetime.now(ET).isoformat(), "completedAt":None,
                    "gridShape":{"tpValues":SWEEP_TP_VALUES, "slValues":SWEEP_SL_VALUES}}
        completed_keys = set()

    log.info(f"Sweep: {total} cells, {len(completed_keys)} already complete, {total-len(completed_keys)} to run")

    try:
        for idx, (tp, sl) in enumerate(grid_cells):
            key = f"{tp}_{sl}"
            if key in completed_keys:
                log.info(f"Sweep cell {idx+1}/{total}: TP {tp}% / SL {sl}% — skipping (cached)")
                continue

            sweep_progress = {
                "phase":"running","current":idx+1,"total":total,
                "currentTP":tp,"currentSL":sl,
                "message":f"Cell {idx+1}/{total}: TP {tp}% / SL {sl}% (break-even {sl/(sl+tp)*100:.1f}%)"
            }

            # v8: sweep values are ATR multipliers, not fixed percentages
            log.info(f"Sweep cell {idx+1}/{total}: starting TP_MULT={tp} SL_MULT={sl}")
            run_training(tp_mult=tp, sl_mult=sl)

            # After training, extract summary from model_meta
            cell_summary = summarize_models_for_sweep(tp, sl)
            existing["grid"].append(cell_summary)
            save_sweep_results(existing)
            log.info(f"Sweep cell {idx+1}/{total} done: avg top10 WR {cell_summary['avg_top10_wr']}%, "
                     f"breakeven {cell_summary['breakeven']}%, edge {cell_summary['avg_edge']}%")

        existing["completedAt"] = datetime.now(ET).isoformat()
        save_sweep_results(existing)
        sweep_progress = {"phase":"done","current":total,"total":total,
            "message":f"Sweep complete. {total} cells evaluated.",
            "currentTP":None,"currentSL":None}
        log.info("Sweep complete.")

    except Exception as e:
        log.error(f"Sweep failed: {e}", exc_info=True)
        sweep_progress = {"phase":"error","current":0,"total":total,"message":str(e),
                         "currentTP":None,"currentSL":None}
    finally:
        sweep_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# PATTERN SEARCH — find narrow feature regions with high win rate
# ═══════════════════════════════════════════════════════════════════
pattern_search_in_progress = False
pattern_search_progress = {"phase":"idle","pct":0,"message":""}

# Features we won't threshold on (derived/contextual, not predictive on their own)
PATTERN_SEARCH_EXCLUDE = {"hours_left", "spy_ret", "spy_momentum", "spy_vol"}

def evaluate_region(df, mask, tp, sl):
    """
    Given a boolean mask over a dataframe, compute pattern statistics.
    Returns dict with: n, win_rate, avg_pnl, breakeven, edge.
    """
    subset = df[mask]
    n = len(subset)
    if n == 0:
        return {"n":0, "win_rate":0, "avg_pnl":0, "edge":0, "breakeven": sl/(sl+tp)*100}
    wr = subset["label"].mean() * 100
    pnl = subset["pnl"].mean()
    be = sl / (sl + tp) * 100
    return {"n":n, "win_rate":round(wr,2), "avg_pnl":round(pnl,3),
            "edge":round(wr-be,2), "breakeven":round(be,2)}

def search_single_feature_patterns(df_train, df_val, tp, sl,
                                     min_samples_train=100, min_samples_val=30,
                                     min_edge_train=5.0, min_edge_val=3.0):
    """
    For each numeric feature, try a set of percentile thresholds.
    Keep patterns where training win rate > breakeven + min_edge_train
    AND validation win rate > breakeven + min_edge_val
    AND sample counts are sufficient in both folds.
    """
    candidates = []
    features_to_try = [f for f in FEATURE_NAMES if f not in PATTERN_SEARCH_EXCLUDE]

    for feat in features_to_try:
        if feat not in df_train.columns: continue
        vals = df_train[feat].values
        if len(vals) == 0: continue

        # Try percentile thresholds: above 60,70,80,90; below 10,20,30,40
        percentiles_hi = [60, 70, 75, 80, 85, 90, 95]
        percentiles_lo = [5, 10, 15, 20, 25, 30, 40]

        for pct in percentiles_hi:
            thresh = float(np.percentile(vals, pct))
            mask_tr = df_train[feat] >= thresh
            mask_va = df_val[feat] >= thresh if feat in df_val.columns else None
            stats_tr = evaluate_region(df_train, mask_tr, tp, sl)
            if stats_tr["n"] < min_samples_train or stats_tr["edge"] < min_edge_train: continue
            stats_va = evaluate_region(df_val, mask_va, tp, sl) if mask_va is not None else None
            if not stats_va or stats_va["n"] < min_samples_val or stats_va["edge"] < min_edge_val: continue
            candidates.append({
                "conditions": [{"feature":feat, "op":">=", "value":round(thresh, 6), "percentile":pct}],
                "train": stats_tr,
                "val": stats_va,
                "feat_count": 1
            })

        for pct in percentiles_lo:
            thresh = float(np.percentile(vals, pct))
            mask_tr = df_train[feat] <= thresh
            mask_va = df_val[feat] <= thresh if feat in df_val.columns else None
            stats_tr = evaluate_region(df_train, mask_tr, tp, sl)
            if stats_tr["n"] < min_samples_train or stats_tr["edge"] < min_edge_train: continue
            stats_va = evaluate_region(df_val, mask_va, tp, sl) if mask_va is not None else None
            if not stats_va or stats_va["n"] < min_samples_val or stats_va["edge"] < min_edge_val: continue
            candidates.append({
                "conditions": [{"feature":feat, "op":"<=", "value":round(thresh, 6), "percentile":pct}],
                "train": stats_tr,
                "val": stats_va,
                "feat_count": 1
            })

    return candidates

def condition_mask(df, conditions):
    """Build a boolean mask from a list of conditions."""
    mask = pd.Series([True] * len(df), index=df.index)
    for c in conditions:
        feat, op, val = c["feature"], c["op"], c["value"]
        if feat not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        if op == ">=":
            mask &= (df[feat] >= val)
        elif op == "<=":
            mask &= (df[feat] <= val)
    return mask

def search_two_feature_patterns(df_train, df_val, tp, sl, single_patterns,
                                 min_samples_train=80, min_samples_val=25,
                                 min_edge_train=8.0, min_edge_val=5.0):
    """
    Take top N single-feature patterns by training edge and try pairwise AND combinations.
    Stricter thresholds because 2-feature combinations have more overfitting risk.
    """
    # Rank single patterns by training edge and keep top 12 for combining
    top_singles = sorted(single_patterns, key=lambda p: p["train"]["edge"], reverse=True)[:12]
    combos = []
    seen = set()

    for i, pa in enumerate(top_singles):
        for j, pb in enumerate(top_singles):
            if i == j: continue
            # Don't combine two conditions on the same feature
            if pa["conditions"][0]["feature"] == pb["conditions"][0]["feature"]: continue
            # Sort conditions for dedup
            key = tuple(sorted([
                (c["feature"], c["op"], c["value"])
                for c in pa["conditions"] + pb["conditions"]
            ]))
            if key in seen: continue
            seen.add(key)

            combined_conds = pa["conditions"] + pb["conditions"]
            mask_tr = condition_mask(df_train, combined_conds)
            stats_tr = evaluate_region(df_train, mask_tr, tp, sl)
            if stats_tr["n"] < min_samples_train or stats_tr["edge"] < min_edge_train: continue
            mask_va = condition_mask(df_val, combined_conds)
            stats_va = evaluate_region(df_val, mask_va, tp, sl)
            if stats_va["n"] < min_samples_val or stats_va["edge"] < min_edge_val: continue
            combos.append({
                "conditions": combined_conds,
                "train": stats_tr,
                "val": stats_va,
                "feat_count": 2
            })

    return combos

# ─── SENSITIVITY SWEEP (diagnostic: how many patterns at various edge bars) ──
sensitivity_in_progress = False
sensitivity_progress = {"phase":"idle","pct":0,"message":""}
SENSITIVITY_PATH = DATA_DIR / "sensitivity_results.json"

# Thresholds tested: (train_edge, val_edge) pairs
SENSITIVITY_BARS = [
    (5.0, 3.0),   # current production bar
    (4.0, 2.0),
    (3.0, 1.0),
    (2.0, 0.0),
    (1.0, -1.0),
    (0.0, -2.0),  # "any above-chance region on val"
]

def collect_all_single_feature_thresholds(df_train, df_val, tp, sl,
                                           min_samples_train=100, min_samples_val=30):
    """
    Like search_single_feature_patterns, but keeps ALL evaluated thresholds that
    simply meet the sample-size bar. Does NOT filter by edge. Used for sensitivity analysis.
    """
    results = []
    features_to_try = [f for f in FEATURE_NAMES if f not in PATTERN_SEARCH_EXCLUDE]

    for feat in features_to_try:
        if feat not in df_train.columns: continue
        vals = df_train[feat].values
        if len(vals) == 0: continue

        percentiles_hi = [60, 70, 75, 80, 85, 90, 95]
        percentiles_lo = [5, 10, 15, 20, 25, 30, 40]

        for pct in percentiles_hi:
            thresh = float(np.percentile(vals, pct))
            mask_tr = df_train[feat] >= thresh
            mask_va = df_val[feat] >= thresh if feat in df_val.columns else None
            stats_tr = evaluate_region(df_train, mask_tr, tp, sl)
            if stats_tr["n"] < min_samples_train: continue
            if mask_va is None: continue
            stats_va = evaluate_region(df_val, mask_va, tp, sl)
            if stats_va["n"] < min_samples_val: continue
            results.append({
                "feature": feat, "op": ">=", "value": round(thresh, 6), "percentile": pct,
                "train": stats_tr, "val": stats_va
            })

        for pct in percentiles_lo:
            thresh = float(np.percentile(vals, pct))
            mask_tr = df_train[feat] <= thresh
            mask_va = df_val[feat] <= thresh if feat in df_val.columns else None
            stats_tr = evaluate_region(df_train, mask_tr, tp, sl)
            if stats_tr["n"] < min_samples_train: continue
            if mask_va is None: continue
            stats_va = evaluate_region(df_val, mask_va, tp, sl)
            if stats_va["n"] < min_samples_val: continue
            results.append({
                "feature": feat, "op": "<=", "value": round(thresh, 6), "percentile": pct,
                "train": stats_tr, "val": stats_va
            })

    return results

def run_sensitivity_sweep(tp_pct=None, sl_pct=None):
    """
    For each scan hour, collect all single-feature threshold evaluations,
    then bucket them by (train_edge, val_edge) bars. Shows how many patterns
    pass at each sensitivity level — diagnostic for signal presence.
    """
    global sensitivity_in_progress, sensitivity_progress

    if sensitivity_in_progress: return
    sensitivity_in_progress = True
    sensitivity_progress = {"phase":"starting","pct":0,"message":"Starting sensitivity sweep..."}

    try:
        if not TRAINING_ROWS_CACHE.exists():
            sensitivity_progress = {"phase":"error","pct":0,
                "message":"No training data cached. Run training first."}
            return

        # v8: barriers are ATR-scaled per stock; report multipliers instead
        use_tp_mult = tp_pct if tp_pct is not None else TP_MULT
        use_sl_mult = sl_pct if sl_pct is not None else SL_MULT

        sensitivity_progress = {"phase":"loading","pct":5,"message":"Loading training data..."}
        rows_per_hour = pickle.loads(TRAINING_ROWS_CACHE.read_bytes())

        all_results = {
            "generatedAt": datetime.now(ET).isoformat(),
            "tp_mult": use_tp_mult, "sl_mult": use_sl_mult,
            "notional_breakeven": round(use_sl_mult/(use_sl_mult+use_tp_mult)*100, 2),
            "bars": [{"train_edge": tb, "val_edge": vb} for tb, vb in SENSITIVITY_BARS],
            "hours": {}
        }

        total_hours = len([h for h in SCAN_HOURS if h in rows_per_hour and len(rows_per_hour[h]) >= 500])
        processed = 0

        for h in SCAN_HOURS:
            rows = rows_per_hour.get(h, [])
            if len(rows) < 500:
                all_results["hours"][str(h)] = {"error":"insufficient data", "n_samples":len(rows)}
                continue

            processed += 1
            pct_base = 5 + int((processed-1)/total_hours * 90)
            sensitivity_progress = {"phase":"searching","pct":pct_base,
                "message":f"Scan {h}:00 — evaluating all thresholds..."}

            df = pd.DataFrame(rows)
            dates = sorted(df["date"].unique())
            split_idx = int(len(dates) * 0.8)
            train_dates = set(dates[:split_idx])
            val_dates = set(dates[split_idx:])
            df_train = df[df["date"].isin(train_dates)].copy()
            df_val = df[df["date"].isin(val_dates)].copy()

            all_thresh = collect_all_single_feature_thresholds(df_train, df_val, use_tp_mult, use_sl_mult)

            # Bucket by bar
            buckets = []
            for tb, vb in SENSITIVITY_BARS:
                passing = [p for p in all_thresh if p["train"]["edge"] >= tb and p["val"]["edge"] >= vb]
                # Show top 3 by val edge, if any
                passing.sort(key=lambda p: p["val"]["edge"], reverse=True)
                top_examples = passing[:3]
                buckets.append({
                    "train_edge": tb, "val_edge": vb,
                    "n_passing": len(passing),
                    "top_examples": top_examples
                })

            # Also emit distribution stats: percentiles of val edge across all evaluated thresholds
            val_edges = [p["val"]["edge"] for p in all_thresh]
            train_edges = [p["train"]["edge"] for p in all_thresh]

            all_results["hours"][str(h)] = {
                "n_train": len(df_train),
                "n_val": len(df_val),
                "base_wr_val": round(df_val["label"].mean()*100, 2),
                "n_thresholds_evaluated": len(all_thresh),
                "val_edge_stats": {
                    "min": round(min(val_edges),2) if val_edges else None,
                    "p10": round(float(np.percentile(val_edges,10)),2) if val_edges else None,
                    "p50": round(float(np.percentile(val_edges,50)),2) if val_edges else None,
                    "p90": round(float(np.percentile(val_edges,90)),2) if val_edges else None,
                    "max": round(max(val_edges),2) if val_edges else None,
                },
                "train_edge_stats": {
                    "min": round(min(train_edges),2) if train_edges else None,
                    "p50": round(float(np.percentile(train_edges,50)),2) if train_edges else None,
                    "max": round(max(train_edges),2) if train_edges else None,
                },
                "buckets": buckets
            }

            log.info(f"Sensitivity {h}:00 — evaluated {len(all_thresh)} thresholds, "
                     f"val edge range [{min(val_edges) if val_edges else 'N/A'}, {max(val_edges) if val_edges else 'N/A'}]")

        SENSITIVITY_PATH.write_text(json.dumps(all_results, indent=2, default=str))
        sensitivity_progress = {"phase":"done","pct":100,
            "message":f"Done. Sensitivity analysis across {processed} scan hours."}
        log.info(f"Sensitivity sweep complete")

    except Exception as e:
        log.error(f"Sensitivity sweep failed: {e}", exc_info=True)
        sensitivity_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        sensitivity_in_progress = False

def load_sensitivity():
    if not SENSITIVITY_PATH.exists(): return None
    try: return json.loads(SENSITIVITY_PATH.read_text())
    except: return None

def run_pattern_search(tp_pct=None, sl_pct=None):
    """
    Search for feature-region patterns in the persisted training data.
    Saves results to PATTERNS_PATH for use by live scans.
    """
    global pattern_search_in_progress, pattern_search_progress

    if pattern_search_in_progress: return
    pattern_search_in_progress = True
    pattern_search_progress = {"phase":"starting","pct":0,"message":"Starting pattern search..."}

    try:
        if not TRAINING_ROWS_CACHE.exists():
            pattern_search_progress = {"phase":"error","pct":0,
                "message":"No training data cached. Run training first."}
            return

        # v8: ATR multipliers
        use_tp_mult = tp_pct if tp_pct is not None else TP_MULT
        use_sl_mult = sl_pct if sl_pct is not None else SL_MULT

        pattern_search_progress = {"phase":"loading","pct":5,"message":"Loading training data..."}
        rows_per_hour = pickle.loads(TRAINING_ROWS_CACHE.read_bytes())

        all_results = {
            "generatedAt": datetime.now(ET).isoformat(),
            "tp_mult": use_tp_mult, "sl_mult": use_sl_mult,
            "notional_breakeven": round(use_sl_mult/(use_sl_mult+use_tp_mult)*100, 2),
            "hours": {}
        }

        total_hours = len([h for h in SCAN_HOURS if h in rows_per_hour and len(rows_per_hour[h]) >= 500])
        processed_hours = 0

        for h in SCAN_HOURS:
            rows = rows_per_hour.get(h, [])
            if len(rows) < 500:
                log.info(f"Pattern search {h}:00 — only {len(rows)} rows, skipping")
                all_results["hours"][str(h)] = {"error":"insufficient data", "n_samples":len(rows)}
                continue

            processed_hours += 1
            pct_base = 5 + int((processed_hours-1)/total_hours * 90)
            pattern_search_progress = {"phase":"searching","pct":pct_base,
                "message":f"Scan {h}:00 — building dataset..."}

            df = pd.DataFrame(rows)
            # Temporal 80/20 split, same as training
            dates = sorted(df["date"].unique())
            split_idx = int(len(dates) * 0.8)
            train_dates = set(dates[:split_idx])
            val_dates = set(dates[split_idx:])
            df_train = df[df["date"].isin(train_dates)].copy()
            df_val = df[df["date"].isin(val_dates)].copy()

            pattern_search_progress = {"phase":"searching","pct":pct_base+2,
                "message":f"Scan {h}:00 — single-feature search ({len(df_train)} train / {len(df_val)} val)..."}

            singles = search_single_feature_patterns(df_train, df_val, use_tp_mult, use_sl_mult)

            pattern_search_progress = {"phase":"searching","pct":pct_base+5,
                "message":f"Scan {h}:00 — two-feature combinations..."}

            pairs = search_two_feature_patterns(df_train, df_val, use_tp_mult, use_sl_mult, singles)

            # Rank by validation edge (the honest measurement)
            all_patterns = singles + pairs
            all_patterns.sort(key=lambda p: p["val"]["edge"], reverse=True)

            # Keep top 20 per hour
            top_patterns = all_patterns[:20]

            # Compute base stats for context
            base_wr = df_val["label"].mean() * 100
            base_pnl = df_val["pnl"].mean()

            all_results["hours"][str(h)] = {
                "n_train": len(df_train),
                "n_val": len(df_val),
                "base_wr_val": round(base_wr, 2),
                "base_pnl_val": round(base_pnl, 3),
                "n_patterns_found": len(all_patterns),
                "patterns": top_patterns
            }

            log.info(f"Pattern search {h}:00 — {len(singles)} single, {len(pairs)} pair, "
                     f"top val edge: {top_patterns[0]['val']['edge'] if top_patterns else 'N/A'}%")

        # Save results
        PATTERNS_PATH.write_text(json.dumps(all_results, indent=2, default=str))

        total_patterns = sum(len(h.get("patterns",[])) for h in all_results["hours"].values() if isinstance(h, dict))
        pattern_search_progress = {"phase":"done","pct":100,
            "message":f"Done. {total_patterns} patterns surfaced across {processed_hours} scan hours."}
        log.info(f"Pattern search complete — {total_patterns} patterns saved to {PATTERNS_PATH}")

    except Exception as e:
        log.error(f"Pattern search failed: {e}", exc_info=True)
        pattern_search_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        pattern_search_in_progress = False

def load_patterns():
    """Load saved patterns from disk."""
    if not PATTERNS_PATH.exists(): return None
    try: return json.loads(PATTERNS_PATH.read_text())
    except: return None

def check_patterns(features_dict, scan_hour, patterns_data):
    """
    Given a stock's features and scan hour, return list of matching patterns.
    Returns: list of dicts, each with pattern conditions and stats.
    """
    if not patterns_data: return []
    hour_data = patterns_data.get("hours", {}).get(str(scan_hour), {})
    if not isinstance(hour_data, dict) or "patterns" not in hour_data: return []

    matches = []
    for pattern in hour_data["patterns"]:
        all_conds_met = True
        for c in pattern["conditions"]:
            feat, op, val = c["feature"], c["op"], c["value"]
            fv = features_dict.get(feat)
            if fv is None:
                all_conds_met = False; break
            if op == ">=" and fv < val:
                all_conds_met = False; break
            if op == "<=" and fv > val:
                all_conds_met = False; break
        if all_conds_met:
            matches.append(pattern)
    return matches

# ═══════════════════════════════════════════════════════════════════
# THRESHOLD ANALYZER — conviction-gated strategy evaluation
# ═══════════════════════════════════════════════════════════════════
thresh_analysis_in_progress = False
thresh_analysis_progress = {"phase":"idle","pct":0,"message":""}
THRESHOLD_RESULTS_PATH = DATA_DIR / "threshold_results.json"
THRESHOLD_CURVE_STEPS = [round(0.50 + i*0.01, 2) for i in range(40)]  # 0.50 to 0.89

def _eval_threshold_on_fold(df_fold, thr, fold_dates):
    """
    Given a fold's data and a threshold, simulate the strategy:
    - On each date, take top-1 stock by calibrated probability IF its prob >= threshold
    - Otherwise no trade that day
    Returns per-fold aggregate stats.
    """
    daily_pnls = []  # per-day PnL (0 for no-trade days)
    trade_labels = []  # labels of trades that fired
    trade_pnls = []  # pnls of trades that fired

    for d in sorted(fold_dates):
        day_df = df_fold[df_fold["date"] == d]
        if len(day_df) == 0:
            continue
        # Top-1 by calibrated probability
        top = day_df.nlargest(1, "cal_prob").iloc[0]
        if top["cal_prob"] >= thr:
            trade_labels.append(int(top["label"]))
            trade_pnls.append(float(top["pnl"]))
            daily_pnls.append(float(top["pnl"]))
        else:
            daily_pnls.append(0.0)

    n_trades = len(trade_labels)
    trading_days = n_trades  # top-1, so trades per day ≤ 1
    total_days = len(fold_dates)

    if n_trades == 0:
        return {
            "threshold": thr, "n_trades": 0, "trading_days": 0, "total_days": total_days,
            "trade_freq": 0, "wr": None, "avg_pnl_trade": None,
            "avg_pnl_day": 0, "std_pnl_day": 0, "cum_pnl": 0,
            "sharpe": None, "pos_days": 0, "pos_day_frac": 0,
        }

    wr = float(np.mean(trade_labels) * 100)
    avg_pnl_trade = float(np.mean(trade_pnls))
    avg_pnl_day = float(np.mean(daily_pnls))
    std_pnl_day = float(np.std(daily_pnls))
    cum_pnl = float(np.sum(daily_pnls))
    trade_freq = trading_days / total_days
    # Sharpe-like: avg daily PnL / std daily PnL (when std > 0)
    sharpe = (avg_pnl_day / std_pnl_day) if std_pnl_day > 1e-6 else 0
    pos_days = sum(1 for p in trade_pnls if p > 0)
    pos_day_frac = pos_days / n_trades if n_trades > 0 else 0

    return {
        "threshold": round(thr, 2),
        "n_trades": int(n_trades),
        "trading_days": int(trading_days),
        "total_days": int(total_days),
        "trade_freq": round(trade_freq, 3),
        "wr": round(wr, 2),
        "avg_pnl_trade": round(avg_pnl_trade, 3),
        "avg_pnl_day": round(avg_pnl_day, 3),
        "std_pnl_day": round(std_pnl_day, 3),
        "cum_pnl": round(cum_pnl, 2),
        "sharpe": round(sharpe, 4),
        "pos_days": int(pos_days),
        "pos_day_frac": round(pos_day_frac, 3),
    }

def run_threshold_analysis():
    """
    Per scan hour:
      1. Split val+test predictions (we already have the three-way split from training).
      2. For each threshold in [0.50, 0.89] step 0.01:
         - Simulate top-1 go/no-go strategy on VAL
         - Check guardrails: ≥10 trading days, ≥20 trades, ≥40% pos_day_frac
         - Record val stats
      3. Pick threshold maximizing VAL Sharpe among those passing guardrails.
      4. Evaluate that chosen threshold on TEST — report test stats side by side.
      5. If no threshold passes guardrails → this hour is "no trade."

    Test stats are the honest out-of-sample numbers. Val stats are in-sample
    (where the threshold was chosen). Gap between them reveals overfitting.
    """
    global thresh_analysis_in_progress, thresh_analysis_progress

    if thresh_analysis_in_progress: return
    thresh_analysis_in_progress = True
    thresh_analysis_progress = {"phase":"starting","pct":0,"message":"Starting threshold analysis (three-way split)..."}

    try:
        if not TRAINING_ROWS_CACHE.exists():
            thresh_analysis_progress = {"phase":"error","pct":0,"message":"No training data. Run training first."}
            return
        if not models:
            thresh_analysis_progress = {"phase":"error","pct":0,"message":"No models loaded."}
            return

        thresh_analysis_progress = {"phase":"loading","pct":5,"message":"Loading training rows..."}
        rows_per_hour = pickle.loads(TRAINING_ROWS_CACHE.read_bytes())

        results = {
            "generatedAt": datetime.now(ET).isoformat(),
            "method": "three_way_split_sharpe_selection",
            "objective": "Maximize Sharpe on val. Guardrails: ≥10 trading days, ≥20 trades, ≥40% winning trades.",
            "rule": "At scan time, if top-1 stock's calibrated prob ≥ chosen_threshold, take that position. Otherwise no trade.",
            "thresholds": THRESHOLD_CURVE_STEPS,
            "hours": {}
        }

        n_hours = len([h for h in SCAN_HOURS if h in models and h in rows_per_hour])
        processed = 0

        for h in SCAN_HOURS:
            if h not in models or h not in rows_per_hour:
                results["hours"][str(h)] = {"error":"no model or data"}
                continue

            processed += 1
            pct = 5 + int((processed-1)/n_hours * 90)
            thresh_analysis_progress = {"phase":"analyzing","pct":pct,
                "message":f"Analyzing {h}:00 — three-way split..."}

            rows = rows_per_hour[h]
            df = pd.DataFrame(rows)
            dates = sorted(df["date"].unique())
            # Must match the training-time split: 60/20/20 temporal
            split_tr = int(len(dates) * 0.6)
            split_va = int(len(dates) * 0.8)
            train_dates = set(dates[:split_tr])
            val_dates = set(dates[split_tr:split_va])
            test_dates = set(dates[split_va:])

            df_val = df[df["date"].isin(val_dates)].copy()
            df_test = df[df["date"].isin(test_dates)].copy()

            # Predict + calibrate for both val and test
            cal = calibrators.get(h)
            if len(df_val) > 0:
                val_raw = models[h].predict(df_val[FEATURE_NAMES].values)
                df_val["cal_prob"] = cal.predict(val_raw) if cal is not None else val_raw
            if len(df_test) > 0:
                test_raw = models[h].predict(df_test[FEATURE_NAMES].values)
                df_test["cal_prob"] = cal.predict(test_raw) if cal is not None else test_raw

            # Base rates on each fold
            base_wr_val = float(df_val["label"].mean() * 100) if len(df_val) > 0 else None
            base_wr_test = float(df_test["label"].mean() * 100) if len(df_test) > 0 else None
            m = model_meta.get(h, {})
            ld = m.get("loss_distribution", {})
            realized_be = ld.get("realized_breakeven_wr")

            # Sweep thresholds on VAL, compute val stats
            val_curve = []
            for thr in THRESHOLD_CURVE_STEPS:
                val_stats = _eval_threshold_on_fold(df_val, thr, val_dates)
                val_curve.append(val_stats)

            # Apply guardrails to val curve: ≥10 trading days, ≥20 trades, ≥40% winning trades
            # Top-1 → trading_days == n_trades, so 10 days == 10 trades. Tighten: ≥15 trades (roughly quarterly).
            eligible = [c for c in val_curve
                        if c["n_trades"] >= 15
                        and c["pos_day_frac"] >= 0.40
                        and c["sharpe"] is not None]

            # Pick the Sharpe-maximizing threshold from eligible set
            best_val = max(eligible, key=lambda c: c["sharpe"], default=None) if eligible else None

            # Evaluate chosen threshold on TEST
            best_test = None
            if best_val is not None and len(df_test) > 0:
                best_test = _eval_threshold_on_fold(df_test, best_val["threshold"], test_dates)

            # Full test curve for display (useful to see how test numbers vary by threshold)
            test_curve = []
            if len(df_test) > 0:
                for thr in THRESHOLD_CURVE_STEPS:
                    test_curve.append(_eval_threshold_on_fold(df_test, thr, test_dates))

            results["hours"][str(h)] = {
                "n_train_dates": split_tr,
                "n_val_dates": len(val_dates),
                "n_test_dates": len(test_dates),
                "base_wr_val": round(base_wr_val, 2) if base_wr_val is not None else None,
                "base_wr_test": round(base_wr_test, 2) if base_wr_test is not None else None,
                "realized_be": realized_be,
                "val_curve": val_curve,
                "test_curve": test_curve,
                "best_val": best_val,
                "best_test": best_test,
                "chosen_threshold": best_val["threshold"] if best_val else None,
                "eligible_count": len(eligible),
            }

            if best_val and best_test:
                log.info(f"Thr {h}:00 — chose {best_val['threshold']} | "
                         f"VAL: Sharpe {best_val['sharpe']}, WR {best_val['wr']}%, PnL/day {best_val['avg_pnl_day']}%, trades {best_val['n_trades']}/{best_val['total_days']} | "
                         f"TEST: Sharpe {best_test['sharpe']}, WR {best_test['wr']}%, PnL/day {best_test['avg_pnl_day']}%, trades {best_test['n_trades']}/{best_test['total_days']}")
            elif best_val:
                log.info(f"Thr {h}:00 — chose {best_val['threshold']} on val but no test data")
            else:
                log.info(f"Thr {h}:00 — NO THRESHOLD passed guardrails on val (no trade hour)")

        THRESHOLD_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
        thresh_analysis_progress = {"phase":"done","pct":100,
            "message":f"Done. Per-hour thresholds chosen on val, reported on test."}
        log.info("Threshold analysis complete (three-way split)")

    except Exception as e:
        log.error(f"Threshold analysis failed: {e}", exc_info=True)
        thresh_analysis_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        thresh_analysis_in_progress = False

def load_threshold_results():
    if not THRESHOLD_RESULTS_PATH.exists(): return None
    try: return json.loads(THRESHOLD_RESULTS_PATH.read_text())
    except: return None

def load_active_thresholds():
    """Return a dict {scan_hour -> threshold} for live gating, using chosen_threshold from each hour's analysis."""
    data = load_threshold_results()
    if not data: return {}
    result = {}
    for h_str, hdata in data.get("hours", {}).items():
        try:
            h = int(h_str)
            chosen = hdata.get("chosen_threshold")
            if chosen is not None:
                result[h] = float(chosen)
        except: pass
    return result

# ═══════════════════════════════════════════════════════════════════
# v9: TECHNICAL SETUP DETECTORS
# ═══════════════════════════════════════════════════════════════════
# Hypothesis-first approach: predefined technical setups, each with precise
# activation conditions. No ML. At scan time, check each stock against each
# setup. If it matches, it's a candidate. Outcome = did price reach
# entry*1.01 before 15:55 close?

SETUP_NAMES = [
    # v9 original 5
    "orb_vol", "vwap_reclaim", "consol_break", "bull_flag", "sector_bounce",
    # v11 expansion — 15 additional well-known technical setups
    "gap_and_go", "gap_fade",
    "orb_60_break", "pivot_break", "yh_break", "inside_bar_break", "three_bar_thrust",
    "hhhl_momentum", "rel_strength_iwm", "opening_drive",
    "midday_reversal", "end_of_day_momentum",
    "vol_dryup_expansion", "vol_climax_reversal",
    "atr_contraction",
]
SETUP_DESCRIPTIONS = {
    "orb_vol": "Opening Range Breakout with volume (>1.5× avg) above VWAP",
    "vwap_reclaim": "VWAP reclaim after gap-up then fill; held above VWAP 2+ bars",
    "consol_break": "Breakout above tight 30-min consolidation (range <0.5%) with elevated volume",
    "bull_flag": "Bull flag: pole ≥2%, shallow 30-60% pullback, consolidation, breakout",
    "sector_bounce": "Intraday-low bounce with sector breadth >50% and green volume bar",
    "gap_and_go": "Gap-up >1% that never fills (low stays above prior close) + above ORH",
    "gap_fade": "Gap-up >1%, filled to prior close, then bounced back up (V-reversal)",
    "orb_60_break": "60-min opening range breakout (wider range, later timing)",
    "pivot_break": "Breakout above classical floor-trader pivot ((H+L+C)/3 of prior day)",
    "yh_break": "Break above yesterday's high with volume > 1.3× avg",
    "inside_bar_break": "Today's range entirely within yesterday's range, then break above",
    "three_bar_thrust": "Three consecutive higher-high bars after a pullback",
    "hhhl_momentum": "Higher-high + higher-low pattern for 4+ consecutive bars",
    "rel_strength_iwm": "Stock outperforming IWM by >0.5% while IWM flat-to-up",
    "opening_drive": "First 15-min close >1% from open with rising volume",
    "midday_reversal": "After 12:00 ET: bounce from new intraday low with volume expansion",
    "end_of_day_momentum": "After 14:30 ET: strong directional bar (>0.5%) with >1.5× volume",
    "vol_dryup_expansion": "3+ bars below-avg volume then >2× surge",
    "vol_climax_reversal": ">3× avg volume bar at intraday low, followed by green close",
    "atr_contraction": "Bar range <0.5× ATR for 3+ bars, then expansion breakout",
}

def _compute_vwap_series(bars):
    """Running VWAP across the bar sequence. Returns list of (vwap_at_bar_i) aligned with bars."""
    vwap = []
    cum_pv, cum_v = 0.0, 0.0
    for b in bars:
        typical = (b["h"] + b["l"] + b["c"]) / 3.0
        cum_pv += typical * b["v"]
        cum_v += b["v"]
        vwap.append(cum_pv / cum_v if cum_v > 0 else b["c"])
    return vwap

def detect_setups(bars_before_scan, scan_price, prev_close, sector_breadth=None,
                  prior_daily=None, iwm_bars=None, scan_minute_et=None):
    """
    Given bars up to (not including) the scan bar, and the current scan_price,
    return a dict of {setup_name: True/False}.

    bars_before_scan: list of 5-min bars from market open up to scan time (exclusive)
    scan_price: price at scan time (entry price proxy)
    prev_close: yesterday's daily close (for gap calc). None → skip gap-dependent setups.
    sector_breadth: fraction of sector stocks currently green. None → skip sector_bounce.
    prior_daily: list of prior daily bars (for yesterday's H/L, ATR, pivot calc). None → skip those.
    iwm_bars: list of IWM 5-min bars up to scan time (for rel_strength_iwm).
    scan_minute_et: scan hour in minutes since midnight ET (e.g., 11:00 → 660). Used by time-
      sensitive setups like midday_reversal (post-12:00) and end_of_day_momentum (post-14:30).
    """
    active = {s: False for s in SETUP_NAMES}

    n = len(bars_before_scan)
    if n < 6: return active  # need at least first 30 min

    bars = bars_before_scan
    vwap_series = _compute_vwap_series(bars)
    current_vwap = vwap_series[-1]
    today_open = bars[0]["o"]
    day_low = min(b["l"] for b in bars)
    day_high = max(b["h"] for b in bars)
    vols = [b["v"] for b in bars]
    avg_vol = sum(vols) / len(vols) if vols else 0
    median_vol = sorted(vols)[len(vols)//2] if vols else 0
    last = bars[-1]

    # Gap percentage (used by gap_and_go, gap_fade)
    gap_pct = None
    if prev_close is not None and prev_close > 0:
        gap_pct = (today_open - prev_close) / prev_close

    # Yesterday's H/L and pivot (used by yh_break, pivot_break, inside_bar_break)
    yh, yl, yc, pivot = None, None, None, None
    if prior_daily and len(prior_daily) >= 1:
        yday = prior_daily[-1]
        yh, yl, yc = yday["h"], yday["l"], yday["c"]
        pivot = (yh + yl + yc) / 3.0

    # Daily ATR for atr_contraction setup (14-day, as % of current price)
    atr_pct = None
    if prior_daily and len(prior_daily) >= 15:
        atr_pct = compute_atr_pct(prior_daily, lookback=14)

    # ── Setup 1: ORB with volume ──────────────────────────────────
    # Opening range = first 6 bars (09:30 – 10:00 ET)
    orh = max(b["h"] for b in bars[:6])
    orl = min(b["l"] for b in bars[:6])
    orb_avg_v = sum(b["v"] for b in bars[:6]) / 6.0
    if (scan_price > orh
        and last["v"] > 1.5 * orb_avg_v
        and scan_price > current_vwap):
        active["orb_vol"] = True

    # ── Setup 2: VWAP reclaim after gap fill ──────────────────────
    if gap_pct is not None and gap_pct > 0.005:  # gap up >0.5%
        touched_vwap = any(bars[i]["l"] <= vwap_series[i] for i in range(n))
        if n >= 2 and touched_vwap:
            held = (bars[-1]["c"] > vwap_series[-1]
                    and bars[-2]["c"] > vwap_series[-2])
            if held and scan_price > current_vwap:
                active["vwap_reclaim"] = True

    # ── Setup 3: Tight consolidation breakout ─────────────────────
    if n >= 12:
        consol = bars[-7:-1]
        consol_high = max(b["h"] for b in consol)
        consol_low = min(b["l"] for b in consol)
        consol_mid = (consol_high + consol_low) / 2.0
        tight_range_pct = (consol_high - consol_low) / consol_mid if consol_mid > 0 else 1
        consol_avg_v = sum(b["v"] for b in consol) / 6.0
        pre = bars[:-7]
        pre_avg_v = (sum(b["v"] for b in pre) / len(pre)) if pre else 0
        vol_elevated = pre_avg_v > 0 and consol_avg_v > 1.3 * pre_avg_v
        if (tight_range_pct < 0.005
            and vol_elevated
            and scan_price > consol_high):
            active["consol_break"] = True

    # ── Setup 4: Bull flag ────────────────────────────────────────
    if n >= 10:
        window = bars[-min(20, n):]
        best_pole = None
        for peak_i in range(2, len(window)):
            peak_price = window[peak_i]["h"]
            prior_lows = [window[j]["l"] for j in range(peak_i)]
            if not prior_lows: continue
            low_price = min(prior_lows)
            low_i = prior_lows.index(low_price)
            if low_i >= peak_i: continue
            pole_pct = (peak_price - low_price) / low_price if low_price > 0 else 0
            if pole_pct >= 0.02:
                if best_pole is None or pole_pct > best_pole[4]:
                    best_pole = (low_i, peak_i, low_price, peak_price, pole_pct)
        if best_pole is not None:
            low_i, peak_i, pl, ph, _ = best_pole
            pole_depth = ph - pl
            flag_bars = window[peak_i+1:]
            if len(flag_bars) >= 4:
                flag_low = min(b["l"] for b in flag_bars[:-1])
                flag_high = max(b["h"] for b in flag_bars[:-1])
                pullback = (ph - flag_low) / pole_depth if pole_depth > 0 else 0
                if 0.30 <= pullback <= 0.60 and scan_price > flag_high:
                    active["bull_flag"] = True

    # ── Setup 5: Sector bounce ────────────────────────────────────
    if sector_breadth is not None and sector_breadth > 0.50:
        recent_low = min(b["l"] for b in bars[-6:])
        if recent_low <= day_low * 1.002:
            is_green = last["c"] > last["o"]
            if is_green and last["v"] > median_vol:
                active["sector_bounce"] = True

    # ── Setup 6: Gap-and-go ───────────────────────────────────────
    # Gap up >1%, low never reached yesterday's close, price above ORH
    if gap_pct is not None and gap_pct > 0.01 and prev_close is not None:
        gap_never_filled = day_low > prev_close
        if gap_never_filled and scan_price > orh:
            active["gap_and_go"] = True

    # ── Setup 7: Gap-fade (V-reversal) ────────────────────────────
    # Gap up >1%, low touched prior close (gap filled), then bounced back up
    if gap_pct is not None and gap_pct > 0.01 and prev_close is not None:
        gap_filled = day_low <= prev_close * 1.002
        bounced = scan_price > prev_close * 1.005  # bounced at least 0.5% off fill
        if gap_filled and bounced:
            active["gap_fade"] = True

    # ── Setup 8: 60-min ORB breakout ──────────────────────────────
    # Requires at least 13 bars (65 min past open)
    if n >= 13:
        orb60 = bars[:12]  # first 12 bars = 60 min
        orh60 = max(b["h"] for b in orb60)
        orb60_avg_v = sum(b["v"] for b in orb60) / 12.0
        if scan_price > orh60 and last["v"] > 1.5 * orb60_avg_v:
            active["orb_60_break"] = True

    # ── Setup 9: Pivot break ──────────────────────────────────────
    # Breakout above classical floor-trader pivot (H+L+C)/3 of prior day
    if pivot is not None:
        # Was price previously below pivot? Then crossed above?
        crossed = any(bars[i]["l"] <= pivot for i in range(n))
        if crossed and scan_price > pivot and last["c"] > pivot:
            active["pivot_break"] = True

    # ── Setup 10: Yesterday's high break ──────────────────────────
    if yh is not None:
        if scan_price > yh and last["v"] > 1.3 * avg_vol:
            # Require price wasn't already above YH for most of day (fresh break)
            bars_above_yh = sum(1 for b in bars[:-1] if b["c"] > yh)
            if bars_above_yh < n * 0.3:  # fresh break
                active["yh_break"] = True

    # ── Setup 11: Inside bar breakout ─────────────────────────────
    # Today's intraday range so far fits entirely within yesterday's range,
    # then breakout above today's high.
    if yh is not None and yl is not None and n >= 8:
        # Only consider the bars BEFORE the last bar for "today's range"
        prior_bars = bars[:-1]
        prior_high = max(b["h"] for b in prior_bars)
        prior_low = min(b["l"] for b in prior_bars)
        if prior_high <= yh and prior_low >= yl:
            if scan_price > prior_high:
                active["inside_bar_break"] = True

    # ── Setup 12: Three-bar thrust ────────────────────────────────
    # Three consecutive bars with strictly higher highs, ideally after a pullback
    if n >= 7:
        last3 = bars[-3:]
        higher_highs = last3[0]["h"] < last3[1]["h"] < last3[2]["h"]
        # Prior 4 bars should show a pullback: lowest low in those 4 < last3 first low
        pullback_zone = bars[-7:-3]
        pullback_low = min(b["l"] for b in pullback_zone)
        was_pullback = pullback_low < last3[0]["l"] and pullback_low < last3[1]["l"]
        if higher_highs and was_pullback and scan_price > last3[2]["h"] * 0.998:
            active["three_bar_thrust"] = True

    # ── Setup 13: Higher-high / higher-low momentum ───────────────
    if n >= 5:
        last4 = bars[-4:]
        all_hh = all(last4[i]["h"] > last4[i-1]["h"] for i in range(1, 4))
        all_hl = all(last4[i]["l"] > last4[i-1]["l"] for i in range(1, 4))
        if all_hh and all_hl and scan_price > current_vwap:
            active["hhhl_momentum"] = True

    # ── Setup 14: Relative strength vs IWM ────────────────────────
    if iwm_bars and len(iwm_bars) >= 2:
        iwm_open = iwm_bars[0]["o"]
        iwm_last = iwm_bars[-1]["c"]
        iwm_pct = (iwm_last - iwm_open) / iwm_open if iwm_open > 0 else 0
        stock_pct = (scan_price - today_open) / today_open if today_open > 0 else 0
        # Stock outperforming IWM by >0.5% AND IWM not down (flat to up)
        if stock_pct - iwm_pct > 0.005 and iwm_pct > -0.002:
            active["rel_strength_iwm"] = True

    # ── Setup 15: Opening drive ───────────────────────────────────
    # First 15-min (3 bars) close >1% from open, with rising volume
    if n >= 3:
        first3 = bars[:3]
        first3_close = first3[-1]["c"]
        first3_open = first3[0]["o"]
        first3_move = (first3_close - first3_open) / first3_open if first3_open > 0 else 0
        vol_rising = first3[0]["v"] < first3[1]["v"] < first3[2]["v"]
        if first3_move > 0.01 and vol_rising and scan_price > first3_close:
            active["opening_drive"] = True

    # ── Setup 16: Midday reversal ─────────────────────────────────
    # Only valid after 12:00 ET. New intraday low in last 6 bars, then bounce with volume.
    if scan_minute_et is not None and scan_minute_et >= 12*60 and n >= 12:
        # Find low in last 6 bars
        recent_window = bars[-6:]
        recent_low = min(b["l"] for b in recent_window)
        recent_low_is_new = recent_low <= day_low * 1.001  # within 0.1% of day low
        # Volume expansion: last bar volume > 1.5× avg of prior 6 bars
        prior_window = bars[-12:-6]
        prior_avg_v = sum(b["v"] for b in prior_window) / 6.0
        vol_expansion = last["v"] > 1.5 * prior_avg_v
        # Bounce: last bar green and price above its midpoint
        last_mid = (last["h"] + last["l"]) / 2
        bounced = last["c"] > last["o"] and last["c"] > last_mid
        if recent_low_is_new and vol_expansion and bounced:
            active["midday_reversal"] = True

    # ── Setup 17: End-of-day momentum ─────────────────────────────
    # Only valid after 14:30 ET. Strong directional bar with volume surge.
    if scan_minute_et is not None and scan_minute_et >= 14*60+30:
        last_move = (last["c"] - last["o"]) / last["o"] if last["o"] > 0 else 0
        if last_move > 0.005 and last["v"] > 1.5 * avg_vol and scan_price > current_vwap:
            active["end_of_day_momentum"] = True

    # ── Setup 18: Volume dry-up followed by expansion ─────────────
    # 3+ bars below avg volume, then latest bar >2× avg volume
    if n >= 5:
        prior_bars = bars[-4:-1]  # three bars before the latest
        all_dry = all(b["v"] < avg_vol * 0.8 for b in prior_bars)
        surge = last["v"] > 2.0 * avg_vol
        is_green = last["c"] > last["o"]
        if all_dry and surge and is_green and scan_price > current_vwap:
            active["vol_dryup_expansion"] = True

    # ── Setup 19: Volume climax reversal ──────────────────────────
    # Recent bar had >3× avg volume at or near intraday low, then followed by green close above
    if n >= 4:
        # Look in the last 3 bars for a climax bar
        for i in range(max(0, n-3), n-1):
            climax = bars[i]
            if climax["v"] > 3.0 * avg_vol and climax["l"] <= day_low * 1.002:
                # Check: latest bar closed green, above climax close
                if last["c"] > last["o"] and last["c"] > climax["c"]:
                    active["vol_climax_reversal"] = True
                    break

    # ── Setup 20: ATR contraction then expansion ──────────────────
    # Last 3-5 bars had range < 0.5 × daily ATR, then current bar breaks out with expansion
    if atr_pct is not None and n >= 5:
        daily_atr_pts = atr_pct * scan_price  # convert % ATR to price points
        recent_ranges = [b["h"] - b["l"] for b in bars[-4:-1]]  # 3 bars before last
        all_contracted = all(r < 0.5 * daily_atr_pts for r in recent_ranges)
        last_range = last["h"] - last["l"]
        expansion = last_range > daily_atr_pts * 0.3  # last bar range jumped
        breakout = scan_price > max(b["h"] for b in bars[-4:-1])  # broke above contraction range
        if all_contracted and expansion and breakout:
            active["atr_contraction"] = True

    return active

def did_hit_target(entry_price, bars_after_entry, target_pct=0.01):
    """Did price reach entry*(1+target_pct) before 15:55 close? Boolean + realized PnL."""
    target = entry_price * (1 + target_pct)
    for b in bars_after_entry:
        bm = bar_to_et_minutes(b)
        if bm is not None and bm >= FORCED_CLOSE_MIN:
            # Evaluate at close if no hit earlier
            return (False, (b["c"] - entry_price) / entry_price * 100)
        if b["h"] >= target:
            return (True, target_pct * 100)
    if bars_after_entry:
        final = bars_after_entry[-1]["c"]
        return (False, (final - entry_price) / entry_price * 100)
    return (False, 0.0)

def did_hit_target_within_horizon(entry_price, bars_after_entry, target_pct, horizon_minutes, scan_minute_et):
    """
    v26: Did price reach entry*(1+target_pct) WITHIN horizon_minutes after scan,
    OR before 15:55 force-close (whichever is earlier)?
    Returns boolean hit.
    """
    target = entry_price * (1 + target_pct)
    # Deadline = min(scan_minute + horizon, FORCED_CLOSE_MIN)
    deadline = min(scan_minute_et + horizon_minutes, FORCED_CLOSE_MIN)
    for b in bars_after_entry:
        bm = bar_to_et_minutes(b)
        if bm is None: continue
        if bm >= deadline:
            return False  # deadline passed, no hit
        if b["h"] >= target:
            return True
    return False

# ═══════════════════════════════════════════════════════════════════
# v9: SETUP EVALUATION — three-way split on historic data
# ═══════════════════════════════════════════════════════════════════
setup_eval_in_progress = False
setup_eval_progress = {"phase":"idle","pct":0,"message":""}
SETUP_RESULTS_PATH = DATA_DIR / "setup_results.json"

def run_setup_evaluation(target_pct=0.01):
    """
    For each scan hour and each setup:
    1. Scan 12 months of historic bar data.
    2. For every (date, ticker) where setup fires at that scan hour, record outcome
       (did price hit entry*1.01 before 15:55 close?).
    3. Three-way split by date: 60% train / 20% val / 20% test.
    4. Report hit rate per fold, base rate per fold, edge.
    5. Test column is the honest number.
    """
    global setup_eval_in_progress, setup_eval_progress
    if setup_eval_in_progress: return
    setup_eval_in_progress = True
    setup_eval_progress = {"phase":"starting","pct":0,"message":"Starting setup evaluation..."}

    try:
        # Load bars from cache — requires that run_training was run at least once
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            setup_eval_progress = {"phase":"error","pct":0,
                "message":"Bar cache missing. Run Training first to populate cache."}
            return

        setup_eval_progress = {"phase":"loading","pct":3,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        # v11 cache integrity fingerprint — compute BEFORE doing any eval work
        # This lets us detect when the underlying bar data changes between runs
        import hashlib
        def compute_cache_fingerprint(intraday_bars_dict):
            """Compact fingerprint of what's in the bar cache."""
            summary_parts = []
            total_bars = 0
            tickers_present = sorted(intraday_bars_dict.keys())
            for t in tickers_present:
                bars = intraday_bars_dict.get(t, [])
                n = len(bars)
                total_bars += n
                # Include first + last bar timestamps if any (detects data drift at edges)
                first_ts = bars[0]["t"] if n > 0 else ""
                last_ts = bars[-1]["t"] if n > 0 else ""
                summary_parts.append(f"{t}:{n}:{first_ts}:{last_ts}")
            fp_str = "|".join(summary_parts)
            h = hashlib.sha256(fp_str.encode()).hexdigest()[:16]
            return {
                "hash": h,
                "n_tickers": len(tickers_present),
                "total_bars": total_bars,
                "has_spy": "SPY" in intraday_bars_dict,
                "has_iwm": "IWM" in intraday_bars_dict,
                "n_spy_bars": len(intraday_bars_dict.get("SPY", [])),
                "n_iwm_bars": len(intraday_bars_dict.get("IWM", [])),
            }
        cache_fp = compute_cache_fingerprint(intraday_bars)
        log.info(f"Cache fingerprint: {cache_fp}")

        # Cache file mtime for audit purposes
        import os as _os
        cache_mtime_iso = datetime.fromtimestamp(
            _os.path.getmtime(BARS_INTRADAY_CACHE), tz=ET
        ).isoformat() if BARS_INTRADAY_CACHE.exists() else None

        # Group bars by ticker, date
        setup_eval_progress = {"phase":"indexing","pct":8,"message":"Indexing bars by date..."}
        by_td = defaultdict(lambda: defaultdict(list))
        # v12: also index SPY + IWM for regime and rel-strength analyses
        for ticker in list(TICKERS) + ["IWM", "SPY"]:
            for b in intraday_bars.get(ticker, []):
                by_td[ticker][b["t"][:10]].append(b)

        # v12 diagnostic: how many SPY/IWM bars actually indexed?
        n_spy_indexed = sum(len(v) for v in by_td.get("SPY", {}).values())
        n_iwm_indexed = sum(len(v) for v in by_td.get("IWM", {}).values())
        n_spy_dates = len(by_td.get("SPY", {}))
        log.info(f"Indexing: SPY {n_spy_indexed} bars across {n_spy_dates} dates; IWM {n_iwm_indexed} bars")
        if n_spy_indexed == 0:
            log.warning("SPY bars missing from cache — spy_dir regime will be unavailable. Retrain to fix.")

        # v12: index SPY daily bars too, for 5-day ATR-based vol regime
        spy_daily = daily_bars.get("SPY", [])
        spy_daily_by_date = {d["t"][:10]: d for d in spy_daily}

        all_dates = sorted(set(d for t in by_td for d in by_td[t]))
        if len(all_dates) < 50:
            setup_eval_progress = {"phase":"error","pct":0,
                "message":f"Only {len(all_dates)} dates in cache — need 50+. Retrain with full history."}
            return

        # Three-way temporal split
        split_tr = int(len(all_dates) * 0.6)
        split_va = int(len(all_dates) * 0.8)
        train_dates = set(all_dates[:split_tr])
        val_dates = set(all_dates[split_tr:split_va])
        test_dates = set(all_dates[split_va:])

        # v12: precompute training-period SPY volatility median (for high_vol/low_vol regime)
        train_dates_list = all_dates[:split_tr]
        spy_5day_atrs_train = []
        for i, date in enumerate(train_dates_list):
            if i < 5: continue
            # 5-day daily ranges from SPY daily bars ending at (date-1)
            recent_5 = []
            for past_date in all_dates[max(0,i-5):i]:
                dd = spy_daily_by_date.get(past_date)
                if dd: recent_5.append(dd["h"] - dd["l"])
            if len(recent_5) >= 3:
                spy_5day_atrs_train.append(sum(recent_5)/len(recent_5))
        spy_vol_median = sorted(spy_5day_atrs_train)[len(spy_5day_atrs_train)//2] if spy_5day_atrs_train else None
        log.info(f"Regime: SPY 5-day ATR median (train-period) = {spy_vol_median}")

        def compute_regime(date, scan_min, ticker_scan_info_local):
            """Return {'spy_dir': ..., 'vol': ..., 'breadth': ...} regime labels."""
            regimes = {}
            # 1. SPY direction at scan time
            spy_day = by_td.get("SPY", {}).get(date, [])
            spy_before = [b for b in spy_day if (bar_to_et_minutes(b) or -1) < scan_min]
            if len(spy_before) >= 2:
                spy_open = spy_before[0]["o"]
                spy_now = spy_before[-1]["c"]
                ch = (spy_now - spy_open) / spy_open if spy_open > 0 else 0
                regimes["spy_dir"] = "SPY_up" if ch > 0.002 else "SPY_down" if ch < -0.002 else "SPY_flat"
            else:
                regimes["spy_dir"] = None
            # 2. Volatility: SPY 5-day high-low average
            idx = all_dates.index(date) if date in all_dates else -1
            if idx >= 5 and spy_vol_median is not None:
                recent = []
                for past in all_dates[max(0,idx-5):idx]:
                    dd = spy_daily_by_date.get(past)
                    if dd: recent.append(dd["h"] - dd["l"])
                if len(recent) >= 3:
                    today_atr = sum(recent)/len(recent)
                    regimes["vol"] = "high_vol" if today_atr > spy_vol_median else "low_vol"
                else:
                    regimes["vol"] = None
            else:
                regimes["vol"] = None
            # 3. R2K-wide breadth at scan time
            if ticker_scan_info_local:
                total = len(ticker_scan_info_local)
                up = 0
                for tup in ticker_scan_info_local.values():
                    before, scan_price, _, _, _ = tup
                    if before and scan_price > before[0]["o"]:
                        up += 1
                frac = up / total if total > 0 else 0
                regimes["breadth"] = "breadth_up" if frac > 0.55 else "breadth_down" if frac < 0.45 else "breadth_flat"
            else:
                regimes["breadth"] = None
            return regimes

        log.info(f"Setup eval: {len(all_dates)} dates total, "
                 f"train {len(train_dates)}d / val {len(val_dates)}d / test {len(test_dates)}d")

        results = {
            "generatedAt": datetime.now(ET).isoformat(),
            "target_pct": target_pct * 100,
            "n_train_dates": len(train_dates),
            "n_val_dates": len(val_dates),
            "n_test_dates": len(test_dates),
            "setups": {s: SETUP_DESCRIPTIONS[s] for s in SETUP_NAMES},
            "hours": {},
            # v11: data provenance / integrity
            "cache_fingerprint": cache_fp,
            "cache_last_modified": cache_mtime_iso,
            "date_range": {"first": all_dates[0] if all_dates else None,
                           "last": all_dates[-1] if all_dates else None},
        }

        total_ops = len(SCAN_HOURS) * len(all_dates)
        processed = 0

        # v12: persist all events across hours for post-hoc analyses
        all_events_by_hour = {}  # scan_hour → {"events": {setup: [records]}, "base": [records]}

        # For each scan hour, iterate over dates, detect setups, record outcomes
        for scan_hour in SCAN_HOURS:
            scan_min = scan_hour * 60
            # Accumulators — lists of (date, ticker, hit_1pct, realized_pnl) per setup
            events = {s: [] for s in SETUP_NAMES}
            base_events = []  # (date, ticker, hit, pnl) over ALL stocks that had valid bars

            for date_idx, date in enumerate(all_dates):
                processed += 1
                if date_idx % 30 == 0:
                    pct = 10 + int(processed / total_ops * 85)
                    setup_eval_progress = {"phase":"scanning","pct":pct,
                        "message":f"{scan_hour}:00 — scanning {date}..."}

                # IWM bars up to scan time for this date (for rel_strength_iwm)
                iwm_day_bars = by_td.get("IWM", {}).get(date, [])
                iwm_before = [b for b in iwm_day_bars if (bar_to_et_minutes(b) or -1) < scan_min]

                # Compute sector breadth for this date + scan hour
                # breadth = fraction of stocks in each sector that are green at scan time
                sector_green = defaultdict(lambda: [0, 0])  # [green_count, total]
                ticker_scan_info = {}  # ticker → (bars_before, scan_price, prev_close, prior_daily, after)
                for ticker in TICKERS:
                    day_bars = by_td[ticker].get(date, [])
                    if len(day_bars) < 12: continue
                    before, after = [], []
                    for b in day_bars:
                        bm = bar_to_et_minutes(b)
                        if bm is None: continue
                        if bm < scan_min: before.append(b)
                        else: after.append(b)
                    if len(before) < 6 or len(after) < 2: continue
                    scan_price = after[0]["o"]
                    open_price = day_bars[0]["o"]
                    # Previous close + full prior daily series for ATR/pivot/YH
                    prior_daily = [d for d in daily_bars.get(ticker, []) if d["t"][:10] < date]
                    prev_close = prior_daily[-1]["c"] if prior_daily else None
                    # Green-ness at scan
                    is_green = scan_price > open_price
                    sect = SECTORS.get(ticker, "?")
                    sector_green[sect][1] += 1
                    if is_green: sector_green[sect][0] += 1
                    ticker_scan_info[ticker] = (before, scan_price, prev_close, prior_daily, after)

                # v12: compute regimes for this date+hour (uses ticker_scan_info for breadth)
                regimes = compute_regime(date, scan_min, ticker_scan_info)

                # Now for each ticker with valid scan info, detect setups and measure outcome
                for ticker, (before, scan_price, prev_close, prior_daily, after) in ticker_scan_info.items():
                    sect = SECTORS.get(ticker, "?")
                    sc = sector_green[sect]
                    breadth = sc[0]/sc[1] if sc[1] > 0 else 0

                    active = detect_setups(
                        before, scan_price, prev_close,
                        sector_breadth=breadth,
                        prior_daily=prior_daily,
                        iwm_bars=iwm_before,
                        scan_minute_et=scan_min,
                    )
                    hit, pnl = did_hit_target(scan_price, after[1:], target_pct=target_pct)
                    active_list = [s for s in SETUP_NAMES if active[s]]

                    # v18: compute lightweight event features for loser-filter analysis.
                    # Only compute if at least one setup fired (saves compute).
                    ev_features = None
                    if active_list:
                        # RSI (14)
                        if len(before) >= 15:
                            gains = [max(0, before[i]["c"]-before[i-1]["c"]) for i in range(len(before)-14, len(before))]
                            losses = [max(0, before[i-1]["c"]-before[i]["c"]) for i in range(len(before)-14, len(before))]
                            ag, al = sum(gains)/len(gains), sum(losses)/len(losses)
                            rsi_val = 100 - (100/(1+ag/al)) if al > 0 else 100
                        else:
                            rsi_val = None
                        # ATR% from prior_daily
                        atr_pct = compute_atr_pct(prior_daily) if prior_daily and len(prior_daily) >= 15 else None
                        # Relative volume: today's intraday so far vs expected prorated ADV
                        rel_vol = None
                        if prior_daily and len(prior_daily) >= 2:
                            adv5 = sum(d["v"] for d in prior_daily[-5:]) / min(5, len(prior_daily))
                            if adv5 > 0:
                                mins_elapsed = scan_min - (9*60+30)  # minutes since open
                                if mins_elapsed > 0:
                                    vol_so_far = sum(b["v"] for b in before)
                                    exp_so_far = adv5 * (mins_elapsed / 390.0)
                                    rel_vol = vol_so_far / exp_so_far if exp_so_far > 0 else None
                        # Day-of-week (Monday=0 .. Friday=4)
                        try:
                            dow = datetime.strptime(date, "%Y-%m-%d").weekday()
                        except Exception:
                            dow = None
                        ev_features = {
                            "rsi": round(rsi_val, 1) if rsi_val is not None else None,
                            "atr_pct": round(atr_pct*100, 2) if atr_pct is not None else None,
                            "rel_volume": round(rel_vol, 2) if rel_vol is not None else None,
                            "dow": dow,
                            "sector": sect,
                        }

                    # v12: enriched event record — regime + full active_setups list
                    # v18: also ev_features for loser-filter analysis
                    ev_record = {
                        "date": date, "ticker": ticker,
                        "hit": hit, "pnl": pnl,
                        "regimes": regimes,
                        "active_setups": active_list,
                        "features": ev_features,
                    }
                    base_events.append(ev_record)
                    for s in active_list:
                        events[s].append(ev_record)

            # Compute stats per fold per setup
            hour_result = {"base": {}, "setups": {}}

            # Base rates
            for fold_name, fold_dates in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
                fold_base = [e for e in base_events if e["date"] in fold_dates]
                n = len(fold_base)
                hits = sum(1 for e in fold_base if e["hit"])
                mean_pnl = (sum(e["pnl"] for e in fold_base) / n) if n > 0 else 0
                hour_result["base"][fold_name] = {
                    "n_observations": n,
                    "hit_rate": round(hits/n*100, 2) if n > 0 else None,
                    "mean_pnl": round(mean_pnl, 3),
                }

            for s in SETUP_NAMES:
                setup_result = {}
                for fold_name, fold_dates in [("train", train_dates), ("val", val_dates), ("test", test_dates)]:
                    fold_events = [e for e in events[s] if e["date"] in fold_dates]
                    n = len(fold_events)
                    hits = sum(1 for e in fold_events if e["hit"])
                    mean_pnl = (sum(e["pnl"] for e in fold_events) / n) if n > 0 else 0
                    # How many distinct trading days had at least one firing?
                    firing_days = len(set(e["date"] for e in fold_events))
                    total_days = len(fold_dates)
                    setup_result[fold_name] = {
                        "n_events": n,
                        "hit_rate": round(hits/n*100, 2) if n > 0 else None,
                        "mean_pnl": round(mean_pnl, 3),
                        "firing_days": firing_days,
                        "total_days": total_days,
                        "firing_day_frac": round(firing_days/total_days, 3) if total_days > 0 else 0,
                        "edge_vs_base": (round(hits/n*100 - hour_result["base"][fold_name]["hit_rate"], 2)
                                         if n > 0 and hour_result["base"][fold_name]["hit_rate"] is not None else None),
                    }
                hour_result["setups"][s] = setup_result

            results["hours"][str(scan_hour)] = hour_result
            # v12: persist events for combinations/regimes analyses
            all_events_by_hour[scan_hour] = {"events": events, "base": base_events}
            log.info(f"{scan_hour}:00 — base test hit_rate {hour_result['base']['test']['hit_rate']}%, "
                     + ", ".join(f"{s}: test n={hour_result['setups'][s]['test']['n_events']} "
                                 f"hit={hour_result['setups'][s]['test']['hit_rate']}%"
                                 for s in SETUP_NAMES))

        # v21.1: Write core results to disk NOW so that if any post-hoc analysis
        # crashes, the core per-hour data is preserved.
        def _save_results():
            try:
                SETUP_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
            except Exception as e:
                log.error(f"Failed to save results: {e}")
        _save_results()
        log.info(f"Saved core results to {SETUP_RESULTS_PATH}")

        # ═══════════════════════════════════════════════════════════════
        # v12: COMBINATIONS analysis (#2)
        # Within each scan hour, for every pair of active setups, compute
        # hit rate when BOTH fire on the same stock at the same time.
        # Only considered "interesting" if combined edge exceeds the better
        # of the two individual edges by ≥2 percentage points (indicates
        # synergy, not just redundant firing).
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"combinations","pct":95,
            "message":"Computing setup combinations..."}

        # Identify tier-qualified setups per hour (from classify_setup_hour on test fold)
        active_at_hour = {}  # scan_hour → list of setup names that passed tier gates
        for h in SCAN_HOURS:
            hr = results["hours"][str(h)]
            actives = []
            for s in SETUP_NAMES:
                t = hr["setups"][s].get("test", {})
                tier = classify_setup_hour(t)
                if tier is None: continue
                # Apply adjacency rule for strong
                if tier == "strong":
                    has_adj = False
                    for off in (-1, 1):
                        ni = SCAN_HOURS.index(h) + off if h in SCAN_HOURS else -1
                        if 0 <= ni < len(SCAN_HOURS):
                            adj_h = SCAN_HOURS[ni]
                            adj_edge = (results["hours"][str(adj_h)]["setups"][s]
                                        .get("test", {}).get("edge_vs_base"))
                            if adj_edge is not None and adj_edge > 0:
                                has_adj = True; break
                    if not has_adj:
                        # Demote to moderate if qualifies
                        if t.get("edge_vs_base", 0) < 3 or t.get("n_events", 0) < 25:
                            continue
                actives.append(s)
            active_at_hour[h] = actives

        combinations_result = {"by_hour": {}}
        for h in SCAN_HOURS:
            actives = active_at_hour[h]
            if len(actives) < 2:
                combinations_result["by_hour"][str(h)] = {"pairs": [], "n_pairs_tested": 0}
                continue

            hr_events = all_events_by_hour[h]
            base_test = [e for e in hr_events["base"] if e["date"] in test_dates]
            base_hits = sum(1 for e in base_test if e["hit"])
            base_test_hr = (base_hits / len(base_test) * 100) if base_test else 0

            individual_edges = {}
            for s in actives:
                s_test = [e for e in hr_events["events"][s] if e["date"] in test_dates]
                s_hits = sum(1 for e in s_test if e["hit"])
                s_hr = (s_hits / len(s_test) * 100) if s_test else 0
                individual_edges[s] = {"hr": s_hr, "edge": s_hr - base_test_hr, "n": len(s_test)}

            pairs = []
            from itertools import combinations as iter_combinations
            for a, b in iter_combinations(actives, 2):
                # Events where BOTH setups fire on same (date, ticker)
                both = [e for e in hr_events["base"] if e["date"] in test_dates
                        and a in e.get("active_setups", []) and b in e.get("active_setups", [])]
                n = len(both)
                if n < 20:  # skip low-sample pairs
                    continue
                hits = sum(1 for e in both if e["hit"])
                hr_pct = hits / n * 100
                edge_vs_base = hr_pct - base_test_hr
                max_individual_edge = max(individual_edges[a]["edge"], individual_edges[b]["edge"])
                lift = edge_vs_base - max_individual_edge
                pairs.append({
                    "setup_a": a, "setup_b": b,
                    "n": n,
                    "hit_rate": round(hr_pct, 2),
                    "edge_vs_base": round(edge_vs_base, 2),
                    "individual_a_edge": round(individual_edges[a]["edge"], 2),
                    "individual_b_edge": round(individual_edges[b]["edge"], 2),
                    "individual_a_n": individual_edges[a]["n"],
                    "individual_b_n": individual_edges[b]["n"],
                    "lift_over_best_individual": round(lift, 2),
                    "is_synergistic": lift >= 2.0,  # threshold for "combination adds edge"
                })
            # Sort by lift descending (most interesting first)
            pairs.sort(key=lambda p: -p["lift_over_best_individual"])
            combinations_result["by_hour"][str(h)] = {
                "pairs": pairs,
                "n_pairs_tested": len(pairs),
                "base_test_hit_rate": round(base_test_hr, 2),
            }

        results["combinations"] = combinations_result
        log.info("Combinations analysis complete")

        # ═══════════════════════════════════════════════════════════════
        # v12: REGIMES analysis (#3)
        # For each surviving (setup, hour), compute hit rate within each
        # regime sub-fold (SPY direction, volatility, breadth). Flag cases
        # where a setup is strong in one regime but absent/negative in another.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"regimes","pct":97,
            "message":"Computing regime splits..."}

        regime_dimensions = ["spy_dir", "vol", "breadth"]
        regimes_result = {"by_hour": {}}
        for h in SCAN_HOURS:
            actives = active_at_hour[h]
            hr_events = all_events_by_hour[h]
            base_test = [e for e in hr_events["base"] if e["date"] in test_dates]

            per_setup = {}
            for s in actives:
                s_test = [e for e in hr_events["events"][s] if e["date"] in test_dates]
                per_setup[s] = {}
                for dim in regime_dimensions:
                    # Collect all unique regime values in this dim from test base events
                    regime_values = sorted({e["regimes"].get(dim) for e in base_test
                                            if e["regimes"].get(dim) is not None})
                    per_regime = {}
                    for rv in regime_values:
                        base_sub = [e for e in base_test if e["regimes"].get(dim) == rv]
                        set_sub = [e for e in s_test if e["regimes"].get(dim) == rv]
                        n_base = len(base_sub)
                        n_set = len(set_sub)
                        if n_base == 0: continue
                        base_hr = sum(1 for e in base_sub if e["hit"]) / n_base * 100
                        set_hr = (sum(1 for e in set_sub if e["hit"]) / n_set * 100) if n_set > 0 else None
                        per_regime[rv] = {
                            "n_events": n_set,
                            "n_base_observations": n_base,
                            "hit_rate": round(set_hr, 2) if set_hr is not None else None,
                            "base_hit_rate": round(base_hr, 2),
                            "edge_vs_base": round(set_hr - base_hr, 2) if set_hr is not None else None,
                        }
                    per_setup[s][dim] = per_regime

                # Check for regime-conditional behavior: big delta between regimes
                for dim, per_regime in per_setup[s].items():
                    # Only flag if 2+ regimes have n≥15
                    qual = [rv for rv, d in per_regime.items()
                            if d.get("n_events", 0) >= 15 and d.get("edge_vs_base") is not None]
                    if len(qual) >= 2:
                        edges = [per_regime[rv]["edge_vs_base"] for rv in qual]
                        spread = max(edges) - min(edges)
                        if spread >= 5.0:
                            per_setup[s][dim]["_flag_conditional"] = True
                            per_setup[s][dim]["_flag_spread"] = round(spread, 2)

            regimes_result["by_hour"][str(h)] = per_setup

        results["regimes"] = regimes_result
        log.info("Regime analysis complete")

        # ═══════════════════════════════════════════════════════════════
        # v14: FOLD-SWAP VALIDATION (priority #1)
        # Classify each surviving (setup, hour) by how robust its edge is
        # across different test fold positions. The original three-way split
        # uses the most recent 57 days as test; fold-swap rotates 5 different
        # non-overlapping 57-day windows as "test", recomputing hit rate and
        # edge vs base rate in each.
        # A setup that holds across 4-5 folds is robust; one that only works
        # in the original most-recent test is fragile/overfit.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"fold_swap","pct":98,
            "message":"Fold-swap validation..."}

        # Define equal-sized test windows sized to the dataset.
        # Original split was 168/56/57 — use 57 as fold size (~3 months each).
        # v17: auto-extend folds to use all available data, so extended history
        # (2+ years) gets rotated through as additional test windows.
        n = len(all_dates)
        fold_size = 57
        # Windows are anchored to the end of data for reproducibility:
        # F_A = last 57, F_B = 57 before that, etc. Only folds with
        # the full fold_size are included (partial folds at the start
        # of data are dropped).
        fold_windows = {}
        # Generate names F_A, F_B, ... up through F_Z if needed (26 max; ~4 years)
        fold_names = [f"F_{chr(ord('A')+i)}" for i in range(26)]
        cursor = n
        for name in fold_names:
            start = cursor - fold_size
            if start < 0:
                # Not enough data for a full-size fold; stop
                break
            fold_windows[name] = set(all_dates[start:cursor])
            cursor = start
            if cursor <= 0:
                break

        log.info(f"Fold windows ({len(fold_windows)} total, each {fold_size}d): "
                 f"{[(k, len(v)) for k,v in fold_windows.items()]}")

        fold_swap_result = {
            "fold_windows": {k: {"n_days": len(v),
                                 "first_date": min(v) if v else None,
                                 "last_date": max(v) if v else None}
                             for k, v in fold_windows.items()},
            "by_hour": {}
        }

        for scan_hour in SCAN_HOURS:
            hr_events = all_events_by_hour[scan_hour]
            hour_folds = {}
            # Per-setup per-fold edge/n
            for s in SETUP_NAMES:
                setup_folds = {}
                for fold_name, fold_set in fold_windows.items():
                    fold_base = [e for e in hr_events["base"] if e["date"] in fold_set]
                    fold_setup = [e for e in hr_events["events"][s] if e["date"] in fold_set]
                    n_base = len(fold_base)
                    n_setup = len(fold_setup)
                    if n_base < 100:  # too small
                        setup_folds[fold_name] = {"edge": None, "hit_rate": None,
                                                  "base_hit_rate": None,
                                                  "n_events": n_setup, "n_base": n_base}
                        continue
                    base_hr = sum(1 for e in fold_base if e["hit"]) / n_base * 100
                    if n_setup == 0:
                        setup_folds[fold_name] = {"edge": None, "hit_rate": None,
                                                  "base_hit_rate": round(base_hr, 2),
                                                  "n_events": 0, "n_base": n_base}
                        continue
                    set_hr = sum(1 for e in fold_setup if e["hit"]) / n_setup * 100
                    setup_folds[fold_name] = {
                        "edge": round(set_hr - base_hr, 2),
                        "hit_rate": round(set_hr, 2),
                        "base_hit_rate": round(base_hr, 2),
                        "n_events": n_setup,
                        "n_base": n_base,
                    }

                # Classify robustness across the 5 folds
                # Only consider folds where we have edge computed AND sample size is reasonable
                valid_folds = [(fn, fd) for fn, fd in setup_folds.items()
                               if fd["edge"] is not None and fd["n_events"] >= 10]
                n_valid = len(valid_folds)
                n_positive = sum(1 for _, fd in valid_folds if fd["edge"] > 0)
                edges = [fd["edge"] for _, fd in valid_folds]
                min_edge = min(edges) if edges else None
                mean_edge = sum(edges) / len(edges) if edges else None

                # v17: classification now scales with number of folds.
                # ROBUST: ≥80% positive folds AND mean ≥3% AND min ≥-2%
                # CONSISTENT: ≥60% positive folds
                # FRAGILE: <60% positive folds
                # Still requires n_valid ≥4 for full classification.
                robustness = None
                positive_frac = (n_positive / n_valid) if n_valid > 0 else 0
                if n_valid >= 4:
                    if positive_frac >= 0.80 and mean_edge is not None and mean_edge >= 3 and min_edge is not None and min_edge >= -2:
                        robustness = "ROBUST"
                    elif positive_frac >= 0.60:
                        robustness = "CONSISTENT"
                    else:
                        robustness = "FRAGILE"
                elif n_valid >= 2:
                    robustness = "LIMITED_DATA"
                else:
                    robustness = "INSUFFICIENT"

                hour_folds[s] = {
                    "folds": setup_folds,
                    "n_valid_folds": n_valid,
                    "n_positive_folds": n_positive,
                    "positive_frac": round(positive_frac, 3),
                    "min_edge": round(min_edge, 2) if min_edge is not None else None,
                    "max_edge": round(max(edges), 2) if edges else None,
                    "mean_edge": round(mean_edge, 2) if mean_edge is not None else None,
                    "robustness": robustness,
                }

            fold_swap_result["by_hour"][str(scan_hour)] = hour_folds

        results["fold_swap"] = fold_swap_result
        log.info("Fold-swap validation complete")

        # Identify which (setup, hour) combos are "original survivors" to focus
        # subsequent analyses on. Uses same tier gate as main eval (edge ≥3, n ≥25 on test fold).
        original_survivors = []  # list of (setup, hour) tuples
        for h in SCAN_HOURS:
            hr_data = results["hours"].get(str(h), {})
            for s in SETUP_NAMES:
                tst = hr_data.get("setups", {}).get(s, {}).get("test", {})
                edge = tst.get("edge_vs_base")
                n = tst.get("n_events", 0)
                if edge is not None and edge >= 3 and n >= 25:
                    original_survivors.append((s, h))
        log.info(f"Original survivors to analyze: {len(original_survivors)}")

        # ═══════════════════════════════════════════════════════════════
        # v16-A2: EX-F_A VALIDATION
        # Recompute hit rate / edge for each original survivor on the UNION
        # of F_B, F_C, F_D, F_E (all folds except the original test fold).
        # If a setup holds up here with similar edge, that's independent
        # confirmation of fold-swap result.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"ex_fa","pct":98.2,
            "message":"Ex-F_A cross-validation..."}

        # Fold F_A is already defined above (most recent 57 days).
        # The "ex-F_A" union = all other folds combined.
        ex_fa_dates = set()
        for fname in ["F_B", "F_C", "F_D", "F_E"]:
            ex_fa_dates |= fold_windows.get(fname, set())

        ex_fa_result = {
            "n_ex_fa_dates": len(ex_fa_dates),
            "date_range": {
                "first": min(ex_fa_dates) if ex_fa_dates else None,
                "last": max(ex_fa_dates) if ex_fa_dates else None,
            },
            "by_hour": {},
        }

        for h in SCAN_HOURS:
            hr_events = all_events_by_hour[h]
            # Base rate across ex-F_A
            ex_base = [e for e in hr_events["base"] if e["date"] in ex_fa_dates]
            n_ex_base = len(ex_base)
            ex_base_hr = (sum(1 for e in ex_base if e["hit"]) / n_ex_base * 100) if n_ex_base > 0 else None
            per_setup = {}
            for (s, hh) in original_survivors:
                if hh != h: continue
                ex_set = [e for e in hr_events["events"][s] if e["date"] in ex_fa_dates]
                n_ex = len(ex_set)
                if n_ex == 0:
                    per_setup[s] = {
                        "ex_fa_n": 0, "ex_fa_hit_rate": None, "ex_fa_edge": None,
                        "fa_edge": results["hours"][str(h)]["setups"][s]["test"].get("edge_vs_base"),
                        "holds_up": False,
                    }
                    continue
                ex_hr = sum(1 for e in ex_set if e["hit"]) / n_ex * 100
                ex_edge = ex_hr - ex_base_hr if ex_base_hr is not None else None
                fa_edge = results["hours"][str(h)]["setups"][s]["test"].get("edge_vs_base")
                holds_up = ex_edge is not None and ex_edge >= 3
                per_setup[s] = {
                    "ex_fa_n": n_ex,
                    "ex_fa_hit_rate": round(ex_hr, 2),
                    "ex_fa_base_hit_rate": round(ex_base_hr, 2) if ex_base_hr is not None else None,
                    "ex_fa_edge": round(ex_edge, 2) if ex_edge is not None else None,
                    "fa_edge": fa_edge,
                    "edge_delta": round(ex_edge - fa_edge, 2) if (ex_edge is not None and fa_edge is not None) else None,
                    "holds_up": holds_up,
                }
            ex_fa_result["by_hour"][str(h)] = {
                "base_hit_rate": round(ex_base_hr, 2) if ex_base_hr is not None else None,
                "n_base": n_ex_base,
                "setups": per_setup,
            }

        results["ex_fa"] = ex_fa_result
        log.info("Ex-F_A validation complete")

        # ═══════════════════════════════════════════════════════════════
        # v16-B1: SETUP OVERLAP (JACCARD)
        # For each scan hour, for each pair of original survivors:
        #   A = set of (date, ticker) where setup A fired
        #   B = set of (date, ticker) where setup B fired
        #   Jaccard = |A ∩ B| / |A ∪ B|
        # High Jaccard = redundant. Low Jaccard = independent signals.
        # Uses test fold only (to match how we think about the signals).
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"overlap","pct":98.5,"message":"Setup overlap analysis..."}

        overlap_result = {"by_hour": {}}
        from itertools import combinations as _iter_comb
        for h in SCAN_HOURS:
            hr_events = all_events_by_hour[h]
            survivors_here = [s for (s, hh) in original_survivors if hh == h]
            if len(survivors_here) < 2:
                overlap_result["by_hour"][str(h)] = {"pairs": [], "n_survivors": len(survivors_here)}
                continue

            # Build firing sets for each survivor (over all test dates)
            sets_by_setup = {}
            for s in survivors_here:
                ev_s = [e for e in hr_events["events"][s] if e["date"] in test_dates]
                sets_by_setup[s] = {(e["date"], e["ticker"]) for e in ev_s}

            pairs = []
            for a, b in _iter_comb(survivors_here, 2):
                A, B = sets_by_setup[a], sets_by_setup[b]
                if not A and not B: continue
                inter = len(A & B)
                union = len(A | B)
                jacc = inter / union if union > 0 else 0
                classification = ("redundant" if jacc >= 0.5
                                  else "correlated" if jacc >= 0.2
                                  else "independent")
                pairs.append({
                    "setup_a": a, "setup_b": b,
                    "n_a_only": len(A - B),
                    "n_b_only": len(B - A),
                    "n_both": inter,
                    "n_union": union,
                    "jaccard": round(jacc, 3),
                    "classification": classification,
                })
            pairs.sort(key=lambda p: -p["jaccard"])
            overlap_result["by_hour"][str(h)] = {
                "pairs": pairs,
                "n_survivors": len(survivors_here),
            }

        results["overlap"] = overlap_result
        log.info("Setup overlap analysis complete")

        # ═══════════════════════════════════════════════════════════════
        # v16-B2: MULTI-SETUP STACKING
        # For each (date, ticker, scan_hour), count how many original-survivor
        # setups fired simultaneously. Then compute hit rate by stack count.
        # Stacks with 3+ setups should have meaningfully higher hit rates if
        # stacking adds conviction.
        # Uses test fold only.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"stacking","pct":98.8,"message":"Multi-setup stacking analysis..."}

        stacking_result = {"by_hour": {}}
        for h in SCAN_HOURS:
            hr_events = all_events_by_hour[h]
            survivors_here = set(s for (s, hh) in original_survivors if hh == h)

            # Base events with survivor-stack counts
            test_base = [e for e in hr_events["base"] if e["date"] in test_dates]
            if not test_base:
                stacking_result["by_hour"][str(h)] = {"by_stack_count": []}
                continue

            base_hits = sum(1 for e in test_base if e["hit"])
            base_hr = base_hits / len(test_base) * 100

            # Count events by stack size (number of SURVIVOR setups firing on this stock)
            by_count = defaultdict(lambda: {"n": 0, "hits": 0, "pnl_sum": 0})
            for e in test_base:
                stack = sum(1 for s in e.get("active_setups", []) if s in survivors_here)
                by_count[stack]["n"] += 1
                if e["hit"]: by_count[stack]["hits"] += 1
                by_count[stack]["pnl_sum"] += e.get("pnl", 0) or 0

            # Aggregate into buckets: 0, 1, 2, 3, 4+
            buckets = []
            for count in sorted(by_count.keys()):
                d = by_count[count]
                n = d["n"]
                hits = d["hits"]
                hit_rate = (hits / n * 100) if n > 0 else None
                edge = (hit_rate - base_hr) if hit_rate is not None else None
                mean_pnl = (d["pnl_sum"] / n) if n > 0 else 0
                buckets.append({
                    "stack_count": count,
                    "n": n,
                    "hit_rate": round(hit_rate, 2) if hit_rate is not None else None,
                    "edge_vs_base": round(edge, 2) if edge is not None else None,
                    "mean_pnl": round(mean_pnl, 3),
                })

            # Is the trend monotonic? (edge increases with stack count)
            edges_by_count = [b["edge_vs_base"] for b in buckets
                              if b["stack_count"] >= 1 and b["edge_vs_base"] is not None]
            monotonic_increasing = len(edges_by_count) >= 2 and all(
                edges_by_count[i] <= edges_by_count[i+1] for i in range(len(edges_by_count)-1)
            )

            stacking_result["by_hour"][str(h)] = {
                "base_hit_rate": round(base_hr, 2),
                "n_base": len(test_base),
                "by_stack_count": buckets,
                "monotonic_increasing": monotonic_increasing,
                "n_survivors": len(survivors_here),
            }

        results["stacking"] = stacking_result
        log.info("Multi-setup stacking analysis complete")

        # ═══════════════════════════════════════════════════════════════
        # v16-A1: SECTOR BREAKDOWN
        # For each original survivor, hit rate per sector (test fold).
        # Flag sectors where edge is ≥5% spread above/below the setup's overall
        # test edge, with n ≥15 per sector.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"sector","pct":99,"message":"Sector breakdown..."}

        sector_result = {"by_hour": {}}
        for h in SCAN_HOURS:
            hr_events = all_events_by_hour[h]
            test_base = [e for e in hr_events["base"] if e["date"] in test_dates]
            # Per-sector base rate
            base_by_sector = defaultdict(lambda: [0, 0])  # [hits, n]
            for e in test_base:
                sec = SECTORS.get(e["ticker"], "?")
                base_by_sector[sec][1] += 1
                if e["hit"]: base_by_sector[sec][0] += 1

            per_setup = {}
            for (s, hh) in original_survivors:
                if hh != h: continue
                test_set = [e for e in hr_events["events"][s] if e["date"] in test_dates]
                n_set = len(test_set)
                overall_hr = (sum(1 for e in test_set if e["hit"]) / n_set * 100) if n_set > 0 else None

                per_sector = {}
                flagged_sectors = []
                for sec, (base_hits, base_n) in base_by_sector.items():
                    sec_set = [e for e in test_set if SECTORS.get(e["ticker"], "?") == sec]
                    n_sec = len(sec_set)
                    if n_sec < 15: continue  # require min sample
                    sec_hr = sum(1 for e in sec_set if e["hit"]) / n_sec * 100
                    base_sec_hr = base_hits / base_n * 100 if base_n > 0 else 0
                    sec_edge = sec_hr - base_sec_hr
                    overall_edge = overall_hr - (sum(1 for e in test_base if e["hit"]) / len(test_base) * 100) if test_base else 0
                    spread = sec_edge - overall_edge  # how much this sector deviates from setup's overall edge
                    per_sector[sec] = {
                        "n": n_sec,
                        "hit_rate": round(sec_hr, 2),
                        "base_sector_hit_rate": round(base_sec_hr, 2),
                        "edge_vs_sector_base": round(sec_edge, 2),
                        "deviation_from_overall": round(spread, 2),
                    }
                    if abs(spread) >= 5:
                        flagged_sectors.append({
                            "sector": sec,
                            "edge": round(sec_edge, 2),
                            "deviation": round(spread, 2),
                            "direction": "stronger" if spread > 0 else "weaker",
                        })

                per_setup[s] = {
                    "overall_test_n": n_set,
                    "overall_test_hit_rate": round(overall_hr, 2) if overall_hr is not None else None,
                    "by_sector": per_sector,
                    "flagged_sectors": flagged_sectors,
                }

            sector_result["by_hour"][str(h)] = per_setup

        results["sector_breakdown"] = sector_result
        log.info("Sector breakdown analysis complete")

        # ═══════════════════════════════════════════════════════════════
        # v18: LOSER-FILTER DISCOVERY/VALIDATION
        # For each ROBUST/CONSISTENT survivor, split firings by 5 feature
        # dimensions (sector, rsi_bucket, atr_bucket, rel_vol_bucket, dow).
        # Discovery = oldest 4 folds (F_F, F_G, F_H, F_I if available).
        # Validation = newest 5 folds (F_A, F_B, F_C, F_D, F_E).
        # Qualify a filter ONLY if:
        #   - Discovery: value's win rate ≥10% relatively lower than setup's
        #     overall win rate (i.e., 0.9× or less of overall)
        #   - Validation: same directional effect (still ≤ 0.9× of overall)
        #   - Affects ≥10% of firings in discovery AND validation
        # Strict methodology to prevent data mining.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"loser_filter","pct":99.3,
            "message":"Loser-filter discovery/validation..."}

        # Identify discovery and validation fold sets
        disc_fold_names = ["F_F", "F_G", "F_H", "F_I"]  # older, used for discovery
        val_fold_names = ["F_A", "F_B", "F_C", "F_D", "F_E"]  # newer, used for validation
        disc_dates = set()
        val_dates_loser = set()
        for name in disc_fold_names:
            disc_dates |= fold_windows.get(name, set())
        for name in val_fold_names:
            val_dates_loser |= fold_windows.get(name, set())
        log.info(f"Loser-filter: discovery={len(disc_dates)}d, validation={len(val_dates_loser)}d")

        # Feature bucket helpers
        def rsi_bucket(v):
            if v is None: return None
            if v < 30: return "oversold"
            if v > 70: return "overbought"
            return "neutral"
        def dow_name(d):
            if d is None: return None
            return ["Mon","Tue","Wed","Thu","Fri"][d] if 0 <= d <= 4 else None

        # Universal buckets for atr_pct and rel_volume: compute tertiles
        # across ALL events (base_events) at each hour — consistent
        # per-hour buckets avoid comparing different hours' volatility.
        def compute_tertile_boundaries(values):
            sv = sorted(v for v in values if v is not None)
            if len(sv) < 30: return None, None
            lo = sv[len(sv)//3]
            hi = sv[2*len(sv)//3]
            return lo, hi
        def tertile_bucket(v, lo, hi):
            if v is None or lo is None or hi is None: return None
            if v <= lo: return "low"
            if v >= hi: return "high"
            return "mid"

        # Determine which setups to analyze: include original survivors PLUS anything newly
        # ROBUST or CONSISTENT via fold-swap. This catches setups that only surfaced after
        # extended history (like end_of_day_momentum @ 15:00 on 9-fold).
        loser_filter_setups = set()
        for (s, hh) in original_survivors:
            loser_filter_setups.add((s, hh))
        for h in SCAN_HOURS:
            fs_hour = fold_swap_result["by_hour"].get(str(h), {})
            for s, sd in fs_hour.items():
                if sd.get("robustness") in ("ROBUST", "CONSISTENT"):
                    loser_filter_setups.add((s, h))

        loser_filter_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "disc_days": len(disc_dates),
            "val_days": len(val_dates_loser),
            "by_hour": {},
        }

        for h in SCAN_HOURS:
            hr_events = all_events_by_hour[h]
            hour_base = hr_events["base"]
            # Compute per-hour tertile boundaries for atr_pct and rel_volume
            atrs = [e["features"]["atr_pct"] for e in hour_base if e.get("features")]
            rvs = [e["features"]["rel_volume"] for e in hour_base if e.get("features")]
            atr_lo, atr_hi = compute_tertile_boundaries(atrs)
            rv_lo, rv_hi = compute_tertile_boundaries(rvs)

            hour_filter_result = {
                "atr_tertile_boundaries": [atr_lo, atr_hi],
                "rel_vol_tertile_boundaries": [rv_lo, rv_hi],
                "setups": {},
            }

            for (s, hh) in loser_filter_setups:
                if hh != h: continue
                all_setup_events = hr_events["events"].get(s, [])
                # Partition into discovery vs validation
                disc_ev = [e for e in all_setup_events if e["date"] in disc_dates]
                val_ev = [e for e in all_setup_events if e["date"] in val_dates_loser]
                if len(disc_ev) < 30 or len(val_ev) < 30:
                    hour_filter_result["setups"][s] = {
                        "status": "insufficient_data",
                        "n_discovery": len(disc_ev),
                        "n_validation": len(val_ev),
                    }
                    continue

                # Overall setup win rate in each fold
                disc_hr = sum(1 for e in disc_ev if e["hit"]) / len(disc_ev) * 100
                val_hr = sum(1 for e in val_ev if e["hit"]) / len(val_ev) * 100

                # Test each feature dimension
                qualified_filters = []
                all_tests = []  # for reference/debug

                def bucket_events(events_list, key_fn):
                    buckets = defaultdict(list)
                    for e in events_list:
                        if not e.get("features"): continue
                        k = key_fn(e["features"])
                        if k is None: continue
                        buckets[k].append(e)
                    return buckets

                # Define 5 feature dimensions
                dim_extractors = {
                    "sector": lambda f: f.get("sector"),
                    "rsi": lambda f: rsi_bucket(f.get("rsi")),
                    "atr": lambda f: tertile_bucket(f.get("atr_pct"), atr_lo, atr_hi),
                    "rel_vol": lambda f: tertile_bucket(f.get("rel_volume"), rv_lo, rv_hi),
                    "dow": lambda f: dow_name(f.get("dow")),
                }

                for dim_name, extractor in dim_extractors.items():
                    disc_buckets = bucket_events(disc_ev, extractor)
                    val_buckets = bucket_events(val_ev, extractor)
                    # Union of bucket values observed
                    bucket_keys = set(disc_buckets.keys()) | set(val_buckets.keys())
                    for bk in bucket_keys:
                        disc_bucket_ev = disc_buckets.get(bk, [])
                        val_bucket_ev = val_buckets.get(bk, [])
                        n_disc = len(disc_bucket_ev)
                        n_val = len(val_bucket_ev)
                        # Min 10% of firings AND min n=15 per bucket in each fold
                        if n_disc < max(15, len(disc_ev)*0.10): continue
                        if n_val < max(15, len(val_ev)*0.10): continue

                        disc_bucket_hr = sum(1 for e in disc_bucket_ev if e["hit"]) / n_disc * 100
                        val_bucket_hr = sum(1 for e in val_bucket_ev if e["hit"]) / n_val * 100

                        # Filter qualifies if in BOTH folds, bucket hit rate is ≥10% relatively lower than setup overall
                        disc_relative = disc_bucket_hr / disc_hr if disc_hr > 0 else 1
                        val_relative = val_bucket_hr / val_hr if val_hr > 0 else 1
                        qualifies = disc_relative <= 0.90 and val_relative <= 0.90

                        test_rec = {
                            "dim": dim_name,
                            "value": str(bk),
                            "n_discovery": n_disc,
                            "n_validation": n_val,
                            "discovery_hit_rate": round(disc_bucket_hr, 2),
                            "validation_hit_rate": round(val_bucket_hr, 2),
                            "setup_discovery_hit_rate": round(disc_hr, 2),
                            "setup_validation_hit_rate": round(val_hr, 2),
                            "discovery_relative": round(disc_relative, 3),
                            "validation_relative": round(val_relative, 3),
                            "qualifies": qualifies,
                        }
                        all_tests.append(test_rec)
                        if qualifies:
                            qualified_filters.append(test_rec)

                hour_filter_result["setups"][s] = {
                    "status": "tested",
                    "n_discovery": len(disc_ev),
                    "n_validation": len(val_ev),
                    "discovery_hit_rate": round(disc_hr, 2),
                    "validation_hit_rate": round(val_hr, 2),
                    "qualified_filters": qualified_filters,
                    "n_tests_run": len(all_tests),
                }

            loser_filter_result["by_hour"][str(h)] = hour_filter_result

        results["loser_filter"] = loser_filter_result
        log.info(f"Loser-filter analysis complete: "
                 f"{sum(len(h.get('setups',{})) for h in loser_filter_result['by_hour'].values())} setups evaluated")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v19: ROBUST LOSER PROFILES
        # For each ROBUST setup from fold-swap, build a complete picture
        # comparing losers (hit=False) to winners (hit=True) across all
        # event features. Unlike loser_filter (which only surfaces qualifying
        # filters), this shows the FULL profile — every feature value,
        # every continuous stat — so you can see what losers look like
        # even when no single filter passes strict criteria.
        # Discovery/validation split preserved for out-of-sample checks.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"robust_profiles","pct":99.7,
            "message":"Robust loser profiles..."}

        robust_setups_by_hour = defaultdict(list)  # hour → list of setup names
        for h in SCAN_HOURS:
            fs_hour = fold_swap_result["by_hour"].get(str(h), {})
            for s, sd in fs_hour.items():
                if sd.get("robustness") == "ROBUST":
                    robust_setups_by_hour[h].append(s)

        def bucket_compare(all_evs, disc_evs, val_evs, extractor, label):
            """
            For a feature dimension, compute winner/loser counts per bucket value
            across all events + per fold. Return list of records.
            """
            def bucket_stats(evs):
                # Group by bucket value; compute hits and total per value
                buckets = defaultdict(lambda: {"n": 0, "hits": 0, "losers": 0})
                for e in evs:
                    if not e.get("features"): continue
                    v = extractor(e["features"])
                    if v is None: continue
                    buckets[v]["n"] += 1
                    if e["hit"]: buckets[v]["hits"] += 1
                    else: buckets[v]["losers"] += 1
                return buckets

            all_b = bucket_stats(all_evs)
            disc_b = bucket_stats(disc_evs)
            val_b = bucket_stats(val_evs)

            total_winners_all = sum(b["hits"] for b in all_b.values())
            total_losers_all = sum(b["losers"] for b in all_b.values())
            total_winners_disc = sum(b["hits"] for b in disc_b.values())
            total_losers_disc = sum(b["losers"] for b in disc_b.values())
            total_winners_val = sum(b["hits"] for b in val_b.values())
            total_losers_val = sum(b["losers"] for b in val_b.values())

            vals = sorted(all_b.keys(), key=lambda x: str(x))
            records = []
            for v in vals:
                ba, bd, bv = all_b.get(v, {"n":0,"hits":0,"losers":0}), disc_b.get(v, {"n":0,"hits":0,"losers":0}), val_b.get(v, {"n":0,"hits":0,"losers":0})
                # Proportion of winners (and losers) that had this value
                pw_disc = (bd["hits"]/total_winners_disc) if total_winners_disc > 0 else 0
                pl_disc = (bd["losers"]/total_losers_disc) if total_losers_disc > 0 else 0
                pw_val = (bv["hits"]/total_winners_val) if total_winners_val > 0 else 0
                pl_val = (bv["losers"]/total_losers_val) if total_losers_val > 0 else 0
                # Diff = losers over-represented vs winners by this much (percentage points)
                diff_disc = (pl_disc - pw_disc) * 100
                diff_val = (pl_val - pw_val) * 100
                # Flag: both folds show |diff| ≥ 5pp in same direction, ≥20 losers in each
                flagged = (
                    abs(diff_disc) >= 5 and abs(diff_val) >= 5
                    and (diff_disc * diff_val) > 0
                    and bd["losers"] >= 20 and bv["losers"] >= 20
                )
                records.append({
                    "value": str(v),
                    "total_n": ba["n"],
                    "disc_n": bd["n"],
                    "val_n": bv["n"],
                    "disc_winners": bd["hits"],
                    "disc_losers": bd["losers"],
                    "val_winners": bv["hits"],
                    "val_losers": bv["losers"],
                    "disc_winner_pct": round(pw_disc*100, 2),
                    "disc_loser_pct": round(pl_disc*100, 2),
                    "val_winner_pct": round(pw_val*100, 2),
                    "val_loser_pct": round(pl_val*100, 2),
                    "disc_diff_pp": round(diff_disc, 2),
                    "val_diff_pp": round(diff_val, 2),
                    "flagged_over_loser": flagged and diff_disc > 0,
                    "flagged_over_winner": flagged and diff_disc < 0,
                })
            return {
                "dim": label,
                "buckets": records,
                "total_winners_disc": total_winners_disc,
                "total_losers_disc": total_losers_disc,
                "total_winners_val": total_winners_val,
                "total_losers_val": total_losers_val,
            }

        def continuous_compare(all_evs, disc_evs, val_evs, extractor, label):
            """
            For a continuous feature, compute mean+std for winners vs losers
            in each fold, and Cohen's d effect size.
            """
            def stats(evs, hit_filter):
                vals = [extractor(e["features"]) for e in evs
                        if e.get("features") and extractor(e["features"]) is not None
                        and e["hit"] == hit_filter]
                n = len(vals)
                if n == 0: return {"n": 0, "mean": None, "std": None}
                m = sum(vals) / n
                var = sum((v - m) ** 2 for v in vals) / max(n-1, 1)
                s = var ** 0.5
                return {"n": n, "mean": m, "std": s}

            def cohens_d(wstats, lstats):
                if not (wstats["n"] and lstats["n"]) or wstats["std"] is None or lstats["std"] is None:
                    return None
                # pooled std
                n1, n2 = wstats["n"], lstats["n"]
                s1, s2 = wstats["std"], lstats["std"]
                if n1 + n2 - 2 <= 0: return None
                sp2 = ((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1 + n2 - 2)
                if sp2 <= 0: return None
                sp = sp2 ** 0.5
                return (lstats["mean"] - wstats["mean"]) / sp  # >0 = losers higher

            disc_w = stats(disc_evs, True)
            disc_l = stats(disc_evs, False)
            val_w = stats(val_evs, True)
            val_l = stats(val_evs, False)
            d_disc = cohens_d(disc_w, disc_l)
            d_val = cohens_d(val_w, val_l)

            # Flagged if |Cohen's d| ≥0.2 in same direction in both folds
            flagged = (d_disc is not None and d_val is not None
                       and abs(d_disc) >= 0.2 and abs(d_val) >= 0.2
                       and (d_disc * d_val) > 0)

            def r(x, p=2):
                if x is None: return None
                return round(x, p)
            return {
                "dim": label,
                "disc_winner": {"n": disc_w["n"], "mean": r(disc_w["mean"]), "std": r(disc_w["std"])},
                "disc_loser": {"n": disc_l["n"], "mean": r(disc_l["mean"]), "std": r(disc_l["std"])},
                "val_winner": {"n": val_w["n"], "mean": r(val_w["mean"]), "std": r(val_w["std"])},
                "val_loser": {"n": val_l["n"], "mean": r(val_l["mean"]), "std": r(val_l["std"])},
                "cohens_d_disc": r(d_disc, 3),
                "cohens_d_val": r(d_val, 3),
                "flagged": flagged,
            }

        robust_profiles_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "by_hour": {},
        }
        for h in SCAN_HOURS:
            hour_robust = robust_setups_by_hour[h]
            if not hour_robust:
                robust_profiles_result["by_hour"][str(h)] = {}
                continue
            hr_events = all_events_by_hour[h]
            atr_lo = loser_filter_result["by_hour"][str(h)]["atr_tertile_boundaries"][0]
            atr_hi = loser_filter_result["by_hour"][str(h)]["atr_tertile_boundaries"][1]
            rv_lo = loser_filter_result["by_hour"][str(h)]["rel_vol_tertile_boundaries"][0]
            rv_hi = loser_filter_result["by_hour"][str(h)]["rel_vol_tertile_boundaries"][1]

            hour_profiles = {}
            for s in hour_robust:
                all_evs = hr_events["events"].get(s, [])
                disc_evs = [e for e in all_evs if e["date"] in disc_dates]
                val_evs = [e for e in all_evs if e["date"] in val_dates_loser]
                n_winners_all = sum(1 for e in all_evs if e["hit"])
                n_losers_all = sum(1 for e in all_evs if not e["hit"])
                if n_losers_all < 30 or n_winners_all < 30:
                    hour_profiles[s] = {"status": "insufficient", "n_losers": n_losers_all, "n_winners": n_winners_all}
                    continue

                # Categorical dims
                cat_dims = [
                    ("sector", lambda f: f.get("sector")),
                    ("dow", lambda f: (["Mon","Tue","Wed","Thu","Fri"][f["dow"]]
                                       if f.get("dow") is not None and 0 <= f["dow"] <= 4 else None)),
                    ("rsi_bucket", lambda f: rsi_bucket(f.get("rsi"))),
                    ("atr_bucket", lambda f: tertile_bucket(f.get("atr_pct"), atr_lo, atr_hi)),
                    ("rel_vol_bucket", lambda f: tertile_bucket(f.get("rel_volume"), rv_lo, rv_hi)),
                ]
                cat_results = [bucket_compare(all_evs, disc_evs, val_evs, ext, name)
                               for name, ext in cat_dims]

                # Continuous dims (raw values)
                cont_dims = [
                    ("rsi", lambda f: f.get("rsi")),
                    ("atr_pct", lambda f: f.get("atr_pct")),
                    ("rel_volume", lambda f: f.get("rel_volume")),
                ]
                cont_results = [continuous_compare(all_evs, disc_evs, val_evs, ext, name)
                                for name, ext in cont_dims]

                hour_profiles[s] = {
                    "status": "tested",
                    "n_winners": n_winners_all,
                    "n_losers": n_losers_all,
                    "n_winners_disc": sum(1 for e in disc_evs if e["hit"]),
                    "n_losers_disc": sum(1 for e in disc_evs if not e["hit"]),
                    "n_winners_val": sum(1 for e in val_evs if e["hit"]),
                    "n_losers_val": sum(1 for e in val_evs if not e["hit"]),
                    "categorical": cat_results,
                    "continuous": cont_results,
                }

            robust_profiles_result["by_hour"][str(h)] = hour_profiles

        results["robust_loser_profiles"] = robust_profiles_result
        log.info(f"Robust loser profiles complete: "
                 f"{sum(len(v) for v in robust_profiles_result['by_hour'].values())} ROBUST setups profiled")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v20: ATR UNIVERSE FILTER TEST
        # For each ROBUST setup, recompute edge under 3 universe filters:
        #   - full:   no filter (baseline)
        #   - mid+hi: exclude low-ATR tertile (what we'd implement)
        #   - hi:     only high-ATR tertile (strictest)
        #
        # Each setup's firings are filtered AND the base rate is recomputed
        # on the filtered universe. This is the correct apples-to-apples
        # comparison: does edge vs base improve when we shrink the universe?
        #
        # Reported separately for discovery (F_F-I) and validation (F_A-E)
        # folds so we know whether the effect holds out-of-sample.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"atr_filter","pct":99.8,
            "message":"ATR universe filter test..."}

        atr_filter_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "by_hour": {},
        }

        for h in SCAN_HOURS:
            hour_robust = robust_setups_by_hour[h]
            if not hour_robust:
                atr_filter_result["by_hour"][str(h)] = {}
                continue

            hr_events = all_events_by_hour[h]
            # Re-use tertile boundaries from loser_filter
            atr_lo_h = loser_filter_result["by_hour"][str(h)]["atr_tertile_boundaries"][0]
            atr_hi_h = loser_filter_result["by_hour"][str(h)]["atr_tertile_boundaries"][1]

            hour_result = {
                "atr_tertile_boundaries": [atr_lo_h, atr_hi_h],
                "setups": {},
            }

            # Event-filter helpers
            def passes_full(e): return True
            def passes_mid_hi(e):
                if not e.get("features") or atr_lo_h is None: return True
                ap = e["features"].get("atr_pct")
                if ap is None: return False  # can't verify — drop to be strict
                return ap > atr_lo_h
            def passes_hi(e):
                if not e.get("features") or atr_hi_h is None: return True
                ap = e["features"].get("atr_pct")
                if ap is None: return False
                return ap >= atr_hi_h

            filters = [
                ("full", passes_full),
                ("mid_hi", passes_mid_hi),
                ("hi", passes_hi),
            ]

            for s in hour_robust:
                all_setup_ev = hr_events["events"].get(s, [])
                per_setup = {}
                for universe_name, fn in filters:
                    # Filter BOTH setup events AND base events
                    base_filtered = [e for e in hr_events["base"] if fn(e)]
                    setup_filtered = [e for e in all_setup_ev if fn(e)]

                    # Split by discovery vs validation fold
                    def fold_stats(base_evs, setup_evs, fold_dates_set):
                        base_fold = [e for e in base_evs if e["date"] in fold_dates_set]
                        setup_fold = [e for e in setup_evs if e["date"] in fold_dates_set]
                        n_base = len(base_fold)
                        n_setup = len(setup_fold)
                        if n_base == 0:
                            return {"n_base": 0, "n_setup": n_setup, "base_hr": None, "setup_hr": None, "edge": None}
                        base_hr = sum(1 for e in base_fold if e["hit"]) / n_base * 100
                        if n_setup == 0:
                            return {"n_base": n_base, "n_setup": 0, "base_hr": round(base_hr,2),
                                    "setup_hr": None, "edge": None}
                        setup_hr = sum(1 for e in setup_fold if e["hit"]) / n_setup * 100
                        return {"n_base": n_base, "n_setup": n_setup,
                                "base_hr": round(base_hr,2), "setup_hr": round(setup_hr,2),
                                "edge": round(setup_hr - base_hr, 2)}

                    disc = fold_stats(base_filtered, setup_filtered, disc_dates)
                    val = fold_stats(base_filtered, setup_filtered, val_dates_loser)
                    all_time = fold_stats(base_filtered, setup_filtered, disc_dates | val_dates_loser)

                    per_setup[universe_name] = {
                        "discovery": disc,
                        "validation": val,
                        "all": all_time,
                    }

                # Compute edge deltas vs full baseline
                full_disc_edge = per_setup["full"]["discovery"].get("edge")
                full_val_edge = per_setup["full"]["validation"].get("edge")
                full_n_setup = per_setup["full"]["all"].get("n_setup", 0)

                for un in ["mid_hi", "hi"]:
                    de = per_setup[un]["discovery"].get("edge")
                    ve = per_setup[un]["validation"].get("edge")
                    n_setup = per_setup[un]["all"].get("n_setup", 0)
                    per_setup[un]["edge_delta_disc"] = (
                        round(de - full_disc_edge, 2) if (de is not None and full_disc_edge is not None) else None
                    )
                    per_setup[un]["edge_delta_val"] = (
                        round(ve - full_val_edge, 2) if (ve is not None and full_val_edge is not None) else None
                    )
                    per_setup[un]["firing_volume_change_pct"] = (
                        round((n_setup - full_n_setup) / full_n_setup * 100, 1) if full_n_setup > 0 else None
                    )

                # Simple verdict per setup
                mid_delta_disc = per_setup["mid_hi"].get("edge_delta_disc")
                mid_delta_val = per_setup["mid_hi"].get("edge_delta_val")
                if mid_delta_disc is not None and mid_delta_val is not None:
                    if mid_delta_disc >= 1.0 and mid_delta_val >= 1.0:
                        verdict = "IMPROVES"
                    elif mid_delta_disc >= 0.5 and mid_delta_val >= 0.5:
                        verdict = "MARGINAL"
                    elif abs(mid_delta_disc) < 0.5 and abs(mid_delta_val) < 0.5:
                        verdict = "NEUTRAL"
                    elif mid_delta_disc < -0.5 or mid_delta_val < -0.5:
                        verdict = "HURTS"
                    else:
                        verdict = "INCONSISTENT"
                else:
                    verdict = "INSUFFICIENT"

                per_setup["verdict"] = verdict
                hour_result["setups"][s] = per_setup

            atr_filter_result["by_hour"][str(h)] = hour_result

        results["atr_filter_test"] = atr_filter_result
        log.info("ATR universe filter test complete")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v21-Test1: SECTOR EXCLUSION TEST
        # For each ROBUST setup, recompute edge with Financial sector
        # excluded from the universe (both setup firings and base rate).
        # Same verdict framework as ATR test: IMPROVES / NEUTRAL / HURTS.
        # Test whether the "Financial is over-loser" finding from v19 is
        # genuinely setup-specific edge improvement or just structural noise.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"sector_excl","pct":99.85,
            "message":"Sector exclusion test..."}

        sector_excl_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "excluded_sector": "Financial",
            "by_hour": {},
        }

        for h in SCAN_HOURS:
            hour_robust = robust_setups_by_hour[h]
            if not hour_robust:
                sector_excl_result["by_hour"][str(h)] = {}
                continue

            hr_events = all_events_by_hour[h]
            hour_result = {"setups": {}}

            def passes_no_financial(e):
                if not e.get("features"): return True
                return e["features"].get("sector") != "Financial"

            for s in hour_robust:
                all_setup_ev = hr_events["events"].get(s, [])

                def fold_stats_filtered(fold_dates_set, filter_fn):
                    base_fold = [e for e in hr_events["base"] if e["date"] in fold_dates_set and filter_fn(e)]
                    setup_fold = [e for e in all_setup_ev if e["date"] in fold_dates_set and filter_fn(e)]
                    n_base = len(base_fold)
                    n_setup = len(setup_fold)
                    if n_base == 0:
                        return {"n_base": 0, "n_setup": n_setup, "base_hr": None, "setup_hr": None, "edge": None}
                    base_hr = sum(1 for e in base_fold if e["hit"]) / n_base * 100
                    if n_setup == 0:
                        return {"n_base": n_base, "n_setup": 0, "base_hr": round(base_hr,2),
                                "setup_hr": None, "edge": None}
                    setup_hr = sum(1 for e in setup_fold if e["hit"]) / n_setup * 100
                    return {"n_base": n_base, "n_setup": n_setup,
                            "base_hr": round(base_hr,2), "setup_hr": round(setup_hr,2),
                            "edge": round(setup_hr - base_hr, 2)}

                def passes_any(e): return True

                # Compute full + no-Financial per fold
                full_disc = fold_stats_filtered(disc_dates, passes_any)
                full_val = fold_stats_filtered(val_dates_loser, passes_any)
                nf_disc = fold_stats_filtered(disc_dates, passes_no_financial)
                nf_val = fold_stats_filtered(val_dates_loser, passes_no_financial)

                full_all = fold_stats_filtered(disc_dates | val_dates_loser, passes_any)
                nf_all = fold_stats_filtered(disc_dates | val_dates_loser, passes_no_financial)

                # Delta vs full baseline
                de_disc = (nf_disc.get("edge") - full_disc.get("edge")
                           if nf_disc.get("edge") is not None and full_disc.get("edge") is not None
                           else None)
                de_val = (nf_val.get("edge") - full_val.get("edge")
                          if nf_val.get("edge") is not None and full_val.get("edge") is not None
                          else None)
                vol_delta = ((nf_all["n_setup"] - full_all["n_setup"]) / full_all["n_setup"] * 100
                             if full_all["n_setup"] > 0 else None)

                # Verdict
                if de_disc is not None and de_val is not None:
                    if de_disc >= 1.0 and de_val >= 1.0:
                        verdict = "IMPROVES"
                    elif de_disc >= 0.5 and de_val >= 0.5:
                        verdict = "MARGINAL"
                    elif abs(de_disc) < 0.5 and abs(de_val) < 0.5:
                        verdict = "NEUTRAL"
                    elif de_disc < -0.5 or de_val < -0.5:
                        verdict = "HURTS"
                    else:
                        verdict = "INCONSISTENT"
                else:
                    verdict = "INSUFFICIENT"

                hour_result["setups"][s] = {
                    "full": {"discovery": full_disc, "validation": full_val, "all": full_all},
                    "no_financial": {
                        "discovery": nf_disc, "validation": nf_val, "all": nf_all,
                        "edge_delta_disc": round(de_disc, 2) if de_disc is not None else None,
                        "edge_delta_val": round(de_val, 2) if de_val is not None else None,
                        "firing_volume_change_pct": round(vol_delta, 1) if vol_delta is not None else None,
                    },
                    "verdict": verdict,
                }

            sector_excl_result["by_hour"][str(h)] = hour_result

        results["sector_exclusion_test"] = sector_excl_result
        log.info("Sector exclusion test complete")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v21-Test2: CONSISTENT FILTER VALIDATION
        # Re-test v18's high-conviction CONSISTENT filters with the same
        # edge-delta framework used for ATR and sector tests.
        # Filters tested:
        #   - opening_drive @ 15:00 skip dow=Thu
        #   - opening_drive @ 15:00 skip rsi>70 (overbought)
        #   - orb_60_break @ 13:00 skip sector=Industrial
        #   - orb_vol @ 15:00 skip dow=Fri
        # IMPROVES = edge delta ≥ +1pp on both folds.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"consistent_filters","pct":99.9,
            "message":"CONSISTENT filter validation..."}

        consistent_filter_tests = [
            {"setup": "opening_drive", "hour": 15, "label": "skip dow=Thu",
             "filter_out": lambda e: (e.get("features") or {}).get("dow") == 3},  # 3=Thu
            {"setup": "opening_drive", "hour": 15, "label": "skip rsi>70",
             "filter_out": lambda e: ((e.get("features") or {}).get("rsi") or 0) > 70},
            {"setup": "orb_60_break", "hour": 13, "label": "skip sector=Industrial",
             "filter_out": lambda e: (e.get("features") or {}).get("sector") == "Industrial"},
            {"setup": "orb_vol", "hour": 15, "label": "skip dow=Fri",
             "filter_out": lambda e: (e.get("features") or {}).get("dow") == 4},  # 4=Fri
        ]

        consistent_filter_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "filters": [],
        }

        for test in consistent_filter_tests:
            s = test["setup"]
            h = test["hour"]
            hr_events = all_events_by_hour[h]
            all_setup_ev = hr_events["events"].get(s, [])

            def passes_filter(e):
                return not test["filter_out"](e)

            def fold_stats_filtered(fold_dates_set, filter_fn):
                base_fold = [e for e in hr_events["base"] if e["date"] in fold_dates_set and filter_fn(e)]
                setup_fold = [e for e in all_setup_ev if e["date"] in fold_dates_set and filter_fn(e)]
                n_base = len(base_fold); n_setup = len(setup_fold)
                if n_base == 0:
                    return {"n_base":0,"n_setup":n_setup,"base_hr":None,"setup_hr":None,"edge":None}
                base_hr = sum(1 for e in base_fold if e["hit"]) / n_base * 100
                if n_setup == 0:
                    return {"n_base":n_base,"n_setup":0,"base_hr":round(base_hr,2),"setup_hr":None,"edge":None}
                setup_hr = sum(1 for e in setup_fold if e["hit"]) / n_setup * 100
                return {"n_base":n_base,"n_setup":n_setup,"base_hr":round(base_hr,2),
                        "setup_hr":round(setup_hr,2),"edge":round(setup_hr-base_hr,2)}

            def passes_any(e): return True

            full_disc = fold_stats_filtered(disc_dates, passes_any)
            full_val = fold_stats_filtered(val_dates_loser, passes_any)
            filt_disc = fold_stats_filtered(disc_dates, passes_filter)
            filt_val = fold_stats_filtered(val_dates_loser, passes_filter)
            full_all = fold_stats_filtered(disc_dates | val_dates_loser, passes_any)
            filt_all = fold_stats_filtered(disc_dates | val_dates_loser, passes_filter)

            de_disc = (filt_disc["edge"] - full_disc["edge"]
                       if filt_disc["edge"] is not None and full_disc["edge"] is not None else None)
            de_val = (filt_val["edge"] - full_val["edge"]
                      if filt_val["edge"] is not None and full_val["edge"] is not None else None)
            vol_delta = ((filt_all["n_setup"] - full_all["n_setup"]) / full_all["n_setup"] * 100
                         if full_all["n_setup"] > 0 else None)

            if de_disc is not None and de_val is not None:
                if de_disc >= 1.0 and de_val >= 1.0: verdict = "IMPROVES"
                elif de_disc >= 0.5 and de_val >= 0.5: verdict = "MARGINAL"
                elif abs(de_disc) < 0.5 and abs(de_val) < 0.5: verdict = "NEUTRAL"
                elif de_disc < -0.5 or de_val < -0.5: verdict = "HURTS"
                else: verdict = "INCONSISTENT"
            else:
                verdict = "INSUFFICIENT"

            consistent_filter_result["filters"].append({
                "setup": s, "hour": h, "label": test["label"],
                "full": {"discovery": full_disc, "validation": full_val, "all": full_all},
                "filtered": {
                    "discovery": filt_disc, "validation": filt_val, "all": filt_all,
                    "edge_delta_disc": round(de_disc, 2) if de_disc is not None else None,
                    "edge_delta_val": round(de_val, 2) if de_val is not None else None,
                    "firing_volume_change_pct": round(vol_delta, 1) if vol_delta is not None else None,
                },
                "verdict": verdict,
            })

        results["consistent_filter_test"] = consistent_filter_result
        log.info("CONSISTENT filter validation complete")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v22: FILTER STACKING TEST for rel_strength_iwm
        # Question: does ATR filter + Financial exclusion stack additively,
        # or is ATR alone sufficient?
        #
        # For rel_strength_iwm @ 12, 13, 14 (ROBUST hours), compare 4 universes:
        #   A: full (baseline)
        #   B: atr_only (ATR > low tertile ~2.85%)  ← current v21 live filter
        #   C: nofin_only (exclude Financial)
        #   D: atr_and_nofin (both stacked)
        #
        # Two verdicts per setup-hour:
        #   - D vs A: is the full stack better than baseline? (expected yes)
        #   - D vs B: is adding Financial exclusion worth it over ATR alone?
        #
        # If D vs B = IMPROVES: wire the stacked filter.
        # If D vs B = MARGINAL/NEUTRAL: ATR alone is sufficient.
        # If D vs B = HURTS: filters don't stack — ATR alone preferred.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"stacking","pct":99.95,
            "message":"Filter stacking test..."}

        stacking_result = {
            "disc_folds": disc_fold_names,
            "val_folds": val_fold_names,
            "by_hour": {},
        }

        target_setup = "rel_strength_iwm"
        stacking_hours = [12, 13, 14]  # where ATR filter showed IMPROVES

        for h in stacking_hours:
            hr_events = all_events_by_hour[h]
            all_setup_ev = hr_events["events"].get(target_setup, [])
            atr_lo_h = loser_filter_result["by_hour"][str(h)]["atr_tertile_boundaries"][0]

            if atr_lo_h is None or not all_setup_ev:
                stacking_result["by_hour"][str(h)] = {"status": "insufficient"}
                continue

            # Universe filter helpers
            def passes_full(e): return True
            def passes_atr_only(e):
                if not e.get("features"): return True
                ap = e["features"].get("atr_pct")
                if ap is None: return False
                return ap > atr_lo_h
            def passes_nofin_only(e):
                if not e.get("features"): return True
                return e["features"].get("sector") != "Financial"
            def passes_atr_and_nofin(e):
                if not e.get("features"): return True
                ap = e["features"].get("atr_pct")
                if ap is None: return False
                if ap <= atr_lo_h: return False
                return e["features"].get("sector") != "Financial"

            universes = [
                ("A_full", passes_full),
                ("B_atr_only", passes_atr_only),
                ("C_nofin_only", passes_nofin_only),
                ("D_atr_and_nofin", passes_atr_and_nofin),
            ]

            def fold_stats_for_universe(fold_dates_set, filter_fn):
                base_fold = [e for e in hr_events["base"] if e["date"] in fold_dates_set and filter_fn(e)]
                setup_fold = [e for e in all_setup_ev if e["date"] in fold_dates_set and filter_fn(e)]
                n_base = len(base_fold); n_setup = len(setup_fold)
                if n_base == 0:
                    return {"n_base":0,"n_setup":n_setup,"base_hr":None,"setup_hr":None,"edge":None}
                base_hr = sum(1 for e in base_fold if e["hit"]) / n_base * 100
                if n_setup == 0:
                    return {"n_base":n_base,"n_setup":0,"base_hr":round(base_hr,2),"setup_hr":None,"edge":None}
                setup_hr = sum(1 for e in setup_fold if e["hit"]) / n_setup * 100
                return {"n_base":n_base,"n_setup":n_setup,"base_hr":round(base_hr,2),
                        "setup_hr":round(setup_hr,2),"edge":round(setup_hr-base_hr,2)}

            hour_data = {"atr_threshold_pct": atr_lo_h, "universes": {}}
            for un_name, fn in universes:
                disc = fold_stats_for_universe(disc_dates, fn)
                val = fold_stats_for_universe(val_dates_loser, fn)
                all_st = fold_stats_for_universe(disc_dates | val_dates_loser, fn)
                hour_data["universes"][un_name] = {
                    "discovery": disc, "validation": val, "all": all_st,
                }

            # Compute key comparisons
            def delta(a, b):
                """Edge delta b vs a (positive = b improves)."""
                if a is None or b is None: return None
                return round(b - a, 2)
            def classify(d_disc, d_val):
                if d_disc is None or d_val is None: return "INSUFFICIENT"
                if d_disc >= 1.0 and d_val >= 1.0: return "IMPROVES"
                if d_disc >= 0.5 and d_val >= 0.5: return "MARGINAL"
                if abs(d_disc) < 0.5 and abs(d_val) < 0.5: return "NEUTRAL"
                if d_disc < -0.5 or d_val < -0.5: return "HURTS"
                return "INCONSISTENT"

            A = hour_data["universes"]["A_full"]
            B = hour_data["universes"]["B_atr_only"]
            C = hour_data["universes"]["C_nofin_only"]
            D = hour_data["universes"]["D_atr_and_nofin"]

            d_vs_a_disc = delta(A["discovery"]["edge"], D["discovery"]["edge"])
            d_vs_a_val = delta(A["validation"]["edge"], D["validation"]["edge"])
            d_vs_b_disc = delta(B["discovery"]["edge"], D["discovery"]["edge"])
            d_vs_b_val = delta(B["validation"]["edge"], D["validation"]["edge"])
            b_n = B["all"]["n_setup"]; d_n = D["all"]["n_setup"]
            a_n = A["all"]["n_setup"]

            hour_data["comparisons"] = {
                "D_vs_A": {
                    "edge_delta_disc": d_vs_a_disc,
                    "edge_delta_val": d_vs_a_val,
                    "volume_change_pct": round((d_n - a_n) / a_n * 100, 1) if a_n > 0 else None,
                    "verdict": classify(d_vs_a_disc, d_vs_a_val),
                },
                "D_vs_B": {
                    "edge_delta_disc": d_vs_b_disc,
                    "edge_delta_val": d_vs_b_val,
                    "volume_change_pct": round((d_n - b_n) / b_n * 100, 1) if b_n > 0 else None,
                    "verdict": classify(d_vs_b_disc, d_vs_b_val),
                },
            }

            stacking_result["by_hour"][str(h)] = hour_data

        results["stacking_filter_test"] = stacking_result
        log.info("Filter stacking test complete")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v23: LIVE-READINESS SIMULATION
        # Simulate what the live scanner would have produced across the
        # most recent 30 trading days, applying the same filters that
        # live code applies (ex_fa gate + breadth_rule + atr_filter_rule).
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"live_sim","pct":99.97,
            "message":"Live-readiness simulation..."}
        log.info("v23.1 live simulation block: ENTRY")

        # Identify active setup list using same logic as live loader:
        #   (1) survivor in original (strong/moderate tier on main test fold)
        #   (2) NOT failed Ex-F_A
        # Survivors come from `original_survivors` computed earlier.
        # Ex-F_A rejects set from ex_fa_result.
        ex_fa_rejects = set()  # (setup, hour) tuples that failed ex_fa
        for h_str, hr_data in ex_fa_result.get("by_hour", {}).items():
            # hr_data has shape {base_hit_rate, n_base, setups: {setup: {ex_fa_edge, holds_up, ...}}}
            for s, rec in (hr_data.get("setups") or {}).items():
                if rec.get("ex_fa_edge") is not None and not rec.get("holds_up", False):
                    ex_fa_rejects.add((s, int(h_str)))

        live_active = set()  # (setup, hour)
        for (s, h) in original_survivors:
            if (s, h) not in ex_fa_rejects:
                live_active.add((s, h))

        # Derive per-(setup, hour) ATR threshold from atr_filter_test
        # (only for setups with verdict=IMPROVES → these get the filter).
        atr_rule_thresholds = {}  # (setup, hour) → threshold_pct
        for h_str, hd in atr_filter_result.get("by_hour", {}).items():
            h_int = int(h_str)
            atr_lo = hd.get("atr_tertile_boundaries", [None, None])[0]
            if atr_lo is None: continue
            for s, sd in hd.get("setups", {}).items():
                if sd.get("verdict") == "IMPROVES":
                    atr_rule_thresholds[(s, h_int)] = atr_lo

        # Determine last 30 trading days present in event data
        all_dates = set()
        for h, hr in all_events_by_hour.items():
            for e in hr["base"]:
                all_dates.add(e["date"])
        sorted_dates = sorted(all_dates, reverse=True)[:30]
        sim_dates = set(sorted_dates)
        sim_date_range = (min(sim_dates), max(sim_dates)) if sim_dates else (None, None)
        log.info(f"Live sim: simulating {len(sim_dates)} trading days "
                 f"from {sim_date_range[0]} to {sim_date_range[1]}")

        # Per-day tally: matches only pass if
        #   (1) setup is in live_active
        #   (2) if (setup, hour) has atr_rule, stock's atr_pct > threshold
        #   (3) breadth_rule — we'd need the r2k_breadth at scan time;
        #       we skip this check in sim because we don't have breadth for
        #       the simulated date. Instead we note that breadth filtering
        #       would further reduce counts. This gives a MAXIMUM firing count.
        per_day = defaultdict(lambda: {
            "total": 0,
            "by_hour": defaultdict(int),
            "by_setup": defaultdict(int),
            "atr_filtered": 0,  # firings the ATR filter would skip
            "simultaneous": 0,  # ticker×hour with ≥2 matches
            "unique_tickers": set(),
        })
        sim_firings = []  # flat list for cross-day aggregation

        for h, hr in all_events_by_hour.items():
            # For this hour, for each event that would fire, check filters
            # events[s] = per-setup event list (only events where setup fired)
            # base events give us per-stock feature info
            # Walk through per-setup events filtered to sim dates
            for (s, active_hour) in live_active:
                if active_hour != h: continue
                ev_list = hr["events"].get(s, [])
                atr_thresh = atr_rule_thresholds.get((s, h))
                for e in ev_list:
                    if e["date"] not in sim_dates: continue
                    day = per_day[e["date"]]
                    features = e.get("features") or {}
                    stock_atr = features.get("atr_pct")
                    # Apply ATR filter rule if applicable
                    if atr_thresh is not None:
                        if stock_atr is None or stock_atr <= atr_thresh:
                            day["atr_filtered"] += 1
                            continue
                    # Record firing
                    day["total"] += 1
                    day["by_hour"][h] += 1
                    day["by_setup"][s] += 1
                    day["unique_tickers"].add((e["ticker"], h))
                    sim_firings.append({
                        "date": e["date"], "ticker": e["ticker"],
                        "hour": h, "setup": s, "hit": e["hit"],
                    })

        # Compute simultaneous firings (same ticker, same date+hour, ≥2 setups)
        from collections import Counter
        ticker_hour_counts = Counter()
        for f in sim_firings:
            ticker_hour_counts[(f["date"], f["ticker"], f["hour"])] += 1
        per_day_simul = defaultdict(int)
        for (date, tkr, h), cnt in ticker_hour_counts.items():
            if cnt >= 2:
                per_day_simul[date] += 1

        # Build clean per-day structure
        per_day_clean = {}
        for date in sorted(sim_dates):
            d = per_day.get(date, {})
            if not d:
                per_day_clean[date] = {
                    "total": 0, "by_hour": {}, "by_setup": {},
                    "atr_filtered": 0, "simultaneous": 0, "unique_tickers": 0,
                }
                continue
            per_day_clean[date] = {
                "total": d["total"],
                "by_hour": {str(k): v for k, v in d["by_hour"].items()},
                "by_setup": dict(d["by_setup"]),
                "atr_filtered": d["atr_filtered"],
                "simultaneous": per_day_simul.get(date, 0),
                "unique_tickers": len(d["unique_tickers"]),
            }

        # Aggregates
        total_firings = sum(x["total"] for x in per_day_clean.values())
        total_atr_filtered = sum(x["atr_filtered"] for x in per_day_clean.values())
        total_simul = sum(x["simultaneous"] for x in per_day_clean.values())
        daily_totals = [x["total"] for x in per_day_clean.values()]
        daily_totals_sorted = sorted(daily_totals)
        n_days = len(daily_totals_sorted)
        if n_days > 0:
            median_daily = (daily_totals_sorted[n_days // 2]
                            if n_days % 2 == 1
                            else (daily_totals_sorted[n_days//2 - 1] + daily_totals_sorted[n_days//2]) / 2)
            mean_daily = sum(daily_totals) / n_days
            max_daily = max(daily_totals)
            min_daily = min(daily_totals)
            n_zero_days = sum(1 for t in daily_totals if t == 0)
            n_high_days = sum(1 for t in daily_totals if t >= 30)
        else:
            median_daily = mean_daily = max_daily = min_daily = 0
            n_zero_days = n_high_days = 0

        # Hour distribution across all firings
        hour_totals = defaultdict(int)
        setup_totals = defaultdict(int)
        for f in sim_firings:
            hour_totals[f["hour"]] += 1
            setup_totals[f["setup"]] += 1

        # Hit rate in sim period (observed — NOT a backtest metric, just a sanity check)
        hits = sum(1 for f in sim_firings if f["hit"])
        sim_hit_rate = (hits / len(sim_firings) * 100) if sim_firings else None

        live_sim_result = {
            "sim_date_range": sim_date_range,
            "n_sim_days": n_days,
            "active_setup_count": len(live_active),
            "atr_rule_setups": [f"{s}@{h}" for (s, h) in sorted(atr_rule_thresholds)],
            "summary": {
                "total_firings": total_firings,
                "mean_per_day": round(mean_daily, 1),
                "median_per_day": median_daily,
                "min_per_day": min_daily,
                "max_per_day": max_daily,
                "zero_firing_days": n_zero_days,
                "high_volume_days_30plus": n_high_days,
                "total_atr_filtered": total_atr_filtered,
                "total_simultaneous": total_simul,
                "sim_observed_hit_rate_pct": round(sim_hit_rate, 2) if sim_hit_rate is not None else None,
            },
            "hour_distribution": {str(k): v for k, v in sorted(hour_totals.items())},
            "setup_distribution": dict(sorted(setup_totals.items(), key=lambda x: -x[1])),
            "per_day": per_day_clean,
            "note": (
                "Simulation excludes breadth_rule filtering (requires r2k breadth "
                "at scan time which isn't stored in events). Firing counts shown "
                "are MAXIMUM — live numbers will be equal or lower on breadth-down days."
            ),
        }
        results["live_simulation"] = live_sim_result
        log.info(f"Live sim complete: {total_firings} firings across {n_days} days, "
                 f"median {median_daily}/day, max {max_daily}/day, "
                 f"{n_zero_days} zero-firing days, {total_atr_filtered} ATR-filtered")
        _save_results()

        # ═══════════════════════════════════════════════════════════════
        # v24: CONVICTION RANKING TEST
        # Core question: does ranking firings by conviction improve hit rate?
        # If top-N hit rate is materially higher than all-firings hit rate,
        # ranking works and we can deploy it in the live scanner.
        # If top-N ≈ all-firings, our scoring isn't capturing edge and we
        # need to revise before deploying.
        #
        # Per-firing score formula:
        #   base = setup.test_edge + setup.test_hit_rate * 0.5
        #   if setup is ROBUST (fold_swap): base *= 1.2
        # Per-stock score (when multiple setups fire on same stock):
        #   sum(base for each firing setup), minus redundancy penalty
        #   (subtract score of weaker setup in known-redundant pairs)
        #
        # For each (date, hour), rank stocks by score; measure hit rate at
        # top-1, top-3, top-5, top-10, top-20, all cutoffs, per fold.
        # ═══════════════════════════════════════════════════════════════
        setup_eval_progress = {"phase":"conv_rank_test","pct":99.98,
            "message":"Conviction ranking test..."}

        # Build per-(setup, hour) base score from test fold numbers (what live uses)
        # This is the score each firing CONTRIBUTES to a stock's total at that hour.
        setup_hour_score = {}  # (setup, hour) → base_score
        for h_str, hr in results.get("hours", {}).items():
            h = int(h_str)
            for s, sd in hr.get("setups", {}).items():
                t = sd.get("test", {})
                hit = t.get("hit_rate")
                edge = t.get("edge_vs_base")
                n = t.get("n_events", 0)
                if hit is None or edge is None: continue
                # Only active setups qualify (tier check)
                tier = None
                if edge >= SETUP_EVIDENCE_THRESHOLDS["strong"]["edge"] and n >= SETUP_EVIDENCE_THRESHOLDS["strong"]["n"]:
                    tier = "strong"
                elif edge >= SETUP_EVIDENCE_THRESHOLDS["moderate"]["edge"] and n >= SETUP_EVIDENCE_THRESHOLDS["moderate"]["n"]:
                    tier = "moderate"
                if not tier: continue
                # Ex-F_A gate
                ex_fa = ex_fa_result.get("by_hour", {}).get(h_str, {}).get("setups", {}).get(s, {})
                if ex_fa.get("ex_fa_edge") is not None and not ex_fa.get("holds_up", False):
                    continue  # dropped
                # ROBUST boost
                fs = fold_swap_result.get("by_hour", {}).get(h_str, {}).get(s, {})
                rob = fs.get("robustness")
                base = edge + hit * 0.5
                if rob == "ROBUST": base *= 1.2
                setup_hour_score[(s, h)] = round(base, 2)

        log.info(f"Conv-rank: {len(setup_hour_score)} active setup-hours with scores")

        # v16 redundant pair list — actual pairs identified in overlap analysis
        REDUNDANT_PAIRS_SERVER = [
            ("orb_vol", "orb_60_break"),  # v16 overlap found J=0.50-0.60 at 13:00 and 15:00
        ]

        # Fold definitions (from earlier in run)
        fold_sets = {"train": train_dates, "val": val_dates, "test": test_dates}

        # Compute per-stock-per-day-per-hour scores from all firing events
        # all_events_by_hour[h]["events"][setup] = [event records, each has date+ticker+hit]
        #
        # For each (date, hour, ticker), aggregate: sum of active setup scores, minus redundancy.
        # Then rank stocks within each (date, hour) and measure hit rate at each cutoff.
        per_hour_stocks = defaultdict(lambda: defaultdict(lambda: {
            "setups_fired": [], "score": 0.0, "hit": None
        }))
        # Structure: per_hour_stocks[(date, hour)][ticker] = {setups_fired, score, hit}

        for h, hr in all_events_by_hour.items():
            events_by_setup = hr.get("events", {})
            for s, ev_list in events_by_setup.items():
                if (s, h) not in setup_hour_score: continue  # not an active setup
                base_score = setup_hour_score[(s, h)]
                for e in ev_list:
                    date = e["date"]
                    ticker = e["ticker"]
                    rec = per_hour_stocks[(date, h)][ticker]
                    rec["setups_fired"].append((s, base_score))
                    rec["hit"] = e["hit"]  # same hit for same (date,hour,ticker)

        # Now compute aggregate score with redundancy penalty
        for (date, h), tickers in per_hour_stocks.items():
            for ticker, rec in tickers.items():
                fired_set = set(s for s, _ in rec["setups_fired"])
                score_map = dict(rec["setups_fired"])
                total = sum(score_map.values())
                # Redundancy: for each redundant pair that fires together, subtract weaker
                for (a, b) in REDUNDANT_PAIRS_SERVER:
                    if a in fired_set and b in fired_set:
                        total -= min(score_map[a], score_map[b])
                rec["score"] = round(total, 2)

        # For each fold, rank per (date, hour) and measure hit rate at cutoffs
        cutoffs = [1, 3, 5, 10, 20]
        conv_rank_result = {
            "setup_hour_scores": {f"{s}@{h}": sc for (s, h), sc in sorted(setup_hour_score.items())},
            "by_fold": {},
        }

        for fold_name, fold_dates_set in fold_sets.items():
            fold_cutoff_stats = {f"top_{c}": {"n_stocks": 0, "n_hits": 0} for c in cutoffs}
            fold_cutoff_stats["all"] = {"n_stocks": 0, "n_hits": 0}
            n_scan_slots = 0  # (date, hour) slots with ≥1 firing

            # Also track per-(date,hour) top-N hit rate for day-level stats
            daily_stats = defaultdict(lambda: {f"top_{c}": {"n": 0, "hits": 0} for c in cutoffs + ["all"]})

            for (date, h), tickers in per_hour_stocks.items():
                if date not in fold_dates_set: continue
                if not tickers: continue
                n_scan_slots += 1

                # Rank by score desc
                ranked = sorted(tickers.items(), key=lambda kv: -kv[1]["score"])

                # Measure hit rate at each cutoff
                for c in cutoffs:
                    top_c = ranked[:c]
                    fold_cutoff_stats[f"top_{c}"]["n_stocks"] += len(top_c)
                    fold_cutoff_stats[f"top_{c}"]["n_hits"] += sum(1 for _, r in top_c if r["hit"])
                # All
                fold_cutoff_stats["all"]["n_stocks"] += len(ranked)
                fold_cutoff_stats["all"]["n_hits"] += sum(1 for _, r in ranked if r["hit"])

            # Compute hit rates
            fold_out = {"n_scan_slots": n_scan_slots, "cutoffs": {}}
            for key, stats in fold_cutoff_stats.items():
                n = stats["n_stocks"]
                hits = stats["n_hits"]
                hr = (hits / n * 100) if n > 0 else None
                fold_out["cutoffs"][key] = {
                    "n_stocks": n,
                    "n_hits": hits,
                    "hit_rate": round(hr, 2) if hr is not None else None,
                }

            # Baseline: hit rate of "all" firings for this fold
            all_hr = fold_out["cutoffs"]["all"]["hit_rate"]
            # Delta vs all
            for key, c_stats in fold_out["cutoffs"].items():
                if c_stats["hit_rate"] is not None and all_hr is not None:
                    c_stats["delta_vs_all"] = round(c_stats["hit_rate"] - all_hr, 2)
                else:
                    c_stats["delta_vs_all"] = None

            conv_rank_result["by_fold"][fold_name] = fold_out

        # Monotonicity check: is top_1 >= top_3 >= top_5 >= top_10 >= top_20 >= all?
        # Validated on val AND test folds for out-of-sample check.
        def monotonic_check(fold_data):
            order = ["top_1", "top_3", "top_5", "top_10", "top_20", "all"]
            hrs = [fold_data["cutoffs"][k]["hit_rate"] for k in order]
            if any(x is None for x in hrs): return False
            return all(hrs[i] >= hrs[i+1] - 0.5 for i in range(len(hrs)-1))  # 0.5pp slop

        conv_rank_result["monotonic_val"] = monotonic_check(conv_rank_result["by_fold"]["val"])
        conv_rank_result["monotonic_test"] = monotonic_check(conv_rank_result["by_fold"]["test"])

        # Verdict: does ranking improve top-10 hit rate by ≥5pp vs all on BOTH val and test?
        val_top10_delta = conv_rank_result["by_fold"]["val"]["cutoffs"]["top_10"].get("delta_vs_all")
        test_top10_delta = conv_rank_result["by_fold"]["test"]["cutoffs"]["top_10"].get("delta_vs_all")
        if val_top10_delta is not None and test_top10_delta is not None:
            if val_top10_delta >= 5.0 and test_top10_delta >= 5.0:
                verdict = "RANKING_WORKS"
            elif val_top10_delta >= 2.0 and test_top10_delta >= 2.0:
                verdict = "RANKING_HELPS_MARGINALLY"
            elif abs(val_top10_delta) < 2.0 and abs(test_top10_delta) < 2.0:
                verdict = "RANKING_NEUTRAL"
            elif val_top10_delta < -2.0 or test_top10_delta < -2.0:
                verdict = "RANKING_HURTS"
            else:
                verdict = "INCONSISTENT"
        else:
            verdict = "INSUFFICIENT"
        conv_rank_result["verdict"] = verdict

        results["conviction_ranking_test"] = conv_rank_result
        log.info(f"Conviction ranking test complete: verdict={verdict}, "
                 f"val Δ={val_top10_delta}, test Δ={test_top10_delta}")
        _save_results()

        SETUP_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
        setup_eval_progress = {"phase":"done","pct":100,
            "message":f"Done. Evaluated {len(SETUP_NAMES)} setups × {len(SCAN_HOURS)} scan hours."}
        log.info("Setup evaluation complete")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"Setup eval failed: {e}\n{tb}", exc_info=True)
        setup_eval_progress = {"phase":"error","pct":0,"message":str(e)}
        # v23.1: record the error into the results file so we can see what went wrong
        # (otherwise it silently overwrites the last _save_results state or stays stale).
        try:
            if 'results' in locals():
                results.setdefault("_analysis_errors", {})["setup_eval"] = f"{type(e).__name__}: {e}"
                results["_analysis_errors"]["setup_eval_trace"] = tb
                SETUP_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
                log.info("Wrote partial results with error info")
        except Exception as e2:
            log.error(f"Also failed to write partial results: {e2}")
    finally:
        setup_eval_in_progress = False

def load_setup_results():
    if not SETUP_RESULTS_PATH.exists(): return None
    try: return json.loads(SETUP_RESULTS_PATH.read_text())
    except: return None

# ═══════════════════════════════════════════════════════════════════
# v25: CONVICTION MODEL — TRAIN + CALIBRATE
# Goal: predict the binary outcome "did stock reach scan_price × 1.01
# before 15:55 ET close?" using all available features, and report
# calibration by probability bucket.
#
# Broad scope: every (stock, scan_hour, date) with sufficient data,
# NOT just current active setup firings. Features include bar-data,
# cross-sectional ranks, SPY/sector relative, gap, regime, AND all
# setup-firing flags as boolean features.
#
# Target: achieve an 80%+ actual hit rate in the model's
# highest-probability bucket (with n≥30 and Wilson CI lower bound ≥75%).
# ═══════════════════════════════════════════════════════════════════
CONVICTION_MODEL_PATH = DATA_DIR / "conviction_model.pkl"
CONVICTION_CALIB_PATH = DATA_DIR / "conviction_calibrator.pkl"
CONVICTION_RESULTS_PATH = DATA_DIR / "conviction_results.json"
conviction_train_in_progress = False
conviction_train_progress = {"phase":"idle","pct":0,"message":""}

def wilson_ci(n_success, n_total, confidence=0.95):
    """Wilson score interval for binomial proportions. Returns (lower, upper)."""
    if n_total == 0: return (0.0, 0.0)
    import math
    z = 1.96  # 95% confidence
    p = n_success / n_total
    denom = 1 + z*z / n_total
    center = (p + z*z / (2*n_total)) / denom
    spread = (z / denom) * math.sqrt(p*(1-p)/n_total + z*z/(4*n_total*n_total))
    return (max(0.0, center - spread), min(1.0, center + spread))

def run_conviction_training():
    """
    Train LightGBM classifier predicting `price hits +1% before 15:55` binary.
    Uses broad scope: every (date, ticker, scan_hour) with features computable,
    NOT filtered to setup firings. Plus setup-firing flags as features.
    """
    global conviction_train_in_progress, conviction_train_progress
    if conviction_train_in_progress:
        log.warning("conviction training already running")
        return
    conviction_train_in_progress = True
    conviction_train_progress = {"phase":"starting","pct":0,"message":"Starting conviction model training..."}

    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No bar cache. Run Training first.")

        conviction_train_progress = {"phase":"loading","pct":2,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        # Determine all trading dates from IWM (ETF, reliable daily bars)
        iwm_daily = daily_bars.get("IWM", [])
        if len(iwm_daily) < 200:
            raise RuntimeError(f"Insufficient IWM daily history: {len(iwm_daily)} bars")
        all_dates = sorted(set(b["t"][:10] for b in iwm_daily))
        log.info(f"Conviction training: {len(all_dates)} trading dates available")

        # Three-way temporal split (same as setup_eval: 60/20/20)
        n = len(all_dates)
        split_tr = int(n * 0.60)
        split_va = int(n * 0.80)
        train_dates = set(all_dates[:split_tr])
        val_dates = set(all_dates[split_tr:split_va])
        test_dates = set(all_dates[split_va:])
        log.info(f"Split: train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)}")

        # Build examples: for each (date, scan_hour, ticker), compute features + target
        # This is the SAME approach as run_setup_evaluation but emits ALL stocks (not just firings)
        # and includes setup-firing flags in the feature vector.

        # Extract trading-hour bars per ticker per date
        def bars_for_date(ticker, date):
            return [b for b in intraday_bars.get(ticker, []) if b["t"][:10] == date]
        def daily_up_to(ticker, date):
            return [b for b in daily_bars.get(ticker, []) if b["t"][:10] < date]

        examples = []  # list of dicts: {fold, date, hour, ticker, features, label, setup_flags}
        from collections import defaultdict as _dd
        scan_hours = SCAN_HOURS  # [11,12,13,14,15]
        tickers = [t for t in TICKERS if t not in ("SPY","IWM")]

        total_dates = len(all_dates)
        for di, date in enumerate(all_dates):
            if (di + 1) % 10 == 0:
                pct = 5 + int((di / total_dates) * 60)
                conviction_train_progress = {"phase":"features","pct":pct,
                    "message":f"Building features for date {di+1}/{total_dates}..."}

            # SPY / IWM context for this date (daily fetch for gap calc; intraday for regime)
            spy_intraday_date = [b for b in intraday_bars.get("SPY", []) if b["t"][:10] == date]
            iwm_intraday_date = [b for b in intraday_bars.get("IWM", []) if b["t"][:10] == date]

            # Sector green counts (recomputed per scan hour below)
            for scan_hour in scan_hours:
                scan_min = scan_hour * 60  # minutes since midnight ET
                # bars strictly before scan; bars at-or-after scan for outcome
                # Using minutes-since-open representation:
                # scan_minute_et = scan_hour * 60 (e.g. 11:00 = 660)
                scan_minute_et = scan_hour * 60

                # SPY context at this scan
                spy_before = [b for b in spy_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                iwm_before = [b for b in iwm_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                spy_ctx = compute_spy_context(spy_before) if spy_before else None

                # Compute sector breadth at scan: fraction of tickers in each sector that are green
                # (close > prev_close). Used by sector_bounce setup.
                sector_green_count = _dd(lambda: [0, 0])  # sector → [green_count, total]
                _per_ticker_scan_price = {}
                _per_ticker_prev_close = {}
                _per_ticker_before_bars = {}
                _per_ticker_after_bars = {}
                _per_ticker_daily = {}
                for tkr in tickers:
                    tbars = bars_for_date(tkr, date)
                    if len(tbars) < 6: continue
                    tb = [b for b in tbars if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                    ta = [b for b in tbars if (bar_to_et_minutes(b) or -1) >= scan_minute_et]
                    if len(tb) < 6 or len(ta) < 2: continue
                    dl = daily_up_to(tkr, date)
                    pc = dl[-1]["c"] if dl else None
                    if pc is None: continue
                    sp = tb[-1]["c"]
                    tsec = SECTORS.get(tkr, "?")
                    sector_green_count[tsec][1] += 1
                    if sp > pc:
                        sector_green_count[tsec][0] += 1
                    _per_ticker_scan_price[tkr] = sp
                    _per_ticker_prev_close[tkr] = pc
                    _per_ticker_before_bars[tkr] = tb
                    _per_ticker_after_bars[tkr] = ta
                    _per_ticker_daily[tkr] = dl

                # Pass 1: build per-ticker feature + setup flags + outcome
                date_hour_rows = []
                for ticker in _per_ticker_scan_price.keys():
                    before = _per_ticker_before_bars[ticker]
                    after = _per_ticker_after_bars[ticker]
                    scan_price = _per_ticker_scan_price[ticker]
                    prev_close = _per_ticker_prev_close[ticker]
                    open_price = before[0]["o"]
                    dl = _per_ticker_daily[ticker]

                    # Compute features
                    feat = compute_features(before, dl, scan_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue

                    # Setup firings at this scan
                    sec = SECTORS.get(ticker, "?")
                    sgc = sector_green_count[sec]
                    breadth = sgc[0] / sgc[1] if sgc[1] > 0 else 0
                    active = detect_setups(before, scan_price, prev_close,
                                           sector_breadth=breadth,
                                           prior_daily=dl, iwm_bars=iwm_before,
                                           scan_minute_et=scan_minute_et)
                    setup_flags = {f"setup_{s}": int(bool(active[s])) for s in SETUP_NAMES}

                    # v26: compute 8 target labels per example
                    # 2 target pcts × 4 horizons
                    labels = {}
                    for target_pct in [0.0075, 0.01]:
                        for horizon_min in [30, 60, 120, 180]:
                            hit = did_hit_target_within_horizon(
                                scan_price, after[1:], target_pct=target_pct,
                                horizon_minutes=horizon_min,
                                scan_minute_et=scan_minute_et,
                            )
                            key = f"t{int(target_pct*10000)}_h{horizon_min}"  # e.g. "t75_h30" or "t100_h60"
                            labels[key] = 1 if hit else 0

                    date_hour_rows.append({
                        "date": date, "hour": scan_hour, "ticker": ticker, "sector": sec,
                        "feat": feat, "setup_flags": setup_flags,
                        "labels": labels,
                    })

                if len(date_hour_rows) < 10: continue
                # Add cross-sectional ranks + sector-relative features
                feats_list = [r["feat"] for r in date_hour_rows]
                sectors_list = [r["sector"] for r in date_hour_rows]
                add_ranks(feats_list)
                add_sector_relative(feats_list, sectors_list)

                # Emit examples
                fold = "train" if date in train_dates else ("val" if date in val_dates else "test")
                for r in date_hour_rows:
                    examples.append({
                        "fold": fold, "date": date, "hour": scan_hour, "ticker": r["ticker"],
                        "feat": r["feat"], "setup_flags": r["setup_flags"],
                        "labels": r["labels"],  # v26: dict of 8 labels
                    })

        log.info(f"Built {len(examples)} examples total")
        conviction_train_progress = {"phase":"vectorize","pct":60,
            "message":f"Vectorizing {len(examples)} examples..."}

        # Build feature matrix ONCE (shared across all 8 target models)
        base_feat_names = FEATURE_NAMES
        setup_feat_names = [f"setup_{s}" for s in SETUP_NAMES]
        all_feat_names = base_feat_names + setup_feat_names + ["hour_11","hour_12","hour_13","hour_14","hour_15"]

        def row_to_vec(ex):
            f = ex["feat"]
            v = [float(f.get(k, 0.0) or 0.0) for k in base_feat_names]
            v += [float(ex["setup_flags"].get(k, 0)) for k in setup_feat_names]
            # Hour one-hot
            v += [1.0 if ex["hour"] == h else 0.0 for h in [11,12,13,14,15]]
            return v

        # Partition by fold once
        X_train_list, labels_train_dict = [], defaultdict(list)
        X_val_list, labels_val_dict = [], defaultdict(list)
        X_test_list, labels_test_dict = [], defaultdict(list)
        meta_test = []
        target_keys = [f"t{int(tp*10000)}_h{h}" for tp in [0.0075, 0.01] for h in [30, 60, 120, 180]]
        log.info(f"v26: training 8 target models: {target_keys}")

        for ex in examples:
            vec = row_to_vec(ex)
            bucket_map = {
                "train": (X_train_list, labels_train_dict),
                "val": (X_val_list, labels_val_dict),
                "test": (X_test_list, labels_test_dict),
            }
            xlist, ylist = bucket_map[ex["fold"]]
            xlist.append(vec)
            for k in target_keys:
                ylist[k].append(ex["labels"].get(k, 0))
            if ex["fold"] == "test":
                meta_test.append({"date": ex["date"], "hour": ex["hour"], "ticker": ex["ticker"]})

        X_train = np.array(X_train_list, dtype=np.float32)
        X_val = np.array(X_val_list, dtype=np.float32)
        X_test = np.array(X_test_list, dtype=np.float32)
        log.info(f"Matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        if len(X_train) < 1000 or len(X_val) < 200:
            raise RuntimeError(f"Insufficient examples: train={len(X_train)}, val={len(X_val)}")

        # Shared LightGBM params
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.85,
            "bagging_freq": 5,
            "lambda_l2": 1.0,
            "verbose": -1,
        }

        # Train all 8 target models
        from sklearn.metrics import roc_auc_score as _auc
        buckets_def = [
            (0.0, 0.3, "0.0-0.3"),
            (0.3, 0.4, "0.3-0.4"),
            (0.4, 0.5, "0.4-0.5"),
            (0.5, 0.6, "0.5-0.6"),
            (0.6, 0.7, "0.6-0.7"),
            (0.7, 0.8, "0.7-0.8"),
            (0.8, 0.9, "0.8-0.9"),
            (0.9, 1.01, "0.9+"),
        ]
        all_target_results = {}

        for ti, tkey in enumerate(target_keys):
            pct_done = 65 + int((ti / len(target_keys)) * 32)
            conviction_train_progress = {"phase":"training","pct":pct_done,
                "message":f"Training target {ti+1}/{len(target_keys)}: {tkey}"}

            y_train = np.array(labels_train_dict[tkey], dtype=np.int32)
            y_val = np.array(labels_val_dict[tkey], dtype=np.int32)
            y_test = np.array(labels_test_dict[tkey], dtype=np.int32)

            # Train
            train_ds = lgb.Dataset(X_train, label=y_train, feature_name=all_feat_names)
            val_ds = lgb.Dataset(X_val, label=y_val, feature_name=all_feat_names, reference=train_ds)
            try:
                model = lgb.train(
                    params, train_ds,
                    num_boost_round=500,
                    valid_sets=[val_ds], valid_names=["val"],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
            except Exception as e:
                log.error(f"Training {tkey} failed: {e}")
                all_target_results[tkey] = {"error": str(e)}
                continue

            # Calibrate on val
            raw_val = model.predict(X_val, num_iteration=model.best_iteration)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_val, y_val)

            # Predict test
            raw_test = model.predict(X_test, num_iteration=model.best_iteration)
            calib_test = calibrator.transform(raw_test)

            try:
                auc_test = round(_auc(y_test, raw_test), 4)
            except Exception:
                auc_test = None

            # Buckets
            bucket_stats = []
            for lo, hi, label in buckets_def:
                mask = (calib_test >= lo) & (calib_test < hi)
                n = int(mask.sum())
                if n == 0:
                    bucket_stats.append({
                        "bucket": label, "n": 0, "n_hits": 0,
                        "hit_rate_pct": None, "ci_lower_pct": None, "ci_upper_pct": None,
                        "mean_predicted_prob": None,
                    })
                    continue
                hits = int(y_test[mask].sum())
                lower, upper = wilson_ci(hits, n)
                bucket_stats.append({
                    "bucket": label, "n": n, "n_hits": hits,
                    "hit_rate_pct": round(hits / n * 100, 2),
                    "ci_lower_pct": round(lower * 100, 2),
                    "ci_upper_pct": round(upper * 100, 2),
                    "mean_predicted_prob": round(float(calib_test[mask].mean()), 3),
                })

            # Verdict
            pass_buckets = []
            for b in bucket_stats:
                if b["bucket"] in ("0.8-0.9", "0.9+") and b["n"] and b["n"] >= 30 and b["ci_lower_pct"] is not None and b["ci_lower_pct"] >= 75:
                    pass_buckets.append(b)
            verdict = "ACHIEVES_80_HIGH_CONFIDENCE" if pass_buckets else "DOES_NOT_ACHIEVE_80"

            # Feature importance
            importance_raw = model.feature_importance(importance_type="gain")
            feat_imp = sorted(
                [(name, float(imp)) for name, imp in zip(all_feat_names, importance_raw)],
                key=lambda x: -x[1]
            )
            total_imp = sum(i for _, i in feat_imp) or 1
            feat_imp_pct = [(n, round(i / total_imp * 100, 2)) for n, i in feat_imp]

            # Parse target key back to human labels
            # e.g. "t75_h30" → +0.75% within 30 min
            tp_code, h_code = tkey.split("_")
            target_pct_pretty = f"+{int(tp_code[1:])/100:.2f}%"  # 75 → 0.75
            horizon_pretty = f"{h_code[1:]}min"

            all_target_results[tkey] = {
                "target_pct": target_pct_pretty,
                "horizon": horizon_pretty,
                "base_rates": {
                    "train": round(float(y_train.mean()) * 100, 2),
                    "val": round(float(y_val.mean()) * 100, 2),
                    "test": round(float(y_test.mean()) * 100, 2),
                },
                "auc_test": auc_test,
                "buckets": bucket_stats,
                "verdict": verdict,
                "top_features": feat_imp_pct[:15],
                "best_iteration": model.best_iteration,
            }
            log.info(f"{tkey} ({target_pct_pretty} in {horizon_pretty}): AUC={auc_test}, verdict={verdict}")

        # Summary
        passing = [k for k, r in all_target_results.items() if r.get("verdict") == "ACHIEVES_80_HIGH_CONFIDENCE"]
        overall_verdict = "ACHIEVES_80_HIGH_CONFIDENCE" if passing else "DOES_NOT_ACHIEVE_80"

        results = {
            "generated_at": datetime.now(ET).isoformat(),
            "fold_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            "n_features": len(all_feat_names),
            "target_keys": target_keys,
            "overall_verdict": overall_verdict,
            "passing_targets": passing,
            "per_target": all_target_results,
        }
        CONVICTION_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

        log.info(f"v26 multi-target training complete. Passing: {passing}")
        conviction_train_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(passing)} of {len(target_keys)} targets passed 80% bar."}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"Conviction training failed: {e}\n{tb}", exc_info=True)
        conviction_train_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        conviction_train_in_progress = False


# ═══════════════════════════════════════════════════════════════════
# v27: PATTERN DISCOVERY — winner profile per scan hour × target
# For each (scan_hour, target_pct):
#   1. Build feature vectors (shared with v25/v26)
#   2. Label each scan: hit = did stock reach +target_pct before 15:55?
#   3. On training fold, compare winner vs loser feature distributions
#   4. Induce a shallow decision tree, extract high-purity leaves
#   5. Validate rules on test fold (n≥30, hit≥base+40pp, CI_lower≥base+30pp)
#   6. Output narrative + distribution comparison + validated rules
# ═══════════════════════════════════════════════════════════════════
PATTERN_RESULTS_PATH = DATA_DIR / "pattern_discovery.json"
pattern_discovery_in_progress = False
pattern_discovery_progress = {"phase":"idle","pct":0,"message":""}

def cohens_d(a, b):
    """Standardized mean difference. a and b are numpy arrays."""
    import numpy as _np
    if len(a) < 2 or len(b) < 2: return None
    na, nb = len(a), len(b)
    sa, sb = _np.std(a, ddof=1), _np.std(b, ddof=1)
    if na + nb - 2 <= 0: return None
    sp2 = ((na-1)*sa*sa + (nb-1)*sb*sb) / (na + nb - 2)
    if sp2 <= 0: return None
    return float((_np.mean(a) - _np.mean(b)) / (sp2 ** 0.5))

def run_pattern_discovery():
    """v27: Discover winner profile for +0.75% AND +1% targets, per scan hour, strict validation."""
    global pattern_discovery_in_progress, pattern_discovery_progress
    if pattern_discovery_in_progress:
        log.warning("pattern discovery already running")
        return
    pattern_discovery_in_progress = True
    pattern_discovery_progress = {"phase":"starting","pct":0,"message":"Starting pattern discovery..."}

    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No bar cache. Run Training first.")

        pattern_discovery_progress = {"phase":"loading","pct":2,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        iwm_daily = daily_bars.get("IWM", [])
        if len(iwm_daily) < 200:
            raise RuntimeError(f"Insufficient IWM daily history")
        all_dates_list = sorted(set(b["t"][:10] for b in iwm_daily))
        n_dates = len(all_dates_list)
        split_tr = int(n_dates * 0.60)
        split_va = int(n_dates * 0.80)
        train_dates = set(all_dates_list[:split_tr])
        val_dates = set(all_dates_list[split_tr:split_va])
        test_dates = set(all_dates_list[split_va:])
        log.info(f"Pattern discovery: {len(train_dates)}/{len(val_dates)}/{len(test_dates)} train/val/test dates")

        def bars_for_date(ticker, date):
            return [b for b in intraday_bars.get(ticker, []) if b["t"][:10] == date]
        def daily_up_to(ticker, date):
            return [b for b in daily_bars.get(ticker, []) if b["t"][:10] < date]

        # Feature schema — shared with v25/v26
        base_feat_names = FEATURE_NAMES
        setup_feat_names = [f"setup_{s}" for s in SETUP_NAMES]
        all_feat_names = base_feat_names + setup_feat_names  # no hour one-hots since we analyze per hour

        tickers = [t for t in TICKERS if t not in ("SPY", "IWM")]
        scan_hours = SCAN_HOURS  # [11,12,13,14,15]

        # Build examples PER SCAN HOUR — dict of hour → {train: [(feat_vec, labels)], val: ..., test: ...}
        # labels is a dict with both target outcomes
        examples_per_hour = {h: {"train": [], "val": [], "test": []} for h in scan_hours}

        for di, date in enumerate(all_dates_list):
            if (di + 1) % 10 == 0:
                pct = 5 + int((di / n_dates) * 55)
                pattern_discovery_progress = {"phase":"features","pct":pct,
                    "message":f"Building features for date {di+1}/{n_dates}..."}
            fold = "train" if date in train_dates else ("val" if date in val_dates else "test")

            spy_intraday_date = [b for b in intraday_bars.get("SPY", []) if b["t"][:10] == date]
            iwm_intraday_date = [b for b in intraday_bars.get("IWM", []) if b["t"][:10] == date]

            for scan_hour in scan_hours:
                scan_minute_et = scan_hour * 60
                spy_before = [b for b in spy_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                iwm_before = [b for b in iwm_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                spy_ctx = compute_spy_context(spy_before) if spy_before else None

                # Compute sector breadth
                sector_green_count = defaultdict(lambda: [0, 0])
                cache_per_tkr = {}
                for tkr in tickers:
                    tbars = bars_for_date(tkr, date)
                    if len(tbars) < 6: continue
                    tb = [b for b in tbars if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                    ta = [b for b in tbars if (bar_to_et_minutes(b) or -1) >= scan_minute_et]
                    if len(tb) < 6 or len(ta) < 2: continue
                    dl = daily_up_to(tkr, date)
                    pc = dl[-1]["c"] if dl else None
                    if pc is None: continue
                    sp = tb[-1]["c"]
                    sec = SECTORS.get(tkr, "?")
                    sector_green_count[sec][1] += 1
                    if sp > pc: sector_green_count[sec][0] += 1
                    cache_per_tkr[tkr] = (tb, ta, sp, pc, dl, sec)

                date_hour_rows = []
                for ticker, (before, after, scan_price, prev_close, dl, sec) in cache_per_tkr.items():
                    open_price = before[0]["o"]
                    feat = compute_features(before, dl, scan_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue
                    sgc = sector_green_count[sec]
                    breadth = sgc[0] / sgc[1] if sgc[1] > 0 else 0
                    active = detect_setups(before, scan_price, prev_close,
                                           sector_breadth=breadth,
                                           prior_daily=dl, iwm_bars=iwm_before,
                                           scan_minute_et=scan_minute_et)
                    setup_flags = {f"setup_{s}": int(bool(active[s])) for s in SETUP_NAMES}

                    # Both target labels
                    hit_75, _ = did_hit_target(scan_price, after[1:], target_pct=0.0075)
                    hit_100, _ = did_hit_target(scan_price, after[1:], target_pct=0.01)

                    date_hour_rows.append({
                        "feat": feat, "setup_flags": setup_flags, "sector": sec,
                        "label_075": 1 if hit_75 else 0, "label_100": 1 if hit_100 else 0,
                    })

                if len(date_hour_rows) < 10: continue
                feats_list = [r["feat"] for r in date_hour_rows]
                sectors_list = [r["sector"] for r in date_hour_rows]
                add_ranks(feats_list)
                add_sector_relative(feats_list, sectors_list)

                for r in date_hour_rows:
                    vec = [float(r["feat"].get(k, 0.0) or 0.0) for k in base_feat_names]
                    vec += [float(r["setup_flags"].get(k, 0)) for k in setup_feat_names]
                    examples_per_hour[scan_hour][fold].append({
                        "vec": vec, "label_075": r["label_075"], "label_100": r["label_100"],
                    })

        log.info(f"Examples per hour: " + ", ".join(
            f"{h}={len(examples_per_hour[h]['train'])+len(examples_per_hour[h]['val'])+len(examples_per_hour[h]['test'])}"
            for h in scan_hours))

        # Now run discovery + validation per (hour, target)
        import numpy as _np
        try:
            from sklearn.tree import DecisionTreeClassifier, _tree
        except Exception as e:
            raise RuntimeError(f"sklearn.tree not available: {e}")

        pattern_discovery_progress = {"phase":"discovery","pct":65,"message":"Discovering patterns..."}

        def extract_rules_from_tree(tree, feat_names, min_leaf_n, min_hit_rate):
            """Walk a decision tree; for each leaf with n≥min_leaf_n and hit_rate≥min_hit_rate,
            return the list of conditions from root to leaf."""
            t = tree.tree_
            FEAT_UNDEFINED = _tree.TREE_UNDEFINED
            rules = []
            def walk(node_id, conditions):
                if t.feature[node_id] != FEAT_UNDEFINED:
                    fname = feat_names[t.feature[node_id]]
                    thresh = float(t.threshold[node_id])
                    walk(t.children_left[node_id], conditions + [(fname, "<=", thresh)])
                    walk(t.children_right[node_id], conditions + [(fname, ">", thresh)])
                else:
                    # Leaf
                    # values shape: (1, n_classes) for sklearn tree classifier
                    vals = t.value[node_id][0]
                    n_total = int(vals.sum())
                    n_pos = int(vals[1]) if len(vals) > 1 else 0
                    if n_total >= min_leaf_n:
                        hr = n_pos / n_total
                        if hr >= min_hit_rate:
                            rules.append({
                                "conditions": conditions,
                                "train_n": n_total, "train_hits": n_pos,
                                "train_hit_rate": hr,
                            })
            walk(0, [])
            return rules

        def evaluate_rule_on_set(rule, vecs, labels, feat_idx):
            """Apply rule conditions to vectors; return (n_match, n_hits)."""
            mask = _np.ones(len(vecs), dtype=bool)
            for (fname, op, thresh) in rule["conditions"]:
                col = vecs[:, feat_idx[fname]]
                if op == "<=":
                    mask &= (col <= thresh)
                else:
                    mask &= (col > thresh)
            n_match = int(mask.sum())
            n_hits = int(labels[mask].sum()) if n_match > 0 else 0
            return n_match, n_hits

        feat_idx = {n: i for i, n in enumerate(all_feat_names)}
        targets = [("0.75%", "label_075", 0.0075), ("1.00%", "label_100", 0.01)]
        per_hour_target_results = {}

        for hi, h in enumerate(scan_hours):
            hour_ex = examples_per_hour[h]
            if not hour_ex["train"] or not hour_ex["test"]:
                continue
            X_train = _np.array([e["vec"] for e in hour_ex["train"]], dtype=_np.float32)
            X_val = _np.array([e["vec"] for e in hour_ex["val"]], dtype=_np.float32) if hour_ex["val"] else _np.zeros((0, X_train.shape[1]))
            X_test = _np.array([e["vec"] for e in hour_ex["test"]], dtype=_np.float32)

            per_hour_target_results[h] = {}

            for ti, (tlabel, label_key, target_pct) in enumerate(targets):
                progress_pct = 70 + int(((hi * len(targets) + ti) / (len(scan_hours) * len(targets))) * 25)
                pattern_discovery_progress = {"phase":"discovery","pct":progress_pct,
                    "message":f"Analyzing {h}:00 × {tlabel}..."}

                y_train = _np.array([e[label_key] for e in hour_ex["train"]], dtype=_np.int32)
                y_val = _np.array([e[label_key] for e in hour_ex["val"]], dtype=_np.int32) if hour_ex["val"] else _np.array([], dtype=_np.int32)
                y_test = _np.array([e[label_key] for e in hour_ex["test"]], dtype=_np.int32)

                train_base = float(y_train.mean())
                val_base = float(y_val.mean()) if len(y_val) > 0 else None
                test_base = float(y_test.mean())
                log.info(f"{h}:00 × {tlabel}: n_train={len(y_train)}, base_train={train_base:.3f}, base_test={test_base:.3f}")

                # Winner vs loser distribution comparison (on train fold)
                win_mask = (y_train == 1)
                los_mask = (y_train == 0)
                distribution = []
                for fn in all_feat_names:
                    col = X_train[:, feat_idx[fn]]
                    win_vals = col[win_mask]
                    los_vals = col[los_mask]
                    if len(win_vals) == 0 or len(los_vals) == 0:
                        continue
                    w_mean = float(_np.mean(win_vals))
                    l_mean = float(_np.mean(los_vals))
                    d = cohens_d(win_vals, los_vals)
                    distribution.append({
                        "feature": fn,
                        "winner_mean": round(w_mean, 4),
                        "loser_mean": round(l_mean, 4),
                        "cohens_d": round(d, 3) if d is not None else None,
                    })
                # Sort by |d| desc
                distribution.sort(key=lambda x: -abs(x["cohens_d"] or 0))

                # Strict validation thresholds (PER YOUR SPEC):
                # test fold n≥30, hit rate ≥ baseline + 40pp, CI_lower ≥ baseline + 30pp
                min_test_hr = test_base + 0.40
                min_test_ci_lower = test_base + 0.30

                # Discovery: decision tree
                # Depth capped at 4 → rules of ≤4 conditions (your "strict").
                # min_samples_leaf = 300 on train to avoid rules too small to validate.
                clf = DecisionTreeClassifier(
                    max_depth=4, min_samples_leaf=300, criterion="gini", random_state=42,
                )
                clf.fit(X_train, y_train)

                # Extract candidate rules from leaves with train hit_rate ≥ baseline + 40pp
                # (matching the eventual test threshold)
                min_leaf_n_train = 300
                min_train_hit_rate = train_base + 0.40
                candidates = extract_rules_from_tree(
                    clf, all_feat_names, min_leaf_n_train, min_train_hit_rate
                )

                # Validate each candidate on test fold
                validated = []
                for cand in candidates:
                    n_match, n_hits = evaluate_rule_on_set(cand, X_test, y_test, feat_idx)
                    if n_match < 30: continue
                    test_hr = n_hits / n_match
                    lower, upper = wilson_ci(n_hits, n_match)
                    passes = test_hr >= min_test_hr and lower >= min_test_ci_lower
                    record = {
                        "conditions": cand["conditions"],
                        "train_n": cand["train_n"],
                        "train_hit_rate": round(cand["train_hit_rate"], 4),
                        "test_n": n_match,
                        "test_hits": n_hits,
                        "test_hit_rate": round(test_hr, 4),
                        "test_ci_lower": round(lower, 4),
                        "test_ci_upper": round(upper, 4),
                        "passes_strict": passes,
                    }
                    validated.append(record)

                validated.sort(key=lambda r: -r["test_hit_rate"])

                per_hour_target_results[h][tlabel] = {
                    "hour": h,
                    "target": tlabel,
                    "n_train": len(y_train),
                    "n_val": len(y_val),
                    "n_test": len(y_test),
                    "base_rate_train": round(train_base * 100, 2),
                    "base_rate_val": round(val_base * 100, 2) if val_base is not None else None,
                    "base_rate_test": round(test_base * 100, 2),
                    "min_test_hit_rate_required_pct": round(min_test_hr * 100, 2),
                    "min_test_ci_lower_required_pct": round(min_test_ci_lower * 100, 2),
                    "n_candidates": len(candidates),
                    "n_validated": len(validated),
                    "n_passing_strict": sum(1 for r in validated if r["passes_strict"]),
                    "top_features": distribution[:20],
                    "validated_rules": validated[:30],  # top 30 by test hit rate
                }

        # Overall verdict
        passing_count = sum(
            per_hour_target_results.get(h, {}).get(tlabel, {}).get("n_passing_strict", 0)
            for h in scan_hours for _, tlabel, _ in [(None, "0.75%", None), (None, "1.00%", None)]
        )

        results = {
            "generated_at": datetime.now(ET).isoformat(),
            "config": {
                "targets": ["0.75%", "1.00%"],
                "scan_hours": scan_hours,
                "tree_max_depth": 4,
                "min_samples_leaf_train": 300,
                "strict_test_threshold_pp_above_base": 40,
                "strict_test_ci_lower_pp_above_base": 30,
            },
            "total_passing_strict_rules": passing_count,
            "per_hour_target": {str(h): per_hour_target_results.get(h, {}) for h in scan_hours},
        }
        PATTERN_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

        log.info(f"Pattern discovery complete. {passing_count} rules pass strict validation.")
        pattern_discovery_progress = {"phase":"done","pct":100,
            "message":f"Done. {passing_count} rules pass strict threshold."}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"Pattern discovery failed: {e}\n{tb}", exc_info=True)
        pattern_discovery_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        pattern_discovery_in_progress = False


# ═══════════════════════════════════════════════════════════════════
# v28: COST-ADJUSTED ANALYSIS
# Runs BOTH the v25-style LightGBM calibration AND the v27-style pattern
# discovery across three targets: +0.30%, +0.40%, +0.50%. These targets
# are derived from round-trip cost estimates (commission 0% on Alpaca,
# plus bid-ask spread and slippage) — represent the minimum price move
# needed for a profitable round-trip.
#
# Target 1 (0.30%): Best-case (tight spreads, good fills)
# Target 2 (0.40%): Typical R2K round-trip cost
# Target 3 (0.50%): Conservative (wider spreads, some slippage)
#
# For each target: one LightGBM calibrated model + one decision-tree
# pattern discovery analysis per scan hour. Same strict validation bars
# as v25/v26 and v27.
# ═══════════════════════════════════════════════════════════════════
V28_RESULTS_PATH = DATA_DIR / "v28_cost_adjusted_results.json"
v28_in_progress = False
v28_progress = {"phase":"idle","pct":0,"message":""}

def run_v28_cost_adjusted():
    """Run LightGBM + pattern discovery across 3 cost-adjusted targets."""
    global v28_in_progress, v28_progress
    if v28_in_progress:
        log.warning("v28 already running"); return
    v28_in_progress = True
    v28_progress = {"phase":"starting","pct":0,"message":"Starting v28 cost-adjusted analysis..."}

    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No bar cache. Run Training first.")

        import numpy as _np
        try:
            from sklearn.tree import DecisionTreeClassifier, _tree
        except Exception as e:
            raise RuntimeError(f"sklearn.tree not available: {e}")
        from sklearn.metrics import roc_auc_score as _auc

        v28_progress = {"phase":"loading","pct":2,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        iwm_daily = daily_bars.get("IWM", [])
        if len(iwm_daily) < 200:
            raise RuntimeError("Insufficient IWM history")
        all_dates_list = sorted(set(b["t"][:10] for b in iwm_daily))
        n_dates = len(all_dates_list)
        split_tr = int(n_dates * 0.60)
        split_va = int(n_dates * 0.80)
        train_dates = set(all_dates_list[:split_tr])
        val_dates = set(all_dates_list[split_tr:split_va])
        test_dates = set(all_dates_list[split_va:])
        log.info(f"v28: {len(train_dates)}/{len(val_dates)}/{len(test_dates)} train/val/test dates")

        def bars_for_date(ticker, date):
            return [b for b in intraday_bars.get(ticker, []) if b["t"][:10] == date]
        def daily_up_to(ticker, date):
            return [b for b in daily_bars.get(ticker, []) if b["t"][:10] < date]

        base_feat_names = FEATURE_NAMES
        setup_feat_names = [f"setup_{s}" for s in SETUP_NAMES]
        all_feat_names_hourless = base_feat_names + setup_feat_names
        all_feat_names_lgbm = base_feat_names + setup_feat_names + ["hour_11","hour_12","hour_13","hour_14","hour_15"]

        tickers_list = [t for t in TICKERS if t not in ("SPY", "IWM")]
        scan_hours = SCAN_HOURS
        TARGET_PCTS = [0.0030, 0.0040, 0.0050]  # cost-adjusted thresholds
        TARGET_LABELS = ["0.30%", "0.40%", "0.50%"]

        # Build shared feature cache (one pass, used by BOTH frameworks)
        v28_progress = {"phase":"features","pct":5,"message":"Building features..."}
        # Per-hour separation used by pattern discovery; flat list used by LightGBM
        examples_per_hour = {h: {"train": [], "val": [], "test": []} for h in scan_hours}
        all_examples = []  # with fold, hour attached

        for di, date in enumerate(all_dates_list):
            if (di + 1) % 10 == 0:
                pct = 5 + int((di / n_dates) * 45)
                v28_progress = {"phase":"features","pct":pct,
                    "message":f"Building features for date {di+1}/{n_dates}..."}
            fold = "train" if date in train_dates else ("val" if date in val_dates else "test")

            spy_intraday_date = [b for b in intraday_bars.get("SPY", []) if b["t"][:10] == date]
            iwm_intraday_date = [b for b in intraday_bars.get("IWM", []) if b["t"][:10] == date]

            for scan_hour in scan_hours:
                scan_minute_et = scan_hour * 60
                spy_before = [b for b in spy_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                iwm_before = [b for b in iwm_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                spy_ctx = compute_spy_context(spy_before) if spy_before else None

                sector_green_count = defaultdict(lambda: [0, 0])
                cache_per_tkr = {}
                for tkr in tickers_list:
                    tbars = bars_for_date(tkr, date)
                    if len(tbars) < 6: continue
                    tb = [b for b in tbars if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                    ta = [b for b in tbars if (bar_to_et_minutes(b) or -1) >= scan_minute_et]
                    if len(tb) < 6 or len(ta) < 2: continue
                    dl = daily_up_to(tkr, date)
                    pc = dl[-1]["c"] if dl else None
                    if pc is None: continue
                    sp = tb[-1]["c"]
                    sec = SECTORS.get(tkr, "?")
                    sector_green_count[sec][1] += 1
                    if sp > pc: sector_green_count[sec][0] += 1
                    cache_per_tkr[tkr] = (tb, ta, sp, pc, dl, sec)

                date_hour_rows = []
                for ticker, (before, after, scan_price, prev_close, dl, sec) in cache_per_tkr.items():
                    open_price = before[0]["o"]
                    feat = compute_features(before, dl, scan_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue
                    sgc = sector_green_count[sec]
                    breadth = sgc[0] / sgc[1] if sgc[1] > 0 else 0
                    active = detect_setups(before, scan_price, prev_close,
                                           sector_breadth=breadth, prior_daily=dl,
                                           iwm_bars=iwm_before, scan_minute_et=scan_minute_et)
                    setup_flags = {f"setup_{s}": int(bool(active[s])) for s in SETUP_NAMES}

                    # Compute 3 target labels (by-close, no horizon bounding)
                    labels = {}
                    for tp in TARGET_PCTS:
                        hit, _ = did_hit_target(scan_price, after[1:], target_pct=tp)
                        labels[f"t{int(tp*10000)}"] = 1 if hit else 0

                    date_hour_rows.append({
                        "feat": feat, "setup_flags": setup_flags, "sector": sec, "labels": labels,
                    })

                if len(date_hour_rows) < 10: continue
                feats_list = [r["feat"] for r in date_hour_rows]
                sectors_list = [r["sector"] for r in date_hour_rows]
                add_ranks(feats_list)
                add_sector_relative(feats_list, sectors_list)

                for r in date_hour_rows:
                    vec_hourless = [float(r["feat"].get(k, 0.0) or 0.0) for k in base_feat_names]
                    vec_hourless += [float(r["setup_flags"].get(k, 0)) for k in setup_feat_names]
                    vec_lgbm = list(vec_hourless) + [1.0 if scan_hour == h else 0.0 for h in [11,12,13,14,15]]
                    rec = {
                        "vec_hourless": vec_hourless,
                        "vec_lgbm": vec_lgbm,
                        "labels": r["labels"],
                        "hour": scan_hour, "fold": fold,
                    }
                    examples_per_hour[scan_hour][fold].append(rec)
                    all_examples.append(rec)

        total_ex = len(all_examples)
        log.info(f"v28: built {total_ex} examples")

        if total_ex < 1000:
            raise RuntimeError(f"Insufficient examples: {total_ex}")

        # ═══════════════════════════════════════════════════════════════
        # FRAMEWORK A: LightGBM calibration per target
        # ═══════════════════════════════════════════════════════════════
        v28_progress = {"phase":"lightgbm","pct":52,"message":"Training LightGBM (3 targets)..."}

        # Build shared LightGBM matrices
        def fold_idx(fold):
            return _np.array([i for i, e in enumerate(all_examples) if e["fold"] == fold], dtype=_np.int32)
        train_idx = fold_idx("train"); val_idx = fold_idx("val"); test_idx = fold_idx("test")
        X_all_lgbm = _np.array([e["vec_lgbm"] for e in all_examples], dtype=_np.float32)
        X_train_lgbm = X_all_lgbm[train_idx]
        X_val_lgbm = X_all_lgbm[val_idx]
        X_test_lgbm = X_all_lgbm[test_idx]
        log.info(f"LightGBM shapes: train={X_train_lgbm.shape}, val={X_val_lgbm.shape}, test={X_test_lgbm.shape}")

        lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 1.0, "verbose": -1,
        }
        buckets_def = [
            (0.0, 0.3, "0.0-0.3"), (0.3, 0.4, "0.3-0.4"), (0.4, 0.5, "0.4-0.5"),
            (0.5, 0.6, "0.5-0.6"), (0.6, 0.7, "0.6-0.7"), (0.7, 0.8, "0.7-0.8"),
            (0.8, 0.9, "0.8-0.9"), (0.9, 1.01, "0.9+"),
        ]

        lgbm_per_target = {}
        for ti, (tp, tlabel) in enumerate(zip(TARGET_PCTS, TARGET_LABELS)):
            tkey = f"t{int(tp*10000)}"
            pct_done = 52 + int((ti / len(TARGET_PCTS)) * 18)
            v28_progress = {"phase":"lightgbm","pct":pct_done,
                "message":f"LightGBM {ti+1}/3: {tlabel}"}

            y_train = _np.array([all_examples[i]["labels"][tkey] for i in train_idx], dtype=_np.int32)
            y_val = _np.array([all_examples[i]["labels"][tkey] for i in val_idx], dtype=_np.int32)
            y_test = _np.array([all_examples[i]["labels"][tkey] for i in test_idx], dtype=_np.int32)

            train_ds = lgb.Dataset(X_train_lgbm, label=y_train, feature_name=all_feat_names_lgbm)
            val_ds = lgb.Dataset(X_val_lgbm, label=y_val, feature_name=all_feat_names_lgbm, reference=train_ds)
            try:
                model = lgb.train(
                    lgbm_params, train_ds, num_boost_round=500,
                    valid_sets=[val_ds], valid_names=["val"],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
            except Exception as e:
                log.error(f"LightGBM {tlabel} failed: {e}")
                lgbm_per_target[tlabel] = {"error": str(e)}
                continue

            raw_val = model.predict(X_val_lgbm, num_iteration=model.best_iteration)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_val, y_val)
            raw_test = model.predict(X_test_lgbm, num_iteration=model.best_iteration)
            calib_test = calibrator.transform(raw_test)
            try:
                auc_test = round(_auc(y_test, raw_test), 4)
            except Exception:
                auc_test = None

            bucket_stats = []
            for lo, hi, lbl in buckets_def:
                mask = (calib_test >= lo) & (calib_test < hi)
                n = int(mask.sum())
                if n == 0:
                    bucket_stats.append({"bucket": lbl, "n": 0, "n_hits": 0,
                        "hit_rate_pct": None, "ci_lower_pct": None, "ci_upper_pct": None,
                        "mean_predicted_prob": None})
                    continue
                hits = int(y_test[mask].sum())
                lower, upper = wilson_ci(hits, n)
                bucket_stats.append({
                    "bucket": lbl, "n": n, "n_hits": hits,
                    "hit_rate_pct": round(hits / n * 100, 2),
                    "ci_lower_pct": round(lower * 100, 2),
                    "ci_upper_pct": round(upper * 100, 2),
                    "mean_predicted_prob": round(float(calib_test[mask].mean()), 3),
                })

            pass_buckets = [b for b in bucket_stats
                            if b["bucket"] in ("0.8-0.9", "0.9+")
                            and b["n"] and b["n"] >= 30
                            and b["ci_lower_pct"] is not None and b["ci_lower_pct"] >= 75]
            verdict = "ACHIEVES_80_HIGH_CONFIDENCE" if pass_buckets else "DOES_NOT_ACHIEVE_80"

            imp_raw = model.feature_importance(importance_type="gain")
            feat_imp = sorted([(n, float(v)) for n, v in zip(all_feat_names_lgbm, imp_raw)],
                              key=lambda x: -x[1])
            total_imp = sum(v for _, v in feat_imp) or 1
            feat_imp_pct = [(n, round(v / total_imp * 100, 2)) for n, v in feat_imp]

            lgbm_per_target[tlabel] = {
                "target_pct": tlabel,
                "base_rates": {
                    "train": round(float(y_train.mean()) * 100, 2),
                    "val": round(float(y_val.mean()) * 100, 2),
                    "test": round(float(y_test.mean()) * 100, 2),
                },
                "auc_test": auc_test,
                "buckets": bucket_stats,
                "verdict": verdict,
                "top_features": feat_imp_pct[:15],
                "best_iteration": model.best_iteration,
            }
            log.info(f"LightGBM {tlabel}: AUC={auc_test}, verdict={verdict}")

        # ═══════════════════════════════════════════════════════════════
        # FRAMEWORK B: Decision tree pattern discovery per (hour × target)
        # ═══════════════════════════════════════════════════════════════
        v28_progress = {"phase":"pattern","pct":70,"message":"Pattern discovery..."}

        def extract_rules_from_tree(tree, feat_names, min_leaf_n, min_hit_rate):
            t = tree.tree_
            FEAT_UNDEFINED = _tree.TREE_UNDEFINED
            rules = []
            def walk(node_id, conditions):
                if t.feature[node_id] != FEAT_UNDEFINED:
                    fname = feat_names[t.feature[node_id]]
                    thresh = float(t.threshold[node_id])
                    walk(t.children_left[node_id], conditions + [(fname, "<=", thresh)])
                    walk(t.children_right[node_id], conditions + [(fname, ">", thresh)])
                else:
                    vals = t.value[node_id][0]
                    n_total = int(vals.sum())
                    n_pos = int(vals[1]) if len(vals) > 1 else 0
                    if n_total >= min_leaf_n:
                        hr = n_pos / n_total
                        if hr >= min_hit_rate:
                            rules.append({"conditions": conditions, "train_n": n_total,
                                          "train_hits": n_pos, "train_hit_rate": hr})
            walk(0, [])
            return rules

        def evaluate_rule_on_set(rule, vecs, labels, feat_idx):
            mask = _np.ones(len(vecs), dtype=bool)
            for (fname, op, thresh) in rule["conditions"]:
                col = vecs[:, feat_idx[fname]]
                mask &= (col <= thresh) if op == "<=" else (col > thresh)
            n_match = int(mask.sum())
            n_hits = int(labels[mask].sum()) if n_match > 0 else 0
            return n_match, n_hits

        feat_idx = {n: i for i, n in enumerate(all_feat_names_hourless)}
        pattern_per_target = {}  # tlabel → per_hour dict

        total_cells = len(scan_hours) * len(TARGET_PCTS)
        cell_i = 0
        for tp, tlabel in zip(TARGET_PCTS, TARGET_LABELS):
            tkey = f"t{int(tp*10000)}"
            pattern_per_target[tlabel] = {}

            for h in scan_hours:
                cell_i += 1
                pct_done = 70 + int((cell_i / total_cells) * 28)
                v28_progress = {"phase":"pattern","pct":pct_done,
                    "message":f"Pattern {tlabel} × {h}:00 ({cell_i}/{total_cells})"}

                hour_ex = examples_per_hour[h]
                if not hour_ex["train"] or not hour_ex["test"]:
                    continue
                X_tr = _np.array([e["vec_hourless"] for e in hour_ex["train"]], dtype=_np.float32)
                X_te = _np.array([e["vec_hourless"] for e in hour_ex["test"]], dtype=_np.float32)
                y_tr = _np.array([e["labels"][tkey] for e in hour_ex["train"]], dtype=_np.int32)
                y_te = _np.array([e["labels"][tkey] for e in hour_ex["test"]], dtype=_np.int32)
                train_base = float(y_tr.mean())
                test_base = float(y_te.mean())

                # Winner vs loser distribution
                win_mask = (y_tr == 1); los_mask = (y_tr == 0)
                distribution = []
                for fn in all_feat_names_hourless:
                    col = X_tr[:, feat_idx[fn]]
                    wv = col[win_mask]; lv = col[los_mask]
                    if len(wv) == 0 or len(lv) == 0: continue
                    d_val = cohens_d(wv, lv)
                    distribution.append({
                        "feature": fn,
                        "winner_mean": round(float(_np.mean(wv)), 4),
                        "loser_mean": round(float(_np.mean(lv)), 4),
                        "cohens_d": round(d_val, 3) if d_val is not None else None,
                    })
                distribution.sort(key=lambda x: -abs(x["cohens_d"] or 0))

                # Strict thresholds
                min_test_hr = test_base + 0.40
                min_test_ci_lower = test_base + 0.30
                min_train_hr = train_base + 0.40

                clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=300,
                                             criterion="gini", random_state=42)
                clf.fit(X_tr, y_tr)
                candidates = extract_rules_from_tree(clf, all_feat_names_hourless, 300, min_train_hr)

                validated = []
                for cand in candidates:
                    n_match, n_hits = evaluate_rule_on_set(cand, X_te, y_te, feat_idx)
                    if n_match < 30: continue
                    test_hr = n_hits / n_match
                    lower, upper = wilson_ci(n_hits, n_match)
                    passes = test_hr >= min_test_hr and lower >= min_test_ci_lower
                    validated.append({
                        "conditions": cand["conditions"],
                        "train_n": cand["train_n"],
                        "train_hit_rate": round(cand["train_hit_rate"], 4),
                        "test_n": n_match, "test_hits": n_hits,
                        "test_hit_rate": round(test_hr, 4),
                        "test_ci_lower": round(lower, 4),
                        "test_ci_upper": round(upper, 4),
                        "passes_strict": passes,
                    })
                validated.sort(key=lambda r: -r["test_hit_rate"])

                pattern_per_target[tlabel][str(h)] = {
                    "hour": h,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "base_rate_train": round(train_base * 100, 2),
                    "base_rate_test": round(test_base * 100, 2),
                    "min_test_hit_rate_required_pct": round(min_test_hr * 100, 2),
                    "min_test_ci_lower_required_pct": round(min_test_ci_lower * 100, 2),
                    "n_candidates": len(candidates),
                    "n_validated": len(validated),
                    "n_passing_strict": sum(1 for r in validated if r["passes_strict"]),
                    "top_features": distribution[:15],
                    "validated_rules": validated[:20],
                }

        # Aggregate verdict
        total_passing_pattern = sum(
            cell.get("n_passing_strict", 0)
            for tlabel_data in pattern_per_target.values()
            for cell in tlabel_data.values()
        )
        total_passing_lgbm = sum(
            1 for t in lgbm_per_target.values()
            if t.get("verdict") == "ACHIEVES_80_HIGH_CONFIDENCE"
        )

        results = {
            "generated_at": datetime.now(ET).isoformat(),
            "config": {
                "targets_pct": [0.30, 0.40, 0.50],
                "scan_hours": scan_hours,
                "tree_max_depth": 4,
                "min_samples_leaf_train": 300,
                "lgbm_bucket_pass_n_min": 30,
                "lgbm_bucket_pass_ci_lower_min_pct": 75,
                "pattern_test_hit_threshold_pp_above_base": 40,
                "pattern_test_ci_lower_pp_above_base": 30,
            },
            "fold_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "n_features_lgbm": len(all_feat_names_lgbm),
            "n_features_pattern": len(all_feat_names_hourless),
            "lgbm_passing_targets": total_passing_lgbm,
            "pattern_passing_rules": total_passing_pattern,
            "lgbm_per_target": lgbm_per_target,
            "pattern_per_target": pattern_per_target,
        }
        V28_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

        log.info(f"v28 complete. LightGBM passing: {total_passing_lgbm}/3, Pattern passing: {total_passing_pattern}")
        v28_progress = {"phase":"done","pct":100,
            "message":f"Done. LightGBM {total_passing_lgbm}/3 targets pass; {total_passing_pattern} pattern rules pass."}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"v28 failed: {e}\n{tb}", exc_info=True)
        v28_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        v28_in_progress = False


# ═══════════════════════════════════════════════════════════════════
# v29: FINE-GRAINED TARGET SWEEP
# Tests every 1bps target from 0.31% to 0.40% using the same LightGBM
# classifier framework as v28. Goal: find the highest target that still
# passes the strict 80% bar (n≥30 in 0.8+ bucket with Wilson CI_lower ≥75%).
#
# v28 results: +0.30% passed (0.9+ bucket 86.96%, CI_lower 80.32%),
# +0.40% failed (0.8-0.9 bucket CI_lower 69.58%), so the failure
# point is between these values. This sweep pinpoints it.
# ═══════════════════════════════════════════════════════════════════
V29_RESULTS_PATH = DATA_DIR / "v29_target_sweep_results.json"
v29_in_progress = False
v29_progress = {"phase":"idle","pct":0,"message":""}

def run_v29_target_sweep():
    """Sweep 0.31% through 0.40% at 1bps increments, LightGBM classifier each."""
    global v29_in_progress, v29_progress
    if v29_in_progress:
        log.warning("v29 already running"); return
    v29_in_progress = True
    v29_progress = {"phase":"starting","pct":0,"message":"Starting v29 fine-grained target sweep..."}

    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No bar cache. Run Training first.")

        import numpy as _np
        from sklearn.metrics import roc_auc_score as _auc

        v29_progress = {"phase":"loading","pct":2,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        iwm_daily = daily_bars.get("IWM", [])
        if len(iwm_daily) < 200:
            raise RuntimeError("Insufficient IWM history")
        all_dates_list = sorted(set(b["t"][:10] for b in iwm_daily))
        n_dates = len(all_dates_list)
        split_tr = int(n_dates * 0.60)
        split_va = int(n_dates * 0.80)
        train_dates = set(all_dates_list[:split_tr])
        val_dates = set(all_dates_list[split_tr:split_va])
        test_dates = set(all_dates_list[split_va:])

        def bars_for_date(ticker, date):
            return [b for b in intraday_bars.get(ticker, []) if b["t"][:10] == date]
        def daily_up_to(ticker, date):
            return [b for b in daily_bars.get(ticker, []) if b["t"][:10] < date]

        base_feat_names = FEATURE_NAMES
        setup_feat_names = [f"setup_{s}" for s in SETUP_NAMES]
        all_feat_names_lgbm = base_feat_names + setup_feat_names + ["hour_11","hour_12","hour_13","hour_14","hour_15"]

        tickers_list = [t for t in TICKERS if t not in ("SPY", "IWM")]
        scan_hours = SCAN_HOURS
        # 10 targets: 0.31%, 0.32%, ..., 0.40%
        TARGET_PCTS = [round(0.0030 + i * 0.0001, 4) for i in range(1, 11)]
        TARGET_LABELS = [f"{tp*100:.2f}%" for tp in TARGET_PCTS]
        log.info(f"v29 sweep targets: {TARGET_LABELS}")

        v29_progress = {"phase":"features","pct":5,"message":"Building features (shared across all 10 targets)..."}
        all_examples = []  # list of {vec, labels_dict, fold}

        for di, date in enumerate(all_dates_list):
            if (di + 1) % 10 == 0:
                pct = 5 + int((di / n_dates) * 55)
                v29_progress = {"phase":"features","pct":pct,
                    "message":f"Building features for date {di+1}/{n_dates}..."}
            fold = "train" if date in train_dates else ("val" if date in val_dates else "test")

            spy_intraday_date = [b for b in intraday_bars.get("SPY", []) if b["t"][:10] == date]
            iwm_intraday_date = [b for b in intraday_bars.get("IWM", []) if b["t"][:10] == date]

            for scan_hour in scan_hours:
                scan_minute_et = scan_hour * 60
                spy_before = [b for b in spy_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                iwm_before = [b for b in iwm_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                spy_ctx = compute_spy_context(spy_before) if spy_before else None

                sector_green_count = defaultdict(lambda: [0, 0])
                cache_per_tkr = {}
                for tkr in tickers_list:
                    tbars = bars_for_date(tkr, date)
                    if len(tbars) < 6: continue
                    tb = [b for b in tbars if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                    ta = [b for b in tbars if (bar_to_et_minutes(b) or -1) >= scan_minute_et]
                    if len(tb) < 6 or len(ta) < 2: continue
                    dl = daily_up_to(tkr, date)
                    pc = dl[-1]["c"] if dl else None
                    if pc is None: continue
                    sp = tb[-1]["c"]
                    sec = SECTORS.get(tkr, "?")
                    sector_green_count[sec][1] += 1
                    if sp > pc: sector_green_count[sec][0] += 1
                    cache_per_tkr[tkr] = (tb, ta, sp, pc, dl, sec)

                date_hour_rows = []
                for ticker, (before, after, scan_price, prev_close, dl, sec) in cache_per_tkr.items():
                    open_price = before[0]["o"]
                    feat = compute_features(before, dl, scan_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue
                    sgc = sector_green_count[sec]
                    breadth = sgc[0] / sgc[1] if sgc[1] > 0 else 0
                    active = detect_setups(before, scan_price, prev_close,
                                           sector_breadth=breadth, prior_daily=dl,
                                           iwm_bars=iwm_before, scan_minute_et=scan_minute_et)
                    setup_flags = {f"setup_{s}": int(bool(active[s])) for s in SETUP_NAMES}

                    # Compute 10 target labels (by-close)
                    labels = {}
                    for tp in TARGET_PCTS:
                        hit, _ = did_hit_target(scan_price, after[1:], target_pct=tp)
                        labels[f"t{int(round(tp*10000))}"] = 1 if hit else 0

                    date_hour_rows.append({
                        "feat": feat, "setup_flags": setup_flags, "sector": sec,
                        "labels": labels, "hour": scan_hour,
                    })

                if len(date_hour_rows) < 10: continue
                feats_list = [r["feat"] for r in date_hour_rows]
                sectors_list = [r["sector"] for r in date_hour_rows]
                add_ranks(feats_list)
                add_sector_relative(feats_list, sectors_list)

                for r in date_hour_rows:
                    vec = [float(r["feat"].get(k, 0.0) or 0.0) for k in base_feat_names]
                    vec += [float(r["setup_flags"].get(k, 0)) for k in setup_feat_names]
                    vec += [1.0 if r["hour"] == h else 0.0 for h in [11,12,13,14,15]]
                    all_examples.append({"vec": vec, "labels": r["labels"], "fold": fold})

        total_ex = len(all_examples)
        log.info(f"v29: built {total_ex} examples")
        if total_ex < 1000:
            raise RuntimeError(f"Insufficient examples: {total_ex}")

        # Build shared matrices
        v29_progress = {"phase":"vectorize","pct":62,"message":"Building matrices..."}
        X_all = _np.array([e["vec"] for e in all_examples], dtype=_np.float32)
        fold_arr = _np.array([0 if e["fold"]=="train" else (1 if e["fold"]=="val" else 2) for e in all_examples])
        train_idx = _np.where(fold_arr == 0)[0]
        val_idx = _np.where(fold_arr == 1)[0]
        test_idx = _np.where(fold_arr == 2)[0]
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_all[test_idx]
        log.info(f"v29 shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 1.0, "verbose": -1,
        }
        buckets_def = [
            (0.0, 0.3, "0.0-0.3"), (0.3, 0.4, "0.3-0.4"), (0.4, 0.5, "0.4-0.5"),
            (0.5, 0.6, "0.5-0.6"), (0.6, 0.7, "0.6-0.7"), (0.7, 0.8, "0.7-0.8"),
            (0.8, 0.9, "0.8-0.9"), (0.9, 1.01, "0.9+"),
        ]

        per_target = {}
        for ti, (tp, tlabel) in enumerate(zip(TARGET_PCTS, TARGET_LABELS)):
            tkey = f"t{int(round(tp*10000))}"
            pct_done = 65 + int((ti / len(TARGET_PCTS)) * 32)
            v29_progress = {"phase":"training","pct":pct_done,
                "message":f"Training target {ti+1}/{len(TARGET_PCTS)}: +{tlabel}"}

            y_train = _np.array([all_examples[i]["labels"][tkey] for i in train_idx], dtype=_np.int32)
            y_val = _np.array([all_examples[i]["labels"][tkey] for i in val_idx], dtype=_np.int32)
            y_test = _np.array([all_examples[i]["labels"][tkey] for i in test_idx], dtype=_np.int32)

            train_ds = lgb.Dataset(X_train, label=y_train, feature_name=all_feat_names_lgbm)
            val_ds = lgb.Dataset(X_val, label=y_val, feature_name=all_feat_names_lgbm, reference=train_ds)
            try:
                model = lgb.train(
                    lgbm_params, train_ds, num_boost_round=500,
                    valid_sets=[val_ds], valid_names=["val"],
                    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
                )
            except Exception as e:
                log.error(f"v29 target {tlabel} failed: {e}")
                per_target[tlabel] = {"error": str(e)}
                continue

            raw_val = model.predict(X_val, num_iteration=model.best_iteration)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_val, y_val)
            raw_test = model.predict(X_test, num_iteration=model.best_iteration)
            calib_test = calibrator.transform(raw_test)
            try:
                auc_test = round(_auc(y_test, raw_test), 4)
            except Exception:
                auc_test = None

            bucket_stats = []
            for lo, hi, lbl in buckets_def:
                mask = (calib_test >= lo) & (calib_test < hi)
                n = int(mask.sum())
                if n == 0:
                    bucket_stats.append({"bucket": lbl, "n": 0, "n_hits": 0,
                        "hit_rate_pct": None, "ci_lower_pct": None, "ci_upper_pct": None,
                        "mean_predicted_prob": None})
                    continue
                hits = int(y_test[mask].sum())
                lower, upper = wilson_ci(hits, n)
                bucket_stats.append({
                    "bucket": lbl, "n": n, "n_hits": hits,
                    "hit_rate_pct": round(hits / n * 100, 2),
                    "ci_lower_pct": round(lower * 100, 2),
                    "ci_upper_pct": round(upper * 100, 2),
                    "mean_predicted_prob": round(float(calib_test[mask].mean()), 3),
                })

            pass_buckets = [b for b in bucket_stats
                            if b["bucket"] in ("0.8-0.9", "0.9+")
                            and b["n"] and b["n"] >= 30
                            and b["ci_lower_pct"] is not None and b["ci_lower_pct"] >= 75]
            verdict = "ACHIEVES_80_HIGH_CONFIDENCE" if pass_buckets else "DOES_NOT_ACHIEVE_80"

            per_target[tlabel] = {
                "target_pct": tlabel,
                "base_rates": {
                    "train": round(float(y_train.mean()) * 100, 2),
                    "val": round(float(y_val.mean()) * 100, 2),
                    "test": round(float(y_test.mean()) * 100, 2),
                },
                "auc_test": auc_test,
                "buckets": bucket_stats,
                "verdict": verdict,
                "best_iteration": model.best_iteration,
                "passing_buckets": [b["bucket"] for b in pass_buckets],
            }
            log.info(f"v29 {tlabel}: AUC={auc_test}, verdict={verdict}, passing={[b['bucket'] for b in pass_buckets]}")

        # Summary: find highest passing target
        passing_labels = [tl for tl in TARGET_LABELS
                          if per_target.get(tl, {}).get("verdict") == "ACHIEVES_80_HIGH_CONFIDENCE"]
        highest_passing = passing_labels[-1] if passing_labels else None

        results = {
            "generated_at": datetime.now(ET).isoformat(),
            "config": {
                "targets_tested": TARGET_LABELS,
                "target_pcts": TARGET_PCTS,
                "bucket_pass_n_min": 30,
                "bucket_pass_ci_lower_min_pct": 75,
                "note": "Sweep between v28's passing +0.30% and failing +0.40% to find breakpoint",
            },
            "fold_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "n_features": len(all_feat_names_lgbm),
            "n_passing_targets": len(passing_labels),
            "passing_targets": passing_labels,
            "highest_passing_target": highest_passing,
            "per_target": per_target,
        }
        V29_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

        log.info(f"v29 complete. Passing targets: {passing_labels}, highest: {highest_passing}")
        v29_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(passing_labels)}/10 targets pass. Highest: {highest_passing or 'none'}"}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"v29 failed: {e}\n{tb}", exc_info=True)
        v29_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        v29_in_progress = False


# ═══════════════════════════════════════════════════════════════════
# v30: CONFIDENT EXAMPLES DUMP
# Trains ONE LightGBM model at +0.32% target (highest-passing from v29),
# identifies test-fold examples where model predicted ≥0.9, and dumps
# 10 winners + 10 losers with full detail (features, price path, setup flags).
# Purpose: enable qualitative examination of confident predictions.
# ═══════════════════════════════════════════════════════════════════
V30_RESULTS_PATH = DATA_DIR / "v30_confident_examples.json"
v30_in_progress = False
v30_progress = {"phase":"idle","pct":0,"message":""}

def run_v30_examples_dump(target_pct=0.0032, n_winners=10, n_losers=10):
    """Train +0.32% model, identify prob≥0.9 test-fold examples, dump 10 winners + 10 losers with full detail."""
    global v30_in_progress, v30_progress
    if v30_in_progress:
        log.warning("v30 already running"); return
    v30_in_progress = True
    v30_progress = {"phase":"starting","pct":0,"message":"Starting v30 examples dump..."}

    try:
        if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
            raise RuntimeError("No bar cache. Run Training first.")

        import numpy as _np

        v30_progress = {"phase":"loading","pct":2,"message":"Loading bars..."}
        daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
        intraday_bars = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())

        iwm_daily = daily_bars.get("IWM", [])
        if len(iwm_daily) < 200:
            raise RuntimeError("Insufficient IWM history")
        all_dates_list = sorted(set(b["t"][:10] for b in iwm_daily))
        n_dates = len(all_dates_list)
        split_tr = int(n_dates * 0.60)
        split_va = int(n_dates * 0.80)
        train_dates = set(all_dates_list[:split_tr])
        val_dates = set(all_dates_list[split_tr:split_va])
        test_dates = set(all_dates_list[split_va:])

        def bars_for_date(ticker, date):
            return [b for b in intraday_bars.get(ticker, []) if b["t"][:10] == date]
        def daily_up_to(ticker, date):
            return [b for b in daily_bars.get(ticker, []) if b["t"][:10] < date]

        base_feat_names = FEATURE_NAMES
        setup_feat_names = [f"setup_{s}" for s in SETUP_NAMES]
        all_feat_names = base_feat_names + setup_feat_names + ["hour_11","hour_12","hour_13","hour_14","hour_15"]

        tickers_list = [t for t in TICKERS if t not in ("SPY", "IWM")]

        # Build examples — keep full metadata for test fold (ticker, date, hour, feature values, price path)
        v30_progress = {"phase":"features","pct":5,"message":"Building features..."}
        all_rows = []  # for all folds: vectors + labels
        test_meta = []  # for test fold only: rich detail

        for di, date in enumerate(all_dates_list):
            if (di + 1) % 10 == 0:
                pct = 5 + int((di / n_dates) * 65)
                v30_progress = {"phase":"features","pct":pct,
                    "message":f"Building features for date {di+1}/{n_dates}..."}
            fold = "train" if date in train_dates else ("val" if date in val_dates else "test")

            spy_intraday_date = [b for b in intraday_bars.get("SPY", []) if b["t"][:10] == date]
            iwm_intraday_date = [b for b in intraday_bars.get("IWM", []) if b["t"][:10] == date]

            for scan_hour in SCAN_HOURS:
                scan_minute_et = scan_hour * 60
                spy_before = [b for b in spy_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                iwm_before = [b for b in iwm_intraday_date if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                spy_ctx = compute_spy_context(spy_before) if spy_before else None

                sector_green_count = defaultdict(lambda: [0, 0])
                cache_per_tkr = {}
                for tkr in tickers_list:
                    tbars = bars_for_date(tkr, date)
                    if len(tbars) < 6: continue
                    tb = [b for b in tbars if (bar_to_et_minutes(b) or -1) < scan_minute_et]
                    ta = [b for b in tbars if (bar_to_et_minutes(b) or -1) >= scan_minute_et]
                    if len(tb) < 6 or len(ta) < 2: continue
                    dl = daily_up_to(tkr, date)
                    pc = dl[-1]["c"] if dl else None
                    if pc is None: continue
                    sp = tb[-1]["c"]
                    sec = SECTORS.get(tkr, "?")
                    sector_green_count[sec][1] += 1
                    if sp > pc: sector_green_count[sec][0] += 1
                    cache_per_tkr[tkr] = (tb, ta, sp, pc, dl, sec)

                date_hour_rows = []
                for ticker, (before, after, scan_price, prev_close, dl, sec) in cache_per_tkr.items():
                    open_price = before[0]["o"]
                    feat = compute_features(before, dl, scan_price, open_price, scan_hour,
                                            spy_context=spy_ctx, prev_close=prev_close)
                    if feat is None: continue
                    sgc = sector_green_count[sec]
                    breadth = sgc[0] / sgc[1] if sgc[1] > 0 else 0
                    active = detect_setups(before, scan_price, prev_close,
                                           sector_breadth=breadth, prior_daily=dl,
                                           iwm_bars=iwm_before, scan_minute_et=scan_minute_et)
                    setup_flags = {f"setup_{s}": int(bool(active[s])) for s in SETUP_NAMES}

                    hit, _ = did_hit_target(scan_price, after[1:], target_pct=target_pct)

                    date_hour_rows.append({
                        "feat": feat, "setup_flags": setup_flags, "sector": sec, "hour": scan_hour,
                        "label": 1 if hit else 0,
                        "ticker": ticker, "date": date, "scan_price": scan_price,
                        "prev_close": prev_close, "open_price": open_price,
                        "after_bars": after, "active_setups": [s for s in SETUP_NAMES if active[s]],
                    })

                if len(date_hour_rows) < 10: continue
                feats_list = [r["feat"] for r in date_hour_rows]
                sectors_list = [r["sector"] for r in date_hour_rows]
                add_ranks(feats_list)
                add_sector_relative(feats_list, sectors_list)

                for r in date_hour_rows:
                    vec = [float(r["feat"].get(k, 0.0) or 0.0) for k in base_feat_names]
                    vec += [float(r["setup_flags"].get(k, 0)) for k in setup_feat_names]
                    vec += [1.0 if r["hour"] == h else 0.0 for h in [11,12,13,14,15]]
                    all_rows.append({"vec": vec, "label": r["label"], "fold": fold})
                    if fold == "test":
                        test_meta.append({
                            "ticker": r["ticker"], "date": r["date"], "hour": r["hour"],
                            "sector": r["sector"],
                            "scan_price": round(r["scan_price"], 4),
                            "prev_close": round(r["prev_close"], 4),
                            "open_price": round(r["open_price"], 4),
                            "label": r["label"],
                            "active_setups": r["active_setups"],
                            "feat": {k: (round(v, 5) if isinstance(v, (int, float)) else v) for k, v in r["feat"].items()},
                            "price_path_post_scan": _build_price_path(r["after_bars"], r["scan_price"]),
                        })

        log.info(f"v30: built {len(all_rows)} total rows, {len(test_meta)} test meta")
        if len(all_rows) < 1000:
            raise RuntimeError(f"Insufficient examples: {len(all_rows)}")

        # Build matrices
        v30_progress = {"phase":"vectorize","pct":72,"message":"Building matrices..."}
        train_rows = [r for r in all_rows if r["fold"] == "train"]
        val_rows = [r for r in all_rows if r["fold"] == "val"]
        test_rows = [r for r in all_rows if r["fold"] == "test"]
        X_train = _np.array([r["vec"] for r in train_rows], dtype=_np.float32)
        X_val = _np.array([r["vec"] for r in val_rows], dtype=_np.float32)
        X_test = _np.array([r["vec"] for r in test_rows], dtype=_np.float32)
        y_train = _np.array([r["label"] for r in train_rows], dtype=_np.int32)
        y_val = _np.array([r["label"] for r in val_rows], dtype=_np.int32)
        y_test = _np.array([r["label"] for r in test_rows], dtype=_np.int32)

        # Train
        v30_progress = {"phase":"training","pct":80,"message":"Training LightGBM at target..."}
        lgbm_params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "bagging_fraction": 0.85, "bagging_freq": 5,
            "lambda_l2": 1.0, "verbose": -1,
        }
        train_ds = lgb.Dataset(X_train, label=y_train, feature_name=all_feat_names)
        val_ds = lgb.Dataset(X_val, label=y_val, feature_name=all_feat_names, reference=train_ds)
        model = lgb.train(lgbm_params, train_ds, num_boost_round=500,
                          valid_sets=[val_ds], valid_names=["val"],
                          callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        # Calibrate
        raw_val = model.predict(X_val, num_iteration=model.best_iteration)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_val, y_val)
        raw_test = model.predict(X_test, num_iteration=model.best_iteration)
        calib_test = calibrator.transform(raw_test)

        # Identify confident predictions (prob ≥ 0.9)
        v30_progress = {"phase":"selecting","pct":92,"message":"Selecting examples..."}
        confident_mask = calib_test >= 0.9
        n_confident = int(confident_mask.sum())

        # Attach predictions to test_meta
        for i, meta in enumerate(test_meta):
            meta["model_prob"] = round(float(calib_test[i]), 4)
            meta["is_confident"] = bool(calib_test[i] >= 0.9)

        # Split into confident winners and confident losers
        confident_winners = [m for m in test_meta if m["is_confident"] and m["label"] == 1]
        confident_losers = [m for m in test_meta if m["is_confident"] and m["label"] == 0]

        # Sort: winners by highest model_prob (most confident first), losers by highest model_prob (most confident first)
        confident_winners.sort(key=lambda m: -m["model_prob"])
        confident_losers.sort(key=lambda m: -m["model_prob"])

        # Take top N of each
        top_winners = confident_winners[:n_winners]
        top_losers = confident_losers[:n_losers]

        log.info(f"v30: {n_confident} confident predictions, {len(confident_winners)} winners, {len(confident_losers)} losers. Returning top {n_winners}W/{n_losers}L.")

        results = {
            "generated_at": datetime.now(ET).isoformat(),
            "target_pct": target_pct,
            "target_label": f"{target_pct*100:.2f}%",
            "model": {
                "n_features": len(all_feat_names),
                "feature_names": all_feat_names,
                "best_iteration": model.best_iteration,
                "fold_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
            },
            "confident_summary": {
                "total_confident_preds": n_confident,
                "n_confident_winners": len(confident_winners),
                "n_confident_losers": len(confident_losers),
                "observed_hit_rate_pct": round((len(confident_winners) / n_confident * 100) if n_confident > 0 else 0, 2),
            },
            "confident_winners": top_winners,
            "confident_losers": top_losers,
        }
        V30_RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))

        log.info(f"v30 complete. Confident: {n_confident}, winners: {len(confident_winners)}, losers: {len(confident_losers)}")
        v30_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(top_winners)} winners + {len(top_losers)} losers ready for download."}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log.error(f"v30 failed: {e}\n{tb}", exc_info=True)
        v30_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        v30_in_progress = False


def _build_price_path(after_bars, scan_price):
    """Summarize the post-scan price path: per-minute OHLCV trimmed to key waypoints.
    Returns a list of {minute_et, open, high, low, close, volume, pct_from_scan} for each bar."""
    path = []
    for b in after_bars:
        m = bar_to_et_minutes(b)
        if m is None: continue
        path.append({
            "minute_et": m,
            "time": f"{m//60:02d}:{m%60:02d}",
            "open": round(b["o"], 4), "high": round(b["h"], 4),
            "low": round(b["l"], 4), "close": round(b["c"], 4),
            "volume": int(b["v"]),
            "pct_from_scan_high": round((b["h"] - scan_price) / scan_price * 100, 3),
            "pct_from_scan_close": round((b["c"] - scan_price) / scan_price * 100, 3),
        })
    return path


# Evidence-quality thresholds for declaring a setup tradable at a given hour.
# A setup is "active" only at scan hours where it survived test-fold evaluation.
# These thresholds are deliberately conservative.
#
# v11 multi-testing protection:
# With 20 setups × 5 scan hours = 100 setup-hour tests, false positives at 5%
# significance would give ~5 spurious results. To reduce this we require that
# STRONG tier setups ALSO show positive test edge in at least one adjacent
# scan hour. Real signals tend to persist across nearby time windows; pure
# noise rarely does. This is a pragmatic substitute for Bonferroni correction.
SETUP_EVIDENCE_THRESHOLDS = {
    "strong":   {"edge": 5.0, "n": 100},  # ≥5% edge over base, ≥100 test events
    "moderate": {"edge": 3.0, "n": 25},   # ≥3% edge over base, ≥25 test events
    # Anything below moderate: not shown
}
STRONG_TIER_ADJACENT_HOURS = 1  # require ≥1 adjacent hour with positive test edge

def classify_setup_hour(test_stats):
    """Given a setup's test-fold stats, return evidence tier based on thresholds alone.
    Does not apply the multi-hour adjacency check — that's done by the caller
    using cross-hour data."""
    if not test_stats: return None
    edge = test_stats.get("edge_vs_base")
    n = test_stats.get("n_events", 0)
    if edge is None: return None
    for tier, crit in SETUP_EVIDENCE_THRESHOLDS.items():
        if edge >= crit["edge"] and n >= crit["n"]:
            return tier
    return None

def _check_adjacent_hour_positive(data, setup_name, current_hour):
    """Check whether a setup has positive test edge in any adjacent scan hour."""
    hours_data = data.get("hours", {})
    # SCAN_HOURS is [11,12,13,14,15]; adjacent to current_hour within same list
    try: idx = SCAN_HOURS.index(current_hour)
    except ValueError: return False
    for offset in (-1, 1):
        adj_idx = idx + offset
        if 0 <= adj_idx < len(SCAN_HOURS):
            adj_hour = SCAN_HOURS[adj_idx]
            adj_stats = (hours_data.get(str(adj_hour), {})
                         .get("setups", {}).get(setup_name, {}).get("test", {}))
            adj_edge = adj_stats.get("edge_vs_base")
            if adj_edge is not None and adj_edge > 0:
                return True
    return False

def derive_breadth_rule(regime_data):
    """
    From a setup's breadth regime data, derive an action rule.

    Evidence tiers:
    - STRONG (hard filter): breadth_down n≥30 AND edge_down ≤ -3 AND spread ≥ 10
      → filter out matches when R2K breadth < 0.45
    - MODERATE (soft warning): breadth_down n≥15 AND edge_down < 0
      → show match but flag "weak breadth" when R2K breadth < 0.45
    - NONE: insufficient evidence, no action
    """
    if not regime_data: return None
    breadth_down = regime_data.get("breadth_down", {})
    breadth_flat = regime_data.get("breadth_flat", {})
    breadth_up = regime_data.get("breadth_up", {})

    ed = breadth_down.get("edge_vs_base")
    nd = breadth_down.get("n_events", 0)
    if ed is None: return None

    edges = [e.get("edge_vs_base") for e in [breadth_down, breadth_flat, breadth_up]
             if e.get("edge_vs_base") is not None]
    spread = max(edges) - min(edges) if len(edges) >= 2 else 0

    if nd >= 30 and ed <= -3 and spread >= 10:
        return {
            "action": "filter_when_down",
            "evidence": "strong",
            "down_edge": round(ed, 2), "n_down": nd, "spread": round(spread, 2),
            "description": f"Hard-filter: historical edge in breadth_down was {ed}% (n={nd}), {spread}% spread vs other regimes",
        }
    if nd >= 15 and ed < 0:
        return {
            "action": "warn_when_down",
            "evidence": "moderate",
            "down_edge": round(ed, 2), "n_down": nd, "spread": round(spread, 2),
            "description": f"Soft-warn: historical edge in breadth_down was {ed}% (n={nd})",
        }
    return None

def derive_atr_filter_rule(atr_test_data, atr_lo_threshold):
    """v21: Derive an ATR filter rule for a setup from the atr_filter_test results.
    Returns a rule dict if the setup's mid_hi verdict is IMPROVES, else None.

    Rule: skip matches where stock's ATR% is below atr_lo_threshold.
    """
    if not atr_test_data: return None
    verdict = atr_test_data.get("verdict")
    if verdict != "IMPROVES": return None
    # Extract edge deltas for confidence
    mid_hi = atr_test_data.get("mid_hi", {})
    return {
        "action": "filter_when_low_atr",
        "atr_threshold_pct": atr_lo_threshold,
        "verdict": verdict,
        "edge_delta_disc": mid_hi.get("edge_delta_disc"),
        "edge_delta_val": mid_hi.get("edge_delta_val"),
        "volume_reduction_pct": mid_hi.get("firing_volume_change_pct"),
        "description": (f"Skip when stock ATR% < {atr_lo_threshold:.2f}%. "
                        f"Historical edge improvement: +{mid_hi.get('edge_delta_disc')}pp (disc), "
                        f"+{mid_hi.get('edge_delta_val')}pp (val). "
                        f"Volume change: {mid_hi.get('firing_volume_change_pct')}%."),
    }

def load_active_setups_for_scanner():
    """
    Returns:
      {
        11: [{"name": "orb_vol", "tier": "strong", "test_hit_rate": 51.3, "test_edge": 7.8, "breadth_rule": {...}, ...}, ...],
        13: [...],
        ...
      }
    Applies tier classification PLUS v11 multi-hour adjacency rule:
    - STRONG tier requires positive edge in ≥1 adjacent scan hour. Setups that
      only show strong edge in a single isolated hour are demoted to MODERATE
      (if they still meet moderate criteria) or excluded.
    - MODERATE tier has no adjacency requirement (we accept more false
      positives at moderate tier in exchange for candidate breadth).
    v13: Also attaches `breadth_rule` derived from regime analysis (if available).
    v17: Also gated by Ex-F_A validation — setups that failed Ex-F_A (edge <3%
         on the combined other 4 folds) are excluded. This drops 3-4 setups
         whose F_A edge didn't replicate on never-used-as-test data.
    """
    data = load_setup_results()
    if not data: return {}
    active = {}
    regimes_all = data.get("regimes", {}).get("by_hour", {})
    ex_fa_all = data.get("ex_fa", {}).get("by_hour", {})
    for h_str, hdata in data.get("hours", {}).items():
        try: h = int(h_str)
        except: continue
        hour_base = hdata.get("base", {}).get("test", {})
        hour_regimes = regimes_all.get(h_str, {})
        hour_ex_fa = ex_fa_all.get(h_str, {}).get("setups", {})
        hour_active = []
        for setup_name, fold_stats in hdata.get("setups", {}).items():
            tst = fold_stats.get("test", {})
            tier = classify_setup_hour(tst)
            if tier is None: continue
            # v11: adjacency check for strong tier
            if tier == "strong":
                has_adjacent = _check_adjacent_hour_positive(data, setup_name, h)
                if not has_adjacent:
                    # Demote: is it still moderate?
                    demoted_tier = None
                    edge = tst.get("edge_vs_base", 0)
                    n = tst.get("n_events", 0)
                    mc = SETUP_EVIDENCE_THRESHOLDS["moderate"]
                    if edge >= mc["edge"] and n >= mc["n"]:
                        demoted_tier = "moderate"
                    if demoted_tier is None: continue
                    tier = demoted_tier
            # v17: Ex-F_A gate — if we have ex_fa data and this setup FAILED it, exclude
            ex_fa_rec = hour_ex_fa.get(setup_name)
            if ex_fa_rec is not None and ex_fa_rec.get("ex_fa_edge") is not None:
                if not ex_fa_rec.get("holds_up", False):
                    # Failed Ex-F_A — don't activate
                    continue
            # v13: derive breadth filter/warning rule from regime data
            setup_regimes = hour_regimes.get(setup_name, {})
            breadth_rule = derive_breadth_rule(setup_regimes.get("breadth", {}))
            # v21: derive ATR filter rule from atr_filter_test results
            atr_test_hour = data.get("atr_filter_test", {}).get("by_hour", {}).get(h_str, {})
            atr_tert = atr_test_hour.get("atr_tertile_boundaries", [None, None])
            atr_lo_threshold = atr_tert[0] if atr_tert else None
            atr_test_setup = atr_test_hour.get("setups", {}).get(setup_name)
            atr_rule = (derive_atr_filter_rule(atr_test_setup, atr_lo_threshold)
                        if (atr_test_setup and atr_lo_threshold is not None) else None)
            hour_active.append({
                "name": setup_name,
                "tier": tier,
                "description": data.get("setups", {}).get(setup_name, ""),
                "test_hit_rate": tst.get("hit_rate"),
                "test_edge": tst.get("edge_vs_base"),
                "test_n": tst.get("n_events"),
                "test_mean_pnl": tst.get("mean_pnl"),
                "firing_day_frac": tst.get("firing_day_frac"),
                "base_hit_rate": hour_base.get("hit_rate"),
                "breadth_rule": breadth_rule,
                "atr_filter_rule": atr_rule,
                # v17: expose ex-F_A edge for UI
                "ex_fa_edge": ex_fa_rec.get("ex_fa_edge") if ex_fa_rec else None,
                "ex_fa_n": ex_fa_rec.get("ex_fa_n") if ex_fa_rec else None,
            })
        if hour_active:
            active[h] = hour_active
    return active

# ─── v10: live setup firing log ──────────────────────────────────
# Every time a setup fires live, persist it so we can later compare live
# hit rate to backtest hit rate. This is the honest feedback loop.
SETUP_FIRING_DIR = DATA_DIR / "setup_firings"
SETUP_FIRING_DIR.mkdir(parents=True, exist_ok=True)

def append_setup_firings(date, scan_hour, stock_result, filtered_setups=None, breadth_context=None):
    """Record each setup firing on a stock to a per-day JSON log.
    Stored fields are the minimum needed to evaluate outcomes later:
    ticker, scan-time price, setups that fired, tier, scan-time timestamp.

    v13 additions:
    - filtered_setups: list of setup metadata dicts that fired but were
      filtered out by the breadth rule. These are recorded so we can
      compare live outcomes of filtered vs unfiltered firings.
    - breadth_context: {"frac": 0.42, "label": "breadth_down"} — what the
      scanner thought the regime was at this scan time.
    """
    path = SETUP_FIRING_DIR / f"{date}.json"
    try: existing = json.loads(path.read_text())
    except: existing = []

    def _setup_brief(m):
        warn = m.get("weak_breadth_warning", False)
        return {
            "name": m["name"], "tier": m["tier"],
            "test_hit_rate": m.get("test_hit_rate"),
            "test_edge": m.get("test_edge"),
            "weak_breadth_warning": warn,
        }

    entry = {
        "date": date,
        "scan_hour": scan_hour,
        "ticker": stock_result["ticker"],
        "price": stock_result["price"],
        "win_prob": stock_result.get("winProb"),
        "setups": [_setup_brief(m) for m in stock_result.get("setupMatches", [])],
        "filtered_setups": [_setup_brief(m) for m in (filtered_setups or [])],
        "breadth_context": breadth_context,
        "timestamp": datetime.now(ET).isoformat(),
    }
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2))

def record_setup_outcomes(date_str=None):
    """For a past date, look up each recorded setup firing, fetch intraday bars,
    and record whether price actually hit entry_price * 1.01 before 15:55 close.
    Writes back into the firing log, adding 'hit_1pct' and 'final_pnl' per entry.
    """
    if date_str is None: date_str = today_et()
    path = SETUP_FIRING_DIR / f"{date_str}.json"
    if not path.exists(): return {"processed": 0, "error": "no firings for date"}
    try: firings = json.loads(path.read_text())
    except Exception as e: return {"processed": 0, "error": str(e)}

    # Only process if at least one entry lacks outcome
    needs_processing = [f for f in firings if "hit_1pct" not in f]
    if not needs_processing: return {"processed": 0, "status": "already processed"}

    if not has_creds(): return {"processed": 0, "error": "no Alpaca credentials"}
    client = alpaca_client()
    try:
        # Fetch intraday bars for all tickers in the log (deduplicated)
        tickers_to_fetch = list({f["ticker"] for f in needs_processing})
        all_bars = fetch_bars(client, tickers_to_fetch, "5Min",
                              f"{date_str}T09:30:00-04:00", f"{date_str}T16:05:00-04:00")
    finally:
        client.close()

    processed = 0
    for f in firings:
        if "hit_1pct" in f: continue
        ticker = f["ticker"]
        scan_hour = f["scan_hour"]
        try: entry_price = float(f["price"])
        except: continue
        scan_min = scan_hour * 60
        day_bars = all_bars.get(ticker, [])
        if not day_bars: continue
        after = [b for b in day_bars if (bar_to_et_minutes(b) or -1) >= scan_min]
        if len(after) < 2: continue
        entry = after[0]["o"]  # realistic entry at next bar open
        hit, pnl = did_hit_target(entry, after[1:], target_pct=0.01)
        f["entry_price"] = round(entry, 4)
        f["hit_1pct"] = bool(hit)
        f["final_pnl"] = round(pnl, 3)
        f["outcome_recorded_at"] = datetime.now(ET).isoformat()
        processed += 1

    path.write_text(json.dumps(firings, indent=2))
    return {"processed": processed, "date": date_str, "total_firings": len(firings)}

def load_live_setup_performance():
    """Aggregate all firing logs and report per-setup live hit rate.
    Compare to backtest test-fold hit rate. Returns a dict {setup_name -> stats}.
    """
    backtest_data = load_setup_results()
    results = defaultdict(lambda: {"firings": [], "n_outcomes": 0, "n_hits": 0,
                                   "total_pnl": 0.0, "by_hour": defaultdict(lambda: {"n":0, "hits":0})})

    for log_file in sorted(SETUP_FIRING_DIR.glob("*.json")):
        try: entries = json.loads(log_file.read_text())
        except: continue
        for entry in entries:
            if "hit_1pct" not in entry: continue
            for s in entry.get("setups", []):
                name = s["name"]
                sh = entry["scan_hour"]
                results[name]["firings"].append({
                    "date": entry["date"], "scan_hour": sh,
                    "ticker": entry["ticker"],
                    "hit": entry["hit_1pct"], "pnl": entry.get("final_pnl", 0)
                })
                results[name]["n_outcomes"] += 1
                if entry["hit_1pct"]:
                    results[name]["n_hits"] += 1
                results[name]["total_pnl"] += entry.get("final_pnl", 0) or 0
                results[name]["by_hour"][sh]["n"] += 1
                if entry["hit_1pct"]:
                    results[name]["by_hour"][sh]["hits"] += 1

    # Compute per-setup summary
    summary = {}
    for name, stats in results.items():
        n = stats["n_outcomes"]
        live_hit_rate = (stats["n_hits"] / n * 100) if n > 0 else None
        mean_pnl = (stats["total_pnl"] / n) if n > 0 else None
        # Backtest comparison: use aggregated test-fold hit rate if available
        bt = None
        if backtest_data:
            total_n, total_hits = 0, 0
            for h_str, hdata in backtest_data.get("hours", {}).items():
                t = hdata.get("setups", {}).get(name, {}).get("test", {})
                if t and t.get("hit_rate") is not None:
                    n_h = t.get("n_events", 0)
                    total_n += n_h
                    total_hits += n_h * t["hit_rate"] / 100
            bt = (total_hits / total_n * 100) if total_n > 0 else None
        by_hour_sum = {}
        for sh, d in stats["by_hour"].items():
            by_hour_sum[str(sh)] = {
                "n": d["n"],
                "hits": d["hits"],
                "hit_rate": round(d["hits"]/d["n"]*100, 2) if d["n"] > 0 else None,
            }
        summary[name] = {
            "n_outcomes": n,
            "n_hits": stats["n_hits"],
            "live_hit_rate": round(live_hit_rate, 2) if live_hit_rate is not None else None,
            "mean_pnl": round(mean_pnl, 3) if mean_pnl is not None else None,
            "backtest_hit_rate": round(bt, 2) if bt is not None else None,
            "drift": round(live_hit_rate - bt, 2) if (live_hit_rate is not None and bt is not None) else None,
            "by_hour": by_hour_sum,
        }
    return summary

# ═══════════════════════════════════════════════════════════════════
# LIVE SCAN
# ═══════════════════════════════════════════════════════════════════
def run_live_scan(scan_hour):
    if scan_hour not in models: raise ValueError(f"No model for {scan_hour}:00")
    t0 = time.time()
    client = alpaca_client()
    today = today_et()

    # Fetch bars for tickers + SPY
    fetch_syms = list(set(TICKERS + ["SPY", "IWM"]))
    intra = fetch_bars(client, fetch_syms, "5Min", f"{today}T09:30:00-04:00", datetime.now(timezone.utc).isoformat())
    snaps = fetch_snapshots(client, TICKERS)
    # Extended daily fetch to support 14-day ATR (need 15+ prior days)
    sd = (datetime.strptime(today,"%Y-%m-%d")-timedelta(days=30)).strftime("%Y-%m-%d")
    daily = fetch_bars(client, fetch_syms, "1Day", sd, today)
    client.close()

    # Compute SPY context
    spy_bars = intra.get("SPY", [])
    spy_ctx = compute_spy_context(spy_bars)

    # v10: For setup detection we need "bars before scan time" per stock
    scan_min = scan_hour * 60
    per_stock_bars_before = {}  # ticker → list of bars up to (exclusive) scan minute
    per_stock_prev_close = {}   # ticker → prior daily close
    per_stock_prior_daily = {}  # ticker → prior daily bars (for YH, pivot, ATR used by setups)

    raw_feats, stock_info, stock_sectors = [], [], []
    for ticker in TICKERS:
        bars = intra.get(ticker, [])
        snap = snaps.get(ticker, {})
        if len(bars) < 15: continue
        cp = snap.get("latestTrade",{}).get("p") or bars[-1]["c"]
        op = bars[0]["o"]
        dl = daily.get(ticker, [])
        # Previous close for gap
        prev_close = dl[-2]["c"] if len(dl) >= 2 else (dl[-1]["c"] if dl else None)

        # v8: per-stock ATR% for barrier placement. Exclude today's bar if present.
        dl_for_atr = [d for d in dl if d["t"][:10] < today][-25:]
        atr_pct = compute_atr_pct(dl_for_atr, lookback=ATR_LOOKBACK_DAYS)
        if atr_pct is None: continue  # Skip stocks with insufficient daily history

        feat = compute_features(bars, dl, cp, op, scan_hour,
                                spy_context=spy_ctx, prev_close=prev_close)
        if feat is None: continue
        raw_feats.append(feat)
        stock_info.append({"ticker":ticker,"sector":SECTORS.get(ticker,"?"),
                           "price":cp,"open":op,"atr_pct":atr_pct})
        stock_sectors.append(SECTORS.get(ticker,"?"))

        # Split bars into before/after scan_min for setup detection
        before = [b for b in bars if (bar_to_et_minutes(b) or -1) < scan_min]
        per_stock_bars_before[ticker] = before
        per_stock_prev_close[ticker] = prev_close
        per_stock_prior_daily[ticker] = dl_for_atr  # same list used for ATR

    # IWM bars up to scan time — for rel_strength_iwm setup
    iwm_all = intra.get("IWM", [])
    iwm_before = [b for b in iwm_all if (bar_to_et_minutes(b) or -1) < scan_min]

    if len(raw_feats) < 5: raise ValueError(f"Only {len(raw_feats)} stocks")
    add_ranks(raw_feats)
    add_sector_relative(raw_feats, stock_sectors)

    # v10: Compute sector breadth at scan time (fraction of sector green)
    sector_counts = defaultdict(lambda: [0, 0])  # [green, total]
    for i, si in enumerate(stock_info):
        sec = si["sector"]
        sector_counts[sec][1] += 1
        if si["price"] > si["open"]:
            sector_counts[sec][0] += 1

    # v13: Compute R2K-wide breadth at scan time (fraction of universe green)
    total_stocks = len(stock_info)
    green_stocks = sum(1 for si in stock_info if si["price"] > si["open"])
    r2k_breadth_frac = (green_stocks / total_stocks) if total_stocks > 0 else 0.5
    r2k_breadth_label = ("breadth_up" if r2k_breadth_frac > 0.55
                         else "breadth_down" if r2k_breadth_frac < 0.45
                         else "breadth_flat")

    # v10: Run setup detectors per stock; only keep results for setups
    # that are "active" at this scan hour (survived test-fold eval).
    active_setups_map = load_active_setups_for_scanner()
    active_at_this_hour = active_setups_map.get(scan_hour, [])
    active_setup_names = {s["name"] for s in active_at_this_hour}
    setup_metadata_by_name = {s["name"]: s for s in active_at_this_hour}

    per_stock_setup_matches = {}  # ticker → list of setup_metadata dicts (final, post-filter)
    per_stock_setup_filtered = {}  # v13: ticker → list of setups filtered out by breadth rule
    if active_setup_names:
        for i, si in enumerate(stock_info):
            ticker = si["ticker"]
            stock_atr_pct = (si.get("atr_pct") or 0) * 100  # convert fraction → percent to match filter threshold units
            before = per_stock_bars_before.get(ticker, [])
            prev_close = per_stock_prev_close.get(ticker)
            prior_daily = per_stock_prior_daily.get(ticker, [])
            sec = si["sector"]
            sc = sector_counts[sec]
            breadth = sc[0]/sc[1] if sc[1] > 0 else 0
            active = detect_setups(
                before, si["price"], prev_close,
                sector_breadth=breadth,
                prior_daily=prior_daily,
                iwm_bars=iwm_before,
                scan_minute_et=scan_min,
            )
            matched = []
            filtered = []
            for n, is_active in active.items():
                if not (is_active and n in active_setup_names):
                    continue
                base_meta = setup_metadata_by_name[n]
                rule = base_meta.get("breadth_rule")
                # v13: apply breadth rule
                # - filter_when_down: skip this match entirely if breadth is down
                # - warn_when_down: include but tag as weak_breadth_warning
                if rule and r2k_breadth_label == "breadth_down":
                    action = rule.get("action")
                    if action == "filter_when_down":
                        # Don't include this match; record as filtered for tracking
                        filtered.append({**base_meta, "filter_reason": rule.get("description","breadth filter")})
                        continue
                    elif action == "warn_when_down":
                        # Include but annotate
                        match_meta = {**base_meta, "weak_breadth_warning": True}
                        matched.append(match_meta)
                        continue
                # v21: apply ATR filter rule (setup-specific filter for low-ATR stocks).
                # Currently active only for rel_strength_iwm (verdict=IMPROVES on mid_hi).
                atr_rule = base_meta.get("atr_filter_rule")
                if atr_rule and atr_rule.get("action") == "filter_when_low_atr":
                    threshold = atr_rule.get("atr_threshold_pct")
                    if threshold is not None and stock_atr_pct > 0 and stock_atr_pct <= threshold:
                        filtered.append({**base_meta, "filter_reason": f"ATR filter: stock ATR% {stock_atr_pct:.2f}% ≤ threshold {threshold:.2f}%"})
                        continue
                matched.append(base_meta)
            if matched:
                per_stock_setup_matches[ticker] = matched
            if filtered:
                per_stock_setup_filtered[ticker] = filtered

    # Load patterns once (if any have been discovered)
    patterns_data = load_patterns()

    # v8: TP/SL are ATR multipliers stored in the trained model's meta.
    # Per-stock barriers computed from that stock's own ATR%.
    meta = model_meta.get(scan_hour, {})
    active_tp_mult = meta.get("tp_mult", TP_MULT)
    active_sl_mult = meta.get("sl_mult", SL_MULT)

    # Load threshold gating config (from last threshold analysis run)
    active_thresholds = load_active_thresholds()
    hour_threshold = active_thresholds.get(scan_hour)

    X = np.array([feat_to_arr(f) for f in raw_feats])
    raw_probs = models[scan_hour].predict(X)
    cal_probs = calibrators[scan_hour].predict(raw_probs) if scan_hour in calibrators else raw_probs

    results = []
    for i in range(len(raw_feats)):
        si, rf = stock_info[i], raw_feats[i]
        wp = float(cal_probs[i])
        # Per-stock barriers in percent terms
        stock_tp_pct = active_tp_mult * si["atr_pct"]
        stock_sl_pct = active_sl_mult * si["atr_pct"]
        # Expected value using per-stock asymmetry
        ev = (wp * stock_tp_pct - (1 - wp) * stock_sl_pct) * 100
        # Concrete price targets for order placement
        tp_price = si["price"] * (1 + stock_tp_pct)
        sl_price = si["price"] * (1 - stock_sl_pct)

        # Check if this stock matches any discovered patterns
        matches = check_patterns(rf, scan_hour, patterns_data) if patterns_data else []
        pattern_summary = None
        if matches:
            matches.sort(key=lambda p: p.get("val",{}).get("edge", 0), reverse=True)
            top = matches[0]
            pattern_summary = {
                "count": len(matches),
                "top_edge_val": top["val"]["edge"],
                "top_wr_val": top["val"]["win_rate"],
                "top_n_val": top["val"]["n"],
                "top_conditions": top["conditions"]
            }

        # Conviction-gate: does this stock clear the threshold for this scan hour?
        clears_threshold = (hour_threshold is not None and wp >= hour_threshold)

        # v10: active setup matches for this stock (from the hypothesis-first scanner)
        setup_matches = per_stock_setup_matches.get(si["ticker"], [])

        results.append({
            "rank":0,"ticker":si["ticker"],"sector":si["sector"],
            "price":f"{si['price']:.2f}",
            "changeFromOpen":f"{((si['price']-si['open'])/si['open']*100):.2f}",
            "winProb":round(wp,4),
            "ev":round(ev,3),
            "rawScore":round(float(raw_probs[i]),4),
            "patternMatch":pattern_summary,
            "clearsThreshold":clears_threshold,
            # v8: per-stock volatility-adjusted barriers
            "atrPct": round(si["atr_pct"]*100, 3),
            "tpPct": round(stock_tp_pct*100, 3),
            "slPct": round(stock_sl_pct*100, 3),
            "tpPrice": round(tp_price, 2),
            "slPrice": round(sl_price, 2),
            # v10: which predefined setups are firing on this stock right now
            "setupMatches": setup_matches,
            "features":{
                "momentum":f"{rf['momentum']:.4f}","relVolume":f"{rf['rel_volume']:.2f}",
                "vwapDist":f"{rf['vwap_dist']*100:.2f}","vwapSlope":f"{rf['vwap_slope']:.4f}",
                "orbStrength":f"{rf['orb_strength']:.3f}","atrReach":f"{rf['atr_reach']:.2f}",
                "realizedVol":f"{rf['realized_vol']:.4f}","trendStr":f"{rf['trend_str']:.4f}",
                "rsi":f"{rf['rsi']:.1f}",
                "retVsSpy":f"{rf['ret_vs_spy']*100:.2f}","retVsSector":f"{rf['ret_vs_sector']*100:.2f}",
                "sectorBreadth":f"{rf['sector_breadth']:.2f}","gapPct":f"{rf['gap_pct']*100:.2f}"
            }
        })

    results.sort(key=lambda x: x["winProb"], reverse=True)
    for i,r in enumerate(results): r["rank"] = i+1

    # Mark tradable = top-1 AND clears threshold
    # Strategy is binary: top pick must meet or exceed threshold, else no trade.
    for r in results:
        r["tradable"] = (r["rank"] == 1
                         and hour_threshold is not None
                         and r["winProb"] >= hour_threshold)

    elapsed = int((time.time()-t0)*1000)
    n_tradable = sum(1 for r in results if r.get("tradable"))

    # Aggregate barrier stats across all stocks (for UI display)
    all_tp = [r["tpPct"] for r in results]
    all_sl = [r["slPct"] for r in results]
    all_atr = [r["atrPct"] for r in results]

    # v10: aggregate setup-firing statistics for the header
    n_stocks_with_any_setup = sum(1 for r in results if r.get("setupMatches"))
    setup_firing_counts = defaultdict(int)
    for r in results:
        for m in r.get("setupMatches", []):
            setup_firing_counts[m["name"]] += 1

    scan_result = {
        "data":results,"timestamp":datetime.now(ET).isoformat(),"source":"live",
        "elapsed":elapsed,"scanHour":scan_hour,
        "modelAUC":meta.get("auc"),"modelWR10":meta.get("avg_win_rate_top10"),
        "modelPnL10":meta.get("avg_pnl_top10"),
        "scoreRange":{"min":results[-1]["rawScore"],"max":results[0]["rawScore"]} if results else None,
        # v8: ATR-multiplier config + universe-wide barrier summary
        "tp_mult": active_tp_mult, "sl_mult": active_sl_mult,
        "atrLookback": meta.get("atr_lookback_days", ATR_LOOKBACK_DAYS),
        "avgTpPct": round(float(np.mean(all_tp)), 2) if all_tp else None,
        "avgSlPct": round(float(np.mean(all_sl)), 2) if all_sl else None,
        "avgAtrPct": round(float(np.mean(all_atr)), 2) if all_atr else None,
        "threshold":hour_threshold,
        "nTradable":n_tradable,
        # v10: setup scanner summary
        "activeSetups": active_at_this_hour,
        "nStocksWithSetup": n_stocks_with_any_setup,
        "setupFiringCounts": dict(setup_firing_counts),
        # v13: R2K-wide breadth + what was filtered by the breadth rule
        "r2kBreadthFrac": round(r2k_breadth_frac, 3),
        "r2kBreadthLabel": r2k_breadth_label,
        "r2kGreenStocks": green_stocks,
        "r2kTotalStocks": total_stocks,
    }

    # v10/v13: persist each setup firing to SETUP_FIRING_LOG for live-validation tracking.
    # v13 also records setups that fired but were filtered out by the breadth rule —
    # lets us later compare live outcomes of filtered vs unfiltered firings to validate the filter.
    breadth_ctx = {"frac": round(r2k_breadth_frac, 3), "label": r2k_breadth_label,
                   "green": green_stocks, "total": total_stocks}
    # Stocks where ANY setup fired (matched or filtered) get logged
    all_stocks_with_activity = set(per_stock_setup_matches.keys()) | set(per_stock_setup_filtered.keys())
    for r in results:
        if r["ticker"] in all_stocks_with_activity:
            try:
                append_setup_firings(
                    today, scan_hour, r,
                    filtered_setups=per_stock_setup_filtered.get(r["ticker"]),
                    breadth_context=breadth_ctx,
                )
            except Exception as e:
                log.warning(f"Setup firing log write failed: {e}")

    sp = SCAN_DIR / f"{today}.json"
    try: saved = json.loads(sp.read_text())
    except: saved = {}
    saved[str(scan_hour)] = results
    sp.write_text(json.dumps(saved))

    last_scans[str(scan_hour)] = scan_result
    LAST_SCAN_PATH.write_text(json.dumps(last_scans, default=str))

    log.info(f"Scan {scan_hour}:00: {len(results)} stocks, {elapsed}ms, "
             f"top5 EV: {[r['ev'] for r in results[:5]]}")
    return scan_result

# ═══════════════════════════════════════════════════════════════════
# OUTCOME RECORDING — FIRST-PASSAGE
# ═══════════════════════════════════════════════════════════════════
def record_outcomes():
    if not has_creds(): return
    today = today_et()
    out_path = OUTCOME_DIR / f"{today}.json"
    if out_path.exists(): log.info(f"Outcomes {today} done."); return

    log.info(f"Recording outcomes {today} (first-passage)...")
    client = alpaca_client()
    try:
        all_bars = fetch_bars(client, TICKERS, "5Min",
                              f"{today}T09:30:00-04:00", f"{today}T16:05:00-04:00")
        client.close()
    except Exception as e:
        log.error(f"Outcome fetch: {e}"); client.close(); return

    sp = SCAN_DIR / f"{today}.json"
    try: today_scans = json.loads(sp.read_text())
    except: today_scans = {}

    outcomes = {}
    for h in SCAN_HOURS:
        outcomes[str(h)] = []
        scan_min = h * 60
        for ticker in TICKERS:
            bars = all_bars.get(ticker, [])
            if len(bars) < 6: continue
            before, after = [], []
            for b in bars:
                bm = bar_to_et_minutes(b)
                if bm is None: continue
                if bm < scan_min: before.append(b)
                else: after.append(b)
            if not before or len(after) < 2: continue

            entry_price = after[0]["o"]

            # v8: retrieve per-stock barriers from saved scan data if available
            stock_tp_pct, stock_sl_pct, raw_score = None, None, None
            scanned = today_scans.get(str(h), [])
            for s in scanned:
                if s["ticker"] == ticker:
                    raw_score = s.get("rawScore")
                    stock_tp_pct = s.get("tpPct")  # in percent units
                    stock_sl_pct = s.get("slPct")
                    break

            # If we have per-stock barriers, use them; else fall back to None (legacy default)
            if stock_tp_pct is not None and stock_sl_pct is not None:
                outcome, pnl, reason = compute_trade_outcome(
                    entry_price, after[1:],
                    tp_pct=stock_tp_pct/100.0, sl_pct=stock_sl_pct/100.0)
            else:
                outcome, pnl, reason = compute_trade_outcome(entry_price, after[1:])

            outcomes[str(h)].append({
                "ticker":ticker,"entryPrice":entry_price,"outcome":outcome,
                "pnl":pnl,"reason":reason,"rawScore":raw_score,
                "tpPct":stock_tp_pct,"slPct":stock_sl_pct
            })

    out_path.write_text(json.dumps({"date":today,"outcomes":outcomes,
        "tp_mult":status.get("activeTpMult",TP_MULT),"sl_mult":status.get("activeSlMult",SL_MULT),
        "recordedAt":datetime.now(ET).isoformat()}, indent=2))

    n_files = len(list(OUTCOME_DIR.glob("*.json")))
    status["outcomeDays"] = n_files
    status["daysSinceRetrain"] = status.get("daysSinceRetrain",0) + 1
    save_status(status)
    log.info(f"Outcomes saved. {n_files} days total.")

    # v10: also record setup firing outcomes for live-validation tracking
    try:
        res = record_setup_outcomes(today)
        log.info(f"Setup outcomes: {res}")
    except Exception as e:
        log.warning(f"Setup outcome recording failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════
app = FastAPI()

@app.get("/api/health")
def health():
    # v8: active barriers are ATR multipliers, not fixed percentages
    active_tp_mult = status.get("activeTpMult", TP_MULT)
    active_sl_mult = status.get("activeSlMult", SL_MULT)
    return {
        "status":"ok","hasCredentials":has_creds(),"marketOpen":market_open(),
        "currentHourET":hour_et(),
        "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
        "outcomeDays":status.get("outcomeDays",0),
        "daysSinceRetrain":status.get("daysSinceRetrain",0),
        "modelsLoaded":list(models.keys()),
        "hasLastScan":bool(last_scans),
        "lastScanHours":list(last_scans.keys()),
        "tp_mult":active_tp_mult,"sl_mult":active_sl_mult,
        "atr_lookback":status.get("atrLookback", ATR_LOOKBACK_DAYS),
        # notional breakeven based on multipliers (true per-stock BE varies)
        "notional_breakeven": round(active_sl_mult/(active_sl_mult+active_tp_mult)*100,1) if (active_sl_mult+active_tp_mult)>0 else 50.0
    }

@app.get("/api/scan/{hour}")
def get_scan(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":f"Scan hour {hour} not supported. Valid: {SCAN_HOURS}"},400)
    # Diagnose why we can't run live
    current_hour_et = hour_et()
    if not market_open():
        reason = "Market is closed."
    elif not has_creds():
        reason = "No Alpaca credentials configured."
    elif hour not in models:
        reason = f"No trained model for {hour}:00 — run Training."
    elif current_hour_et < hour:
        reason = f"Scan hour {hour}:00 ET hasn't arrived yet (current hour {current_hour_et}:00 ET). Market must be past scan time with enough bars (~90 min after 09:30)."
    else:
        reason = None  # all prerequisites met, try live scan

    if reason is None:
        try: return run_live_scan(hour)
        except Exception as e:
            log.error(f"Scan: {e}")
            reason = f"Scan failed: {str(e)}"

    cached = last_scans.get(str(hour))
    if cached: return {**cached, "source":"cached", "message":reason}
    return {"data":[], "source":"offline", "timestamp":datetime.now(ET).isoformat(),
            "message":reason}

@app.post("/api/scan/{hour}/refresh")
def refresh(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":f"Scan hour {hour} not supported. Valid: {SCAN_HOURS}"},400)
    if not market_open(): return JSONResponse({"error":"Market closed"},400)
    if hour not in models: return JSONResponse({"error":"No model"},400)
    return run_live_scan(hour)

class TrainRequest(BaseModel):
    # v8: ATR multipliers replace fixed tp_pct/sl_pct.
    # Legacy names kept for back-compat; server infers meaning from magnitude
    # (< 3 interpreted as multiplier; those values are rejected anyway under old semantics).
    tp_pct: Optional[float] = None  # now ATR multiplier (e.g. 0.5)
    sl_pct: Optional[float] = None  # now ATR multiplier (e.g. 2.5)
    tp_mult: Optional[float] = None
    sl_mult: Optional[float] = None

@app.post("/api/train")
def trigger_train(bg: BackgroundTasks, req: Optional[TrainRequest] = None):
    if training_in_progress: return {"status":"already_running"}
    # Prefer explicit tp_mult/sl_mult; fall back to tp_pct/sl_pct as multipliers
    tp_m = (req.tp_mult if req and req.tp_mult is not None
            else (req.tp_pct if req and req.tp_pct is not None else None))
    sl_m = (req.sl_mult if req and req.sl_mult is not None
            else (req.sl_pct if req and req.sl_pct is not None else None))
    # Bounds check: multipliers must be 0.1-5.0 (reasonable ATR range)
    if tp_m is not None and not (0.1 <= tp_m <= 5.0): return JSONResponse({"error":"tp_mult must be 0.1-5.0"},400)
    if sl_m is not None and not (0.1 <= sl_m <= 5.0): return JSONResponse({"error":"sl_mult must be 0.1-5.0"},400)
    bg.add_task(run_training, tp_m, sl_m)
    return {"status":"started","tp_mult":tp_m or TP_MULT,"sl_mult":sl_m or SL_MULT}

@app.post("/api/cache/clear")
def clear_cache():
    """Delete cached bar data to force fresh fetch on next training."""
    if training_in_progress: return JSONResponse({"error":"Cannot clear during training"},400)
    deleted = []
    for f in [BARS_DAILY_CACHE, BARS_INTRADAY_CACHE]:
        if f.exists():
            f.unlink()
            deleted.append(f.name)
    return {"status":"ok","deleted":deleted}

# v17: extend cache backward in history without retraining
@app.post("/api/cache/extend_history")
def extend_history_endpoint(bg: BackgroundTasks, months: int = 12):
    if extend_history_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Cannot extend during training"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"No existing cache. Run Training first."},400)
    if not (1 <= months <= 36):
        return JSONResponse({"error":"months must be 1-36"},400)
    bg.add_task(run_extend_history, months)
    return {"status":"started","months_back":months}

@app.get("/api/cache/extend_history/progress")
def extend_history_progress_endpoint():
    return {"inProgress":extend_history_in_progress, **extend_history_progress}

# v18: SPY+IWM repair endpoints
@app.post("/api/cache/repair_etf")
def repair_etf_endpoint(bg: BackgroundTasks):
    if repair_etf_in_progress: return {"status":"already_running"}
    if training_in_progress or extend_history_in_progress:
        return JSONResponse({"error":"Cannot repair during training/extend"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"No existing cache"},400)
    bg.add_task(run_repair_etf)
    return {"status":"started"}

@app.get("/api/cache/repair_etf/progress")
def repair_etf_progress_endpoint():
    return {"inProgress":repair_etf_in_progress, **repair_etf_progress}

@app.get("/api/cache/status")
def cache_status():
    return {
        "daily": {"exists":BARS_DAILY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_DAILY_CACHE),1)},
        "intraday": {"exists":BARS_INTRADAY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_INTRADAY_CACHE),1)},
        "max_age_hours":CACHE_MAX_AGE_HOURS
    }

@app.get("/api/training/progress")
def progress():
    return {"inProgress":training_in_progress,**training_progress,
            "meta":{str(h):model_meta[h] for h in model_meta}}

# ─── SWEEP endpoints ──────────────────────────────────────────────
@app.post("/api/sweep")
def trigger_sweep(bg: BackgroundTasks):
    if sweep_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress; wait for it to finish"},400)
    bg.add_task(run_sweep, True)  # resume=True
    total = len(SWEEP_TP_VALUES) * len(SWEEP_SL_VALUES)
    return {"status":"started","total_cells":total,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.post("/api/sweep/reset")
def reset_sweep():
    if sweep_in_progress: return JSONResponse({"error":"Cannot reset during sweep"},400)
    if SWEEP_RESULTS_PATH.exists(): SWEEP_RESULTS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/sweep/status")
def sweep_status():
    return {"inProgress":sweep_in_progress, **sweep_progress,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.get("/api/sweep/results")
def sweep_results():
    return load_sweep_results()

# ─── PATTERN SEARCH endpoints ─────────────────────────────────────
class PatternSearchRequest(BaseModel):
    tp_pct: Optional[float] = None
    sl_pct: Optional[float] = None

@app.post("/api/patterns/search")
def trigger_pattern_search(bg: BackgroundTasks, req: Optional[PatternSearchRequest] = None):
    if pattern_search_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress"},400)
    if not TRAINING_ROWS_CACHE.exists():
        return JSONResponse({"error":"No training data cached. Run training first."},400)
    tp = (req.tp_pct/100.0) if req and req.tp_pct is not None else None
    sl = (req.sl_pct/100.0) if req and req.sl_pct is not None else None
    if tp is not None and not (0.001 <= tp <= 0.10): return JSONResponse({"error":"tp_pct must be 0.1-10.0"},400)
    if sl is not None and not (0.001 <= sl <= 0.10): return JSONResponse({"error":"sl_pct must be 0.1-10.0"},400)
    bg.add_task(run_pattern_search, tp, sl)
    return {"status":"started"}

@app.post("/api/patterns/reset")
def reset_patterns():
    if pattern_search_in_progress: return JSONResponse({"error":"Cannot reset during search"},400)
    if PATTERNS_PATH.exists(): PATTERNS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/patterns/progress")
def patterns_progress():
    return {"inProgress":pattern_search_in_progress, **pattern_search_progress,
            "hasTrainingData":TRAINING_ROWS_CACHE.exists()}

@app.get("/api/patterns/results")
def patterns_results():
    data = load_patterns()
    return data or {"hours":{}, "generatedAt":None}

# ─── SENSITIVITY SWEEP endpoints ─────────────────────────────────
@app.post("/api/sensitivity/run")
def trigger_sensitivity(bg: BackgroundTasks, req: Optional[PatternSearchRequest] = None):
    if sensitivity_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress"},400)
    if not TRAINING_ROWS_CACHE.exists():
        return JSONResponse({"error":"No training data cached. Run training first."},400)
    tp = (req.tp_pct/100.0) if req and req.tp_pct is not None else None
    sl = (req.sl_pct/100.0) if req and req.sl_pct is not None else None
    bg.add_task(run_sensitivity_sweep, tp, sl)
    return {"status":"started"}

@app.get("/api/sensitivity/progress")
def sensitivity_progress_endpoint():
    return {"inProgress":sensitivity_in_progress, **sensitivity_progress}

@app.get("/api/sensitivity/results")
def sensitivity_results_endpoint():
    data = load_sensitivity()
    return data or {"hours":{}, "generatedAt":None}

# ─── THRESHOLD ANALYZER endpoints ─────────────────────────────────
@app.post("/api/threshold/run")
def trigger_threshold_analysis(bg: BackgroundTasks):
    if thresh_analysis_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress"},400)
    if not TRAINING_ROWS_CACHE.exists():
        return JSONResponse({"error":"No training data cached. Run training first."},400)
    if not models:
        return JSONResponse({"error":"No models loaded."},400)
    bg.add_task(run_threshold_analysis)
    return {"status":"started"}

@app.post("/api/threshold/reset")
def reset_threshold():
    if thresh_analysis_in_progress: return JSONResponse({"error":"Cannot reset during analysis"},400)
    if THRESHOLD_RESULTS_PATH.exists(): THRESHOLD_RESULTS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/threshold/progress")
def threshold_progress_endpoint():
    return {"inProgress":thresh_analysis_in_progress, **thresh_analysis_progress,
            "hasTrainingData":TRAINING_ROWS_CACHE.exists(),
            "hasModels":bool(models)}

@app.get("/api/threshold/results")
def threshold_results_endpoint():
    data = load_threshold_results()
    return data or {"hours":{}, "generatedAt":None}

# ─── v9: SETUP EVALUATION endpoints ──────────────────────────────
@app.post("/api/setup/run")
def trigger_setup_evaluation(bg: BackgroundTasks):
    if setup_eval_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing. Run Training first."},400)
    bg.add_task(run_setup_evaluation)
    return {"status":"started"}

# v25: conviction model training endpoints
@app.post("/api/conviction/train")
def trigger_conviction_training(bg: BackgroundTasks):
    if conviction_train_in_progress: return {"status":"already_running"}
    if training_in_progress or setup_eval_in_progress:
        return JSONResponse({"error":"Another task running"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing"},400)
    bg.add_task(run_conviction_training)
    return {"status":"started"}

@app.get("/api/conviction/progress")
def conviction_progress_endpoint():
    return {"inProgress":conviction_train_in_progress, **conviction_train_progress}

@app.get("/api/conviction/results")
def conviction_results_endpoint():
    if not CONVICTION_RESULTS_PATH.exists():
        return {"status":"no_results"}
    try:
        return json.loads(CONVICTION_RESULTS_PATH.read_text())
    except Exception as e:
        return JSONResponse({"error":str(e)},500)

# v27: pattern discovery endpoints
@app.post("/api/pattern/train")
def trigger_pattern_discovery(bg: BackgroundTasks):
    if pattern_discovery_in_progress: return {"status":"already_running"}
    if training_in_progress or setup_eval_in_progress or conviction_train_in_progress:
        return JSONResponse({"error":"Another task running"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing"},400)
    bg.add_task(run_pattern_discovery)
    return {"status":"started"}

@app.get("/api/pattern/progress")
def pattern_progress_endpoint():
    return {"inProgress":pattern_discovery_in_progress, **pattern_discovery_progress}

@app.get("/api/pattern/results")
def pattern_results_endpoint():
    if not PATTERN_RESULTS_PATH.exists():
        return {"status":"no_results"}
    try:
        return json.loads(PATTERN_RESULTS_PATH.read_text())
    except Exception as e:
        return JSONResponse({"error":str(e)},500)

# v28: cost-adjusted analysis endpoints (runs LightGBM + pattern both, across 3 targets)
@app.post("/api/v28/train")
def trigger_v28(bg: BackgroundTasks):
    if v28_in_progress: return {"status":"already_running"}
    if (training_in_progress or setup_eval_in_progress or
        conviction_train_in_progress or pattern_discovery_in_progress):
        return JSONResponse({"error":"Another task running"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing"},400)
    bg.add_task(run_v28_cost_adjusted)
    return {"status":"started"}

@app.get("/api/v28/progress")
def v28_progress_endpoint():
    return {"inProgress":v28_in_progress, **v28_progress}

@app.get("/api/v28/results")
def v28_results_endpoint():
    if not V28_RESULTS_PATH.exists():
        return {"status":"no_results"}
    try:
        return json.loads(V28_RESULTS_PATH.read_text())
    except Exception as e:
        return JSONResponse({"error":str(e)},500)

# v29: fine-grained target sweep endpoints
@app.post("/api/v29/train")
def trigger_v29(bg: BackgroundTasks):
    if v29_in_progress: return {"status":"already_running"}
    if (training_in_progress or setup_eval_in_progress or
        conviction_train_in_progress or pattern_discovery_in_progress or v28_in_progress):
        return JSONResponse({"error":"Another task running"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing"},400)
    bg.add_task(run_v29_target_sweep)
    return {"status":"started"}

@app.get("/api/v29/progress")
def v29_progress_endpoint():
    return {"inProgress":v29_in_progress, **v29_progress}

@app.get("/api/v29/results")
def v29_results_endpoint():
    if not V29_RESULTS_PATH.exists():
        return {"status":"no_results"}
    try:
        return json.loads(V29_RESULTS_PATH.read_text())
    except Exception as e:
        return JSONResponse({"error":str(e)},500)

# v30: confident examples dump endpoints
@app.post("/api/v30/train")
def trigger_v30(bg: BackgroundTasks):
    if v30_in_progress: return {"status":"already_running"}
    if (training_in_progress or setup_eval_in_progress or
        conviction_train_in_progress or pattern_discovery_in_progress or
        v28_in_progress or v29_in_progress):
        return JSONResponse({"error":"Another task running"},400)
    if not (BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()):
        return JSONResponse({"error":"Bar cache missing"},400)
    bg.add_task(run_v30_examples_dump)
    return {"status":"started"}

@app.get("/api/v30/progress")
def v30_progress_endpoint():
    return {"inProgress":v30_in_progress, **v30_progress}

@app.get("/api/v30/results")
def v30_results_endpoint():
    if not V30_RESULTS_PATH.exists():
        return {"status":"no_results"}
    try:
        return json.loads(V30_RESULTS_PATH.read_text())
    except Exception as e:
        return JSONResponse({"error":str(e)},500)






@app.post("/api/setup/reset")
def reset_setup():
    if setup_eval_in_progress: return JSONResponse({"error":"Cannot reset during eval"},400)
    if SETUP_RESULTS_PATH.exists(): SETUP_RESULTS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/setup/progress")
def setup_progress_endpoint():
    return {"inProgress":setup_eval_in_progress, **setup_eval_progress,
            "hasBars":BARS_DAILY_CACHE.exists() and BARS_INTRADAY_CACHE.exists()}

@app.get("/api/setup/results")
def setup_results_endpoint():
    data = load_setup_results()
    return data or {"hours":{}, "generatedAt":None}

@app.get("/api/setup/active")
def setup_active_endpoint():
    """Returns per-hour active setups (survived test-fold) for display in the UI."""
    return {"active": load_active_setups_for_scanner()}

@app.get("/api/setup/live")
def setup_live_performance():
    """Live firing outcomes vs backtest hit rate — the honest feedback loop."""
    return load_live_setup_performance()

@app.post("/api/setup/record_outcomes")
def trigger_setup_outcome_recording():
    """Manually trigger outcome recording for today's firings (for testing;
    normally runs on the 16:12 ET cron)."""
    return record_setup_outcomes()

@app.get("/api/outcomes/summary")
def outcome_summary():
    files = sorted(OUTCOME_DIR.glob("*.json"))
    if not files: return {"totalDays":0,"recent":[]}
    recent = []
    for f in files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hs = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            top10 = scored[:10]
            wins = sum(1 for e in top10 if e["outcome"]==1)
            avg_pnl = np.mean([e["pnl"] for e in top10]) if top10 else 0
            base_wr = np.mean([e["outcome"] for e in entries]) if entries else 0
            reasons = {}
            for e in entries:
                r = e.get("reason","?")
                reasons[r] = reasons.get(r,0)+1
            hs[str(h)] = {"total":len(entries),"top10wins":wins,
                      "top10pnl":round(avg_pnl,3),"baseWR":round(base_wr*100,1),
                      "reasons":reasons}
        recent.append({"date":d["date"],"hours":hs})
    return {"totalDays":len(files),"recent":recent}

@app.get("/api/diagnostic")
def diagnostic():
    outcome_files = sorted(OUTCOME_DIR.glob("*.json"))
    outcomes = []
    for f in outcome_files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hd = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            t10 = scored[:10]
            hd[str(h)] = {
                "totalStocks":len(entries),
                "baseWinRate":round(np.mean([e["outcome"] for e in entries])*100,1) if entries else None,
                "top10":[{"ticker":e["ticker"],"score":e["rawScore"],"outcome":e["outcome"],"pnl":e["pnl"],"reason":e["reason"]} for e in t10],
                "top10wins":sum(1 for e in t10 if e["outcome"]==1),
                "top10pnl":round(np.mean([e["pnl"] for e in t10]),3) if t10 else 0,
                "reasons":{r:sum(1 for e in entries if e.get("reason")==r) for r in set(e.get("reason","?") for e in entries)}
            }
        outcomes.append({"date":d["date"],"hours":hd})

    scans = {}
    for h_str, scan in last_scans.items():
        scans[h_str] = {
            "timestamp":scan.get("timestamp"),"source":scan.get("source"),
            "scoreRange":scan.get("scoreRange"),
            "top20":(scan.get("data") or [])[:20]
        }

    return JSONResponse({
        "_type":"r2k_scanner_diagnostic","_version":"5.0_r2k_first_passage",
        "generatedAt":datetime.now(ET).isoformat(),
        "strategy":{"tp_mult":status.get("activeTpMult",TP_MULT),"sl_mult":status.get("activeSlMult",SL_MULT),"atr_lookback":ATR_LOOKBACK_DAYS,"forced_close":"15:55 ET","entry_delay":"1 bar"},
        "server":{
            "hasCredentials":has_creds(),"marketOpen":market_open(),"currentHourET":hour_et(),
            "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
            "outcomeDays":status.get("outcomeDays",0),"daysSinceRetrain":status.get("daysSinceRetrain",0)
        },
        "modelMeta":{str(h):model_meta[h] for h in model_meta},
        "lastScans":scans,
        "outcomes":outcomes,
        "outcomeSummary":{"totalDays":len(outcome_files),
            "dateRange":{"first":outcome_files[0].stem,"last":outcome_files[-1].stem} if outcome_files else None}
    }, headers={"Content-Disposition":f'attachment; filename="r2k_diagnostic_{today_et()}.json"'})

# SPA fallback
dist_path = Path(__file__).parent / "dist"
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=dist_path/"assets"), name="assets")
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        fp = dist_path / full_path
        if fp.is_file(): return FileResponse(fp)
        return FileResponse(dist_path / "index.html")

# ═══════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════
scheduler = BackgroundScheduler(timezone=ET)
if has_creds():
    def cron_scan():
        h = hour_et()
        if h in SCAN_HOURS and market_open() and h in models:
            try: run_live_scan(h)
            except Exception as e: log.error(f"Cron scan: {e}")
    scheduler.add_job(cron_scan, "cron", hour="11,12,13,14,15", minute=5, day_of_week="mon-fri")
    scheduler.add_job(record_outcomes, "cron", hour=16, minute=12, day_of_week="mon-fri")
    scheduler.start()
    log.info("Scheduler: scans 11-15 ET :05, outcomes 16:12 ET")
