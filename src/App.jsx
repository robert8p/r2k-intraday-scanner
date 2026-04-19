import { useState, useEffect, useCallback, Fragment, Component } from "react";

const SCAN_HOURS = [11,12,13,14,15];
const F = "'JetBrains Mono','SF Mono','Fira Code','Cascadia Code',monospace";

// v11: Error boundary. Catches any rendering crash in a subtree and shows a
// helpful message instead of blanking the entire app. Without this, a single
// bad row can take the whole UI offline.
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, errorInfo: null };
  }
  static getDerivedStateFromError(error) {
    return { error };
  }
  componentDidCatch(error, errorInfo) {
    this.setState({ error, errorInfo });
    console.error("ErrorBoundary caught:", error, errorInfo);
  }
  reset = () => this.setState({ error: null, errorInfo: null });
  render() {
    if (this.state.error) {
      return (
        <div style={{padding:20,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.3)",borderRadius:8,fontFamily:F,color:"#fca5a5"}}>
          <div style={{fontSize:13,fontWeight:700,marginBottom:8}}>⚠ UI render error (app kept alive by error boundary)</div>
          <div style={{fontSize:11,color:"#f87171",marginBottom:6}}>{String(this.state.error?.message || this.state.error)}</div>
          {this.state.errorInfo?.componentStack && (
            <details style={{fontSize:10,color:"#94a3b8",marginTop:6}}>
              <summary style={{cursor:"pointer",color:"#64748b"}}>Stack trace</summary>
              <pre style={{whiteSpace:"pre-wrap",wordBreak:"break-word",marginTop:4}}>{this.state.errorInfo.componentStack}</pre>
            </details>
          )}
          <button onClick={this.reset} style={{marginTop:10,padding:"4px 10px",background:"#ef4444",color:"white",border:"none",borderRadius:4,cursor:"pointer",fontSize:11,fontFamily:F}}>Try Again</button>
        </div>
      );
    }
    return this.props.children;
  }
}

const Box = ({children,style})=><div style={{background:"rgba(255,255,255,0.015)",borderRadius:8,border:"1px solid rgba(255,255,255,0.04)",padding:16,...style}}>{children}</div>;
const Lbl = ({children})=><div style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:10}}>{children}</div>;
const Btn = ({children,active,color="#3b82f6",onClick,disabled,style:s})=>(
  <button onClick={onClick} disabled={disabled} style={{padding:"4px 10px",borderRadius:4,fontSize:11,fontFamily:F,
    cursor:disabled?"default":"pointer",border:`1px solid ${active?color:"rgba(255,255,255,0.06)"}`,
    background:active?`${color}18`:"transparent",color:active?color:disabled?"#334155":"#64748b",
    opacity:disabled?0.5:1,transition:"all 0.15s",...s}}>{children}</button>);

function SourceBadge({source,trained}) {
  const cfg={live:{c:"#22c55e",l:"● LIVE"},cached:{c:"#eab308",l:"LAST SCAN"},offline:{c:"#64748b",l:"OFFLINE"},loading:{c:"#64748b",l:"..."},error:{c:"#ef4444",l:"ERROR"}}[source]||{c:"#64748b",l:"?"};
  return (
    <div style={{display:"flex",gap:6,alignItems:"center"}}>
      <span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:`${cfg.c}18`,color:cfg.c,border:`1px solid ${cfg.c}30`}}>{cfg.l}</span>
      {trained&&<span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:"rgba(139,92,246,0.12)",color:"#8b5cf6",border:"1px solid rgba(139,92,246,0.2)"}}>LGBM FIRST-PASSAGE</span>}
    </div>);
}

function WinBar({winProb,ev,breakeven=0.612}) {
  const wp = typeof winProb === "number" && !isNaN(winProb) ? winProb : 0;
  const e = typeof ev === "number" && !isNaN(ev) ? ev : 0;
  const pct=(wp*100).toFixed(1);
  const c=wp>breakeven+0.09?"#22c55e":wp>breakeven+0.04?"#a3e635":wp>breakeven?"#eab308":wp>breakeven-0.06?"#f97316":"#6b7280";
  const evColor = e>0?"#22c55e":e<0?"#ef4444":"#64748b";
  return (
    <div style={{display:"flex",alignItems:"center",gap:6,minWidth:180}}>
      <div style={{width:50,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden",flexShrink:0}}>
        <div style={{width:`${Math.max(2,wp*100)}%`,height:"100%",background:c,borderRadius:3,transition:"width 0.4s"}}/>
      </div>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:12,color:c,fontWeight:600,minWidth:40}}>{pct}%</span>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:11,color:evColor,fontWeight:500,minWidth:44}}>
        {e>0?"+":""}{e.toFixed(2)}%
      </span>
    </div>);
}

function Fc({value,label}) {
  const v=parseFloat(value);let c="#94a3b8";
  if(["momentum","vwapDist","vwapSlope","trendStr","orbStrength"].includes(label)) c=v>0.4?"#22c55e":v>0.15?"#a3e635":v>0?"#94a3b8":v>-0.2?"#f97316":"#ef4444";
  else if(label==="relVolume") c=v>1.8?"#22c55e":v>1.2?"#a3e635":"#94a3b8";
  else if(label==="atrReach") c=v<0.8?"#22c55e":v<1.2?"#eab308":"#ef4444";
  return <span style={{color:c,fontVariantNumeric:"tabular-nums",fontSize:11.5}}>{value}</span>;
}

// v15: trigger a browser download of an object as JSON file
function downloadJson(data, filename) {
  try {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename || "download.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (e) {
    console.error("Download failed", e);
    alert("Download failed: " + e.message);
  }
}

// ─── SCANNER ─────────────────────────────────────────────────────
function ScannerTab({data,scanHour,source,elapsed,message,modelWR10,modelPnL10,health,scanInfo}) {
  const [mode,setMode]=useState("setup");

  if(source==="offline"||!data||data.length===0) return (
    <Box style={{padding:40,textAlign:"center"}}>
      <div style={{fontSize:14,color:"#64748b",marginBottom:8}}>{source==="offline"?"Scan unavailable":"No data"}</div>
      <div style={{fontSize:12,color:"#475569",maxWidth:600,margin:"0 auto"}}>{message||"Train model, then scan during market hours."}</div>
    </Box>);

  // v8: multipliers-based; derive notional BE from universe average
  const activeTpMult = scanInfo?.tp_mult ?? health?.tp_mult ?? 0.5;
  const activeSlMult = scanInfo?.sl_mult ?? health?.sl_mult ?? 2.5;
  const avgTp = scanInfo?.avgTpPct;
  const avgSl = scanInfo?.avgSlPct;
  const notionalBE = (activeSlMult/(activeSlMult+activeTpMult));
  const beThresh = notionalBE;
  const beThresh5 = beThresh + 0.05;

  // v17: Conviction = deduplicated count of firing setups per stock.
  // orb_vol + orb_60_break at the same scan hour count as 1 (redundant signal).
  const REDUNDANT_PAIRS = [["orb_vol", "orb_60_break"]];
  const convictionScore = (setupMatches) => {
    if (!Array.isArray(setupMatches) || setupMatches.length === 0) return 0;
    const names = setupMatches.map(m => m?.name).filter(Boolean);
    let count = names.length;
    // Dedup: for each redundant pair, if both present count only once
    for (const [a, b] of REDUNDANT_PAIRS) {
      if (names.includes(a) && names.includes(b)) count -= 1;
    }
    return Math.max(0, count);
  };

  const filtered = mode==="setup" ? data.filter(s=>s.setupMatches && s.setupMatches.length>0)
    : mode==="conv2" ? data.filter(s=>convictionScore(s.setupMatches) >= 2)
    : mode==="tradable" ? data.filter(s=>s.tradable)
    : mode==="be" ? data.filter(s=>s.winProb>=beThresh)
    : mode==="be5" ? data.filter(s=>s.winProb>=beThresh5)
    : mode==="posEV" ? data.filter(s=>s.ev>0)
    : data.slice(0, mode==="top10"?10:20);
  const posEV = data.filter(s=>s.ev>0);
  const tradable = data.filter(s=>s.tradable);
  const withSetup = data.filter(s=>s.setupMatches && s.setupMatches.length>0);
  const highConv = data.filter(s=>convictionScore(s.setupMatches) >= 2);
  const avgEV = posEV.length>0 ? posEV.reduce((s,r)=>s+r.ev,0)/posEV.length : 0;

  return (
    <div>
      <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:16,flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5}}>Show</span>
          {[["setup","★ Setup Active"],["conv2","★★ Conv≥2"],["tradable","✓ Tradable"],["posEV","+EV"],["be",`Win>${(beThresh*100).toFixed(0)}%`],["be5",`Win>${(beThresh5*100).toFixed(0)}%`],["top10","Top 10"],["top20","Top 20"]].map(([m,l])=>
            <Btn key={m} active={mode===m} onClick={()=>setMode(m)} color={m==="setup"?"#a855f7":m==="conv2"?"#d946ef":m==="tradable"?"#06b6d4":undefined}>{l}</Btn>)}
        </div>
        {scanInfo?.r2kBreadthLabel && (() => {
          const bl = scanInfo.r2kBreadthLabel;
          const frac = scanInfo.r2kBreadthFrac;
          const c = bl === "breadth_up" ? "#22c55e" : bl === "breadth_down" ? "#ef4444" : "#eab308";
          const pct = frac != null ? (frac*100).toFixed(0) : "?";
          const label = bl === "breadth_up" ? "BREADTH UP" : bl === "breadth_down" ? "BREADTH DOWN" : "BREADTH FLAT";
          const title = `R2K breadth: ${scanInfo.r2kGreenStocks}/${scanInfo.r2kTotalStocks} stocks green at ${scanHour}:00 ET (${pct}%). ${bl === "breadth_down" ? "Some setups historically underperform in breadth_down — affected matches are filtered or flagged." : ""}`;
          return <span title={title} style={{display:"inline-block",padding:"3px 8px",borderRadius:3,background:`${c}18`,border:`1px solid ${c}40`,color:c,fontSize:10,fontWeight:700,letterSpacing:0.5}}>
            {label} {pct}%
          </span>;
        })()}
        <span style={{fontSize:11,color:"#334155"}}>
          {scanHour}:00 ET —{" "}
          {withSetup.length > 0 ? (
            <>
              <span style={{color:"#a855f7",fontWeight:700}}>
                {withSetup.length} STOCK{withSetup.length===1?"":"S"} WITH FIRING SETUP{withSetup.length===1?"":"S"}
              </span>
              {highConv.length > 0 && (
                <span style={{color:"#d946ef",fontWeight:700,marginLeft:6}}>
                  · {highConv.length} w/ CONV≥2
                </span>
              )}
              {scanInfo?.setupFiringCounts && Object.keys(scanInfo.setupFiringCounts).length > 0 && (
                <span style={{color:"#64748b",marginLeft:8,fontSize:10}}>
                  ({Object.entries(scanInfo.setupFiringCounts).map(([n,c])=>`${n}:${c}`).join(", ")})
                </span>
              )}
            </>
          ) : scanInfo?.activeSetups?.length > 0 ? (
            <span style={{color:"#64748b"}}>
              0 setups firing ({scanInfo.activeSetups.map(s=>s.name).join(", ")} active at this hour)
            </span>
          ) : (
            <span style={{color:"#64748b"}}>no setups tracked at this hour</span>
          )}
          {scanInfo?.threshold && (
            <>
              {" · "}
              <span style={{color:tradable.length>0?"#06b6d4":"#64748b"}}>
                {tradable.length} tradable
              </span>
            </>
          )}
          {avgTp!=null&&avgSl!=null&&` — avg TP +${avgTp}% / SL -${avgSl}%`}
          {elapsed!=null&&` — ${elapsed}ms`}
          {modelWR10!=null&&` — val WR@10 ${(modelWR10*100).toFixed(0)}%`}
          {modelPnL10!=null&&` — val PnL@10 ${modelPnL10>0?"+":""}${modelPnL10}%`}
        </span>
      </div>

      {filtered.length===0 ? (
        <Box style={{padding:20,textAlign:"center",color:"#64748b",fontSize:12}}>
          {mode==="setup" ? (
            <>
              <div>No stocks currently have firing setups at {scanHour}:00 ET.</div>
              {scanInfo?.activeSetups?.length > 0 ? (
                <div style={{marginTop:6,fontSize:11,color:"#475569"}}>
                  Setups tracked at this hour: {scanInfo.activeSetups.map(s=>s.name).join(", ")}. None are currently matching any stock.
                </div>
              ) : (
                <div style={{marginTop:6,fontSize:11,color:"#475569"}}>
                  No setups have been validated for {scanHour}:00 ET. Run setup evaluation in the Setups tab.
                </div>
              )}
            </>
          ) : mode==="tradable" ? (
            "No stocks currently clear the ML threshold. (Note: ML threshold gating is deprecated — use ★ Setup Active instead.)"
          ) : mode==="posEV" ? (
            "No stocks with positive EV at this scan hour. (Note: EV is from the deprecated ML model — use ★ Setup Active to see setup firings.)"
          ) : (
            `No stocks match ${mode} filter at this scan hour.`
          )}
        </Box>
      ) : (
        <Box style={{padding:12}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
            <span style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase"}}>
              {filtered.length} stocks — TP {activeTpMult}×ATR / SL {activeSlMult}×ATR{avgTp!=null&&` (avg ${avgTp}% / ${avgSl}%)`} / Close 15:55 (notional BE {(notionalBE*100).toFixed(0)}%)
            </span>
            {posEV.length>0&&<span style={{fontSize:11,color:"#22c55e"}}>Avg EV (positive): +{avgEV.toFixed(3)}%</span>}
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
              <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                {["#","Ticker","Conv","Sector","Price","TP $","SL $","Chg%","Win%","EV","Mom","RelVol","VWAP%","ATR","Vol","RSI","vsSPY","vsSect","Breadth","Gap"].map(h=>(
                  <th key={h} style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase",whiteSpace:"nowrap"}}>{h}</th>))}
              </tr></thead>
              <tbody>{filtered.map((s,i)=>{const chg=parseFloat(s.changeFromOpen);const evPos=s.ev>0;return(
                <tr key={s.ticker+s.rank} style={{borderBottom:"1px solid rgba(255,255,255,0.03)",
                  background:evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}}
                  onMouseEnter={e=>e.currentTarget.style.background="rgba(255,255,255,0.04)"}
                  onMouseLeave={e=>e.currentTarget.style.background=evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}>
                  <td style={{padding:"5px 6px",color:"#475569",fontWeight:600,fontSize:11}}>{s.rank}</td>
                  <td style={{padding:"5px 6px",fontWeight:700,color:s.tradable?"#06b6d4":"#e2e8f0",letterSpacing:0.3}}>
                    {s.tradable && <span title="TRADABLE — top-1 pick and clears threshold for this scan hour" style={{color:"#06b6d4",marginRight:4,fontSize:13,fontWeight:900}}>✓</span>}
                    {s.setupMatches && s.setupMatches.length > 0 && s.setupMatches.map((m,mi)=>{
                      const warn = m?.weak_breadth_warning;
                      const tierColor = warn ? "#f97316" : m?.tier==="strong" ? "#22c55e" : "#eab308";
                      const name = typeof m?.name === "string" ? m.name : "unknown";
                      const baseTitle = `Setup: ${name} (${m?.tier||"?"}). Test hit rate ${m?.test_hit_rate}% vs base ${m?.base_hit_rate}% (edge +${m?.test_edge}%). ${m?.description||""}`;
                      const warnTitle = warn ? ` ⚠ WEAK BREADTH WARNING: this setup historically underperformed in breadth_down regimes. Current R2K breadth is down.` : "";
                      const bgRgb = warn ? '249,115,22' : m?.tier==='strong' ? '34,197,94' : '234,179,8';
                      return <span key={mi} title={baseTitle + warnTitle} style={{display:"inline-block",padding:"1px 5px",marginRight:3,borderRadius:2,background:`rgba(${bgRgb},${warn?0.2:0.15})`,border:warn?`1px solid #f97316`:"none",color:tierColor,fontSize:9,fontWeight:700,letterSpacing:0.3,textTransform:"uppercase"}}>
                        {warn && <span style={{marginRight:3}}>⚠</span>}{name.replace(/_/g,"·")}
                      </span>;
                    })}
                    {s.patternMatch && <span title={`${s.patternMatch.count} pattern match(es). Best: val WR ${s.patternMatch.top_wr_val}%, edge +${s.patternMatch.top_edge_val}%, n_val ${s.patternMatch.top_n_val}`} style={{color:"#f59e0b",marginRight:4,fontSize:13}}>⚡</span>}
                    {s.ticker}
                  </td>
                  <td style={{padding:"5px 6px",fontVariantNumeric:"tabular-nums"}}>
                    {(()=>{
                      const conv = convictionScore(s.setupMatches);
                      const c = conv>=3?"#d946ef":conv===2?"#a855f7":conv===1?"#94a3b8":"#334155";
                      const w = conv>=2?700:400;
                      const title = conv===0?"No survivor setups firing":
                        conv===1?"1 setup firing":
                        `${conv} independent setups firing simultaneously (orb_vol + orb_60_break count as 1 — redundant per v16 overlap analysis). Stacking analysis shows multi-setup firings have meaningfully higher hit rates.`;
                      return <span title={title} style={{color:c,fontWeight:w}}>{conv}</span>;
                    })()}
                  </td>
                  <td style={{padding:"5px 6px",color:"#64748b",fontSize:11}}>{s.sector}</td>
                  <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>${s.price}</td>
                  <td title={`+${s.tpPct}% (ATR ${s.atrPct}%)`} style={{padding:"5px 6px",color:"#22c55e",fontVariantNumeric:"tabular-nums",fontSize:11,fontWeight:500}}>{s.tpPrice!=null?`$${s.tpPrice}`:"—"}</td>
                  <td title={`-${s.slPct}% (ATR ${s.atrPct}%)`} style={{padding:"5px 6px",color:"#ef4444",fontVariantNumeric:"tabular-nums",fontSize:11,fontWeight:500}}>{s.slPrice!=null?`$${s.slPrice}`:"—"}</td>
                  <td style={{padding:"5px 6px",color:chg>0?"#22c55e":chg<0?"#ef4444":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{isNaN(chg)?"—":`${chg>0?"+":""}${chg}%`}</td>
                  <td style={{padding:"5px 6px"}} colSpan={2}><WinBar winProb={s.winProb} ev={s.ev} breakeven={beThresh}/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.momentum ?? "—"} label="momentum"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.relVolume ?? "—"} label="relVolume"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.vwapDist ?? "—"} label="vwapDist"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.atrReach ?? "—"} label="atrReach"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.realizedVol ?? "—"} label="realizedVol"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.rsi ?? "—"} label="rsi"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.retVsSpy ?? "—"} label="retVsSpy"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.retVsSector ?? "—"} label="retVsSector"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.sectorBreadth ?? "—"} label="sectorBreadth"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features?.gapPct ?? "—"} label="gapPct"/></td>
                </tr>);})}</tbody>
            </table>
          </div>
        </Box>
      )}
    </div>);
}

// ─── TRAINING ────────────────────────────────────────────────────
// ─── SWEEP (grid search over TP/SL) ──────────────────────────────
function SweepSection() {
  const [status,setStatus]=useState(null);
  const [results,setResults]=useState(null);
  const [topN,setTopN]=useState(10);  // which top-N to highlight in the heatmap

  const poll=useCallback(()=>{
    fetch('/api/sweep/status').then(r=>r.json()).then(setStatus).catch(()=>{});
    fetch('/api/sweep/results').then(r=>r.json()).then(setResults).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,3000);return()=>clearInterval(iv);},[poll]);

  const runSweep=async()=>{
    if(!confirm("Run grid search? Each cell trains a full model suite. 3 cells × ~3-5 min each ≈ 10-15 minutes. Scanner offline during sweep. Runs in background — safe to close browser.")) return;
    await fetch('/api/sweep',{method:'POST'});
    poll();
  };
  const resetSweep=async()=>{
    if(!confirm("Discard all sweep results? The next sweep will start from scratch.")) return;
    await fetch('/api/sweep/reset',{method:'POST'});
    poll();
  };

  const ip = status?.inProgress;
  const grid = status?.grid;
  const gridResults = results?.grid || [];
  const resultMap = {};
  gridResults.forEach(r => { resultMap[`${r.tp_pct}_${r.sl_pct}`] = r; });
  const total = grid ? grid.tp.length * grid.sl.length : 0;
  const done = gridResults.length;

  // Find best cell by selected top-N edge (falls back to legacy avg_edge)
  const best = gridResults.filter(r=>r[`avg_top${topN}_edge`]!=null || r.avg_edge!=null)
    .sort((a,b)=>((b[`avg_top${topN}_edge`]??b.avg_edge)||0)-((a[`avg_top${topN}_edge`]??a.avg_edge)||0))[0];

  // Color scale for edge values: red (negative) -> yellow (0) -> green (positive)
  const edgeColor = (edge) => {
    if (edge == null) return "rgba(100,116,139,0.1)";
    if (edge >= 5) return "rgba(34,197,94,0.4)";
    if (edge >= 3) return "rgba(34,197,94,0.25)";
    if (edge >= 1) return "rgba(163,230,53,0.22)";
    if (edge >= 0) return "rgba(234,179,8,0.18)";
    if (edge >= -3) return "rgba(249,115,22,0.18)";
    return "rgba(239,68,68,0.22)";
  };

  return (
    <Box>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
        <Lbl>TP/SL Grid Search — {(grid?.tp?.length||0)} TP × {(grid?.sl?.length||0)} SL Combinations</Lbl>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <span style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5}}>Strategy:</span>
          {[1,3,5,10].map(N=><Btn key={N} active={topN===N} onClick={()=>setTopN(N)} style={{padding:"3px 8px",fontSize:10}}>Top-{N}</Btn>)}
          <span style={{color:"#334155",margin:"0 4px"}}>|</span>
          <Btn onClick={runSweep} disabled={ip} color="#f97316" style={{padding:"4px 10px",fontSize:11}}>
            {ip?`Running ${status?.current}/${status?.total}`:done>0&&done<total?`Resume (${done}/${total})`:"Run Sweep"}
          </Btn>
          {done>0&&!ip&&<Btn onClick={resetSweep} color="#ef4444" style={{padding:"4px 10px",fontSize:11}}>Reset</Btn>}
        </div>
      </div>

      {ip&&(
        <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(249,115,22,0.08)",border:"1px solid rgba(249,115,22,0.2)"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
            <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${(status.current/status.total)*100}%`,height:"100%",background:"#f97316",borderRadius:3,transition:"width 0.5s"}}/>
            </div>
            <span style={{fontSize:11,color:"#f97316",fontWeight:600}}>{status.current}/{status.total}</span>
          </div>
          <div style={{fontSize:11,color:"#94a3b8"}}>{status.message}</div>
        </div>
      )}

      {done===0&&!ip ? (
        <div style={{fontSize:12,color:"#64748b",padding:"20px 0",textAlign:"center"}}>
          Grid: TP {grid?.tp.join("%, ")||""}% × SL {grid?.sl.join("%, ")||""}%<br/>
          Each cell requires a full train cycle. First cell fetches bars (~15 min), subsequent use cache (~2-3 min).<br/>
          Safe to close browser — runs in background. Results save after each cell.
        </div>
      ) : (
        <>
          {best && (
            <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(34,197,94,0.06)",border:"1px solid rgba(34,197,94,0.2)"}}>
              <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:2}}>Best cell so far</div>
              <div style={{fontSize:13,color:"#e2e8f0"}}>
                <span style={{color:"#22c55e",fontWeight:700}}>TP {best.tp_pct}% / SL {best.sl_pct}%</span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Top-{topN} WR: <span style={{color:"#e2e8f0",fontWeight:600}}>{best[`avg_top${topN}_wr`] ?? best.avg_top10_wr}%</span></span>
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                {best.avg_realized_be!=null ? (
                  <>
                    <span>Realized BE: <span style={{color:"#eab308",fontWeight:600}}>{best.avg_realized_be}%</span></span>
                    <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                    <span>Edge: <span style={{color:(best[`avg_top${topN}_edge`] ?? best.realized_edge)>0?"#22c55e":"#ef4444",fontWeight:700}}>{(best[`avg_top${topN}_edge`] ?? best.realized_edge)>0?"+":""}{best[`avg_top${topN}_edge`] ?? best.realized_edge}%</span></span>
                  </>
                ) : (
                  <>
                    <span>BE: <span style={{color:"#eab308",fontWeight:600}}>{best.breakeven}%</span></span>
                    <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                    <span>Edge: <span style={{color:best.avg_edge>0?"#22c55e":"#ef4444",fontWeight:700}}>{best.avg_edge>0?"+":""}{best.avg_edge}%</span></span>
                  </>
                )}
                <span style={{margin:"0 10px",color:"#475569"}}>|</span>
                <span>Top-{topN} PnL: <span style={{color:(best[`avg_top${topN}_pnl`] ?? best.avg_top10_pnl)>0?"#22c55e":"#ef4444",fontWeight:600}}>{(best[`avg_top${topN}_pnl`] ?? best.avg_top10_pnl)>0?"+":""}{best[`avg_top${topN}_pnl`] ?? best.avg_top10_pnl}%</span></span>
              </div>
            </div>
          )}

          <div style={{marginBottom:8,fontSize:10,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase"}}>
            Viewing Top-{topN} strategy. Toggle buttons above to compare. Realized BE uses actual loser P&L distribution.
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{borderCollapse:"collapse",fontSize:11}}>
              <thead>
                <tr>
                  <th style={{padding:"6px 10px",textAlign:"right",color:"#64748b",fontSize:10,fontWeight:500}}>SL ↓ / TP →</th>
                  {grid?.tp.map(tp => (
                    <th key={tp} style={{padding:"6px 10px",textAlign:"center",color:"#94a3b8",fontSize:11,fontWeight:700}}>{tp}%</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid?.sl.map(sl => (
                  <tr key={sl}>
                    <td style={{padding:"6px 10px",textAlign:"right",color:"#94a3b8",fontWeight:700,fontSize:11}}>{sl}%</td>
                    {grid.tp.map(tp => {
                      const cell = resultMap[`${tp}_${sl}`];
                      const running = ip && status?.currentTP===tp && status?.currentSL===sl;
                      // Use selected top-N metrics if available
                      const cellWR = cell?.[`avg_top${topN}_wr`] ?? cell?.avg_top10_wr;
                      const cellPnL = cell?.[`avg_top${topN}_pnl`] ?? cell?.avg_top10_pnl;
                      const cellEdge = cell?.[`avg_top${topN}_edge`] ?? cell?.realized_edge ?? cell?.avg_edge;
                      return (
                        <td key={tp} style={{
                          padding:"6px",border:"1px solid rgba(255,255,255,0.06)",
                          background:running?"rgba(249,115,22,0.2)":cell?edgeColor(cellEdge):"rgba(100,116,139,0.05)",
                          minWidth:140,textAlign:"center"
                        }}>
                          {running ? (
                            <div style={{color:"#f97316",fontSize:10,fontWeight:600}}>Running...</div>
                          ) : cell ? (
                            <div>
                              <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5}}>Top-{topN}</div>
                              <div style={{fontSize:14,fontWeight:700,color:cellEdge>0?"#22c55e":cellEdge<-3?"#ef4444":"#eab308",fontVariantNumeric:"tabular-nums"}}>
                                {cellEdge>0?"+":""}{cellEdge}%
                              </div>
                              {cell.avg_realized_be != null ? (
                                <div style={{fontSize:9,color:"#94a3b8",marginTop:1}}>
                                  WR {cellWR}% / rBE {cell.avg_realized_be}%
                                </div>
                              ) : (
                                <div style={{fontSize:9,color:"#94a3b8",marginTop:1}}>
                                  WR {cellWR}% / BE {cell.breakeven}%
                                </div>
                              )}
                              <div style={{fontSize:9,color:cellPnL>0?"#22c55e":"#ef4444",marginTop:1,fontVariantNumeric:"tabular-nums"}}>
                                PnL {cellPnL>0?"+":""}{cellPnL}%
                              </div>
                            </div>
                          ) : (
                            <div style={{color:"#334155",fontSize:10}}>—</div>
                          )}
                        </td>);
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:8,lineHeight:1.5}}>
            rBE = realized break-even (computed from actual loser P&L distribution). When SL is set wide (e.g. 5%), losers that close at 15:55 lose far less than the full stop — realized break-even is lower than nominal. Edge uses realized BE when available.
          </div>
        </>
      )}
    </Box>);
}

function TrainingTab() {
  const [d,setD]=useState(null);
  const [ld,setLd]=useState(true);
  const [sh,setSh]=useState(11);

  const poll=useCallback(()=>{fetch('/api/training/progress').then(r=>r.json()).then(d=>{setD(d);setLd(false);}).catch(()=>setLd(false));},[]);
  useEffect(()=>{poll();const iv=setInterval(poll,2000);return()=>clearInterval(iv);},[poll]);

  // v8: sliders are ATR multipliers, not fixed percentages
  const [tp,setTp]=useState(0.5);
  const [sl,setSl]=useState(2.5);
  const notionalBE = (sl/(sl+tp)*100).toFixed(1);

  // v17: extend history state
  const [extMonths, setExtMonths] = useState(12);
  const [extProg, setExtProg] = useState(null);
  const pollExt = useCallback(()=>{
    fetch('/api/cache/extend_history/progress').then(r=>r.json()).then(setExtProg).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollExt();
    const iv=setInterval(pollExt, 3000);
    return ()=>clearInterval(iv);
  },[pollExt]);

  // v18: ETF repair state (fetch SPY + IWM for full cache range)
  const [repairProg, setRepairProg] = useState(null);
  const pollRepair = useCallback(()=>{
    fetch('/api/cache/repair_etf/progress').then(r=>r.json()).then(setRepairProg).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollRepair();
    const iv=setInterval(pollRepair, 3000);
    return ()=>clearInterval(iv);
  },[pollRepair]);

  // v25: conviction model training state
  const [convProg, setConvProg] = useState(null);
  const [convResults, setConvResults] = useState(null);
  const pollConv = useCallback(()=>{
    fetch('/api/conviction/progress').then(r=>r.json()).then(setConvProg).catch(()=>{});
    fetch('/api/conviction/results').then(r=>r.json()).then(d=>{
      if (d && !d.status) setConvResults(d);
    }).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollConv();
    const iv=setInterval(pollConv, 3000);
    return ()=>clearInterval(iv);
  },[pollConv]);

  // v27: pattern discovery state
  const [patProg, setPatProg] = useState(null);
  const [patResults, setPatResults] = useState(null);
  const pollPat = useCallback(()=>{
    fetch('/api/pattern/progress').then(r=>r.json()).then(setPatProg).catch(()=>{});
    fetch('/api/pattern/results').then(r=>r.json()).then(d=>{
      if (d && !d.status) setPatResults(d);
    }).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollPat();
    const iv=setInterval(pollPat, 3000);
    return ()=>clearInterval(iv);
  },[pollPat]);

  // v28: cost-adjusted analysis state
  const [v28Prog, setV28Prog] = useState(null);
  const [v28Results, setV28Results] = useState(null);
  const pollV28 = useCallback(()=>{
    fetch('/api/v28/progress').then(r=>r.json()).then(setV28Prog).catch(()=>{});
    fetch('/api/v28/results').then(r=>r.json()).then(d=>{
      if (d && !d.status) setV28Results(d);
    }).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollV28();
    const iv=setInterval(pollV28, 3000);
    return ()=>clearInterval(iv);
  },[pollV28]);

  // v29: fine-grained target sweep state
  const [v29Prog, setV29Prog] = useState(null);
  const [v29Results, setV29Results] = useState(null);
  const pollV29 = useCallback(()=>{
    fetch('/api/v29/progress').then(r=>r.json()).then(setV29Prog).catch(()=>{});
    fetch('/api/v29/results').then(r=>r.json()).then(d=>{
      if (d && !d.status) setV29Results(d);
    }).catch(()=>{});
  },[]);
  useEffect(()=>{
    pollV29();
    const iv=setInterval(pollV29, 3000);
    return ()=>clearInterval(iv);
  },[pollV29]);

  const trigTrain=async()=>{
    await fetch('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({tp_mult:tp,sl_mult:sl})});
    poll();
  };
  const clearCache=async()=>{
    if(!confirm("Clear cached bar data? Next training will refetch 12 months from Alpaca (10-15 min).")) return;
    const r=await fetch('/api/cache/clear',{method:'POST'});
    const d=await r.json();
    alert(d.error?`Error: ${d.error}`:`Cleared: ${d.deleted.join(", ")||"nothing"}`);
  };
  const extendHistory=async()=>{
    if(!confirm(`Fetch ${extMonths} months of bars BEFORE the current cache's earliest date? This will take ~${(extMonths*1.2).toFixed(0)}-${(extMonths*1.7).toFixed(0)} minutes and append to existing cache without overwriting. After completion, re-run Setup Evaluation to use the extended data.`)) return;
    const r=await fetch(`/api/cache/extend_history?months=${extMonths}`,{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollExt();
  };
  const repairEtf=async()=>{
    if(!confirm("Fetch SPY + IWM bars for the full cache date range? Needed because prior extend_history runs lost ETF coverage. Takes ~3-5 minutes.")) return;
    const r=await fetch('/api/cache/repair_etf',{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollRepair();
  };
  const trainConvictionModel=async()=>{
    if(!confirm("Train conviction model? This builds features for every (date × stock × scan hour) in 2 years of data, trains LightGBM to predict `hit +1% before close` binary, and reports calibration by probability bucket. Takes ~15-30 minutes.")) return;
    const r=await fetch('/api/conviction/train',{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollConv();
  };
  const runPatternDiscovery=async()=>{
    if(!confirm("Run pattern discovery? Builds features for every (date × stock × scan hour), then for each (scan_hour × target +0.75%/+1%) induces a decision tree (depth 4) and validates rules on test fold (strict: n≥30, hit ≥ base+40pp, CI lower ≥ base+30pp). Takes ~20-30 minutes.")) return;
    const r=await fetch('/api/pattern/train',{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollPat();
  };
  const runV28=async()=>{
    if(!confirm("Run v28 cost-adjusted analysis? Runs BOTH LightGBM calibration AND decision-tree pattern discovery across 3 cost-adjusted targets (+0.30%, +0.40%, +0.50%). Takes ~30-45 minutes.")) return;
    const r=await fetch('/api/v28/train',{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollV28();
  };
  const runV29=async()=>{
    if(!confirm("Run v29 fine-grained target sweep? Tests every 1bps target from +0.31% to +0.40% using LightGBM classifier. Pinpoints the highest target that still passes the strict 80% bar. Takes ~25-40 minutes.")) return;
    const r=await fetch('/api/v29/train',{method:'POST'});
    const d=await r.json();
    if(d.error) alert(`Error: ${d.error}`);
    pollV29();
  };

  if(ld) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  const ip=d?.inProgress,pg=d||{},meta=d?.meta||{};
  const sm=meta[String(sh)];
  const activeTpMult = Object.values(meta)[0]?.tp_mult;
  const activeSlMult = Object.values(meta)[0]?.sl_mult;
  const activeNotionalBE = activeTpMult && activeSlMult ? (activeSlMult/(activeSlMult+activeTpMult)*100).toFixed(1) : null;

  const Slider = ({label,value,setValue,min,max,step,color,suffix="×ATR"})=>(
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
        <span style={{fontSize:11,color:"#94a3b8",fontWeight:500}}>{label}</span>
        <span style={{fontSize:13,color:color,fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{value.toFixed(2)}{suffix}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e=>setValue(parseFloat(e.target.value))}
        disabled={ip}
        style={{width:"100%",accentColor:color,cursor:ip?"not-allowed":"pointer"}}/>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#475569",marginTop:2}}>
        <span>{min}{suffix}</span><span>{max}{suffix}</span>
      </div>
    </div>);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Model Training — Volatility-Adjusted First-Passage (v8)</Lbl>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          <div>
            <div style={{fontSize:11,color:"#64748b",marginBottom:10,textTransform:"uppercase",letterSpacing:0.5}}>
              Strategy Parameters — ATR Multipliers
            </div>
            <Slider label="Take Profit" value={tp} setValue={setTp} min={0.1} max={3.0} step={0.05} color="#22c55e"/>
            <Slider label="Stop Loss" value={sl} setValue={setSl} min={0.1} max={5.0} step={0.05} color="#ef4444"/>
            <div style={{marginTop:12,padding:"8px 10px",borderRadius:4,
              background:"rgba(6,182,212,0.06)",border:"1px solid rgba(6,182,212,0.15)"}}>
              <div style={{fontSize:11,color:"#94a3b8",marginBottom:2}}>Volatility-adjusted barriers</div>
              <div style={{fontSize:12,color:"#06b6d4",fontWeight:600,marginBottom:4}}>
                TP = entry × (1 + {tp}×ATR) · SL = entry × (1 − {sl}×ATR)
              </div>
              <div style={{fontSize:10,color:"#64748b",lineHeight:1.5}}>
                ATR = 14-day Average True Range as % of close. A stock with 2% ATR and TP={tp}×ATR gets TP at +{(tp*2).toFixed(2)}%.
                Notional breakeven {notionalBE}% (actual per-stock BE varies with volatility). R:R = {(tp/sl).toFixed(2)}:1 in ATR units.
              </div>
            </div>
          </div>
          <div>
            <div style={{fontSize:11,color:"#64748b",marginBottom:10,textTransform:"uppercase",letterSpacing:0.5}}>
              Current Status
            </div>
            <div style={{fontSize:12,lineHeight:2,color:"#94a3b8",marginBottom:14}}>
              {[
                {l:"Models",v:Object.keys(meta).length>0?`${Object.keys(meta).length} hours trained`:"None",ok:Object.keys(meta).length>0},
                {l:"Active multipliers",v:activeTpMult?`${activeTpMult}×ATR / ${activeSlMult}×ATR`:"—",ok:!!activeTpMult},
                {l:"Notional breakeven",v:activeNotionalBE?`${activeNotionalBE}%`:"—",ok:!!activeNotionalBE},
                {l:"Trained",v:Object.values(meta)[0]?.trained_at?new Date(Object.values(meta)[0].trained_at).toLocaleString():"Never",ok:Object.keys(meta).length>0},
              ].map((c,i)=>(
                <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
                  <span style={{width:12,height:12,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
                    background:c.ok?"rgba(34,197,94,0.15)":"rgba(100,116,139,0.15)",color:c.ok?"#22c55e":"#64748b",fontSize:9,fontWeight:900}}>{c.ok?"✓":"·"}</span>
                  <span style={{minWidth:130,fontSize:11}}>{c.l}</span>
                  <span style={{color:c.ok?"#e2e8f0":"#64748b",fontWeight:500}}>{c.v}</span>
                </div>))}
            </div>
            <div style={{display:"flex",gap:8}}>
              <Btn onClick={trigTrain} disabled={ip} color="#8b5cf6" style={{padding:"8px 16px",fontSize:12}}>
                {ip?"Training...":`Train (TP ${tp}% / SL ${sl}%)`}
              </Btn>
              <Btn onClick={clearCache} disabled={ip} color="#ef4444" style={{padding:"8px 12px",fontSize:11}}>
                Clear cache
              </Btn>
            </div>
            {ip&&(
              <div style={{marginTop:10}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${pg.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
                  </div>
                  <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600}}>{pg.pct||0}%</span>
                </div>
                <div style={{fontSize:11,color:"#64748b"}}>{pg.message}</div>
              </div>)}
            <div style={{marginTop:10,fontSize:10,color:"#475569",lineHeight:1.5}}>
              First training fetches 12 months of bars (~15 min). Re-training with different TP/SL uses cached bars (~2-3 min).
            </div>
          </div>
        </div>
      </Box>

      {/* v17: Extend History — fetch additional months before existing cache */}
      <Box>
        <Lbl>Extend History (v17) — add older data to enrich fold-swap validation</Lbl>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:20}}>
          <div>
            <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.6,marginBottom:12}}>
              Fetches N months of 5-min bars BEFORE the current cache's earliest date and appends them. Existing data is preserved. After completion, re-run Setup Evaluation to include the extended data — fold-swap will automatically produce additional test folds (F_F, F_G, ...) for stronger generalization testing.
            </div>
            <div style={{marginBottom:10}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                <span style={{fontSize:11,color:"#94a3b8"}}>Months to extend</span>
                <span style={{fontSize:13,color:"#06b6d4",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{extMonths} mo</span>
              </div>
              <input type="range" min={3} max={36} step={3} value={extMonths}
                onChange={e=>setExtMonths(parseInt(e.target.value))}
                disabled={extProg?.inProgress}
                style={{width:"100%",accentColor:"#06b6d4",cursor:extProg?.inProgress?"not-allowed":"pointer"}}/>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#475569",marginTop:2}}>
                <span>3 mo</span><span>36 mo</span>
              </div>
            </div>
            <div style={{fontSize:10,color:"#475569",lineHeight:1.5,marginBottom:12}}>
              Estimated time: ~{(extMonths*1.2).toFixed(0)}-{(extMonths*1.7).toFixed(0)} min (depends on Alpaca rate limits).
            </div>
          </div>
          <div>
            <div style={{fontSize:11,color:"#64748b",marginBottom:10,textTransform:"uppercase",letterSpacing:0.5}}>
              Status
            </div>
            <div style={{fontSize:12,lineHeight:2,color:"#94a3b8",marginBottom:14}}>
              <div style={{display:"flex",alignItems:"center",gap:8}}>
                <span style={{width:12,height:12,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
                  background:extProg?.inProgress?"rgba(6,182,212,0.15)":"rgba(100,116,139,0.15)",
                  color:extProg?.inProgress?"#06b6d4":"#64748b",fontSize:9,fontWeight:900}}>
                  {extProg?.inProgress?"⟳":"·"}
                </span>
                <span style={{minWidth:100,fontSize:11}}>Phase</span>
                <span style={{color:"#e2e8f0",fontWeight:500}}>{extProg?.phase||"idle"}</span>
              </div>
            </div>
            <div style={{display:"flex",gap:8}}>
              <Btn onClick={extendHistory} disabled={extProg?.inProgress||repairProg?.inProgress||ip} color="#06b6d4" style={{padding:"8px 16px",fontSize:12}}>
                {extProg?.inProgress?"Extending...":`Extend History ${extMonths}mo`}
              </Btn>
              <Btn onClick={repairEtf} disabled={extProg?.inProgress||repairProg?.inProgress||ip} color="#f97316" style={{padding:"8px 12px",fontSize:12}} title="Fetch SPY + IWM bars for full cache date range. Fixes ETF coverage gaps.">
                {repairProg?.inProgress?"Repairing...":"Repair SPY/IWM"}
              </Btn>
            </div>
            {extProg?.inProgress&&(
              <div style={{marginTop:10}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${extProg.pct||0}%`,height:"100%",background:"#06b6d4",borderRadius:3,transition:"width 0.5s"}}/>
                  </div>
                  <span style={{fontSize:11,color:"#06b6d4",fontWeight:600}}>{extProg.pct||0}%</span>
                </div>
                <div style={{fontSize:11,color:"#64748b"}}>{extProg.message}</div>
              </div>)}
            {repairProg?.inProgress&&(
              <div style={{marginTop:10}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${repairProg.pct||0}%`,height:"100%",background:"#f97316",borderRadius:3,transition:"width 0.5s"}}/>
                  </div>
                  <span style={{fontSize:11,color:"#f97316",fontWeight:600}}>{repairProg.pct||0}%</span>
                </div>
                <div style={{fontSize:11,color:"#64748b"}}>{repairProg.message}</div>
              </div>)}
            {!extProg?.inProgress && extProg?.phase === "done" && (
              <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(34,197,94,0.08)",border:"1px solid rgba(34,197,94,0.2)"}}>
                <div style={{fontSize:11,color:"#22c55e",fontWeight:600}}>✓ {extProg.message}</div>
              </div>
            )}
            {!extProg?.inProgress && extProg?.phase === "error" && (
              <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
                <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {extProg.message}</div>
              </div>
            )}
            {!repairProg?.inProgress && repairProg?.phase === "done" && (
              <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(34,197,94,0.08)",border:"1px solid rgba(34,197,94,0.2)"}}>
                <div style={{fontSize:11,color:"#22c55e",fontWeight:600}}>✓ {repairProg.message}</div>
              </div>
            )}
            {!repairProg?.inProgress && repairProg?.phase === "error" && (
              <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
                <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {repairProg.message}</div>
              </div>
            )}
          </div>
        </div>
      </Box>

      {/* v25: CONVICTION MODEL — train & calibrate */}
      <Box>
        <Lbl>Conviction Model (v25) — train classifier + report calibration by probability bucket</Lbl>
        <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
          Trains a LightGBM binary classifier on the full 2-year dataset (every date × stock × scan hour, NOT just setup firings). Target: <b>did price hit scan × 1.01 before 15:55 ET close?</b> Uses 36 base features + 20 setup-firing flags + scan-hour one-hots. Three-way temporal split. Isotonic calibration on val fold. Reports calibration buckets on held-out test fold with Wilson 95% confidence intervals. <b>Pass condition: n≥30 in 0.8-0.9 or 0.9+ bucket AND CI lower bound ≥75%.</b>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
          <Btn onClick={trainConvictionModel} disabled={convProg?.inProgress||ip||extProg?.inProgress||repairProg?.inProgress} color="#8b5cf6" style={{padding:"8px 16px",fontSize:12}}>
            {convProg?.inProgress?"Training...":"Train Conviction Model"}
          </Btn>
          {convResults && !convProg?.inProgress && (
            <>
              <Btn onClick={()=>downloadJson(convResults,`conviction_results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"8px 12px",fontSize:12}}>
                Download JSON
              </Btn>
              <span style={{fontSize:11,color:"#64748b"}}>
                Last trained: {convResults.generated_at?.slice(0,19).replace("T"," ")}
              </span>
            </>
          )}
        </div>
        {convProg?.inProgress && (
          <div style={{marginTop:10,marginBottom:10}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${convProg.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600}}>{convProg.pct||0}%</span>
            </div>
            <div style={{fontSize:11,color:"#64748b"}}>{convProg.message}</div>
          </div>
        )}
        {!convProg?.inProgress && convProg?.phase === "error" && (
          <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
            <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {convProg.message}</div>
          </div>
        )}
        {convResults && (
          <div style={{marginTop:14}}>
            {/* Overall verdict */}
            <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,
              background: convResults.overall_verdict==="ACHIEVES_80_HIGH_CONFIDENCE"?"rgba(34,197,94,0.08)":"rgba(234,179,8,0.08)",
              border: `1px solid ${convResults.overall_verdict==="ACHIEVES_80_HIGH_CONFIDENCE"?"rgba(34,197,94,0.25)":"rgba(234,179,8,0.25)"}`}}>
              <span style={{fontSize:11,color:"#64748b",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3}}>Overall verdict</span>
              <span style={{marginLeft:10,color: convResults.overall_verdict==="ACHIEVES_80_HIGH_CONFIDENCE"?"#22c55e":"#eab308",fontWeight:700,letterSpacing:0.3}}>{convResults.overall_verdict}</span>
              <span style={{marginLeft:16,color:"#94a3b8",fontSize:11}}>
                Passing: <b>{(convResults.passing_targets||[]).length}</b>/{(convResults.target_keys||[]).length}
                {"  ·  "}Features: <b>{convResults.n_features}</b>
              </span>
            </div>

            {/* Fold sizes */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8,marginBottom:14}}>
              {["train","val","test"].map(f=>{
                const n = convResults.fold_sizes?.[f];
                return <div key={f} style={{padding:"8px 10px",borderRadius:4,background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)"}}>
                  <div style={{fontSize:9,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",marginBottom:3}}>{f}</div>
                  <div style={{fontSize:13,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
                    <b>{n?.toLocaleString() ?? "—"}</b> <span style={{color:"#64748b",fontSize:11}}>examples</span>
                  </div>
                </div>;
              })}
            </div>

            {/* Matrix summary: 2 targets × 4 horizons */}
            <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:4}}>Target × Horizon Matrix</div>
            <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:14}}>
              <thead><tr>
                <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Target</th>
                {[30,60,120,180].map(h=>
                  <th key={h} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderLeft:"1px solid rgba(255,255,255,0.08)"}}>{h}min</th>
                )}
              </tr></thead>
              <tbody>
                {[[75,"+0.75%"],[100,"+1.00%"]].map(([tp,tpLabel])=>
                  <tr key={tp} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                    <td style={{padding:"4px 8px",color:"#e2e8f0",fontWeight:600}}>{tpLabel}</td>
                    {[30,60,120,180].map(h=>{
                      const tkey = `t${tp}_h${h}`;
                      const tres = convResults.per_target?.[tkey];
                      if (!tres) return <td key={h} style={{padding:"4px 8px",borderLeft:"1px solid rgba(255,255,255,0.08)",color:"#475569"}}>—</td>;
                      const passes = tres.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
                      // Find best high-conviction bucket
                      const highBuckets = (tres.buckets||[]).filter(b=>["0.8-0.9","0.9+"].includes(b.bucket));
                      const bestBucket = highBuckets.find(b=>b.n>=30 && b.ci_lower_pct!=null && b.ci_lower_pct>=75);
                      const auc = tres.auc_test;
                      const br = tres.base_rates?.test;
                      return <td key={h} style={{padding:"4px 8px",borderLeft:"1px solid rgba(255,255,255,0.08)",background:passes?"rgba(34,197,94,0.08)":"transparent"}}>
                        <div style={{fontSize:10,color:passes?"#22c55e":"#94a3b8",fontWeight:passes?700:500}}>
                          {passes?"✓ PASSES":"✗ below bar"}
                        </div>
                        <div style={{fontSize:10,color:"#64748b",fontVariantNumeric:"tabular-nums",marginTop:2}}>
                          AUC {auc?.toFixed(3)} · base {br}%
                        </div>
                        {bestBucket && <div style={{fontSize:10,color:"#22c55e",fontVariantNumeric:"tabular-nums",marginTop:2}}>
                          {bestBucket.bucket}: {bestBucket.hit_rate_pct}% (n={bestBucket.n})
                        </div>}
                      </td>;
                    })}
                  </tr>
                )}
              </tbody>
            </table>

            {/* Per-target detail */}
            <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:4}}>Per-target Calibration Detail</div>
            {(convResults.target_keys||[]).map(tkey=>{
              const tres = convResults.per_target?.[tkey];
              if (!tres) return null;
              const passes = tres.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
              return <details key={tkey} style={{marginBottom:8}} open={passes}>
                <summary style={{cursor:"pointer",padding:"6px 10px",borderRadius:4,
                  background:passes?"rgba(34,197,94,0.06)":"rgba(255,255,255,0.02)",
                  border:`1px solid ${passes?"rgba(34,197,94,0.25)":"rgba(255,255,255,0.04)"}`,fontSize:11}}>
                  <span style={{color:passes?"#22c55e":"#c4b5fd",fontWeight:600}}>{tres.target_pct} in {tres.horizon}</span>
                  <span style={{marginLeft:10,color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>
                    AUC test: <b>{tres.auc_test ?? "—"}</b>
                    {"  ·  "}base rates (tr/va/te): <b>{tres.base_rates?.train}%/{tres.base_rates?.val}%/{tres.base_rates?.test}%</b>
                  </span>
                  <span style={{marginLeft:10,color:passes?"#22c55e":"#eab308",fontWeight:700,fontSize:10,letterSpacing:0.3}}>
                    {passes?"✓ PASSES":"✗ BELOW"}
                  </span>
                </summary>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginTop:8,marginBottom:8}}>
                  <thead><tr>
                    {["Bucket","n","Hit%","CI 95%","Mean Predicted","Status"].map(col=>
                      <th key={col} style={{padding:"3px 8px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                  </tr></thead>
                  <tbody>
                    {(tres.buckets||[]).map((b,bi)=>{
                      const isHigh = b.bucket==="0.8-0.9" || b.bucket==="0.9+";
                      const bPass = isHigh && b.n>=30 && b.ci_lower_pct!=null && b.ci_lower_pct>=75;
                      const hr = b.hit_rate_pct;
                      const hrColor = hr==null?"#64748b":hr>=75?"#22c55e":hr>=60?"#a3e635":hr>=50?"#eab308":hr>=40?"#94a3b8":"#ef4444";
                      return <tr key={bi} style={{borderTop:"1px solid rgba(255,255,255,0.03)",background:isHigh?"rgba(139,92,246,0.04)":"transparent"}}>
                        <td style={{padding:"3px 8px",color:isHigh?"#c4b5fd":"#e2e8f0",fontWeight:isHigh?600:400}}>{b.bucket}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.n}</td>
                        <td style={{padding:"3px 8px",color:hrColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{hr!=null?`${hr}%`:"—"}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:10}}>
                          {b.ci_lower_pct!=null && b.ci_upper_pct!=null ? `[${b.ci_lower_pct}%, ${b.ci_upper_pct}%]` : "—"}
                        </td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.mean_predicted_prob!=null?`${(b.mean_predicted_prob*100).toFixed(1)}%`:"—"}</td>
                        <td style={{padding:"3px 8px",fontSize:9,fontWeight:600,letterSpacing:0.3}}>
                          {bPass ? <span style={{color:"#22c55e"}}>✓ PASSES</span> : (isHigh && b.n>0 ? <span style={{color:"#eab308"}}>✗ below</span> : "")}
                        </td>
                      </tr>;
                    })}
                  </tbody>
                </table>
                {/* Top features for this target */}
                <div style={{fontSize:9,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",marginBottom:2,marginTop:4}}>Top 15 features ({tkey})</div>
                <div style={{fontSize:10,color:"#94a3b8"}}>
                  {(tres.top_features||[]).map(([name,pct])=>`${name} (${pct}%)`).join(" · ")}
                </div>
              </details>;
            })}
          </div>
        )}
      </Box>

      {/* v27: PATTERN DISCOVERY — winner profile per scan hour × target */}
      <Box>
        <Lbl>Pattern Discovery (v27) — winner profile per scan hour × target</Lbl>
        <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
          Builds features for every date × stock × scan hour. For each scan_hour × target combination [+0.75%, +1%]: step one compares winner vs loser feature distributions via Cohen's d, step two induces a decision tree at depth 4 with min leaf 300, step three validates leaf-rules on held-out test fold. <b>Strict validation</b>: test n ≥ 30, hit rate ≥ baseline + 40pp, Wilson CI lower bound ≥ baseline + 30pp. Rules passing strict are the deployable winner profile. <i>Honest prior: v26 showed AUC 0.70-0.79 on this data; at strict +40pp threshold, very few, possibly zero, rules will validate. The distribution comparison is still informative either way.</i>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
          <Btn onClick={runPatternDiscovery} disabled={patProg?.inProgress||ip||extProg?.inProgress||repairProg?.inProgress||convProg?.inProgress} color="#f59e0b" style={{padding:"8px 16px",fontSize:12}}>
            {patProg?.inProgress?"Running...":"Run Pattern Discovery"}
          </Btn>
          {patResults && !patProg?.inProgress && (
            <>
              <Btn onClick={()=>downloadJson(patResults,`pattern_discovery_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"8px 12px",fontSize:12}}>
                Download JSON
              </Btn>
              <span style={{fontSize:11,color:"#64748b"}}>
                Last run: {patResults.generated_at?.slice(0,19).replace("T"," ")}
              </span>
            </>
          )}
        </div>
        {patProg?.inProgress && (
          <div style={{marginTop:10,marginBottom:10}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${patProg.pct||0}%`,height:"100%",background:"#f59e0b",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#f59e0b",fontWeight:600}}>{patProg.pct||0}%</span>
            </div>
            <div style={{fontSize:11,color:"#64748b"}}>{patProg.message}</div>
          </div>
        )}
        {!patProg?.inProgress && patProg?.phase === "error" && (
          <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
            <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {patProg.message}</div>
          </div>
        )}
        {patResults && (
          <div style={{marginTop:14}}>
            {/* Overall summary */}
            <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,
              background: patResults.total_passing_strict_rules>0?"rgba(34,197,94,0.08)":"rgba(234,179,8,0.08)",
              border: `1px solid ${patResults.total_passing_strict_rules>0?"rgba(34,197,94,0.25)":"rgba(234,179,8,0.25)"}`}}>
              <span style={{fontSize:11,color:"#64748b",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3}}>Rules passing strict</span>
              <span style={{marginLeft:10,color: patResults.total_passing_strict_rules>0?"#22c55e":"#eab308",fontWeight:700,letterSpacing:0.3,fontSize:14}}>
                {patResults.total_passing_strict_rules}
              </span>
              <span style={{marginLeft:16,color:"#94a3b8",fontSize:11}}>
                across {SCAN_HOURS.length} hours × 2 targets. Threshold: hit ≥ base+40pp, CI lower ≥ base+30pp, n ≥ 30.
              </span>
            </div>

            {/* Per-hour × target detail */}
            {SCAN_HOURS.map(h=>{
              const hourData = patResults.per_hour_target?.[String(h)];
              if (!hourData) return null;
              return <div key={h} style={{marginBottom:16}}>
                <div style={{fontSize:12,color:"#e2e8f0",fontWeight:600,marginBottom:6,paddingBottom:4,borderBottom:"1px solid rgba(255,255,255,0.1)"}}>{h}:00 ET</div>
                {["0.75%","1.00%"].map(tlabel=>{
                  const td = hourData[tlabel];
                  if (!td) return <div key={tlabel} style={{fontSize:11,color:"#475569",marginLeft:8,marginBottom:8}}>+{tlabel} — no data</div>;
                  const anyPass = td.n_passing_strict > 0;
                  return <div key={tlabel} style={{marginLeft:0,marginBottom:12,padding:"8px 10px",borderRadius:4,background:"rgba(255,255,255,0.02)",border:`1px solid ${anyPass?"rgba(34,197,94,0.25)":"rgba(255,255,255,0.04)"}`}}>
                    <div style={{fontSize:11,marginBottom:6,display:"flex",flexWrap:"wrap",gap:12}}>
                      <span style={{color:"#c4b5fd",fontWeight:600}}>+{tlabel}</span>
                      <span style={{color:"#94a3b8"}}>base (tr/va/te): <b>{td.base_rate_train}%/{td.base_rate_val ?? "—"}%/{td.base_rate_test}%</b></span>
                      <span style={{color:"#94a3b8"}}>n test: <b>{td.n_test?.toLocaleString()}</b></span>
                      <span style={{color:"#94a3b8"}}>required test hit ≥ <b>{td.min_test_hit_rate_required_pct}%</b>, CI lower ≥ <b>{td.min_test_ci_lower_required_pct}%</b></span>
                      <span style={{color:anyPass?"#22c55e":"#eab308",fontWeight:700,fontSize:10,letterSpacing:0.3}}>
                        {td.n_candidates} candidates → {td.n_validated} validated → <b>{td.n_passing_strict} pass strict</b>
                      </span>
                    </div>

                    {/* Top 10 discriminating features */}
                    <details style={{marginBottom:6}}>
                      <summary style={{cursor:"pointer",fontSize:10,color:"#64748b",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3}}>
                        Top discriminating features (winner vs loser, by |Cohen's d|)
                      </summary>
                      <table style={{width:"100%",fontSize:10,borderCollapse:"collapse",marginTop:4}}>
                        <thead><tr>
                          {["Feature","Winner mean","Loser mean","Cohen's d"].map(col=>
                            <th key={col} style={{padding:"3px 8px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                        </tr></thead>
                        <tbody>
                          {(td.top_features||[]).slice(0,10).map((f,fi)=>{
                            const d = f.cohens_d;
                            const dColor = d==null?"#64748b":Math.abs(d)>=0.5?"#22c55e":Math.abs(d)>=0.3?"#a3e635":Math.abs(d)>=0.15?"#eab308":"#94a3b8";
                            return <tr key={fi} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                              <td style={{padding:"3px 8px",color:"#e2e8f0"}}>{f.feature}</td>
                              <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{f.winner_mean}</td>
                              <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{f.loser_mean}</td>
                              <td style={{padding:"3px 8px",color:dColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{d!=null?(d>0?"+":"")+d:"—"}</td>
                            </tr>;
                          })}
                        </tbody>
                      </table>
                    </details>

                    {/* Validated rules */}
                    {(td.validated_rules||[]).length > 0 && (
                      <details open={anyPass}>
                        <summary style={{cursor:"pointer",fontSize:10,color:"#64748b",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3}}>
                          Validated rules ({td.validated_rules.length}, top 30 by test hit rate)
                        </summary>
                        <table style={{width:"100%",fontSize:10,borderCollapse:"collapse",marginTop:4}}>
                          <thead><tr>
                            {["Conditions","Train n","Train hit%","Test n","Test hit%","CI 95%","Strict"].map(col=>
                              <th key={col} style={{padding:"3px 8px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                          </tr></thead>
                          <tbody>
                            {td.validated_rules.map((r,ri)=>{
                              const thr = (r.test_hit_rate*100).toFixed(1);
                              const thrColor = r.test_hit_rate>=0.7?"#22c55e":r.test_hit_rate>=0.55?"#a3e635":r.test_hit_rate>=0.45?"#eab308":"#94a3b8";
                              const condStr = r.conditions.map(c=>`${c[0]} ${c[1]} ${typeof c[2]==="number"?c[2].toFixed(3):c[2]}`).join(" AND ");
                              return <tr key={ri} style={{borderTop:"1px solid rgba(255,255,255,0.03)",background:r.passes_strict?"rgba(34,197,94,0.08)":"transparent"}}>
                                <td style={{padding:"3px 8px",color:"#e2e8f0",fontSize:10,fontFamily:F}}>{condStr}</td>
                                <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{r.train_n}</td>
                                <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{(r.train_hit_rate*100).toFixed(1)}%</td>
                                <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{r.test_n}</td>
                                <td style={{padding:"3px 8px",color:thrColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{thr}%</td>
                                <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:9}}>
                                  [{(r.test_ci_lower*100).toFixed(1)}%, {(r.test_ci_upper*100).toFixed(1)}%]
                                </td>
                                <td style={{padding:"3px 8px",fontSize:9,fontWeight:700,letterSpacing:0.3}}>
                                  {r.passes_strict?<span style={{color:"#22c55e"}}>✓ STRICT</span>:<span style={{color:"#eab308"}}>✗</span>}
                                </td>
                              </tr>;
                            })}
                          </tbody>
                        </table>
                      </details>
                    )}
                    {(td.validated_rules||[]).length === 0 && (
                      <div style={{fontSize:10,color:"#64748b",fontStyle:"italic"}}>
                        No validated rules (decision tree found {td.n_candidates} leaf candidates on train fold matching base+40pp, but none survived test-fold validation at n≥30).
                      </div>
                    )}
                  </div>;
                })}
              </div>;
            })}
          </div>
        )}
      </Box>

      {/* v28: COST-ADJUSTED ANALYSIS — LightGBM + pattern discovery across 0.30/0.40/0.50 */}
      <Box>
        <Lbl>Cost-Adjusted Analysis (v28) — LightGBM + pattern discovery across round-trip-cost targets</Lbl>
        <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
          Retargets prediction at thresholds derived from round-trip transaction cost: +0.30% (best-case tight spreads), +0.40% (typical R2K cost), +0.50% (conservative). For each target, runs BOTH: (A) LightGBM calibrated classifier at same strict validation as v26; (B) decision-tree pattern discovery at same strict validation as v27. If any combination produces a high-confidence bucket or validated rule, we have a deployable cost-positive signal.
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
          <Btn onClick={runV28} disabled={v28Prog?.inProgress||ip||extProg?.inProgress||repairProg?.inProgress||convProg?.inProgress||patProg?.inProgress} color="#ec4899" style={{padding:"8px 16px",fontSize:12}}>
            {v28Prog?.inProgress?"Running...":"Run v28 Cost-Adjusted Analysis"}
          </Btn>
          {v28Results && !v28Prog?.inProgress && (
            <>
              <Btn onClick={()=>downloadJson(v28Results,`v28_cost_adjusted_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"8px 12px",fontSize:12}}>
                Download JSON
              </Btn>
              <span style={{fontSize:11,color:"#64748b"}}>
                Last run: {v28Results.generated_at?.slice(0,19).replace("T"," ")}
              </span>
            </>
          )}
        </div>
        {v28Prog?.inProgress && (
          <div style={{marginTop:10,marginBottom:10}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${v28Prog.pct||0}%`,height:"100%",background:"#ec4899",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#ec4899",fontWeight:600}}>{v28Prog.pct||0}%</span>
            </div>
            <div style={{fontSize:11,color:"#64748b"}}>{v28Prog.message}</div>
          </div>
        )}
        {!v28Prog?.inProgress && v28Prog?.phase === "error" && (
          <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
            <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {v28Prog.message}</div>
          </div>
        )}
        {v28Results && (
          <div style={{marginTop:14}}>
            {/* Overall summary */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:8,marginBottom:14}}>
              <div style={{padding:"8px 12px",borderRadius:4,
                background:v28Results.lgbm_passing_targets>0?"rgba(34,197,94,0.08)":"rgba(234,179,8,0.08)",
                border:`1px solid ${v28Results.lgbm_passing_targets>0?"rgba(34,197,94,0.25)":"rgba(234,179,8,0.25)"}`}}>
                <div style={{fontSize:10,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>LightGBM passing</div>
                <div style={{fontSize:18,color:v28Results.lgbm_passing_targets>0?"#22c55e":"#eab308",fontWeight:700}}>{v28Results.lgbm_passing_targets}/3 targets</div>
              </div>
              <div style={{padding:"8px 12px",borderRadius:4,
                background:v28Results.pattern_passing_rules>0?"rgba(34,197,94,0.08)":"rgba(234,179,8,0.08)",
                border:`1px solid ${v28Results.pattern_passing_rules>0?"rgba(34,197,94,0.25)":"rgba(234,179,8,0.25)"}`}}>
                <div style={{fontSize:10,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>Pattern rules passing</div>
                <div style={{fontSize:18,color:v28Results.pattern_passing_rules>0?"#22c55e":"#eab308",fontWeight:700}}>{v28Results.pattern_passing_rules} rules</div>
              </div>
              <div style={{padding:"8px 12px",borderRadius:4,background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)"}}>
                <div style={{fontSize:10,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>Folds</div>
                <div style={{fontSize:12,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
                  tr={v28Results.fold_sizes?.train?.toLocaleString()} / va={v28Results.fold_sizes?.val?.toLocaleString()} / te={v28Results.fold_sizes?.test?.toLocaleString()}
                </div>
              </div>
            </div>

            {/* LightGBM results per target */}
            <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:6,marginTop:10}}>LightGBM Calibration — per target</div>
            <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:16}}>
              <thead><tr>
                {["Target","AUC","Base (test)","Best high-conv bucket","Verdict"].map(col=>
                  <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
              </tr></thead>
              <tbody>
                {["0.30%","0.40%","0.50%"].map(tlabel=>{
                  const t = v28Results.lgbm_per_target?.[tlabel];
                  if (!t || t.error) return <tr key={tlabel} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                    <td style={{padding:"4px 8px",color:"#e2e8f0"}}>+{tlabel}</td>
                    <td colSpan={4} style={{padding:"4px 8px",color:"#ef4444"}}>{t?.error || "—"}</td>
                  </tr>;
                  const passes = t.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
                  const highBuckets = (t.buckets||[]).filter(b=>["0.8-0.9","0.9+"].includes(b.bucket));
                  const bestPass = highBuckets.find(b=>b.n>=30 && b.ci_lower_pct!=null && b.ci_lower_pct>=75);
                  const bestAny = highBuckets.filter(b=>b.n>0).sort((a,b)=>(b.hit_rate_pct||0)-(a.hit_rate_pct||0))[0];
                  const best = bestPass || bestAny;
                  return <tr key={tlabel} style={{borderTop:"1px solid rgba(255,255,255,0.04)",background:passes?"rgba(34,197,94,0.06)":"transparent"}}>
                    <td style={{padding:"4px 8px",color:"#c4b5fd",fontWeight:600}}>+{tlabel}</td>
                    <td style={{padding:"4px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{t.auc_test}</td>
                    <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{t.base_rates?.test}%</td>
                    <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:10}}>
                      {best ? `${best.bucket}: ${best.hit_rate_pct}% (n=${best.n}, CI ${best.ci_lower_pct}-${best.ci_upper_pct}%)` : "—"}
                    </td>
                    <td style={{padding:"4px 8px",fontWeight:700,fontSize:10,letterSpacing:0.3,color:passes?"#22c55e":"#eab308"}}>
                      {passes ? "✓ PASSES 80%" : "✗ below bar"}
                    </td>
                  </tr>;
                })}
              </tbody>
            </table>

            {/* LightGBM detail per target (expandable) */}
            {["0.30%","0.40%","0.50%"].map(tlabel=>{
              const t = v28Results.lgbm_per_target?.[tlabel];
              if (!t || t.error) return null;
              const passes = t.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
              return <details key={"lgbm-"+tlabel} style={{marginBottom:6}} open={passes}>
                <summary style={{cursor:"pointer",padding:"4px 8px",borderRadius:4,
                  background:passes?"rgba(34,197,94,0.06)":"rgba(255,255,255,0.02)",
                  border:`1px solid ${passes?"rgba(34,197,94,0.25)":"rgba(255,255,255,0.04)"}`,fontSize:11}}>
                  <span style={{color:"#c4b5fd",fontWeight:600}}>LightGBM calibration: +{tlabel}</span>
                  <span style={{marginLeft:10,color:"#94a3b8"}}>(click to expand full bucket table)</span>
                </summary>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginTop:6,marginBottom:8}}>
                  <thead><tr>
                    {["Bucket","n","Hit%","CI 95%","Mean Pred","Status"].map(col=>
                      <th key={col} style={{padding:"3px 8px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                  </tr></thead>
                  <tbody>
                    {(t.buckets||[]).map((b,bi)=>{
                      const isHigh = b.bucket==="0.8-0.9" || b.bucket==="0.9+";
                      const bPass = isHigh && b.n>=30 && b.ci_lower_pct!=null && b.ci_lower_pct>=75;
                      const hrColor = b.hit_rate_pct==null?"#64748b":b.hit_rate_pct>=75?"#22c55e":b.hit_rate_pct>=60?"#a3e635":b.hit_rate_pct>=50?"#eab308":b.hit_rate_pct>=40?"#94a3b8":"#ef4444";
                      return <tr key={bi} style={{borderTop:"1px solid rgba(255,255,255,0.03)",background:isHigh?"rgba(139,92,246,0.04)":"transparent"}}>
                        <td style={{padding:"3px 8px",color:isHigh?"#c4b5fd":"#e2e8f0",fontWeight:isHigh?600:400}}>{b.bucket}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.n}</td>
                        <td style={{padding:"3px 8px",color:hrColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{b.hit_rate_pct!=null?`${b.hit_rate_pct}%`:"—"}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:10}}>{b.ci_lower_pct!=null?`[${b.ci_lower_pct}%, ${b.ci_upper_pct}%]`:"—"}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.mean_predicted_prob!=null?`${(b.mean_predicted_prob*100).toFixed(1)}%`:"—"}</td>
                        <td style={{padding:"3px 8px",fontSize:9,fontWeight:700,letterSpacing:0.3,color:bPass?"#22c55e":(isHigh && b.n>0?"#eab308":"#64748b")}}>
                          {bPass?"✓":(isHigh && b.n>0?"✗":"")}
                        </td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </details>;
            })}

            {/* Pattern discovery per target x hour matrix */}
            <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:6,marginTop:16}}>Pattern Discovery — per (target × hour)</div>
            <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:10}}>
              <thead><tr>
                <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Target</th>
                {SCAN_HOURS.map(h=>
                  <th key={h} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderLeft:"1px solid rgba(255,255,255,0.08)"}}>{h}:00</th>
                )}
              </tr></thead>
              <tbody>
                {["0.30%","0.40%","0.50%"].map(tlabel=>{
                  const tData = v28Results.pattern_per_target?.[tlabel] || {};
                  return <tr key={tlabel} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                    <td style={{padding:"4px 8px",color:"#c4b5fd",fontWeight:600}}>+{tlabel}</td>
                    {SCAN_HOURS.map(h=>{
                      const cell = tData[String(h)];
                      if (!cell) return <td key={h} style={{padding:"4px 8px",borderLeft:"1px solid rgba(255,255,255,0.08)",color:"#475569"}}>—</td>;
                      const anyPass = cell.n_passing_strict > 0;
                      return <td key={h} style={{padding:"4px 8px",borderLeft:"1px solid rgba(255,255,255,0.08)",background:anyPass?"rgba(34,197,94,0.08)":"transparent"}}>
                        <div style={{fontSize:10,color:anyPass?"#22c55e":"#94a3b8",fontWeight:anyPass?700:500}}>
                          {anyPass?`✓ ${cell.n_passing_strict} rule(s)`:"✗ 0 pass"}
                        </div>
                        <div style={{fontSize:9,color:"#64748b",fontVariantNumeric:"tabular-nums",marginTop:2}}>
                          base {cell.base_rate_test}% · cand {cell.n_candidates} → val {cell.n_validated}
                        </div>
                      </td>;
                    })}
                  </tr>;
                })}
              </tbody>
            </table>

            {/* Pattern detail: show validated rules for any passing cells */}
            {["0.30%","0.40%","0.50%"].map(tlabel=>{
              const tData = v28Results.pattern_per_target?.[tlabel] || {};
              const passingCells = Object.entries(tData).filter(([h,c])=>c.n_passing_strict > 0);
              if (passingCells.length === 0) return null;
              return <details key={"pat-"+tlabel} open style={{marginTop:10,marginBottom:6}}>
                <summary style={{cursor:"pointer",padding:"4px 8px",borderRadius:4,
                  background:"rgba(34,197,94,0.06)",border:"1px solid rgba(34,197,94,0.25)",fontSize:11}}>
                  <span style={{color:"#22c55e",fontWeight:600}}>✓ Passing pattern rules for +{tlabel}</span>
                </summary>
                {passingCells.map(([h,cell])=>
                  <div key={h} style={{marginTop:6,padding:"6px 10px",borderRadius:4,background:"rgba(255,255,255,0.02)"}}>
                    <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:4}}>{h}:00 ET</div>
                    {cell.validated_rules.filter(r=>r.passes_strict).map((r,ri)=>{
                      const condStr = r.conditions.map(c=>`${c[0]} ${c[1]} ${typeof c[2]==="number"?c[2].toFixed(3):c[2]}`).join(" AND ");
                      return <div key={ri} style={{fontSize:10,color:"#94a3b8",marginBottom:3}}>
                        <span style={{color:"#22c55e",fontWeight:700,marginRight:6}}>
                          test hr {(r.test_hit_rate*100).toFixed(1)}% (n={r.test_n}, CI {(r.test_ci_lower*100).toFixed(1)}-{(r.test_ci_upper*100).toFixed(1)}%)
                        </span>
                        <span style={{fontFamily:F,color:"#e2e8f0"}}>{condStr}</span>
                      </div>;
                    })}
                  </div>
                )}
              </details>;
            })}
          </div>
        )}
      </Box>

      {/* v29: FINE-GRAINED TARGET SWEEP — pinpoint where the 80% bar breaks */}
      <Box>
        <Lbl>Fine-Grained Target Sweep (v29) — find the highest target that still passes 80% bar</Lbl>
        <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
          v28 showed +0.30% PASSES the strict 80% bar (0.9+ bucket 86.96%, CI lower 80.32%) and +0.40% FAILS. This sweep tests every 1bps target between them: +0.31%, +0.32%, ..., +0.40%. Each trains a separate LightGBM classifier with same features and strict validation as v28. Output: highest target where 0.8+ bucket still has n≥30 and CI lower ≥75%.
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:10}}>
          <Btn onClick={runV29} disabled={v29Prog?.inProgress||ip||extProg?.inProgress||repairProg?.inProgress||convProg?.inProgress||patProg?.inProgress||v28Prog?.inProgress} color="#14b8a6" style={{padding:"8px 16px",fontSize:12}}>
            {v29Prog?.inProgress?"Running...":"Run v29 Target Sweep"}
          </Btn>
          {v29Results && !v29Prog?.inProgress && (
            <>
              <Btn onClick={()=>downloadJson(v29Results,`v29_target_sweep_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"8px 12px",fontSize:12}}>
                Download JSON
              </Btn>
              <span style={{fontSize:11,color:"#64748b"}}>
                Last run: {v29Results.generated_at?.slice(0,19).replace("T"," ")}
              </span>
            </>
          )}
        </div>
        {v29Prog?.inProgress && (
          <div style={{marginTop:10,marginBottom:10}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${v29Prog.pct||0}%`,height:"100%",background:"#14b8a6",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#14b8a6",fontWeight:600}}>{v29Prog.pct||0}%</span>
            </div>
            <div style={{fontSize:11,color:"#64748b"}}>{v29Prog.message}</div>
          </div>
        )}
        {!v29Prog?.inProgress && v29Prog?.phase === "error" && (
          <div style={{marginTop:10,padding:"6px 10px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.2)"}}>
            <div style={{fontSize:11,color:"#fca5a5",fontWeight:600}}>✗ {v29Prog.message}</div>
          </div>
        )}
        {v29Results && (
          <div style={{marginTop:14}}>
            {/* Headline: highest passing target */}
            <div style={{marginBottom:12,padding:"10px 14px",borderRadius:4,
              background: v29Results.highest_passing_target?"rgba(34,197,94,0.08)":"rgba(234,179,8,0.08)",
              border: `1px solid ${v29Results.highest_passing_target?"rgba(34,197,94,0.3)":"rgba(234,179,8,0.25)"}`}}>
              <div style={{fontSize:10,color:"#64748b",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:3}}>Highest target passing strict 80% bar</div>
              <div style={{fontSize:20,color:v29Results.highest_passing_target?"#22c55e":"#eab308",fontWeight:700}}>
                {v29Results.highest_passing_target ? `+${v29Results.highest_passing_target}` : "none in range"}
              </div>
              <div style={{fontSize:11,color:"#94a3b8",marginTop:4}}>
                {v29Results.n_passing_targets}/10 targets pass. Targets tested: +{v29Results.config?.targets_tested?.join(", +")}
              </div>
            </div>

            {/* Per-target sweep summary table */}
            <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:14}}>
              <thead><tr>
                {["Target","AUC","Base","0.8-0.9 n","0.8-0.9 hit%","0.8-0.9 CI lower","0.9+ n","0.9+ hit%","0.9+ CI lower","Verdict"].map(col=>
                  <th key={col} style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
              </tr></thead>
              <tbody>
                {(v29Results.config?.targets_tested||[]).map(tlabel=>{
                  const t = v29Results.per_target?.[tlabel];
                  if (!t || t.error) return <tr key={tlabel} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                    <td style={{padding:"4px 6px",color:"#e2e8f0"}}>+{tlabel}</td>
                    <td colSpan={9} style={{padding:"4px 6px",color:"#ef4444"}}>{t?.error || "—"}</td>
                  </tr>;
                  const passes = t.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
                  const b89 = (t.buckets||[]).find(b=>b.bucket==="0.8-0.9");
                  const b90 = (t.buckets||[]).find(b=>b.bucket==="0.9+");
                  const ci89ok = b89?.ci_lower_pct !== null && b89?.ci_lower_pct >= 75 && b89?.n >= 30;
                  const ci90ok = b90?.ci_lower_pct !== null && b90?.ci_lower_pct >= 75 && b90?.n >= 30;
                  const rowBg = passes ? "rgba(34,197,94,0.06)" : "transparent";
                  return <tr key={tlabel} style={{borderTop:"1px solid rgba(255,255,255,0.04)",background:rowBg}}>
                    <td style={{padding:"4px 6px",color:passes?"#22c55e":"#c4b5fd",fontWeight:600}}>+{tlabel}</td>
                    <td style={{padding:"4px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{t.auc_test}</td>
                    <td style={{padding:"4px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{t.base_rates?.test}%</td>
                    <td style={{padding:"4px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b89?.n ?? "—"}</td>
                    <td style={{padding:"4px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b89?.hit_rate_pct!=null?`${b89.hit_rate_pct}%`:"—"}</td>
                    <td style={{padding:"4px 6px",color:ci89ok?"#22c55e":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:ci89ok?600:400}}>{b89?.ci_lower_pct!=null?`${b89.ci_lower_pct}%`:"—"}</td>
                    <td style={{padding:"4px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b90?.n ?? "—"}</td>
                    <td style={{padding:"4px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b90?.hit_rate_pct!=null?`${b90.hit_rate_pct}%`:"—"}</td>
                    <td style={{padding:"4px 6px",color:ci90ok?"#22c55e":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:ci90ok?600:400}}>{b90?.ci_lower_pct!=null?`${b90.ci_lower_pct}%`:"—"}</td>
                    <td style={{padding:"4px 6px",fontWeight:700,fontSize:10,letterSpacing:0.3,color:passes?"#22c55e":"#eab308"}}>
                      {passes ? "✓ PASSES" : "✗ below"}
                    </td>
                  </tr>;
                })}
              </tbody>
            </table>

            {/* Expandable detail per target */}
            {(v29Results.config?.targets_tested||[]).map(tlabel=>{
              const t = v29Results.per_target?.[tlabel];
              if (!t || t.error) return null;
              const passes = t.verdict === "ACHIEVES_80_HIGH_CONFIDENCE";
              return <details key={"v29-"+tlabel} style={{marginBottom:4}} open={passes}>
                <summary style={{cursor:"pointer",padding:"3px 8px",borderRadius:4,
                  background:passes?"rgba(34,197,94,0.04)":"rgba(255,255,255,0.02)",
                  border:`1px solid ${passes?"rgba(34,197,94,0.2)":"rgba(255,255,255,0.04)"}`,fontSize:10}}>
                  <span style={{color:passes?"#22c55e":"#c4b5fd",fontWeight:600}}>+{tlabel} full calibration</span>
                </summary>
                <table style={{width:"100%",fontSize:10,borderCollapse:"collapse",marginTop:4,marginBottom:4}}>
                  <thead><tr>
                    {["Bucket","n","Hit%","CI 95%","Mean Pred"].map(col=>
                      <th key={col} style={{padding:"2px 8px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                  </tr></thead>
                  <tbody>
                    {(t.buckets||[]).map((b,bi)=>{
                      const isHigh = b.bucket==="0.8-0.9" || b.bucket==="0.9+";
                      return <tr key={bi} style={{borderTop:"1px solid rgba(255,255,255,0.03)",background:isHigh?"rgba(139,92,246,0.04)":"transparent"}}>
                        <td style={{padding:"2px 8px",color:isHigh?"#c4b5fd":"#e2e8f0",fontWeight:isHigh?600:400}}>{b.bucket}</td>
                        <td style={{padding:"2px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.n}</td>
                        <td style={{padding:"2px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b.hit_rate_pct!=null?`${b.hit_rate_pct}%`:"—"}</td>
                        <td style={{padding:"2px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:9}}>{b.ci_lower_pct!=null?`[${b.ci_lower_pct}%, ${b.ci_upper_pct}%]`:"—"}</td>
                        <td style={{padding:"2px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.mean_predicted_prob!=null?`${(b.mean_predicted_prob*100).toFixed(1)}%`:"—"}</td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </details>;
            })}
          </div>
        )}
      </Box>

      {Object.keys(meta).length>0&&(
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
            <Lbl>Validation Results</Lbl>
            <div style={{display:"flex",gap:4,marginBottom:10}}>
              {SCAN_HOURS.filter(h=>meta[String(h)]).map(h=><Btn key={h} active={h===sh} onClick={()=>setSh(h)}>{h}:00</Btn>)}
            </div>
          </div>
          {sm?(
            <>
              {/* Top-N comparison table — most important view */}
              {sm.topN && (
                <div style={{marginBottom:14,padding:"10px 12px",borderRadius:4,background:"rgba(6,182,212,0.04)",border:"1px solid rgba(6,182,212,0.15)"}}>
                  <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Top-N Strategy Comparison (per val day, {sm.val_dates} days)</div>
                  <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                    <thead><tr>
                      {["Pick","Win Rate","Avg PnL","StdDev","Min Day","Max Day","+Days"].map(h=><th key={h} style={{padding:"4px 8px",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",textAlign:"left"}}>{h}</th>)}
                    </tr></thead>
                    <tbody>
                      {[1,3,5,10].map(N=>{
                        const t=sm.topN[String(N)]||sm.topN[N];
                        if(!t) return null;
                        const rBE = sm.loss_distribution?.realized_breakeven_wr;
                        const wrPct = t.avg_wr*100;
                        const beat = rBE ? wrPct > rBE : t.avg_pnl > 0;
                        return <tr key={N} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                          <td style={{padding:"5px 8px",color:"#e2e8f0",fontWeight:700}}>Top-{N}</td>
                          <td style={{padding:"5px 8px",color:beat?"#22c55e":"#ef4444",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{wrPct.toFixed(1)}%
                            {rBE && <span style={{color:"#475569",fontSize:10,marginLeft:6,fontWeight:400}}>vs rBE {rBE}%</span>}
                          </td>
                          <td style={{padding:"5px 8px",color:t.avg_pnl>0?"#22c55e":"#ef4444",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{t.avg_pnl>0?"+":""}{t.avg_pnl}%</td>
                          <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{t.std_pnl}%</td>
                          <td style={{padding:"5px 8px",color:"#ef4444",fontVariantNumeric:"tabular-nums"}}>{t.min_pnl}%</td>
                          <td style={{padding:"5px 8px",color:"#22c55e",fontVariantNumeric:"tabular-nums"}}>+{t.max_pnl}%</td>
                          <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{t.pnl_positive_days}/{t.n_days}</td>
                        </tr>;
                      })}
                    </tbody>
                  </table>
                  <div style={{fontSize:9,color:"#475569",marginTop:6,lineHeight:1.5}}>
                    Narrowing from top-10 to top-3 or top-1 amplifies any real edge but also amplifies noise. Watch: do win rate and PnL rise as N decreases, or stay flat/degrade? Rising = real ordering ability. Flat = model can't rank its own picks. Also check StdDev: if top-1 has +2% PnL but ±3% stddev, the strategy is still too volatile to trade.
                  </div>
                </div>
              )}

              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Key Metrics</div>
                <div style={{fontSize:12,lineHeight:2.2,color:"#94a3b8"}}>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>AUC:</span>
                    <span style={{color:sm.auc>0.6?"#22c55e":"#eab308",fontWeight:700,fontSize:14}}>{sm.auc}</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Win Rate:</span>
                    <span style={{color:sm.avg_win_rate_top10>sm.val_win_rate?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{(sm.avg_win_rate_top10*100).toFixed(1)}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>vs base {(sm.val_win_rate*100).toFixed(1)}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Avg P&L:</span>
                    <span style={{color:sm.avg_pnl_top10>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.avg_pnl_top10>0?"+":""}{sm.avg_pnl_top10}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;61% stocks):</span>
                    <span style={{color:sm.ev_above_breakeven>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven>0?"+":""}{sm.ev_above_breakeven}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven} stocks)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;66% stocks):</span>
                    <span style={{color:sm.ev_above_breakeven_plus5>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven_plus5>0?"+":""}{sm.ev_above_breakeven_plus5}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven_plus5} stocks)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Top-10 default):</span>
                    <span style={{color:sm.ev_above_50pct>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_50pct>0?"+":""}{sm.ev_above_50pct}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_50pct} samples @ &gt;50%)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Exit reasons (val):</span>
                    <span style={{fontSize:11}}>{sm.val_exit_reasons?Object.entries(sm.val_exit_reasons).map(([r,n])=>`${r}: ${n}`).join(", "):""}</span></div>
                </div>
              </div>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Feature Importance</div>
                {sm.importance&&Object.entries(sm.importance).sort(([,a],[,b])=>b-a).slice(0,12).map(([name,val])=>{
                  const max=Math.max(...Object.values(sm.importance));
                  return (
                    <div key={name} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                      <span style={{width:100,fontSize:10,color:"#94a3b8",textAlign:"right",flexShrink:0}}>{name}</span>
                      <div style={{flex:1,height:12,background:"rgba(255,255,255,0.04)",borderRadius:2,overflow:"hidden"}}>
                        <div style={{width:`${(val/max)*100}%`,height:"100%",borderRadius:2,background:"#8b5cf6"}}/>
                      </div>
                      <span style={{fontSize:10,color:"#64748b",minWidth:32,fontVariantNumeric:"tabular-nums"}}>{(val*100).toFixed(1)}%</span>
                    </div>);})}
              </div>
            </div>
            </>
          ):<div style={{color:"#475569",fontSize:12}}>Select scan hour</div>}
        </Box>)}
      <SweepSection/>
    </div>);
}

// ─── PATTERNS ────────────────────────────────────────────────────
function formatCondition(c) {
  const op = c.op === ">=" ? "≥" : "≤";
  const val = Math.abs(c.value) < 0.01 ? c.value.toExponential(2) : c.value.toFixed(3);
  return `${c.feature} ${op} ${val}`;
}

function PatternsTab({health}) {
  const [prog,setProg]=useState(null);
  const [results,setResults]=useState(null);
  const [selHour,setSelHour]=useState(11);

  const poll=useCallback(()=>{
    fetch('/api/patterns/progress').then(r=>r.json()).then(setProg).catch(()=>{});
    fetch('/api/patterns/results').then(r=>r.json()).then(setResults).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,3000);return()=>clearInterval(iv);},[poll]);

  const run=async()=>{
    if(!confirm("Run pattern search on current training data? Takes ~1-3 minutes. Uses the same TP/SL you trained with.")) return;
    const body={};
    if(health?.tp_mult) body.tp_pct = health.tp_mult;  // backend now interprets as multiplier
    if(health?.sl_mult) body.sl_pct = health.sl_mult;
    await fetch('/api/patterns/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    poll();
  };
  const reset=async()=>{
    if(!confirm("Delete all discovered patterns?")) return;
    await fetch('/api/patterns/reset',{method:'POST'});
    poll();
  };

  const ip = prog?.inProgress;
  const hasData = prog?.hasTrainingData;
  const hours = results?.hours || {};
  const hourKeys = Object.keys(hours).sort();
  const currentHour = hours[String(selHour)];
  const patterns = currentHour?.patterns || [];

  // Count stats
  const totalPatterns = hourKeys.reduce((s,h)=>s+((hours[h]?.patterns||[]).length),0);
  const hoursWithPatterns = hourKeys.filter(h=>(hours[h]?.patterns||[]).length>0).length;

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
          <Lbl>Pattern Search</Lbl>
          <div style={{display:"flex",gap:6}}>
            <Btn onClick={run} disabled={ip||!hasData} color="#f59e0b" style={{padding:"4px 10px",fontSize:11}}>
              {ip?"Searching...":totalPatterns>0?"Re-run Search":"Run Pattern Search"}
            </Btn>
            {totalPatterns>0&&!ip&&<Btn onClick={reset} color="#ef4444" style={{padding:"4px 10px",fontSize:11}}>Reset</Btn>}
          </div>
        </div>

        {!hasData && <div style={{fontSize:12,color:"#64748b",padding:"12px 0"}}>
          No training data cached. Run Training first — pattern search reads from the enriched training dataset.
        </div>}

        {ip && (
          <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(245,158,11,0.08)",border:"1px solid rgba(245,158,11,0.2)"}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${prog.pct}%`,height:"100%",background:"#f59e0b",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#f59e0b",fontWeight:600}}>{prog.pct}%</span>
            </div>
            <div style={{fontSize:11,color:"#94a3b8"}}>{prog.message}</div>
          </div>
        )}

        {totalPatterns===0 && !ip && hasData && (
          <div style={{fontSize:12,color:"#64748b",padding:"12px 0",lineHeight:1.6}}>
            Searches training data for narrow feature regions where historical win rate exceeded break-even by a meaningful margin on both training AND validation splits.<br/>
            Single-feature thresholds are tested first (25+ features × 14 percentiles). The top 12 single-feature patterns are then combined pairwise.<br/>
            Only patterns with n≥50 val samples and val edge ≥+3% are kept.<br/>
            If no patterns survive, it means the current features don't contain a narrow enough signal — and we should be honest about that rather than lowering the bar.
          </div>
        )}

        {totalPatterns>0 && (
          <div style={{fontSize:11,color:"#94a3b8",marginBottom:4}}>
            <span style={{color:"#f59e0b",fontWeight:700}}>{totalPatterns}</span> patterns across <span style={{color:"#f59e0b",fontWeight:700}}>{hoursWithPatterns}</span> scan hour{hoursWithPatterns!==1?"s":""}
            {results?.generatedAt && <span style={{color:"#475569"}}> · generated {new Date(results.generatedAt).toLocaleString()}</span>}
          </div>
        )}
      </Box>

      {totalPatterns>0 && (
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:12}}>
            <Lbl>Patterns by Scan Hour</Lbl>
            <div style={{flex:1}}/>
            {SCAN_HOURS.map(h=>{
              const count = (hours[String(h)]?.patterns||[]).length;
              return <Btn key={h} active={h===selHour} onClick={()=>setSelHour(h)} style={{padding:"3px 8px",fontSize:10}}>
                {h}:00 ({count})
              </Btn>;
            })}
          </div>

          {!currentHour || currentHour.error ? (
            <div style={{color:"#64748b",fontSize:12,padding:"12px 0"}}>
              No patterns for {selHour}:00 — {currentHour?.error || "search has not run for this hour"}
            </div>
          ) : patterns.length===0 ? (
            <div style={{color:"#64748b",fontSize:12,padding:"12px 0"}}>
              No patterns met criteria at {selHour}:00 (base WR {currentHour.base_wr_val}%, n_val {currentHour.n_val}).
            </div>
          ) : (
            <>
              <div style={{fontSize:10,color:"#475569",marginBottom:8,letterSpacing:0.3}}>
                {currentHour.n_train} training rows / {currentHour.n_val} validation rows · base WR {currentHour.base_wr_val}% · base PnL {currentHour.base_pnl_val>0?"+":""}{currentHour.base_pnl_val}%
              </div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                  <thead>
                    <tr>
                      {["#","Type","Conditions","Train N","Train WR","Train Edge","Val N","Val WR","Val Edge","Val PnL"].map(h=>
                        <th key={h} style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase"}}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {patterns.map((p,i)=>(
                      <tr key={i} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                        <td style={{padding:"5px 6px",color:"#475569"}}>{i+1}</td>
                        <td style={{padding:"5px 6px",color:p.feat_count===2?"#f59e0b":"#94a3b8",fontSize:10}}>{p.feat_count===2?"2F":"1F"}</td>
                        <td style={{padding:"5px 6px",color:"#e2e8f0",fontFamily:"inherit",fontSize:10,lineHeight:1.4}}>
                          {p.conditions.map((c,j)=><div key={j}>{formatCondition(c)}</div>)}
                        </td>
                        <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.train.n}</td>
                        <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.train.win_rate}%</td>
                        <td style={{padding:"5px 6px",color:p.train.edge>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{p.train.edge>0?"+":""}{p.train.edge}%</td>
                        <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.val.n}</td>
                        <td style={{padding:"5px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{p.val.win_rate}%</td>
                        <td style={{padding:"5px 6px",color:p.val.edge>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:700}}>{p.val.edge>0?"+":""}{p.val.edge}%</td>
                        <td style={{padding:"5px 6px",color:p.val.avg_pnl>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums"}}>{p.val.avg_pnl>0?"+":""}{p.val.avg_pnl}%</td>
                      </tr>))}
                  </tbody>
                </table>
              </div>
              <div style={{fontSize:10,color:"#475569",marginTop:8,lineHeight:1.5}}>
                Val Edge is the honest measurement — it's the margin by which this pattern's top-picks beat break-even on held-out days. If Train Edge is high but Val Edge is low, the pattern is overfit. The scanner will flag live stocks matching any pattern here with a "⚡" icon.
              </div>
            </>
          )}
        </Box>
      )}
      <SensitivitySection/>
    </div>);
}

// ─── SENSITIVITY SWEEP ───────────────────────────────────────────
function SensitivitySection() {
  const [prog,setProg]=useState(null);
  const [results,setResults]=useState(null);
  const [selHour,setSelHour]=useState(11);

  const poll=useCallback(()=>{
    fetch('/api/sensitivity/progress').then(r=>r.json()).then(setProg).catch(()=>{});
    fetch('/api/sensitivity/results').then(r=>r.json()).then(setResults).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,3000);return()=>clearInterval(iv);},[poll]);

  const run=async()=>{
    if(!confirm("Run sensitivity sweep? Evaluates every single-feature threshold and buckets by (train_edge, val_edge) bars. Takes ~1-2 min.")) return;
    await fetch('/api/sensitivity/run',{method:'POST'});
    poll();
  };

  const ip = prog?.inProgress;
  const hours = results?.hours || {};
  const hourKeys = Object.keys(hours).sort();
  const hasResults = hourKeys.length > 0;
  const cur = hours[String(selHour)];

  const fmtEdge = (v) => v==null ? "—" : `${v>0?"+":""}${v}%`;
  const edgeColor = (v) => v==null ? "#64748b" : v>=3 ? "#22c55e" : v>=0 ? "#eab308" : "#ef4444";

  return (
    <Box>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
        <Lbl>Sensitivity Sweep — How Many Patterns At Each Threshold Bar?</Lbl>
        <Btn onClick={run} disabled={ip} color="#06b6d4" style={{padding:"4px 10px",fontSize:11}}>
          {ip?"Running...":hasResults?"Re-run":"Run Sensitivity"}
        </Btn>
      </div>

      {ip && (
        <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(6,182,212,0.08)",border:"1px solid rgba(6,182,212,0.2)"}}>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
            <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${prog.pct}%`,height:"100%",background:"#06b6d4",borderRadius:3,transition:"width 0.5s"}}/>
            </div>
            <span style={{fontSize:11,color:"#06b6d4",fontWeight:600}}>{prog.pct}%</span>
          </div>
          <div style={{fontSize:11,color:"#94a3b8"}}>{prog.message}</div>
        </div>
      )}

      {!hasResults && !ip && (
        <div style={{fontSize:12,color:"#64748b",padding:"12px 0",lineHeight:1.6}}>
          The main Pattern Search uses a strict bar (+5% train / +3% val). This tool tests many bars at once to show if there's ANY signal present — even weak signal. If even the loosest bar (0% train, -2% val) finds zero patterns, the features have no edge at all. If looser bars find patterns, we'll see where signal transitions from "real" to "noise".
        </div>
      )}

      {hasResults && (
        <>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:12}}>
            {SCAN_HOURS.map(h=>(
              <Btn key={h} active={h===selHour} onClick={()=>setSelHour(h)} style={{padding:"3px 8px",fontSize:10}}>
                {h}:00
              </Btn>))}
          </div>

          {!cur || cur.error ? (
            <div style={{color:"#64748b",fontSize:12}}>No data for {selHour}:00</div>
          ) : (
            <>
              <div style={{fontSize:10,color:"#475569",marginBottom:8,letterSpacing:0.3}}>
                {cur.n_train} train / {cur.n_val} val · base WR {cur.base_wr_val}% · {cur.n_thresholds_evaluated} thresholds evaluated
              </div>

              {/* Val edge distribution */}
              <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(255,255,255,0.02)"}}>
                <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Validation Edge Distribution Across All Thresholds</div>
                <div style={{display:"flex",gap:16,fontSize:11,color:"#94a3b8"}}>
                  <span>Min: <span style={{color:edgeColor(cur.val_edge_stats.min),fontWeight:600}}>{fmtEdge(cur.val_edge_stats.min)}</span></span>
                  <span>P10: <span style={{color:edgeColor(cur.val_edge_stats.p10),fontWeight:600}}>{fmtEdge(cur.val_edge_stats.p10)}</span></span>
                  <span>Median: <span style={{color:edgeColor(cur.val_edge_stats.p50),fontWeight:600}}>{fmtEdge(cur.val_edge_stats.p50)}</span></span>
                  <span>P90: <span style={{color:edgeColor(cur.val_edge_stats.p90),fontWeight:600}}>{fmtEdge(cur.val_edge_stats.p90)}</span></span>
                  <span>Max: <span style={{color:edgeColor(cur.val_edge_stats.max),fontWeight:600}}>{fmtEdge(cur.val_edge_stats.max)}</span></span>
                </div>
              </div>

              {/* Buckets table */}
              <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
                <thead><tr>
                  {["Train Edge ≥","Val Edge ≥","# Patterns Passing","Best Example"].map(h=>
                    <th key={h} style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase"}}>{h}</th>)}
                </tr></thead>
                <tbody>
                  {cur.buckets.map((b,i)=>{
                    const top = b.top_examples && b.top_examples[0];
                    return <tr key={i} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b.train_edge>0?"+":""}{b.train_edge}%</td>
                      <td style={{padding:"6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b.val_edge>0?"+":""}{b.val_edge}%</td>
                      <td style={{padding:"6px",color:b.n_passing>0?"#22c55e":"#64748b",fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{b.n_passing}</td>
                      <td style={{padding:"6px",color:"#94a3b8",fontSize:10}}>
                        {top ? (
                          <span>{top.feature} {top.op==">=" ? "≥" : "≤"} {Math.abs(top.value)<0.01?top.value.toExponential(2):top.value.toFixed(3)} · train {top.train.edge>0?"+":""}{top.train.edge}% / val <span style={{color:edgeColor(top.val.edge),fontWeight:600}}>{top.val.edge>0?"+":""}{top.val.edge}%</span> (n={top.val.n})</span>
                        ) : <span style={{color:"#475569"}}>—</span>}
                      </td>
                    </tr>;
                  })}
                </tbody>
              </table>

              <div style={{fontSize:10,color:"#475569",marginTop:8,lineHeight:1.5}}>
                If the top row (+5%/+3%) shows 0 but lower rows show many, there's weak signal that didn't pass the strict bar. If all rows show 0 even the bottom one (0%/-2%), the features have no predictive edge at all. Reading down the table shows where "signal" transitions to "noise" for this scan hour.
              </div>
            </>
          )}
        </>
      )}
    </Box>);
}


// ─── THRESHOLDS (v7 three-way split, Sharpe-selected) ────────────
function ThresholdsTab({health}) {
  const [prog,setProg]=useState(null);
  const [results,setResults]=useState(null);
  const [selHour,setSelHour]=useState(11);

  const poll=useCallback(()=>{
    fetch('/api/threshold/progress').then(r=>r.json()).then(setProg).catch(()=>{});
    fetch('/api/threshold/results').then(r=>r.json()).then(setResults).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,3000);return()=>clearInterval(iv);},[poll]);

  const run=async()=>{
    if(!confirm("Run threshold analysis?\n\nStrategy: at each scan hour, take top-1 stock IF its calibrated probability ≥ chosen threshold, else no trade.\n\nThreshold is chosen on VAL to maximize Sharpe (with guardrails), then evaluated on TEST (held out). Test numbers are the honest out-of-sample estimate.")) return;
    await fetch('/api/threshold/run',{method:'POST'});
    poll();
  };
  const reset=async()=>{
    if(!confirm("Delete threshold results? Live scanner will stop gating trades (no 'tradable' stocks).")) return;
    await fetch('/api/threshold/reset',{method:'POST'});
    poll();
  };

  const ip = prog?.inProgress;
  const hasData = prog?.hasTrainingData && prog?.hasModels;
  const hours = results?.hours || {};
  const hasResults = Object.values(hours).some(h => h?.val_curve);
  const cur = hours[String(selHour)];

  // Render a stats row for either val or test best result
  const StatsRow = ({label, stats, color}) => {
    if (!stats) return (
      <div style={{padding:"8px 12px",borderRadius:4,background:"rgba(100,116,139,0.05)",border:"1px solid rgba(100,116,139,0.15)",marginBottom:6}}>
        <span style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginRight:10}}>{label}</span>
        <span style={{fontSize:12,color:"#475569"}}>no data</span>
      </div>);
    return (
      <div style={{padding:"8px 12px",borderRadius:4,background:`rgba(${color},0.05)`,border:`1px solid rgba(${color},0.2)`,marginBottom:6}}>
        <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:3}}>{label}</div>
        <div style={{fontSize:12,lineHeight:1.7,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
          <span>Sharpe <b style={{color:stats.sharpe>0?"#22c55e":"#ef4444"}}>{stats.sharpe!=null?stats.sharpe:"—"}</b></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>WR <b>{stats.wr!=null?`${stats.wr}%`:"—"}</b></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>PnL/trade <b style={{color:stats.avg_pnl_trade>0?"#22c55e":"#ef4444"}}>{stats.avg_pnl_trade!=null?`${stats.avg_pnl_trade>0?"+":""}${stats.avg_pnl_trade}%`:"—"}</b></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>PnL/day <b style={{color:stats.avg_pnl_day>0?"#22c55e":"#ef4444"}}>{stats.avg_pnl_day!=null?`${stats.avg_pnl_day>0?"+":""}${stats.avg_pnl_day}%`:"—"}</b></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>CumPnL <b style={{color:stats.cum_pnl>0?"#22c55e":"#ef4444"}}>{stats.cum_pnl>0?"+":""}{stats.cum_pnl}%</b></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>Trades <b>{stats.n_trades}/{stats.total_days}</b> <span style={{color:"#64748b"}}>({(stats.trade_freq*100).toFixed(0)}% days)</span></span>
          <span style={{margin:"0 8px",color:"#334155"}}>·</span>
          <span>+days <b>{stats.pos_days}/{stats.n_trades}</b></span>
        </div>
      </div>);
  };

  // Detect overfitting by comparing val sharpe to test sharpe
  const overfitBadge = (valS, testS) => {
    if (valS == null || testS == null) return null;
    const gap = valS - testS;
    let color, label;
    if (gap < 0.1) { color = "#22c55e"; label = "HOLDS"; }
    else if (gap < 0.3) { color = "#eab308"; label = "SOME DROP"; }
    else { color = "#ef4444"; label = "LARGE DROP"; }
    return <span style={{padding:"2px 6px",borderRadius:3,background:`rgba(${color==="#22c55e"?"34,197,94":color==="#eab308"?"234,179,8":"239,68,68"},0.15)`,color,fontSize:10,fontWeight:700,letterSpacing:0.5,marginLeft:8}}>{label}</span>;
  };

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
          <Lbl>Conviction-Gated Strategy — Three-Way-Split Analyzer</Lbl>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            <Btn onClick={run} disabled={ip||!hasData} color="#06b6d4" style={{padding:"4px 10px",fontSize:11}}>
              {ip?"Running...":hasResults?"Re-run":"Run Analysis"}
            </Btn>
            {hasResults&&!ip&&<Btn onClick={()=>downloadJson(results,`threshold_results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"4px 10px",fontSize:11}}>Download JSON</Btn>}
            {hasResults&&!ip&&<Btn onClick={reset} color="#ef4444" style={{padding:"4px 10px",fontSize:11}}>Reset</Btn>}
          </div>
        </div>

        {!hasData && <div style={{fontSize:12,color:"#64748b",padding:"12px 0"}}>Requires trained models + training data cache. Run Training first (v7 uses 60/20/20 split — older models trained on 80/20 will be rejected).</div>}

        {ip && (
          <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(6,182,212,0.08)",border:"1px solid rgba(6,182,212,0.2)"}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${prog.pct}%`,height:"100%",background:"#06b6d4",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#06b6d4",fontWeight:600}}>{prog.pct}%</span>
            </div>
            <div style={{fontSize:11,color:"#94a3b8"}}>{prog.message}</div>
          </div>
        )}

        {!hasResults && !ip && hasData && (
          <div style={{fontSize:12,color:"#94a3b8",padding:"12px 0",lineHeight:1.7}}>
            <div style={{marginBottom:8}}><b style={{color:"#e2e8f0"}}>Method:</b> For each scan hour, split training data 60/20/20 temporally into train/val/test. Model trains on train (val for early stopping). Threshold is chosen on val to maximize Sharpe (avg daily PnL / std daily PnL), subject to guardrails: ≥15 trades, ≥40% winning trades. Chosen threshold is then applied to test — the held-out set — for an honest out-of-sample estimate.</div>
            <div><b style={{color:"#e2e8f0"}}>Rule:</b> At each scan, take top-1 stock IF its calibrated probability ≥ chosen threshold. Otherwise no trade.</div>
          </div>
        )}

        {hasResults && results?.generatedAt && (
          <div style={{fontSize:11,color:"#94a3b8"}}>
            Generated {new Date(results.generatedAt).toLocaleString()} · Objective: Sharpe on val with guardrails (≥15 trades, ≥40% winning)
          </div>
        )}
      </Box>

      {hasResults && (
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:12,flexWrap:"wrap"}}>
            <Lbl>Per-Hour Results</Lbl>
            <div style={{flex:1}}/>
            {SCAN_HOURS.map(h=>{
              const hd = hours[String(h)];
              const ct = hd?.chosen_threshold;
              const label = ct!=null ? `${h}:00 (${ct})` : `${h}:00 ✕`;
              return <Btn key={h} active={h===selHour} onClick={()=>setSelHour(h)} style={{padding:"3px 8px",fontSize:10}}>{label}</Btn>;
            })}
          </div>

          {!cur || cur.error ? (
            <div style={{color:"#64748b",fontSize:12}}>No data for {selHour}:00 — {cur?.error||"no analysis"}</div>
          ) : (
            <>
              <div style={{fontSize:10,color:"#475569",marginBottom:10,letterSpacing:0.3,lineHeight:1.6}}>
                <b>Split:</b> train {cur.n_train_dates}d, val {cur.n_val_dates}d, test {cur.n_test_dates}d ·{" "}
                <b>Base WR:</b> val {cur.base_wr_val}%, test {cur.base_wr_test}% ·{" "}
                <b>Realized BE:</b> {cur.realized_be}% ·{" "}
                <b>Eligible thresholds on val:</b> {cur.eligible_count}
              </div>

              {cur.chosen_threshold == null ? (
                <div style={{padding:"12px 14px",borderRadius:4,background:"rgba(239,68,68,0.08)",border:"1px solid rgba(239,68,68,0.25)",marginBottom:10}}>
                  <div style={{fontSize:12,color:"#ef4444",fontWeight:700,letterSpacing:0.5,marginBottom:2}}>NO-TRADE HOUR</div>
                  <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.5}}>No threshold passed the guardrails on val (≥15 trades, ≥40% winning trades, Sharpe defined). Live scanner will not mark any stock tradable at this hour.</div>
                </div>
              ) : (
                <div style={{marginBottom:14,padding:"10px 12px",borderRadius:4,background:"rgba(6,182,212,0.06)",border:"1px solid rgba(6,182,212,0.25)"}}>
                  <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>
                    Chosen threshold: <b style={{color:"#06b6d4",fontSize:14}}>≥ {cur.chosen_threshold}</b>
                    {overfitBadge(cur.best_val?.sharpe, cur.best_test?.sharpe)}
                  </div>
                  <StatsRow label="VAL (in-sample, threshold chosen here)" stats={cur.best_val} color="234,179,8"/>
                  <StatsRow label="TEST (out-of-sample, honest estimate)" stats={cur.best_test} color="34,197,94"/>
                </div>
              )}

              <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,margin:"14px 0 6px"}}>
                Full curve — val vs test by threshold
              </div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                  <thead>
                    <tr>
                      <th rowSpan={2} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderRight:"1px solid rgba(255,255,255,0.06)"}}>Thr</th>
                      <th colSpan={4} style={{padding:"4px 8px",textAlign:"center",color:"#eab308",fontSize:10,fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",borderRight:"1px solid rgba(255,255,255,0.06)"}}>VAL (in-sample)</th>
                      <th colSpan={4} style={{padding:"4px 8px",textAlign:"center",color:"#22c55e",fontSize:10,fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>TEST (out-of-sample)</th>
                    </tr>
                    <tr>
                      {["Trades","WR","Sharpe","Cum"].map((h,i)=>
                        <th key={"v"+i} style={{padding:"3px 6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderRight:i===3?"1px solid rgba(255,255,255,0.06)":"none"}}>{h}</th>)}
                      {["Trades","WR","Sharpe","Cum"].map((h,i)=>
                        <th key={"t"+i} style={{padding:"3px 6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {(cur.val_curve||[]).map((vc,i)=>{
                      const tc = (cur.test_curve||[])[i];
                      const isChosen = cur.chosen_threshold === vc.threshold;
                      const row = (c, isLast=false) => c ? (
                        <>
                          <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{c.n_trades}</td>
                          <td style={{padding:"3px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{c.wr!=null?`${c.wr}%`:"—"}</td>
                          <td style={{padding:"3px 6px",color:c.sharpe>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{c.sharpe!=null?c.sharpe:"—"}</td>
                          <td style={{padding:"3px 6px",color:c.cum_pnl>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums",borderRight:isLast?"none":"1px solid rgba(255,255,255,0.06)"}}>{c.cum_pnl>0?"+":""}{c.cum_pnl}%</td>
                        </>
                      ) : (
                        <>
                          <td colSpan={4} style={{padding:"3px 6px",color:"#334155",fontVariantNumeric:"tabular-nums",borderRight:isLast?"none":"1px solid rgba(255,255,255,0.06)"}}>—</td>
                        </>
                      );
                      return <tr key={i} style={{borderTop:"1px solid rgba(255,255,255,0.04)",background:isChosen?"rgba(6,182,212,0.08)":"transparent"}}>
                        <td style={{padding:"3px 6px",color:isChosen?"#06b6d4":"#e2e8f0",fontWeight:isChosen?700:500,fontVariantNumeric:"tabular-nums",borderRight:"1px solid rgba(255,255,255,0.06)"}}>{vc.threshold}{isChosen&&" ★"}</td>
                        {row(vc)}
                        {row(tc, true)}
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>

              <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.6}}>
                <b style={{color:"#94a3b8"}}>How to read:</b> The chosen threshold row (cyan, ★) is the one picked by maximum Sharpe on val with guardrails. VAL columns show that threshold's in-sample numbers (potentially optimistic). TEST columns are the same threshold evaluated on held-out data — this is the honest estimate of live performance. If TEST Sharpe / WR are close to VAL, signal is real. If TEST numbers collapse vs VAL, the chosen threshold was overfit.
              </div>
            </>
          )}
        </Box>
      )}
    </div>);
}


// ─── SETUPS (v9: hypothesis-first technical pattern scanner) ─────
function SetupsTab() {
  const [prog,setProg]=useState(null);
  const [results,setResults]=useState(null);
  const [live,setLive]=useState(null);
  const [selHour,setSelHour]=useState(11);

  const poll=useCallback(()=>{
    fetch('/api/setup/progress').then(r=>r.json()).then(setProg).catch(()=>{});
    fetch('/api/setup/results').then(r=>r.json()).then(setResults).catch(()=>{});
    fetch('/api/setup/live').then(r=>r.json()).then(setLive).catch(()=>{});
  },[]);
  useEffect(()=>{poll();const iv=setInterval(poll,5000);return()=>clearInterval(iv);},[poll]);

  const run=async()=>{
    if(!confirm("Run setup evaluation?\n\nScans 12 months of bar data for 5 predefined technical setups. Three-way split (60/20/20). Each setup's hit rate on +1% target reported per fold. TEST column is held out from training selection — honest out-of-sample numbers.\n\nTakes ~2-3 minutes.")) return;
    await fetch('/api/setup/run',{method:'POST'});
    poll();
  };
  const reset=async()=>{
    if(!confirm("Delete setup results?")) return;
    await fetch('/api/setup/reset',{method:'POST'});
    poll();
  };

  const ip = prog?.inProgress;
  const hasBars = prog?.hasBars;
  const hours = results?.hours || {};
  const hasResults = Object.keys(hours).length > 0;
  const setupDefs = results?.setups || {};
  const setupNames = Object.keys(setupDefs);
  const cur = hours[String(selHour)];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
          <Lbl>Technical Setups — Hypothesis-First Pattern Scanner (v9)</Lbl>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            <Btn onClick={run} disabled={ip||!hasBars} color="#a855f7" style={{padding:"4px 10px",fontSize:11}}>
              {ip?"Running...":hasResults?"Re-run":"Run Setup Evaluation"}
            </Btn>
            {hasResults&&!ip&&<Btn onClick={()=>downloadJson(results,`setup_results_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.json`)} color="#06b6d4" style={{padding:"4px 10px",fontSize:11}}>Download JSON</Btn>}
            {hasResults&&!ip&&<Btn onClick={reset} color="#ef4444" style={{padding:"4px 10px",fontSize:11}}>Reset</Btn>}
          </div>
        </div>

        {!hasBars && <div style={{fontSize:12,color:"#64748b",padding:"12px 0"}}>Requires cached bar data. Run Training first to populate cache (~20 min on first run, or already exists if Training has been run).</div>}

        {ip && (
          <div style={{marginBottom:12,padding:"8px 12px",borderRadius:4,background:"rgba(168,85,247,0.08)",border:"1px solid rgba(168,85,247,0.2)"}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                <div style={{width:`${prog.pct}%`,height:"100%",background:"#a855f7",borderRadius:3,transition:"width 0.5s"}}/>
              </div>
              <span style={{fontSize:11,color:"#a855f7",fontWeight:600}}>{prog.pct}%</span>
            </div>
            <div style={{fontSize:11,color:"#94a3b8"}}>{prog.message}</div>
          </div>
        )}

        {!hasResults && !ip && hasBars && (
          <div style={{fontSize:12,color:"#94a3b8",padding:"12px 0",lineHeight:1.7}}>
            <div style={{marginBottom:10}}><b style={{color:"#e2e8f0"}}>Target:</b> price reaches entry × 1.01 (+1%) at any point before 15:55 ET force-close. Binary hit/miss outcome per event.</div>
            <div style={{marginBottom:10}}><b style={{color:"#e2e8f0"}}>Method:</b> For each scan hour and each predefined setup, detect every historical firing. Three-way split (60% train / 20% val / 20% test) by date. Hit rate computed per fold. Test column is held out and reveals if the setup's edge is real or in-sample overfit.</div>
            <div><b style={{color:"#e2e8f0"}}>Five setups to test:</b>
              <ul style={{margin:"6px 0 0 20px",fontSize:11,color:"#64748b"}}>
                <li><b>ORB + Volume:</b> Break above first 30-min high with &gt;1.5× avg volume, above VWAP</li>
                <li><b>VWAP Reclaim:</b> Gap up &gt;0.5%, gap filled (touched VWAP), reclaimed and held above 2+ bars</li>
                <li><b>Consolidation Breakout:</b> Tight 30-min range (&lt;0.5%) with elevated volume, break above range</li>
                <li><b>Bull Flag:</b> 2%+ pole, 30-60% retrace, 4+ bar consolidation, breakout above flag</li>
                <li><b>Sector Bounce:</b> At intraday low with sector breadth &gt;50% green, green volume bar</li>
              </ul>
            </div>
          </div>
        )}

        {hasResults && (
          <div style={{fontSize:11,color:"#94a3b8"}}>
            Generated {new Date(results.generatedAt).toLocaleString()} · Target +{results.target_pct}% before 15:55 ET ·{" "}
            Split: train {results.n_train_dates}d / val {results.n_val_dates}d / test {results.n_test_dates}d
            {results.cache_fingerprint && (
              <div style={{marginTop:4,fontSize:10,color:"#475569",fontFamily:"monospace"}}>
                Cache fingerprint: <span style={{color:"#64748b"}}>{results.cache_fingerprint.hash}</span>
                {" · "}{results.cache_fingerprint.n_tickers} tickers
                {" · "}{results.cache_fingerprint.total_bars.toLocaleString()} bars
                {" · "}IWM: {results.cache_fingerprint.has_iwm ? `${results.cache_fingerprint.n_iwm_bars} bars` : <span style={{color:"#ef4444"}}>MISSING</span>}
                {results.date_range && <> · range: {results.date_range.first} → {results.date_range.last}</>}
                {results.cache_last_modified && <> · cache built: {new Date(results.cache_last_modified).toLocaleString()}</>}
              </div>
            )}
          </div>
        )}
      </Box>

      {hasResults && (
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:12,flexWrap:"wrap"}}>
            <Lbl>Per-Hour Setup Performance</Lbl>
            <div style={{flex:1}}/>
            {SCAN_HOURS.map(h=><Btn key={h} active={h===selHour} onClick={()=>setSelHour(h)} style={{padding:"3px 8px",fontSize:10}}>{h}:00</Btn>)}
          </div>

          {!cur ? (
            <div style={{color:"#64748b",fontSize:12}}>No data for {selHour}:00</div>
          ) : (
            <>
              {/* Base rate header */}
              <div style={{marginBottom:14,padding:"10px 12px",borderRadius:4,background:"rgba(100,116,139,0.06)",border:"1px solid rgba(100,116,139,0.15)"}}>
                <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:4}}>Base rate @{selHour}:00 — all stocks with valid bars, what fraction hit +1% before close?</div>
                <div style={{fontSize:12,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
                  <span>Train: <b>{cur.base.train.hit_rate}%</b> (n={cur.base.train.n_observations})</span>
                  <span style={{margin:"0 10px",color:"#334155"}}>|</span>
                  <span>Val: <b>{cur.base.val.hit_rate}%</b> (n={cur.base.val.n_observations})</span>
                  <span style={{margin:"0 10px",color:"#334155"}}>|</span>
                  <span>Test: <b style={{color:"#eab308"}}>{cur.base.test.hit_rate}%</b> (n={cur.base.test.n_observations})</span>
                </div>
                <div style={{fontSize:10,color:"#475569",marginTop:3}}>Setups must beat these base rates to be useful. Edge = setup hit rate − base rate.</div>
              </div>

              {/* Setups table */}
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                  <thead>
                    <tr>
                      <th rowSpan={2} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderRight:"1px solid rgba(255,255,255,0.06)"}}>Setup</th>
                      <th colSpan={3} style={{padding:"4px 8px",textAlign:"center",color:"#94a3b8",fontSize:10,fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",borderRight:"1px solid rgba(255,255,255,0.06)"}}>Train</th>
                      <th colSpan={3} style={{padding:"4px 8px",textAlign:"center",color:"#eab308",fontSize:10,fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",borderRight:"1px solid rgba(255,255,255,0.06)"}}>Val</th>
                      <th colSpan={3} style={{padding:"4px 8px",textAlign:"center",color:"#22c55e",fontSize:10,fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>Test (honest)</th>
                    </tr>
                    <tr>
                      {["n","Hit%","Edge","n","Hit%","Edge","n","Hit%","Edge"].map((h,i)=>
                        <th key={i} style={{padding:"3px 6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderRight:(i===2||i===5)?"1px solid rgba(255,255,255,0.06)":"none"}}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {setupNames.map(s=>{
                      const ss = cur.setups[s];
                      const testEdge = ss.test.edge_vs_base;
                      const testN = ss.test.n_events;
                      // v11: STRONG tier requires positive edge in ≥1 adjacent scan hour.
                      // Isolated strong-looking results get demoted to guard against multi-testing false positives.
                      const curIdx = SCAN_HOURS.indexOf(selHour);
                      let hasAdjacent = false;
                      for(const off of [-1,1]) {
                        const ai = curIdx + off;
                        if (ai >= 0 && ai < SCAN_HOURS.length) {
                          const adjH = String(SCAN_HOURS[ai]);
                          const adjEdge = hours[adjH]?.setups?.[s]?.test?.edge_vs_base;
                          if (adjEdge != null && adjEdge > 0) { hasAdjacent = true; break; }
                        }
                      }
                      // Match server-side SETUP_EVIDENCE_THRESHOLDS + adjacency rule
                      let verdict;
                      if (testEdge == null) verdict = null;
                      else if (testEdge >= 5 && testN >= 100 && hasAdjacent) verdict = "STRONG";
                      else if (testEdge >= 3 && testN >= 25) verdict = "MODERATE";
                      else if (testEdge >= 5 && testN >= 100) verdict = "MODERATE"; // demoted strong
                      else if (testEdge > 0) verdict = "WEAK";
                      else if (testEdge > -2) verdict = "NONE";
                      else verdict = "NEGATIVE";
                      const verdictColor = {
                        STRONG:"#22c55e", MODERATE:"#a3e635", WEAK:"#eab308",
                        NONE:"#64748b", NEGATIVE:"#ef4444"
                      }[verdict];
                      return <tr key={s} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                        <td style={{padding:"6px 8px",borderRight:"1px solid rgba(255,255,255,0.06)"}}>
                          <div style={{color:"#e2e8f0",fontWeight:600}}>{s}</div>
                          <div style={{color:"#475569",fontSize:9,marginTop:2}}>{setupDefs[s]}</div>
                          {verdict && <div style={{marginTop:4,display:"inline-block",padding:"2px 6px",borderRadius:2,background:`rgba(${verdictColor==='#22c55e'?'34,197,94':verdictColor==='#a3e635'?'163,230,53':verdictColor==='#eab308'?'234,179,8':verdictColor==='#ef4444'?'239,68,68':'100,116,139'},0.15)`,color:verdictColor,fontSize:9,fontWeight:700,letterSpacing:0.5}}>{verdict}</div>}
                        </td>
                        {["train","val","test"].map((fold,fi)=>{
                          const f = ss[fold];
                          const lastCol = fi===2;
                          return <Fragment key={fold}>
                            <td style={{padding:"4px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{f.n_events}</td>
                            <td style={{padding:"4px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{f.hit_rate!=null?`${f.hit_rate}%`:"—"}</td>
                            <td style={{padding:"4px 6px",color:f.edge_vs_base>0?"#22c55e":f.edge_vs_base<0?"#ef4444":"#64748b",fontVariantNumeric:"tabular-nums",fontWeight:600,borderRight:lastCol?"none":"1px solid rgba(255,255,255,0.06)"}}>{f.edge_vs_base!=null?`${f.edge_vs_base>0?"+":""}${f.edge_vs_base}%`:"—"}</td>
                          </Fragment>;
                        })}
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>

              <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.6}}>
                <b style={{color:"#94a3b8"}}>How to read:</b> Each setup fires N times per fold. Hit% = fraction of firings that reached +1% before close. Edge = Hit% − base rate. TEST column is held-out data (setup wasn't selected using it). Verdict badges: <b style={{color:"#22c55e"}}>STRONG</b> (edge ≥5% AND n≥100 AND positive edge in an adjacent scan hour — real signals tend to persist across nearby hours); <b style={{color:"#a3e635"}}>MODERATE</b> (edge ≥3% AND n≥25, or isolated strong result); <b style={{color:"#eab308"}}>WEAK</b> (positive but below thresholds); <b style={{color:"#64748b"}}>NONE</b> (no edge); <b style={{color:"#ef4444"}}>NEGATIVE</b> (below base rate). The adjacency rule guards against multi-testing false positives — with 20 setups × 5 hours = 100 tests, some spurious "strong" results are expected by chance; real signals generalize to adjacent hours.
              </div>

              {/* Days with firings — to see if these fire often enough to be useful */}
              <div style={{marginTop:14}}>
                <div style={{fontSize:10,color:"#64748b",textTransform:"uppercase",letterSpacing:0.5,marginBottom:6}}>Firing frequency (test fold)</div>
                <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:8}}>
                  {setupNames.map(s=>{
                    const t = cur.setups[s].test;
                    return <div key={s} style={{padding:"6px 8px",borderRadius:3,background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)"}}>
                      <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,marginBottom:2}}>{s}</div>
                      <div style={{fontSize:11,color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>
                        {t.firing_days}/{t.total_days} days <span style={{color:"#64748b"}}>({(t.firing_day_frac*100).toFixed(0)}%)</span>
                      </div>
                      <div style={{fontSize:10,color:"#475569"}}>{t.n_events} events total</div>
                    </div>;
                  })}
                </div>
              </div>
            </>
          )}
        </Box>
      )}

      {/* v12: COMBINATIONS analysis */}
      {hasResults && results.combinations && (
        <Box>
          <Lbl>Setup Combinations (v12) — which pairs of setups, when firing simultaneously, lift the edge</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each scan hour, pairs of surviving setups that fire together on the same stock. "Combined edge" = hit rate of both-firing vs base rate. "Lift" = combined edge − best individual edge. A positive lift ≥2% means the pair adds real synergy beyond the best single signal. Tested on TEST fold only, min n=20 per pair.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.combinations.by_hour?.[String(h)] || {pairs:[], n_pairs_tested:0};
            if (hd.pairs.length === 0) {
              return <div key={h} style={{marginBottom:10,fontSize:11,color:"#475569"}}>
                <b style={{color:"#94a3b8"}}>{h}:00</b> · {hd.n_pairs_tested===0?"no testable pairs":"no pairs met n≥20"}
              </div>;
            }
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>{h}:00 ET · base hit {hd.base_test_hit_rate}% · {hd.pairs.length} pair{hd.pairs.length===1?"":"s"} tested</div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Setup A + Setup B","Both n","Combined Hit%","Combined Edge","A edge (solo)","B edge (solo)","Lift","Synergy"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {hd.pairs.map((p,pi)=>{
                    const liftColor = p.lift_over_best_individual >= 2 ? "#22c55e" : p.lift_over_best_individual >= 0 ? "#eab308" : "#ef4444";
                    return <tr key={pi} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"4px 8px",color:"#e2e8f0"}}>{p.setup_a} + {p.setup_b}</td>
                      <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.n}</td>
                      <td style={{padding:"4px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{p.hit_rate}%</td>
                      <td style={{padding:"4px 8px",color:p.edge_vs_base>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{p.edge_vs_base>0?"+":""}{p.edge_vs_base}%</td>
                      <td style={{padding:"4px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>+{p.individual_a_edge}% <span style={{color:"#475569"}}>(n{p.individual_a_n})</span></td>
                      <td style={{padding:"4px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>+{p.individual_b_edge}% <span style={{color:"#475569"}}>(n{p.individual_b_n})</span></td>
                      <td style={{padding:"4px 8px",color:liftColor,fontVariantNumeric:"tabular-nums",fontWeight:700}}>{p.lift_over_best_individual>0?"+":""}{p.lift_over_best_individual}%</td>
                      <td style={{padding:"4px 8px",fontSize:10}}>{p.is_synergistic ? <span style={{color:"#22c55e",fontWeight:700}}>✓ synergy</span> : <span style={{color:"#64748b"}}>—</span>}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v12: REGIMES analysis */}
      {hasResults && results.regimes && (
        <Box>
          <Lbl>Regime Analysis (v12) — does each setup work equally in all market conditions?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each surviving setup, hit rate split by three regime dimensions: <b>spy_dir</b> (SPY up/flat/down at scan time), <b>vol</b> (SPY 5-day ATR high/low), <b>breadth</b> (% of R2K universe green at scan). A ⚑ flag appears when a regime split shows ≥5% edge spread across regimes with n≥15 per regime — i.e. the setup behaves meaningfully differently in different conditions. Test fold only.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.regimes.by_hour?.[String(h)] || {};
            const setups = Object.keys(hd);
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:16}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:6}}>{h}:00 ET</div>
              {setups.map(s=>{
                const sd = hd[s];
                return <div key={s} style={{marginBottom:10,paddingLeft:12,borderLeft:"2px solid rgba(255,255,255,0.06)"}}>
                  <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:4}}>{s}</div>
                  {["spy_dir","vol","breadth"].map(dim=>{
                    const rd = sd[dim] || {};
                    const flagged = rd._flag_conditional;
                    const spread = rd._flag_spread;
                    const regime_keys = Object.keys(rd).filter(k=>!k.startsWith("_"));
                    if (regime_keys.length === 0) return null;
                    return <div key={dim} style={{fontSize:10,color:"#94a3b8",marginBottom:3,fontFamily:"monospace"}}>
                      <span style={{color:"#64748b",display:"inline-block",minWidth:70}}>{dim}:</span>
                      {regime_keys.map(rv=>{
                        const r = rd[rv];
                        const edgeColor = r.edge_vs_base>3?"#22c55e":r.edge_vs_base>0?"#a3e635":r.edge_vs_base>-3?"#eab308":"#ef4444";
                        return <span key={rv} style={{marginRight:12}}>
                          <span style={{color:"#64748b"}}>{rv}</span> <span style={{color:edgeColor,fontWeight:600}}>{r.edge_vs_base>0?"+":""}{r.edge_vs_base}%</span> <span style={{color:"#475569"}}>(n{r.n_events})</span>
                        </span>;
                      })}
                      {flagged && <span style={{color:"#f59e0b",fontWeight:700,marginLeft:8}}>⚑ {spread}% spread</span>}
                    </div>;
                  })}
                </div>;
              })}
            </div>;
          })}
        </Box>
      )}

      {/* v14: FOLD-SWAP validation */}
      {hasResults && results.fold_swap && (
        <Box>
          <Lbl>Fold-Swap Robustness (v14) — does each setup's edge hold across different test windows?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Original three-way split used the most recent 57 days as test. Fold-swap rotates 5 non-overlapping 57-day windows (F_A = most recent → F_E = oldest) as "test" and recomputes each setup's edge. A real signal generalizes across folds; an overfit signal only works in the original test window. <b>ROBUST</b> = edge &gt;0 in ≥4/5 folds AND mean ≥3% AND min ≥-2%. <b>CONSISTENT</b> = edge &gt;0 in 3/5. <b>FRAGILE</b> = edge &gt;0 in fewer than 3 folds.
          </div>
          {results.fold_swap.fold_windows && (
            <div style={{fontSize:10,color:"#64748b",marginBottom:10,fontFamily:"monospace"}}>
              {Object.entries(results.fold_swap.fold_windows).map(([name,fw])=>
                <span key={name} style={{marginRight:16}}>{name}: {fw.first_date}→{fw.last_date} ({fw.n_days}d)</span>
              )}
            </div>
          )}
          {SCAN_HOURS.map(h=>{
            const hr = results.fold_swap.by_hour?.[String(h)] || {};
            const setups = Object.keys(hr);
            // Only show setups that are in the active-setup list from main eval (survived tier gate)
            const mainHr = results.hours?.[String(h)] || {};
            const mainSetups = mainHr.setups || {};
            const activeSetups = setups.filter(s=>{
              const tst = mainSetups[s]?.test||{};
              return tst.edge_vs_base!=null && tst.edge_vs_base >= 3 && tst.n_events >= 25;
            });
            if (activeSetups.length === 0) return null;
            return <div key={h} style={{marginBottom:16}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:6}}>{h}:00 ET</div>
              <div style={{overflowX:"auto"}}>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                  <thead><tr>
                    <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Setup</th>
                    {["F_A","F_B","F_C","F_D","F_E"].map(fn=>
                      <th key={fn} style={{padding:"4px 6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",minWidth:80}}>{fn}</th>)}
                    <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>+/neg</th>
                    <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Mean</th>
                    <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Min</th>
                    <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Robustness</th>
                  </tr></thead>
                  <tbody>
                    {activeSetups.map(s=>{
                      const sd = hr[s];
                      const rob = sd.robustness;
                      const robColor = rob === "ROBUST" ? "#22c55e" : rob === "CONSISTENT" ? "#a3e635" : rob === "FRAGILE" ? "#ef4444" : "#64748b";
                      const robBg = rob === "ROBUST" ? "rgba(34,197,94,0.18)" : rob === "CONSISTENT" ? "rgba(163,230,53,0.15)" : rob === "FRAGILE" ? "rgba(239,68,68,0.15)" : "rgba(100,116,139,0.10)";
                      return <tr key={s} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                        <td style={{padding:"5px 8px",color:"#e2e8f0",fontWeight:500}}>{s}</td>
                        {["F_A","F_B","F_C","F_D","F_E"].map(fn=>{
                          const f = sd.folds?.[fn];
                          if (!f || f.edge == null) {
                            return <td key={fn} style={{padding:"5px 6px",color:"#475569",fontVariantNumeric:"tabular-nums",fontSize:10}}>{f?.n_events != null ? `n=${f.n_events}` : "—"}</td>;
                          }
                          const c = f.edge > 3 ? "#22c55e" : f.edge > 0 ? "#a3e635" : f.edge > -3 ? "#eab308" : "#ef4444";
                          return <td key={fn} style={{padding:"5px 6px",color:c,fontVariantNumeric:"tabular-nums",fontWeight:600,fontSize:10}}>
                            {f.edge>0?"+":""}{f.edge}% <span style={{color:"#475569",fontWeight:400}}>(n{f.n_events})</span>
                          </td>;
                        })}
                        <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums",fontSize:11}}>{sd.n_positive_folds}/{sd.n_valid_folds}</td>
                        <td style={{padding:"5px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{sd.mean_edge!=null?`${sd.mean_edge>0?"+":""}${sd.mean_edge}%`:"—"}</td>
                        <td style={{padding:"5px 8px",color:sd.min_edge<0?"#ef4444":"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{sd.min_edge!=null?`${sd.min_edge>0?"+":""}${sd.min_edge}%`:"—"}</td>
                        <td style={{padding:"5px 8px"}}>
                          <span style={{display:"inline-block",padding:"2px 7px",borderRadius:2,background:robBg,color:robColor,fontSize:9,fontWeight:700,letterSpacing:0.5}}>{rob}</span>
                        </td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>
            </div>;
          })}
        </Box>
      )}

      {/* v16-A2: EX-F_A VALIDATION */}
      {hasResults && results.ex_fa && (
        <Box>
          <Lbl>Ex-F_A Cross-Validation (v16) — does each survivor hold up on the combined other 4 folds?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Independent cross-check of fold-swap. Each survivor's edge is re-measured on the UNION of F_B∪F_C∪F_D∪F_E ({results.ex_fa.n_ex_fa_dates} days total, {results.ex_fa.date_range?.first} → {results.ex_fa.date_range?.last}). "Holds up" = edge ≥3% on this combined alternate fold. "Edge Δ" = Ex-F_A edge minus original F_A edge; positive means the setup actually looks better outside its original test window.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.ex_fa.by_hour?.[String(h)] || {setups:{}};
            const setups = Object.keys(hd.setups || {});
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>{h}:00 ET · Ex-F_A base hit {hd.base_hit_rate}% (n={hd.n_base})</div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Setup","F_A edge","Ex-F_A n","Ex-F_A edge","Edge Δ","Holds?"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {setups.map(s=>{
                    const d = hd.setups[s];
                    const holds = d.holds_up;
                    const deltaColor = d.edge_delta>0?"#22c55e":d.edge_delta<-2?"#ef4444":"#eab308";
                    return <tr key={s} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"5px 8px",color:"#e2e8f0",fontWeight:500}}>{s}</td>
                      <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{d.fa_edge!=null?`${d.fa_edge>0?"+":""}${d.fa_edge}%`:"—"}</td>
                      <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{d.ex_fa_n}</td>
                      <td style={{padding:"5px 8px",color:d.ex_fa_edge>3?"#22c55e":d.ex_fa_edge>0?"#a3e635":d.ex_fa_edge>-3?"#eab308":"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{d.ex_fa_edge!=null?`${d.ex_fa_edge>0?"+":""}${d.ex_fa_edge}%`:"—"}</td>
                      <td style={{padding:"5px 8px",color:deltaColor,fontVariantNumeric:"tabular-nums"}}>{d.edge_delta!=null?`${d.edge_delta>0?"+":""}${d.edge_delta}%`:"—"}</td>
                      <td style={{padding:"5px 8px"}}>{holds ? <span style={{color:"#22c55e",fontWeight:700}}>✓ holds</span> : <span style={{color:"#ef4444"}}>✗</span>}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v16-B1: SETUP OVERLAP */}
      {hasResults && results.overlap && (
        <Box>
          <Lbl>Setup Overlap (v16) — are our setups independent or do they fire on the same stocks?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Jaccard overlap of firing sets per (date, ticker): how often do two setups fire on the <i>same stock at the same scan time</i>? High overlap = redundant signals. Low overlap = independent signals. Jaccard = |A ∩ B| / |A ∪ B|. <b style={{color:"#ef4444"}}>redundant</b> (≥0.5), <b style={{color:"#eab308"}}>correlated</b> (0.2–0.5), <b style={{color:"#22c55e"}}>independent</b> (&lt;0.2). Test fold only.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.overlap.by_hour?.[String(h)];
            const pairs = hd?.pairs || [];
            if (pairs.length === 0) return null;
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>{h}:00 ET · {hd.n_survivors} survivors, {pairs.length} pairs</div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Setup A","Setup B","A only","B only","Both","Jaccard","Class"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {pairs.map((p,pi)=>{
                    const cls = p.classification;
                    const clsColor = cls==="redundant"?"#ef4444":cls==="correlated"?"#eab308":"#22c55e";
                    return <tr key={pi} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"4px 8px",color:"#e2e8f0"}}>{p.setup_a}</td>
                      <td style={{padding:"4px 8px",color:"#e2e8f0"}}>{p.setup_b}</td>
                      <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.n_a_only}</td>
                      <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{p.n_b_only}</td>
                      <td style={{padding:"4px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{p.n_both}</td>
                      <td style={{padding:"4px 8px",color:clsColor,fontVariantNumeric:"tabular-nums",fontWeight:700}}>{p.jaccard}</td>
                      <td style={{padding:"4px 8px",color:clsColor,fontSize:10,fontWeight:700,textTransform:"uppercase"}}>{cls}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v16-B2: MULTI-SETUP STACKING */}
      {hasResults && results.stacking && (
        <Box>
          <Lbl>Multi-Setup Stacking (v16) — does more simultaneous-firing setups = higher hit rate?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Per scan hour, for each stock at that scan time, count how many of the original-survivor setups fire on it. Then compute hit rate grouped by that count. If stacking adds conviction, higher counts should correlate with higher hit rates. Test fold only.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.stacking.by_hour?.[String(h)];
            if (!hd || !hd.by_stack_count) return null;
            const buckets = hd.by_stack_count;
            if (buckets.length === 0) return null;
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>
                {h}:00 ET · {hd.n_survivors} survivors · base hit {hd.base_hit_rate}% (n={hd.n_base})
                {hd.monotonic_increasing && <span style={{color:"#22c55e",marginLeft:8,fontWeight:700}}>✓ MONOTONIC (more stack = more edge)</span>}
              </div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Stack Count","N","Hit %","Edge vs base","Mean PnL"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {buckets.map((b,bi)=>{
                    const edgeColor = b.edge_vs_base>5?"#22c55e":b.edge_vs_base>2?"#a3e635":b.edge_vs_base>-2?"#94a3b8":"#ef4444";
                    return <tr key={bi} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"5px 8px",color:"#e2e8f0",fontWeight:600,fontVariantNumeric:"tabular-nums"}}>{b.stack_count}{b.stack_count===0?" (no setups)":b.stack_count===1?" (single)":""}</td>
                      <td style={{padding:"5px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.n}</td>
                      <td style={{padding:"5px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{b.hit_rate!=null?`${b.hit_rate}%`:"—"}</td>
                      <td style={{padding:"5px 8px",color:edgeColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{b.edge_vs_base!=null?`${b.edge_vs_base>0?"+":""}${b.edge_vs_base}%`:"—"}</td>
                      <td style={{padding:"5px 8px",color:b.mean_pnl>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums"}}>{b.mean_pnl>0?"+":""}{b.mean_pnl}%</td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v16-A1: SECTOR BREAKDOWN */}
      {hasResults && results.sector_breakdown && (
        <Box>
          <Lbl>Sector Breakdown (v16) — do survivors work better in specific sectors?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each original survivor, test-fold hit rate per sector (min n=15 per sector). <b>Flagged sectors</b>: absolute deviation of the sector's edge-vs-sector-base from the setup's overall edge is ≥5%. "stronger" = sector shows bigger edge than setup average; "weaker" = sector shows smaller or negative edge. Informational — not used to filter live scanner yet.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.sector_breakdown.by_hour?.[String(h)] || {};
            const setups = Object.keys(hd);
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:16}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:6}}>{h}:00 ET</div>
              {setups.map(s=>{
                const sd = hd[s];
                const flagged = sd.flagged_sectors || [];
                return <div key={s} style={{marginBottom:10,paddingLeft:12,borderLeft:"2px solid rgba(255,255,255,0.06)"}}>
                  <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:4}}>
                    {s} <span style={{color:"#64748b",fontWeight:400,marginLeft:8}}>overall hit {sd.overall_test_hit_rate}% (n={sd.overall_test_n})</span>
                  </div>
                  {flagged.length === 0 ? (
                    <div style={{fontSize:10,color:"#475569"}}>No sectors deviate meaningfully (all within 5% of overall edge)</div>
                  ) : (
                    <div style={{fontSize:10,color:"#94a3b8",fontFamily:"monospace"}}>
                      {flagged.map((f,fi)=>{
                        const c = f.direction==="stronger" ? "#22c55e" : "#ef4444";
                        return <span key={fi} style={{marginRight:16}}>
                          <span style={{color:"#64748b"}}>{f.sector}</span>{" "}
                          <span style={{color:c,fontWeight:700}}>{f.direction}</span>{" "}
                          <span style={{color:"#94a3b8"}}>(edge {f.edge>0?"+":""}{f.edge}%, Δ {f.deviation>0?"+":""}{f.deviation}%)</span>
                        </span>;
                      })}
                    </div>
                  )}
                </div>;
              })}
            </div>;
          })}
        </Box>
      )}

      {/* v24: CONVICTION RANKING TEST */}
      {hasResults && results.conviction_ranking_test && (
        <Box>
          <Lbl>Conviction Ranking Test (v24) — does ranking by conviction actually improve hit rate?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each (date, hour), ranks firings by conviction score (test_edge + test_hit_rate × 0.5, with +20% ROBUST boost, redundant-pair dedup). Measures hit rate of top-N cutoffs vs all-firings hit rate across train / val / test folds. <b>Verdict</b>: RANKING_WORKS (top-10 Δ ≥+5pp on val+test), RANKING_HELPS_MARGINALLY (≥+2pp), RANKING_NEUTRAL (|Δ|&lt;2pp), RANKING_HURTS (any fold ≤-2pp). If RANKING_WORKS: deploy in live scanner. If not: revise scoring formula.
          </div>
          {(() => {
            const cr = results.conviction_ranking_test;
            const verdict = cr.verdict;
            const vColor = verdict==="RANKING_WORKS"?"#22c55e":verdict==="RANKING_HELPS_MARGINALLY"?"#a3e635":verdict==="RANKING_NEUTRAL"?"#94a3b8":verdict==="RANKING_HURTS"?"#ef4444":"#eab308";
            const cutoffKeys = ["top_1","top_3","top_5","top_10","top_20","all"];
            const folds = ["train","val","test"];
            return <>
              <div style={{marginBottom:10,padding:"6px 10px",borderRadius:4,background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)"}}>
                <span style={{fontSize:11,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase"}}>Verdict</span>
                <span style={{marginLeft:10,color:vColor,fontWeight:700,letterSpacing:0.3}}>{verdict}</span>
                <span style={{marginLeft:16,color:"#94a3b8",fontSize:11}}>Monotonic val: <b>{cr.monotonic_val?"✓":"✗"}</b></span>
                <span style={{marginLeft:10,color:"#94a3b8",fontSize:11}}>Monotonic test: <b>{cr.monotonic_test?"✓":"✗"}</b></span>
              </div>

              {/* Cutoff × fold table */}
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:14}}>
                <thead><tr>
                  <th style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>Cutoff</th>
                  {folds.map(f=><th key={f} colSpan={3} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase",borderLeft:"1px solid rgba(255,255,255,0.08)"}}>{f.toUpperCase()}</th>)}
                </tr><tr>
                  <th></th>
                  {folds.map(f=><Fragment key={f}>
                    <th style={{padding:"2px 8px",textAlign:"left",color:"#475569",fontSize:9,fontWeight:500,borderLeft:"1px solid rgba(255,255,255,0.08)"}}>n</th>
                    <th style={{padding:"2px 8px",textAlign:"left",color:"#475569",fontSize:9,fontWeight:500}}>Hit%</th>
                    <th style={{padding:"2px 8px",textAlign:"left",color:"#475569",fontSize:9,fontWeight:500}}>Δ vs all</th>
                  </Fragment>)}
                </tr></thead>
                <tbody>
                  {cutoffKeys.map(c=>{
                    const isAll = c==="all";
                    return <tr key={c} style={{borderTop:"1px solid rgba(255,255,255,0.03)",background:isAll?"rgba(255,255,255,0.02)":"transparent"}}>
                      <td style={{padding:"3px 8px",color:isAll?"#64748b":"#e2e8f0",fontWeight:isAll?400:600,fontStyle:isAll?"italic":"normal"}}>{c.replace("_"," ")}</td>
                      {folds.map(f=>{
                        const fd = cr.by_fold?.[f]?.cutoffs?.[c];
                        if (!fd) return <Fragment key={f}>
                          <td style={{padding:"3px 8px",borderLeft:"1px solid rgba(255,255,255,0.08)"}}>—</td><td></td><td></td>
                        </Fragment>;
                        const d = fd.delta_vs_all;
                        const dColor = d==null?"#64748b":d>=5?"#22c55e":d>=2?"#a3e635":d<-2?"#ef4444":"#94a3b8";
                        return <Fragment key={f}>
                          <td style={{padding:"3px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums",borderLeft:"1px solid rgba(255,255,255,0.08)"}}>{fd.n_stocks}</td>
                          <td style={{padding:"3px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{fd.hit_rate!=null?`${fd.hit_rate}%`:"—"}</td>
                          <td style={{padding:"3px 8px",color:dColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{isAll?"—":(d!=null?`${d>0?"+":""}${d}pp`:"—")}</td>
                        </Fragment>;
                      })}
                    </tr>;
                  })}
                </tbody>
              </table>

              {/* Per-setup scores reference */}
              <details>
                <summary style={{cursor:"pointer",fontSize:11,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3}}>Per-setup base scores (click to expand)</summary>
                <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginTop:8}}>
                  <thead><tr>
                    {["Setup","Score"].map(col=>
                      <th key={col} style={{padding:"3px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                  </tr></thead>
                  <tbody>
                    {Object.entries(cr.setup_hour_scores||{}).sort((a,b)=>b[1]-a[1]).map(([k,sc])=>
                      <tr key={k} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                        <td style={{padding:"3px 8px",color:"#e2e8f0"}}>{k}</td>
                        <td style={{padding:"3px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{sc}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </details>
            </>;
          })()}
        </Box>
      )}

      {/* v23: LIVE-READINESS SIMULATION */}
      {hasResults && results.live_simulation && (
        <Box>
          <Lbl>Live-Readiness Simulation (v23) — what would live scanner produce over last 30 days?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Simulates live scanner behavior on recent 30 days using <b>current active-setup list</b> (post Ex-F_A gate) + <b>ATR filter rule</b> (rel_strength_iwm @ 12/13/14 skip ATR% ≤ ~2.85%). Breadth filtering NOT applied (can't reconstruct live breadth retroactively) — so these counts are a MAXIMUM, real live will be equal or lower on breadth-down days.
          </div>

          {(() => {
            const ls = results.live_simulation;
            const sum = ls.summary || {};
            const dateRange = ls.sim_date_range || ["—","—"];
            return <>
              <div style={{fontSize:11,color:"#64748b",marginBottom:10}}>
                <b style={{color:"#94a3b8"}}>Period:</b> {dateRange[0]} → {dateRange[1]} · <b style={{color:"#94a3b8"}}>{ls.n_sim_days} trading days</b> · <b style={{color:"#94a3b8"}}>{ls.active_setup_count} active setups</b> (post-Ex_FA) · ATR filter on: {ls.atr_rule_setups?.join(", ")||"none"}
              </div>

              {/* Summary cards */}
              <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(120px,1fr))",gap:8,marginBottom:14}}>
                {[
                  {label:"Total firings",val:sum.total_firings,color:"#e2e8f0"},
                  {label:"Median/day",val:sum.median_per_day,color:"#06b6d4"},
                  {label:"Mean/day",val:sum.mean_per_day,color:"#06b6d4"},
                  {label:"Max/day",val:sum.max_per_day,color:sum.max_per_day>30?"#eab308":"#e2e8f0"},
                  {label:"Min/day",val:sum.min_per_day,color:"#e2e8f0"},
                  {label:"Zero-firing days",val:sum.zero_firing_days,color:sum.zero_firing_days>3?"#f97316":"#e2e8f0"},
                  {label:"30+ firing days",val:sum.high_volume_days_30plus,color:sum.high_volume_days_30plus>5?"#eab308":"#e2e8f0"},
                  {label:"ATR-filtered",val:sum.total_atr_filtered,color:"#94a3b8"},
                  {label:"Simultaneous",val:sum.total_simultaneous,color:"#94a3b8"},
                  {label:"Observed hit%",val:sum.sim_observed_hit_rate_pct!=null?`${sum.sim_observed_hit_rate_pct}%`:"—",color:"#22c55e"},
                ].map((c,ci)=>
                  <div key={ci} style={{padding:"8px 10px",borderRadius:4,background:"rgba(255,255,255,0.02)",border:"1px solid rgba(255,255,255,0.04)"}}>
                    <div style={{fontSize:9,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",marginBottom:3}}>{c.label}</div>
                    <div style={{fontSize:16,color:c.color,fontWeight:700,fontVariantNumeric:"tabular-nums"}}>{c.val ?? "—"}</div>
                  </div>
                )}
              </div>

              {/* Distributions */}
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12,marginBottom:14}}>
                <div>
                  <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:4}}>Hour Distribution</div>
                  <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                    <tbody>
                      {Object.entries(ls.hour_distribution||{}).sort((a,b)=>+a[0]-+b[0]).map(([h,n])=>{
                        const pct = sum.total_firings > 0 ? (n/sum.total_firings*100) : 0;
                        return <tr key={h} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                          <td style={{padding:"2px 6px",color:"#e2e8f0",width:60}}>{h}:00 ET</td>
                          <td style={{padding:"2px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",width:60}}>{n}</td>
                          <td style={{padding:"2px 6px",width:"100%"}}>
                            <div style={{height:6,background:"rgba(255,255,255,0.05)",borderRadius:2,overflow:"hidden"}}>
                              <div style={{width:`${pct}%`,height:"100%",background:"#06b6d4"}}/>
                            </div>
                          </td>
                          <td style={{padding:"2px 6px",color:"#64748b",fontSize:10,fontVariantNumeric:"tabular-nums",width:50}}>{pct.toFixed(1)}%</td>
                        </tr>;
                      })}
                    </tbody>
                  </table>
                </div>
                <div>
                  <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:4}}>Setup Distribution</div>
                  <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                    <tbody>
                      {Object.entries(ls.setup_distribution||{}).map(([s,n])=>{
                        const pct = sum.total_firings > 0 ? (n/sum.total_firings*100) : 0;
                        return <tr key={s} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                          <td style={{padding:"2px 6px",color:"#e2e8f0",width:140,fontSize:10}}>{s}</td>
                          <td style={{padding:"2px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",width:50}}>{n}</td>
                          <td style={{padding:"2px 6px",width:"100%"}}>
                            <div style={{height:6,background:"rgba(255,255,255,0.05)",borderRadius:2,overflow:"hidden"}}>
                              <div style={{width:`${pct}%`,height:"100%",background:"#22c55e"}}/>
                            </div>
                          </td>
                          <td style={{padding:"2px 6px",color:"#64748b",fontSize:10,fontVariantNumeric:"tabular-nums",width:50}}>{pct.toFixed(1)}%</td>
                        </tr>;
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Per-day table */}
              <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:4}}>Per-Day Firings</div>
              <div style={{maxHeight:300,overflow:"auto"}}>
                <table style={{width:"100%",fontSize:10,borderCollapse:"collapse"}}>
                  <thead style={{position:"sticky",top:0,background:"#020617"}}>
                    <tr>
                      {["Date","Total","ATR filt","Simul","Unique","11","12","13","14","15","Top setups"].map(col=>
                        <th key={col} style={{padding:"3px 6px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(ls.per_day||{}).sort((a,b)=>b[0].localeCompare(a[0])).map(([date,d])=>{
                      const totalC = d.total===0?"#ef4444":d.total>=30?"#eab308":"#e2e8f0";
                      const topSetups = Object.entries(d.by_setup||{}).sort((a,b)=>b[1]-a[1]).slice(0,3).map(([s,n])=>`${s}(${n})`).join(", ");
                      return <tr key={date} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                        <td style={{padding:"3px 6px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{date}</td>
                        <td style={{padding:"3px 6px",color:totalC,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{d.total}</td>
                        <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{d.atr_filtered}</td>
                        <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{d.simultaneous}</td>
                        <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{d.unique_tickers}</td>
                        {[11,12,13,14,15].map(h=>
                          <td key={h} style={{padding:"3px 6px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{(d.by_hour||{})[h]||0}</td>
                        )}
                        <td style={{padding:"3px 6px",color:"#94a3b8",fontSize:9}}>{topSetups}</td>
                      </tr>;
                    })}
                  </tbody>
                </table>
              </div>
              <div style={{fontSize:10,color:"#475569",marginTop:8,fontStyle:"italic"}}>{ls.note}</div>
            </>;
          })()}
        </Box>
      )}

      {/* v22: FILTER STACKING TEST */}
      {hasResults && results.stacking_filter_test && (
        <Box>
          <Lbl>Filter Stacking Test (v22) — does ATR + Financial stack add edge over ATR alone for rel_strength_iwm?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For rel_strength_iwm @ 12/13/14, compare 4 universes: <b>A</b> full, <b>B</b> ATR>low only (current live filter), <b>C</b> no-Financial only, <b>D</b> ATR>low + no-Financial stacked. Two verdicts per hour: <b>D vs A</b> (stack vs no filter) and <b>D vs B</b> (stack vs ATR alone — the key question). IMPROVES = Δedge ≥+1pp both folds.
          </div>
          {[12, 13, 14].map(h=>{
            const hd = results.stacking_filter_test.by_hour?.[String(h)];
            if (!hd || !hd.universes) return null;
            const universes = hd.universes;
            const comp = hd.comparisons || {};
            const dVsA = comp.D_vs_A || {};
            const dVsB = comp.D_vs_B || {};
            const vColor = (v)=>v==="IMPROVES"?"#22c55e":v==="MARGINAL"?"#a3e635":v==="NEUTRAL"?"#94a3b8":v==="HURTS"?"#ef4444":"#eab308";
            return <div key={h} style={{marginBottom:16}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>
                rel_strength_iwm @ {h}:00 ET · ATR threshold {hd.atr_threshold_pct?.toFixed(2)}%
              </div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse",marginBottom:6}}>
                <thead><tr>
                  {["Universe","Disc edge","Val edge","n firings"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {["A_full","B_atr_only","C_nofin_only","D_atr_and_nofin"].map((k,ki)=>{
                    const u = universes[k] || {};
                    const labels = {"A_full":"A: full","B_atr_only":"B: ATR only","C_nofin_only":"C: no-Fin only","D_atr_and_nofin":"D: ATR + no-Fin"};
                    const color = k==="B_atr_only"?"#06b6d4":k==="D_atr_and_nofin"?"#d946ef":"#94a3b8";
                    return <tr key={k} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"3px 8px",color:color,fontWeight:500}}>{labels[k]}</td>
                      <td style={{padding:"3px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{u.discovery?.edge!=null?`${u.discovery.edge>0?"+":""}${u.discovery.edge}%`:"—"}</td>
                      <td style={{padding:"3px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums"}}>{u.validation?.edge!=null?`${u.validation.edge>0?"+":""}${u.validation.edge}%`:"—"}</td>
                      <td style={{padding:"3px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{u.all?.n_setup!=null?u.all.n_setup:"—"}</td>
                    </tr>;
                  })}
                </tbody>
              </table>
              <div style={{display:"flex",gap:16,fontSize:11}}>
                <div style={{flex:1,padding:"6px 10px",borderRadius:4,background:"rgba(217,70,239,0.04)",border:"1px solid rgba(217,70,239,0.15)"}}>
                  <div style={{fontSize:10,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",marginBottom:3}}>D vs A (stack vs baseline)</div>
                  <span style={{color:vColor(dVsA.verdict),fontWeight:700,letterSpacing:0.3}}>{dVsA.verdict||"—"}</span>
                  <span style={{color:"#94a3b8",marginLeft:10,fontVariantNumeric:"tabular-nums"}}>
                    Δdisc {dVsA.edge_delta_disc!=null?`${dVsA.edge_delta_disc>0?"+":""}${dVsA.edge_delta_disc}pp`:"—"} · Δval {dVsA.edge_delta_val!=null?`${dVsA.edge_delta_val>0?"+":""}${dVsA.edge_delta_val}pp`:"—"} · vol {dVsA.volume_change_pct!=null?`${dVsA.volume_change_pct}%`:"—"}
                  </span>
                </div>
                <div style={{flex:1,padding:"6px 10px",borderRadius:4,background:"rgba(6,182,212,0.04)",border:"1px solid rgba(6,182,212,0.15)"}}>
                  <div style={{fontSize:10,color:"#64748b",fontWeight:600,letterSpacing:0.3,textTransform:"uppercase",marginBottom:3}}>D vs B (stack vs ATR alone) — KEY Q</div>
                  <span style={{color:vColor(dVsB.verdict),fontWeight:700,letterSpacing:0.3}}>{dVsB.verdict||"—"}</span>
                  <span style={{color:"#94a3b8",marginLeft:10,fontVariantNumeric:"tabular-nums"}}>
                    Δdisc {dVsB.edge_delta_disc!=null?`${dVsB.edge_delta_disc>0?"+":""}${dVsB.edge_delta_disc}pp`:"—"} · Δval {dVsB.edge_delta_val!=null?`${dVsB.edge_delta_val>0?"+":""}${dVsB.edge_delta_val}pp`:"—"} · vol {dVsB.volume_change_pct!=null?`${dVsB.volume_change_pct}%`:"—"}
                  </span>
                </div>
              </div>
            </div>;
          })}
        </Box>
      )}

      {/* v21: SECTOR EXCLUSION TEST */}
      {hasResults && results.sector_exclusion_test && (
        <Box>
          <Lbl>Financial Sector Exclusion Test (v21) — does excluding Financial improve edges?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each ROBUST setup, recompute edge with Financial stocks excluded (both setup firings AND base rate, apples-to-apples). <b>Verdict</b>: IMPROVES (Δedge ≥+1pp both folds), MARGINAL (≥+0.5pp), NEUTRAL (|Δ|&lt;0.5), HURTS (≤-0.5), INCONSISTENT. Tests whether v19's "Financial over-loser" finding is actionable.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.sector_exclusion_test.by_hour?.[String(h)] || {};
            const setups = Object.keys(hd.setups || {});
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>
                {h}:00 ET · {setups.length} ROBUST setup{setups.length===1?"":"s"}
              </div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Setup","Disc full→noFin","Δdisc","Val full→noFin","Δval","Vol Δ","Verdict"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {setups.map(s=>{
                    const sd = hd.setups[s];
                    const verdict = sd.verdict;
                    const vColor = verdict==="IMPROVES"?"#22c55e":verdict==="MARGINAL"?"#a3e635":verdict==="NEUTRAL"?"#94a3b8":verdict==="HURTS"?"#ef4444":"#eab308";
                    const fd = sd.full?.discovery||{}, fv = sd.full?.validation||{};
                    const nd = sd.no_financial?.discovery||{}, nv = sd.no_financial?.validation||{};
                    const dedD = sd.no_financial?.edge_delta_disc;
                    const dedV = sd.no_financial?.edge_delta_val;
                    const volChg = sd.no_financial?.firing_volume_change_pct;
                    const deC = dedD==null?"#64748b":dedD>0.5?"#22c55e":dedD<-0.5?"#ef4444":"#94a3b8";
                    const veC = dedV==null?"#64748b":dedV>0.5?"#22c55e":dedV<-0.5?"#ef4444":"#94a3b8";
                    return <tr key={s} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                      <td style={{padding:"4px 8px",color:"#e2e8f0",fontWeight:500}}>{s}</td>
                      <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{fd.edge!=null?`${fd.edge>0?"+":""}${fd.edge}%`:"—"} → {nd.edge!=null?`${nd.edge>0?"+":""}${nd.edge}%`:"—"}</td>
                      <td style={{padding:"4px 8px",color:deC,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{dedD!=null?`${dedD>0?"+":""}${dedD}pp`:"—"}</td>
                      <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{fv.edge!=null?`${fv.edge>0?"+":""}${fv.edge}%`:"—"} → {nv.edge!=null?`${nv.edge>0?"+":""}${nv.edge}%`:"—"}</td>
                      <td style={{padding:"4px 8px",color:veC,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{dedV!=null?`${dedV>0?"+":""}${dedV}pp`:"—"}</td>
                      <td style={{padding:"4px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{volChg!=null?`${volChg}%`:"—"}</td>
                      <td style={{padding:"4px 8px"}}><span style={{color:vColor,fontWeight:700,fontSize:10,letterSpacing:0.3}}>{verdict}</span></td>
                    </tr>;
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v21: CONSISTENT FILTER VALIDATION */}
      {hasResults && results.consistent_filter_test && (
        <Box>
          <Lbl>CONSISTENT Filter Validation (v21) — do v18's high-conviction filters survive edge-delta testing?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Re-tests v18's 4 high-conviction CONSISTENT filters with the same edge-delta framework used for ATR and sector tests. Filter IMPROVES only if edge gains ≥1pp on both discovery and validation folds.
          </div>
          <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
            <thead><tr>
              {["Setup","Hour","Filter","Disc full→filt","Δdisc","Val full→filt","Δval","Vol Δ","Verdict"].map(col=>
                <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
            </tr></thead>
            <tbody>
              {(results.consistent_filter_test.filters||[]).map((f,fi)=>{
                const verdict = f.verdict;
                const vColor = verdict==="IMPROVES"?"#22c55e":verdict==="MARGINAL"?"#a3e635":verdict==="NEUTRAL"?"#94a3b8":verdict==="HURTS"?"#ef4444":"#eab308";
                const fd = f.full?.discovery||{}, fv = f.full?.validation||{};
                const td = f.filtered?.discovery||{}, tv = f.filtered?.validation||{};
                const ded = f.filtered?.edge_delta_disc;
                const dev = f.filtered?.edge_delta_val;
                const volChg = f.filtered?.firing_volume_change_pct;
                const deC = ded==null?"#64748b":ded>0.5?"#22c55e":ded<-0.5?"#ef4444":"#94a3b8";
                const veC = dev==null?"#64748b":dev>0.5?"#22c55e":dev<-0.5?"#ef4444":"#94a3b8";
                return <tr key={fi} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                  <td style={{padding:"4px 8px",color:"#e2e8f0",fontWeight:500}}>{f.setup}</td>
                  <td style={{padding:"4px 8px",color:"#94a3b8"}}>{f.hour}:00</td>
                  <td style={{padding:"4px 8px",color:"#f97316"}}>{f.label}</td>
                  <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{fd.edge!=null?`${fd.edge>0?"+":""}${fd.edge}%`:"—"} → {td.edge!=null?`${td.edge>0?"+":""}${td.edge}%`:"—"}</td>
                  <td style={{padding:"4px 8px",color:deC,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{ded!=null?`${ded>0?"+":""}${ded}pp`:"—"}</td>
                  <td style={{padding:"4px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{fv.edge!=null?`${fv.edge>0?"+":""}${fv.edge}%`:"—"} → {tv.edge!=null?`${tv.edge>0?"+":""}${tv.edge}%`:"—"}</td>
                  <td style={{padding:"4px 8px",color:veC,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{dev!=null?`${dev>0?"+":""}${dev}pp`:"—"}</td>
                  <td style={{padding:"4px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{volChg!=null?`${volChg}%`:"—"}</td>
                  <td style={{padding:"4px 8px"}}><span style={{color:vColor,fontWeight:700,fontSize:10,letterSpacing:0.3}}>{verdict}</span></td>
                </tr>;
              })}
            </tbody>
          </table>
        </Box>
      )}

      {/* v20: ATR UNIVERSE FILTER TEST */}
      {hasResults && results.atr_filter_test && (
        <Box>
          <Lbl>ATR Universe Filter Test (v20) — does excluding low-ATR stocks improve ROBUST setup edges?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each ROBUST setup, compares edge on 3 universes: <b>full</b> (baseline), <b>mid_hi</b> (excludes low-ATR tertile — what we'd implement), <b>hi</b> (strictest, top tertile only). Base rate is recomputed on each filtered universe — apples-to-apples. <b>Verdict</b> on mid_hi: IMPROVES (Δedge ≥+1pp both folds), MARGINAL (≥+0.5pp), NEUTRAL (|Δ|&lt;0.5), HURTS (drops ≥-0.5), INCONSISTENT. Firing volume change = how many setup firings remain under this filter.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.atr_filter_test.by_hour?.[String(h)] || {};
            const setups = Object.keys(hd.setups || {});
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>
                {h}:00 ET · {setups.length} ROBUST setup{setups.length===1?"":"s"} · ATR tertile boundaries: {hd.atr_tertile_boundaries?.map(v=>v!=null?v.toFixed(2):"—").join(" / ")}
              </div>
              <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
                <thead><tr>
                  {["Setup","Universe","Disc edge","Val edge","Δedge disc","Δedge val","n firings","vol Δ","Verdict"].map(col=>
                    <th key={col} style={{padding:"4px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                </tr></thead>
                <tbody>
                  {setups.map(s=>{
                    const sd = hd.setups[s];
                    const verdict = sd.verdict;
                    const vColor = verdict==="IMPROVES"?"#22c55e":verdict==="MARGINAL"?"#a3e635":verdict==="NEUTRAL"?"#94a3b8":verdict==="HURTS"?"#ef4444":"#eab308";
                    return ["full","mid_hi","hi"].map(un=>{
                      const univ = sd[un] || {};
                      const isFull = un === "full";
                      const isMidHi = un === "mid_hi";
                      const dEdge = univ.edge_delta_disc;
                      const vEdge = univ.edge_delta_val;
                      const volChg = univ.firing_volume_change_pct;
                      const deColor = dEdge==null?"#64748b":dEdge>0.5?"#22c55e":dEdge<-0.5?"#ef4444":"#94a3b8";
                      const veColor = vEdge==null?"#64748b":vEdge>0.5?"#22c55e":vEdge<-0.5?"#ef4444":"#94a3b8";
                      return <tr key={s+un} style={{borderTop:un==="full"?"1px solid rgba(255,255,255,0.06)":"1px solid rgba(255,255,255,0.02)"}}>
                        <td style={{padding:"3px 8px",color:"#e2e8f0",fontWeight:isFull?600:400}}>{isFull?s:""}</td>
                        <td style={{padding:"3px 8px",color:isFull?"#94a3b8":isMidHi?"#06b6d4":"#d946ef",fontWeight:500}}>{un}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{univ.discovery?.edge!=null?`${univ.discovery.edge>0?"+":""}${univ.discovery.edge}%`:"—"}</td>
                        <td style={{padding:"3px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{univ.validation?.edge!=null?`${univ.validation.edge>0?"+":""}${univ.validation.edge}%`:"—"}</td>
                        <td style={{padding:"3px 8px",color:deColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{isFull?"—":(dEdge!=null?`${dEdge>0?"+":""}${dEdge}pp`:"—")}</td>
                        <td style={{padding:"3px 8px",color:veColor,fontVariantNumeric:"tabular-nums",fontWeight:600}}>{isFull?"—":(vEdge!=null?`${vEdge>0?"+":""}${vEdge}pp`:"—")}</td>
                        <td style={{padding:"3px 8px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{univ.all?.n_setup!=null?univ.all.n_setup:"—"}</td>
                        <td style={{padding:"3px 8px",color:volChg<-50?"#ef4444":volChg<-20?"#eab308":"#64748b",fontVariantNumeric:"tabular-nums"}}>{isFull?"—":(volChg!=null?`${volChg}%`:"—")}</td>
                        <td style={{padding:"3px 8px"}}>{isMidHi?<span style={{color:vColor,fontWeight:700,fontSize:10,letterSpacing:0.3}}>{verdict}</span>:""}</td>
                      </tr>;
                    });
                  })}
                </tbody>
              </table>
            </div>;
          })}
        </Box>
      )}

      {/* v19: ROBUST LOSER PROFILES */}
      {hasResults && results.robust_loser_profiles && (
        <Box>
          <Lbl>ROBUST Setup Loser Profiles (v19) — what distinguishes losers from winners within ROBUST setups?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each ROBUST setup (survived fold-swap at ≥80% positive across 9 folds), full loser-vs-winner profile. Categorical dims show what % of losers fall in each value vs what % of winners. <b style={{color:"#f97316"}}>⚑ over-loser</b> = ≥5pp more losers than winners in both discovery AND validation folds. Continuous dims show winner mean vs loser mean with Cohen's d effect size. <b style={{color:"#f97316"}}>⚑ d≥0.2</b> = small-but-real effect in both folds. Everything shown — not just qualifying — so you see the full picture.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.robust_loser_profiles.by_hour?.[String(h)] || {};
            const setups = Object.keys(hd);
            if (setups.length === 0) return null;
            return <div key={h} style={{marginBottom:16}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:6}}>{h}:00 ET · {setups.length} ROBUST setup{setups.length===1?"":"s"}</div>
              {setups.map(s=>{
                const sd = hd[s];
                if (sd.status === "insufficient") {
                  return <div key={s} style={{fontSize:10,color:"#475569",marginBottom:6,paddingLeft:12}}>
                    <span style={{color:"#64748b",fontWeight:500}}>{s}</span> — insufficient data (winners={sd.n_winners}, losers={sd.n_losers})
                  </div>;
                }
                return <div key={s} style={{marginBottom:14,paddingLeft:12,borderLeft:"2px solid rgba(34,197,94,0.2)"}}>
                  <div style={{fontSize:12,color:"#e2e8f0",fontWeight:600,marginBottom:6}}>
                    {s}
                    <span style={{color:"#64748b",fontWeight:400,marginLeft:8,fontSize:10}}>
                      {sd.n_winners} winners · {sd.n_losers} losers · disc {sd.n_winners_disc}W/{sd.n_losers_disc}L · val {sd.n_winners_val}W/{sd.n_losers_val}L
                    </span>
                  </div>

                  {/* Categorical dimensions */}
                  {(sd.categorical||[]).map((cd,ci)=>{
                    const anyFlagged = cd.buckets.some(b=>b.flagged_over_loser||b.flagged_over_winner);
                    return <div key={ci} style={{marginBottom:6}}>
                      <div style={{fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:2}}>
                        {cd.dim}
                        {anyFlagged && <span style={{color:"#f97316",marginLeft:6,fontWeight:700}}>⚑</span>}
                      </div>
                      <table style={{width:"100%",fontSize:10,borderCollapse:"collapse"}}>
                        <thead><tr>
                          {["Value","total n","Disc W%","Disc L%","Disc Δpp","Val W%","Val L%","Val Δpp","Flag"].map(col=>
                            <th key={col} style={{padding:"2px 6px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                        </tr></thead>
                        <tbody>
                          {cd.buckets.map((b,bi)=>{
                            const overLoser = b.flagged_over_loser;
                            const overWinner = b.flagged_over_winner;
                            const flagColor = overLoser?"#f97316":overWinner?"#22c55e":"transparent";
                            const diffColor = (v)=>v>5?"#ef4444":v<-5?"#22c55e":"#94a3b8";
                            return <tr key={bi} style={{borderTop:"1px solid rgba(255,255,255,0.02)"}}>
                              <td style={{padding:"2px 6px",color:overLoser?"#fca5a5":overWinner?"#86efac":"#e2e8f0",fontWeight:overLoser||overWinner?600:400}}>{b.value}</td>
                              <td style={{padding:"2px 6px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{b.total_n}</td>
                              <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.disc_winner_pct}%</td>
                              <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.disc_loser_pct}%</td>
                              <td style={{padding:"2px 6px",color:diffColor(b.disc_diff_pp),fontVariantNumeric:"tabular-nums",fontWeight:600}}>{b.disc_diff_pp>0?"+":""}{b.disc_diff_pp}</td>
                              <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.val_winner_pct}%</td>
                              <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{b.val_loser_pct}%</td>
                              <td style={{padding:"2px 6px",color:diffColor(b.val_diff_pp),fontVariantNumeric:"tabular-nums",fontWeight:600}}>{b.val_diff_pp>0?"+":""}{b.val_diff_pp}</td>
                              <td style={{padding:"2px 6px",color:flagColor,fontWeight:700}}>{overLoser?"⚑over-loser":overWinner?"⚑over-winner":""}</td>
                            </tr>;
                          })}
                        </tbody>
                      </table>
                    </div>;
                  })}

                  {/* Continuous dimensions */}
                  <div style={{marginTop:8,fontSize:10,color:"#94a3b8",fontWeight:600,textTransform:"uppercase",letterSpacing:0.3,marginBottom:2}}>
                    CONTINUOUS (winner vs loser means + Cohen's d)
                  </div>
                  <table style={{width:"100%",fontSize:10,borderCollapse:"collapse"}}>
                    <thead><tr>
                      {["Feature","Disc Win μ±σ","Disc Loser μ±σ","Disc d","Val Win μ±σ","Val Loser μ±σ","Val d","Flag"].map(col=>
                        <th key={col} style={{padding:"2px 6px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                    </tr></thead>
                    <tbody>
                      {(sd.continuous||[]).map((cd,ci)=>{
                        const dColor = (d)=>d===null||d===undefined?"#64748b":Math.abs(d)>=0.2?(d>0?"#fca5a5":"#86efac"):"#94a3b8";
                        return <tr key={ci} style={{borderTop:"1px solid rgba(255,255,255,0.02)"}}>
                          <td style={{padding:"2px 6px",color:"#e2e8f0",fontWeight:500}}>{cd.dim}</td>
                          <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{cd.disc_winner.mean!=null?`${cd.disc_winner.mean}±${cd.disc_winner.std}`:"—"}</td>
                          <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{cd.disc_loser.mean!=null?`${cd.disc_loser.mean}±${cd.disc_loser.std}`:"—"}</td>
                          <td style={{padding:"2px 6px",color:dColor(cd.cohens_d_disc),fontVariantNumeric:"tabular-nums",fontWeight:600}}>{cd.cohens_d_disc!=null?cd.cohens_d_disc:"—"}</td>
                          <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{cd.val_winner.mean!=null?`${cd.val_winner.mean}±${cd.val_winner.std}`:"—"}</td>
                          <td style={{padding:"2px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{cd.val_loser.mean!=null?`${cd.val_loser.mean}±${cd.val_loser.std}`:"—"}</td>
                          <td style={{padding:"2px 6px",color:dColor(cd.cohens_d_val),fontVariantNumeric:"tabular-nums",fontWeight:600}}>{cd.cohens_d_val!=null?cd.cohens_d_val:"—"}</td>
                          <td style={{padding:"2px 6px"}}>{cd.flagged?<span style={{color:"#f97316",fontWeight:700}}>⚑</span>:null}</td>
                        </tr>;
                      })}
                    </tbody>
                  </table>
                </div>;
              })}
            </div>;
          })}
        </Box>
      )}

      {/* v18: LOSER-FILTER DISCOVERY/VALIDATION */}
      {hasResults && results.loser_filter && (
        <Box>
          <Lbl>Loser Filters (v18) — what features distinguish losing firings from winners?</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            For each ROBUST/CONSISTENT survivor, tests 5 feature dimensions (sector, RSI bucket, ATR% tertile, rel_volume tertile, day-of-week).
            Discovery fold: {results.loser_filter.disc_folds?.join(", ")} ({results.loser_filter.disc_days} days).
            Validation fold: {results.loser_filter.val_folds?.join(", ")} ({results.loser_filter.val_days} days).
            A filter qualifies only if (1) the bucket's hit rate is ≥10% relatively lower than setup's overall in BOTH discovery AND validation folds, and (2) the bucket holds ≥10% of firings with n≥15 in each fold. Strict methodology to avoid data mining.
          </div>
          {SCAN_HOURS.map(h=>{
            const hd = results.loser_filter.by_hour?.[String(h)];
            if (!hd || !hd.setups) return null;
            const setups = Object.keys(hd.setups);
            if (setups.length === 0) return null;
            const totalQualified = setups.reduce((a,s)=>a+((hd.setups[s].qualified_filters||[]).length),0);
            return <div key={h} style={{marginBottom:14}}>
              <div style={{fontSize:11,color:"#94a3b8",fontWeight:600,marginBottom:4}}>
                {h}:00 ET · {setups.length} setups tested · {totalQualified} qualified filters found
              </div>
              {setups.map(s=>{
                const sd = hd.setups[s];
                if (sd.status === "insufficient_data") {
                  return <div key={s} style={{fontSize:10,color:"#475569",marginBottom:6,paddingLeft:12}}>
                    <span style={{color:"#64748b",fontWeight:500}}>{s}</span> — insufficient data (disc n={sd.n_discovery}, val n={sd.n_validation})
                  </div>;
                }
                const qf = sd.qualified_filters || [];
                return <div key={s} style={{marginBottom:10,paddingLeft:12,borderLeft:"2px solid rgba(255,255,255,0.06)"}}>
                  <div style={{fontSize:11,color:"#e2e8f0",fontWeight:600,marginBottom:4}}>
                    {s} <span style={{color:"#64748b",fontWeight:400,marginLeft:8}}>
                      disc hit {sd.discovery_hit_rate}% (n={sd.n_discovery}) · val hit {sd.validation_hit_rate}% (n={sd.n_validation})
                    </span>
                  </div>
                  {qf.length === 0 ? (
                    <div style={{fontSize:10,color:"#475569"}}>
                      No filters qualified on {sd.n_tests_run} dimension-bucket tests. Setup is insensitive to these features (or filter effect isn't strong enough to validate).
                    </div>
                  ) : (
                    <div>
                      <table style={{width:"100%",fontSize:10,borderCollapse:"collapse",marginTop:4}}>
                        <thead><tr>
                          {["Filter","Disc hit%","Disc relative","Val hit%","Val relative","n (disc/val)"].map(col=>
                            <th key={col} style={{padding:"3px 6px",textAlign:"left",color:"#64748b",fontSize:9,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{col}</th>)}
                        </tr></thead>
                        <tbody>
                          {qf.map((f,fi)=>
                            <tr key={fi} style={{borderTop:"1px solid rgba(255,255,255,0.03)"}}>
                              <td style={{padding:"3px 6px",color:"#fca5a5",fontWeight:600}}>
                                {f.dim}={f.value}
                              </td>
                              <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{f.discovery_hit_rate}%</td>
                              <td style={{padding:"3px 6px",color:"#f97316",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{f.discovery_relative}×</td>
                              <td style={{padding:"3px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{f.validation_hit_rate}%</td>
                              <td style={{padding:"3px 6px",color:"#ef4444",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{f.validation_relative}×</td>
                              <td style={{padding:"3px 6px",color:"#64748b",fontVariantNumeric:"tabular-nums"}}>{f.n_discovery}/{f.n_validation}</td>
                            </tr>)}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>;
              })}
            </div>;
          })}
        </Box>
      )}

      {/* v10: Live-vs-backtest performance tracking */}
      {live && Object.keys(live).length > 0 && (
        <Box>
          <Lbl>Live Performance vs Backtest — the honest feedback loop</Lbl>
          <div style={{fontSize:10,color:"#475569",marginBottom:10,lineHeight:1.5}}>
            Every live setup firing is recorded. Outcomes are checked at 16:12 ET (did price hit +1% before close?). If live hit rate tracks backtest hit rate within ~5 points, the setup is real. If live collapses below backtest, we found a subtle overfit. This is the ultimate reality check.
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",fontSize:11,borderCollapse:"collapse"}}>
              <thead><tr>
                {["Setup","Live n","Live Hit%","Backtest Hit%","Drift","Mean PnL","Per-hour live hit rates"].map(h=>
                  <th key={h} style={{padding:"5px 8px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.3,textTransform:"uppercase"}}>{h}</th>)}
              </tr></thead>
              <tbody>
                {Object.entries(live).map(([name,stats])=>{
                  const drift = stats.drift;
                  const driftColor = drift == null ? "#64748b" : drift > -3 ? "#22c55e" : drift > -7 ? "#eab308" : "#ef4444";
                  return <tr key={name} style={{borderTop:"1px solid rgba(255,255,255,0.04)"}}>
                    <td style={{padding:"6px 8px",color:"#e2e8f0",fontWeight:600}}>{name}</td>
                    <td style={{padding:"6px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{stats.n_outcomes}</td>
                    <td style={{padding:"6px 8px",color:"#e2e8f0",fontVariantNumeric:"tabular-nums",fontWeight:600}}>{stats.live_hit_rate!=null?`${stats.live_hit_rate}%`:"—"}</td>
                    <td style={{padding:"6px 8px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{stats.backtest_hit_rate!=null?`${stats.backtest_hit_rate}%`:"—"}</td>
                    <td style={{padding:"6px 8px",color:driftColor,fontVariantNumeric:"tabular-nums",fontWeight:700}}>{drift!=null?`${drift>0?"+":""}${drift}%`:"—"}</td>
                    <td style={{padding:"6px 8px",color:stats.mean_pnl>0?"#22c55e":"#ef4444",fontVariantNumeric:"tabular-nums"}}>{stats.mean_pnl!=null?`${stats.mean_pnl>0?"+":""}${stats.mean_pnl}%`:"—"}</td>
                    <td style={{padding:"6px 8px",fontSize:10,color:"#64748b",fontVariantNumeric:"tabular-nums"}}>
                      {Object.entries(stats.by_hour||{}).sort().map(([h,d])=>
                        <span key={h} style={{marginRight:8}}>{h}:00 <span style={{color:"#e2e8f0"}}>{d.hit_rate!=null?`${d.hit_rate}%`:"—"}</span> <span style={{color:"#475569"}}>(n{d.n})</span></span>
                      )}
                    </td>
                  </tr>;
                })}
              </tbody>
            </table>
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:10,lineHeight:1.5}}>
            <b style={{color:"#94a3b8"}}>Drift color:</b> Green = live within 3% of backtest (signal holding). Yellow = 3-7% drop. Red = &gt;7% drop (likely overfit discovered). Need n≥30 firings per setup for statistical reliability.
          </div>
        </Box>
      )}
    </div>);
}

// ─── OUTCOMES ────────────────────────────────────────────────────
function OutcomesTab() {
  const [d,setD]=useState(null);
  useEffect(()=>{fetch('/api/outcomes/summary').then(r=>r.json()).then(setD).catch(()=>{});},[]);
  if(!d) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  if(d.totalDays===0) return <Box><div style={{color:"#475569",fontSize:12,padding:20,textAlign:"center"}}>No outcomes yet. Recorded at 16:12 ET each trading day.</div></Box>;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Top-10 Win Rate & P&L — {d.totalDays} days</Lbl>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
            <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
              <th style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10}}>DATE</th>
              {SCAN_HOURS.map(h=><th key={h} style={{padding:"6px",textAlign:"center",color:"#64748b",fontSize:10}}>{h}:00</th>)}
            </tr></thead>
            <tbody>{d.recent.map((day,i)=>(
              <tr key={i} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{day.date}</td>
                {SCAN_HOURS.map(h=>{
                  const s=day.hours[String(h)];
                  if(!s) return <td key={h} style={{padding:"5px 6px",textAlign:"center",color:"#334155"}}>—</td>;
                  const wr=s.top10wins*10;
                  const pnl=s.top10pnl||0;
                  const bwr=s.baseWR||0;
                  const c=wr>bwr+10?"#22c55e":wr>bwr?"#eab308":"#ef4444";
                  const pc=pnl>0?"#22c55e":"#ef4444";
                  return <td key={h} style={{padding:"5px 6px",textAlign:"center",fontVariantNumeric:"tabular-nums"}}>
                    <span style={{color:c,fontWeight:600}}>{wr}%</span>
                    <span style={{fontSize:10,color:"#475569",marginLeft:3}}>({s.top10wins}/10)</span>
                    <div style={{fontSize:10,color:pc,fontWeight:500}}>{pnl>0?"+":""}{pnl}%</div>
                    <div style={{fontSize:9,color:"#334155"}}>base {bwr}%</div>
                  </td>;
                })}
              </tr>))}</tbody>
          </table>
        </div>
      </Box>
    </div>);
}

// ─── STATUS ──────────────────────────────────────────────────────
function StatusTab({health}) {
  return <Box><Lbl>Server</Lbl>
    {health?<div style={{fontSize:12,lineHeight:2,color:"#94a3b8"}}>
      {[
        {l:"Server",v:"Online",ok:true},
        {l:"Alpaca",v:health.hasCredentials?"OK":"NOT SET",ok:health.hasCredentials},
        {l:"Market",v:health.marketOpen?"Open":"Closed",ok:health.marketOpen},
        {l:"Models",v:health.modelsLoaded?.length>0?health.modelsLoaded.join(", "):"None",ok:health.modelsLoaded?.length>0},
        {l:"Strategy",v:health.tp_mult!=null?`TP ${health.tp_mult}×ATR / SL ${health.sl_mult}×ATR`:"Not trained",ok:health.tp_mult!=null},
        {l:"Outcome days",v:String(health.outcomeDays||0),ok:(health.outcomeDays||0)>0},
        {l:"Cached scans",v:health.lastScanHours?.length>0?health.lastScanHours.join(", "):"None",ok:health.hasLastScan},
      ].map((c,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{width:14,height:14,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
            background:c.ok?"rgba(34,197,94,0.15)":"rgba(239,68,68,0.15)",color:c.ok?"#22c55e":"#ef4444",fontSize:10,fontWeight:900}}>{c.ok?"✓":"✗"}</span>
          <span style={{minWidth:140}}>{c.l}</span>
          <span style={{color:c.ok?"#e2e8f0":"#ef4444",fontWeight:500}}>{c.v}</span>
        </div>))}
    </div>:<div style={{color:"#ef4444"}}>Cannot reach server</div>}
  </Box>;
}

// ─── MAIN ────────────────────────────────────────────────────────
export default function R2KScanner() {
  const [scanHour,setScanHour]=useState(11);
  const [tab,setTab]=useState("scanner");
  const [data,setData]=useState([]);
  const [source,setSource]=useState("loading");
  const [loading,setLoading]=useState(true);
  const [lastUpdate,setLastUpdate]=useState(null);
  const [elapsed,setElapsed]=useState(null);
  const [modelWR10,setModelWR10]=useState(null);
  const [modelPnL10,setModelPnL10]=useState(null);
  const [scanInfo,setScanInfo]=useState(null);
  const [health,setHealth]=useState(null);
  const [error,setError]=useState(null);
  const [message,setMessage]=useState(null);

  useEffect(()=>{fetch('/api/health').then(r=>r.json()).then(setHealth).catch(()=>{});},[]);

  const fetchScan=useCallback(async(hour,force=false)=>{
    setLoading(true);setError(null);setMessage(null);
    try{
      const url=force?`/api/scan/${hour}/refresh`:`/api/scan/${hour}`;
      const r=await fetch(url,force?{method:'POST'}:{});
      if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error||`HTTP ${r.status}`);}
      const d=await r.json();
      setData(d.data||[]);setSource(d.source||"offline");setLastUpdate(d.timestamp);
      setElapsed(d.elapsed||null);setModelWR10(d.modelWR10||null);setModelPnL10(d.modelPnL10||null);
      setScanInfo({
        tp_mult: d.tp_mult, sl_mult: d.sl_mult,
        avgTpPct: d.avgTpPct, avgSlPct: d.avgSlPct, avgAtrPct: d.avgAtrPct,
        threshold: d.threshold, nTradable: d.nTradable,
        activeSetups: d.activeSetups, nStocksWithSetup: d.nStocksWithSetup,
        setupFiringCounts: d.setupFiringCounts,
        r2kBreadthFrac: d.r2kBreadthFrac, r2kBreadthLabel: d.r2kBreadthLabel,
        r2kGreenStocks: d.r2kGreenStocks, r2kTotalStocks: d.r2kTotalStocks,
      });
      setMessage(d.message||null);
    }catch(err){setError(err.message);setSource("error");setData([]);}
    finally{setLoading(false);}
  },[]);

  useEffect(()=>{fetchScan(scanHour);},[scanHour,fetchScan]);
  useEffect(()=>{if(source!=="live")return;const iv=setInterval(()=>fetchScan(scanHour),5*60*1000);return()=>clearInterval(iv);},[source,scanHour,fetchScan]);

  const downloadDiag=useCallback(async()=>{
    try{const r=await fetch('/api/diagnostic');const b=await r.blob();
      const fn=r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1]||`diag.json`;
      const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=fn;document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(u);
    }catch(e){alert(e.message);}
  },[]);

  const tabs=[{id:"scanner",l:"Scanner"},{id:"training",l:"Training",c:"#8b5cf6"},{id:"setups",l:"Setups",c:"#a855f7"},{id:"patterns",l:"Patterns",c:"#f59e0b"},{id:"thresholds",l:"Thresholds",c:"#06b6d4"},{id:"outcomes",l:"Outcomes",c:"#22c55e"},{id:"status",l:"Status"}];

  return (
    <div style={{fontFamily:F,background:"#0c0f14",color:"#e2e8f0",minHeight:"100vh"}}>
      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"12px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontSize:15,fontWeight:800,letterSpacing:1}}>RUSSELL 2000 SCANNER</div>
          <SourceBadge source={loading?"loading":source} trained={health?.trained}/>
        </div>
        <div style={{display:"flex",gap:4,fontSize:11,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{color:"#eab308",fontWeight:600}}>
            {health ? (
              health.tp_mult != null
                ? `TP ${health.tp_mult}×ATR / SL ${health.sl_mult}×ATR / Close 15:55 (notional BE ${health.notional_breakeven}%)`
                : `Not trained yet`
            ) : "Loading..."}
          </span>
          {lastUpdate&&<><span style={{color:"#334155",margin:"0 4px"}}>|</span><span style={{color:"#94a3b8"}}>{new Date(lastUpdate).toLocaleString()}</span></>}
        </div>
      </div>

      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"8px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5,marginRight:4}}>Scan</span>
          {SCAN_HOURS.map(h=><Btn key={h} active={h===scanHour} onClick={()=>setScanHour(h)}>{h}:00</Btn>)}
          <Btn onClick={()=>fetchScan(scanHour,true)} disabled={!health?.marketOpen||!health?.trained} style={{marginLeft:4}}>↻ Refresh</Btn>
          <Btn onClick={downloadDiag} color="#f97316" style={{marginLeft:4}}>⬇ Diagnostic</Btn>
        </div>
        <div style={{display:"flex",gap:2}}>
          {tabs.map(t=><Btn key={t.id} active={t.id===tab} onClick={()=>setTab(t.id)} color={t.c||"#3b82f6"}>{t.l}</Btn>)}
        </div>
      </div>

      {error&&<div style={{margin:"12px 20px 0",padding:"8px 12px",borderRadius:6,background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>{error}</div>}

      <div style={{padding:"16px 20px"}}>
        {tab==="scanner"&&<ErrorBoundary><ScannerTab data={data} scanHour={scanHour} source={source} elapsed={elapsed} message={message} modelWR10={modelWR10} modelPnL10={modelPnL10} health={health} scanInfo={scanInfo}/></ErrorBoundary>}
        {tab==="training"&&<ErrorBoundary><TrainingTab/></ErrorBoundary>}
        {tab==="patterns"&&<ErrorBoundary><PatternsTab health={health}/></ErrorBoundary>}
        {tab==="thresholds"&&<ErrorBoundary><ThresholdsTab health={health}/></ErrorBoundary>}
        {tab==="setups"&&<ErrorBoundary><SetupsTab/></ErrorBoundary>}
        {tab==="outcomes"&&<ErrorBoundary><OutcomesTab/></ErrorBoundary>}
        {tab==="status"&&<ErrorBoundary><StatusTab health={health}/></ErrorBoundary>}
      </div>
    </div>);
}
