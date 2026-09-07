"""Five-seed fourth-order fixed-SP-PF replay detector evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json, os
from pathlib import Path
from time import perf_counter
from typing import Any
import numpy as np
import pandas as pd

from attacks.replay import ReplayConfig, inject_replay
from experiments.run_ablation import AblationConfig, _detector_data, _run_sppf, _shared_case, aggregate_metrics, classification_metrics


@dataclass(frozen=True)
class ReplayComparisonConfig:
    seeds: tuple[int,...]=(20260907,20260908,20260909,20260910,20260911)
    intensities: tuple[float,...]=(0.0,0.01,0.02,0.05,0.075,0.10,0.15,0.20,0.30)
    replay_delay_steps: int=20
    output_directory: str="results/replay_comparison"


def _base(c: ReplayComparisonConfig)->AblationConfig:
    return AblationConfig(seeds=c.seeds,particle_count=350,output_directory=c.output_directory,make_plots=False)


def run_replay_case(config: ReplayComparisonConfig, seed: int, intensity: float) -> list[dict[str,Any]]:
    base=_base(config); case=_shared_case(base,seed,0.0)
    # Current replay definition: source is clean noisy history k-delay, all 15
    # fourth-order measurement channels, blend in [0,1]. No interpolation.
    replay=ReplayConfig(base.attack_start_index,base.attack_end_index,config.replay_delay_steps,tuple(range(case["system"].measurement_dimension)),blend=float(intensity))
    attacked,mask,vector=inject_replay(case["clean"],replay); case["attacked"]=attacked; case["attack_mask"]=mask
    started=perf_counter(); est=_run_sppf(case,base,adaptive=False); runtime=perf_counter()-started
    err=case["truth"]-est["estimates"]; delta=np.array([0,4,8]); omega=np.array([1,5,9]); rows=[]
    for detector in ("residual","likelihood"):
        scores,flags,threshold,details=_detector_data(detector,est,case,base); hits=np.flatnonzero(flags&mask); idx=int(hits[0]) if hits.size else None
        metrics=classification_metrics(flags,mask,base.calibration_start_index,base.attack_end_index)
        rows.append({"seed":seed,"model_order":4,"estimator_mode":"fixed_sppf","detector_type":detector,"replay_intensity":float(intensity),"intensity_definition":"blend: 0=current measurement, 1=full delayed replay","replay_delay_steps":config.replay_delay_steps,"replay_delay_seconds":config.replay_delay_steps*base.dt,"attack_start":base.attack_start_index,"attack_end":base.attack_end_index,"attacked_channels":list(range(case["system"].measurement_dimension)),"detected":bool(idx is not None),"detection_index":idx,"detection_delay":None if idx is None else idx-base.attack_start_index,"false_alarm_rate":metrics["false_alarm_rate"],"threshold":threshold,"max_score":float(np.max(scores)),"state_rmse":float(np.sqrt(np.mean(err**2))),"delta_mse":float(np.mean(err[:,delta]**2)),"omega_mse":float(np.mean(err[:,omega]**2)),"runtime_seconds":runtime,"scores":scores,"flags":flags,"clean":case["clean"],"attacked":attacked,"attack_vector":vector,**metrics,**details})
    return rows


def _aggregate(raw:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for (detector,intensity),g in raw.groupby(["detector_type","replay_intensity"],sort=False):
        delays=g.detection_delay.dropna(); row={"detector_type":detector,"replay_intensity":intensity,"seed_count":len(g),"detection_rate":float(g.detected.mean()),"detected_runs":int(g.detected.sum()),"missed_runs":int((~g.detected).sum()),"mean_detection_delay":float(delays.mean()) if len(delays) else None,"std_detection_delay":float(delays.std(ddof=0)) if len(delays) else None}
        for col in ("false_alarm_rate","state_rmse","delta_mse","omega_mse","max_score","threshold","runtime_seconds"):
            row[f"mean_{col}"]=float(g[col].mean()); row[f"std_{col}"]=float(g[col].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(raw:pd.DataFrame,summary:pd.DataFrame,c:ReplayComparisonConfig,d:Path)->None:
    cache=d/".matplotlib";cache.mkdir(parents=True,exist_ok=True);os.environ["MPLCONFIGDIR"]=str(cache.resolve());import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    x=np.asarray(c.intensities)*100
    def v(det,col): return summary[summary.detector_type==det].sort_values("replay_intensity")[col].astype(float).to_numpy()
    for name,col,ylabel,title in (("replay_detection_rate.png","detection_rate","Detection rate","Replay detection rate"),("replay_detection_delay.png","mean_detection_delay","Mean delay (samples)","Replay detection delay"),("replay_state_rmse.png","mean_state_rmse","State RMSE","Replay state RMSE")):
        fig,ax=plt.subplots(figsize=(8,4.5));[ax.plot(x,v(det,col),marker="o",label=det) for det in ("residual","likelihood")];ax.set(xlabel="Replay blend intensity (%)",ylabel=ylabel,title=title);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(d/name,dpi=160);plt.close(fig)
    rep=raw[(raw.detector_type=="residual")&(raw.replay_intensity==0.30)].iloc[0];lik=raw[(raw.detector_type=="likelihood")&(raw.replay_intensity==0.30)].iloc[0];t=np.arange(len(rep.scores))*.02
    for name,row,ylabel in (("residual_score_vs_threshold.png",rep,"J-statistic"),("likelihood_score_vs_threshold.png",lik,"log(Lref / Lk)")):
        fig,ax=plt.subplots(figsize=(9,4));ax.plot(t,row.scores);ax.axhline(row.threshold,color="r",ls="--");ax.axvspan(t[int(row.attack_start)],t[int(row.attack_end)-1],color="r",alpha=.12);ax.set(xlabel="Time (s)",ylabel=ylabel,title=f"Replay {row.detector_type} score, blend=0.30");ax.grid(alpha=.25);fig.tight_layout();fig.savefig(d/name,dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));idx=12;ax.plot(t,rep.clean[:,idx],label="clean");ax.plot(t,rep.attacked[:,idx],label="replayed");ax.axvspan(t[int(rep.attack_start)],t[int(rep.attack_end)-1],color="r",alpha=.12);ax.set(xlabel="Time (s)",ylabel="Pe1 measurement",title="Representative replay measurement (blend=0.30)");ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(d/"representative_replay_measurement.png",dpi=160);plt.close(fig)


def run_replay_comparison(config:ReplayComparisonConfig=ReplayComparisonConfig())->dict[str,Any]:
    rows=[]
    for intensity in config.intensities:
        for seed in config.seeds: rows+=run_replay_case(config,seed,intensity)
    raw=pd.DataFrame(rows);summary=_aggregate(raw);d=Path(config.output_directory);d.mkdir(parents=True,exist_ok=True)
    csv=raw.copy()
    for col in ("scores","flags","clean","attacked","attack_vector"):csv[col]=csv[col].apply(lambda x:json.dumps(np.asarray(x).tolist()))
    csv.attacked_channels=csv.attacked_channels.apply(lambda x:",".join(map(str,x)));csv.to_csv(d/"replay_results.csv",index=False)
    compact=[]
    for row in raw.to_dict(orient="records"):
        for col in ("scores","flags","clean","attacked","attack_vector"):row.pop(col,None)
        compact.append(row)
    (d/"replay_results.json").write_text(json.dumps({"configuration":asdict(config),"per_seed_results":compact,"aggregate_results":summary.to_dict(orient="records")},indent=2),encoding="utf-8")
    (d/"summary.md").write_text("# Replay Detection Evaluation\n\nReplay uses a contiguous delayed window from the clean noisy synthetic measurement history: source k-20, samples 100:155, all 15 fourth-order channels, no interpolation. Intensity is the dimensionless blend in [0,1], not a physical percentage. Results below are from fixed SP-PF with 350 particles per partition and five seeds.\n\n"+summary.to_csv(index=False),encoding="utf-8")
    (d/"paper_comparison.md").write_text("# Paper comparison\n\nThe paper motivates replay detection through likelihood-ratio behaviour under its own system and observation model. This prototype uses delayed synthetic noisy measurements and a normal-reference particle likelihood degradation. Thresholds and raw scores are not comparable. This experiment reports whether that existing detector actually detects replay; no threshold was retuned for replay.\n",encoding="utf-8")
    _plot(raw,summary,config,d);return {"raw":raw,"summary":summary}

if __name__=="__main__":
    r=run_replay_comparison();print(r["summary"].to_string(index=False))
