"""Fair 1050-total-particle fourth-order PF/SP-PF FDIA comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.run_ablation import AblationConfig, DETECTOR_TYPES, ESTIMATOR_MODES, aggregate_metrics, run_estimator_case


@dataclass(frozen=True)
class EqualBudgetConfig:
    seeds: tuple[int, ...] = (20260907, 20260908, 20260909, 20260910, 20260911)
    attack_magnitudes: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    total_particles: int = 1050
    sppf_particles_per_partition: int = 350
    output_directory: str = "results/equal_budget"


def _base(config: EqualBudgetConfig, particles: int) -> AblationConfig:
    return AblationConfig(seeds=config.seeds, attack_magnitudes=config.attack_magnitudes, particle_count=particles, output_directory=config.output_directory, make_plots=False)


def _summary_markdown(summary: pd.DataFrame, config: EqualBudgetConfig) -> str:
    return "\n".join([
        "# Equal-Total-Particle-Budget Ablation", "",
        "All modes use the same fourth-order trajectory/noise/FDIA timing, five seeds and calibrated detector procedure.", "",
        f"Full PF uses exactly {config.total_particles} particles. Fixed SP-PF uses 3 × {config.sppf_particles_per_partition} = {config.total_particles} particles.",
        "Adaptive-KL starts with that same 1050 total. The existing reconfiguration retains 350 particles per active filter; therefore a merge reduces total particles to 700 and this is logged per sample. It is approximately equal only before repartitioning.", "",
        "## Aggregate results", "",
        summary[["estimator_mode","detector_type","attack_magnitude","detection_rate","mean_detection_delay","mean_false_alarm_rate","mean_state_rmse","mean_runtime_seconds","mean_num_partitions"]].to_csv(index=False),
        "No superiority claim is made unless these metrics support it.", "",
    ])


def _plot(raw: pd.DataFrame, summary: pd.DataFrame, config: EqualBudgetConfig, directory: Path) -> None:
    cache=directory/".matplotlib"; cache.mkdir(parents=True, exist_ok=True); os.environ["MPLCONFIGDIR"]=str(cache.resolve())
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x=np.asarray(config.attack_magnitudes)*100
    def vals(mode: str, detector: str, col: str) -> np.ndarray:
        d=summary[(summary.estimator_mode==mode)&(summary.detector_type==detector)].sort_values("attack_magnitude")
        return d[col].astype(float).to_numpy()
    for filename,col,ylabel,title in (("rmse_vs_magnitude.png","mean_state_rmse","Mean state RMSE","Equal-budget RMSE"),("detection_rate_vs_magnitude.png","detection_rate","Detection rate","Equal-budget detection rate"),("detection_delay_vs_magnitude.png","mean_detection_delay","Mean delay (samples)","Equal-budget detection delay"),("false_alarm_rate.png","mean_false_alarm_rate","False-alarm rate","Equal-budget false alarms")):
        fig,ax=plt.subplots(figsize=(9,4.5))
        for mode in ESTIMATOR_MODES:
            for detector in DETECTOR_TYPES: ax.plot(x,vals(mode,detector,col),marker="o",label=f"{mode}: {detector}")
        ax.set(xlabel="FDIA magnitude (%)",ylabel=ylabel,title=title); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(directory/filename,dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); d=summary.groupby("estimator_mode").mean(numeric_only=True).reindex(ESTIMATOR_MODES); ax.bar(d.index,d.mean_runtime_seconds); ax.set(ylabel="Mean runtime (s)",title="Equal-budget runtime by estimator"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(directory/"runtime_by_estimator.png",dpi=160); plt.close(fig)
    adaptive_rows=raw[(raw.estimator_mode=="adaptive_kl")&(raw.detector_type=="likelihood")]
    rep=adaptive_rows[adaptive_rows.attack_magnitude==0.05]
    rep=(rep.iloc[0] if not rep.empty else adaptive_rows.iloc[0])
    fig,ax=plt.subplots(figsize=(9,4)); ax.step(np.arange(len(rep.particle_count_history))*.02,np.asarray(rep.particle_count_history),where="post"); ax.set(xlabel="Time (s)",ylabel="Total active particles",title="Adaptive-KL particle count (5% FDIA)"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(directory/"adaptive_particle_count_vs_time.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for mode in ESTIMATOR_MODES:
        for detector in DETECTOR_TYPES: ax.plot(x,vals(mode,detector,"detection_rate"),marker="o",label=f"{mode}: {detector}")
    ax.set(xlabel="FDIA magnitude (%)",ylabel="Detection rate",title="Residual vs likelihood at equal particle budget"); ax.grid(alpha=.25); ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(directory/"residual_vs_likelihood.png",dpi=160); plt.close(fig)


def run_equal_budget(config: EqualBudgetConfig = EqualBudgetConfig()) -> dict[str, Any]:
    if config.total_particles != 3*config.sppf_particles_per_partition: raise ValueError("budget must equal three initial SP-PF partitions")
    rows=[]
    for magnitude in config.attack_magnitudes:
        for seed in config.seeds:
            rows += run_estimator_case(_base(config,config.total_particles),seed,magnitude,"full_pf")
            rows += run_estimator_case(_base(config,config.sppf_particles_per_partition),seed,magnitude,"fixed_sppf")
            rows += run_estimator_case(_base(config,config.sppf_particles_per_partition),seed,magnitude,"adaptive_kl")
    raw=pd.DataFrame(rows); summary=aggregate_metrics(raw); directory=Path(config.output_directory); directory.mkdir(parents=True,exist_ok=True)
    csv=raw.copy()
    for col in ("scores","log_likelihoods","partition_counts","particle_count_history"): csv[col]=csv[col].apply(lambda x:json.dumps(np.asarray(x).tolist()))
    csv.attack_channels=csv.attack_channels.apply(lambda x:",".join(map(str,x))); csv.to_csv(directory/"equal_budget_results.csv",index=False)
    compact=[]
    for row in raw.to_dict(orient="records"):
        for col in ("scores","log_likelihoods","partition_counts","particle_count_history"): row.pop(col,None)
        compact.append(row)
    (directory/"equal_budget_results.json").write_text(json.dumps({"configuration":asdict(config),"per_seed_results":compact,"aggregate_results":summary.to_dict(orient="records")},indent=2),encoding="utf-8")
    (directory/"summary.md").write_text(_summary_markdown(summary,config),encoding="utf-8"); _plot(raw,summary,config,directory)
    return {"raw":raw,"summary":summary}


if __name__ == "__main__":
    result=run_equal_budget(); print(result["summary"][["estimator_mode","detector_type","attack_magnitude","detection_rate","mean_state_rmse","mean_runtime_seconds"]].to_string(index=False))
