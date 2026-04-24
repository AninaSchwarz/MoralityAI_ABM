"""
Simplified Networked Hegselmann–Krause (HK) model with adaptive epsilon (ε)

Kept mechanisms ONLY:
1) incident + threat
2) norm/context violation
3) adaptive ε update
4) HK averaging on a social network
5) polarization metric

Outputs:
- Saves key figures to a folder named "figure" NEXT TO THIS SCRIPT
- Saves key simulation timeseries data to that folder as .npz and .csv

Dependencies: numpy, matplotlib
Run:
  python hk_simplified.py
Optional:
  python hk_simplified.py --reps 15 --N 400 --T 200
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import numpy as np

# Non-interactive backend so saving works in VS Code / headless runs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Parameters
# -----------------------------

@dataclass(frozen=True)
class Params:
    N: int = 300
    T: int = 140
    seed: int = 7

    # network
    p_edge: float = 0.04

    # opinions
    op_init_sigma: float = 0.35
    op_clip: float = 1.0

    # HK + adaptive epsilon
    eps0: float = 0.35
    eps_min: float = 0.05
    eps_max: float = 0.85
    eps_shrink_rate: float = 0.10
    eps_expand_rate: float = 0.06

    # policy → incidents
    incident_base: float = 0.28
    incident_k: float = 1.55
    safeguard_effect_cost: float = 0.22

    # violation model (reduced form)
    violation_scale: float = 0.75
    safeguard_violation_reduction: float = 0.70

    # threat dynamics
    threat_decay: float = 0.85
    threat_from_incident: float = 0.40
    threat_noise: float = 0.03

    # opinion response to threat
    threat_polarize_strength: float = 0.035
    noise_sigma: float = 0.007


# -----------------------------
# Utilities
# -----------------------------

def ensure_outdir() -> Path:
    """Create ./figure relative to this script file."""
    here = Path(__file__).resolve().parent
    outdir = here / "figure"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def make_network(N: int, p_edge: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Undirected Erdos–Renyi adjacency list; ensures every node has >=1 neighbor."""
    while True:
        A = rng.random((N, N)) < p_edge
        A = np.triu(A, 1)
        A = A + A.T
        np.fill_diagonal(A, 0)
        if np.all(A.sum(axis=1) > 0):
            break
    return [np.where(A[i] > 0)[0] for i in range(N)]


def polarization_metric(ops: np.ndarray) -> float:
    """Conceptual polarization: spread of opinions (std dev)."""
    return float(np.std(ops))


def save_timeseries_csv(path: Path, t: np.ndarray, series: dict[str, np.ndarray]) -> None:
    keys = list(series.keys())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + keys)
        for i in range(len(t)):
            w.writerow([int(t[i])] + [float(series[k][i]) for k in keys])


# -----------------------------
# Core dynamics
# -----------------------------

def incident_probability(p: Params, S: float, I: float, G: float) -> float:
    """
    Incident probability decreases with intensity (S+I).
    Safeguards reduce effectiveness (-> slightly higher incidents).
    """
    x = np.clip(S + I, 0.0, 2.0) / 2.0
    k_eff = p.incident_k * (1.0 - p.safeguard_effect_cost * G)
    return float(p.incident_base * np.exp(-k_eff * x))


def violation_level(p: Params, S: float, I: float, G: float) -> float:
    """Violation increases with (S+I), reduced by safeguards."""
    x = np.clip(S + I, 0.0, 2.0) / 2.0
    v = p.violation_scale * x * (1.0 - p.safeguard_violation_reduction * G)
    return float(np.clip(v, 0.0, 1.0))


def hk_update(ops: np.ndarray, eps: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    """
    Networked HK averaging:
    each agent averages neighbor opinions within |op_j - op_i| <= eps_i.
    """
    N = ops.shape[0]
    new_ops = ops.copy()
    for i in range(N):
        ni = neighbors[i]
        within = ni[np.abs(ops[ni] - ops[i]) <= eps[i]]
        if within.size:
            new_ops[i] = float(np.mean(ops[within]))
    return new_ops


def run_simulation(p: Params, *, S: float, I: float, G: float, neighbors: list[np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """
    Run simplified HK model for T steps under fixed policy (S, I, G).

    NOTE: incidents are iid Bernoulli with p=incident_probability(...). In this simplified
    model, incidents do NOT depend on opinions, so Figure 1 should be computed directly
    from incident_probability (no need to run HK many times).
    """
    rng = np.random.default_rng(p.seed)
    if neighbors is None:
        neighbors = make_network(p.N, p.p_edge, rng)

    ops = rng.normal(0.0, p.op_init_sigma, size=p.N)
    ops = np.clip(ops, -p.op_clip, p.op_clip)
    eps = np.full(p.N, p.eps0, dtype=float)
    threat = np.zeros(p.N, dtype=float)

    inc_p = incident_probability(p, S, I, G)
    viol = violation_level(p, S, I, G)

    incident_ts = np.zeros(p.T, dtype=float)
    threat_ts = np.zeros(p.T, dtype=float)
    eps_ts = np.zeros(p.T, dtype=float)
    pol_ts = np.zeros(p.T, dtype=float)
    opinions_ts = np.zeros((p.T, p.N), dtype=float)

    for t in range(p.T):
        incidents = rng.random(p.N) < inc_p
        incident_ts[t] = float(np.mean(incidents))

        threat = p.threat_decay * threat + p.threat_from_incident * incidents.astype(float)
        threat += rng.normal(0.0, p.threat_noise, size=p.N)
        threat = np.clip(threat, 0.0, 1.5)
        threat_ts[t] = float(np.mean(threat))

        safety = 1.0 - np.clip(threat, 0.0, 1.0)
        eps = eps + p.eps_expand_rate * (safety - 0.5) - p.eps_shrink_rate * viol
        eps = np.clip(eps, p.eps_min, p.eps_max)
        eps_ts[t] = float(np.mean(eps))

        ops = hk_update(ops, eps, neighbors)

        ops = ops + p.threat_polarize_strength * (threat - 0.4) * np.sign(ops + 1e-9)
        ops = ops + rng.normal(0.0, p.noise_sigma, size=p.N)
        ops = np.clip(ops, -p.op_clip, p.op_clip)

        pol_ts[t] = polarization_metric(ops)
        opinions_ts[t] = ops

    return {
        "incident": incident_ts,
        "threat_mean": threat_ts,
        "eps_mean": eps_ts,
        "polarization": pol_ts,
        "opinions": opinions_ts,
        "incident_prob": np.array([inc_p], dtype=float),
        "violation": np.array([viol], dtype=float),
    }


# -----------------------------
# Plotting + saving
# -----------------------------

def plot_and_save_incident_vs_intensity(p: Params, outdir: Path) -> None:
    """
    Figure 1: incident rate vs (S+I), with and without safeguards.

    IMPORTANT: This is computed directly from incident_probability(...) so it runs fast.
    (If we ran the full HK simulation for each point and replicate, it would be extremely slow.)
    """
    xs = np.linspace(0, 1, 101)
    y_no = np.zeros_like(xs)
    y_sg = np.zeros_like(xs)

    for k, x in enumerate(xs):
        S = x / 2
        I = x / 2
        y_no[k] = incident_probability(p, S, I, G=0.0)
        y_sg[k] = incident_probability(p, S, I, G=1.0)

    fig_path = outdir / "fig1_incident_vs_intensity.png"
    plt.figure(figsize=(9, 4.2))
    plt.plot(xs, y_no, label="Higher effectiveness (fewer incidents)")
    plt.plot(xs, y_sg, label="With safeguards (slightly higher incidents)")
    plt.xlabel("Surveillance + inference intensity (S + I)")
    plt.ylabel("Incident rate (conceptual)")
    plt.title("Privacy–Security: safeguards can soften harm, but reduce effectiveness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    np.savez(outdir / "fig1_incident_vs_intensity.npz", x=xs, incident_no_safeguards=y_no, incident_with_safeguards=y_sg)
    with (outdir / "fig1_incident_vs_intensity.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_S_plus_I", "incident_no_safeguards", "incident_with_safeguards"])
        for i in range(len(xs)):
            w.writerow([float(xs[i]), float(y_no[i]), float(y_sg[i])])

    print(f"Saved: {fig_path}")


def plot_and_save_timeseries_bundle(p: Params, outdir: Path, *, S: float, I: float, reps: int = 8) -> None:
    """
    Figure 2: polarization over time under high S/I comparing G=0 vs G=1.
    Also saves eps, threat, incident over time and final opinion histogram.

    NOTE: reps default is modest for speed. Increase with --reps if needed.
    """
    pol_no, pol_sg = [], []
    eps_no, eps_sg = [], []
    thr_no, thr_sg = [], []
    inc_no, inc_sg = [], []
    final_ops_no, final_ops_sg = [], []

    # Fix one network across reps for speed and comparability
    rng_net = np.random.default_rng(p.seed)
    neighbors = make_network(p.N, p.p_edge, rng_net)

    for r in range(reps):
        pr = Params(**{**p.__dict__, "seed": p.seed + r})
        out0 = run_simulation(pr, S=S, I=I, G=0.0, neighbors=neighbors)
        out1 = run_simulation(pr, S=S, I=I, G=1.0, neighbors=neighbors)

        pol_no.append(out0["polarization"]); pol_sg.append(out1["polarization"])
        eps_no.append(out0["eps_mean"]);     eps_sg.append(out1["eps_mean"])
        thr_no.append(out0["threat_mean"]);  thr_sg.append(out1["threat_mean"])
        inc_no.append(out0["incident"]);     inc_sg.append(out1["incident"])
        final_ops_no.append(out0["opinions"][-1]); final_ops_sg.append(out1["opinions"][-1])

    pol_no = np.mean(np.vstack(pol_no), axis=0)
    pol_sg = np.mean(np.vstack(pol_sg), axis=0)
    eps_no = np.mean(np.vstack(eps_no), axis=0)
    eps_sg = np.mean(np.vstack(eps_sg), axis=0)
    thr_no = np.mean(np.vstack(thr_no), axis=0)
    thr_sg = np.mean(np.vstack(thr_sg), axis=0)
    inc_no = np.mean(np.vstack(inc_no), axis=0)
    inc_sg = np.mean(np.vstack(inc_sg), axis=0)
    final_ops_no = np.mean(np.vstack(final_ops_no), axis=0)
    final_ops_sg = np.mean(np.vstack(final_ops_sg), axis=0)

    t = np.arange(p.T)

    def save_lineplot(y1, y2, title, ylabel, filename, label1, label2):
        fp = outdir / filename
        plt.figure(figsize=(9, 4.2))
        plt.plot(t, y1, label=label1)
        plt.plot(t, y2, label=label2)
        plt.xlabel("Time (t)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fp, dpi=200)
        plt.close()
        print(f"Saved: {fp}")

    save_lineplot(
        pol_no, pol_sg,
        "Bounded confidence mechanism: norm violations can amplify polarization",
        "Polarization index (conceptual)",
        "fig2_polarization_over_time.png",
        "High surveillance, norm violation → ε shrinks",
        "Safeguards + context limits → ε stays larger",
    )
    save_lineplot(
        eps_no, eps_sg,
        "Adaptive confidence bound (ε) over time",
        "Mean ε",
        "fig2_eps_over_time.png",
        "ε mean (no safeguards)",
        "ε mean (with safeguards)",
    )
    save_lineplot(
        thr_no, thr_sg,
        "Threat over time",
        "Mean threat",
        "fig2_threat_over_time.png",
        "Threat mean (no safeguards)",
        "Threat mean (with safeguards)",
    )
    save_lineplot(
        inc_no, inc_sg,
        "Incident rate over time",
        "Incident rate",
        "fig2_incident_over_time.png",
        "Incident rate (no safeguards)",
        "Incident rate (with safeguards)",
    )

    fp_hist = outdir / "fig2_final_opinion_hist.png"
    plt.figure(figsize=(9, 4.2))
    plt.hist(final_ops_no, bins=30, alpha=0.6, label="Final opinions (no safeguards)")
    plt.hist(final_ops_sg, bins=30, alpha=0.6, label="Final opinions (with safeguards)")
    plt.xlabel("Opinion")
    plt.ylabel("Count")
    plt.title("Final opinion distribution")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fp_hist, dpi=200)
    plt.close()
    print(f"Saved: {fp_hist}")

    bundle = {
        "t": t,
        "polarization_no_safeguards": pol_no,
        "polarization_with_safeguards": pol_sg,
        "eps_mean_no_safeguards": eps_no,
        "eps_mean_with_safeguards": eps_sg,
        "threat_mean_no_safeguards": thr_no,
        "threat_mean_with_safeguards": thr_sg,
        "incident_no_safeguards": inc_no,
        "incident_with_safeguards": inc_sg,
        "final_ops_no_safeguards": final_ops_no,
        "final_ops_with_safeguards": final_ops_sg,
        "S": np.array([S], dtype=float),
        "I": np.array([I], dtype=float),
        "reps": np.array([reps], dtype=int),
        "N": np.array([p.N], dtype=int),
        "T": np.array([p.T], dtype=int),
    }
    np.savez(outdir / "fig2_timeseries_avg.npz", **bundle)
    save_timeseries_csv(
        outdir / "fig2_timeseries_avg.csv",
        t,
        {
            "polarization_no_safeguards": pol_no,
            "polarization_with_safeguards": pol_sg,
            "eps_mean_no_safeguards": eps_no,
            "eps_mean_with_safeguards": eps_sg,
            "threat_mean_no_safeguards": thr_no,
            "threat_mean_with_safeguards": thr_sg,
            "incident_no_safeguards": inc_no,
            "incident_with_safeguards": inc_sg,
        },
    )
    print(f"Saved: {outdir / 'fig2_timeseries_avg.npz'}")
    print(f"Saved: {outdir / 'fig2_timeseries_avg.csv'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=300, help="Number of agents")
    ap.add_argument("--T", type=int, default=140, help="Time steps")
    ap.add_argument("--reps", type=int, default=8, help="Replicates for time-series (Figure 2)")
    ap.add_argument("--S", type=float, default=0.8, help="Surveillance level for Figure 2")
    ap.add_argument("--I", type=float, default=0.8, help="Inference level for Figure 2")
    ap.add_argument("--seed", type=int, default=7, help="Random seed base")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = ensure_outdir()
    print(f"Output folder: {outdir}")

    p = Params(N=args.N, T=args.T, seed=args.seed)

    # Fast analytic Figure 1 (no long HK sweeps)
    plot_and_save_incident_vs_intensity(p, outdir)

    # HK-based Figure 2 and extra key outputs
    plot_and_save_timeseries_bundle(p, outdir, S=args.S, I=args.I, reps=args.reps)

    print("Done.")


if __name__ == "__main__":
    main()
