# NMC-Gradient-Tunneling

Official code for the manuscript

> **Local Spike Timing Enables Gradient-based Learning in Neural Microcircuits**
>
> Xiangnan Zhang<sup>1</sup>, Jingxin Liu<sup>2,3</sup>, Ranqi Lu<sup>4</sup>, Jingyu Liu<sup>2,3</sup>, Qunxi Dong<sup>2,3</sup>, Fuze Tian<sup>5</sup>, Lixian Zhu<sup>2,3</sup>, Bin Hu<sup>2,3</sup>, Björn W. Schuller<sup>6,7,8</sup>
>
> arXiv, 2026. doi: [10.48550/arXiv.2609.08070](https://doi.org/10.48550/arXiv.2609.08070)
>
> <sup>1</sup> School of Future Technologies, Beijing Institute of Technology, Beijing, China  
> <sup>2</sup> Key Laboratory of Brain Health Intelligent Evaluation and Intervention (Beijing Institute of Technology), Ministry of Education, Beijing, China  
> <sup>3</sup> School of Medical Technology, Beijing Institute of Technology, Beijing, China  
> <sup>4</sup> Department of Health Data Science, School of Medical Technology, Capital Medical University, Beijing, China  
> <sup>5</sup> School of Information Science and Engineering, Lanzhou University, Lanzhou, China  
> <sup>6</sup> Chair of Health Informatics (CHI), Technische Universität München, Munich, Germany  
> <sup>7</sup> Munich Data Science Institute (MDSI), Munich, Germany  
> <sup>8</sup> Munich Center for Machine Learning (MCML), Munich, Germany

This repository implements the **NMC Feedback Learning Framework**: the **Gradient Tunneling (GT)** algorithm and the **lead-lag expansion** technique, which train spiking **neural microcircuits (NMCs)** endowed with sparse, trainable feedback connections under a purely **spike-timing-dependent**, gradient-based learning rule. The framework needs neither surrogate gradients nor backpropagation-through-time inside the recurrent population, and is compatible with ANN–SNN hybrid architectures.

## Repository layout

| Path | Contents |
|---|---|
| [`NMC_lib/`](NMC_lib) | The framework implementation: GT primitives, feedback-NMC building blocks and training utilities. |
| [`Examples/`](Examples) | Entry scripts and analysis scripts. |
| [`Experiments&Logs/`](Experiments&Logs) | Recorded TensorBoard events, console logs where available, and the scripts behind the paper's figures and tables. |
| `LICENSE` | MIT license. |

## Method overview

**The problem.** Neural microcircuits (NMCs) — recurrent populations of spiking neurons — are computationally powerful, yet for almost two decades no practical *online* learning rule has existed for training their **feedback connections** ([Maass et al., 2007](https://doi.org/10.1371/journal.pcbi.0030165)). Dominant SNN approaches train recurrent weights with surrogate gradients over a differentiable computation graph, which decouples learning from spike timing.

**The idea.** This work reframes temporal credit assignment as a **state separation** problem: extract the task-relevant components induced by historical perturbations directly from the *current* neural state, instead of tracing errors backward through time. A stochastic model of NMC dynamics (firing rates as expectations over spike events, with the recurrent drive playing the role of intrinsic white noise) makes the network differentiable in its **distributional parameters**, not in its spike events. This yields the **causality-gradient theorem**: the Jacobian of postsynaptic firing rates w.r.t. presynaptic firing rates can be estimated from nothing but the *timing* of pre- and postsynaptic spikes, inside a short stationary window (lead-lag expansion).

**Gradient Tunneling (GT)** implements this theorem in three steps:

1. **Local tracing** — causality between current and lagged spikes is accumulated by an iterative-moving-average filter into an eligibility trace and a Jacobian trace;
2. **Global tunneling** — downstream learning signals tunnel *through* the recurrent population along the Jacobian trace, providing gradient estimates for the trainable feedback weights;
3. **Firing-rate regularization** — keeps population activity near a target rate so that rate estimates stay accurate.

Feedback is wired into the NMC as extra input channels sampled sparsely and uniformly from the delayed population state, so the only trainable recurrent weights are these few feedback columns.

## Headline results

- **Theoretical validation.** The GT-estimated Jacobian closely matches numerical differentiation (correlation ≈1, consistently above a recurrent-free LIF layer); under strongly non-stationary EEG with a frozen readout, GT still improves loss and accuracy, and post-training NMCs pass the white-noise test. This validates the causality-gradient theorem and motivates firing-rate regularization.
- **Evidence integration (T-maze).** Trained from a terminal signal alone, feedback NMCs learn evidence integration while the fixed Lyapunov-tuned LSM fails; learned feedback separates population states into decision clusters and holds decisions at high certainty throughout inference.
- **Incremental add task.** Stepping the effective length from L = 15 to 75 (SNR ≈−45 dB), the frozen LSM's error never recovers while the feedback NMC returns to low error at each step; incremental pre-training yields a curriculum-like warm-up that exceeds the fixed-reservoir memory limit.
- **Real-world benchmarks.** SHD: 73.61 % without surrogate gradients (significantly above e-prop 67.54 % and FPTT 67.24 %; lr = 0 collapses it to ≈54 %). LibEER EEG: best among reproduced models — SEED 60.22 % acc. / 58.04 % F1, DEAP-Valence 73.93 %, best precision/recall/F1 on DEAP-Arousal. All with ≈0.43 % of a dense SRNN's trainable recurrent connections and ≈1.41 s/iteration (≈2× faster than LTC-SNN+FPTT, ≈6.9× faster than SRNN+D-RTRL).

## Installation and quick start

**Requirements.** Python 3.10+ with PyTorch (CUDA recommended; developed under PyTorch 2.x), plus `numpy`, `h5py`, `tqdm`, `scipy`, `matplotlib`, `tensorboard`, `scikit-learn`, `pandas` and `umap-learn`. `NMC_lib/` is imported directly from the repository root — no build step:

```bash
pip install torch numpy h5py tqdm scipy matplotlib tensorboard scikit-learn pandas umap-learn
```

**Datasets.** Data paths are hard-coded in each example script, so download the data and edit the path before running:

- **SHD** (`Examples/SHD/shd.py`): the official HDF5 event files from the Heidelberg spiking datasets (download them from the canonical source cited in the manuscript); edit `shd_dir`.
- **SEED / DEAP** (EEG emotion recognition): the `fSEED.h5` and `fDEAP.h5` files used in the paper; edit the hard-coded `fseed_file` / `fdeap_file` paths in the EEG entry scripts.
- Synthetic tasks (add task, evidence collection) generate their own spike trains and need no data download.

Replay any run's TensorBoard log, e.g.

```bash
tensorboard --logdir "Experiments&Logs/SHD/GT_NMC/seed42/logs"
```

## Code overview

Everything a user needs is re-exported by `NMC_lib` (e.g. `NMC_lib.GradientTunneler`, `NMC_lib.reservoir.FeedbackReservoirDefault`).

| Module | Contents |
|---|---|
| `_base.py` | The GT engine: spiking neurons (`ALIF`), `IterativeMovingAverage` / `LowPassFilter`, spike monitors, `BSAEncoder`, `GradientTunneler`, `ReservoirBase`, and the spike-train / add-task / evidence-collection data generators. |
| `reservoir.py` | LIF kernel `Reservoir`, feedback microcircuits (`FeedbackReservoir`, `FeedbackReservoirDefault`, `HierarchicalFeedbackCircuit`), `ReservoirOfflineWrapper`, Lyapunov-tuned kernel factories. |
| `models.py` | Readout models and NMC assemblies: `LSMBase`, `ParallelReservoirs`, `BandsDenseLSM`, `AddTaskNMC`, `ResidualReadout`, shared by the proposed method and fixed-reservoir comparisons. |
| `EEG.py` | EEG preprocessing: band filtering, frequency-domain entropy sequences (FES), SEED/DEAP HDF5 loaders, electrode index tables. |
| `training_utils.py` | Training loops: `ISIter` (subject splits + BSA encoding), `AddTaskExecutor`, `EvidenceCollectionExecutor`, `NetworkTrainer`, benchmarks and metric aggregation. |

In the code, each trainable microcircuit carries a **`GradientTunneler`** (collected via `get_tunneler_list()`); the paper's local tracing and global tunneling correspond to `get_eligibility_jacobian_traces()` and `GradientTunneler.tunnel()`. The training loop calls `loss.backward()`, tunnels the learning signal through the tunneler list, then steps AdamW in two parameter groups — reservoir weights at `reservoir_lr`, readout at `readout_lr`. Real-input columns are masked (`masked_amount`), so GT updates only the feedback columns; reducing the batch size to 1 makes the rule strictly online.

`Examples/Analysis/jacobian_eval.py` validates the GT Jacobian estimate against numerical differentiation.

## Experiments&Logs

Recorded runs and the analysis scripts behind the paper's figures and tables:

| Family | Contents |
|---|---|
| `SHD/` | Speech-recognition runs — GT method, GT ablation (lr = 0) and the FPTT/e-prop/pp-prop baselines, seeds 42–45 each — with metrics, significance and ablation-plot scripts. |
| `SEED/` | EEG runs: population-scale and feedback-ratio sweeps, ablation studies, and ANN/SNN baseline logs. |
| `DEAP/` | DEAP Valence/Arousal runs over 32 folds — one directory per model (f-NMC and every reproduced baseline). |
| `AddTask/` | Incremental add-task runs for the feedback NMC (FNMC) and the LSM with the RMSE-vs-L plotting scripts. |
| `EvidenceCollection/` | Feedback-ratio sweeps (FNMC 5/15/25 % and LSM) and the analysis scripts behind the evidence figures. |
| `Jacobian/` | Jacobian evaluation script (Fig. 2a). |

## Citation

If you use this work, please cite:

```bibtex
@misc{zhang2026gradient,
  title        = {Local Spike Timing Enables Gradient-based Learning in
                  Neural Microcircuits},
  author       = {Zhang, Xiangnan and Liu, Jingxin and Lu, Ranqi and Liu, Jingyu
                  and Dong, Qunxi and Tian, Fuze and Zhu, Lixian and Hu, Bin and
                  Schuller, Bj{\"o}rn W.},
  year         = {2026},
  eprint       = {2609.08070},
  archivePrefix = {arXiv},
  primaryClass = {cs.NE},
  url          = {https://arxiv.org/abs/2609.08070}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Datasets are third-party and subject to their own licenses (SHD © Heidelberg University; SEED © Shanghai Jiao Tong University; DEAP © Queen Mary University of London).
