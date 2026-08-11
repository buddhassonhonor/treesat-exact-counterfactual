# TreeSAT-Exact: Exact Counterfactual Explanations for Decision Trees via Leaf-Box Projection

This repository contains the code and figures for the paper:

> **Exact Counterfactual Explanations for Decision Trees via Leaf-Box Projection**  
> Zhigao Huang*, Yipu Yuan*, Shiyan Zheng†, Mianmian Zhou, Jinmei Wu, Jinfa Wei, Shufen Li  
> (* Equal contribution, † Corresponding author: buddhasson@icloud.com)  
> Key Laboratory of Information Functional Material for Fujian Higher Education, Quanzhou Normal University, Quanzhou 362000, China

## Overview

**TreeSAT-Exact** is a closed-form, exact counterfactual solver for binary decision-tree classifiers. It caches target-class leaf boxes once per trained tree and computes the global L₂-optimal counterfactual by projection-and-selection over axis-aligned boxes in O(md) time per query — with no external MIP or SAT solver.

Key properties:
- ✅ Provably global-optimal (see Theorem 1 in the paper)
- ✅ Sub-millisecond median runtime (0.063 ms on benchmark datasets)
- ✅ 14–18× faster than naive leaf enumeration at scale
- ✅ Stable across 20 random seeds and Gaussian noise up to σ = 0.10

## Repository Structure

```
experiment/
    run_experiments.py      # Main experiment runner (all three rounds)
experiment_counterfactual.py  # Standalone solver and utility functions
figures/
    stage1_main_metrics.png          # Round 1 aggregate metrics
    stage2_scalability_depth.png     # Round 2 depth sweep
    stage2_scalability_features.png  # Round 2 feature sweep
    stage2_ablation.png              # Round 2 ablation
    stage3_stability_boxplot.png     # Round 3 stability
    stage3_noise_robustness.png      # Round 3 noise robustness
exp/
    stage1/    # Round 1 result JSON files
    stage2/    # Round 2 result JSON files
    stage3/    # Round 3 result JSON files
    reproducibility.json
    final_assessment.json
```

## Environment

```
Python  3.13
NumPy   2.3.1
pandas  2.3.3
scikit-learn  1.7.2
Matplotlib    3.10.8
```

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Reproducing All Results

### Round 1 — Main accuracy/optimality results
```bash
python experiment/run_experiments.py --round 1 --seeds 5 --queries 60
```

### Round 2 — Scalability and ablation
```bash
python experiment/run_experiments.py --round 2
```

### Round 3 — Stability, noise robustness, and case studies
```bash
python experiment/run_experiments.py --round 3 --seeds 20 --queries 30
```

All figures are written to `figures/` and all numeric results to `exp/`.

## Datasets

All datasets are standard publicly available benchmarks:
- **Breast Cancer**, **Wine**, **Iris**, **Digits** — from the UCI Machine Learning Repository as shipped with scikit-learn (`sklearn.datasets`)
- **Synthetic-20** — a 20-feature binary classification dataset generated procedurally; the generation script is included in `experiment_counterfactual.py`

No proprietary or restricted data were used.

## Citation

If you find this work useful, please cite:

```bibtex
@article{huang2025treesat,
  title={Exact Counterfactual Explanations for Decision Trees via Leaf-Box Projection},
  author={Huang, Zhigao and Yuan, Yipu and Zheng, Shiyan and Zhou, Mianmian and Wu, Jinmei and Wei, Jinfa and Li, Shufen},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
