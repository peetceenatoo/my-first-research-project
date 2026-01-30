## Generalized Stratified Estimator

This repository provides code and experiments for a **generalized stratification-based approach to offline evaluation of recommender systems with implicit feedback**.

The implemented method, the **Generalized Stratified (GS) estimator**, addresses the limitations of standard offline evaluators under **Missing-Not-At-Random (MNAR)** logging policies. While classical estimators such as **Average-Over-All (AOA)** are biased and **Inverse Propensity Scoring (IPS)** often suffers from high variance (although unbiased), the GS estimator introduces a **controlled bias–variance trade-off**.

By stratifying items (e.g., by popularity) and aggregating feedback within strata, GS:
- Generalizes IPS as a special case
- Reduces estimation variance compared to IPS
- Remains applicable to **implicit feedback** settings where explicit ratings are unavailable

This repository provides experimental comparison of **Naive (AOA)**, **IPS**, and **GS** estimators on real-world datasets through ranking-based metrics.

## Environment Setup

> ⚠️ Python 3.7 is mandatory due to TensorFlow 1.x compatibility.

Install Miniconda: https://docs.conda.io/projects/miniconda/en/latest/

Create and activate the environment:

```bash
conda create -n RecSys-Evaluation python=3.7 anaconda
conda activate RecSys-Evaluation
```

I thank my colleague and friend [Jacopo Piazzalunga](https://github.com/Jacopopiazza) who worked with me on this project.
