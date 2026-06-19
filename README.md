# Coupled Pendulum Complexity Scaling

**Paper:** Power-law scaling of effective degrees of freedom in heterogeneous coupled pendulum systems across spatial dimensions

**Authors:**
- Bang Won Kyu — Heatmap Research, Adelante Inc., Republic of Korea
- Bang Jun Suk — Department of Mechanical Engineering, Ritsumeikan University, Japan

---

## Abstract

We numerically investigate how effective dynamical complexity scales with system size, spatial dimension, and parameter heterogeneity in damped coupled pendulum systems under a fixed total energy constraint. Key findings:

- Homogeneous 1D chains (N=2-1000): PR proportional to N^-0.858
- PCA effective dimension saturates at D_eff = 16 (95.5% cumulative variance) for N >= 200
- Energy robustness within +-10% over E_total in [1, 100]
- Critical disorder threshold sigma_c approx 0.4
- 2D scaling: alpha = 0.628 (L=2-10); 3D scaling: alpha = 0.789 (L=2-15, N up to 3375)
- KZ spectrum verification: exponent -0.567, R-squared = 0.170

---

## Repository Structure

- `pendulum_simulation.py` — Full simulation suite
- `data/` — CSV results (1D, 2D, 3D scaling, disorder, energy sensitivity, D_eff saturation, KZ spectrum, summary)
- `figures/` — Generated figures (PDF and PNG)
- `paper/` — Paper PDF and LaTeX source

---

## How to Run

### Google Colab (Recommended)

```python
# Cell 1
!pip install scipy numpy matplotlib pandas tqdm

# Cell 2
!wget https://raw.githubusercontent.com/mypolian-cyber/coupled-pendulum-complexity-scaling/main/pendulum_simulation.py
exec(open('pendulum_simulation.py').read())
```

### Local

```bash
pip install scipy numpy matplotlib pandas tqdm
python pendulum_simulation.py
```

Estimated runtime: several hours on CPU (1D N=1000 and 3D L=15 are computationally intensive).

---

## Requirements

- python >= 3.8
- scipy >= 1.7
- numpy >= 1.21
- matplotlib >= 3.4
- pandas >= 1.3
- tqdm >= 4.62

---

## Citation

```bibtex
@article{bang2026pendulum,
  title   = {Power-law scaling of effective degrees of freedom in
             heterogeneous coupled pendulum systems across spatial dimensions},
  author  = {Bang, Won Kyu and Bang, Jun Suk},
  year    = {2026}
}
```

---

## License

MIT License — see LICENSE file.

## DOI

DOI will be assigned via Zenodo upon release.
