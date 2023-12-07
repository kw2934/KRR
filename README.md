# Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift

Paper: (https://arxiv.org/abs/2302.10160).


## Demonstration

See `Demo.ipynb` for a synthetic data experiment. 
- Feature space: $\mathcal{X} = [0, 1]$.
- Response model: $y|x \sim N( f^\*(x) , \sigma^2 )$ with $f^\* (x) = \cos(2\pi x) - 1$ and $\sigma = 0.5$.
- Source covariate distribution: $P = \frac{B}{B + 1} \mathcal{U} [0, 1/2] + \frac{1}{B + 1} \mathcal{U} [1/2, 1]$ with $B = 5$.
- Target covariate distribution: $Q = \frac{1}{B + 1} \mathcal{U} [0, 1/2] + \frac{B}{B + 1} \mathcal{U} [1/2, 1]$ with $B = 5$.
- Samples sizes: 1000 (source) and 500 (target).
- Kernel: First-order Sobolev kernel $K(z, w) = \min \lbrace z , w \rbrace $.

We compare model selection methods that are based on different validation datasets.
- New method (red): target data with pseudo-labels;
- Oracle method (cyan): target data with noiseless responses;
- Naive method (blue): half of source data.

<p align="center">
    <img src="demo.png" alt="Demonstration" width="500" height="400" />
</p>

## Citation
```
@article{wang2023pseudo,
  title={Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift},
  author={Wang, Kaizheng},
  journal={arXiv preprint arXiv:2302.10160},
  year={2023}
}
```
