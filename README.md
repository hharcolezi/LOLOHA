# LOngitudinal LOcal HAshing (LOLOHA)
Repository for the paper: *[Héber H. Arcolezi](https://hharcolezi.github.io/), [Carlos Pinzón](https://www.caph.info/), [Catuscia Palamidessi](http://www.lix.polytechnique.fr/Labo/Catuscia.Palamidessi/), [Sébastien Gambs](https://sebastiengambs.openum.ca/). "Frequency Estimation of Evolving Data Under Local Differential Privacy" (2022)*. <https://arxiv.org/abs/2210.00262>.

If our codes and work are useful to you, we would appreciate a reference to:

```
@article{Arcolezi2022,
  title={Frequency Estimation of Evolving Data Under Local Differential Privacy},
  author={Arcolezi, Héber H. and Pinzón, Carlos and Palamidessi, Catuscia and Gambs, Sébastien},
  journal={arXiv preprint arXiv:2210.00262},
  year={2022}
}
```

## Codes & Datasets
All experiments in the paper are repeated over 20 iterations. Here we provide four Jupyter notebooks that use a reduced fraction of the respective dataset to decrease execution time through 5 iterations. Please use the whole dataset (frac=1) and all iterations (nb_seed=20) to fully reproduce the paper's results.

- The [LDP](https://github.com/hharcolezi/LOLOHA/tree/main/LDP) folder has all developed longitudinal LDP protocols.
- The [datasets](https://github.com/hharcolezi/LOLOHA/tree/main/datasets) folder has all used datastes.
- Experiments:
  - The [Experiments_Adult.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_Adult.ipynb) Jupyter notebook has the experimental evaluation with the Adult dataset.
  - The [Experiments_Syn.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_Syn.ipynb) Jupyter notebook has the experimental evaluation with the Synthetic dataset.
  - The [Experiments_DB_MT.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_DB_MT.ipynb) Jupyter notebook has the experimental evaluation with the DB_MT dataset.
  - The [Experiments_DB_DE.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_DB_DE.ipynb) Jupyter notebook has the experimental evaluation with the DB_DE dataset.
- Appendix:
  - The [Appendix_Theoretical_Analysis.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Appendix_LOLOHA_Analysis.ipynb) Jupyter notebook has the theoretical analysis of our LOLOHA protocol (privacy levels, estimator, variance, and the optimization of parameter g) and of state-of-the-art protocols.
  - The [Appendix_Variances.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Appendix_Variances.ipynb) Jupyter notebook has the theorical variances (Table 1) and the numerical analysis of variances (Fig. 2).

## Environment
Our codes were developed using Python 3 with numpy, pandas, and numba libaries. The versions are listed below:

- Python 3.8.8
- Numpy 1.23.1
- Pandas 1.2.4
- Numba 0.53.1

## Contact
For any question, please contact [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr

## License
[MIT](https://github.com/hharcolezi/LOLOHA/blob/main/LICENSE)
