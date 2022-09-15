# LOngitudinal LOcal HAshing (LOLOHA)
Repository for the paper: *[Héber H. Arcolezi](https://hharcolezi.github.io/), [Carlos Pinzón](https://www.caph.info/), [Catuscia Palamidessi](http://www.lix.polytechnique.fr/Labo/Catuscia.Palamidessi/), [Sébastien Gambs](https://sebastiengambs.openum.ca/). "Frequency Estimation of Evolving Data Under Local Differential Privacy" (2022). **Full Version**: .*

If our codes and work are useful to you, we would appreciate a reference to:

```
@article{Arcolezi2022,
  title={Frequency Estimation of Evolving Data Under Local Differential Privacy},
  author={Arcolezi, Héber H. and Pinzón, Carlos and Palamidessi, Catuscia and Gambs, Sébastien},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```

## Codes & Datasets
All experiments in the paper are repeated over 20 iterations. Here we provide two Jupyter notebooks that use a reduced fraction of the respective dataset to decrease execution time through 5 iterations. Please use the whole dataset (frac=1) and all iterations (nb_seed=20) to fully reproduce the paper's results.

- The [LDP](https://github.com/hharcolezi/LOLOHA/tree/main/LDP) folder has all developed longitudinal LDP protocols.
- The [datasets](https://github.com/hharcolezi/LOLOHA/tree/main/datasets) folder has all used datastes.
- The [1_LOLOHA_Analysis.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/1_LOLOHA_Analysis.ipynb) Jupyter notebook has all analytical analysis of the protocols (estimators, variance, privacy levels), the optimization of LOLOHA g parameter, and the analytical variance evaluation.
- The [2_LOLOHA_Adult.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/2_LOLOHA_Adult.ipynb) Jupyter notebook has the experimental evaluation with the Adult dataset.
- The [2_LOLOHA_PWGTP_Delaware.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/2_LOLOHA_PWGTP_Delaware.ipynb) Jupyter notebook has the experimental evaluation with the DB_DE (PWGTP_Delaware) dataset.

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
