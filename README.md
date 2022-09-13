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

- The [codes](https://github.com/hharcolezi/LOLOHA/tree/main/codes) folder has all developed longitudinal LDP protocols.
- The [datasets](https://github.com/hharcolezi/LOLOHA/tree/main/datasets) folder has all used datastes.
- The [1_LOLOHA_Analysis.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/1_LOLOHA_Analysis.ipynb) Jupyter notebook has all analytical analysis of the protocols (estimators, variance, privacy levels), the optimization of LOLOHA g parameter, and the analytical variance evaluation.
- The [2_LOLOHA_Adult.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/2_LOLOHA_Adult.ipynb) Jupyter notebook has the (simplified) experimental evaluation with the Adult dataset.

## Environment
I mainly used Python 3 with numpy, pandas, and numba libaries. The versions I use are listed below:

- Python 3.8.8
- Numpy 1.23.1
- Pandas 1.2.4
- Numba 0.53.1

## Contact
For any question, please contact [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr

## License
[MIT](https://github.com/hharcolezi/LOLOHA/blob/main/LICENSE)
