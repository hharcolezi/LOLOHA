# LOngitudinal LOcal HAshing (LOLOHA)
Repository for the paper: *[Héber H. Arcolezi](https://hharcolezi.github.io/), [Carlos Pinzón](https://www.caph.info/), [Catuscia Palamidessi](http://www.lix.polytechnique.fr/Labo/Catuscia.Palamidessi/), [Sébastien Gambs](https://sebastiengambs.openum.ca/). "Frequency Estimation of Evolving Data Under Local Differential Privacy"*. In: Proceedings of the 26th International Conference on Extending Database Technology, EDBT 2023, Ioannina, Greece, March 28 - March 31, 2023. pp. 512–525. <http://dx.doi.org/10.48786/edbt.2023.44>.

If our codes and work are useful to you, we would appreciate a reference to:

```
@inproceedings{Arcolezi2023,
  author    = {Arcolezi,  Héber H. and Pinzón,  Carlos A and Palamidessi,  Catuscia and Gambs,  Sébastien},
  title     = {Frequency Estimation of Evolving Data Under Local Differential Privacy},
  booktitle = {Proceedings of the 26th International Conference on Extending Database
               Technology, {EDBT} 2023, Ioannina, Greece, March 28 - March 31, 2023},
  pages     = {512--525},
  publisher = {OpenProceedings.org},
  year      = {2023},
  doi       = {10.48786/EDBT.2023.44},
}
```

## Codes & Datasets
All experiments in the paper are repeated over 20 iterations. Here we provide four Jupyter notebooks that use a reduced fraction of the respective dataset to decrease execution time through 5 iterations. Please use the whole dataset (**frac=1**) and all iterations (**nb_seed=20**) to fully reproduce the paper's results.

- The [LDP](https://github.com/hharcolezi/LOLOHA/tree/main/LDP) folder has all developed longitudinal LDP protocols.
- The [datasets](https://github.com/hharcolezi/LOLOHA/tree/main/datasets) folder has all used datastes.
- Experiments:
  - The [Experiments_Adult.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_Adult.ipynb) Jupyter notebook has the experimental evaluation with the **Adult dataset**.
  - The [Experiments_Syn.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_Syn.ipynb) Jupyter notebook has the experimental evaluation with the **Synthetic dataset**.
  - The [Experiments_DB_MT.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_DB_MT.ipynb) Jupyter notebook has the experimental evaluation with the **DB_MT dataset**.
  - The [Experiments_DB_DE.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Experiments_DB_DE.ipynb) Jupyter notebook has the experimental evaluation with the **DB_DE dataset**.
- Appendix:
  - The [Appendix_Theoretical_Analysis.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Appendix_LOLOHA_Analysis.ipynb) Jupyter notebook has the theoretical analysis of our LOLOHA protocol (privacy levels, estimator, variance, and the optimization of parameter g) and of state-of-the-art LDP protocols.
  - The [Appendix_Variances.ipynb](https://github.com/hharcolezi/LOLOHA/blob/main/Appendix_Variances.ipynb) Jupyter notebook has the theoretical variances and the numerical analysis of variances (Fig. 2).

**We have implemented LOLOHA mechanisms into our [multi-freq-ldpy](https://github.com/hharcolezi/multi-freq-ldpy) Python package.**

## Environment
Our codes were developed using Python 3 with numpy, pandas, and numba libraries. The versions are listed below:

- Python 3.8.8
- Numpy 1.23.1
- Pandas 1.2.4
- Numba 0.53.1

## Contact
For any questions, please contact:
- [Héber H. Arcolezi](https://hharcolezi.github.io/): heber.hwang-arcolezi [at] inria.fr
- [Carlos Pinzón](https://www.caph.info/): carlos.pinzon [at] inria.fr

## License
[MIT](https://github.com/hharcolezi/LOLOHA/blob/main/LICENSE)