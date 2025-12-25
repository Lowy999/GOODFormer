# Invariant Graph Transformer for Out-of-Distribution Generalization

[![arXiv](https://img.shields.io/badge/arXiv-2508.00304-b31b1b.svg)](https://arxiv.org/abs/2508.00304)
[![License][license-image]][license-url]
[![DOI](https://zenodo.org/badge/1121006924.svg)](https://doi.org/10.5281/zenodo.18055149)

This is the official code for the implementation of "Invariant Graph Transformer for Out-of-Distribution Generalization"
which is accepted by KDD 2026.

[license-url]: https://github.com/Lowy999/GOODFormer/blob/master/LICENSE
[license-image]:https://img.shields.io/badge/license-GPL3.0-green.svg


## Table of contents

* [Installation](#installation)
* [Run](#run)
* [Citing](#citing)
* [License](#license)
* [Contact](#contact)


## Installation 

### Conda dependencies

```shell
conda env create -f environment.yml
conda activate GOODFormer
```

### Project installation

```shell
pip install -e .
```

## Run

```shell
bash run.sh
```


## Citing
If you find this repository helpful, please cite our [preprint](https://arxiv.org/abs/2508.00304).
```
@article{liao2025invariant,
  title={Invariant Graph Transformer for Out-of-Distribution Generalization},
  author={Liao, Tianyin and Zhang, Ziwei and Sun, Yufei and Hu, Chunyu and Li, Jianxin},
  journal={arXiv preprint arXiv:2508.00304},
  year={2025}
}
```

## License

### Datasets
The GOOD datasets are released under the [MIT License](https://drive.google.com/file/d/1xA-5q3YHXLGLz7xV2tT69a9dcVmiJmiV/view?usp=sharing).

### Code
The GOODFormer codebase is licensed under **GPLv3**:
- Architecture builds upon [GOOD](https://github.com/divelab/GOOD.git) (GPLv3)
- and [LECI](https://github.com/divelab/LECI.git) (GPLv3)
- with some code adapted from [GraphGPS](https://github.com/rampasek/GraphGPS) (MIT License)
- See full license in [LICENSE](LICENSE)

## Contact

Please feel free to contact [Tianyin Liao](mailto:1120230329@mail.nankai.edu.cn)!

