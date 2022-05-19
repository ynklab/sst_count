# Logical Inference for Counting on Semi-structured Tables

## Overview


## Usage

### Environment
- Linux (WSL; Ubuntu 20.04)
- Python 3.7 (venv)

### Preparation
```
pip install -r requirements.txt
pip install depccg==1.1.0
python -m spacy download en_core_web_sm
```
Because `depccg_en download` does not work (confirmed in May 2022), you may have to download from the [direct link](https://drive.google.com/file/d/1mxl1HU99iEQcUYhWhvkowbE4WOH0UKxv) for the depccg_en model and locate the proper directory.

### Experiment 1 (General)
```
python experiments/general_exp.py
```

### Experiment 2 (Running Time)
```
./experiments/time_exp.sh
```

## Citation
If you use either our dataset or our work, or both in any published research, please cite the following:
* Tomoya Kurosawa and Hitomi Yanaka. 2022. [Logical Inference for Counting on Semi-structured Tables](https://aclanthology.org/2022.acl-srw.8/)
In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, pages 84–96, Dublin, Ireland. Association for Computational Linguistics.


```
@inproceedings{kurosawa-yanaka-2022-logical,
    title = "Logical Inference for Counting on Semi-structured Tables",
    author = "Kurosawa, Tomoya  and
      Yanaka, Hitomi",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-srw.8",
    pages = "84--96"
}
```

Our dataset comes from [InfoTabS (Gupta et al., 2020)](https://infotabs.github.io/).

## Contact
For questions and usage issues, please contact kurosawa-tomoya@is.s.u-tokyo.ac.jp .

## License
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)