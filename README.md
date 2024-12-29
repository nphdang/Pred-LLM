# Pred-LLM: Tabular Data Generation via Large Language Models (LLMs)
This is the implementation of the Pred-LLM method in the paper "Generating Realistic Tabular Data with Large Language Models", ICDM 2024: https://icdm2024.org/

# Framework
The method Pred-LLM includes three phases: (1) fine-tuning a pre-trained LLM with the real dataset, (2) sampling synthetic samples conditioned on each feature, and (3) constructing prompts based on the generated data to query labels.

![framework](https://github.com/nphdang/Pred-LLM/blob/main/predllm_method.jpg)

# Installation
Please refer to the file "requirements.txt"

# How to run
## For classification tasks:
```
python -W ignore pred_llm.py --dataset iris --method original --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
python -W ignore pred_llm.py --dataset iris --method pred_llm --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
```
## For regression tasks:
```
python -W ignore pred_llm_reg.py --dataset california --method original --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
python -W ignore pred_llm_reg.py --dataset california --method pred_llm --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
```
## To run for all datasets and all methods:
```
python -W ignore pred_llm.py --dataset classification --method all --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
python -W ignore pred_llm_reg.py --dataset regression --method all --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3
```

# Citation
```
@inproceedings{nguyen2024predllm,
  title={Generating Realistic Tabular Data with Large Language Models},
  author={Nguyen, Dang and Gupta, Sunil and Do, Kien and Nguyen, Thin and Venkatesh, Svetha},
  booktitle={IEEE International Conference on Data Mining (ICDM)},  
  year={2024},  
}
```

# Acknowledgements
- Great: As Pred-LLM heavily relies on Great, we thank the authors of Great for the publicly shared code: https://github.com/kathrinse/be_great
- Synthcity: We also thank the authors of Synthcity for the shared code to evaluate the quality and diversity of synthetic samples: https://github.com/vanderschaarlab/synthcity
