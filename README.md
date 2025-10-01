# Vision-Language Models Align with Human Concept Processing

This repo contains the code to replicate results from the paper _Vision-Language Models Align with Human Concept Processing_.

[Experiment Overview Image.pdf](https://github.com/user-attachments/files/22643600/Experiment.Overview.Image.pdf)

The content of this repository is structured as follows:
```
.
├── LICENSE
├── README.md
├── data
│   ├── README.md
│   ├── concepts.csv
│   ├── concepts_with_start_end_idxs.csv
│   └── screen_sentences.csv
├── representation_extraction
│   ├── README.md
│   ├── align.py
│   ├── bert.py
│   ├── clip.py
│   ├── idefics2.py
│   ├── llama3.py
│   ├── llava.py
│   ├── lxmert.py
│   ├── mistral.py
│   └── visualbert.py
├── rsa
│   ├── README.md
│   ├── getting_brain_rdms.ipynb
│   ├── partial_correlations.ipynb
│   └── rsa.ipynb
├── setup.py
└── src
    ├── __init__.py
    ├── best_layers.py
    ├── embedding_extraction.py
    ├── paths.py
    ├── statistical_tests.py
    └── utils.py

```

The Python scripts used to extract model representations are provided within `representation_extraction`.

Jupyter notebooks to replicate results from representational similarity analysis (RSA) and ablation studies are included in `rsa`. More specifically, see `rsa/rsa.ipynb` for code to compute RSA results for the main experiments and the ablation study where only concept-words were passed to vision-language models; see `rsa/partial_correlations.ipynb` for code about the ablation study where we regressed language-only models' RDMs (representational dissimilarity matrices) out of vision-language models' RDMs.


The model-derived RDMs will be made publicly available shortly, while the brain-derived RDMs can be recreated using the code provided in `rsa/getting_brain_rdms.ipynb`.
