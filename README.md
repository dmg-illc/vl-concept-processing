# Vision-Language Models Align with Human Neural Representations in Concept Processing

This repo contains the code to replicate results from the paper [_Vision-Language Models Align with Human Neural Representations in Concept Processing_](https://aclanthology.org/2026.eacl-long.150/), by [Anna Bavaresco](https://annabavaresco.github.io/), [Marianne de Heer Kloots](https://mdhk.net/), [Sandro Pezzelle](https://sandropezzelle.github.io/) and [Raquel Fernández](https://staff.fnwi.uva.nl/r.fernandezrovira/).

<img width="1004" height="300" alt="Experiment Overview Image" src="https://github.com/user-attachments/assets/d747cd7d-6e73-4406-8709-a751e4457cd1" />


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


The model-derived RDMs are publicly available at [https://zenodo.org/records/15221180](https://zenodo.org/records/15221180), while the brain-derived RDMs can be recreated using the code provided in `rsa/getting_brain_rdms.ipynb`.
