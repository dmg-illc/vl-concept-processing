# Embedding extraction

This folder contains the code we used to extract model representations in the different conditions.

The HuggingFace implementations of LXMERT and VisualBERT include only the language model, but not the feature extractor. However, there is a [HuggingFace Project](https://github.com/huggingface/transformers-research-projects/tree/main/lxmert) providing some code to work around the issue. The modules that we call at the beginning of `lxmert.py` and `visualbert.py` are the same as those included in the HF project.

As for the GloVe representations included in our study, they were extracted using the [official vectors](https://nlp.stanford.edu/projects/glove/) pretrained on CommonCrawl (840B tokens version).