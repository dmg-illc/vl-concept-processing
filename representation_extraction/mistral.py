from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
from PIL import Image
import torch
import pandas as pd
import os
import argparse
import numpy as np
from src.embedding_extraction import find_target_word_pos_with_tens_mapping
from src.paths import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'sentences', 'single_concept']) 

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "mistralai/Mistral-7B-v0.1"

sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')
sentences = sent_df.Sentence_Screen.tolist()
df = pd.read_csv(ROOT / 'data/concepts_with_start_end_idxs.csv')

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

if args.context =='sentences':

    concepts = df.concepts

    ret_dict = {}
    for i in range(len(sentences)):
    # for i in range(2):
  
        inputs = tokenizer(sentences[i], return_tensors="pt", return_offsets_mapping=True, padding=False)
        inputs.to(device)
        offset_mapping = inputs['offset_mapping']
        inputs.pop('offset_mapping')
        output = model(
                    **inputs,
                    output_hidden_states=True     
                    )
        hidden_states = [h.detach().cpu().float().numpy() for h in output.hidden_states]
        print(offset_mapping)
        print(inputs.input_ids.size())
        indices = find_target_word_pos_with_tens_mapping(sentences[i], sent_df, offset_mapping)
        if sent_df.Concept[i].lower() in ret_dict:
            ret_dict[sent_df.Concept[i].lower()][sentences[i]] = {             
                                                      'hidden_states': np.vstack([h[0,indices,:].mean(axis=0) for h in hidden_states])
                                                      }
        else:
            ret_dict[sent_df.Concept[i].lower()] = {sentences[i] : {
                                                      'hidden_states': np.vstack([h[0,indices,:].mean(axis=0) for h in hidden_states])}
                                                    }
        del inputs, output
        torch.cuda.empty_cache()



elif args.context =='picture':

    concepts = df.concepts.unique()

    ret_dict = {}
    for i in range(len(concepts)):
    
        inputs = tokenizer(concepts[i], return_tensors="pt", return_offsets_mapping=True, padding=False)
        inputs.to(device)
        offset_mapping = inputs['offset_mapping']
        inputs.pop('offset_mapping')
        output = model(
                    **inputs,
                    output_hidden_states=True     
                    )
        hidden_states = [h.detach().cpu().float().numpy() for h in output.hidden_states]
        ret_dict[concepts[i]] = {'hidden_states': np.vstack([h[0,:,:].mean(axis=0) for h in hidden_states])}      # if sent_df.Concept[i].lower() in ret_dict:
        del inputs, output
        torch.cuda.empty_cache()

config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': 'hidden_states'}

pickle. dump({'config': config, 'out_dict': ret_dict}, open(ROOT / f"embeddings/mistral/mistral_{args.context}.pkl", "wb"))