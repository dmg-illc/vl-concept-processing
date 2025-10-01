# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import pickle
from PIL import Image
import torch
# import os, json
from os.path import join
import pandas as pd
import os
import argparse
import numpy as np
from src.embedding_extraction import get_relevant_tokens_indices, find_target_word_pos_with_tens_mapping
from src.paths import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'sentences', 'single_concept']) # can be picture or sentences
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "llava-hf/llama3-llava-next-8b-hf"


sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')
sentences = sent_df.Sentence_Screen.tolist()
df = pd.read_csv(ROOT / 'data/concepts_with_start_end_idxs.csv')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

llava = LlavaNextForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")
processor = LlavaNextProcessor.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)

if args.context =='picture':

    im_paths = [join(ROOT / 'data/pereira_data', p)for p in df.image_paths]
    concepts = df.concepts

    ret_dict = {}
    for i in range(len(df)):
        print(df.image_paths[i].split('/')[-1])
        image = Image.open(im_paths[i])
        template = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{concepts[i]}"
        offset_mapping = processor(image, template, return_tensors="pt", padding = False, return_offsets_mapping=True)['offset_mapping']
        inputs = processor(image, template, return_tensors="pt", padding = False).to(device)
        output = llava(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        _, text_start_ind = torch.where(inputs['input_ids']==198) # 198 is the last input id before the text starts
        text_start_ind = text_start_ind.item()
        ret_dict[df.image_paths[i].split('/')[-1]] = {'hidden_states': np.vstack([hidden_states[i][0,text_start_ind+1:,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])}
        del inputs, output, hidden_states
        torch.cuda.empty_cache()

elif args.context =='sentences':

    concepts = df.concepts

    ret_dict = {}
    for i in range(len(sentences)):
        template = f"<|start_header_id|>user<|end_header_id|>\n\n{sentences[i]}"

        inputs = processor(text=template, return_tensors="pt", padding = False, return_offsets_mapping=True).to(device)
        print(inputs.keys())
        offset_mapping = inputs['offset_mapping'][0, 5:,:]-42 # that ensures we only consider the sentence-relevant part of the input
        inputs.pop('offset_mapping')
        output = llava(**inputs, output_hidden_states=True)
        hidden_states = [h[:, 5:,:] for h in output.hidden_states]
        print(hidden_states[0].size())
    
        indices = find_target_word_pos_with_tens_mapping(sentences[i], sent_df, offset_mapping)
        print(indices)
        if sent_df.Concept[i].lower() in ret_dict:
            ret_dict[sent_df.Concept[i].lower()][sentences[i]] = {'mapping': indices,
                                                        'offsets_mapping': offset_mapping.detach().cpu().numpy(),
                                                    #   'input_ids' : inputs['input_ids'].detach().cpu().numpy(), 
                                                      'hidden_states': np.vstack([hidden_states[i][0,indices,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])
                                                      }
        else:
            ret_dict[sent_df.Concept[i].lower()] = {sentences[i] : {'mapping': indices,
                                                        'offsets_mapping': offset_mapping.detach().cpu().numpy(),
                                                    #   'input_ids' : inputs['input_ids'].detach().cpu().numpy()},
                                                      'hidden_states': np.vstack([hidden_states[i][0,indices,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])}
                                                      }
        del inputs, output, hidden_states
        torch.cuda.empty_cache()

elif args.context =='single_concept':
        concepts = df.concepts

        ret_dict = {}
        templates = [f"<|start_header_id|>user<|end_header_id|>\n\n{concept}" for concept in concepts]
        
        inputs = processor(text=templates, return_tensors="pt", padding = False, return_offsets_mapping=True).to(device)
        print(inputs.keys())
        offset_mapping = [inputs['offset_mapping'][i, 5:,:]-42 for i, _ in enumerate(concepts)] # that ensures we only consider the sentence-relevant part of the input
        inputs.pop('offset_mapping')
        outputs = llava(**inputs, output_hidden_states=True)
        ret_dict = {concept: np.stack([state[ind,get_relevant_tokens_indices(offset_mapping[ind], concept),:].detach().cpu().numpy().mean(axis=0) for state in outputs.hidden_states]) for ind, concept in enumerate(concepts)}
        
config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': 'hidden_states'}

pickle. dump({'config': config, 'out_dict': ret_dict}, open(ROOT / "results/embeddings/llava/llava_{args.context}.pkl", "wb"))