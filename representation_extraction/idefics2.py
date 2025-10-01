# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import Idefics2Processor, Idefics2ForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
import pickle
from PIL import Image
import torch
from os.path import join
import pandas as pd
import argparse
import numpy as np
from src.embedding_extraction import get_relevant_tokens_indices, find_target_word_pos_with_tens_mapping
from src.paths import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'sentences', 'single_concept']) # can be picture or sentences
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = "HuggingFaceM4/idefics2-8b"

sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')
sentences = sent_df.Sentence_Screen.tolist()
df = pd.read_csv(ROOT / 'data/concepts_with_start_end_idxs.csv')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
idefics = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.float16,    
    quantization_config=quantization_config,
    ).to(device)

processor = AutoProcessor.from_pretrained(model_name, do_image_splitting=False)

if args.context =='picture':

    im_paths = [join(ROOT / 'data/pereira_data', p) for p in df.image_paths]
    concepts = df.concepts

    ret_dict = {}
    for i in range(len(df)):

        print(df.image_paths[i].split('/')[-1])

        image = Image.open(im_paths[i])
        template = f"{concepts[i]}<image>"
    
        inputs = processor(image, template, return_tensors="pt", padding = False)
       
        end_id = np.where(inputs.input_ids.detach().cpu().numpy()[0]==32000)[0][0]
      
        output = idefics(**inputs.to(device), output_hidden_states=True)
        hidden_states = output.hidden_states
        print(output.hidden_states[0].size())
     
        ret_dict[df.image_paths[i].split('/')[-1]] = {'hidden_states': np.vstack([hidden_states[i][0,1:end_id,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])}
        del inputs, output, hidden_states
        torch.cuda.empty_cache()

elif args.context =='sentences':

    concepts = df.concepts

    ret_dict = {}

    for i in range(len(sentences)):
 
        inputs = processor(text=sentences[i], return_tensors="pt", padding = False, return_offsets_mapping=True).to(device)
        print(inputs.keys())
        offset_mapping = inputs['offset_mapping']
        inputs.pop('offset_mapping')
        output = idefics(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
    
        indices = find_target_word_pos_with_tens_mapping(sentences[i], sent_df, offset_mapping)
        if sent_df.Concept[i].lower() in ret_dict:
            ret_dict[sent_df.Concept[i].lower()][sentences[i]] = {
                                                      'hidden_states': np.vstack([hidden_states[i][0,indices,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])
                                                      }
        else:
            ret_dict[sent_df.Concept[i].lower()] = {sentences[i] : {
                                                      'hidden_states': np.vstack([hidden_states[i][0,indices,:].detach().cpu().numpy().mean(axis=0) for i in range(len(hidden_states))])}
                                                      }
        del inputs, output, hidden_states
        torch.cuda.empty_cache()

elif args.context =='single_concept':
    concepts = df.concepts

    inputs = processor(text=concepts, return_tensors="pt", padding = False, return_offsets_mapping=True).to(device)
    print(inputs.keys())
    offset_mapping = inputs['offset_mapping']
    inputs.pop('offset_mapping')
    outputs = idefics(**inputs, output_hidden_states=True)

    ret_dict = {concept: np.stack([state[ind,get_relevant_tokens_indices(offset_mapping[ind], concept),:].detach().cpu().numpy().mean(axis=0) for state in outputs.hidden_states]) for ind, concept in enumerate(concepts)}


config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': 'hidden_states'}

pickle. dump({'config': config, 'out_dict': ret_dict}, open(ROOT / f"results/embeddings/idefics2/idefics2_{args.context}.pkl", "wb"))