import torch
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from transformers import BertTokenizer, BertTokenizerFast, VisualBertModel
from frcnn_utils import Config
import pickle
# import os, json
from os.path import join
import pandas as pd
import os
import argparse
import numpy as np
from src.embedding_extraction import get_relevant_tokens_indices, find_target_word_pos_with_tens_mapping
from src.paths import ROOT


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'sentences', 'single_concept'])
parser.add_argument('-s','--spec', default='')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'



model_name = "uclanlp/visualbert-vqa-coco-pre"
model_component = ['language_hidden_states', 'pooler_output']

# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)

bert_tokenizer = BertTokenizerFast.from_pretrained('google-bert/bert-base-uncased')
visualbert = VisualBertModel.from_pretrained(model_name).to(device)
# print(visualbert)


df = pd.read_csv(ROOT / 'data/concepts_with_start_end_idxs.csv')

if args.context =='picture':


    im_paths = [join(ROOT / 'data/pereira_data', p) for p in df.image_paths]
    concepts = df.concepts

    ret_dict = {}

    for i in range(len(df)):
        # run frcnn
        images, sizes, scales_yx = image_preprocess(im_paths[i])
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt"
        )

        # Very important that the boxes are normalized
        normalized_boxes = output_dict.get("normalized_boxes").to(device)
        features = output_dict.get("roi_features").to(device)

        # run visualbert

        inputs = bert_tokenizer(
            concepts[i],
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(device)

        output_vbert = visualbert(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_embeds=features,
        visual_attention_mask=torch.ones(features.shape[:-1]).to(device),
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
        return_dict=True,
        output_hidden_states=True
        )

        hidden_states = output_vbert['hidden_states']
        pooler_output = output_vbert['pooler_output'].detach().cpu().numpy()
        ret_hidden = np.vstack([h.detach().cpu().numpy() for h in hidden_states])

        ret_dict[df.image_paths[i].split('/')[-1]] = {'hidden_states' : ret_hidden, 
                                                      'pooler_output': pooler_output,
                                                      'mapping': inputs['offset_mapping'].detach().cpu().numpy()}




if args.context =='sentences':

    sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')

    sentences = sent_df.Sentence_Screen.tolist()
 
    ret_dict = {}

    for i in range(len(sentences)):

        # run visualbert

        inputs = bert_tokenizer(
            sentences[i],
       
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(device)

        output_vbert = visualbert(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
        return_dict=True,
        output_hidden_states=True
        )

        hidden_states = output_vbert['hidden_states']
        pooler_output = output_vbert['pooler_output'].detach().cpu().numpy()
        ret_hidden = np.vstack([h.detach().cpu().numpy() for h in hidden_states])
        indices = find_target_word_pos_with_tens_mapping(sentences[i], sent_df, inputs['offset_mapping'])
    

        if sent_df.Concept[i].lower() in ret_dict:
            ret_dict[sent_df.Concept[i].lower()][sentences[i]] = {'mapping': indices,
                                                      'hidden_states' : ret_hidden, 
                                                      'pooler_output': pooler_output}
        else:
            ret_dict[sent_df.Concept[i].lower()] = {sentences[i] : {'mapping': indices,
                                                        'hidden_states' : ret_hidden, 
                                                        'pooler_output': pooler_output}}
        
if args.context =='single_concept':
    
    concepts = df.concepts
    
    tokenizer = bert_tokenizer
    model = visualbert

    model_component = "hidden_states"

    inputs = tokenizer(text=concepts, padding=True, return_tensors="pt", return_offsets_mapping=True)
    mapping = inputs['offset_mapping'].numpy()
    del inputs['offset_mapping']


    model.eval()
    with torch.no_grad():
        outputs = model(**inputs.to(device), output_hidden_states=True)

    hidden_states = outputs[model_component]
    emb_dict = {concept: np.stack([state[ind,get_relevant_tokens_indices(mapping[ind], concept),:].detach().cpu().numpy().mean(axis=0) for state in hidden_states]) for ind, concept in enumerate(concepts)}



config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': model_component}

pickle.dump({'config': config, 'out_dict': ret_dict}, open(ROOT / f"results/embeddings/visualbert/visualbert_{args.context}.pkl"), "wb")
