import torch
from processing_image import Preprocess
from modeling_frcnn import GeneralizedRCNN
from transformers import LxmertTokenizer, LxmertTokenizerFast, LxmertForPreTraining
from frcnn_utils import Config
import pickle
from os.path import join
import pandas as pd
import os
import argparse
import numpy as np
from src.embedding_extraction import get_relevant_tokens_indices
from src.paths import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'sentences', 'single_concept','sentence_w_images']) 
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_name = "unc-nlp/lxmert-base-uncased"
model_component = 'language_hidden_states'

# load models and model components
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)


lxmert_base = LxmertForPreTraining.from_pretrained(model_name).to(device)


df = pd.read_csv(ROOT / 'data/concepts_with_start_end_idxs.csv')

if args.context =='picture':

    lxmert_tokenizer = LxmertTokenizer.from_pretrained(model_name)

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

        # run lxmert

        inputs = lxmert_tokenizer(
            concepts[i],
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(device)

        # print(inputs)

        output_lxmert = lxmert_base(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True
        )

        hidden_states = output_lxmert['language_hidden_states']
        
        ret_hidden = np.vstack([h.detach().cpu().numpy() for h in hidden_states])

        ret_dict[df.image_paths[i].split('/')[-1]] = ret_hidden




elif args.context =='sentences':

    lxmert_tokenizer = LxmertTokenizerFast.from_pretrained(model_name)

    im_dir = ROOT / 'data/pereira/stimuli/noise_images'
    sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')
    concepts = sent_df.Concept.apply(lambda x: x.lower()).unique()
    ret_dict = {}

    for concept in concepts:
        ret_dict[concept] = {}
        im_paths = [join(im_dir, concept, p) for p in os.listdir(join(im_dir, concept))]
        print(im_paths)
        sentences = sent_df.Sentence_Screen[sent_df.Concept.apply(lambda x: x.lower()) == concept].tolist()

    # run frcnn
        images, sizes, scales_yx = image_preprocess(im_paths)
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt"
        )

        normalized_boxes = output_dict.get("normalized_boxes").to(device)
        features = output_dict.get("roi_features").to(device)
        inputs = lxmert_tokenizer(
            sentences,
            padding="max_length",
            max_length=30,  # 20
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True
        ).to(device)

        output_lxmert = lxmert_base(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            visual_feats=features,
            visual_pos=normalized_boxes,
            token_type_ids=inputs.token_type_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True
        )

        # hidden_states = output_lxmert['language_hidden_states']
        hidden_states = [state.detach().cpu().numpy() for state in output_lxmert['language_hidden_states']]
        for i in range(len(sentences)):
            ret_dict[concept][sentences[i]] = {'hidden_states' : np.stack([layer[i,:] for layer in hidden_states]),
                                                'mapping': inputs['offset_mapping'][i].detach().cpu().numpy()}


elif args.context == 'single_concept':

    im_dir = ROOT / 'data/pereira/stimuli/noise_images'
    concepts = df.concepts
    tokenizer = LxmertTokenizerFast.from_pretrained(model_name)
    model = lxmert_base

    model_component = "language_hidden_states"
    im_paths = [str(im_dir / concept / 'noise1.jpg') for concept in concepts]

    ret_dict = dict()

    for i, concept in enumerate(concepts):
        images, sizes, scales_yx = image_preprocess([im_paths[i]])
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
        inputs = tokenizer(
            concept,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        mapping = inputs['offset_mapping'].numpy()
        del inputs['offset_mapping']

        inputs = inputs.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                visual_feats=features,
                visual_pos=normalized_boxes,
                token_type_ids=inputs.token_type_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True
            )

        hidden_states = outputs[model_component]
        # considering only the first hidden state because I'm looking at one word at a time
        ret_dict[concept] = np.stack([state[0,get_relevant_tokens_indices(mapping[0], concept),:].detach().cpu().numpy().mean(axis=0) for state in hidden_states])
        del hidden_states, inputs, features, normalized_boxes, output_dict
        





config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': model_component}

pickle.dump({'config': config, 'out_dict': ret_dict}, open(ROOT / "results/embeddings/lxmert/lxmert_{args.context}.pkl"), "wb")



