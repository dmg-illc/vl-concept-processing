import torch
from PIL import Image
import pandas as pd
import os
from PIL import Image
from transformers import AlignProcessor, AlignModel, AutoTokenizer, AlignTextModel
import argparse
import numpy as np
import pickle
import sys
from src.embedding_extraction import get_relevant_tokens_indices, find_target_word_pos_with_tens_mapping
from src.paths import ROOT



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', choices=['picture', 'text', 'sentences', 'single_concept']) 
args = parser.parse_args()



model_name = "kakaobrain/align-base"

concepts_df = pd.read_csv(ROOT / 'data/concepts.csv')
concepts_df['Pereira_image_paths'] = [pd.eval(paths_list, engine='python') for paths_list in concepts_df['Pereira_image_paths']]
concepts_df['Pereira_sentences'] = [pd.eval(sent, engine='python') for sent in concepts_df['Pereira_sentences']]

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AlignModel.from_pretrained(model_name).to(device)
processor = AlignProcessor.from_pretrained(model_name)


if args.context =='picture':
    model_component = "image_embeds"
    im_paths = []
    for el in concepts_df.Pereira_image_paths:
        new_l = [os.path.join(ROOT / 'data/pereira_data', path) for path in el]
        im_paths += new_l

    im_arr = np.array(im_paths).reshape(20,54) 

    concepts = [el.split('/')[-1][:-4]for el in im_paths]

    model.eval()
    with torch.no_grad():
        for i, im_batch in enumerate(im_arr):
            images = [Image.open(img).convert('RGB') for img in im_batch] 
            inputs = processor(text="", images = images, return_tensors = 'pt').to(device)
            outputs = model(**inputs)
            if i == 0:
                img_embeds = outputs[model_component].detach().cpu().numpy()
            else:
                img_embeds = np.vstack([img_embeds, outputs[model_component].detach().cpu().numpy()])


    emb_dict = {c: e for c,e in zip(concepts, img_embeds)}
    config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': model_component}
    
    embs_to_save = {'config':config, 'emb_dict':emb_dict}


elif args.context =='text':
    model_component = "text_embeds"
    concepts = concepts_df.concept_name.values.tolist()

    # putting one image just for the argument not to be empty
    path = concepts_df.Pereira_image_paths[0][0]
    img = Image.open(os.path.join(ROOT / 'data/pereira_data', path))
    inputs = processor(text=concepts, images = img, return_tensors = 'pt', padding=True).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    txt_embeds = outputs[model_component].detach().cpu().numpy()
    emb_dict = {c: e for c,e in zip(concepts, txt_embeds)}
    config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': model_component}
    
    embs_to_save = {'config':config, 'emb_dict':emb_dict}


elif args.context =='sentences':

    sent_df = pd.read_csv(ROOT / 'data/screen_sentences.csv')


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AlignTextModel.from_pretrained(model_name).to(device)

    model_component = "hidden_states"
    sentences = sent_df.Sentence_Screen.tolist()

    inputs = tokenizer(text=sentences, padding=True, return_tensors="pt", return_offsets_mapping=True).to(device)
    mapping = inputs['offset_mapping']
    del inputs['offset_mapping']
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs[model_component]
    emb_dict = {sentences[ind]: np.stack([state[ind,:,:].detach().cpu().numpy() for state in hidden_states]) for ind in range(len(sentences))}

    indices = [find_target_word_pos_with_tens_mapping(sentence=s, st_end_df=sent_df, mapping=m) for s,m in zip(sentences, mapping)]

    map_dict = {s: i for s,i in zip(sentences, indices)}

    config = {'context' : args.context,
            'model_name' : model_name,
            'model_component': model_component}
    
    embs_to_save = {'config':config, 'emb_dict':emb_dict, 'map_dict':map_dict}



elif args.context =='single_concept':

    concepts = concepts_df.concept_name.values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AlignTextModel.from_pretrained(model_name).to(device)

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
    
    embs_to_save = {'config':config, 'emb_dict':emb_dict}
    
    
pickle.dump(embs_to_save, open(ROOT / f"data/embeddings/align/align_{args.context}.pkl", "wb"))

