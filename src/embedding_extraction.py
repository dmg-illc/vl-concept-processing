import numpy as np

def get_relevant_tokens_indices(offset_mapping, target_word):
  
  """
    Offset mapping should be a numpy array with the offset mapping of a single concept. 
    Target word is the word that should be found within the sentence, e.g., 'corkscrew'.

    The function returns the indices of the offset mappings that correspond to the 
    target words (so the indices that can be used for the indexing the hidden states).
  """

  start_ind = 0
  end_ind = start_ind + len(target_word)

  ret_ind = []
  for i in range(offset_mapping.shape[0]):
    if offset_mapping[i,0] != offset_mapping[i,1]:
        if (offset_mapping[i,0] >= start_ind) & (offset_mapping[i,1]<= end_ind):
            ret_ind.append(i)

  return np.array(ret_ind)

def find_target_word_pos_with_tens_mapping(sentence, st_end_df, mapping):

  """
      This is for when the mapping is for a single sentence
  """
  
  start, end = st_end_df.loc[st_end_df.Sentence_Screen == sentence, ['Word_Start', 'Word_End']].values[0]
 
  ret = []
  for i, (s,e) in enumerate(mapping.squeeze()):

    s, e = s.item(), e.item()
    
    if s!=e:
      if (s>=start) & (e<=end):
        ret.append(i)
  
  if len(ret) == 0:
     print(sentence[start:end])
  return np.array(ret)