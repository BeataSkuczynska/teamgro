import pandas as pd 
import random as rand
import numpy as np
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import models
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from statistics import *
import pickle

#functions for saving/loading the dictionary
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#load in the datasets
eu_vs = pd.read_excel("/Users/borismarinov/Desktop/University/Masters/Year2/Hackathon/EUvsDisinfo/eu_vs_disinfo.xlsx")
fake_covid = pd.read_csv("/Users/borismarinov/Desktop/University/Masters/Year2/Hackathon/teamgro/fake_covid_rehydrated.csv")

#function to preprocess the text (str)
def preprocess_text(text):
    #first lower
    text = text.lower()

    #remove links:
    text = re.sub(r'http\S+', '', text)
    
    #remove emojis, and non alphabetic characters. This keeps @mentions and #hashtags, we can see if we want to remove. Keep . since we might want to know sentence structures
    text = ''.join(c for c in text if c.isalnum() or c == ' ' or c[0] == "@" or c[0] == "#" or c == "." or c == "\n")
    
    return text


#Create clean text columns to use for embeddings
eu_vs['cleaned_text'] = eu_vs.Summary.apply(lambda x: preprocess_text(x))
fake_covid['cleaned_text'] = fake_covid.text.apply(lambda x: preprocess_text(x))


#Looking at number of tokens and sentences for each
eu_vs['token_length'] = eu_vs.cleaned_text.apply(lambda x: len(word_tokenize(x)))
eu_vs['sentence_length'] = eu_vs.cleaned_text.apply(lambda x: len(sent_tokenize(x)))
fake_covid['token_length'] = fake_covid.cleaned_text.apply(lambda x: len(word_tokenize(x)))
fake_covid['sentence_length'] = fake_covid.cleaned_text.apply(lambda x: len(sent_tokenize(x)))


# eu_vs.to_csv("eu_vs_cleaned.csv")
# fake_covid.to_csv("fake_covid_cleaned.csv")


############## Embeddings ############
# Use BERT for mapping tokens to embeddings
model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')

#get all sentences from EU vs Disinfo in form of lists
all_eu_vs_sentences = []

for index in tqdm(range(len(eu_vs))):
    summary = eu_vs.cleaned_text[index]
    all_eu_vs_sentences.append(sent_tokenize(summary))

#need to change to a list only
all_eu_vs_sentences = [item for sublist in all_eu_vs_sentences for item in sublist]

#Create all embeddings
sentence_embeddings = model.encode(all_eu_vs_sentences)

#Create a dictionary for all sentences and embeddings for EU vs Disinfo
eu_embedding_dict = {}
#Add each to dict (sentence as key, embedding as value)
for sentence, embedding in zip(all_eu_vs_sentences, sentence_embeddings):
    eu_embedding_dict[sentence] = embedding

#create new dict to store summaries and average BERT embeddings
average_dict = {}

#Now go over the summaries in the original data
for index in range(len(eu_vs)):

    #get individual summary
    summary = eu_vs.cleaned_text[index]
    summary_sentences = sent_tokenize(summary)

    #iterate over the sentence dict, find matches, and get embeddings for average
    average_embedding = []
    for key, value in eu_embedding_dict.items():
        #if sentence found in summary
        if key in summary_sentences:
            #add the embeddings to list so can get average after
            average_embedding.append(list(value))
    #change to numpy
    average_embedding=np.array([np.array(xi) for xi in average_embedding])
    # average_embedding = [item for sublist in average_embedding for item in sublist]
    if len(average_embedding) > 1:
        average_embedding = np.average(average_embedding, axis=0)
    elif len(average_embedding) == 1:
        average_embedding = average_embedding[0]
    
    #Now that we have the average embedding sorted properly in a nice array
    average_dict[summary] = average_embedding

#Save the dict so can use later and don't need to retrain
save_obj(average_dict, "eu_summaries_dict")


