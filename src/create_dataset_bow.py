import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import argparse
from keras.preprocessing.text import one_hot
from gensim.models import Word2Vec
#HELPERS
def read_data(raw_path, pp_path):
    df = pd.read_csv(raw_path)
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    
    PP_df = pd.read_csv(pp_path)
    PP_df = PP_df[['id','techniques']]

    #use only recipes from the PP_recipes df, also take the techniques column from it. Dont use their tokenization though...
    df_merged = pd.merge(df, PP_df, left_on='id', right_on='id', how='right')
    return df_merged

def filter_recipes(recipes, vocab_set):
    """
    removes recipe if ingredients in recipe is not in our ingredients list
    """
    recipes_ids = []
    for i, recipe in enumerate(recipes):
        found = True
        for word in recipe:
            if word not in vocab_set:
                found = False
                break
        if found and len(recipe) > 0:
            recipes_ids.append(i)
    return recipes_ids

def save_pkl(obj, path, ingr_embd_dim, min_count):
    if(path == None):
        path = "./pkl_files/{:%d%m%y}_{}_{}.pkl".format(datetime.datetime.now(), ingr_embd_dim, min_count)
    pickle_out = open(path, "wb")
    pickle.dump(obj, pickle_out)
    print('Saved ', path)
    pickle_out.close()

def load_pkl(path):
    pickle_in = open(path, "rb")
    print('Loaded ', path)
    return pickle.load(pickle_in)

def get_vocab_size(recipes):
    #We are only using word2vec because it is east to obtain vocab size, probably easier way to do this though... But I could just copy paste this so...
    model = Word2Vec(min_count=0,
                        window=3,
                        size=10,
                        sg=0
                        )
    model.build_vocab(recipes, progress_per=100)
    return len(list(model.wv.vocab))

def main(args):
    pkl_path = args.pkl_path
    raw_path = args.raw_path
    pp_path = args.pp_path

    #merges raw_recipes and pp_recipes, so that we have df with techniques
    df = read_data(raw_path, pp_path)
    recipes = df['ingredients']

    #temporary size
    vocab_size = get_vocab_size(recipes)
    print(len(vocab_size))

    recipes_str = list(map(lambda x : ' '.join(x), recipes))
    encoded_recipes = [one_hot(r, vocab_size) for r in recipes_str]

    maxlen = max(list(map(len, encoded_recipes)))

    print(len(recipes))
    print(len(encoded_recipes))
    
    #save_pkl((df_filtered, X, y), pkl_path, ingr_embd_dim, min_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    parser.add_argument('-rrp', '--RAW_recipes_path', action='store', dest='raw_path', help='path to data', default='./data/RAW_recipes.csv')
    parser.add_argument('-ppp', '--pp_recipes_path', action='store', dest='pp_path', help='path to data', default='./data/PP_recipes.csv')
    args = parser.parse_args()

    main(args)