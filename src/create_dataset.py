import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import argparse
from functools import reduce
import datetime
#HELPERS
def read_data(raw_path, pp_path, rip_path):
    rip_df = pd.read_csv(rip_path)
    rip_df = rip_df[['recipe_id', 'rating']]
    rip_df = rip_df.groupby(['recipe_id']).mean()

    raw_df = pd.read_csv(raw_path)
    raw_df['ingredients'] = raw_df['ingredients'].apply(ast.literal_eval)

    raw_df = pd.merge(raw_df, rip_df, left_on='id', right_on='recipe_id', how='left')
 
    PP_df = pd.read_csv(pp_path)
    PP_df = PP_df[['id','techniques']]

    #use only recipes from the PP_recipes df, also take the techniques column from it. Dont use their tokenization though...
    df_merged = pd.merge(raw_df, PP_df, left_on='id', right_on='id', how='right')
    return df_merged

def create_model(min_count=1000, size=50):
    model = Word2Vec(min_count=min_count,
                        window=3,
                        size=size,
                        sg=0
                        )
    return model

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

def create_vocab(model, recipes):
    model.build_vocab(recipes, progress_per=100)
    vocab = list(model.wv.vocab)
    vocab_set = set(vocab)
    return vocab, vocab_set

def train_model(model, recipes):
    model.train(recipes, total_examples=model.corpus_count, epochs=30, report_delay=1)

def get_recipe_embeddings(model, recipes, maxlen, output_dim):

    def embed_recipe(recipe, maxlen, model):
        for i, ingredient in enumerate(recipe):
            word_vec = model.wv[ingredient]
            if(i == 0):
                recipe_embedding = word_vec
            else:
                recipe_embedding = np.vstack((recipe_embedding, word_vec))
        #PAD
        ingredients_to_pad = (maxlen - len(recipe))
        for i in range(0, ingredients_to_pad):
            recipe_embedding = np.vstack((recipe_embedding, np.zeros(output_dim)))
        return recipe_embedding
        
    #embed all ingredients in each recipe
    recipe_embeddings = list(map(lambda x : embed_recipe(x, maxlen, model), tqdm(recipes)))

    return recipe_embeddings

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

def main(args):
    pkl_path = args.pkl_path
    raw_path = args.raw_path
    pp_path = args.pp_path
    rip_path = args.rip_path
    ingr_embd_dim = int(args.ingr_embd_dim)
    min_count = int(args.min_count)
    df = read_data(raw_path, pp_path, rip_path)
    recipes = df['ingredients']
    model = create_model(min_count=min_count, size=ingr_embd_dim)
    vocab, vocab_set = create_vocab(model, recipes)
    recipes_keep_ids = filter_recipes(recipes, vocab_set)

    #the filtered recipes are the recipes which have ingredients bigger than min_count
    df_filtered = df.iloc[recipes_keep_ids]
    filtered_recipes = df_filtered['ingredients']

    maxlen = max(list(filtered_recipes.apply(len)))

    train_model(model, filtered_recipes)

    X_r = get_recipe_embeddings(model, filtered_recipes, maxlen, ingr_embd_dim)
    X_r = np.array(X_r)
    #X_r = X_r.reshape(-1,1)
    print(X_r[:5].shape)
    X_t = list(df_filtered['techniques'])
    X_t = list(map(ast.literal_eval, X_t))
    X_t = np.array(X_t)
    print("abc__",X_t.shape)
    #X_ratings = list(df_filtered['rating'])
    #X_ratings = np.array(X_ratings)
    #X_ratings = X_ratings.reshape(-1,1)

    #print(X_r.shape, X_t.shape, X_ratings.shape)
    #X = np.concatenate((X_r, X_t, X_ratings), axis=1)
    y = np.array(df_filtered['minutes'])

    print(X_r.shape)
    print(y.shape)

    save_pkl((df_filtered, X_r, y), pkl_path, ingr_embd_dim, min_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    parser.add_argument('-rrp', '--RAW_recipes_path', action='store', dest='raw_path', help='path to data', default='./data/RAW_recipes.csv')
    parser.add_argument('-rip', '--RAW_interactions_path', action='store', dest='rip_path', help='path to data', default='./data/RAW_interactions.csv')
    parser.add_argument('-ppp', '--pp_recipes_path', action='store', dest='pp_path', help='path to data', default='./data/PP_recipes.csv')
    parser.add_argument('-ied', '--ingr_embd_dim', action='store', dest='ingr_embd_dim', help='nr of dims for feature vectors', default=50)
    parser.add_argument('-mc', '--min_count', action='store', dest='min_count', help='min_count', default=0)
    args = parser.parse_args()

    main(args)