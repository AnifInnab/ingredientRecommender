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
#HELPERS
def read_data(path):
    df = pd.read_csv(path)
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    
    PP_df = pd.read_csv('../data/PP_recipes.csv')
    PP_df = PP_df[['id','techniques']]

    #use only recipes from the PP_recipes df, also take the techniques column from it. Dont use their tokenization though...
    df_merged = pd.merge(df, PP_df, left_on='id', right_on='id', how='right')
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
        recipe_embedding = np.zeros(0)
        for i, ingredient in enumerate(recipe):
            word_vec = model.wv[ingredient]
            recipe_embedding = np.concatenate((recipe_embedding, word_vec))
        #PAD
        zeros_to_pad = output_dim * (maxlen - len(recipe))
        recipe_embedding = np.pad(recipe_embedding, (0,zeros_to_pad), 'constant')
        return recipe_embedding

    #embed all ingredients in each recipe
    recipe_embeddings = list(map(lambda x : embed_recipe(x, maxlen, model), tqdm(recipes)))

    return recipe_embeddings

def save_pkl(obj, path):
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
    data_path = args.data_path
    ingr_embd_dim = int(args.ingr_embd_dim)
    min_count = int(args.min_count)

    df = read_data(data_path)
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

    X_t = list(df_filtered['techniques'])
    X_t = list(map(ast.literal_eval, X_t))
    X_t = np.array(X_t)

    X = np.concatenate((X_r, X_t), axis=1)
    y = np.array(df_filtered['minutes'])

    print(X.shape)
    print(y.shape)

    save_pkl((df_filtered, X, y), pkl_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    parser.add_argument('-d', '--data_path', action='store', dest='data_path', help='path to data')
    parser.add_argument('-ied', '--ingr_embd_dim', action='store', dest='ingr_embd_dim', help='nr of dims for feature vectorss', default=50)
    parser.add_argument('-mc', '--min_count', action='store', dest='min_count', help='min_count', default=0)
    args = parser.parse_args()

    main(args)