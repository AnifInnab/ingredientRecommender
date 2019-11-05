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

#HELPERS
def read_data(path):
    df = pd.read_csv(path)
    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    
    PP_df = pd.read_csv('../data/PP_recipes.csv')
    PP_df = PP_df[['id','techniques']]

    #use only recipes from the PP_recipes df, also take the techniques column from it. Dont use their tokenization though...
    df_merged = pd.merge(df, PP_df, left_on='id', right_on='id', how='right')
    return df_merged

def create_model(min_count=1000):
    model = Word2Vec(min_count=min_count,
                        window=3,
                        size=50,
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

def get_recipe_embeddings(model, recipes):

    def embed_recipe(recipe, model):
        recipe_embedding = np.empty([len(recipe), model.vector_size])
        for i, ingredient in enumerate(recipe):
            recipe_embedding[i] = model.wv[ingredient]
        return recipe_embedding

    def add_recipe_embedding(recipe_embedding, model):
        resultant = np.zeros(model.vector_size)
        print(recipe_embedding.size)
        for ingr_embd in recipe_embedding:
            resultant += ingr_embd
        return resultant

    #embed all ingredients in each recipe
    recipe_embeddings = list(map(lambda x : embed_recipe(x, model), tqdm(recipes)))

    #add all ingredients eembedding in each recipe
    added_recipe_embeddings = list(map(lambda x : add_recipe_embedding(x, model), tqdm(recipe_embeddings)))
    added_recipe_embeddings = np.array(added_recipe_embeddings)
    return added_recipe_embeddings
    
def tsne(X):
    #TSNE on recipe embeddings
    tsne = TSNE(n_components=2, verbose=1, n_jobs=2)
    X_tsne = tsne.fit_transform(X)
    
    return X_tsne

def hover_plot(X_tsne, df):

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        hover_idx = ind['ind'][0]
        text = "{} \n {}".format(df['name'].iloc[hover_idx], df['ingredients'].iloc[hover_idx])
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    x = X_tsne[:,0]
    y = X_tsne[:,1]

    norm = plt.Normalize(1,4)
    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots()
    sc = plt.scatter(x,y, s=1, cmap=cmap, norm=norm)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)

    plt.show()

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
    load = args.load

    if load:
        df, df_filtered, model, recipes_ids, X, X_tsne, vocab, vocab_set = load_pkl(pkl_path)
    else:
        df = read_data(data_path)
        recipes = df['ingredients']

        model = create_model(min_count=2000)
        vocab, vocab_set = create_vocab(model, recipes)
        recipes_keep_ids = filter_recipes(recipes, vocab_set)

        df_filtered = df.iloc[recipes_keep_ids]
        filtered_recipes = df_filtered['ingredients']
        train_model(model, filtered_recipes)

        X = get_recipe_embeddings(model, filtered_recipes)

        X_tsne = tsne(X)
        save_pkl((df, df_filtered, model, recipes_keep_ids, X, X_tsne, vocab, vocab_set), pkl_path)
        

    #print(X_tsne)
    hover_plot(X_tsne, df_filtered)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    parser.add_argument('-d', '--data_path', action='store', dest='data_path', help='path to data')
    parser.add_argument('-l', '--load', action='store_true', dest='load', help='flag deciding if we load pkl object or not')
    args = parser.parse_args()

    main(args)