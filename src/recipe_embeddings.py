import pandas as pd
import numpy as np
import ast
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle
import sys
import argparse

#HELPERS
def filter_recipes(recipes, ingredients_set):
    """
    removes recipe if ingredients in recipe is not in our ingredients list
    """
    filtered_recipes = []
    recipes_ids = []
    for i, recipe in enumerate(recipes):
        found = True
        for word in recipe:
            if word not in ingredients_set:
                found = False
                break
        if found and len(recipe) > 0:
            filtered_recipes.append(recipe)
            recipes_ids.append(i)
    return filtered_recipes, recipes_ids

def embed_recipe(recipe, model):
    recipe_embedding = np.empty([len(recipe), model.vector_size])
    for i, ingredient in enumerate(recipe):
        recipe_embedding[i] = model.wv[ingredient]
    
    return normalize(recipe_embedding)

def add_recipe_embedding(recipe_embedding, model):
    resultant = np.zeros(model.vector_size)
    for ingr_embd in recipe_embedding:
        resultant += ingr_embd
    return resultant

def read_data(path):
    recipes = pd.read_csv(path)
    recipes_df = recipes['ingredients']
    recipes_df = recipes_df.apply(ast.literal_eval)
    recipes = list(recipes_df)
    return recipes, recipes_df

def create_train_model(recipes, min_count=800):
    model = Word2Vec(min_count=min_count,
                        window=3,
                        size=50,
                        sg=0
                        )
    model.build_vocab(recipes, progress_per=100)
    vocab = list(model.wv.vocab)
    vocab_set = set(vocab)

    #Find new recipes which only has ingredients present in vocab
    filtered_recipes, recipes_ids = filter_recipes(recipes, vocab_set)

    model.train(filtered_recipes, total_examples=model.corpus_count, epochs=30, report_delay=1)

    return model, filtered_recipes, recipes_ids

def get_recipe_embeddings(filtered_recipes, model):
    #embed all ingredients in each recipe
    recipe_embeddings = list(map(lambda x: embed_recipe(x, model), tqdm(filtered_recipes)))

    #add all ingredients eembedding in each recipe
    added_recipe_embeddings = list(map(lambda x : add_recipe_embedding(x, model), tqdm(recipe_embeddings)))
    added_recipe_embeddings = np.array(added_recipe_embeddings)

    return added_recipe_embeddings

def tsne(X):
    #TSNE on recipe embeddings
    tsne = TSNE(n_components=2, verbose=1)
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def hover_plot(X_tsne, names):

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind['ind'][0]]
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


class Step:
    def __init__(self, load, path):
        self.load = load
        self.path = path

def main(args):
    pkl_path = args.pkl_path
    data_path = args.data_path
    load = args.load
    if load:
        recipes, recipes_df, model, filtered_recipes, recipes_ids, X, X_tsne = load_pkl(pkl_path)
    else:
        recipes, recipes_df = read_data(data_path)
        model, filtered_recipes, recipes_ids = create_train_model(recipes)
        X = get_recipe_embeddings(filtered_recipes, model)
        X_tsne = tsne(X)
        save_pkl((recipes,recipes_df, model, filtered_recipes, recipes_ids, X, X_tsne), pkl_path)
        
    names = list(recipes_df[recipes_ids])
    hover_plot(X_tsne, names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pkl_path', action='store', dest='pkl_path', help='path to pkl file')
    parser.add_argument('-d', '--data_path', action='store', dest='data_path', help='path to data')
    parser.add_argument('-l', '--load', action='store_true', dest='load', help='flag deciding if we load pkl object or not')
    args = parser.parse_args()

    main(args)