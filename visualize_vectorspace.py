import re
from collections import Counter
from matplotlib import colors
import numpy as np
from matplotlib.lines import Line2D
from nltk.corpus import stopwords
import gensim
from matplotlib import pyplot
from sklearn.decomposition import PCA
import gensim.downloader as api
from gensim.models import Word2Vec

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


# todos
# häufigste gruppen-konstellationen einfärben
# Atommasse wie Preis Plotten mit Zusammensetzung





ms_model = Word2Vec.load('./data/material2vec/pretrained_embeddings')
vocab = [x for x in ms_model.wv.key_to_index]

heusler_df = pd.read_csv('./data/heusler.csv', sep=';', header=None, names=['compound', 'heusler', 'groups', 'price', 'weight'])

# filter for heuserl, half_heusler
heusler_df = heusler_df[heusler_df['heusler'] != 'unknown']
heusler_df = heusler_df[heusler_df['heusler'] != 'off-stoichiometric']
heusler_df = heusler_df[heusler_df['heusler'] != 'sums-up-to-100']


# filter for compounds in our embedding space
heusler_df = heusler_df[heusler_df['compound'].isin(vocab)]

cmap = 'RdYlGn_r'


# --- vis price ---
colormap = matplotlib.cm.get_cmap(name=cmap)

heusler_price_df = heusler_df[heusler_df['price'] != 'unknown'].copy()
heusler_price_df['price'] = pd.to_numeric(heusler_price_df['price'])

# normalize to [0-1]
heusler_price_df['price'] = heusler_price_df['price'].div(heusler_price_df['price'].max())
heusler_price_df['price'] = heusler_price_df['price'][abs(heusler_price_df['price'] - np.median(heusler_price_df['price'])) < 0.1 * np.std(heusler_price_df['price'])]
heusler_price_df['price'] /= heusler_price_df['price'].max()
heusler_price_df = heusler_price_df.dropna()
hist = np.histogram(heusler_price_df['price'])

X = ms_model.wv[heusler_price_df['compound']]
pca = TSNE(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1], color=colormap(1-heusler_price_df['price']))

plt.legend()
plt.title("Composition Price")
plt.set_cmap(cmap)
cbar = plt.colorbar()
cbar.ax.set_ylabel('normalized price from cheap (0) to expensive (1) ', rotation=270, labelpad=20)
plt.tight_layout()
figure = plt.gcf() # get current figure
figure.set_size_inches(16, 9)

plt.savefig("./data/heusler_price_no_labels.svg")

for i, word in enumerate(heusler_price_df['compound']):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='black')


plt.savefig("./data/heusler_price_with_labels.svg")
plt.show()


# --- vis weight ---
heusler_weight_df = heusler_df[heusler_df['weight'] != 0].copy()


# normalize to [0-1]
heusler_weight_df['weight'] = heusler_weight_df['weight'].div(heusler_weight_df['weight'].max())

print(heusler_weight_df['weight'])

heusler_weight_df = heusler_weight_df.dropna()
hist = np.histogram(heusler_weight_df['weight'])

X = ms_model.wv[heusler_weight_df['compound']]
pca = TSNE(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1], color=colormap(1-heusler_weight_df['weight']))

plt.legend()
plt.title("Composition Weight")
plt.set_cmap(cmap)
cbar = plt.colorbar()
cbar.ax.set_ylabel('normalized price from light (0) to heavy (1) ', rotation=270, labelpad=20)
plt.tight_layout()
figure = plt.gcf()  # get current figure
figure.set_size_inches(16, 9)
plt.savefig("./data/heusler_weight_no_labels.svg")

for i, word in enumerate(heusler_weight_df['compound']):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='black')


plt.savefig("./data/heusler_weight_with_labels.svg")
plt.show()


# --- vis element typs ---
elements_df = pd.read_csv('data/elements.csv').fillna('unknown')
elements_df = elements_df[elements_df['Symbol'].isin(list(ms_model.wv.key_to_index))]

X = ms_model.wv[elements_df['Symbol']]

pca = TSNE(n_components=2)
result = pca.fit_transform(X)

cmap = {'Halogen': '#a6cee3', 'Noble Gas': '#1f78b4', 'Metal': '#b2df8a', 'Transactinide': '#33a02c', 'Metalloid': '#fb9a99', 'Lanthanide': '#e31a1c', 'Alkali Metal': '#fdbf6f', 'Transition Metal': '#ff7f00', 'Actinide': '#cab2d6', 'Alkaline Earth Metal': '#6a3d9a', 'Nonmetal': '#ffff99', 'unknown': '#b15928'}

scatter = pyplot.scatter(result[:, 0], result[:, 1], label=[cmap[x] for x in elements_df['Type']], color=[cmap[x] for x in elements_df['Type']], s=50)
colours = [cmap[e] for e in  elements_df['Type']]

plt.legend()
plt.title("Types of Elements")
plt.tight_layout()
figure = plt.gcf() # get current figure
figure.set_size_inches(16, 9)
for i, word in enumerate(elements_df['Symbol']):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='black')

legend_elements = []
for type, color in cmap.items():
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=type,
                          markerfacecolor=color, markersize=15))

plt.legend(handles=legend_elements, loc='upper right', title="Types")
plt.savefig("./data/element_types.svg")

pyplot.show()


# --- vis groups ---
colormap = matplotlib.cm.get_cmap(name='tab10')

heusler_groups_df = heusler_df[heusler_df['groups'] != 'unknown'].copy()
counter = Counter(heusler_groups_df['groups'].tolist())
most_frequent = list(sorted(counter.keys(), key=counter.get, reverse=True))[:10]
heusler_groups_df = heusler_groups_df[heusler_groups_df['groups'].isin(most_frequent)]

X = ms_model.wv[heusler_groups_df['compound']]
pca = TSNE(n_components=2)
result = pca.fit_transform(X)

colors = []
for c in heusler_groups_df['groups']:
    if c in most_frequent:
        colors.append(colormap(most_frequent.index(c)))

    else:
        colors.append('red')

plt.scatter(result[:, 0], result[:, 1], c=colors)
for i, word in enumerate( heusler_groups_df['compound']):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='black')


plt.title("Compounds by Element Groups")
plt.tight_layout()
figure = plt.gcf() # get current figure
figure.set_size_inches(16, 9)

legend_elements = []
for index, type in enumerate(most_frequent):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=type,
                          markerfacecolor=colormap(index), markersize=15))

plt.legend(handles=legend_elements, loc='upper right', title="Types")
plt.savefig("./data/heusler_groups.svg")

plt.show()


# --- vis heusler types ---
materials = heusler_df['compound'].tolist()
types = heusler_df['heusler'].tolist()


X = ms_model.wv[materials]

pca = TSNE(n_components=2)
result = pca.fit_transform(X)


cmap = {'half_heusler': 'orange', 'unknown': 'green', 'full_heusler': 'yellowgreen', 'off-stoichiometric': 'yellow', 'sums-up-to-100': 'orange'}


pyplot.scatter(result[:, 0], result[:, 1], color=[cmap[x] for x in types])




legend_elements = [Line2D([0], [0], marker='o', color='w', label="Half Heusler",
                          markerfacecolor='orange', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label="Full Heusler",
                          markerfacecolor='yellowgreen', markersize=15)
                   ]


plt.legend(handles=legend_elements, loc='upper right', title="Types")
plt.title("Full vs. Half Heusler")

figure = plt.gcf() # get current figure
figure.set_size_inches(16, 9)
plt.savefig("./data/heusler_types_no_labels.svg")

for i, word in enumerate(materials):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), color='black')

plt.savefig("./data/heusler_types_with_labels.svg")

plt.show()
