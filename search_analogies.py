import numpy as np
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

model = Word2Vec.load("word2vec.model")

# --- Examples ---
print('Most similar to manganese:', model.wv.most_similar(positive=['manganese']))

# manganese - copper = iron - ?
# manganese - copper - iron = ?
print('copper to manganese is like ? to iron:',
      model.wv.most_similar(positive=['manganese'], negative=['copper',  'iron']))

labels = [l for (l, _) in model.wv.most_similar(positive=['manganese'], topn=100)]
vectors = np.array([model.wv[label] for label in labels])

# --- plot word vectors ---
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(vectors)
x, y = new_values[:, 0], new_values[:, 1]

fig, axs = plt.subplots(figsize=(16, 9))
axs.scatter(x, y)
for (x_c, y_c, label_c) in zip(x, y, labels):
    axs.annotate(label_c, xy=(x_c, y_c), ha='center')
plt.show()


# --- plot difference vectors ---
# complexity: x²/2
num_comparisons = vectors.shape[0] ** 2 - vectors.shape[0]

diff_matrix = np.zeros((num_comparisons, vectors.shape[1]))
diff_matrix_labels_lower = []
diff_matrix_labels_upper = []


for i in range(1, vectors.shape[0]):
    start_index = sum(range(0, i - 1))
    diff = vectors[0:i] - vectors[i]
    diff_matrix[start_index:start_index + i] = diff
    diff_matrix[num_comparisons - start_index - i:num_comparisons - start_index] = -diff
    diff_matrix_labels_lower.extend([labels[i] + ' to ' + label] for label in labels[0:i])
    diff_matrix_labels_upper.extend([label + ' to ' + labels[i]] for label in labels[0:i])

diff_matrix_labels = diff_matrix_labels_lower + diff_matrix_labels_upper[::-1]
# flatten the list
diff_matrix_labels = [item for sublist in diff_matrix_labels for item in sublist]

tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(diff_matrix)
x, y = new_values[:, 0], new_values[:, 1]
fig, axs = plt.subplots(figsize=(16, 9))
axs.scatter(x, y)
for (x_c, y_c, label_c) in zip(x, y, diff_matrix_labels):
    axs.annotate(label_c, xy=(x_c, y_c), ha='center')
plt.show()
