import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

model = Word2Vec.load("word2vec.model")

labels = [label for label in list(model.wv.index_to_key) if label not in stopwords.words('english')]
vectors = np.array([model.wv[label] for label in labels])

# complexity: x²/2
num_comparisons = vectors.shape[0] ** 2

diff_matrix = np.zeros((num_comparisons, vectors.shape[1]))

for i in range(vectors.shape[0]):
    start_index = sum(range(0, i+1))
    diff = vectors[0:i + 1] - vectors[i]
    diff_matrix[start_index:start_index + i + 1] = diff
    diff_matrix[num_comparisons - start_index - i - 1:num_comparisons - start_index] = -diff

#diff_matrix = diff_matrix[~np.all(diff_matrix == 0, axis=1)]
print(diff_matrix)
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
new_values = tsne_model.fit_transform(vectors)
x, y = new_values[:, 0], new_values[:, 1]
plt.scatter(x, y)
plt.show()











