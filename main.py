import json
import numpy as np
import argparse
import os.path as osp
import nltk
from os import listdir
from gensim.models import Word2Vec
from chemdataextractor import Document
from chemdataextractor.reader import PdfReader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.corpus import stopwords

nltk.download('stopwords')
parser = argparse.ArgumentParser(description='Generate Word Embeddings')
parser.add_argument('--min', type=int, default=10, help='minimum token frequency for training and plot')
parser.add_argument('--max', type=int, default=1000, help='minimum token frequency for plot')
parser.add_argument('--source', type=str, default='', help='directory of PDF publications')
args = parser.parse_args()

print('min token frequency:', args.min)
print('max token frequency:', args.max)


def tsne_plot(model):
    labels = [label for label in list(model.wv.index_to_key) if label not in stopwords.words('english')]
    print('number of removed stopwords:', len(list(model.wv.index_to_key)) - len(labels))

    counts = [model.wv.get_vecattr(label, "count") for label in labels]
    count_filter = list(map(lambda c: c < args.max, counts))

    labels = np.array(labels)[count_filter]
    vectors = np.array([model.wv[label] for label in labels])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)
    x, y = new_values[:, 0], new_values[:, 1]

    fig, axs = plt.subplots(figsize=(16, 9))

    axs.scatter(x, y)
    for (x_c, y_c, label_c) in zip(x, y, labels):
        axs.annotate(label_c, xy=(x_c, y_c), ha='center')

    plt.show()


if __name__ == '__main__':

    if args.source != '':
        pdf_dir = './data/pdf'
        sentences = []
        files = list(filter(lambda x: not x.startswith('.'), listdir(pdf_dir)))
        for file in files:
            f = open(osp.join(pdf_dir, file), 'rb')
            doc = Document.from_file(f, readers=[PdfReader()])
            for element in doc.elements:
                current_sentences = []
                element_tokens = element.tokens
                for sentence_tokens in element_tokens:
                    tokens = [token.text for token in sentence_tokens]
                    sentences.append(tokens)

        with open('./data/tokenized.json', 'w') as sentences_out:
            json.dump(sentences, sentences_out)
    else:
        with open('./data/tokenized.json') as sentences_in:
            sentences = json.load(sentences_in)

    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=args.min, workers=4,
                     compute_loss=True)
    model.save("word2vec.model")

    tsne_plot(model)


