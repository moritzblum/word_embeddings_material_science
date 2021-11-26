import json
from os import listdir
import os.path as osp
from gensim.models import Word2Vec
from chemdataextractor import Document
from chemdataextractor.reader import PdfReader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne_plot(model):
    labels = list(model.wv.index_to_key)
    vectors = [model.wv[label] for label in labels]

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x, y = new_values[:, 0], new_values[:, 1]

    plt.figure()
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == '__main__':

    preprocess = False

    if preprocess:
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



    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=20, workers=4,
                     compute_loss=True)
    model.save("word2vec.model")

    #model.most_similar('trump')
    tsne_plot(model)
