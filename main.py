from os import listdir
import os.path as osp
from chemdataextractor import Document
from chemdataextractor.reader import PdfReader
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec


class MaterialScienceCorpus:
    def __init__(self, path):
        """Iterates over sentences form a set of papers."""
        self.path = path

    def __iter__(self):
        self.files = list(filter(lambda x: not x.startswith('.'), listdir(self.path)))
        self.current_file = []
        self.current_sentences = []
        return self

    def __next__(self):
        # stop iteration if all data is consumed
        if len(self.current_sentences) == len(self.files) == 0:
            raise StopIteration

        # load all data form one file if all sentences are consumed
        while len(self.current_sentences) == 0:

            # load new paper
            self.current_file = osp.join(self.path, self.files.pop(0))
            f = open(self.current_file, 'rb')
            doc = Document.from_file(f, readers=[PdfReader()])
            for element in doc.elements:
                element_tokens = element.tokens
                for sentence_tokens in element_tokens:
                    tokens = [token.text for token in sentence_tokens]
                    self.current_sentences.append(tokens)

        return self.current_sentences.pop(0)


if __name__ == '__main__':
    class callback(CallbackAny2Vec):
        '''Callback to print loss after each epoch.'''

        def __init__(self):
            self.epoch = 0

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
            self.epoch += 1

    dir = './data'
    model = Word2Vec(sentences=MaterialScienceCorpus(dir), vector_size=100, window=5, min_count=1, workers=4,
                     compute_loss=True, callbacks=[callback()])
    model.save("word2vec.model")
