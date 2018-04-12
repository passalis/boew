import numpy as np
# from glove import Glove
from tqdm import tqdm


class WordVectorizer():
    """
    Reads a collection of documents and assigns an id to each word (also supports trimming the rare words and loading
    an initial embedding)
    """

    def __init__(self):
        self.word2id = {}
        self.n_words = 0
        self.word_embedding_dim = 300

    def fit(self, texts, min_tf=5):
        """
        Learn the mapping between each word and each id
        :param texts:
        :return:
        """
        word2counts = {}
        # Count word occurrences
        for document in tqdm(texts):
            for word in document.split():
                if word not in self.word2id:
                    self.word2id[word] = self.n_words
                    self.n_words += 1
                    word2counts[word] = 0
                else:
                    word2counts[word] += 1

        # Pass from the words and trim words with less than min_tf occurrences
        word_2idx = {}
        id2word = []
        counter = 0
        for x in word2counts:
            if word2counts[x] > min_tf:
                word_2idx[x] = counter
                id2word.append(x)
                counter += 1

        self.word2id = word_2idx
        self.id2word = id2word
        self.n_words = counter

    def transform(self, texts):
        """
        Transform text to word ids
        :param texts:
        :return:
        """
        word_vectors = []
        # Count word occurrences
        for document in tqdm(texts):
            vec = []
            for word in document.split():
                if word in self.word2id:
                    vec.append(self.word2id[word])

            word_vectors.append(np.asarray(vec))
        return word_vectors

    def get_word_list(self):
        return [x for x in self.id2word]

    def get_glove_embedding(self, embedding_file='data/glove.6B.300d.txt'):
        word_list = self.get_word_list()
        dictionary = read_glove(embedding_file)
        # model = Glove.load_stanford(embedding_file)
        embedding = 0.1 * np.random.randn(len(word_list) + 1, self.word_embedding_dim)
        found_words = 0
        for word in word_list:
            if word in dictionary:
                found_words += 1
                i = self.word2id[word]
                embedding[i, :] = dictionary[word]
        print "words found (%): ", float(found_words) / len(word_list)
        return embedding


def read_glove(file):
    """
    Reads the initial glove embedding
    :param file:
    :return:
    """

    word2vec = {}  # skip information on first line

    fin = open(file)
    for line in fin:
        items = line.replace('\r', '').replace('\n', '').split(' ')
        if len(items) < 10: continue
        word = items[0]
        vect = np.array([float(i) for i in items[1:]])
        word2vec[word] = vect

    return word2vec


def mask_document_text(texts, max_length=100):
    texts = np.asarray(texts)

    data = np.zeros((texts.shape[0], max_length), dtype='int32')
    mask = np.zeros((texts.shape[0], max_length), dtype='int32')
    for i, text in enumerate(texts):
        text_len = min(text.shape[0], max_length)
        data[i, :text_len] = text[:text_len]
        mask[i, :text_len] = 1

    return data, np.float32(mask)
