import cPickle as pickle
import os.path
from os.path import exists, join
import numpy as np

from word_vectorizer import WordVectorizer

text_datasets_path = 'data'


def load_20ng():
    """
    Loads the preprocessed 20NG data
    :return:
    """
    path_20ng = join(text_datasets_path, '20ng.pickle')
    if not exists(path_20ng):
        transform_20ng_dataset(path_20ng)

    with open(path_20ng, "rb") as f:
        data = pickle.load(f)

    return data['train_data'], data['train_labels'], data['test_data'], data['test_labels'], data['embedding']


def transform_20ng_dataset(ouput_file):
    """
    Preprocess the 20NG dataset and stores the preprocessed data
    :param ouput_file:
    :return:
    """

    train_data, train_labels = get_20ng(split='train')
    test_data, test_labels = get_20ng(split='test')

    wv = WordVectorizer()
    wv.fit(train_data)
    glove_embedding = wv.get_glove_embedding()

    train_data = wv.transform(train_data)
    test_data = wv.transform(test_data)
    data = {}
    data['train_data'] = train_data
    data['test_data'] = test_data
    data['train_labels'] = train_labels
    data['test_labels'] = test_labels
    data['word2id'] = wv.word2id
    data['id2word'] = wv.id2word
    data['embedding'] = glove_embedding

    with open(ouput_file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_20ng(split='train'):
    """
    Loads the raw 20ng data
    :param split:
    :return:
    """

    mapping = {
        'alt.atheism': 0, 'comp.graphics': 1, 'comp.os.ms-windows.misc': 2, 'comp.sys.ibm.pc.hardware': 3,
        'comp.sys.mac.hardware': 4, 'comp.windows.x': 5, 'misc.forsale': 6, 'rec.autos': 7, 'rec.motorcycles': 8,
        'rec.sport.baseball': 9, 'rec.sport.hockey': 10, 'sci.crypt': 11, 'sci.electronics': 12, 'sci.med': 13,
        'sci.space': 14, 'soc.religion.christian': 15, 'talk.politics.guns': 16, 'talk.politics.mideast': 17,
        'talk.politics.misc': 18,
        'talk.religion.misc': 19}
    data, labels = [], []

    if split == 'test':
        filename = os.path.join(text_datasets_path, '20ng-test-all-terms.txt')
    else:
        filename = os.path.join(text_datasets_path, '20ng-train-all-terms.txt')

    with open(filename) as f:
        for line in f:
            cur = line.split('\t')
            labels.append(mapping[cur[0]])
            data.append(unicode(cur[1].decode('utf-8')))
    return data, np.asarray(labels)
