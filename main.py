from time import time
import numpy as np
import datasets
from boew import BoEW, get_class_mapping
from database_evaluation import Database
from word_vectorizer import mask_document_text


def evaluate_bof_family(emb, train_data, train_mask, train_labels, test_data, test_mask, test_labels,
                        entropy_distance='cosine', retrieval_distance='cosine', n_codewords=16, g=0.1, m=0.1,
                        iters=10, batch_size=100, eta=0.01, eta_g=0.001):
    np.random.seed(1)
    print "Distance metric:", retrieval_distance
    n_classes = np.unique(train_labels).shape[0]
    model = BoEW(embedding=emb, eta_W=eta, eta_V=eta, eta_emb=eta, eta_g=eta_g, g=g, m=m,
                 n_codewords=n_codewords, entropy_distance=entropy_distance, n_classes=n_classes)

    # Initialize the dictionary and the entropy centers
    model.init(train_data, train_mask, train_labels)

    # Create the database and evaluate the initial performance
    database = Database(model.encode(train_data, train_mask), train_labels, metric=retrieval_distance)
    train_res = database.evaluate(model.encode(train_data, train_mask), train_labels)
    test_encoded = model.encode(test_data, test_mask)
    test_res = database.evaluate(test_encoded, test_labels)
    print "BoEW (Initial)"
    print "mAP TRAIN = ", train_res.get_metrics()['map_mean']
    print "mAP TEST = ", test_res.get_metrics()['map_mean']

    # Optimize the model
    model.fit(train_data, train_mask, train_labels, n_epochs=iters, batch_size=batch_size)

    # Re-evaluate the model
    database = Database(model.encode(train_data, train_mask), train_labels, metric=retrieval_distance)
    train_res = database.evaluate(model.encode(train_data, train_mask), train_labels)
    test_encoded = model.encode(test_data, test_mask)
    test_res = database.evaluate(test_encoded, test_labels)
    print "BoEW (Optimized)"
    print "mAP TRAIN = ", train_res.get_metrics()['map_mean']
    print "mAP TEST = ", test_res.get_metrics()['map_mean']


def run_test():
    print "Loading dataset"
    train_data, train_labels, test_data, test_labels, emb = datasets.load_20ng()
    train_labels = np.int32(train_labels)
    test_labels = np.int32(test_labels)
    np.random.seed(1)

    max_len = 500
    train_data, train_mask = mask_document_text(train_data, max_length=max_len)
    test_data, test_mask = mask_document_text(test_data, max_length=max_len)

    evaluate_bof_family(emb, train_data, train_mask, train_labels, test_data, test_mask, test_labels,
                        entropy_distance='cosine',
                        retrieval_distance='cosine', n_codewords=64, g=0.5, m=0.005, iters=10, batch_size=100)

    evaluate_bof_family(emb, train_data, train_mask, train_labels, test_data, test_mask, test_labels,
                        entropy_distance='euclidean',
                        retrieval_distance='euclidean', n_codewords=64, g=0.5, m=0.01, iters=10, batch_size=100)


run_test()
