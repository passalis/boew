import theano.gradient
import sklearn.cluster as cluster
import numpy as np
import theano
import theano.tensor as T
import lasagne
from tqdm import tqdm

floatX = theano.config.floatX

class BoEW():
    def __init__(self, embedding=None, eta_W=0.001, eta_emb=0.001, eta_V=0.001, eta_g=0.001,
                 n_codewords=32, g=0.1, m=0.1, n_classes=20, entropy_distance='cosine'):

        # Input variables
        self.input = T.imatrix('input')
        self.input_mask = T.fmatrix('input_mask')
        self.label_mapping = T.imatrix('mapping')
        self.emb = theano.shared(np.float32(embedding), 'embedding')

        self.nbof = BoEWFeatureLayer(n_codewords=n_codewords, feature_dimension=embedding.shape[1], g=g)
        self.feature_dim = n_codewords

        self.W = theano.shared(np.float32(np.ones((n_codewords,))), name='W')

        # Define the computational graph
        self.input_emb = self.emb[self.input, :]
        self.features = self.nbof.sym_histograms(self.input_emb, self.input_mask)
        self.features = self.features * self.W.reshape((1, -1))

        # Set the loss function
        self.entropy_loss = SoftEntropy(m=m, n_classes=n_classes, n_dim=self.feature_dim, distance=entropy_distance)
        loss = self.entropy_loss.sym_entropy(self.features, self.label_mapping)

        grad = T.grad(loss, self.emb)
        grad = T.switch(T.isnan(grad), 0, grad)
        updates = lasagne.updates.adam([grad], [self.emb], learning_rate=eta_emb)

        grad = T.grad(loss, self.nbof.V)
        grad = T.switch(T.isnan(grad), 0, grad)
        updates.update(lasagne.updates.adam([grad], [self.nbof.V], learning_rate=eta_V))

        grad = T.grad(loss, self.nbof.g)
        grad = T.switch(T.isnan(grad), 0, grad)
        updates.update(lasagne.updates.adam([grad], [self.nbof.g], learning_rate=eta_g))

        grad = T.grad(loss, self.W)
        grad = T.switch(T.isnan(grad), 0, grad)
        updates.update(lasagne.updates.adam([grad], [self.W], learning_rate=eta_W))

        self.train = theano.function(inputs=[self.input, self.input_mask, self.label_mapping], outputs=loss,
                                     updates=updates)
        updates_W = lasagne.updates.adam(loss, [self.W], learning_rate=eta_W)
        self.train_rf = theano.function(inputs=[self.input, self.input_mask, self.label_mapping], outputs=loss,
                                        updates=updates_W)
        self.encode_fn = theano.function(inputs=[self.input, self.input_mask], outputs=self.features)
        self.loss_fn = theano.function(inputs=[self.input, self.input_mask, self.label_mapping], outputs=loss)

    def init(self, data, mask, labels):
        """
        Initializes the BoEW model and the centers of the entropy
        :param data:
        :param mask:
        :param labels:
        :return:
        """
        self.nbof.initialize_dictionary(self.emb.get_value())
        S = self.encode(data, mask)
        self.entropy_loss.init_centers(S, labels)

    def fit(self, train_data, train_mask, train_labels, batch_size=64, n_epochs=10, rf=False):
        """
        Optimizes the representation using the proposed entropy-based algorithm
        :param train_data:
        :param train_mask:
        :param train_labels:
        :param batch_size:
        :param n_epochs:
        :param rf:
        :return:
        """
        idx = np.arange(train_labels.shape[0])
        n_batches = int(len(idx) / batch_size)
        loss_hist = np.zeros((n_epochs,))
        for j in tqdm(range(n_epochs)):
            np.random.shuffle(idx)
            loss = 0

            for i in (range(n_batches)):
                cur_idx = idx[i * batch_size:(i + 1) * batch_size]
                cur_data = train_data[cur_idx]
                cur_labels = train_labels[cur_idx]
                cur_mask = train_mask[cur_idx]
                cur_labels = get_class_mapping(cur_labels)

                if rf:
                    cur_loss = self.train_rf(cur_data, cur_mask, cur_labels)
                else:
                    cur_loss = self.train(cur_data, cur_mask, cur_labels)
                loss += cur_loss

            if n_batches * batch_size < len(idx):
                cur_idx = idx[n_batches * batch_size:]
                cur_data = train_data[cur_idx]
                cur_mask = train_mask[cur_idx]
                cur_labels = train_labels[cur_idx]
                cur_labels = get_class_mapping(cur_labels)

                loss += self.train(cur_data, cur_mask, cur_labels)
            loss_hist[j] = loss
        return loss_hist

    def encode(self, data, mask, batch_size=128):
        """
        Encodes the documents using the BoEW model
        :param data:
        :param mask:
        :param batch_size:
        :return:
        """
        features = np.zeros((len(data), self.feature_dim))
        n_batches = int(len(data) / batch_size)

        for i in tqdm(range(n_batches)):
            cur_data = data[i * batch_size:(i + 1) * batch_size]
            cur_mask = mask[i * batch_size:(i + 1) * batch_size]
            features[i * batch_size:(i + 1) * batch_size] = self.encode_fn(cur_data, cur_mask)
        if n_batches * batch_size < len(data):
            cur_data = data[n_batches * batch_size:]
            cur_mask = mask[n_batches * batch_size:]
            features[n_batches * batch_size:] = self.encode_fn(cur_data, cur_mask)

        return features


class BoEWFeatureLayer:
    """
    Defines a BoEW input layer
    """

    def __init__(self, g=0.1, feature_dimension=10, n_codewords=16):
        self.Nk = n_codewords
        self.D = feature_dimension
        self.g = theano.shared(value=np.float32(g), name='g')
        # RBF-centers / codewords
        V = np.random.rand(self.Nk, self.D).astype(dtype=floatX)
        self.V = theano.shared(value=V, name='V', borrow=True)
        # Input weights for the RBF neurons

        # Tensor of input objects (n_objects, n_features, self.D)
        self.X = T.tensor3(name='X', dtype=floatX)
        self.mask_X = T.imatrix(name='X_mask')
        self.mask_x = T.ivector(name='x_mask')

        # Feature matrix of an object (n_features, self.D)
        self.x = T.matrix(name='x', dtype=floatX)

        self.encode_objects_theano = theano.function(inputs=[self.X, self.mask_X],
                                                     outputs=self.sym_histograms(self.X, self.mask_X))
        self.encode_object_theano = theano.function(inputs=[self.x, self.mask_x],
                                                    outputs=self.sym_histogram(self.x, self.mask_x))

    def sym_histogram(self, X, mask=None):
        """
        Computes a soft-quantized histogram of a set of feature vectors (X is a matrix).
        :param X: matrix of feature vectors
        :return:
        """
        distances = euclidean_distance(X, self.V)
        membership = T.nnet.softmax(-distances / self.g ** 2)

        if mask is not None:
            histogram = membership * T.reshape(mask, (mask.shape[0], 1))
            histogram = T.sum(histogram, axis=0) / T.sum(mask, axis=0)
        else:
            histogram = T.mean(membership, axis=0)
        return histogram

    def sym_histograms(self, X, masks=None):
        """
        Encodes a set of objects (X is a tensor3)
        :param X: tensor3 containing the feature vectors for each object
        :return:
        """
        if masks is None:
            histograms, updates = theano.map(self.sym_histogram, sequences=(X,))
        else:
            histograms, updates = theano.map(self.sym_histogram, sequences=(X, masks))
        return histograms

    def initialize_dictionary(self, X, max_iter=100, redo=5):
        """
        Samples some feature vectors from X and learns an initial dictionary
        :param X: list of objects
        :param max_iter: maximum k-means iters
        :param redo: number of times to repeat k-means clustering
        :param n_samples: number of feature vectors to sample from the objects
        :param normalize: use l_2 norm normalization for the feature vectors
        """

        print "Clustering feature vectors..."
        features = np.float64(X)
        V = cluster.k_means(features, n_clusters=self.Nk, max_iter=max_iter, n_init=redo)
        self.V.set_value(np.asarray(V[0], dtype=theano.config.floatX))


class SoftEntropy:
    """
    Defines the symbolic calculation of the soft entropy
    """

    def __init__(self, m=0.1, n_classes=1, n_dim=1, distance='euclidean'):
        """
        Initializes the Soft Entropy Class
        """
        self.m = m
        self.distance = distance
        # Histograms
        self.S = T.fmatrix(name='S')
        self.mapping = T.imatrix(name='class_mapping')

        # Entropy centers
        self.C = theano.shared(np.asarray(np.zeros((n_classes, n_dim)), dtype=floatX), name='C')

        # Compile functions
        self.calculate_soft_entropy_fn = theano.function([self.S, self.mapping], self.sym_entropy(self.S, self.mapping))

    def init_centers(self, S, labels):
        """
        Gets the vectors and positions one center above each class
        """
        unique_labels = np.unique(labels)
        centers = None

        for label in unique_labels:
            idx = np.squeeze(labels == label)
            cur_S = S[idx, :]
            cur_center = np.mean(cur_S, axis=0)
            if centers is None:
                centers = cur_center
            else:
                centers = np.vstack((centers, cur_center))
        centers = np.asarray(centers, dtype=floatX)
        self.C.set_value(centers)

    def sym_entropy(self, S, mapping):
        """
        Defines the symbolic calculation of the soft entropy
        """
        if self.distance == 'euclidean':
            distances = euclidean_distance(S, self.C)
        else:
            distances = cosine_distance(S, self.C)
        Q = T.nnet.softmax(-distances / self.m)

        # Calculates the fuzzy membership vector for each histogram S
        # Q, scan_u = theano.map(fn=self.sym_get_similarity, sequences=[S])

        Nk = T.sum(Q, axis=0)

        H = T.dot(mapping.T, Q)
        P = H / Nk

        entropy_per_cluster = P * T.log2(P)
        entropy_per_cluster = T.switch(T.isnan(entropy_per_cluster), 0, entropy_per_cluster)
        entropy_per_cluster = entropy_per_cluster.sum(axis=0)

        Rk = Nk / Nk.sum()
        E = -(entropy_per_cluster * Rk).sum()
        return T.squeeze(E)

    def calculate_soft_entropy(self, data, labels):
        return self.calculate_soft_entropy_fn(data, get_class_mapping(labels))


def get_class_mapping(labels):
    """
    Returns the pi_ij matrix (1: if the i-th object belongs to class j), 0 otherwise)
    :param labels the labels
    :return the pi (label to instance mapping) matrix
    """
    unique = np.unique(labels)
    mapping = np.zeros((labels.shape[0], unique.shape[0]))
    for i in range(labels.shape[0]):
        idx = np.where(labels[i] == unique)
        mapping[i, idx] = 1

    return np.int32(mapping)


def euclidean_distance(A, B):
    """
    Defines the symbolic matrix that contains the distances between the vectors of A and B
    To accelerate the calculations it exploits the fact that (a-b)^2 = a^2 -2ab + b^2
    :param A: matrix of vectors
    :param B: matrix of vectors
    :return: the euclidean distances between A and B
    """
    aa = T.sum(A * A, axis=1)
    bb = T.sum(B * B, axis=1)
    AB = T.dot(A, T.transpose(B))

    AA = T.transpose(T.tile(aa, (bb.shape[0], 1)))
    BB = T.tile(bb, (aa.shape[0], 1))

    D = AA + BB - 2 * AB

    # Sanity checks
    D = T.maximum(D, 0)
    D = T.sqrt(D)
    return D


def cosine_distance(A, B):
    """
    Defines the symbolic matrix that contains the cosine distance between the vectors of A and B
    :param A: matrix of vectors
    :param B: matrix of vectors
    :return: the cosine distances between A and B
    """

    A = A / T.sqrt(T.sum(A ** 2, axis=1)).reshape((-1, 1))
    B = B / T.sqrt(T.sum(B ** 2, axis=1)).reshape((-1, 1))
    D = T.dot(A, T.transpose(B))

    return 1 - D
