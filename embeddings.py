import torch
from tqdm import tqdm
import numpy as np

from data_utils_new import read_data


def load_word_embeddings(lang):
    embeddings_index = {}
    f = open('/home/halshater/Downloads/glove.6B/glove.6B.300d.txt')

    # i = 0
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        # i+=1
        # if i > 1000:
        #     break
    f.close()

    embedding_matrix = np.zeros((len(lang.word2index) + 2, 300), dtype=np.float32)
    for word in tqdm(lang.word2index.keys()):
        if word in embeddings_index:
            embedding_matrix[lang.word2index[word]] = embeddings_index[word]
        else:
            embedding_matrix[lang.word2index[word]] = embeddings_index["the"]

    return torch.FloatTensor(embedding_matrix)

if __name__ == "__main__":
    lang, pairs, labels = read_data()
    mat = load_word_embeddings(lang)
    print (mat.size())