import argparse
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 100
HIDDEN_DIM = 128
TAGSET_SIZE = 46
torch.manual_seed(0)

class RNNTagger(nn.Module):
    def __init__(self, hidden_dim, tagset_size, weight_mat):
        super(RNNTagger, self).__init__()
        ############################################
        # TODO: Add pytorch cuda device to use gpu #
        ############################################
        self.cuda0 = torch.device('cuda:0')
        ############################################
        # TODO END                                 #
        ############################################
        self.hidden_dim = hidden_dim
        self.embedding, embedding_dim = create_emb_layer(weight_mat, self.cuda0)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=-1)
        


    def forward(self, sentence, lens_in_batch):
        sentence = torch.LongTensor(sentence)
        ############################################
        # TODO: Add pytorch cuda device to use gpu #
        ############################################
        sentence.to(self.cuda0)
        ############################################
        # TODO END                                 #
        ############################################
        sentence = torch.nn.utils.rnn.pack_padded_sequence(
            self.embedding(sentence),
            lens_in_batch,
            batch_first = True,
            enforce_sorted = False
        )

        lstm_out, _ = self.lstm(sentence)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        tag_space = self.dropout(lstm_out)
        tag_space = self.hidden2tag(tag_space)
        tag_scores = self.softmax(tag_space)
        return tag_scores
    

def create_emb_layer(weight_mat, device=None):
    weight_mat = torch.from_numpy(weight_mat)
    if device:
        weight_mat.to(device)
    num_embeddings, embedding_dim = weight_mat.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weight_mat})
    emb_layer.weight.requires_grad = False
    return emb_layer, embedding_dim

def train(training_file, pre_trained_embedding_file):
    assert os.path.isfile(training_file), "Training file does not exist"

    # Your code starts here

    # load data and store words and tags 
    train_txt = open(training_file).readlines()
    words_by_sentence = []
    tags_by_sentence = []
    for sentence in train_txt:
        str_list = sentence.split()
        word_list = str_list[0::2]
        tag_list = str_list[1::2]
        words_by_sentence.append(word_list)
        tags_by_sentence.append(tag_list)

    # get counts of each word in dic
    word_freq = {}
    print('##############################')
    print('get counts of each word in dic')
    print('##############################')
    for sentence in tqdm(words_by_sentence):
        for word in sentence:
            if (word in word_freq):
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    
    # get UNKA words and replace them in words_by_sentence 
    unka_set = set()
    print('##############################')
    print('get UNKA words and replace them in words_by_sentence')
    print('##############################')
    for sentence in tqdm(words_by_sentence):
        for word in sentence:
            if (word_freq[word] <= 2):
                unka_set.add(word)
    for idx, sentence in enumerate(words_by_sentence):
        words_by_sentence[idx] = ["UNKA" if word in unka_set else word for word in sentence]

    # get unique tags and place in dict with index as values
    tags_set = set()
    print('##############################')
    print('get unique tags and place in dict with index as values')
    print('##############################')
    for sentence_tags in tqdm(tags_by_sentence):
        tags_set.update(set(sentence_tags))
    tags_list = sorted(list(tags_set))
    tags_list.insert(0, "<PAD>") # insert <PAD> at index 0 for padded data
    tags_dict = dict(zip(tags_list, range(len(tags_list))))
    
    # create vocabulary dictionary with index as values
    vocab_words_set = set()
    print('##############################')
    print('create vocabulary dictionary with index as values')
    print('##############################')
    for sentence in tqdm(words_by_sentence):
        vocab_words_set.update(set(sentence))
    vocab_word_list = sorted(list(vocab_words_set))
    vocab_word_list.insert(0, "<PAD>") # insert <PAD> at index 0 for padded data
    vocab_dict = dict(zip(vocab_word_list, range(len(vocab_word_list))))

    # convert words and tags by sentence to their index
    print('##############################')
    print('convert words and tags by sentence to their index')
    print('##############################')
    for sentence in tqdm(words_by_sentence):
        for i in range(len(sentence)):
            sentence[i] = vocab_dict.get(sentence[i])
    for sentence in tqdm(tags_by_sentence):
        for i in range(len(sentence)):
            sentence[i] = tags_dict.get(sentence[i])

    # create batches for words and tags
    n_sentences = len(words_by_sentence)
    if ((n_sentences % BATCH_SIZE) == 0):
        n_batches = n_sentences / BATCH_SIZE
    else:
        n_batches = (n_sentences // BATCH_SIZE) + 1
    list_of_word_batches = [] # list of lists of lists
    list_of_tag_batches = []
    print('##############################')
    print('create batches for words and tags')
    print('##############################')
    for i in tqdm(range(n_batches)):
        word_batch = []
        tag_batch = []
        for j in range(BATCH_SIZE):
            sentence_idx = i * BATCH_SIZE + j
            if sentence_idx < n_sentences:
                word_batch.append(words_by_sentence[sentence_idx])
                tag_batch.append(tags_by_sentence[sentence_idx])
        list_of_word_batches.append(word_batch)
        list_of_tag_batches.append(tag_batch)

    # pad each batch for words and tags
    list_of_lens_in_batch = []
    print('##############################')
    print('pad each batch for words and tags')
    print('##############################')
    for i in tqdm(range(len(list_of_word_batches))):
        sentence_lengths = [len(sentence) for sentence in list_of_word_batches[i]]
        list_of_lens_in_batch.append(sentence_lengths)
        max_sentence_len = max(sentence_lengths)
        padded_word_batch = np.zeros((BATCH_SIZE, max_sentence_len), dtype=int)
        padded_tag_batch = padded_word_batch.copy()
        for j, sentence_len in enumerate(sentence_lengths):
            word_seq = list_of_word_batches[i][j]
            padded_word_batch[j, 0:sentence_len] = word_seq[:sentence_len]
            tag_seq = list_of_tag_batches[i][j]
            padded_tag_batch[j, 0:sentence_len] = tag_seq[:sentence_len]
        padded_tag_batch = [i for i in padded_tag_batch if any(i)]
        list_of_word_batches[i] = padded_word_batch
        list_of_tag_batches[i] = padded_tag_batch
    ##################################################
    #   TODO : Load Glovetwitt, EmoTag embedding     #                 
    ##################################################
    print('##############################')
    print('Load Pre-trained Embedding')
    print('##############################')
    if pre_trained_embedding_file == 'glove200d':
        # get glove embeddings from file and store in dict
        # "glove.6B/glove.6B.200d.txt"
        embedding_file = "Project/glove.6B.200d.txt"
        Loaded_Emb_dict = {}
        with open(embedding_file) as e:
            x = e.readlines()
            for sentence in tqdm(x):
                embedding_list = sentence.split()
                target_word = embedding_list[0]
                embeddings = embedding_list[1:]
                embeddings = [float(val) for val in embeddings]
                Loaded_Emb_dict[target_word] = embeddings

    elif pre_trained_embedding_file == 'glovetwit200d':
        # get glove embeddings from file and store in dict
        # "glove.twitter.27B.200d.txt"
        embedding_file = "Project/glove.twitter.27B.200d.txt"
        Loaded_Emb_dict = {}
        with open(embedding_file) as e:
            x = e.readlines()
            for sentence in tqdm(x):
                embedding_list = sentence.split()
                target_word = embedding_list[0]
                embeddings = embedding_list[1:]
                embeddings = [float(val) for val in embeddings]
                Loaded_Emb_dict[target_word] = embeddings
    
    elif pre_trained_embedding_file == 'Emotag':
        # get EmoTag embbeddings from tile and stor in dict
        # 'EmoTag-Vectors-620d/emotag-vectors.csv'
        embedding_file = 'Project/EmoTag-Vectors-620d/emotag-vectors.csv'
        Loaded_Emb_dict = {}
        with open(embedding_file, 'r') as e:
            x = e.readlines()
            for sentence in tqdm(x):
                embedding_list = sentence.split()
                embedding_list[-1] = embedding_list[-1].replace('\n','')
                target_word = embedding_list[0]
                embeddings = embedding_list[1:]
                embeddings = [float(val) for val in embeddings]
                Loaded_Emb_dict[target_word] = embeddings
    else:
        print("wrong emb file name chosse one name between 'glove200d', 'glovetwit200d', 'Emotag'")
    ##################################################
    #                 END of TODO                    #                 
    ##################################################

    # create weight matrix of pre-trained embeddings
    print('##############################')
    print('create weight matrix of pre-trained embeddings')
    print('##############################')
    mat_len = len(vocab_word_list)
    embedding_dim = len(embedding_list) - 1
    print(f"Embedding dimension : {embedding_dim}")

    weight_mat = np.zeros((mat_len, embedding_dim))
    for i, word in tqdm(enumerate(vocab_word_list)):
        if (i > 0): # leave the 0th row as 0s for <PAD> embedding
            try:
                weight_mat[i] = Loaded_Emb_dict[word]
            except KeyError:
                # use random values from a normal dist for words not in glove
                weight_mat[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

    # save dictionaries for vocab and tags and weight matrix for testing
    with open('vocab_tag_weight.pickle', 'wb') as handle:
        pickle.dump([vocab_dict, tags_dict, weight_mat], handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('vocab_tag_weight saved')
    cuda0 = torch.device('cuda:0') # .to(cuda0)
    # train model
    learning_rate = 0.01
    n_epochs = 30
    model = RNNTagger(HIDDEN_DIM, TAGSET_SIZE, weight_mat)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print('##############################')
    print('Train Model')
    print('##############################')
    for epoch in tqdm(range(n_epochs)):
        for batch_no in range(len(list_of_word_batches)):
            padded_word_batch = list_of_word_batches[batch_no]
            padded_tag_batch = list_of_tag_batches[batch_no]
            lens_in_batch = list_of_lens_in_batch[batch_no]
            optimiser.zero_grad()
            predictions = model(padded_word_batch, lens_in_batch)
            predictions = predictions.view(-1, predictions.shape[-1])
            padded_tag_batch = torch.LongTensor(padded_tag_batch).view(-1)
            ############################################
            # TODO: Add pytorch cuda device to use gpu #
            ############################################
            padded_tag_batch.to(cuda0)
            ############################################
            # TODO END                                 #
            ############################################
            loss = criterion(predictions, padded_tag_batch)
            loss.backward()
            optimiser.step()
        print("FINISHED EPOCH: " + str(epoch + 1))

    # Your code ends here

    return model
    

def test(model_file, data_file, label_file):
    assert os.path.isfile(model_file), "Model file does not exist"
    assert os.path.isfile(data_file), "Data file does not exist"
    assert os.path.isfile(label_file), "Label file does not exist"

    # Your code starts here
    # load test data
    test_txt = open(data_file).readlines()
    words_by_sentence = []
    for sentence in test_txt:
        word_list = sentence.split()
        words_by_sentence.append(word_list)
    # load truth
    truth_txt = open(label_file).readlines()
    tags_by_sentence = []
    for sentence in truth_txt:
        str_list = sentence.split()
        tag_list = str_list[1::2]
        tags_by_sentence.append(tag_list)

    # load dictionaries from training of vocab and tag indexes
    with open('vocab_tag_weight.pickle', 'rb') as handle:
        tmp = pickle.load(handle)
    vocab_dict = tmp[0]
    tags_dict = tmp[1]
    weight_mat = tmp[2]

    # convert words and tags by sentence to their index
    for sentence in words_by_sentence:
        for i in range(len(sentence)):
            sentence[i] = vocab_dict.get(sentence[i])
    for sentence in tags_by_sentence:
        for i in range(len(sentence)):
            sentence[i] = tags_dict.get(sentence[i])

    # pad words and tags
    sentence_lengths = [len(sentence) for sentence in words_by_sentence]
    max_sentence_len = max(sentence_lengths)
    padded_word_mat = np.zeros((len(words_by_sentence), max_sentence_len))
    padded_tag_mat = padded_word_mat.copy()
    for i in range(len(words_by_sentence)):
        word_seq = words_by_sentence[i]
        tag_seq = tags_by_sentence[i]
        padded_word_mat[i, 0:sentence_lengths[i]] = word_seq[:sentence_lengths[i]]
        padded_tag_mat[i, 0:sentence_lengths[i]] = tag_seq[:sentence_lengths[i]]

    # get predictions from loaded model
    model = RNNTagger(HIDDEN_DIM, TAGSET_SIZE, weight_mat)
    model.load_state_dict(torch.load(model_file))
    
    prediction = model(padded_word_mat, sentence_lengths)
    prediction = prediction.view(-1, prediction.shape[-1])
    prediction = torch.argmax(prediction, -1).cpu()
    prediction = prediction.numpy()
    padded_tags = torch.LongTensor(padded_tag_mat)
    padded_tags = padded_tags.view(-1)
    padded_tags = padded_tags.numpy()
    non_pad_elems = (padded_tags != 0).nonzero()
    non_pad_idx = non_pad_elems[0]
    prediction = prediction[non_pad_idx]
    ground_truth = padded_tags[non_pad_idx]
    
    # Your code ends here

    score = accuracy_score(prediction, ground_truth)
    print(f"The accuracy of the model is {100*score:6.2f}%")


def main(params):
    # if params.train:
    #     model = train(params.training_file, params.pre_trained_embedding_file)
    #     torch.save(model.state_dict(), params.model_file)
    # else:
    #     test(params.model_file, params.data_file, params.label_file)

    model = train(params.training_file, params.pre_trained_embedding_file)
    torch.save(model.state_dict(), params.model_file)
    test(params.model_file, params.data_file, params.label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action="store_const", const=True, default=False)
    parser.add_argument("--model_file", type=str, default="model.torch")
    parser.add_argument("--training_file", type=str, default="")
    parser.add_argument("--data_file", type=str, default="")
    parser.add_argument("--label_file", type=str, default="")
    parser.add_argument("--pre_trained_embedding_file", type=str, default="glove200d")

    main(parser.parse_args())
