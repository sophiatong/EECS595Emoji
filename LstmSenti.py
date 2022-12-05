import argparse
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import pyarrow as pa
import os
import random
import argparse
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)


def Load_parquet(folder_path, num_of_file=2):
  #folder_path: folder contains parquet files ex) "parquet/2013/1"
  #num_of_file: how many files you want to load from the path folder
  if not os.path.exists(folder_path):
    print("Path doesn't exist")
    return 
  
  for idx, filename in enumerate(os.listdir(folder_path)):
    if idx >= num_of_file:
      break
    Data = pq.read_table(folder_path + "/" + filename)
    Data = Data.to_pandas()
    
    if idx == 0:
      Full_DATA = Data
    else:
      Full_DATA = pd.concat([Full_DATA, Data], ignore_index=True)

  return Full_DATA

def create_emb_layer(weight_mat, device=None):
    weight_mat = torch.from_numpy(weight_mat)
    if device:
        weight_mat.to(device)
    num_embeddings, embedding_dim = weight_mat.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weight_mat})
    emb_layer.weight.requires_grad = False
    return emb_layer, embedding_dim

def Get_UNKA(words_by_sentence, frequency=2):
  #frequency : standard for replace word as UNKA
  word_freq = {}
  for sentence in words_by_sentence:
    for word in sentence:
      if (word in word_freq):
        word_freq[word] +=1
      else:
        word_freq[word] = 1
  
  unka_set = set()
  for sentence in words_by_sentence:
        for word in sentence:
            if (word_freq[word] <= frequency):
                unka_set.add(word)
  for idx, sentence in enumerate(words_by_sentence):
      words_by_sentence[idx] = ["UNKA" if word in unka_set else word for word in sentence]
  return words_by_sentence

def Create_Vocab(words_by_sentence):
  vocab_words_set = set()
  for sentence in words_by_sentence:
    vocab_words_set.update(set(sentence))
  vocab_word_list = sorted(vocab_words_set)
  vocab_word_list.insert(0, "<PAD>") # insert <PAD> at index 0 for padded data
  vocab_dict = dict(zip(vocab_word_list, range(len(vocab_word_list))))
  
  return vocab_dict, vocab_word_list

def Tokenize_with_Vocab(words_by_sentence, vocab_dict):
    for sentence in words_by_sentence:
        for i in range(len(sentence)):
            if vocab_dict.get(sentence[i]) == None:
              sentence[i] = vocab_dict.get("UNKA")
            else:
              sentence[i] = vocab_dict.get(sentence[i])



def Creat_Batch(words_by_sentence, Label,train_batch):
  n_sentences = len(words_by_sentence)
  if ((n_sentences % train_batch) == 0):
      n_batches = n_sentences / train_batch
  else:
      n_batches = (n_sentences // train_batch) # if left one is less than bathsize, drop it+ 1
  list_of_word_label_batches = [] # list of lists of lists
  for i in range(n_batches):
      word_label_batche = []
      for j in range(train_batch):
          sentence_idx = i * train_batch + j
          if sentence_idx < n_sentences:
              word_label_batche.append((words_by_sentence[sentence_idx], Label[sentence_idx]))
      list_of_word_label_batches.append(word_label_batche)
  return list_of_word_label_batches

def Padding_and_zip_with_label(words_by_sentence, Label):
  sentence_lengths = [len(sentence) for sentence in words_by_sentence]
  max_sentence_len = max(sentence_lengths)
  padded_word_mat = np.zeros((len(words_by_sentence), max_sentence_len))
  padded_label_mat = np.zeros(len(words_by_sentence))
  for i in range(len(words_by_sentence)):
      word_seq = words_by_sentence[i]
      padded_word_mat[i, 0:sentence_lengths[i]] = word_seq[:sentence_lengths[i]]
      label = Label[i]
      padded_label_mat[i] = label

  return (padded_word_mat, padded_label_mat), sentence_lengths

def Train_Batch_Padding(Train_batches, train_batch):
  list_of_lens_in_batch = []
  for i in range(len(Train_batches)):
      sentence_lengths = [len(data[0]) for data in Train_batches[i]]
      list_of_lens_in_batch.append(sentence_lengths)
      max_sentence_len = max(sentence_lengths)
      padded_word_batch = np.zeros((train_batch, max_sentence_len), dtype=int)
      padded_label_batch = np.zeros(train_batch, dtype=np.int)
      for j, sentence_len in enumerate(sentence_lengths):
          word_seq = Train_batches[i][j][0]
          padded_word_batch[j, 0:sentence_len] = word_seq[:sentence_len]
          label = Train_batches[i][j][1]
          padded_label_batch[j] = label
      Train_batches[i] = (padded_word_batch, padded_label_batch)
  return Train_batches, list_of_lens_in_batch

def Create_weightmat(pre_trained_embedding_file, vocab_word_list):
  if pre_trained_embedding_file == 'glove200d':
      # get glove embeddings from file and store in dict
      # "glove.6B/glove.6B.200d.txt"
      embedding_file = "glove.6B.200d.txt"
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
      embedding_file = "glove.twitter.27B.200d.txt"
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
      embedding_file = 'EmoTag-Vectors-620d/emotag-vectors.csv'
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
  return weight_mat

def Preprocess(Full_Data, test_ratio=0.2, evaluation_ratio=0.1, seed=595, train_batch=32, pre_trained_embedding_file='glove200d'):
  # Full_Data : Concatenated pandas data from parquet files
  # test_ratio : ratio of test set
  # evaluation_ratio : ratio of eval set
  random.seed(seed)
  Text = Full_Data['tweet_text'].to_list()
  random.shuffle(Text)  
  Label = Full_Data['sentiment'].to_list()
  random.shuffle(Label)

  N = len(Text)
  test_len = round(N * test_ratio)
  eval_len = round(N * evaluation_ratio)
  train_len = N - test_len - eval_len

  words_by_sentence = [sentence.split() for sentence in Text]
  words_by_sentence[:train_len] = Get_UNKA(words_by_sentence[:train_len])
  vocab_dict, vocab_word_list = Create_Vocab(words_by_sentence[:train_len])
  Tokenize_with_Vocab(words_by_sentence, vocab_dict)
  Train_batches = Creat_Batch(words_by_sentence[:train_len], Label[:train_len], train_batch)

  weight_mat = Create_weightmat(pre_trained_embedding_file, vocab_word_list)

  
  Train_set, list_of_lens_in_batch = Train_Batch_Padding(Train_batches, train_batch)
  Eval_set, Eval_len = Padding_and_zip_with_label(words_by_sentence[train_len: train_len+eval_len], Label[train_len: train_len+eval_len])
  Test_set, Test_len = Padding_and_zip_with_label(words_by_sentence[train_len+eval_len: N], Label[train_len+eval_len: N])

  return Train_set, Eval_set, Test_set, vocab_dict, weight_mat, list_of_lens_in_batch, Eval_len, Test_len



class RNNSentiment(nn.Module):
    def __init__(self, hidden_dim, target_size, weight_mat):
        super(RNNSentiment, self).__init__()
        ############################################
        # TODO: Add pytorch cuda device to use gpu #
        ############################################
        self.cuda0 = torch.device('cuda:0')
        self.target_size = target_size
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
        self.hidden2tag = nn.Linear(hidden_dim, target_size)
        self.softmax = nn.Softmax(dim=-1)
        


    def forward(self, sentence, lens_in_batch):
        sentence = torch.LongTensor(sentence)
        # print('sentence shape')
        # print(sentence.shape)
        ############################################
        # TODO: Add pytorch cuda device to use gpu #
        ############################################
        sentence.to(self.cuda0)
        ############################################
        # TODO END                                 #
        ############################################
        # print('embedding shape')
        # print(self.embedding(sentence).shape)
        sentence = torch.nn.utils.rnn.pack_padded_sequence(
            self.embedding(sentence),
            lens_in_batch,
            batch_first = True,
            enforce_sorted = False
        )
        
        lstm_out, _ = self.lstm(sentence)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        label_space = self.dropout(lstm_out)
        # print('after lstm')
        # print(label_space.shape)
        label_space = label_space.view(label_space.shape[0], -1)
        # print('after view')
        # print(label_space.shape)
        fc1 = nn.Linear(label_space.shape[-1], self.target_size)
        label_space = fc1(label_space)#self.hidden2tag(label_space)
        # print('after linear')
        # print(label_space.shape)
      
        label_scores = self.softmax(label_space)
        return label_scores

def train(Train_set, Eval_set, HIDDEN_DIM, TAGSET_SIZE, weight_mat, list_of_lens_in_batch, Eval_len, learning_rate = 0.01, n_epochs = 30):
  cuda0 = torch.device('cuda:0') # .to(cuda0)
  # train model
  
  model = RNNSentiment(HIDDEN_DIM, TAGSET_SIZE, weight_mat)
  criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
  optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
  print('##############################')
  print('Train Model')
  print('##############################')
  
  for epoch in tqdm(range(n_epochs)):
      print(f'\nepoch: {epoch+1}')
      
      for batch_no in tqdm(range(len(Train_set))):

          padded_word_batch = Train_set[batch_no][0]
          padded_label_batch = Train_set[batch_no][1]
          lens_in_batch = list_of_lens_in_batch[batch_no]
          optimiser.zero_grad()
          predictions = model(padded_word_batch, lens_in_batch)
          # print(predictions.shape)
          # print(padded_label_batch.shape)
          predictions = predictions.view(-1, predictions.shape[-1])
          padded_label_batch = torch.LongTensor(padded_label_batch).view(-1)
          ############################################
          # TODO: Add pytorch cuda device to use gpu #
          ############################################
          padded_label_batch.to(cuda0)
          ############################################
          # TODO END                                 #
          ############################################
          loss = criterion(predictions, padded_label_batch)
          loss.backward()
          optimiser.step()
      print("FINISHED EPOCH: " + str(epoch + 1))
      print(f'train_loss: {loss}')
      
      with torch.no_grad():
        # length = torch.tensor([len(Eval_set[0])], dtype=torch.int64)
        eval_pred = model(Eval_set[0], Eval_len)
        eval_pred = eval_pred.view(-1, eval_pred.shape[-1])
        labels = Eval_set[1]
        labels = torch.LongTensor(labels).view(-1)
        labels.to(cuda0)
        eval_loss = criterion(eval_pred, labels)
        print(f'\nEvaluation LOSS: {eval_loss}')
        eval_pred = torch.argmax(eval_pred, -1).cpu()
        score = accuracy_score(eval_pred, labels)
        print(f"Evaluation ACC: {100*score:6.2f}%")

  # Your code ends here

  return model

def test(Test_set, HIDDEN_DIM, TAGSET_SIZE, weight_mat, Test_len, model_file):
  cuda0 = torch.device('cuda:0')
  criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
  with torch.no_grad():
      model = RNNSentiment(HIDDEN_DIM, TAGSET_SIZE, weight_mat)
      model.load_state_dict(torch.load(model_file))
      # length = torch.tensor([len(Eval_set[0])], dtype=torch.int64)
      test_pred = model(Test_set[0], Test_len)
      test_pred = test_pred.view(-1, test_pred.shape[-1])
      labels = Test_set[1]
      labels = torch.LongTensor(labels).view(-1)
      labels.to(cuda0)
      test_loss = criterion(test_pred, labels)
      print(f'\nTEST LOSS: {test_loss}')
      test_pred = torch.argmax(test_pred, -1).cpu()
      score = accuracy_score(test_pred, labels)
      print(f"TEST ACC: {100*score:6.2f}%")

def main(params):
    #'glove200d', 'glovetwit200d', 'Emotag'
    BATCH_SIZE = params.batch #32
    lr = params.lr
    epochs = params.epochs
    num_of_file = params.num_of_file
    HIDDEN_DIM = 128
    TAGSET_SIZE = 2 #46
    torch.manual_seed(0)
    model_file = params.model_file
    pre_trained_embedding_file = params.pre_trained_embedding_file
    Full_Data = Load_parquet(params.training_file, num_of_file=num_of_file)
    Train_set, Eval_set, Test_set, vocab_dict, weight_mat, list_of_lens_in_batch, Eval_len, Test_len = Preprocess(Full_Data, test_ratio=0.2, evaluation_ratio=0.1, seed=595, train_batch=BATCH_SIZE, pre_trained_embedding_file=pre_trained_embedding_file)
    
    model = train(Train_set, Eval_set, HIDDEN_DIM, TAGSET_SIZE, weight_mat, list_of_lens_in_batch, Eval_len, learning_rate=lr, n_epochs=epochs)
    torch.save(model.state_dict(), model_file)

    test(Test_set, HIDDEN_DIM, TAGSET_SIZE, weight_mat, Test_len, model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    #parser.add_argument("--train", action="store_const", const=True, default=False)
    parser.add_argument("--model_file", type=str, default="model.torch")
    parser.add_argument("--training_file", type=str, default="")
    # parser.add_argument("--data_file", type=str, default="")
    # parser.add_argument("--label_file", type=str, default="")
    parser.add_argument("--pre_trained_embedding_file", type=str, default="glove200d")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_of_file", type=int, default=2)

    main(parser.parse_args())