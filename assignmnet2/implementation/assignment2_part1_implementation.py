import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle

words = pickle.load(open(f"../dataset/words.pkl", "rb"))
word2idx = pickle.load(open(f"../dataset/word2idx.pkl", "rb"))
vectors = pickle.load(open(f"../dataset/vectors.pkl", "rb"))
train_vocab_list = pickle.load(open(f"../dataset/train_vocab_list.pkl", "rb"))
train_target_list = pickle.load(open(f"../dataset/train_target_list.pkl", "rb"))
test_vocab_list = pickle.load(open(f"../dataset/test_vocab_list.pkl", "rb"))
test_target_list = pickle.load(open(f"../dataset/test_target_list.pkl", "rb"))


def get_weights_matrix(target_vocab):
    target_vocab_list = target_vocab.split(" ")
    weights_matrix = np.zeros((1, 50))
    words_found = 0

    for i, word in enumerate(target_vocab_list):
        try:
            weights_matrix += vectors[word2idx[word]]
            words_found += 1
        except KeyError:
            weights_matrix += np.random.normal(scale=0.6, size=(50,))

    return torch.from_numpy(weights_matrix).unsqueeze(0)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, hidden = self.rnn(x)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

    def predict(self, x):
        prediction = F.softmax(self.forward(x), dim=1).data[0][0]
        return 1 if prediction > 0.5 else 0


n_epochs = 3
train_len = 67349
test_len = 1821
train_predictions = np.zeros([n_epochs, train_len])
train_losses = np.zeros(n_epochs)
test_predictions = np.zeros([n_epochs, test_len])
test_losses = np.zeros(n_epochs)

model = RNN(50, 300, 2).double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if __name__ == "__main__":
    for epoch in range(n_epochs):
        for i in range(train_len):
            vocab, target = train_vocab_list[i], train_target_list[i]
            inputs = get_weights_matrix(vocab)
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            train_predictions[epoch][i] = int(predicted)
            loss = criterion(output, torch.tensor([int(target)], dtype=int))
            train_losses[epoch] += loss.item()
            loss.backward()
            optimizer.step()
        for i in range(test_len):
            vocab, target = test_vocab_list[i], test_target_list[i]
            inputs = get_weights_matrix(vocab)
            predicted = model.predict(inputs)
            test_predictions[epoch][i] = int(predicted)
            loss = criterion(output, torch.tensor([int(target)], dtype=int))
            test_losses[epoch] += loss.item()
        print(train_losses[epoch])
        print(test_losses[epoch])

    torch.save(model, "../trained_models/part1_state.chkpt")
