import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, source_vocab_size, embedding_dimension, hidden_dimension, num_layers, p):
        """
        vocab_size = the number of unique tokens in our source text
        embedding_dimension = how how many dimensions a single word vector will have
        hidden_dimension = usually set to something in the range of 100, but a larger value could do better (but more processing time)
        n_layers is the number of layers in the RNN.
        dropout is the amount of dropout to use. This is a regularization parameter to prevent overfitting. Check out this for more details about dropout.
        """
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_dimension = hidden_dimension
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(source_vocab_size, embedding_dimension)
        self.LSTM = nn.LSTM(embedding_dimension, hidden_dimension, num_layers, batch_first=True)  #batch_first=True
        self.output_layer = nn.Linear(hidden_dimension, source_vocab_size)

    def forward(self, x):  #defines how the components get stitched together in the model
        "outputs the final hidden state and cell that combine to create the context vector used in the decoder model"
        # This function needs to do the following computations
        # e = embed x using the embedding layer
        # lstm_out, new_hidden = passing e and hidden (if it is not None) through the LSTM

        x_transp = torch.transpose(x,0,1)
        # x_transp = sequence length x batch size
        e = self.embedding_layer(x_transp)
        # e = sequence length x batch_size x embedding dimensions

        encoder_out, (new_hidden, cell) = self.LSTM(e)  # hidden

        return encoder_out, new_hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):

        # query = query.repeat(1, keys.size(1), 1)  # now shape [batch, seq_len, hidden_dim]
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # [batch, seq_len, 1]
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, hidden_size, num_layers, p):
        super().__init__() #Decoder, self
        self.target_vocab_size = target_vocab_size
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, encoder_out, x, hidden, cell):
        x = x.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(x))  # [1, batch_size, embedding_size]

        # Attention uses last hidden state and encoder output
        context, _ = self.attention(hidden[-1].unsqueeze(1), encoder_out)  # [batch_size, 1, hidden_size]
        context = context.permute(1, 0, 2)  # [1, batch_size, hidden_size]

        # Combine context + embedded input
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, embedding_size + hidden_size]

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [1, batch, hidden]
        predictions = self.fc(outputs.squeeze(0))  # [batch, vocab_size]

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__() # Seq2Seq, self
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # self.word_to_index = word_to_index

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.target_vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        # source = source.permute(1, 0)  # [seq_len, batch_size] → [batch_size, seq_len]
        encoder_out, hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(encoder_out, x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
