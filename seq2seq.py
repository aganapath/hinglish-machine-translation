import torch
import torch.nn as nn
import random

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

        lstm_out, (new_hidden, cell) = self.LSTM(e)  # hidden

        return new_hidden, cell


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, hidden_size, num_layers, p):
        super().__init__() #Decoder, self
        self.target_vocab_size = target_vocab_size
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, x, hidden, cell):
        """
        x = single input token from target sentence
        hidden = last hidden state made by Encoder.forward
        cell = last cell made by Encoder.forward
        """
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        # possibly need to execute this before running output through self.fc (the linear layer)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__() # Seq2Seq, self
        self.encoder = encoder
        self.decoder = decoder
        # self.word_to_index = word_to_index

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = target.shape[1]
        target_len = target.shape[0]
        target_vocab_size = self.decoder.target_vocab_size

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        # source = source.permute(1, 0)  # [seq_len, batch_size] → [batch_size, seq_len]
        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

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
