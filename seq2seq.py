import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dimension, hidden_dimension): #defines components of the model
        # vocab_size = the number of possible words that the model can generate (tokens)
        # embedding_dimension = how many dimensions a single word vector will have
        # hidden_dimension = usually set to something in the range of 100, but a larger value could do better (but more processing time)
        super().__init__()
        self.hidden_dimension = hidden_dimension
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dimension)
        self.LSTM = nn.LSTM(embedding_dimension, hidden_dimension, batch_first=True)
        self.output_layer = nn.Linear(hidden_dimension, vocab_size)

    def forward(self, x, hidden): #defines how the components get stitched together in the model

        # This function needs to do the following computations
        # e = embed x using the embedding layer
        # lstm_out, new_hidden = passing e and hidden (if it is not None) through
        # the LSTM

        e = self.embedding_layer(x) #e = batch_size x context_size x embedding dimensions - a matrix with 50 rows, each row including the embedding vector for that character in the context

        if hidden is not None:
          lstm_out, new_hidden = self.LSTM(e, hidden) # lstm_out: batch_size x content_length x hidden_dimension
          # this basically generates matrices for every character in the context - so 50 matrices produced in parallel
        else:
          lstm_out, new_hidden = self.LSTM(e)

        # Now lstm_out is [batch_size, seq_len, hidden_dim]
        logits = self.output_layer(lstm_out)  # [batch_size, seq_len, vocab_size]

        # Flatten for loss: [batch_size * seq_len, vocab_size]
        logits = logits.view(-1, logits.size(-1))

        # this extracts the last LSTM hidden representation in case we input multiple tokens into the LSTM
        # lstm_out = lstm_out[:, -1, :]
        # lstm_out: batch_size x context_length x hidden --> batch_size x hidden

        # logits = passing the last LSTM hidden representation through the output layer
        # logits = self.output_layer(lstm_out) #logits: batch_size x n_characters

        return logits, new_hidden


class Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
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
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        # target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

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