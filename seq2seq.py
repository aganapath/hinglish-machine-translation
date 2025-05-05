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
        self.LSTM = nn.LSTM(embedding_dimension, hidden_dimension, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dimension, source_vocab_size)

    def forward(self, x):  #defines how the components get stitched together in the model
        "outputs the final hidden state and cell that combine to create the context vector used in the decoder model"
        # This function needs to do the following computations
        # e = embed x using the embedding layer
        # lstm_out, new_hidden = passing e and hidden (if it is not None) through the LSTM

        # x_transp = torch.transpose(x,0,1)
        # x_transp = sequence length x batch size
        e = self.embedding_layer(x)
        # print("embedding shape", e.shape)
        # e = batch_size x sequence length x embedding dimensions

        encoder_out, (new_hidden, cell) = self.LSTM(e)

        return encoder_out, new_hidden, cell

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, device):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        self.device = device

    def forward(self, query, keys):

        # query = query.repeat(1, keys.size(1), 1)  # now shape [batch, seq_len, hidden_dim]
        query = query.to(self.device).float()
        keys = keys.to(self.device).float()

        # print(f"query shape: {query.shape}")
        # print(f"keys shape: {keys.shape}")

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  # [batch, seq_len, 1]
        scores = scores.squeeze(2).unsqueeze(1) #scores = [batch, 1, seq_len]
        # print("scores shape", scores.shape)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        # print("weights shape", weights.shape)
        # print("context shape", context.shape)

        return context, weights

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embedding_size, hidden_size, num_layers, p, device):
        super().__init__() #Decoder, self
        self.target_vocab_size = target_vocab_size
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.device = device
        self.attention = BahdanauAttention(hidden_size, device).to(device)
        self.rnn = nn.LSTM(embedding_size + hidden_size, hidden_size, num_layers, dropout=p, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, encoder_out, x, hidden, cell):
        # print(f"x shape: {x.shape}")
        x = x.unsqueeze(1)  # [1, batch_size]
        # print(f"x unsqueezed shape: {x.shape}")
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_size]

        # Attention uses last hidden state and encoder output
        context, _ = self.attention(hidden[-1].unsqueeze(1), encoder_out)  # [batch_size, 1, hidden_size]

        # Combine context + embedded input
        # print("embedded shape", embedded.shape)
        # print("context shape", context.shape)
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, embedding_size + hidden_size]

        # print("rnn shape", rnn_input.shape)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))  # output: [1, batch, hidden]
        predictions = self.fc(outputs.squeeze(1))  # [batch, vocab_size]

        # print("outputs shape", outputs.shape)
        # print("hidden shape", hidden.shape)
        # print("cell shape", cell.shape)
        # print("predictions shape", predictions.shape)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__() # Seq2Seq, self
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        # print("source shape", source.shape)
        # print("target shape", target.shape)
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.target_vocab_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        # print("outputs shape", outputs.shape)

        # source = source.permute(1, 0)  # [seq_len, batch_size] → [batch_size, seq_len]
        encoder_out, hidden, cell = self.encoder(source)
        # print("encoder out shape", encoder_out.shape)
        # print("encoder final hidden state shape", hidden.shape)

        # Grab the first input to the Decoder which will be <SOS> token
        x = source[:, 0]
        # print("x shape", x.shape)

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(encoder_out, x, hidden, cell)

            # Store next output prediction
            outputs[:, t, :] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[:, t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def translate(self, src_batch, sos_idx):
        # set model to evaluation mode to ensure deterministic output + no dropout
        self.eval()
        src_batch = src_batch.to(self.device)
        batch_size = src_batch.shape[0]
        max_length = src_batch.shape[1]
        #creates a list of #[batch size] empty lists that we append decoded tokens to.
        decoded_tokens = [[] for i in range(batch_size)]

        with torch.no_grad():
            # initialize the first decoder hidden layer to be the final hidden layer of the encoder
            encoder_out, decoder_hidden, decoder_cell = self.encoder(src_batch)
            decoder_input = torch.tensor([sos_idx] * batch_size, device=self.device)
            # print(f"encoder_out shape: {encoder_out.shape}")

            # print(f"decoder_input shape [8, 30, 512]: {decoder_input.shape}")
            # print("decoder hidden shape [8, 1, 512]", decoder_hidden.shape)
            # print("decoder cell shape ", decoder_cell.shape)

            for i in range(max_length):
                # decoder_input = decoder_input.unsqueeze(1)
                # print(f"decoder_input (unsqueezed) shape: {decoder_input.shape}")
                # print("src_batch[:, i] ", src_batch[:, i].shape)
                decoder_output, decoder_hidden, decoder_cell = self.decoder(encoder_out=encoder_out, x=decoder_input, hidden=decoder_hidden, cell=decoder_cell)
                # output highest probability token ids based on decoder output (greedy decoding)
                tokens = decoder_output.argmax(1)

                # print("decoder output shape ", decoder_output.shape)
                # print("tokens shape:", tokens.shape)
                # print("batch_size:", batch_size)
                # print("max_length", max_length)


                # append predicted token for each sentence in batch to decoded_tokens
                for b in range(batch_size):
                    decoded_tokens[b].append(tokens[b].item())

                decoder_input = tokens
                # print("new decoder input shape", decoder_input.shape)

        return decoded_tokens
