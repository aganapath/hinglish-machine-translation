import tqdm
from nltk.translate.bleu_score import corpus_bleu
import torch
import torch.nn as nn

def train(model, loss_fn, optimizer, dataloader, device, epochs=150, num_batches=None):
    model.train()

    for epoch in range(epochs):
        epoch_losses = []

        for i, (X, y) in enumerate(tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            if num_batches is not None and i >= num_batches:
                break  # Stop after `num_batches`

            X = X.to(device)
            y = y.to(device)
            h = None

            optimizer.zero_grad()
            output = model(X, y)

            output = output[1:].reshape(-1, output.shape[2])
            y = y[1:].reshape(-1)

            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return avg_loss


def decode_tokens(token_list, vocab):
  decoded_tokens = []
  for t in token_list:
    decoded_tokens.append(vocab.get_itos()[t])

  return decoded_tokens

def evaluate(model, dataloader, vocab, device, sos_idx, num_batches=None):
    model.eval()
    predictions = []
    targets = []

    for i, (X, y) in enumerate(tqdm.tqdm(dataloader)):
        if num_batches is not None and i >= num_batches:
            break  # Stop after `num_batches`

        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)
            y_hat = model.translate(X, sos_idx)
            y = y.tolist()

            for target, prediction in zip(y, y_hat):
                decoded_prediction = decode_tokens(prediction, vocab)
                decoded_target = decode_tokens(target, vocab)
                # print("\n", decoded_prediction, " decoded prediction")
                # print(decoded_target, " decoded target")

                predictions.append(decoded_prediction)
                targets.append(decoded_target)

    # compute bleu score between all target and predicted sentences
    bleu_score = corpus_bleu(targets, predictions)
    print(f"BLEU: {bleu_score}")


    return bleu_score