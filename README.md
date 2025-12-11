# LSTM Seq2Seq Chatbot

This project implements a simple conversational chatbot using a Sequence-to-Sequence (Seq2Seq) model with LSTM encoder-decoder architecture. The model is trained on the Cornell Movie Dialogs Corpus and learns to generate short conversational responses.

## **Features**

LSTM Encoder-Decoder with teacher forcing

Custom vocabulary and tokenization

Data preprocessing & cleaning

Training with gradient clipping and LR scheduling

Greedy decoding + Top-K / Top-P sampling

Interactive chat mode

## **Dataset**

The chatbot is trained on the Cornell Movie Dialogs Corpus, downloaded automatically using kagglehub. Dialogue pairs are extracted by linking consecutive movie lines.

## **Model**

* Encoder: Embedding → Multi-layer LSTM

* Decoder: Embedding → LSTM → Linear output

* Seq2Seq: Handles decoding loop and teacher forcing

## **Training**
* Loss: CrossEntropy (ignoring <pad>)

* Optimizer: Adam

* Early stopping + LR scheduler

* Best model saved as best_chatbot_model.pt

## **Inference**

You can test the chatbot using:

* predict(model, "Hello, how are you?")

or start interactive chat:

* chat(model, word2idx, idx2word, device)

## **File Structure**
LSTM Seq2Seq Chatbot.ipynb   # Main notebook
README.md                    # Documentation

## **Requirements**
* Python 3+
* PyTorch
* numpy, pandas
* kagglehub
* matplotlib
