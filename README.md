# Machine Translation with Attention-Based Seq2Seq Model

This project implements a machine translation system using a Seq2Seq model with attention mechanisms. The model translates between English and Vietnamese sentences, leveraging modern deep learning techniques such as recurrent neural networks (RNN) and attention mechanisms.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Overview

The objective of this project is to build a machine translation model that can translate English sentences into Vietnamese (or vice versa) by using an Attention-based Seq2Seq model. The Seq2Seq (sequence-to-sequence) model is a popular architecture for tasks where input and output sequences have different lengths. Attention mechanisms enhance the Seq2Seq model by allowing the decoder to "focus" on different parts of the input sequence during the translation process.

## Features
- Seq2Seq model with LSTM-based Encoder and Decoder.
- Bahdanau Attention mechanism.
- Custom data preprocessing and vocabulary building.
- Teacher Forcing for efficient training.
- Tokenization and padding support.
- Evaluation on BLEU Score.

## Dataset

The dataset used for training the model is **English-Vietnamese parallel sentences** from the IWSLT (International Workshop on Spoken Language Translation) dataset. The dataset contains approximately 126,000 sentence pairs for training and 6,000 pairs for validation/testing.

### Example Sentence Pair:
- **English**: "I love programming."
- **Vietnamese**: "Tôi yêu lập trình."

You can download the dataset from [IWSLT official repository](https://wit3.fbk.eu/).

## Requirements

To run the project, you will need to install the following dependencies:

- Python >= 3.8
- PyTorch >= 1.7
- tqdm
- numpy
- matplotlib

Use the following command to install the dependencies:

```bash
pip install -r requirements.txt
