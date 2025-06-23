# Sentiment Identification Project

A machine learning project for sentiment analysis of movie reviews using a Recurrent Neural Network (RNN) with LSTM architecture. This project can classify text reviews as positive or negative sentiment.

## Project Overview

This project implements a sentiment analysis system that:
- Preprocesses movie review data
- Trains an RNN model with LSTM layers for binary sentiment classification
- Provides both command-line and web interface for making predictions
- Includes comprehensive logging and evaluation metrics

## Architecture

The model architecture consists of:
- **Embedding Layer**: Converts words to dense vectors
- **LSTM Layer**: Processes sequential text data
- **Dropout Layers**: Prevents overfitting
- **Fully Connected Layers**: Final classification layers
- **Sigmoid Activation**: Outputs probability for binary classification

