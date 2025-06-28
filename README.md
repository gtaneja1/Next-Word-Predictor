This project implements a next-word prediction model using LSTM (Long Short-Term Memory) in PyTorch.
Given a sequence of words, the model predicts the most likely next word, trained on domain-specific text data. This project showcases practical understanding of deep learning, NLP preprocessing, model training, and inference generation in Python.

FEATURES:
-Tokenization and custom vocabulary generation using NLTK
- Cleaned and preprocessed training data from raw text
- Sequence slicing using sliding windows
- LSTM-based neural network built from scratch in PyTorch
- Custom training loop using CrossEntropyLoss and Adam optimizer
- Prediction function to generate 1 or more future words from a seed
- Optional training continuation using saved states
