# 📝 Next-Word Prediction with LSTM (PyTorch)

A **deep learning project** implementing a next-word prediction model using an **LSTM-based neural network** in PyTorch. The model learns from text sequences and predicts context-aware future words.

---

## 🚀 Features

* **Custom Training Pipeline**: Built with NLTK for tokenization, vocabulary generation, and sequence slicing.
* **LSTM Model**: Multi-layer LSTM trained to predict the next word in a sequence.
* **Optimization**: Uses CrossEntropyLoss + Adam optimizer with checkpoint saving.
* **Prediction**: Generates one or more words following an input sequence.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NLTK
* Jupyter Notebook

---

## 📂 Project Structure

```
next-word-prediction/
│── notebooks/
│   └── next_word_prediction.ipynb   # model dev & experiments
│── src/
│   ├── train.py                     # training pipeline
│   ├── predict.py                   # word prediction
│── requirements.txt
│── README.md
```

---

## 🔧 How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/<your-username>/next-word-prediction.git
   cd next-word-prediction
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:

   ```bash
   jupyter notebook notebooks/next_word_prediction.ipynb
   ```

4. **Or train via script**:

   ```bash
   python src/train.py
   ```

5. **Predict next words**:

   ```bash
   python src/predict.py --input "The future of AI"
   ```

---

## 📌 Example Output

```
Input:  "The future of AI"
Output: "The future of AI is bright"
```

---

## 🌟 Future Improvements

* Add transformer-based model for comparison (BERT/ GPT-style).
* Deploy as a web app with Streamlit.
* Experiment with larger datasets for better accuracy.

---
