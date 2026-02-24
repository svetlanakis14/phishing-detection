import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from preprocessing import (load_dataset, initial_cleaning, preprocess_texts, split_data, build_tfidf, load_vectorizer, build_lstm, padding, load_tokenizer, encode_labels, MAX_VOCABULARY, MAX_LENGTH)
from model_tfidf import (build_tfidf_model, train_tfidf, save_tfidf)
from model_lstm import (build_lstm_model, train_lstm_model, save_lstm_model)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
TFIDF_DIR = os.path.join(MODELS_DIR, 'tfidf')
LSTM_DIR = os.path.join(MODELS_DIR, 'lstm')

def plot_history(history, save_path=None):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    if save_path:
        plt.savefig(save_path)
    
def main():
   os.makedirs(MODELS_DIR, exist_ok=True)
   os.makedirs(RESULTS_DIR, exist_ok=True)

   df = load_dataset()
   df = initial_cleaning(df)
   df = preprocess_texts(df)

   train, test = split_data(df)

   y_train = encode_labels(train)
   y_test = encode_labels(test)

   vectorizer_path = os.path.join(TFIDF_DIR, 'tfidf_vectorizer.pkl')

   X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf(train['clean_text'], test['clean_text'], save_path=vectorizer_path)
   
   print('Tf-idf model summary:')
   tfidf_model = build_tfidf_model(input_dim=X_train_tfidf.shape[1])
   tfidf_model.summary()
   
   print('PRE ATTACK: Tf-idf model training:')
   history_tfidf = train_tfidf(tfidf_model, X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test, checkpoint_dir=TFIDF_DIR)
   save_tfidf(tfidf_model, os.path.join(TFIDF_DIR, 'tfidf_model.keras'))
   plot_path = os.path.join(TFIDF_DIR, 'tfidf_loss.png')
   plot_history(history_tfidf, save_path=plot_path)
 
   tok_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
   tokenizer = build_lstm(train['clean_text'], save_path=tok_path)

   X_train_seq = padding(train['clean_text'], tokenizer)
   X_test_seq  = padding(test['clean_text'],  tokenizer)
   
   print('LSTM model summary:')
   lstm_model = build_lstm_model(vocab_size=MAX_VOCABULARY, max_seq_length=MAX_LENGTH)
   lstm_model.summary()
   
   print('PRE ATTACK: LSTM model training:')
   hist_lstm = train_lstm_model(lstm_model, X_train_seq, y_train, X_test_seq,  y_test, epochs=10, batch_size=64, checkpoint_dir=LSTM_DIR)
   save_lstm_model(lstm_model, os.path.join(LSTM_DIR, 'lstm_model.keras'))
   plot_lstm_path = os.path.join(LSTM_DIR, 'lstm_loss.png')
   plot_history(hist_lstm, save_path=plot_lstm_path)

if __name__ == "__main__":
    main()