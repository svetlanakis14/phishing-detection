import os
import numpy as np
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from preprocessing import (load_dataset, initial_cleaning, preprocess_texts, split_data, build_tfidf, build_lstm, padding, encode_labels, MAX_VOCABULARY, MAX_LENGTH)
from model_tfidf import (build_tfidf_model, train_tfidf, save_tfidf)
from model_lstm import (build_lstm_model, train_lstm_model, save_lstm_model)
from evaluation import (plot_history, evaluate_model, plot_comparison_bar)
from adversarial_attacks import (apply_adversarial_attacks)
import matplotlib
matplotlib.use('Agg')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
TFIDF_DIR = os.path.join(MODELS_DIR, 'tfidf')
LSTM_DIR = os.path.join(MODELS_DIR, 'lstm')
TFIDF_ATTACK_DIR = os.path.join(TFIDF_DIR, 'attacks')
LSTM_ATTACK_DIR = os.path.join(LSTM_DIR, 'attacks')
TFIDF_RETRAIN_DIR = os.path.join(TFIDF_DIR, 'post_train')
LSTM_RETRAIN_DIR = os.path.join(LSTM_DIR, 'post_train')


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TFIDF_ATTACK_DIR, exist_ok=True)
    os.makedirs(LSTM_ATTACK_DIR, exist_ok=True)
    os.makedirs(TFIDF_RETRAIN_DIR, exist_ok=True)
    os.makedirs(LSTM_RETRAIN_DIR, exist_ok=True)

    df = load_dataset()
    df = initial_cleaning(df)
    df = preprocess_texts(df)

    train, test = split_data(df)

    y_train = encode_labels(train)
    y_test = encode_labels(test)

    vectorizer_path = os.path.join(TFIDF_DIR, 'tfidf_vectorizer.pkl')
    X_train_tfidf, X_test_tfidf, vectorizer = build_tfidf(train['clean_text'], test['clean_text'], save_path=vectorizer_path)

    print('\nTf-idf model summary:')
    tfidf_model = build_tfidf_model(input_dim=X_train_tfidf.shape[1])
    tfidf_model.summary()

    print('\nPRE ATTACK: Tf-idf model training:')
    history_tfidf = train_tfidf(tfidf_model, X_train_tfidf.toarray(), y_train, X_test_tfidf.toarray(), y_test, checkpoint_dir=TFIDF_DIR)
    save_tfidf(tfidf_model, os.path.join(TFIDF_DIR, 'tfidf_model.keras'))
    plot_history(history_tfidf, save_path=os.path.join(TFIDF_DIR, 'tfidf_loss.png'))

    print('\nPRE ATTACK: TF-IDF evaluation')
    tfidf_metrics = evaluate_model(tfidf_model, X_test_tfidf.toarray(), y_test, model_name='tfidf', metrics_path=os.path.join(TFIDF_DIR, 'tfidf_metrics.json'), cm_path=os.path.join(TFIDF_DIR, 'tfidf_confusion_matrix.png'))

    tok_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
    tokenizer = build_lstm(train['clean_text'], save_path=tok_path)

    X_train_seq = padding(train['clean_text'], tokenizer)
    X_test_seq = padding(test['clean_text'], tokenizer)

    print('\nLSTM model summary:')
    lstm_model = build_lstm_model(vocab_size=MAX_VOCABULARY, max_seq_length=MAX_LENGTH)
    lstm_model.summary()

    print('\nPRE ATTACK: LSTM model training:')
    hist_lstm = train_lstm_model(lstm_model, X_train_seq, y_train, X_test_seq, y_test, epochs=10, batch_size=64, checkpoint_dir=LSTM_DIR)
    save_lstm_model(lstm_model, os.path.join(LSTM_DIR, 'lstm_model.keras'))
    plot_history(hist_lstm, save_path=os.path.join(LSTM_DIR, 'lstm_loss.png'))

    print('\nPRE ATTACK: LSTM evaluation')
    lstm_metrics = evaluate_model(lstm_model, X_test_seq, y_test, model_name='lstm', metrics_path=os.path.join(LSTM_DIR, 'lstm_metrics.json'), cm_path=os.path.join(LSTM_DIR, 'lstm_confusion_matrix.png'))

    print('\nATTACKING:')
    attacked = apply_adversarial_attacks(test['clean_text'].tolist())

    for attack_name, attack_texts in attacked.items():
        print(f"\n  Attack: {attack_name}")

        X_att_tfidf = vectorizer.transform(attack_texts)
        evaluate_model(tfidf_model, X_att_tfidf.toarray(), y_test, model_name=f'tfidf_{attack_name}', metrics_path=os.path.join(TFIDF_ATTACK_DIR, f'tfidf_{attack_name}_metrics.json'), cm_path=os.path.join(TFIDF_ATTACK_DIR, f'tfidf_{attack_name}_confusion_matrix.png'))

        X_att_seq = padding(attack_texts, tokenizer)
        evaluate_model(lstm_model, X_att_seq, y_test, model_name=f'lstm_{attack_name}', metrics_path=os.path.join(LSTM_ATTACK_DIR, f'lstm_{attack_name}_metrics.json'), cm_path=os.path.join(LSTM_ATTACK_DIR, f'lstm_{attack_name}_confusion_matrix.png'))

    print('\nRetraining with attacked emails...')
    all_adv_texts = []
    all_adv_labels = []
    for attack_texts in attacked.values():
        all_adv_texts.extend(attack_texts)
        all_adv_labels.extend(y_test.tolist())

    all_adv_labels = np.array(all_adv_labels)
    augmented_texts = train['clean_text'].tolist() + all_adv_texts
    augmented_labels = np.concatenate([y_train, all_adv_labels])

    print("\n  Retrain TF-IDF model...")
    X_aug_tfidf = vectorizer.transform(augmented_texts).toarray()
    tfidf_model_v2 = build_tfidf_model(input_dim=X_train_tfidf.shape[1])
    train_tfidf(tfidf_model_v2, X_aug_tfidf, augmented_labels, X_test_tfidf.toarray(), y_test, epochs=10, batch_size=64)
    save_tfidf(tfidf_model_v2, os.path.join(TFIDF_DIR, 'tfidf_model_v2.keras'))

    print('\nPOST RETRAIN: TF-IDF evaluation')
    evaluate_model(tfidf_model_v2, X_test_tfidf.toarray(), y_test, model_name='tfidf_retrain', metrics_path=os.path.join(TFIDF_RETRAIN_DIR, 'tfidf_retrain_metrics.json'), cm_path=os.path.join(TFIDF_RETRAIN_DIR, 'tfidf_retrain_confusion_matrix.png'))

    print("\n  Retrain LSTM model...")
    X_aug_seq = padding(augmented_texts, tokenizer)
    lstm_model_v2 = build_lstm_model(vocab_size=MAX_VOCABULARY, max_seq_length=MAX_LENGTH)
    train_lstm_model(lstm_model_v2, X_aug_seq, augmented_labels, X_test_seq, y_test, epochs=10, batch_size=64)
    save_lstm_model(lstm_model_v2, os.path.join(LSTM_DIR, 'lstm_model_v2.keras'))

    print('\nPOST RETRAIN: LSTM evaluation')
    evaluate_model(lstm_model_v2, X_test_seq, y_test, model_name='lstm_retrain', metrics_path=os.path.join(LSTM_RETRAIN_DIR, 'lstm_retrain_metrics.json'), cm_path=os.path.join(LSTM_RETRAIN_DIR, 'lstm_retrain_confusion_matrix.png'))

    all_metrics = [tfidf_metrics, lstm_metrics]
    plot_comparison_bar(all_metrics, save_path=os.path.join(RESULTS_DIR, 'all_models_comparison.png'))


if __name__ == "__main__":
    main()
