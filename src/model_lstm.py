import os
from keras import Sequential, load_model, Dense, Dropout, Adam, EarlyStopping, ModelCheckpoint, Embedding, LSTM, Bidirectional

def build_lstm_model(vocab_size, embedding_dim = 128, lstm_units = 64, max_seq_length = 200, dropout_rate = 0.3):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length, mask_zero=True),
        Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.2)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate / 2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, checkpoint_dir=None):
    
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)]

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'lstm_best.keras')
        callbacks.append(ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=0))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history


def save_lstm_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_lstm_model(path: str):
    return load_model(path)