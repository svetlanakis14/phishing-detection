import os 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def build_tfidf_model(input_dim, dropout_rate=0.5):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_tfidf(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64, checkpoint_dir=None):
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, 'tfidf_best.keras')
        callbacks.append(ModelCheckpoint(path, monitor='val_loss', save_best_only=True, verbose=0))
    
    history = model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=epochs,
       batch_size=batch_size,
       callbacks=callbacks,
       verbose = 1
    )
    
    return history 

def save_tfidf(model, path):
    os.makedirs(os.path.dirname(path), exist_ok = True)
    model.save(path)

def load_tfidf(path):
    return load_model(path)