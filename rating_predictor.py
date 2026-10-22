import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/content/amazon.csv')

text_col = [col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()][0]
rating_col = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()][0]

df[text_col] = df[text_col].astype(str).fillna('')

if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
    df['price'] = df['price'].fillna(df['price'].median())
else:
    df['price'] = np.random.uniform(5.0, 150.0, size=len(df))

scaler = StandardScaler()
df['price_scaled'] = scaler.fit_transform(df[['price']])

df[rating_col] = pd.to_numeric(df[rating_col].astype(str).str.replace('|', '', regex=False), errors='coerce')
df = df.dropna(subset=[rating_col]).reset_index(drop=True)

target = df[rating_col].astype(float).astype(int).values
target = np.clip(target, 1, 5) - 1

vocab_size = 5000
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df[text_col].values)

sequences = tokenizer.texts_to_sequences(df[text_col].values)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

meta_features = df[['price_scaled']].values

X_text_train, X_text_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    padded_sequences, meta_features, target, test_size=0.2, random_state=42
)

text_input = Input(shape=(max_length,), name='text_input')
embedding = Embedding(input_dim=vocab_size, output_dim=32)(text_input)
lstm_layer = LSTM(32)(embedding)

meta_input = Input(shape=(meta_features.shape[1],), name='meta_input')
meta_dense = Dense(16, activation='relu')(meta_input)

combined = concatenate([lstm_layer, meta_dense])
dense1 = Dense(16, activation='relu')(combined)
dropout = Dropout(0.2)(dense1)
output = Dense(5, activation='softmax', name='output')(dropout)

model = Model(inputs=[text_input, meta_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_text_train, X_meta_train],
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=([X_text_test, X_meta_test], y_test),
    verbose=1
)

loss, accuracy = model.evaluate([X_text_test, X_meta_test], y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

predictions = model.predict([X_text_test, X_meta_test])
predicted_classes = np.argmax(predictions, axis=1) + 1
actual_classes = y_test + 1

comparison_df = pd.DataFrame({
    'Actual Rating': actual_classes,
    'Predicted Rating': predicted_classes
})
display(comparison_df.head(10))