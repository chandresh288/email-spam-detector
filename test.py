import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
data = pd.read_csv('spam.csv')
data.head()

# Analyze data
data.groupby('Category').describe().T

import seaborn as sns
plt.figure(figsize=(12,14))
sns.countplot(data['Category'])

# Downsampling
ham_msg = data[data.Category == 'ham']
spam_msg = data[data.Category == 'spam']
ham_msg = ham_msg.sample(n=len(spam_msg), random_state=42)

print(ham_msg.shape, spam_msg.shape)

# Combine and reset index
balanced_data = pd.concat([ham_msg, spam_msg]).reset_index(drop=True)
plt.figure(figsize=(8, 6))
sns.countplot(balanced_data.Category)
plt.title('Distribution of ham and spam email messages (after downsampling)')
plt.xlabel('Message types')

# Prepare data for training
balanced_data['label'] = balanced_data['Category'].map({'ham': 0, 'spam': 1})
train_msg, test_msg, train_labels, test_labels = train_test_split(
    balanced_data['Message'],
    balanced_data['label'],
    test_size=0.2,
    random_state=434
)

# Tokenization
vocab_size = 500
oov_tok = '<OOV>'
max_len = 50

token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
token.fit_on_texts(train_msg)

# Word index
word_index = token.word_index

# Padding sequences
padding_type = 'post'
truncate_type = 'post'
Trainning_seq = token.texts_to_sequences(train_msg)
Trainning_pad = pad_sequences(Trainning_seq, maxlen=50, padding=padding_type, truncating=truncate_type)

Testing_seq = token.texts_to_sequences(test_msg)
Testing_pad = pad_sequences(Testing_seq, maxlen=50, padding=padding_type, truncating=truncate_type)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=50),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
    optimizer='adam'
)

# Train the model
epoch = 30
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(
    Trainning_pad, 
    train_labels, 
    validation_data=(Testing_pad, test_labels), 
    epochs=epoch, 
    callbacks=[early_stop], 
    verbose=2
)

# Evaluate the model
model.evaluate(Testing_pad, test_labels)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

