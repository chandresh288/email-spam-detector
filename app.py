from flask import Flask, request, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and tokenizer
model = tf.keras.models.load_model('spam_ham_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    result = 'Spam' if prediction > 0.5 else 'Ham'
    return render_template('index.html', prediction_text=f'The message is: {result}')

if __name__ == '__main__':
    app.run(debug=True)
