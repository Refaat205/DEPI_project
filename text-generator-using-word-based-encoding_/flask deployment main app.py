from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

# Simulated model loading (you'd replace this with your actual trained model)
class TextClassificationModel:
    def __init__(self):
        # Tokenizer setup
        self.tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        
        # Example training data (you'd use your actual training data)
        sample_texts = [
            "help me obi-wan kenobi youre my only hope",
            "star wars is awesome",
            "science fiction movie quote",
            "famous movie line"
        ]
        self.tokenizer.fit_on_texts(sample_texts)
        
        # Create a simple model
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(10,)),  # Explicitly define input shape
            tf.keras.layers.Embedding(1000, 16, input_length=10),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Dummy training (replace with actual training)
        sequences = self.tokenizer.texts_to_sequences(sample_texts)
        padded_sequences = pad_sequences(sequences, maxlen=10, padding='post', truncating='post')
        labels = np.array([1, 1, 0, 0], dtype=np.float32)  # Explicit float32 conversion
        
        # Convert to TensorFlow tensor
        padded_tensor = tf.convert_to_tensor(padded_sequences, dtype=tf.int32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Use tf.data.Dataset for training
        dataset = tf.data.Dataset.from_tensor_slices((padded_tensor, labels_tensor))
        dataset = dataset.shuffle(buffer_size=4).batch(2)
        
        # Train the model
        self.model.fit(dataset, epochs=10, verbose=0)

    def predict(self, text):
        # Preprocess the input
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=10, padding='post', truncating='post')
        
        # Make prediction
        prediction = self.model.predict(padded_sequence)[0][0]
        return prediction

# Create model instance
text_model = TextClassificationModel()

@app.route('/')
def home():
    return render_template('text_index.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.form['user_input']
    try:
        # Process text input through the model
        prediction = text_model.predict(user_input)
        
        # Convert prediction to a meaningful output
        confidence = prediction * 100
        category = "Movie Quote" if prediction > 0.5 else "Not a Movie Quote"
        
        output = f"Analysis: {category} (Confidence: {confidence:.2f}%)"
    except Exception as e:
        output = f"Error processing input: {str(e)}"
    
    return render_template('text_index.html', user_input=user_input, output=output)

if __name__ == '__main__':
    app.run(debug=True)