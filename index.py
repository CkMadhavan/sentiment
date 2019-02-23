from flask import Flask,jsonify
import pickle
import tensorflow as tf
import numpy as np

app = Flask(__name__)

@app.route('/<process>')
def main(process):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(20000 , 32 , input_length = 500))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dense(50))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1 , activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
        
    model.load_weights('sentiment_final.h5')

    with open('token' , "rb") as file:
        t = pickle.load(file)
    
    text = [process]

    word = tf.keras.preprocessing.sequence.pad_sequences(t.texts_to_sequences(text) , 500)
    x = str(model.predict(np.array(word))[0][0])
    tf.keras.backend.clear_session()
    
    return jsonify({"process" :x})

if __name__=="__main__":
    app.run()
