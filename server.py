from flask import Flask, request, jsonify
import json
import random
import numpy as np
from keras.models import load_model
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json', encoding="utf8").read())


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_json, ints):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


@app.route('/api/chatbot', methods=['POST'])
def chatbot():

    data = request.get_json()
    user_message = data['message']
    ints = predict_class(user_message, model)
    if (len(ints) > 0):
        response = get_response(intents, ints)
        return jsonify({"response": response, "status": 'success'})
    else:
        return jsonify({"response": 'Xin lỗi vấn đề này tôi cần phải tìm hiểu thêm', "status": 'success'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
