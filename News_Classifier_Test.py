import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model

model = load_model("Fake_True_News_Classifier/news_classifier.h5")
use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
text = input("Title of the news article: ")
while text.lower()!="done":
	text = [text]
	arr = []
	for c in text:
		embedding = use([c])
		true_embed = tf.reshape(embedding, [-1]).numpy()
		arr.append(true_embed)
	arr = np.array(arr)
	y_pred = model.predict(arr)
	if y_pred[0][0] > .65:
		print(f"Fake - Certainty: {y_pred[0][0]*100}%")
	elif y_pred[0][1] > 0.65:
		print(f"Real - Certainty: {y_pred[0][1]*100}%")
	else:
		print(f"Unsure - Certainty: {y_pred[0][0]*100}%-Real   ----    {y_pred[0][1]*100}%-Fake")
	text = input("Title of the news article: ")
