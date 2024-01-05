import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tqdm import tqdm

pd.set_option("display.WIDTH", 1000)
pd.set_option("display.max_columns", 15)

true_data = pd.read_csv("Fake_and_True_News/True.csv")
fake_data = pd.read_csv("Fake_and_True_News/Fake.csv")
# print(true_data.head())
# print(fake_data.head())

true_data = true_data.drop(["subject", "date"], axis=1)
fake_data = fake_data.drop(["subject", "date"], axis=1)
true_data['Category'] = ["True" for _ in range(len(true_data))]
fake_data['Category'] = ["Fake" for _ in range(len(fake_data))]

data = pd.concat([true_data, fake_data])
data.reset_index(inplace=True, drop=True)
data = shuffle(data)
data.reset_index(inplace=True, drop=True)
print(data)
# sns.countplot(data.Category)
# plt.show()

# Universal sentence encoder
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
	data.Category.to_numpy().reshape(-1, 1)
)

print(type_one_hot.shape)

train_news, test_news, y_train, y_test = train_test_split(data.title, type_one_hot, test_size=0.1,
													random_state=101)
print(train_news[0])
X_train = []
for c in tqdm(train_news):
	embedding = use([c])
	news_emb = tf.reshape(embedding, [-1]).numpy()
	X_train.append(news_emb)
X_train = np.array(X_train)

X_test = []
for c in tqdm(test_news):
	embedding = use([c])
	news_emb = tf.reshape(embedding, [-1]).numpy()
	X_test.append(news_emb)
X_test = np.array(X_test)

print(X_train[0].shape, X_test[0].shape)
model = Sequential()
model.add(Dense(256, input_shape=[X_train.shape[1]], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer='Adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=4, batch_size=16, validation_split=0.1, shuffle=True)

loss = pd.DataFrame(model.history.history)

loss[['loss', 'val_loss']].plot()
plt.show()
loss[['accuracy', 'val_accuracy']].plot()
plt.show()
print(model.evaluate(X_test, y_test))

model.save('news_classifier.h5')
with open("Embedding.txt", "w") as f:
	f.write('''
	use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')\n
	arr = []\n
	embedding = use([c])\n
	true_embed = tf.reshape(true_embed, [-1]).numpy()\n
	arr.append(true_embed)\n
	arr = np.array(X_test)''')

print("\n\n\n")
print(test_news.iloc[59])
print("Fake" if y_test[59][0] == 1 else "True")
print(model.predict(X_test[59:60]))
