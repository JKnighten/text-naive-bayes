import pickle

from naivebayes.models.dictionary import Multinomial

#####################
# Wine Data Example #
#####################

with open('./sample_data/nltk/wine/wine_data.pkl', 'rb') as f:
        wine_data = pickle.load(f)

wine_bow_data = wine_data["bagofwords"]

freq_data = wine_bow_data["freq_data"]
label_counts = wine_bow_data["label_counts"]
raw_data = wine_bow_data["raw_data"]
labels = wine_bow_data["labels"]

model = Multinomial()
model.train(label_counts, freq_data)

print(model.predict(raw_data))
print(labels)
