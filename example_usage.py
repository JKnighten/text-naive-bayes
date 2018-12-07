import pickle

from naivebayes.models.dictionary import Multinomial

#####################
# Wine Data Example #
#####################

with open('./sample_data/nltk/wine/wine_data.pkl', 'rb') as f:
        wine_data = pickle.load(f)

# Remove Top N Most Common Words
n = 50
label_0_top_50 = {word for word, freq in wine_data[0][0].most_common(n)}
label_1_top_50 = {word for word, freq in wine_data[0][1].most_common(n)}

for word_to_remove in label_0_top_50.union(label_1_top_50):
    del wine_data[0][0][word_to_remove]
    del wine_data[0][1][word_to_remove]

model = Multinomial()
model.train(wine_data[1], wine_data[0])

print(model.predict(wine_data[2]))
print(wine_data[3])



