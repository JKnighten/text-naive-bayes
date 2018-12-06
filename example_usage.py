import pickle

#####################
# Wine Data Example #
#####################

with open('sample_data/nltk/wine/wine_data.pkl', 'rb') as f:
        wine_data = pickle.load(f)

print(wine_data)