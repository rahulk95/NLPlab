#Name : Rahul Sunil Kolase
#Roll No: 34
#Assignment : 02


import gensim
from gensim import corpora, models
from gensim.matutils import np
from gensim.utils import simple_preprocess

text1 = ["""ou probably already know that it is important to have a king-size breakfast every morning. 
         do you know why Your body is hungry in the morning because you haven’t eaten for about 8-10 hours? Breakfast is therefore the first meal of the day, and therefore, the most important. 
         Imagine driving without fuel; This is exactly how your body feels without fuel from a nutritious breakfast. Nowadays many people skip breakfast to lose weight. 
         Nutritionists are alarmed by this trend, as it is mandatory to eat breakfast within two hours of waking up. Depriving the body of energy can lead to serious health problems in the long run. 
         Forget silly celebrities and their absurd ways to lose weight. 
         Never miss breakfast!"""]

tokens1 = [[item for item in line.split()] for line in text1]
g_dict1 = corpora.Dictionary(tokens1)

print("The dictionary has: " +str(len(g_dict1)) + " tokens\n")
print(g_dict1.token2id)

text = ["The food is excellent but the service can be better",
        "The food is always delicious and loved the service",
        "The food was mediocre and the service was terrible"]

g_dict = corpora.Dictionary([simple_preprocess(line) for line in text])
g_bow = [g_dict.doc2bow(simple_preprocess(line)) for line in text]

print("Dictionary : ")
for item in g_bow:
    print([[g_dict[id], freq] for id, freq in item])

g_tfidf = models.TfidfModel(g_bow, smartirs='ntc')

print("TF-IDF Vector:")
for item in g_tfidf[g_bow]:
    print([[g_dict[id], np.around(freq, decimals=2)] for id, freq in item])
    
    corpus_text = 'n'.join(rev[:1000]['Text'])
data = []
# iterate through each sentence in the file
for i in sent_tokenize(corpus_text):
    temp = []
    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)
    from gensim.models import Word2Vec

#loading the downloaded model
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)

#the model is loaded. It can be used to perform all of the tasks mentioned above.

# getting word vectors of a word
dog = model['dog']

#performing king queen magic
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

#picking odd one out
print(model.doesnt_match("breakfast cereal dinner lunch".split()))

#printing similarity index
print(model.similarity('woman', 'man'))
    
# Output:-
# The dictionary has: 92 tokens

# {'8-10': 0, 'Breakfast': 1, 'Depriving': 2, 'Forget': 3, 'Imagine': 4, 'Never': 5, 'Nowadays': 6, 'Nutritionists': 7, 'This': 8, 'Your': 9, 'a': 10, 'about': 11, 'absurd': 12, 'alarmed': 13, 'already': 14, 'and': 15, 'are': 16, 'as': 17, 'because': 18, 'body': 19, 'breakfast': 20, 'breakfast!': 21, 'breakfast.': 22, 'by': 23, 'can': 24, 'celebrities': 25, 'day,': 26, 'do': 27, 'driving': 28, 'eat': 29, 'eaten': 30, 'energy': 31, 'every': 32, 'exactly': 33, 'feels': 34, 'first': 35, 'for': 36, 'from': 37, 'fuel': 38, 'fuel;': 39, 'have': 40, 'haven’t': 41, 'health': 42, 'hours': 43, 'hours?': 44, 'how': 45, 'hungry': 46, 'important': 47, 'important.': 48, 'in': 49, 'is': 50, 'it': 51, 'king-size': 52, 'know': 53, 'lead': 54, 'long': 55, 'lose': 56, 'mandatory': 57, 'many': 58, 'meal': 59, 'miss': 60, 'morning': 61, 'morning.': 62, 'most': 63, 'nutritious': 64, 'of': 65, 'ou': 66, 'people': 67, 'probably': 68, 'problems': 69, 'run.': 70, 'serious': 71, 'silly': 72, 'skip': 73, 'that': 74, 'the': 75, 'their': 76, 'therefore': 77, 'therefore,': 78, 'this': 79, 'to': 80, 'trend,': 81, 'two': 82, 'up.': 83, 'waking': 84, 'ways': 85, 'weight.': 86, 'why': 87, 'within': 88, 'without': 89, 'you': 90, 'your': 91}
# Dictionary : 
# [['be', 1], ['better', 1], ['but', 1], ['can', 1], ['excellent', 1], ['food', 1], ['is', 1], ['service', 1], ['the', 2]]
# [['food', 1], ['is', 1], ['service', 1], ['the', 2], ['always', 1], ['and', 1], ['delicious', 1], ['loved', 1]]
# [['food', 1], ['service', 1], ['the', 2], ['and', 1], ['mediocre', 1], ['terrible', 1], ['was', 2]]
# TF-IDF Vector:
# [['be', 0.43], ['better', 0.43], ['but', 0.43], ['can', 0.43], ['excellent', 0.43], ['food', 0.09], ['is', 0.21], ['service', 0.09], ['the', 0.18]]
# [['food', 0.11], ['is', 0.26], ['service', 0.11], ['the', 0.21], ['always', 0.52], ['and', 0.26], ['delicious', 0.52], ['loved', 0.52]]
# [['food', 0.08], ['service', 0.08], ['the', 0.16], ['and', 0.2], ['mediocre', 0.39], ['terrible', 0.39], ['was', 0.78]]