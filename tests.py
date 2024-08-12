from nltk_utils import tokenize, stem, bag_of_words

item = "how long does shipping take?"
print(item)
print(tokenize(item))
words = ["Organic", "shipping", "take", "long", "how", "does"]
print([stem(word) for word in words])

print("##################Bag of words##################")
sentence = ["hello", "how", "are", "you"]
words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
bag =   [  0,      1,     0,    1,    0,      0,       0]
print(bag_of_words(sentence, words))