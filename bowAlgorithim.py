import re
import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))
    return words


def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab))
    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))

        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        print("{0}\n{1}\n".format(sentence, numpy.array(bag_vector)))


tempText = ["Joe waited for the train",
            "The train was late",
            "Mary and Samantha took the bus",
            "I looked for Mary and Samantha at the bus station ",
            "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]

countVectorizer = CountVectorizer()
tfidVectorizer = TfidfVectorizer()

countTest = countVectorizer.fit_transform(tempText)
print(countTest.toarray())

tfidTest = tfidVectorizer.fit_transform(tempText)
print(tfidTest.shape)
