#######################################################
# Imports
#######################################################
import io
import os
import string
import warnings

import aiml
import nltk
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ImageTest import preprocess_image

# Unsafe
warnings.filterwarnings("ignore")
#######################################################
# Initialise Keras Model
#######################################################
test_dir = "data/fruits-360_dataset/fruits-360/Test"
model_dir = "data/models/pani_rmsprop_cnn.hdf5"
test_datagen = ImageDataGenerator(rescale=1 / 255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(100, 100))
label_map = test_generator.class_indices

model = load_model(model_dir)

#######################################################
# Initialise NLTK Agent
#######################################################
data = "data/data.txt"
readFile = io.open(data, 'r')
corpus = readFile.read()
lowerCorpus = corpus.lower()

# Uncomment these on first run
# nltk.download('punkt')
# nltk.download('wordnet')

corpal_sentences = nltk.sent_tokenize(lowerCorpus)
corpal_words = nltk.word_tokenize(lowerCorpus)

#######################################################
# Normalize tokens using Lemmatization
#######################################################
lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    lemmedTokens = []
    for token in tokens:
        lemmedTokens.append(lemmer.lemmatize(token))

    return lemmedTokens


# Dictionary of punctuations in unicode
remove_punct_dict = {}
for punctuation in string.punctuation:
    remove_punct_dict[ord(punctuation)] = None


def lemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


#######################################################
#  Similarity-Based Response Generation
#######################################################
def response(user_response):
    chatBot_Response = ""
    corpal_sentences.append(user_response)
    tfidf = TfidfVectorizer(tokenizer=lemNormalize, stop_words='english').fit_transform(corpal_sentences)
    cosSimiliarity = cosine_similarity(tfidf[-1], tfidf)
    idx = cosSimiliarity.argsort()[0][-2]
    flat = cosSimiliarity.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatBot_Response += "Sorry, I don't know what that is"
        return chatBot_Response
    else:
        # TODO parse response to remove the found keyword
        chatBot_Response += "\t" + corpal_sentences[idx].split("\n")[1]
        return chatBot_Response


#######################################################
#  Initialise AIML agent
#######################################################
kern = aiml.Kernel()
kern.setTextEncoding(None)
braindir = "data/models/brain.brn"
if os.path.isfile(braindir):
    kern.bootstrap(brainFile=braindir)
else:
    kern.bootstrap(learnFiles="std-startup.xml", commands="LOAD BRAIN")
    kern.saveBrain(braindir)

#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the Cayman Islands Chat Bot!"
                  "\nAsk me anything about the Cayman islands including"
                  " it's geography, national animals and so on."
                  "\n\n** NEW UPDATE **"
                  "\n\nMy owner has now trained me on over 60k images of fruits! Upload images in .jpg format to the"
                  " upload folder and I'll try and predict what it is!")
print(welcomeMessage)
#######################################################
# Main loop
#######################################################
while True:
    # get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    # activate selected response agent
    answer = kern.respond(userInput)
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 10:
            print("Let me think..", end="")
            print(response(userInput))
            corpal_sentences.remove(userInput)
        elif cmd == 11:
            uploaddir = "data/upload/"
            filetype = ".jpg"
            img_tensor = preprocess_image(uploaddir + userInput + filetype)
            classes = model.predict_classes(img_tensor)
            for label, num in label_map.items():
                if num == classes:
                    print("I think this is a:", label)

        elif cmd == 99:
            print("Sorry repeat that please")
    else:
        print(answer)
