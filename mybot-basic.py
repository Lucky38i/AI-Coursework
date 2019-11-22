#######################################################
# Imports
#######################################################
import io
import json
import os
import string
import warnings

import aiml
import nltk
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ImageClassification import preprocess_image

# Unsafe
warnings.filterwarnings("ignore")
#######################################################
# Initialise Keras Model
#######################################################
label_map_file = "data/label_map.json"
model_dir = "data/models/pani_adam_cnn.hdf5"
with open(label_map_file, 'r') as file:
    label_map = json.load(file)

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


def lem_tokens(tokens):
    lemmed_tokens = []
    for token in tokens:
        lemmed_tokens.append(lemmer.lemmatize(token))

    return lemmed_tokens


# Dictionary of punctuations in unicode
remove_punct_dict = {}
for punctuation in string.punctuation:
    remove_punct_dict[ord(punctuation)] = None


def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


#######################################################
#  Similarity-Based Response Generation
#######################################################
def response(user_response):
    chatBot_Response = ""
    corpal_sentences.append(user_response)
    tfidf = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english').fit_transform(corpal_sentences)
    cos_similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = cos_similarity.argsort()[0][-2]
    flat = cos_similarity.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatBot_Response += "Sorry, I don't know what that is"
        return chatBot_Response
    else:
        chatBot_Response += "\t" + corpal_sentences[idx].split("\n")[1]
        return chatBot_Response


#######################################################
#  Initialise AIML agent
#######################################################
brain_dir = "data/models/brain.brn"
startup_dir = "data/std-startup.xml"

kern = aiml.Kernel()
kern.setTextEncoding(None)

if os.path.isfile(brain_dir):
    kern.bootstrap(brainFile=brain_dir)
else:
    kern.bootstrap(learnFiles=startup_dir, commands="LOAD BRAIN")
    kern.saveBrain(brain_dir)

#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the Cayman Islands Chat Bot!"
                  "\nAsk me anything about the Cayman islands including"
                  " it's geography, national animals and so on."
                  "\n\n** NEW UPDATE **"
                  "\nMy owner has now trained me on over 60k images of fruits! ask me about"
                  " the images in the upload folder and I'll try and predict what it is!")
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
            file = userInput.split(" ")
            img_tensor = preprocess_image(uploaddir + file[len(file) - 1] + filetype)
            classes = model.predict_classes(img_tensor)
            for label, num in label_map.items():
                if num == classes:
                    print("Uhhh... This looks like a(n)", label)

        elif cmd == 99:
            print("Sorry repeat that please")
    else:
        print(answer)
