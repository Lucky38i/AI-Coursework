#######################################################
# Imports
#######################################################
import aiml
import string
import wikipediaapi
import configparser
import os
import io
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Unsafe
warnings.filterwarnings("ignore")

#######################################################
# Initialise config parser
#######################################################
config = configparser.ConfigParser()
config.read('config.ini')

wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
# Initialise NLTK Agent
#######################################################
data = "data.txt"
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
brain = "brain.brn"
if os.path.isfile(brain):
    kern.bootstrap(brainFile=brain)
else:
    kern.bootstrap(learnFiles="std-startup.xml", commands="LOAD BRAIN")
    kern.saveBrain(brain)

#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the Cayman Islands Chat Bot!"
                  "Ask me anything about the Cayman islands including"
                  "it's geography, national animals and so on")
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
        elif cmd == 99:
            print("Sorry repeat that please")
    else:
        print(answer)
