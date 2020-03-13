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

from ImageClassification_Training import preprocess_image
import Transformer_Training as tt

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
#  Transformer Model Interface
#######################################################
encoder_path = "data/checkpoints/encoder_epoch_68.h5"
decoder_path = "data/checkpoints/decoder_epoch_68.h5"

dataset, info = tt.create_dataset(tt.BATCH_SIZE)
encoder, decoder = tt.create_transformer(info['vocab_size'], tt.MODEL_SIZE,
                                         info['max_length'], tt.NUM_LAYERS, tt.ATTENTION_HEADS)
encoder.load_weights(encoder_path)
decoder.load_weights(decoder_path)

print(info)

#######################################################
#  POL model Interface
#######################################################
val = """
avocado => {}
mango => {}
bananas => {}
breadfruit => {}
starfruit => {}
guinep => {}
ackee => {}
limes => {}
coconut => {}
tamarind => {}
farm => f1 
backyard => b1
farm2 => f2
be_in => {} 
George Town => GT
Bodden Town => BT
East End => EE
North Side => NS
West Bay => WB
districts => {GT, WB, BT, EE, NS, WB}
south_of => {(WB,GT), (NS, BT)}
west_of => {(EE, NS), (EE, BT), (EE, WB), (EE, GT), (NS, GT), (NS, WB)}
"""
folval = nltk.Valuation.fromstring(val)
grammar_file = 'data/grammars/CIGrammarFile.fcfg'
objectCounter = 0


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
startup_dir = "data/std-startup.xml"

kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles=startup_dir, commands="LOAD BRAIN")

#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the Cayman Islands Chat Bot!"
                  "\nAsk me anything about the Cayman islands including"
                  " it's geography, national animals and so on."
                  "\n\n** NEW UPDATE **"
                  "\nMy owner has implemented a Transformer Model to train me on the SQuAD"
                  "\nDataset!, refer to their explore page for questions you can ask me.")
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

        elif cmd == 4:
            o = 'o' + str(objectCounter)
            objectCounter += 1
            folval['o' + o] = o  # insert constant
            try:
                if len(folval[params[1]]) == 1:  # clean up if necessary
                    if ('',) in folval[params[1]]:
                        folval[params[1]].clear()
                folval[params[1]].add((o,))  # insert type of fruit information
                if len(folval["be_in"]) == 1:  # clean up if necessary
                    if ('',) in folval["be_in"]:
                        folval["be_in"].clear()
                folval["be_in"].add((o, folval[params[2]]))  # insert location
            except nltk.sem.evaluate.Undefined:
                print("Those are invalid objects, try again")

        elif cmd == 5 or cmd == 6:  # Are there any x in y or # Are all x in y
            try:
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                allOrSome = 'some '
                if cmd == 6:
                    allOrSome = 'all '
                sent = allOrSome + params[1] + ' are_in ' + params[2]
                results = nltk.evaluate_sents([sent], grammar_file, m, g)[0][0]
                if results[2]:
                    print("Yes.")
                else:
                    print("No.")
            except ValueError:
                print("Those are invalid objects, try checking for something else")

        elif cmd == 7:  # Which fruits are in ...
            try:
                g = nltk.Assignment(folval.domain)
                m = nltk.Model(folval.domain, folval)
                e = nltk.Expression.fromstring("be_in(x," + params[1] + ")")
                results = m.satisfiers(e, "x", g)
                if len(results) == 0:
                    print("There are none, sorry!")
                else:
                    # find satisfying objects in the valuation dictionary,
                    # #and print their type names
                    sol = folval.values()
                    for so in results:
                        for symbol, value in folval.items():
                            if len(value) > 0:
                                valList = list(value)
                                if len(valList[0]) == 1:
                                    for i in valList:
                                        if i[0] == so:
                                            print(symbol)
                                            break
            except nltk.sem.evaluate.Undefined:
                print("That's not a valid place, try again")
        elif cmd == 8:  # All districts in cayman
            grammar = nltk.Assignment(folval.domain)
            model = nltk.Model(folval.domain, folval)
            expression = nltk.Expression.fromstring("districts(x)")
            results = model.satisfiers(expression, "x", grammar)
            sol = folval.values()
            for result in results:
                for symbol, value in folval.items():
                    if len(value) > 0:
                        if value == result:
                            print(symbol)
                            break
        # Manage image classification
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
            input_sentence = [userInput]
            tt.predict(encoder, decoder, info['tokenizer'], input_sentence, info['max_length'])
    else:
        print(answer)
