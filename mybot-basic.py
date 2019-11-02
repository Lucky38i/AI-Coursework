#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic chatbot design --- for your own modifications
"""
#######################################################
# Initialise Wikipedia agent
#######################################################
import aiml
import json
import requests
import wikipediaapi
import configparser
import os

#######################################################
# Initialise config parser
#######################################################
config = configparser.ConfigParser()
config.read('config.ini')

wiki_wiki = wikipediaapi.Wikipedia('en')
wikipediaapi.log.setLevel(level=wikipediaapi.logging.ERROR)

#######################################################
# Initialise weather agent
#######################################################
APIkey = config['OPENWEATHERMAP']['ApiKey']

#######################################################
#  Initialise AIML agent
#######################################################
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
# Use the Kernel's bootstrap() method to initialize the Kernel. The
# optional learnFiles argument is a file (or list of files) to load.
# The optional commands argument is a command (or list of commands)
# to run after the files are loaded.
# The optional brainFile argument specifies a brain file to load.
brain = "brain.brn"
if os.path.isfile(brain):
    kern.bootstrap(brainFile=brain)
else:
    kern.bootstrap(learnFiles="std-startup.xml", commands="LOAD AIML")
    kern.saveBrain(brain)

#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the Forrest Gump Foodie Chat Bot"
                  "Ask me anything about Bubba Buford's wild list of shrimp"
                  "creations!")
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
            # TODO implement NLTK
            print("Test")
            break
        elif cmd == 99:
            print("Sorry repeat that please")
    else:
        print(answer)
