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
kern.bootstrap(learnFiles="mybot-basic.xml")
#######################################################
# Welcome user
#######################################################
welcomeMessage = ("Welcome to the WW2 Weather Chat Bot. Please feel free to ask questions about"
                  "The weather during WW2. Curious about the day when the German took over Paris? What about D-Day?"
                  "Ask Away!")
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
    # pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    # activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    # post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1:
            wpage = wiki_wiki.page(params[1])
            if wpage.exists():
                print(wpage.summary)
                print("Learn more at", wpage.canonicalurl)
            else:
                print("Sorry, I don't know what that is.")
        elif cmd == 2:
            succeeded = False
            api_url = r"http://api.openweathermap.org/data/2.5/weather?q="
            response = requests.get(api_url + params[1] + r"&units=metric&APPID=" + APIkey)
            if response.status_code == 200:
                response_json = json.loads(response.content)
                if response_json:
                    t = response_json['main']['temp']
                    tmi = response_json['main']['temp_min']
                    tma = response_json['main']['temp_max']
                    hum = response_json['main']['humidity']
                    wsp = response_json['wind']['speed']
                    wdir = response_json['wind']['deg']
                    conditions = response_json['weather'][0]['description']
                    print("The temperature is", t, "°C, varying between", tmi, "and", tma, "at the moment, humidity is",
                          hum, "%, wind speed ", wsp, "m/s,", conditions)
                    succeeded = True
            if not succeeded:
                print("Sorry, I could not resolve the location you gave me.")
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)
