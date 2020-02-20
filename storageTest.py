import nltk

v = """
lettuce => {}
cabbages => {}
mustards => {}
potatoes => {}
onion => {}
carrots => {}
beans => {}
peas => {}
field1 => {}
field2 => {}
field3 => {}
field4 => {}
be_in => {}
"""
valDict = nltk.Valuation.fromstring(v)
grammar_file = 'data/grammars/simple-sem.fcfg'
objectCounter = 0
sent = ""



