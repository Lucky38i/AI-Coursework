import nltk

value = """
George Town => GT
Bodden Town => BT
East End => EE
North Side => NS
West Bay => WB
Districts => {GT, WB, BT, EE, NS, WB}
north_of => {(WB,GT)}
east_of => {(GT, EE), (GT, BT), (GT, NS), (BT, EE)}
south_of => {(WB,GT), (NS, BT)}
west_of => {(EE, NS), (EE, BT), (EE, WB), (EE, GT), (NS, GT), (NS, WB)}
Capital => {GT}
"""

read_expr = nltk.sem.Expression.fromstring

valDict = nltk.Valuation.fromstring(value)
grammar_file = 'data/grammars/DistrictGrammar.fcfg'

grammar = nltk.Assignment(valDict.domain)
R = nltk.sem.Expression.fromstring('all x. all y. (north_of(x, y) -> -north_of(y, x))')
test = nltk.sem.Expression.fromstring('-north_of(GT, WB)')
a1 = read_expr('west_of(EE, NS)')
prover = nltk.Prover9()
print (prover.prove(test, [R, a1]))



