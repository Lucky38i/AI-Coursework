import nltk

read_expr = nltk.Expression.fromstring
# v = """
# george_town => GT
# bodden_town => BT
# east_end => EE
# north_side => NS
# west_bay => WB
# districts => {GT, WB, BT, EE, NS, WB}
# north_of => {(WB,GT)}
# east_of => {(GT, EE), (GT, BT), (GT, NS), (BT, EE)}
# south_of => {(WB,GT), (NS, BT)}
# west_of => {(EE, NS), (EE, BT), (EE, WB), (EE, GT), (NS, GT), (NS, WB)}
# capital => {GT}
# """
# valuation = nltk.Valuation.fromstring(v)
# domain = {'GT'}
# grammar = nltk.Assignment(valuation.domain, [('x', 'GT')])
# model2 = nltk.Model(valuation.domain, valuation)
#
# symbol = 'george_town'
# fmla1 = read_expr(symbol + '(x) -> districts(x)')
# print(model2.satisfiers(fmla1, 'x', grammar))


domain = {'b', 'o', 'c'}
v = """
bertie => b
olive => o
cyril => c
boy => {b}
girl => {o}
dog => {c}
walk => {o, c}
see => {(b, o), (c, b), (o, c)}
"""
val = nltk.Valuation.fromstring(v)
g = nltk.Assignment(domain, [('x', 'o')])
m = nltk.Model(domain, val)
fmla1 = read_expr('exist x.see(x)')
print(m.satisfiers(fmla1, 'x', g))