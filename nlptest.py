import nltk

# fs0 = nltk.FeatStruct("""[NAME=Lee,
#                             ADDRESS=[NUMBER=74,
#                                 STREET='rue Pascal'],
#                             SPOUSE=[NAME=Kim,
#                                 ADDRESS=[NUMBER=74,
#                                             STREET='rue Pascal']]]""")
# fs1 = nltk.FeatStruct("[SPOUSE = [ADDRESS = [CITY = Paris]]]")
# fs2 = nltk.FeatStruct("""[NAME=Lee, ADDRESS=(1)[NUMBER=74, STREET='rue Pascal'],
#                             SPOUSE=[NAME=Kim, ADDRESS->(1)]]""")
#
# # print(fs0)
# # print(fs1.unify(fs0))
# print(fs1.unify(fs2))

# fs1 = nltk.FeatStruct("[ADDRESS1=[NUMBER=74, STREET='rue Pascal']]")
# fs2 = nltk.FeatStruct("[ADDRESS1=?x, ADDRESS2=?x]")
# print(fs2)
# print("\n")
# print(fs2.unify(fs1))

# fs1 = nltk.FeatStruct("[A = ?x, B= [C = ?x]]")
# fs2 = nltk.FeatStruct("[B = [D = d]]")
# fs3 = nltk.FeatStruct("[B = [C = d]]")
# print(fs3.unify(fs1))
# cp = nltk.load_parser('data/grammars/sq10pop.fcfg', trace=3)
# query = 'What cities are located in China'
# query1 = 'What cities have populations above 1,000,000'
# trees = list(cp.parse(query.split()))
# answer = trees[0].label()['SEM']
# answer = [s for s in answer if s]
# q = ' '.join(answer)
# print(q)

# nltk.download_gui()


v = """
George Town => GT
Bodden Town => BT
East End => EE
North Side => NS
West Bay => WB
districts => {GT, WB, BT, EE, NS, WB}
south_of => {(WB,GT), (NS, BT)}
west_of => {(EE, NS), (EE, BT), (EE, WB), (EE, GT), (NS, GT), (NS, WB)}
"""
read_expr = nltk.Expression.fromstring
prover = nltk.Prover9()

valuation = nltk.Valuation.fromstring(v)
domain = valuation.domain
model = nltk.Model(domain, valuation)
grammar = nltk.Assignment(domain)
cp = nltk.load_parser('data/grammars/sq10.fcfg',trace=3)

north_assum = read_expr('all x. all y. (north_of(x, y) -> -north_of(y, x))')
north_a1 = read_expr('north_of(GT, WB)')

east_assum = read_expr('all x. all y. (east_of(x, y) -> -east_of(y, x))')
east_a1 = read_expr('east_of(GT,EE)')
east_a2 = read_expr('east_of(GT, BT)')
east_a3 = read_expr('east_of(GT, NS)')
east_a4 = read_expr('east_of(BT, EE)')

west_assum = read_expr('all x. all y. (west_of(x, y) -> -west_of(y, x))')
west_a1 = read_expr('west_of(EE, NS)')
west_a2 = read_expr('west_of(EE, BT)')
west_a3 = read_expr('west_of(EE, WB)')
west_a4 = read_expr('west_of(EE, GT)')
west_a5 = read_expr('west_of(NS, GT)')
west_a6 = read_expr('west_of(NS, WB)')

south_assum = read_expr('all x. all y. (south_of(x, y) -> -south_of(y, x))')
south_a1 = read_expr('south_of(WB,GT)')
south_a2 = read_expr('south_of(NS, BT)')

assumptions = [east_assum, east_a1, east_a2, east_a3, east_a4,
               north_assum, north_a1,
               west_assum, west_a1, west_a2, west_a3, west_a4, west_a5, west_a6,
               south_assum, south_a1, south_a2]

sent = read_expr('north_of(EE,GT)')
# print(prover.prove(sent, assumptions))

query = 'which city is north of west bay'








# trees = list(cp.parse(query.split()))
# answer = trees[0].label()['SEM']
# answer = [s for s in answer if s]
# q = ' '.join(answer)
# print(q)


