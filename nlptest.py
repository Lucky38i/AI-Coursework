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

nltk.download_gui()


read_expr = nltk.sem.Expression.fromstring
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
# print(val)

g = nltk.Assignment(domain, [('x', 'o'), ('y', 'c')])
# print(g)

model = nltk.Model(domain, val)

fmla1 = read_expr('girl(x) | boy(x)')
fmla2 = read_expr('girl(x) -> walk(x)')
fmla3 = read_expr('walk(x) -> girl(x)')

v2 = """
bruce => b
elspeth => e
julia => j
matthew => m
person => {b, e, j, m}
admire => {(b,e), (m, j)}
"""
val2 = nltk.Valuation.fromstring(v2)
domain2 = val2.domain
model2 = nltk.Model(domain2, val2)
g2 = nltk.Assignment(domain2)

