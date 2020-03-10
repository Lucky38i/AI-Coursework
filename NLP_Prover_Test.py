import nltk

read_expr = nltk.Expression.fromstring
prover = nltk.Prover9()

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
print(prover.prove(sent, assumptions))

