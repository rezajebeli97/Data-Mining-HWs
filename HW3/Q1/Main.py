from HW3.Q1.Q1 import Apriori, Rule

############################################################################ Q1_part1
transactions = [
        ['a', 'b', 'd', 'g'],
        ['b', 'd', 'e'],
        ['a', 'b', 'c', 'e', 'f'],
        ['b', 'd', 'e', 'g'],
        ['a', 'b', 'c', 'e', 'f'],
        ['b', 'e', 'g'],
        ['a', 'c', 'd', 'e'],
        ['b', 'e'],
        ['a', 'b', 'e', 'f'],
        ['a', 'c', 'd', 'e']
    ]

availableItems = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

ap = Apriori(transactions, availableItems)


frequent_itemsets = ap.extract_items(4, 7)      #a
print(frequent_itemsets[1])                     #a

frequent_itemsets = ap.extract_items(4, 7)      #b
print(frequent_itemsets[2])                     #b

frequent_itemsets = ap.extract_items(7, 7)      #c
print(frequent_itemsets[2])                     #c

frequent_itemsets = ap.extract_items(4, 7)                  #d
confident_rules = ap.extract_rules(1, frequent_itemsets, 6) #d
print([str(x) for x in confident_rules[0]])                 #d
print([str(x) for x in confident_rules[1]])                 #d

conf = ap.confidence(Rule({'b'}, {'e'}))    #e
print(conf)                                 #e



################################################################################# Q1_part2
transactions = [
        ['a', 'c', 'd'],
        ['b', 'c', 'e'],
        ['a', 'b', 'c', 'e'],
        ['b', 'e']
    ]

availableItems = ['a', 'b', 'c', 'd', 'e']

ap = Apriori(transactions, availableItems)


frequent_itemsets = ap.extract_items(2, 5)      #a
print(frequent_itemsets[1])                     #a
print(frequent_itemsets[2])                     #a
print(frequent_itemsets[3])                     #a

frequent_itemsets = ap.extract_items(2, 5)                                          #b
confident_rules = ap.extract_rules(0.65, frequent_itemsets, 4)                      #b
print(len(confident_rules[0]) + len(confident_rules[1]) + len(confident_rules[2]))  #b

frequent_itemsets = ap.extract_items(2, 5)                      #c
confident_rules = ap.extract_rules(0.8, frequent_itemsets, 4)   #c
print([str(x) for x in confident_rules[0]])                     #c

conf = ap.confidence(Rule({'e'}, {'c'}))    #d
print(conf)                                 #d

supp = ap.support({'b', 'c'})       #e
print(supp)                         #e
