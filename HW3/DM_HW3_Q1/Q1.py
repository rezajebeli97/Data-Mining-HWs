import copy

class Rule:
    left_items: {} = None
    right_items: {} = None

    def __init__(self, left_items, right_items):
        self.left_items = left_items
        self.right_items = right_items

    def __eq__(self, other):
        if self.left_items == other.left_items and self.right_items == other.right_items:
            return True
        else:
            return False

    def __str__(self):
        return str(self.left_items) + " -> " + str(self.right_items)



class Apriori:
    def __init__(self,transactions, availableItems):
        self.transactions = transactions
        self.availableItems = availableItems


    def extract_items(self, support_threshold , depth):
        frequent_itemsets = []
        frequent_itemsets.append([])
        for i in range(1, depth+1):
            if i == 1:
                frequent_itemsets_in_this_depth = []
                for item in self.availableItems:
                    itemSet = {item}
                    if self.support(itemSet) >= support_threshold:
                        frequent_itemsets_in_this_depth.append(itemSet)
            else:
                frequent_itemsets_in_this_depth = []
                for f in frequent_itemsets[i - 1]:
                    for item in self.availableItems:
                        frequent_itemset_in_last_depth = copy.deepcopy(f)
                        if item in frequent_itemset_in_last_depth:
                            continue
                        frequent_itemset_in_last_depth.add(item)
                        itemSet = frequent_itemset_in_last_depth
                        if self.support(itemSet) >= support_threshold and itemSet not in frequent_itemsets_in_this_depth:
                            frequent_itemsets_in_this_depth.append(itemSet)
            frequent_itemsets.append(frequent_itemsets_in_this_depth)
        return frequent_itemsets

    def extract_rules(self,confidence_threshold, frequent_itemsets, depth):
        frequent_rules = []

        frequent_rules_depth1 = []
        for itemSetRow in frequent_itemsets:
            for itemSet in itemSetRow:
                if len(itemSet) < 2:
                    continue
                for item in itemSet:
                    tempItemSet = copy.deepcopy(itemSet)
                    tempItemSet.remove(item)
                    rule = Rule(tempItemSet, {item})
                    if self.confidence(rule) >= confidence_threshold:
                        frequent_rules_depth1.append(rule)
        frequent_rules.append(frequent_rules_depth1)

        for i in range(1, depth):
            frequent_rules_depthi = []
            for rule in frequent_rules[i-1]:
                if len(rule.left_items) == 1:
                    continue
                for item in rule.left_items:
                    tempRule = copy.deepcopy(rule)
                    tempRule.left_items.remove(item)
                    tempRule.right_items.add(item)
                    if self.confidence(tempRule) >= confidence_threshold:
                        if tempRule not in frequent_rules_depthi:
                            frequent_rules_depthi.append(tempRule)
            frequent_rules.append(frequent_rules_depthi)

        return frequent_rules


    def confidence(self, rule):
        left = 0
        all = 0
        for transaction in self.transactions:
            tmp = True
            for item in rule.left_items:
                if item not in transaction:
                    tmp = False
                    break
            if tmp:
                left+=1
                tmp2 = True
                for item2 in rule.right_items:
                    if item2 not in transaction:
                        tmp2 = False
                        break
                if tmp2:
                    all+=1
        return all / left

    def support(self, itemSet):
        support = 0
        for transaction in self.transactions:
            tmp = True
            for item in itemSet:
                if item not in transaction:
                    tmp = False
                    break
            if tmp:
                support+=1
        return support