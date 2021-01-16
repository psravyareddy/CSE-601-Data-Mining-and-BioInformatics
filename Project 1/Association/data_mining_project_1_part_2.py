# -*- coding: utf-8 -*-
"""
Project 1 - Part 2 - Association Analysis

Team members: Sai Hari Charan, Shravya Pentaparthi, Hemant Koti <br>

In this notebook, we will use the Apriori algorithm to find all frequent itemsets for the gene expressions dataset. <br>
"""

import traceback 
import pandas as pd
import numpy as np
import argparse
import itertools
import sys

df_genes = pd.read_csv('association-rule-test-data.txt', sep='\t', header=None)
all_transactions = []

pd.set_option('display.max_rows', 1000)

def parse_args():
    parser = argparse.ArgumentParser(description="Data Mining Project 1 - Dimensionality Reduction and Association Analysis.")
    parser.add_argument("--filepath", type=str, default="association-rule-test-data.txt", help="Path to the association rules dataset")
    parser.add_argument("--support", type=int, default="30", help="Support for Apriori algorithm")
    parser.add_argument("--confidence", type=int, default="70", help="Confidence for Apriori algorithm")
    args = parser.parse_args()
    return args


def getLength1FrequentItems(support):
    frequent_items = set()
    for i in range(len(df_genes.columns)):
        frequent_items = frequent_items.union({label for (label, count) in zip(
            df_genes[i].value_counts().index.tolist(), df_genes[i].value_counts().tolist()) if count >= support})
    return frequent_items


def getLength1Combinations(items):
    return [frozenset(item) for item in list(itertools.combinations(items, 1))]


def getCombinations(items, k):
    return set([item1.union(item2) for item1 in items for item2 in items if len(item1.union(item2)) == k])


def apriori(support=50):
    frequent_items_dict = {}
    print('Support is set to be:', support)

    for i in range(len(df_genes.columns) - 1):
        frequent_item_combinations = getLength1Combinations(getLength1FrequentItems(support)) if i == 0 else getCombinations(frequent_items_temp, i + 1)
        frequent_items_temp = []

        for item in frequent_item_combinations:
            transaction_subset = [transaction for transaction in all_transactions if item.issubset(transaction)]

            if (item != None and len(transaction_subset) >= support):
                frequent_items_temp.append(item)
                frequent_items_dict[str(set(sorted(list(item))))] = len(transaction_subset)

        if (len(frequent_items_temp) > 0):
            print('Number of length ' + str(i + 1) + ' frequent itemsets:', str(len(frequent_items_temp)))
        else:
            break

    print('Number of all lengths frequent itemsets:', len(frequent_items_dict))
    return frequent_items_dict, i


def generateRules(frequent_items_dict, k, minconfidence):
    print('\nConfidence is set to be:', minconfidence)
    generatedrules = pd.DataFrame(data=None, columns=['rule', 'head', 'body', 'confidence'])

    for itemsets in range(k):
        for frequentitem in [eval(item) for item in frequent_items_dict if(len(eval(item)) == itemsets + 1)]:
            for i in range(len(frequentitem) - 1):
                for subset in [set(k) for k in list(itertools.combinations(frequentitem, i + 1))]:
                    confidence = frequent_items_dict[str(set(sorted(list(frequentitem))))]/frequent_items_dict[str(set(sorted(list(frequentitem.difference(subset)))))]
                    if(confidence >= minconfidence):
                        generatedrules.loc[len(generatedrules)] = pd.Series({'rule': str(frequentitem).upper(), 'body': str(subset).upper(), 
                        'head': str(frequentitem.difference(subset)).upper(), 'confidence': confidence})

    return generatedrules.drop_duplicates()


# Code: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html
def asso_rule_template1(query, generatedrules):

    result = pd.DataFrame(data=None, columns=generatedrules.columns)
    rule, count, itemset = [q.strip() for q in query.upper().split('|')]
    itemset = [i.strip() for i in itemset.split(',')]

    print('Template 1 query:', rule, count, itemset)
    rule = rule.lower()
    if (count == "ANY"):
        for item in itemset:
            queryfilter = generatedrules[generatedrules[rule].str.contains(item)]
            result = result.append(queryfilter)
    elif (count == "NONE"):
        result = generatedrules.copy()
        for item in itemset:
            queryfilter = ~result[rule].str.contains(item)
            result = result[queryfilter]
    elif (count.isdigit()):
        for index, _rule in generatedrules.iterrows():
            ruleset = eval(_rule[rule])
            intersect = set(itemset).intersection(ruleset)
            if (len(intersect) == int(count)):
                result.loc[len(result)] = _rule

    return result.drop_duplicates()


def asso_rule_template2(query, generatedrules):
    result = pd.DataFrame(data=None, columns=generatedrules.columns)
    rule, count = [q.strip() for q in query.upper().split('|')]
    count = int(count)

    print('Template 2 query:', rule, count)
    rule = rule.lower()
    
    for i in range(len(generatedrules)):
        if ((rule == "RULE" and len(eval(generatedrules['body'].iloc[i])) + len(eval(generatedrules['head'].iloc[i])) >= count)
            or (len(eval(generatedrules[rule].iloc[i])) >= count)):
             result = result.append(generatedrules.iloc[i])

    return result.drop_duplicates()


def asso_rule_template3(query, generatedrules):    
    result = pd.DataFrame(data=None, columns=generatedrules.columns)
    query = query.split("|")

    if (query[0].lower() == "1or1"):
        query1 = "|".join(query[1:4])
        query2 = "|".join(query[4:7])
        result = asso_rule_template1(query1, generatedrules).append(asso_rule_template1(query2, generatedrules))
    elif (query[0].lower() == "1or2"):
        query1 = "|".join(query[1:4])
        query2 = "|".join(query[4:6])
        result = asso_rule_template1(query1, generatedrules).append(asso_rule_template2(query2, generatedrules))
    elif (query[0].lower() == "2or2"):
        query1 = "|".join(query[1:3])
        query2 = "|".join(query[3:5])
        result = asso_rule_template2(query1, generatedrules).append(asso_rule_template2(query2, generatedrules))        
    elif (query[0].lower() == "1and1"):
        query1 = "|".join(query[1:4])
        query2 = "|".join(query[4:7])
        result = pd.merge(asso_rule_template1(query1, generatedrules), asso_rule_template1(query2, generatedrules), how='inner')
    elif (query[0].lower() == "1and2"):
        query1 = "|".join(query[1:4])
        query2 = "|".join(query[4:6])
        result = pd.merge(asso_rule_template1(query1, generatedrules), asso_rule_template2(query2, generatedrules), how='inner')   
    elif (query[0].lower() == "2and2"):
        query1 = "|".join(query[1:3])
        query2 = "|".join(query[3:5])
        result = pd.merge(asso_rule_template2(query1, generatedrules), asso_rule_template2(query2, generatedrules), how='inner')
    
    print('Template 3 query:', query[0], query1, query2)
    return result.drop_duplicates()


def main():
    try:
        args = parse_args()

        print(df_genes.shape)
        print(df_genes.info())
        print(df_genes.iloc[:, -1].unique())

        # Code: https://stackoverflow.com/a/20027386
        for i in range(len(df_genes.columns) - 1):
            prefix = 'G' + str(i+1) + "_"
            df_genes[i] = prefix + df_genes[i]

        for i in range(len(df_genes)):
            all_transactions.append(set(df_genes.iloc[i]))

        frequent_items_dict, k = apriori(args.support)

        rules = generateRules(frequent_items_dict, k, (args.confidence / 100))
        print(rules)
        print('Number of rules:', len(rules))
        print()

    except Exception as ex:
        print(traceback.print_exc())

    while (True):

        template_number = 0
        try:
            template_number = int(input('Enter the template number or 0 to exit: '))
        except Exception as ex:
            print('Incorrect input entered! Enter an integer value')
            continue

        try:
            if (template_number == 0):
                break
            elif (template_number < 0 or template_number > 3):
                print('Incorrect template number')
                continue

            query = input("Enter '|' separated rule for template " + str(template_number) + ": ")
            if (template_number == 1):
                result = asso_rule_template1(query, rules)
            elif (template_number == 2):
                result = asso_rule_template2(query, rules)
            else:
                result = asso_rule_template3(query, rules)           

            print('Number of rules:', len(result))
            print(result)
            print()
        except Exception as ex:
            print(traceback.print_exc())


if __name__ == "__main__":
    main()

"""
References

Code
1. https://stackoverflow.com/a/20027386
2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html

Readings
1. https://www.hackerearth.com/blog/developers/beginners-tutorial-apriori-algorithm-data-mining-r-implementation/
2. https://www.geeksforgeeks.org/frozenset-in-python/

"""
