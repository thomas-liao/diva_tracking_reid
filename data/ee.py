import csv

f = open('market1501_train.csv', 'r')
a = set()
for line in f:
  idx, _ = line.split(',')
  a.add(idx)
print(len(list(a)))

