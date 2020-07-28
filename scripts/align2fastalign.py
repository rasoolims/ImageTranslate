import os
import sys

r1 = open(os.path.abspath(sys.argv[1]), 'r')
r2 = open(os.path.abspath(sys.argv[2]), 'r')
w = open(os.path.abspath(sys.argv[3]), 'w')

l1 = r1.readline()
while l1:
    l2 = r2.readline()
    w.write(l1.strip() + ' ||| ' + l2.strip() + '\n')
    l1 = r1.readline()
w.close()
