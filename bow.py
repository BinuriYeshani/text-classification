#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:33:55 2019

@author: binuri
"""
import codecs
import fetchData
import collections, re

dirName = '/home/binuri/work_Msc/IR_Assignment3/20_newsgroup';
listOfFiles = fetchData.main(dirName)
fileList = list()
for f in listOfFiles:
    fileList.append(codecs.open(f, "r", encoding="ISO-8859-1").read())

#texts = ['John likes to watch movies. Mary likes too.',
#   'John also likes to watch football games.']
    
texts = fileList
bagsofwords = [ collections.Counter(re.findall(r'\w+', txt))
            for txt in texts]


print(bagsofwords)