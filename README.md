DataScience
===========
Instructions for interpreting code in the final project:

I'll go through different files in order of their use for processing data. 

1. extract_data.sh
This file is used to run sql queries and save the output to files.  Helpful while making frequent changes to the queries.
2. extract_emai_test.py
This script is called to import the email body from requests in a batches.  It will grab 100 requests at a time, strip the html, and save them to files in groups of 1000.
3. load_data.py
This is where it gets interesting!  In this file there are many different ways to parse data and create features.  When you run this script, you can specify whether you'd like a piece of data rebuilt, or whether you'd prefer to load pre-built data from a pickled file.  This allowed me to parallelize feature extraction and classification training.  I was able to run a classifier while building a different feature set on the next run.  Every time a new set of features is saved, it is pickled to a file for re-loading.
4. exploration(0-3).py
These files are all pretty much the same.  They contain slight differences so that multiple routes could be run simulaneously.  The bulk of the interesting stuff is conatined in the main function, which decides feature sets to use and executes over a loop of parameters different models, logging results to a logging file.
