import csv
import os
import pandas as pd

def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

kmer = 3

trainEnhancerPath = "data/layer2/train_strong_enhancers.fa"
trainNonEnhancerPath = "data/layer2/train_weak_enhancers.fa"
testEnhancerPath = "data/layer2/test_strong_enhancers.fa"
testNonEnhancerPath = "data/layer2/test_weak_enhancers.fa"

train = "data/layer2_" + str(kmer) + "mer/train.tsv"
dev = "data/layer2_" + str(kmer) + "mer/dev.tsv"

if not os.path.exists("data/layer2_" + str(kmer) + "mer"):
    os.mkdir("data/layer2_" + str(kmer) + "mer")

trainFile = open(train, 'a')
devFile = open(dev, 'a')

trainFile.write('sequence' + '\t' + 'label' + '\n')
devFile.write('sequence' + '\t' + 'label' + '\n')

with open(trainEnhancerPath) as f:
    reader = csv.reader(f)
    count = 0
    for i, row in enumerate(reader):
        if((i+1)%2 == 0):
            trainFile.write(seq2kmer(str(row).upper()[2:-2], kmer) + '\t' + '1' + '\n')
            count += 1
    print(count)

with open(trainNonEnhancerPath) as f:
    reader = csv.reader(f)
    count = 0
    for i, row in enumerate(reader):
        if((i+1)%2 == 0):
            trainFile.write(seq2kmer(str(row).upper()[2:-2], kmer) + '\t' + '0' + '\n')
            count += 1
    print(count)

with open(testEnhancerPath) as f:
    reader = csv.reader(f)
    count = 0
    for i, row in enumerate(reader):
        if((i+1)%2 == 0):
            devFile.write(seq2kmer(str(row).upper()[2:-2], kmer) + '\t' + '1' + '\n')
            count += 1
    print(count)

with open(testNonEnhancerPath) as f:
    reader = csv.reader(f)
    count = 0
    for i, row in enumerate(reader):
        if((i+1)%2 == 0):
            devFile.write(seq2kmer(str(row).upper()[2:-2], kmer) + '\t' + '0' + '\n')
            count += 1
    print(count)

trainFile.close()
devFile.close()

data = pd.read_csv("data/layer2_" + str(kmer) + "mer/dev.tsv", sep='\t')
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("data/layer2_" + str(kmer) + "mer/dev.tsv", sep='\t', index=False)

data = pd.read_csv("data/layer2_" + str(kmer) + "mer/train.tsv", sep='\t')
data = data.sample(frac=1).reset_index(drop=True)
data.to_csv("data/layer2_" + str(kmer) + "mer/train.tsv", sep='\t', index=False)
