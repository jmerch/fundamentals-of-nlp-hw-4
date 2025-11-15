import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.
NEAREST = {
    "a" : ["q", "w", "s", "z"], 
    "e" : ["w", "3", "4", "r", "d", "s", "f"], 
    "i" : ["u", "8", "9", "o", "k", "j"], 
    "o" : ["i", "9", "0", "p", "l", "k"], 
    "u" : ["y", "7", "8", "i", "j", "h"]
}

def get_synonym(word):
    try:
        wordnet.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')

    syns = wordnet.synsets(word)
    if syns:
        syn = wordnet.synsets(word)[0]
        alts = syn.lemmas()
        if len(alts) > 1:
            alternative = alts[1].name()
            return alternative
        else:
            return word
    else:
        return word

def get_typoed(word):
    word = list(word)
    for i in range(len(word)):
        if rand_bool(0.4) and word[i] in "aeiou":
            word[i] = random.choice(NEAREST[word[i]])
    return "".join(word)

def rand_bool(p):
    return random.random() < p
            
def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    words = example["text"].split(" ")
    for i in range(len(words)):
        if rand_bool(0.7): # 50% of words will be modified with one of the transformations
            if rand_bool(0.5):
                words[i] = get_synonym(words[i].lower())
            else:
                words[i] = get_typoed(words[i])
            
    transformed = " ".join(words)
    example["text"] = transformed
    ##### YOUR CODE ENDS HERE ######

    return example
