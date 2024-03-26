import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

SOS_token = 0
EOS_token = 1

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cuda:0')


class Preprocessing:
    def __init__(self, filename):
        self.filename = filename
        self.word2index = {}
        self.word2count = {}

        self.sentences = []
        self.index2word = {
            0: "SOS", 1: "EOS"
        }
        self.n_words = 2

        with open(self.filename) as file:
            for i, line in enumerate(file.readlines()):
                line = line.strip()
                self.sentences.append(line)
        
        self.allow_list = [True] * len(self.sentences)
        self.target_sentences = self.sentences[::]

    def get_sentences(self):
        return self.sentences[::]
    
    def get_sentence(self, index):
        return self.sentences[index]
    
    def choice(self):
        while True:
            index = random.randint(0, len(self.sentences) - 1)
            if self.allow_list[index]:
                break
            return self.sentences[index], index
        
    def generate_allow_list(self, max_length):
        allow_list = []

        for sentence in self.sentences:
            if len(sentence.split()) < max_length:
                allow_list.append(True)
            else: 
                allow_list.append(False)

        return allow_list
    
    def addSentence( self, sentence ):
        for word in sentence.split():
            self.addWord(word)
    def addWord( self, word ):
        if word not in self.word2index:
            self.word2index[ word ] = self.n_words
            self.word2count[ word ] = 1
            self.index2word[ self.n_words ] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def load_file(self, allow_list=[]):
        if allow_list:
            self.allow_list = [x and y for (x,y) in zip(self.allow_list, allow_list)]
        
        self.target_sentences =[]
        for i, sentence in enumerate(self.sentences):
            if self.allow_list[i]:
                self.addSentence(sentence)
                self.target_sentences.append(sentence)

def tensorFromSentence( lang, sentence ):
    indexes = [ lang.word2index[ word ] for word in sentence.split(' ') ]
    indexes.append( EOS_token )
    return torch.tensor( indexes, dtype=torch.long ).to( device ).view(-1, 1)
    
def tensorsFromPair( input_lang, output_lang ):
    input_sentence, index = input_lang.choice()
    output_sentence       = output_lang.get_sentence( index )
    input_tensor  = tensorFromSentence( input_lang, input_sentence )
    output_tensor = tensorFromSentence( output_lang, output_sentence )
    return (input_tensor, output_tensor)    




            