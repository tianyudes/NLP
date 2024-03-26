# defination of encoder and decoder
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device('cuda:0')


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size).to(device)

    def forward(self, input_word, hidden_state):
        word_embedding = self.embedding(input_word).view( 1, 1, -1 )
        out, new_hidden = self.gru(word_embedding, hidden_state)

        return out, new_hidden
    

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        # refer to the input of the network
        self.gru= nn.GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim = 1)

    def forword(self, input_word, hidden_state):
        embedding = self.embedding(input_word).view( 1, 1, -1 )

        relu_embedding = F.relu(embedding)

        gru_output, hidden_state = self.gru(relu_embedding, hidden_state)
        result = self.softmax(gru_output[0])
        return result, hidden_state
    
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size).to(device)
    


