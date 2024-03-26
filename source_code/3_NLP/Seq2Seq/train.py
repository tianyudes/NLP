import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from preprocessing import *
from network import *

def main():
    n_iters       = 75000
    learning_rate = 0.01 * 0.8
    embedding_size = 256
    hidden_size   = 256
    max_length    = 30

    input_lang  = Preprocessing( 'jpn.txt' )
    output_lang = Preprocessing( 'eng.txt')
                                                                                                                                                                                                                                                                                                                               
    allow_list = [x and y for (x,y) in zip( input_lang.get_allow_list( max_length ), output_lang.get_allow_list( max_length ) ) ]
                                                                                                                                                                                                                                                                                                                                        
    input_lang.load_file( allow_list )
    output_lang.load_file( allow_list )
                                                                                                                                                                                                                                                                                                                                                                              
    encoder           = Encoder( input_lang.n_words, embedding_size, hidden_size ).to( device )
    decoder           = Decoder( hidden_size, embedding_size, output_lang.n_words ).to( device )
                                                                                                                                                                                                                                                                                                                                                                                      
    encoder_optimizer = optim.SGD( encoder.parameters(), lr=learning_rate )
    decoder_optimizer = optim.SGD( decoder.parameters(), lr=learning_rate )
                                                                                                                                                                                                                                                                                                                
    training_pairs = [ tensorsFromPair( input_lang, output_lang ) for i in range( n_iters ) ]
                                                                                                                                                                                                                                                                                                                                                                                            
    criterion      = nn.NLLLoss()

    for epoch in range( 1, n_iters + 1):
                                                                                                                                                                                                                                                                                                                                                                      
        input_tensor, output_tensor = training_pairs[ epoch - 1 ]
                                                                                                                                                                                                                                                                                                                                                                                  
        encoder_hidden              = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_length  = input_tensor.size(0)
        output_length = output_tensor.size(0)

        # Encoder phese                                                                                                                                                                                                                                                                                                                                                                                     
        for i in range( input_length ):
            encoder_output, encoder_hidden = encoder( input_tensor[ i ], encoder_hidden )

        # Decoder phese                                                                                                                                                                                                                                                                                                                                                                                     
        loss = 0
        decoder_input  = torch.tensor( [ [ SOS_token ] ] ).to( device )
        decoder_hidden = encoder_hidden
        for i in range( output_length ):
            decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden )                                                                                                                                                                                                                                                                                                                                                                           
            decoder_input = output_tensor[ i ]
                                                                                                                                                                                                                                                                                                          
            if random.random() < 0.5:                                                                                                                                                                                                                                                                                                                                                           
                topv, topi                     = decoder_output.topk( 1 )                                                                                                                                                                                                                                                                                                                                                
                decoder_input                  = topi.squeeze().detach()

                                                                                                                                                                                                                                                                                                                                                                                     
            loss += criterion( decoder_output, output_tensor[ i ] )
                                                                                                                                                                                                                                                                                                                                                                  
            if decoder_input.item() == EOS_token: break
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
                                                                                                                                                                                                                                                                                                                                                                                    
        if epoch % 50 == 0:
            print( "[epoch num %d (%d)] [ loss: %f]" % ( epoch, n_iters, loss.item() / output_length ) )