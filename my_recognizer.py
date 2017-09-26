import warnings
from asl_data import SinglesData
import math


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    # return probabilities, guesses
    for n in range(test_set.num_items):
        prob = float("-inf")
        
        word_logLvalue = dict()
        X, length = test_set.get_item_Xlengths(n)

        for word, trained_model in models.items():
            try:
                word_logLvalue[word] = trained_model.score(X,length)
            except Exception as inst:
                word_logLvalue[word] = float("-inf")
                
				
            if word_logLvalue[word] > prob:
                prob = word_logLvalue[word]
                guess = word		   		
                
        probabilities.append(word_logLvalue)
        guesses.append(guess)
	
    return probabilities, guesses