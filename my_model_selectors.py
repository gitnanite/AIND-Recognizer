import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        logN = np.log(self.X.shape[0])
        d = self.X.shape[1]
        n_components = self.min_n_components
        model = self.base_model(n_components) 
        bic_score = float("inf")
		#bic_score = - 2 * model.score(self.X, self.lengths)	+ (n_components**2 + 2 * d * n_components-1) * logN)
        if not bic_score: return self.base_model(n_components)
	
        for n in range(self.min_n_components, self.max_n_components+1):
		
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
			    # per Dana Sheahen on slack
		        # number of parameters is given by p = n^2 + 2*d*n - 1  
				#, with n: number of features, d :number of HMM states
		        # https://ai-nd.slack.com/archives/C4GQUB39T/p1491489096823164
                p =  n**2 + 2 * d * n - 1
                BIC = -2 * logL + p * logN
                if BIC < bic_score:
                    n_components = n
                    bic_score = BIC
            except Exception as inst:
                continue
        return self.base_model(n_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
 
        # implement model selection based on DIC scores
        n_components = self.min_n_components
        model = self.base_model(n_components)
        M = len(self.hwords)
        dic_score = float("-inf")
		
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                for hword in self.hwords:
                    logL = model.score(self.X, self.lengths)
                    logAll_Minusi = 0
                    if hword != self.this_word:
                        X_hword, lengths_hword = self.hwords[hword]
                        logAll_Minusi += model.score(X_hword, lengths_hword)
        
                DIC = logL - logAll_Minusi/(M-1)
                if DIC > dic_score:
                    n_components = n
                    dic_score = DIC
            except Exception as inst:
                if self.verbose:
                    print('An error occured in DIC: ')
                    print(inst)
                continue
            	
        return self.base_model(n_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        n_components = self.min_n_components+1
        cv_score = float("-inf")
        cv_model = self.base_model(n_components)
        		
        if len(self.sequences) < 3: 
		    # not training on smaller samples
            return cv_model
        else:
            n_splits = 3
		
		# code taken in part from http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        kf = KFold(n_splits=n_splits)
        for n in range(self.min_n_components,self.max_n_components+1):
            model = self.base_model(n)
            scores = []
            
            for train_index, test_index in kf.split(self.sequences):
                X_train, train_lengths = combine_sequences(train_index,self.sequences)
                X_test, test_lengths = combine_sequences(test_index,self.sequences)
            
                try:
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter =1000,random_state=self.random_state, verbose=False).fit(X_train, train_lengths)
                    score = model.score(X_test, test_lengths)
                    scores.append(score)
                except Exception as inst:
                    if self.verbose:
                        print("An error occurred in CV:")
                        print(inst)
                    #return self.base_model(n)
                    continue

            CV = np.mean(scores)
            if CV > cv_score:
                n_components = n
                cv_score = CV
		
        return self.base_model(n_components)
