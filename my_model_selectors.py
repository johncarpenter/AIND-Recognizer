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
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
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

        # TODO implement model selection based on BIC scores

        bic_score = float('inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):
            model = self.base_model(n_components)

            #skip if the model isn't there
            if model is None:
                #print("Missing Model")
                continue

            #hmm errors in some version? from forums
            try:
                logL = model.score(self.X, self.lengths)
            except:
                continue



            # Not sure about this one. Worked it out from here;
            # https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
            # and from https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/11
            # Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
            is_prob = n_components
            tr_prob = n_components * (n_components-1)
            em_prob = n_components * self.X.shape[1]*2

            #p = is_prob + tr_prob + em_prob

            # Based on feedback from another student, not sure why it works?
            p = n_components**2 + (2 * len(self.X[0]) * n_components)

            logN = np.log(len(self.X))

            BIC = -2 * logL + p * logN

            if  BIC < bic_score:
                bic_score = BIC
                best_model = model


        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        dic_score = float('-inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):

            model = self.base_model(n_components)

            try:
                logL = model.score(self.X, self.lengths)
            except:
                continue

            partial_list = []
            for word in self.hwords:
                if word != self.this_word :
                    h_X, h_lengths = self.hwords[word]
                    try:
                        partial_list.append(model.score(h_X, h_lengths))
                    except:
                        print("Model error")
                        continue

            DIC = logL - np.average(partial_list)

            #dic_score, best_model = max((dic_score,best_model),(DIC,model))
            if  DIC > dic_score:
                dic_score = DIC
                best_model = model
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        n_splits = 3
        if len(self.sequences) < 2:
            return None
        elif len(self.sequences) == 2:
            n_splits = 2

        cv_score = float('-inf')
        best_model = None
        split_method = KFold(n_splits=n_splits)

        for n_components in range(self.min_n_components, self.max_n_components+1):

            partial_list = []

            for train_idx, test_idx in split_method.split(self.sequences):

                # Combine training sequences
                self.X, self.lengths = combine_sequences(train_idx, self.sequences)

                model = self.base_model(n_components)

                test_X, test_lengths = combine_sequences(test_idx, self.sequences)

                try:
                    partial_list.append(model.score(test_X, test_lengths))
                except Exception as e:
                    #print("Model Errors: {}".format(e))
                    continue

            if model is None or partial_list == []:
                continue


            score = np.mean(partial_list)
            if  score > cv_score:
                cv_score = score
                best_model = model

        return best_model
