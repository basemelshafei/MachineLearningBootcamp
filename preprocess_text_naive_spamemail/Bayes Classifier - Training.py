# Notebook Imports
import pandas as pd
import numpy as np

# Constants
VOCAB_SIZE = 2500

TRAINING_DATA_FILE = '02_Training/train-data.txt'
TEST_DATA_FILE = '02_Training/test-data.txt'

TOKEN_SPAM_PROB_FILE = '03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = '03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = '03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = '03_Testing/test-features.txt'
TEST_TARGET_FILE = '03_Testing/test-target.txt'

# Read and load features from .txt Files into NumPy array

sparse_train_data = np.loadtxt(TRAINING_DATA_FILE, delimiter=' ', dtype=int)
sparse_test_data = np.loadtxt(TEST_DATA_FILE, delimiter=' ', dtype=int)
# To get an idea of the size of this data, lets print the number of rows in this array
# We have done this with a dataframe now we do this using an array
print('Nr of rows in the training file:', sparse_train_data.shape[0])
print('Nr of rows in the testing file:', sparse_test_data.shape[0])

print('Nr of emails in training file:', np.unique(sparse_train_data[:, 0]).size)
print('Nr of emails in testing file:', np.unique(sparse_test_data[:, 0]).size)

# How to crate an empty dataframe
column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
print(column_names[:5])
len(column_names)

index_names = np.unique(sparse_train_data[:, 0])
print(index_names)

full_train_data = pd.DataFrame(index=index_names, columns=column_names)
full_train_data.fillna(value=0, inplace=True)
full_train_data.head()

# Create a Full Matrix from sparse Matrix
def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, cat_idx=2, freq_idx=3):
    """
    Form a full matrix from a sparse matrix. Return a pandas dataframe.
    Keyword arguments:
    sparse_matrix -- numpy array
    nr_words -- size of the vocabulary. Total number of tokens.
    doc_idx -- position of the document id in the sparse matrix. Default: 1st column
    word_idx -- position of the word id in the sparse matrix. Default: 2nd column
    cat_idx -- position of the label (spam is 1, nonspam is 0). Default: 3rd column
    freq_idx -- position of occurrence of word in sparse matrix. Default: 4th column
    """
    column_names = ['DOC_ID'] + ['CATEGORY'] + list(range(0, VOCAB_SIZE))
    doc_id_names = np.unique(sparse_matrix[:, 0])
    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)

    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        label = sparse_matrix[i][cat_idx]
        occurrence = sparse_matrix[i][freq_idx]
        # Basically alloting each value, and use declared variables in function defn so that changing them is easy
        # Keeping the i same we are going through the compelete row using each column name for each word and storing data

        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, 'CATEGORY'] = label
        full_matrix.at[doc_nr, word_id] = occurrence
        # Here we are filling the values into the empty data frame

    full_matrix.set_index('DOC_ID', inplace=True)
    return full_matrix


full_train_data = make_full_matrix(sparse_train_data, VOCAB_SIZE)
print(full_train_data.head())

# Training the Naive Bayes Model
# Calculating the Probability of Spam

print(full_train_data.CATEGORY.size)
print(full_train_data.CATEGORY.sum())

prob_spam = full_train_data.CATEGORY.sum() / full_train_data.CATEGORY.size
print('Probability of spam is', prob_spam)

# Very smart method for this, as category 1 caters to spam
# Total Number of Words / Tokens
full_train_features = full_train_data.loc[:, full_train_data.columns != 'CATEGORY']
# just removing the categories column, and then generating a dataframe
full_train_features.head()

email_lengths = full_train_features.sum(axis=1)
print(email_lengths.shape)

print(email_lengths[:5])
# Summing up row wise, to get total number of tokens in a particular email

total_wc = email_lengths.sum()
print(total_wc)
# Summing up all the rows to get total number of tokens we have

# Number of Tokens in Spam & Ham Emails
spam_lengths = email_lengths[full_train_data.CATEGORY == 1]
print(spam_lengths.shape)

spam_wc = spam_lengths.sum()
print(spam_wc)

ham_lengths = email_lengths[full_train_data.CATEGORY == 0]
print(ham_lengths.shape)

nonspam_wc = ham_lengths.sum()
print(nonspam_wc)

# Confirming things
print(email_lengths.shape[0] - spam_lengths.shape[0] - ham_lengths.shape[0])

print(spam_wc + nonspam_wc - total_wc)

print('Average nr of words in spam emails {:.0f}'.format(spam_wc / spam_lengths.shape[0]))
print('Average nr of words in ham emails {:.3f}'.format(nonspam_wc / ham_lengths.shape[0]))

# THIS 0 AND 3 control the numbers after decimal points
# Summing the Tokens Occuring in Spam
train_spam_tokens = full_train_features.loc[full_train_data.CATEGORY == 1]
train_spam_tokens.head()
train_spam_tokens.tail()

# Summing in columns
summed_spam_tokens = train_spam_tokens.sum(axis=0) + 1
# adding 1 to make all calculation as non 0 while applying bayes
# Easy to do and does not change anything much
print(summed_spam_tokens)

# Summing the Tokens Occuring in Ham
train_ham_tokens = full_train_features.loc[full_train_data.CATEGORY == 0]
summed_ham_tokens = train_ham_tokens.sum(axis=0) + 1
print(summed_ham_tokens)

# P(Token | Spam) - Probability that a Token Occurs given the Email is Spam
# Now as we added 1 in all words, now we need to increase the total word count as well before applying conditional probability
# This will minimize the error

prob_tokens_spam = summed_spam_tokens / (spam_wc + VOCAB_SIZE)
prob_tokens_spam.head()

prob_tokens_spam.sum()

# P(Token | Ham) - Probability that a Token Occurs given the Email is Nonspam
prob_tokens_nonspam = summed_ham_tokens / (nonspam_wc + VOCAB_SIZE)
prob_tokens_nonspam.sum()

# P(Token) - Probability that Token Occurs
prob_tokens_all = full_train_features.sum(axis=0) / total_wc

# Save the training model
np.savetxt(TOKEN_SPAM_PROB_FILE, prob_tokens_spam)
np.savetxt(TOKEN_HAM_PROB_FILE, prob_tokens_nonspam)
np.savetxt(TOKEN_ALL_PROB_FILE, prob_tokens_all)

# Prepare Test Data
full_test_data = make_full_matrix(sparse_test_data, VOCAB_SIZE)

X_test = full_test_data.loc[:, full_test_data.columns != 'CATEGORY']
y_test = full_test_data.CATEGORY
np.savetxt(TEST_TARGET_FILE, y_test)
np.savetxt(TEST_FEATURE_MATRIX, X_test)
