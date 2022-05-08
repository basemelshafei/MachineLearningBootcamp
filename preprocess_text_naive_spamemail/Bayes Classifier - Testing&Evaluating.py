import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TOKEN_SPAM_PROB_FILE = '03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = '03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = '03_Testing/prob-all-tokens.txt'

TEST_FEATURE_MATRIX = '03_Testing/test-features.txt'
TEST_TARGET_FILE = '03_Testing/test-target.txt'

VOCAB_SIZE = 2500

# Features
X_test = np.loadtxt(TEST_FEATURE_MATRIX, delimiter=' ')
# Target
y_test = np.loadtxt(TEST_TARGET_FILE, delimiter=' ')
# Token Probabilities
prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')

print(X_test[:5])

# calculating the joint probability

PROB_SPAM = 0.31116
np.log(prob_token_spam)

# Joint probability in log format (easier math)
joint_log_spam = X_test.dot(np.log(prob_token_spam)- np.log(prob_all_tokens))+np.log(PROB_SPAM)
print(joint_log_spam[:5])

joint_log_ham = X_test.dot(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1-PROB_SPAM)
print(joint_log_ham[:5])

# making predictions
# checking for higher probability
prediction = joint_log_spam > joint_log_ham
print(prediction[-5:]*1)

# This *1 converts boolean into integers
# simplifying the probabilities by removing the common division probability
joint_log_spam = X_test.dot(np.log(prob_token_spam))+np.log(PROB_SPAM)
joint_log_ham = X_test.dot(np.log(prob_token_ham)) + np.log(1-PROB_SPAM)

#Probability of an email being spam or non spam does not depent on probability of the token occuring


# metrics and evaluation
# accurcay
correct_doc = (y_test == prediction).sum()
print('Docs classified correctly:', correct_doc)

numdocs_wrong = X_test.shape[0] - correct_doc
print('Docs classified incorrectly', numdocs_wrong)
#Only 50 classified incorrectly

#Accuracy
correct_doc/len(X_test)

fraction_wrong = numdocs_wrong/len(X_test)
print('Fraction classified incorrectly is {:.2%}'.format(fraction_wrong))
print('Accuracy of the model is {:.2%}'.format(1-fraction_wrong))

# visualizing results
# Chart Styling Info
yaxis_label = 'P(X | Spam)'
xaxis_label = 'P(X | Nonspam)'

linedata = np.linspace(start=-14000, stop=1, num=1000)
plt.figure(figsize = (11,7))
plt.xlabel(xaxis_label, fontsize = 14)
plt.ylabel(yaxis_label, fontsize = 14)

plt.xlim([-14000, 1])
plt.ylim([-14000, 1])

plt.scatter(joint_log_ham, joint_log_spam, color= 'navy')
plt.show()

# the decision boundary
plt.figure(figsize = (16,7))

plt.subplot(1, 2, 1)
plt.xlabel(xaxis_label, fontsize = 14)
plt.ylabel(yaxis_label, fontsize = 14)

plt.xlim([-14000, 1])
plt.ylim([-14000, 1])
plt.plot(linedata, linedata, color = 'orange')

plt.scatter(joint_log_ham, joint_log_spam, color= 'navy', alpha = 0.5, s = 25)

plt.subplot(1, 2, 2)
plt.xlabel(xaxis_label, fontsize = 14)
plt.ylabel(yaxis_label, fontsize = 14)

plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.plot(linedata, linedata, color = 'orange')

plt.scatter(joint_log_ham, joint_log_spam, color= 'navy', alpha = 0.5, s = 3)

plt.show()

#To make our charts more visual we need to use Seaborn
sns.set_style('whitegrid')
labels = 'Actual Category'

summary_df = pd.DataFrame({yaxis_label: joint_log_spam, xaxis_label: joint_log_ham, labels: y_test})

#Seaborn works well with DataFrames, hence we created one

sns.lmplot(x = xaxis_label, y = yaxis_label, data = summary_df, height = 6.5, fit_reg=False,
           scatter_kws={'alpha':0.5, 's':25})
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.plot(linedata, linedata, color = 'orange')

sns.lmplot(x = xaxis_label, y = yaxis_label, data = summary_df, height = 6.5, fit_reg=False, legend = False,
           scatter_kws={'alpha':0.5, 's':25}, hue = labels, markers=['o', 'x'], palette = 'hls')
plt.xlim([-2000, 1])
plt.ylim([-2000, 1])
plt.plot(linedata, linedata, color = 'black')
plt.legend(('Decision Boundary', 'NonSpam', 'Spam'), loc = 'lower right', fontsize = 14)
plt.show()
# Making a customised color pallete and zooming in further

my_colours = ['#4A71C0', '#AB3A2C']

sns.lmplot(x=xaxis_label, y=yaxis_label, data=summary_df, height=6.5, fit_reg=False, legend=False,
          scatter_kws={'alpha': 0.7, 's': 25}, hue=labels, markers=['o', 'x'], palette=my_colours)

plt.xlim([-500, 1])
plt.ylim([-500, 1])

plt.plot(linedata, linedata, color='black')

plt.legend(('Decision Boundary', 'Nonspam', 'Spam'), loc='lower right', fontsize=14)
plt.show()

# false positives and false negatives
print(np.unique(prediction, return_counts=True))
true_pos = (y_test == 1) & (prediction == 1)
true_pos.sum()

false_pos = (y_test == 0) & (prediction == 1)
false_pos.sum()

false_neg = (y_test == 1) & (prediction == 0)
false_neg.sum()

recall_score = true_pos.sum() / (true_pos.sum() + false_neg.sum())
print('Recall score is {:.2%}'.format(recall_score))

precision_score = true_pos.sum() / (true_pos.sum() + false_pos.sum())
print('Precision score is {:.2%}'.format(precision_score))

# Very Often there is a tradeoff between recall and precision
#Think more on this

# F-1Score and F-Score
f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
print('F Score is {:.2}'.format(f1_score))

# F score is the harmonic mean of Recall and Precision, hence it takes both false positives
# and false negatives into account





