# Notebook imports
import sys
from os import walk
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

from bs4 import BeautifulSoup
# from wordcloud import WordCloud
from PIL import Image
import numpy as np

VOCAB_SIZE = 2500

example_file = "01_Processing/practice_email.txt"
easy_nonspam_1_path = "01_Processing/spam_assassin_corpus/easy_ham_1"
easy_nonspam_2_path = "01_Processing/spam_assassin_corpus/easy_ham_2"
spam_1_path = "01_Processing/spam_assassin_corpus/spam_1"
spam_2_path = "01_Processing/spam_assassin_corpus/spam_2"

spam_cat = 1
ham_cat = 0

DATA_JSON_FILE = '01_Processing/email-text-data.json'
WORD_ID_FILE = '01_Processing/word-by-id.csv'

TRAINING_DATA_FILE = '02_Training/train-data.txt'
TEST_DATA_FILE = '02_Training/test-data.txt'

WHALE_FILE = '01_Processing/wordcloud_resources/whale-icon.png'
SKULL_FILE = '01_Processing/wordcloud_resources/skull-icon.png'
THUMBS_UP_FILE = '01_Processing/wordcloud_resources/thumbs-up.png'
THUMBS_DOWN_FILE = '01_Processing/wordcloud_resources/thumbs-down.png'
CUSTOM_FONT_FILE = '01_Processing/wordcloud_resources/OpenSansCondensed-Bold.ttf'

# reading files
with open(example_file) as stream:
    is_body = False
    lines = []
    for line in stream:
        if is_body:
            lines.append(line)
        elif line == "\n":
            is_body=True

email_body="\n".join(lines)
print(email_body)

# Generator function
# email body extraction
def email_body_generator(path):

    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            filepath = join(root, file_name)
            with open(filepath, encoding='latin-1') as stream:
                is_body = False
                lines = []
                for line in stream:
                    if is_body:
                        lines.append(line)
                    elif line == "\n":
                        is_body = True

            email_body = "\n".join(lines)
            yield file_name, email_body

def df_from_directory(path, classification):
    rows = []
    row_names = []
    for file_name, email_body in email_body_generator(path) :
        rows.append({'message': email_body, "category": classification})
        row_names.append(file_name)

    return pd.DataFrame(rows, index=row_names)

spam_emails = df_from_directory(spam_1_path, spam_cat)
spam_emails = spam_emails.append(df_from_directory(spam_2_path, spam_cat))

nonspam_emails = df_from_directory(easy_nonspam_1_path, ham_cat)
nonspam_emails = spam_emails.append(df_from_directory(easy_nonspam_2_path, ham_cat))

data = pd.concat([spam_emails, nonspam_emails])

# data cleaning: checking for missing values
# check if any messages are null
print(data['message'].isnull().values.any())

# check if there are empty emails
print((data.message.str.len() == 0).any())

# how many trues as true equates to 1
print((data.message.str.len() == 0).sum())

# how many nulls
print(data.message.isnull().sum())

# locate 4 empty messages
print(data[data.message.str.len() == 0].index)

# entry of message with missing data
print(data.index.get_loc('cmds'))

# how to drop unwanted entries (null/empty) from the dataframe
data.drop(['cmds'], inplace=True)

# add document IDs to track emails in dataset
document_IDs = range(0, len(data.index))
data['DOC_ID'] = document_IDs
data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)

print(data.head())
print(data.tail())

# save to file using pandas
data.to_json(DATA_JSON_FILE)

# number of spam messages visualised (pie charts)
data.category.value_counts()

amount_of_spam = data.category.value_counts()[1]
amount_of_ham = data.category.value_counts()[0]

category_names = ['spam', 'legit mail']
sizes = [amount_of_spam, amount_of_ham]

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90,autopct='%1.2f%%')
plt.show()

category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]

custom_colours = ['#ff7675', '#74b9ff']
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90,autopct='%1.2f%%',
       colors=['#ff7675', '#74b9ff'], explode=[0, 0.05])
plt.show()

category_names = ['Spam', 'Legit Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90,
       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8)

# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)

plt.show()

category_names = ['Spam', 'Legit Mail', 'Updates', 'Promotions']
sizes = [55, 66, 77, 99]
custom_colours = ['#ff7675', '#74b9ff', 'yellow', 'green']
abc=[0.05, 0.05, 0.05, 0.05]
plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90,
       autopct='%1.0f%%', colors=custom_colours, pctdistance=0.8, explode=abc)

# draw circle
centre_circle = plt.Circle((0, 0), radius=0.6, fc='white')
plt.gca().add_artist(centre_circle)

plt.show()

# Natural Language processing
#  text preprocessing
# convert to lower case
msg = 'All WoRK and NO PlaY makeS JaCK a DUll Boy'
msg=msg.lower()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('shakespeare')

# Removing stop words and converting to set
type(stopwords.words('english'))
stop_words = set(stopwords.words('english'))
filtered_words=[]
words = word_tokenize(msg.lower())
#Important loop
for word in words:
    if word not in stop_words:
        filtered_words.append(word)

print(filtered_words)

# word stems and stemming
filtered_words=[]

#stemmer = PorterStemmer
stemmer = SnowballStemmer('english')
#Important loop
for word in words:
    if word not in stop_words :
        # removing punctuation
        if word.isalpha():
            stemmed_word=stemmer.stem(word)
            filtered_words.append(stemmed_word)

print(filtered_words)

# removing HTML tags from emails
soup = BeautifulSoup(data.at[2, 'message'], 'html.parser')
print(soup.prettify())

# Functions for email processing
def clean_message(message, stemmer=PorterStemmer(),
                  stop_words=set(stopwords.words('english'))):
    # Converts to Lower Case and splits up the words

    words = word_tokenize(message.lower())

    filtered_words = []

    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))

        return filtered_words

print(clean_message(email_body))

# Challenge: Modify function to remove HTML tags. Then test on Email with DOC_ID 2.
def clean_msg_no_html(message, stemmer=PorterStemmer(),
    stop_words=set(stopwords.words('english'))):
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()

    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())

    filtered_words = []

    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
    #             filtered_words.append(word)

    return filtered_words

print(clean_msg_no_html(data.at[2, 'message']))

# applying cleaning and tokenization to all messages
print(data.iat[2, 0])
#With data.at we work using Column name and with data.iat we use the location
print(data.iloc[0:])

first_emails = data.message.iloc[0:3]

nested_list = first_emails.apply(clean_message)

# flat_list = []
# for sublist in nested_list:
#     for item in sublist:
#         flat_list.append(item)

flat_list = [item for sublist in nested_list for item in sublist]

#both the codes do the same job second is list comprehension

# use apply() on all the messages in the dataframe
nested_list = data.message.apply(clean_msg_no_html)

# Using Logic to Slice Dataframes
print(data[data.CATEGORY == 1].shape)
print(data[data.CATEGORY == 1].tail())
doc_ids_spam = data[data.CATEGORY == 1].index
doc_ids_ham = data[data.CATEGORY == 0].index

# Subsetting a series with an index
print(type(doc_ids_ham))
print(type(nested_list))

#Now to make a smaller list with only ham mails
nested_list_ham = nested_list.loc[doc_ids_ham]
print(nested_list_ham.shape)
nested_list_spam = nested_list.loc[doc_ids_spam]
print(nested_list_spam.shape)

flat_list_ham = [item for sublist in nested_list_ham for item in sublist]
normal_words = pd.Series(flat_list_ham)

print(normal_words.shape[0]) #total number of unique words in the non-spam messages
print(normal_words.value_counts()[:10])
print(normal_words.value_counts().shape[0])

flat_list_spam = [item for sublist in nested_list_spam for item in sublist]
spammy_words = pd.Series(flat_list_spam)

print(spammy_words.shape[0]) #total number of unique wordsin the spam messages
print(spammy_words.value_counts()[:10])

# wordclouds
# word_cloud = WordCloud().generate(email_body)
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# Word Cloud of Ham and Spam messages
# icon = Image.open(THUMBS_UP_FILE)
# image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
# image_mask.paste(icon, box=icon)
#
# rgb_array = np.array(image_mask) # converts the image object to an array

# Generate the text as a string for the word cloud
# ham_str = ' '.join(flat_list_ham)
#
# word_cloud = WordCloud(mask=rgb_array, background_color='white',
#                       max_words=500, colormap='winter')
#
# word_cloud.generate(ham_str)
#
# plt.figure(figsize=[16, 8])
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
#
# icon = Image.open(THUMBS_DOWN_FILE)
# image_mask = Image.new(mode='RGB', size=icon.size, color=(255, 255, 255))
# image_mask.paste(icon, box=icon)
#
# rgb_array = np.array(image_mask) # converts the image object to an array
#
# # Generate the text as a string for the word cloud
# spam_str = ' '.join(flat_list_spam)
#
# word_cloud = WordCloud(mask=rgb_array, background_color='white', max_font_size=300,
#                       max_words=2000, colormap='gist_heat', font_path=CUSTOM_FONT_FILE)
#
# word_cloud.generate(spam_str.upper())#Converting to upper case
#
# plt.figure(figsize=[16, 8])
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()

# generate markdown and dictionary
stemmed_nested_list = data.message.apply(clean_msg_no_html)
flat_stemmed_list =[item for sublist in stemmed_nested_list for item in sublist]

unique_words = pd.Series(flat_stemmed_list).value_counts()
unique_words.shape[0]

frequent_words = unique_words[0:VOCAB_SIZE]
frequent_words

# Create Vocabulary Dataframe with a WORD_ID
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
vocab.head()
vocab.to_csv(WORD_ID_FILE, index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)
#Here we are able to link by providing the index_label and header straightaway from Vocab

# Exercise: Checking if a Word is Part of the Vocabulary
any(vocab.VOCAB_WORD=='machine')
#Set method
'machine' in set(vocab.VOCAB_WORD)
#Much faster...

# Exercise: Find the Email with the Most Number of Words

# For loop
clean_email_lengths = []
for sublist in stemmed_nested_list:
    clean_email_lengths.append(len(sublist))

print(clean_email_lengths)
#Python list comprehension
clean_email_lengths = [len(sublist) for sublist in stemmed_nested_list]
print('Nr words in the longest email:', max(clean_email_lengths))

print('Email position in the list(and the data dataframe)', np.argmax(clean_email_lengths))

print(stemmed_nested_list[np.argmax(clean_email_lengths)])

print(data.at[np.argmax(clean_email_lengths), 'message'])

# Generate Features & a Sparse Matrix

type(stemmed_nested_list.tolist())

word_columns_df = pd.DataFrame.from_records(stemmed_nested_list.tolist())
print(word_columns_df)

# Splitting the Data into a Training and Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(word_columns_df, data.CATEGORY, test_size=0.3, random_state=42)
print('Nr of training samples', X_train.shape[0])
print('Fraction of training set', X_train.shape[0] / word_columns_df.shape[0])
X_train.index.name = X_test.index.name = 'DOC_ID'
X_train.head()
y_train.head()

# Create a Sparse Matrix for the Training Data
word_index = pd.Index(vocab.VOCAB_WORD)
word_index.get_loc('thu')
X_train.index[0]
y_train.at[4844]


def make_sparse_matrix(df, indexed_words, labels):
    """
    Returns sparse matrix as dataframe.

    df: A dataframe with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)

    """
    nr_rows = df.shape[0]
    nr_columns = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for i in range(nr_rows):
        for j in range(nr_columns):
            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)
                category = labels.at[doc_id]
                item = {'LABEL': category, 'DOC_ID': doc_id,
                        'OCCURENCE': 1, 'WORD_ID': word_id}

                dict_list.append(item)

    return pd.DataFrame(dict_list)


sparse_train_df = make_sparse_matrix(X_train, word_index, y_train)
print(sparse_train_df[:5])
print(sparse_train_df.shape)

# Combine Occurrences with the Pandas groupby() Method
train_grouped = sparse_train_df.groupby(['DOC_ID', 'WORD_ID', 'LABEL']).sum()
train_grouped.head()

print(vocab.at[0, 'VOCAB_WORD'])

print(data.message[0])

train_grouped = train_grouped.reset_index()
train_grouped.head()

train_grouped.tail()

print(vocab.at[1895, 'VOCAB_WORD'])

print(train_grouped.shape)

# Save Training Data as .txt File
np.savetxt(TRAINING_DATA_FILE, train_grouped, fmt='%d')

print(train_grouped.columns)






