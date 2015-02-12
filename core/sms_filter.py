import csv

import pandas as pd
from textblob import TextBlob

__author__ = 'laurogama'

# messages = [line.rstrip() for line in open('../datasets/sms_spam/SMSSpamCollection')]
# print len(messages)
# for message_no, message in enumerate(messages[:10]):
#     print message_no, message
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words


messages = pd.read_csv('../datasets/sms_spam/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,
                           names=["label", "message"])
# print messages
print messages.groupby('label').describe()



messages.message.head().apply(split_into_tokens)