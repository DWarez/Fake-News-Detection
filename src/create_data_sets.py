import pandas as pd
from sklearn.model_selection import train_test_split
import sys

#Reading the input

if len(sys.argv) < 2:
    raise Exception(
        f"Usage: python {sys.argv[0]} <% training>[directory csv]")

training_percentual = int(sys.argv[1])
test_percentual = 100 - training_percentual

if training_percentual + test_percentual != 100:
    raise Exception("% training + % test should be 100")

csv_path = "data"
if len(sys.argv) == 3:
    csv_path = sys.argv[2]

#read the csv
training_set = pd.read_csv(f"../{csv_path}/Task3a_training.csv", usecols=[1, 3])

#divide in test and training randomly
y = training_set.label
X = training_set.drop('label', 1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=test_percentual/100)

train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)
data_set = train_set.append(test_set)

# #Calculate Statistic informations
# percentual_labels_in_train = train_set['label'].value_counts(normalize=True) * 100
# print(f"Train set info:\n-size: {len(train_set.index)}\n-%true news: {percentual_labels_in_train} %\n-%fake news: {percentual_labels_in_train} %")

# percentual_labels_in_test = test_set['label'].value_counts(normalize=True) * 100
# print(f"Test set info:\n-size: {len(test_set.index)}\n-%true news: {percentual_labels_in_test[1]} %\n-%fake news: {percentual_labels_in_test[0]} %")

percentual_original = data_set['label'].value_counts(normalize=True) * 100
print(f"Original set info:\n-size: {len(data_set.index)}\n {percentual_original}%")

#write result
data_set.to_csv(f"../data/Data_set.csv", index=False)
train_set.to_csv(f"../data/train_set.csv", index=False)
test_set.to_csv(f"../data/test_set.csv", index=False)