from data import process

fname_vocab = "/data2/jose/data_autoria/vocabulario"
root_path = "/data2/jose/data_autoria/"
train_path = root_path+"train"
test_path = root_path+"test"

print("Loading train dataset...")
dt_test = process.canon60Dataset(test_path)
print("Loaded")
vocab_freq = process.read_vocab(fname_vocab)
vocab = list(vocab_freq.keys())

print(dt_test.X)
print(len(dt_test.X))

palNum = {'unk':0}
numPal = {0:'unk'}

for i, word in enumerate(vocab):
    palNum[word] = i+1
    numPal[i+1] = word

X = []
y = []
for sentence in dt_test.X:
    tempX = []
    for words in sentence:
        words = words.split()
        for word in words:
            tempX.append(palNum[word])
    X.append(tempX)


print(len(X))