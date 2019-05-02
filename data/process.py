import glob, random
from xml.dom import minidom
import logging
from nltk.tokenize import TweetTokenizer
import numpy as np
import os
from textblob import TextBlob

NL = "NL"

class canon60Dataset():
    """
    Class to handle the canon60Dataset feeding
    """

    def __init__(self, path, join_all=False, logger=None):

        self.logger = logger or logging.getLogger(__name__)
        if path.endswith('/'):
            path = path[:-1]
        self.join_all = join_all
        self.path = path
        self.flists = self.get_files()
        self.X, self.y, self.fnames = self.read_files()

    def get_files(self, ext="txt"):
        """
        Return the files on self.path with terminates on ext
        :param ext:
        :return:
        """
        f = glob.glob(self.path+"/*{}".format(ext))
        return f

    def read_files(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X,y = [], []
        fnames = []
        for fname in self.flists:
            with open(fname) as f:
                content = f.readlines()

            author = content[0].split(";")[0][2:]
            text = content[1:]
            text = [x.lower() for x in text]
            if self.join_all:
                text = ' '.join([x.replace("\n", " ") for x in text])
            else:
                text = [x.replace("\n", " {}".format(NL)) for x in text]

            X.append(text)
            y.append(author)
            fnames.append(fname)

        return X,y, fnames

class PAN2019():
    """

    """

    def __init__(self, path, txt, join_all=False, name="train", mode="CNN3D", sentiment=True):
        self.path = path
        self.mode = mode
        self.txt = txt
        self.join_all = join_all
        self.id = name
        if sentiment:
            self.X, self.y, self.fnames, self.y2, self.sentiment = self.read_files_sentiment()
        else:
            self.X, self.y, self.fnames, self.y2 = self.read_files()


    def get_vocab(self):
        tknzr = TweetTokenizer()
        # load
        try:
            X_train = np.load("./tmp/{}_X.npy".format(self.id))
            vocab = np.load("./tmp/{}_vocab.npy".format(self.id))
            print("Loaded data from vocab")
        except Exception as e:
            print(e)
            exit()
            print("Tokenizing")
            X_train = [tknzr.tokenize(x) for x in self.X]
            print("Sum")
            vocab_dict = {}
            print("Getting vocab")
            for tweet_tok in X_train:
                for tok in tweet_tok:
                    vocab_dict[tok] = vocab_dict.get(tok, 0) + 1
            vocab = list(vocab_dict.keys())
            # save vocab
            print("Saving data from vocab")
            np.save("./tmp/{}_X".format(self.id), X_train)
            np.save("./tmp/{}_vocab".format(self.id), vocab)
        return vocab, X_train

    def read_files(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X,y, y2, fnames = [], [], [], []
        txt_path = self.txt
        with open(txt_path) as f:
            lines = f.readlines()

        random.shuffle(lines)
        print("Join ALL: {}".format(self.join_all))
        for line in lines:
            fname, clase, gender = line.split(":::")
            # print("{} - {} - {}".format(fname, clase, gender))
            fname_xml = "{}/{}.xml".format(self.path, fname)
            xmldoc = minidom.parse(fname_xml)
            docs = xmldoc.getElementsByTagName("document")
            text = ""
            if self.join_all:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue + " "
                X.append(text)
                y.append(clase)
                y2.append(gender)
                fnames.append(fname_xml)

            else:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue
                    X.append(text)
                    if self.mode == "CNN":
                        y.append(clase)
                        y2.append(gender)
                        fnames.append(fname_xml)

                if self.mode == "CNN3D":
                    y.append(clase)
                    y2.append(gender)
                    fnames.append(fname_xml)

        return X,y, fnames, y2

    def read_files_sentiment(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X,y, y2, fnames = [], [], [], []
        sentiment = []
        txt_path = self.txt
        with open(txt_path) as f:
            lines = f.readlines()

        random.shuffle(lines)
        print("Join ALL: {}".format(self.join_all))
        for line in lines:
            fname, clase, gender = line.split(":::")
            # print("{} - {} - {}".format(fname, clase, gender))
            fname_xml = "{}/{}.xml".format(self.path, fname)
            xmldoc = minidom.parse(fname_xml)
            docs = xmldoc.getElementsByTagName("document")
            text = ""
            if self.join_all:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue + " "
                blob = TextBlob(text)
                sentiment.append([blob.sentiment.polarity, blob.sentiment.subjectivity])

                X.append(text)
                y.append(clase)
                y2.append(gender)
                fnames.append(fname_xml)

            else:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue
                    X.append(text)
                    if self.mode == "CNN":
                        y.append(clase)
                        y2.append(gender)
                        fnames.append(fname_xml)

                if self.mode == "CNN3D":
                    y.append(clase)
                    y2.append(gender)
                    fnames.append(fname_xml)

        return X,y, fnames, y2, sentiment


class PAN2019_Test():
    """

    """

    def __init__(self, path, join_all=False, sentiment=True):
        self.path = path
        self.join_all = join_all
        self.files = self.get_files()
        if sentiment:
            self.X, self.fnames, self.sentiment = self.read_files_sentiment()
        else:
            self.X, self.fnames = self.read_files()

    def read_files(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X, fnames = [], []

        #print("Join ALL: {}".format(self.join_all))
        for fname_xml in self.files:

            xmldoc = minidom.parse(fname_xml)
            docs = xmldoc.getElementsByTagName("document")
            text = ""
            if self.join_all:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue + " "
                X.append(text)

            else:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue
                    X.append(text)



            fnames.append(fname_xml)


        return X, fnames

    def read_files_sentiment(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X, fnames = [], []
        sent = []
        #print("Join ALL: {}".format(self.join_all))
        for fname_xml in self.files:

            xmldoc = minidom.parse(fname_xml)
            docs = xmldoc.getElementsByTagName("document")
            text = ""
            if self.join_all:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue + " "
                X.append(text)
                blob = TextBlob(text)
                sent.append([blob.sentiment.polarity, blob.sentiment.subjectivity])
            else:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue
                    X.append(text)



            fnames.append(fname_xml)


        return X, fnames, sent

    def get_files(self, ext="xml"):
        """
        Return the files on self.path with terminates on ext
        :param ext:
        :return:
        """
        f = glob.glob(self.path+"/*{}".format(ext))
        return f

def read_vocab(fname,join_all=True):
    """
    Return a dict with the vocab and the freqs
    :return:
    """
    X = {}
    i = 0
    if not join_all:
        X["NL"] = 0
        i += 1
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            freq, vocab = line.split()
            ind = i
            vocab = vocab.lower()
            freq = int(freq)
            X[vocab] = i
            i+=1

    return X

def read_vocab_list(fname):
    """
    Return a dict with the vocab and the freqs
    :return:
    """
    X = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            line = line.lower().strip()
            X.append(line)

    return X

def write_from_array(array, path, x_train=[], texts_rep_train=[], y2_train=[], x_test=[], text_test_rep=[], fnames_test=[]):
    """
    Aux method to call write_output from an array of results
    :return:
    """

    id_human = []	
    for a in array:
        id, lang, type = a
	
        if type == "bot":
            write_output(id, lang, type, path=path, gender=type)
        else:
            id_human.append(id)
    
    gender_ids = get_gender(id_human, lang, x_train, texts_rep_train, y2_train, x_test, text_test_rep, fnames_test)

    for b in gender_ids:
        id, lang, gender = b
        write_output(id, lang, "human", path=path, gender=gender.strip()) 

def write_output(id, lang, type, path, gender): #set
    """
    <author id="author-id" lang="en|es" type="bot|human" gender="bot|male|female" /> 
    gender not predicted at the moment
    :param id:
    :return:
    """
    id_file = id.split(".")[0]
    str = '<author id="{}" lang="{}" type="{}" gender="{}"/>'.format(id_file, lang, type, gender) #set
    with open('{}/{}'.format(path, id), 'w') as the_file:
        the_file.write(str)
        the_file.close()

from sklearn.neural_network import MLPClassifier
from data import linguistic_process
 
def get_gender(id_human, lang, x_train, texts_rep_train, y_train, x_test, text_test_rep, fnames_test):

    info = []
    '''
    if lang == "en":
        arr_ling = linguistic_process.sentiment_analisis_textblob(x_train)
        texts_rep_train = np.column_stack((arr_ling,texts_rep))
    '''
    for i,e in enumerate(y_train):
        if e=="bot":
            texts_rep_train.pop(i)
            y_train.pop(i)

    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32))
    clf.fit(texts_rep_train, y_train)	

    x_test_new = []
    id_test = []

    for i,fname in enumerate(fnames_test):
        fname = fname.split("/")[-1]
        if fname in id_human:
            x_test_new.append(text_test_rep[i])
            id_test.append(fname)
    '''
    if lang == "en":
        arr_ling = linguistic_process.sentiment_analisis_textblob(x_test)
        x_test_new = np.column_stack((arr_ling,x_test_new))    
    '''
    pred = clf.predict(x_test_new) 

    for i,gend in enumerate(pred):
        info.append((id_test[i],lang,gend))

    return info



if __name__ == "__main__":

    path = "/data2/jose/data_autoria/VocAllm"
    l = read_vocab_list(path)
    print(l)
    print(len(l))
