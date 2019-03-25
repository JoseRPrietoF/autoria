import glob

NL = "NL"

class canon60Dataset():
    """
    Class to handle the canon60Dataset feeding
    """

    def __init__(self, path, join_all=True):

        if path.endswith('/'):
            path = path[:-1]
        self.join_all = join_all
        self.path = path
        self.flists = self.get_files()
        self.X, self.y = self.read_files()

    def get_files(self, ext="txt"):
        """
        Return the files on self.path with terminates on ext
        :param ext:
        :return:
        """
        f = glob.glob(self.path+"/*/*{}".format(ext))
        return f

    def read_files(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X,y = [], []
        for fname in self.flists:
            author = fname.split('/')[3]
            with open(fname) as f:
                content = f.readlines()

            text = [x.lower() for x in content]
            if self.join_all:
                text = ' '.join([x.replace("\n", " ") for x in text])
            else:
                text = [x.replace("\n", " {}".format(NL)) for x in text]

            X.append(text)
            y.append(author)
        return X,y


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


