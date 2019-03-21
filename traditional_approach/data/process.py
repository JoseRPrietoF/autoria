import glob

NL = "NL"

class canon60Dataset():
    """
    Class to handle the canon60Dataset feeding
    """

    def __init__(self, path):

        if path.endswith('/'):
            path = path[:-1]
        self.path = path
        self.flists = self.get_files()
        self.X, self.y = self.read_files()

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
        for fname in self.flists:
            with open(fname) as f:
                content = f.readlines()

            author = content[0].split(";")[0][2:]
            text = content[1:]
            # text = [x.replace("\n", " {}".format(NL)) for x in text]
            text = ' '.join([x.replace("\n", " ") for x in text])
            '''
            from nltk import ngrams
            n = 2
            ngramas = ngrams(text.split(), n)
            text = ' '.join(['_'.join(gram) for gram in ngramas])
            ''' 
            X.append(text)
            y.append(author)

        return X,y


def read_vocab(fname):
    """
    Return a dict with the vocab and the freqs
    :return:
    """
    #X = {"NL":0}
    X = {}
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            freq, vocab = line.split()
            freq = int(freq)
            X[vocab] = freq

    return X


if __name__ == "__main__":
    # path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/train"
    path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/test"

    dt = canon60Dataset(path)
