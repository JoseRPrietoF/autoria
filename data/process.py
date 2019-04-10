import glob
from xml.dom import minidom
import logging
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

    def __init__(self, path, txt, join_all=False):
        self.path = path
        self.txt = txt
        self.join_all = join_all
        self.X, self.y, self.fnames = self.read_files()

    def read_files(self):
        """
        Return the contents of files with the gt
        :return:
        """
        X,y, fnames = [], [], []
        txt_path = self.txt
        with open(txt_path) as f:
            lines = f.readlines()

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
            else:
                for txt_region in docs:
                    text += txt_region.firstChild.nodeValue
                    X.append(text)
                    y.append(clase)

            fnames.append(fname_xml)

        return X,y, fnames

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

def write_from_array(array, path):
    """
    Aux method to call write_output from an array of results
    :return:
    """
    for a in array:
        id, lang, type = a
        write_output(id, lang, type, path=path)

def write_output(id, lang, type, path):
    """
    <author id="author-id" lang="en|es" type="bot|human" gender="bot|male|female" />
    gender not predicted at the moment
    :param id:
    :return:
    """
    str = '<author id="{}" lang="{}" type="{}" />'.format(id, lang, type)
    with open('{}/{}'.format(path, id), 'w') as the_file:
        the_file.write(str)
        the_file.close()


if __name__ == "__main__":
    # path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/train"
    # path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/test"
    path = "/data2/jose/data/pan19-author-profiling-training-2019-02-18/es"
    txt = "truth.txt"
    # dt = canon60Dataset(path)
    dt = PAN2019(path=path, txt=txt)
    import numpy as np
    classes = np.unique(dt.y, return_counts=True)
    print(classes)
