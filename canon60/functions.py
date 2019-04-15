import numpy as np

def classify_per_doc(classifieds, logger=None):
    """
    Clasificamos por documento.
    Votacion
    Devuelve una lista de tuplas con (hyp, gt)
    :param classifieds:
    :return:
    """
    per_doc = {}
    new_classifieds = []
    for i in range(len(classifieds)):
        hyp, fname, y_gt = classifieds[i]
        hyp = np.argmax(hyp, axis=-1)
        y_gt = np.argmax(y_gt, axis=-1)
        doc_name = fname.decode("utf-8").split("/")[-1]
        doc_name = doc_name.split("_")[-1].split(".")[0]

        docs = per_doc.get(doc_name, [])
        docs.append((hyp, fname, y_gt))
        per_doc[doc_name] = docs

    for k in per_doc.keys():

        counts = []
        for a in per_doc[k]:
            hyp, fname, y_gt = a
            counts.append(hyp)
        # counts = np.unique(counts, return_counts=True)
        counts = np.bincount(counts)
        hyp = np.argmax(counts)
        new_classifieds.append((hyp, y_gt))
        logger.info("Clasificando : {} en autor {} siendo realmente del autor {}".format(k, hyp, y_gt))

        logger.info("---------------- \n\n\n")
    return new_classifieds