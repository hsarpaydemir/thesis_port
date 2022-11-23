import numpy as np
import random
import matplotlib.pyplot as plt

def activateRPandMask_v2_f(idRefs, labSems, ccDsks, rpBWDsks, refPts_l, keep_prob):
    '''
    activates gateways and selects the corresp. masks

    in:   idRefs: reference points with IDs (corresponding to #channel in labSems)
          labSems: 3D stack of instance masks 
          ccDsks: colorised disks 
          rpBWDsks : binary disks
          refPts_l: a list of RP's IDs which are within the image.
          keep_prob: probability to keep a mask and the corresponding ref point
    out:  labSemRems: 3D stack of activated instance masks
          rpBwRems: segmentation masks of activated disks
    '''

    unRPs_l = np.unique(idRefs)
    labSemRems = np.zeros(labSems.shape, dtype=np.uint8)
    rpBwRems = np.zeros(rpBWDsks.shape, dtype=np.uint8)

    for un_i in unRPs_l.astype(int):
        if un_i == 0:
            continue
        if (not np.isin(un_i,refPts_l)) and (random.random() <= keep_prob):
            idM = idRefs == un_i
            refID = ccDsks[idM]
            refM = ccDsks == refID
            labSemRems[:,:,un_i - 1] = labSems[:,:,un_i - 1]
            rpBwRems[refM] = rpBWDsks[refM]

    return labSemRems, rpBwRems