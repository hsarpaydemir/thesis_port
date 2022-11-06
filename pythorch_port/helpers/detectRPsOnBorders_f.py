import numpy as np

def detectRPsOnBorders_f(idRefs, ccDsks):
    #detects RP-disks and RPs with IDs on the tile borders. 

    #in:  idRefs: reference points with ID which is corresponding to a mask's channel
    #    ccDsks: colorised disks

    #out: refPts_l: list of RP's IDs
    #    brdDsks: disk segmentaion mask with disks on borders
    
    cc_l = np.append(np.append(np.append(np.unique(ccDsks[:, 0]), np.unique(ccDsks[:, -1])), np.unique(ccDsks[0, :])), np.unique(ccDsks[-1, :]))
    cc_l = cc_l[cc_l != 0]

    refPts_l = []
    brdDsks = np.zeros(ccDsks.shape, dtype=np.uint8)

    for cdi in cc_l:
        cc_m = ccDsks == cdi
        refPts_l = np.append(refPts_l, np.amax(idRefs[cc_m]))
        brdDsks[cc_m] = 1
    
    return brdDsks, refPts_l