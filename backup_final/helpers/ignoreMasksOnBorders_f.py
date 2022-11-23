from pickletools import uint8
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def ignoreMasksOnBorders_f(tmp_augL, tmp_augC):
    #finds and ignores instance masks on borders

    #input:
    # tmp_augL: a stack of instance masks
    #output:
    # ignoreOB: ignore mask

    augLBord = np.zeros(tmp_augL.shape, dtype=int)
    augLBord[:, 0, :] = 1
    augLBord[:, -1, :] = 1
    augLBord[0, :, :] = 1
    augLBord[-1, :, :] = 1
    ignoreOB = tmp_augL * augLBord
    #np.amax(np.concatenate(((np.ones(ignoreOB[:, :, 0].shape, dtype=int) * 0.1)[:, :, np.newaxis], ignoreOB), axis=2), axis=2) Ones array always returns 0.1,returns 0 in matlab code
    ignoreOB = np.argmax(np.concatenate(((np.ones(ignoreOB[:, :, 0].shape, dtype=int) * 0.1)[:, :, np.newaxis], ignoreOB), axis=2), axis=2)
    ignoreOB -= 1
    un2ignore = np.unique(ignoreOB)

    # ignore intsnce masks
    for ug_i in un2ignore:
        if ug_i == -1:
            continue
        ig_m = tmp_augL[:, :, ug_i] > 0
        ignoreOB[ig_m] = 1
    ignoreOB[ignoreOB < 0] = 0
    return tmp_augL, tmp_augC, ignoreOB
