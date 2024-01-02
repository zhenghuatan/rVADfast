from __future__ import division
import numpy
import sys
import os
import math
import struct
import scipy.io.wavfile as wav
from scipy.fftpack import dct
from scipy.signal import lfilter
from copy import deepcopy
import code


# References
# Z.-H. Tan and B. Lindberg, Low-complexity variable frame rate analysis for speech recognition and voice activity detection.
# IEEE Journal of Selected Topics in Signal Processing, vol. 4, no. 5, pp. 798-807, 2010.
# Achintya Kumar Sarkar and Zheng-Hua Tan 2017
# Version: 02 Dec 2017


def speech_wave(fileName_):
    (fs, sig) = wav.read(fileName_)
    if sig.dtype == 'int16':
        nb = 16  # -> 16-bit wav files
    elif sig.dtype == 'int32':
        nb = 32  # -> 32-bit wav files
    max_nb = float(2 ** (nb - 1))
    sig = sig / (max_nb + 1.0)
    return fs, sig


def enframe(speech, fs, winlen, ovrlen):
    N, flth, foVr = len(speech), int(numpy.fix(fs * winlen)), int(numpy.fix(fs * ovrlen))

    if len(speech) < flth:
        print("speech file length shorter than window length")
        exit()

    frames = int(numpy.ceil((N - flth + foVr) / foVr))
    slen = (frames - 1) * foVr + flth

    if len(speech) < slen:
        signal = numpy.concatenate((speech, numpy.zeros((slen - N))))

    else:
        signal = deepcopy(speech)

    idx = numpy.tile(numpy.arange(0, flth), (frames, 1)) + numpy.tile(numpy.arange(0, (frames) * foVr, foVr),
                                                                      (flth, 1)).T
    idx = numpy.array(idx, dtype=numpy.int64)

    return signal[idx]


def sflux(data, fs, winlen, ovrlen, nftt):
    eps = numpy.finfo(float).eps

    xf = enframe(data, fs, winlen, ovrlen)  # framing
    w = numpy.matrix(numpy.hamming(int(fs * winlen)))
    w = numpy.tile(w, (numpy.size(xf, axis=0), 1))

    xf = numpy.multiply(xf, w)  # apply window

    # fft
    ak = numpy.abs(numpy.fft.fft(xf, nftt))
    idx = range(0, int(nftt / 2) + 1)
    ak = ak[:, idx]
    Num = numpy.exp(float(1 / len(idx)) * numpy.sum(numpy.log(ak + eps), axis=1))
    Den = float(1 / len(idx)) * numpy.sum(ak, axis=1)

    ft = (Num + eps) / (Den + eps)

    flen, fsh10 = int(numpy.fix(fs * winlen)), int(numpy.fix(fs * ovrlen))
    nfr10 = int(numpy.floor((len(data) - (flen - fsh10)) / fsh10))

    # syn frames as per nfr10
    if nfr10 < len(ft):
        ft = ft[range(nfr10)]
    else:
        ft = numpy.concatenate((ft, numpy.repeat(ft[:1], nfr10 - len(ft), axis=0)))

    return ft, flen, fsh10, nfr10


def snre_highenergy(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk):
    ## ---*******- important *******
    # here [0] index array element has  not used

    Dexpl = 18;
    Dexpr = 18;
    segThres = 0.25

    fdata_ = deepcopy(fdata);
    pv01_ = deepcopy(pv01);
    pvblk_ = deepcopy(pvblk)

    fdata_ = numpy.insert(fdata_, 0, 'inf')
    pv01_ = numpy.insert(pv01_, 0, 'inf')
    pvblk_ = numpy.insert(pvblk_, 0, 'inf')

    # energy estimation
    e = numpy.zeros(nfr10, dtype='float64')
    e = numpy.insert(e, 0, 'inf')

    for i in range(1, nfr10 + 1):
        for j in range(1, flen + 1):
            e[i] = e[i] + numpy.square(fdata_[(i - 1) * fsh10 + j])

        if numpy.less_equal(e[i], ENERGYFLOOR):
            e[i] = ENERGYFLOOR

    emin = numpy.ones(nfr10)
    emin = numpy.insert(emin, 0, 'inf')
    NESEG = 200

    if numpy.less(nfr10, NESEG):
        NESEG = nfr10

    for i in range(1, int(numpy.floor(nfr10 / NESEG)) + 1):
        eY = numpy.sort(e[range((i - 1) * NESEG + 1, (i * NESEG) + 1)])
        eY = numpy.insert(eY, 0, 'inf')

        emin[range((i - 1) * NESEG + 1, i * NESEG + 1)] = eY[int(numpy.floor(NESEG * 0.1))]
        if numpy.not_equal(i, 1):
            emin[range((i - 1) * NESEG + 1, i * NESEG + 1)] = 0.9 * emin[(i - 1) * NESEG] + 0.1 * emin[
                (i - 1) * NESEG + 1]

    if numpy.not_equal(i * NESEG, nfr10):
        eY = numpy.sort(e[range((i - 1) * NESEG + 1, nfr10 + 1)])
        eY = numpy.insert(eY, 0, 'inf')

        emin[range(i * NESEG + 1, nfr10 + 1)] = eY[int(numpy.floor((nfr10 - (i - 1) * NESEG) * 0.1))]
        emin[range(i * NESEG + 1, nfr10 + 1)] = 0.9 * emin[i * NESEG] + 0.1 * emin[i * NESEG + 1]

    D = numpy.zeros(nfr10)
    D = numpy.insert(D, 0, 'inf')

    postsnr = numpy.zeros(nfr10)
    postsnr = numpy.insert(postsnr, 0, 'inf')

    for i in range(2, nfr10 + 1):
        postsnr[i] = numpy.log10(e[i]) - numpy.log10(emin[i])
        if numpy.less(postsnr[i], 0):
            postsnr[i] = 0

        D[i] = numpy.sqrt(numpy.abs(e[i] - e[i - 1]) * postsnr[i])
    D[1] = D[2]

    tm1 = numpy.hstack((numpy.ones(Dexpl) * D[1], D[1:len(D)]))
    Dexp = numpy.hstack((tm1, numpy.ones(Dexpr) * D[nfr10]))
    Dexp = numpy.insert(Dexp, 0, 'inf')

    Dsmth = numpy.zeros(nfr10, dtype='float64')
    Dsmth = numpy.insert(Dsmth, 0, 'inf')

    Dsmth_max = deepcopy(Dsmth)

    for i in range(1, nfr10 + 1):
        Dsmth[i] = sum(Dexp[range(i, i + Dexpl + Dexpr + 1)])

    for i in range(1, int(numpy.floor(nfr10 / NESEG)) + 1):
        Dsmth_max[range((i - 1) * NESEG + 1, i * NESEG + 1)] = numpy.amax(
            e[range((i - 1) * NESEG + 1, i * NESEG + 1)]);  # numpy.amax(Dsmth[range((i-1)*NESEG+1, i*NESEG+1)])

    if numpy.not_equal(i * NESEG, nfr10):
        Dsmth_max[range(i * NESEG + 1, nfr10 + 1)] = numpy.amax(
            e[range((i - 1) * NESEG + 1, nfr10 + 1)])  # numpy.amax(Dsmth[range((i-1)*NESEG+1, nfr10+1)])

    snre_vad = numpy.zeros(nfr10)
    snre_vad = numpy.insert(snre_vad, 0, 'inf')

    for i in range(1, nfr10 + 1):
        if numpy.greater(Dsmth[i], Dsmth_max[i] * segThres):
            snre_vad[i] = 1

    # block based processing to remove noise part by using snre_vad1.
    sign_vad = 0
    noise_seg = numpy.zeros(int(numpy.floor(nfr10 / 1.6)));
    noise_seg = numpy.insert(noise_seg, 0, 'inf')

    noise_samp = numpy.zeros((nfr10, 2))
    n_noise_samp = -1

    for i in range(1, nfr10 + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):  # % start of a segment
            sign_vad = 1
            nstart = i
        elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):  # % end of a segment
            sign_vad = 0
            nstop = i - 1
            if numpy.equal(sum(pv01_[range(nstart, nstop + 1)]), 0):
                noise_seg[range(int(numpy.round(nstart / 1.6)), int(numpy.floor(nstop / 1.6)) + 1)] = 1
                n_noise_samp = n_noise_samp + 1
                noise_samp[n_noise_samp, :] = numpy.array([(nstart - 1) * fsh10 + 1, nstop * fsh10])

    noise_samp = noise_samp[:n_noise_samp + 1, ]

    # syn  from [0] index
    noise_samp = noise_samp - 1
    noise_seg = noise_seg[1:len(noise_seg)]

    return noise_samp, noise_seg, len(noise_samp)


def snre_vad(fdata, nfr10, flen, fsh10, ENERGYFLOOR, pv01, pvblk, vadThres):
    ## ---*******- important *******
    # here [0] index array element has  not used

    Dexpl, Dexpr = 18, 18
    Dsmth = numpy.zeros(nfr10, dtype='float64');
    Dsmth = numpy.insert(Dsmth, 0, 'inf')

    fdata_ = deepcopy(fdata)
    pv01_ = deepcopy(pv01)
    pvblk_ = deepcopy(pvblk)

    fdata_ = numpy.insert(fdata_, 0, 'inf')
    pv01_ = numpy.insert(pv01_, 0, 'inf')
    pvblk_ = numpy.insert(pvblk_, 0, 'inf')

    # energy estimation
    e = numpy.zeros(nfr10, dtype='float64')
    e = numpy.insert(e, 0, 'inf')

    for i in range(1, nfr10 + 1):
        for j in range(1, flen + 1):
            e[i] = e[i] + numpy.square(fdata_[(i - 1) * fsh10 + j])

        if numpy.less_equal(e[i], ENERGYFLOOR):
            e[i] = ENERGYFLOOR

    segsnr = numpy.zeros(nfr10);
    segsnr = numpy.insert(segsnr, 0, 'inf')
    segsnrsmth = 1
    sign_segsnr = 0
    D = numpy.zeros(nfr10);
    D = numpy.insert(D, 0, 'inf')
    postsnr = numpy.zeros(nfr10, dtype='float64');
    postsnr = numpy.insert(postsnr, 0, 'inf')
    snre_vad = numpy.zeros(nfr10);
    snre_vad = numpy.insert(snre_vad, 0, 'inf')
    sign_pv = 0

    for i in range(1, nfr10 + 1):

        if (pvblk_[i] == 1) and (sign_pv == 0):
            nstart = i
            sign_pv = 1

        elif ((pvblk_[i] == 0) or (i == nfr10)) and (sign_pv == 1):

            nstop = i - 1
            if i == nfr10:
                nstop = i
            sign_pv = 0
            datai = fdata_[range((nstart - 1) * fsh10 + 1, (nstop - 1) * fsh10 + flen - fsh10 + 1)]
            datai = numpy.insert(datai, 0, 'inf')

            for j in range(nstart, nstop - 1 + 1):  # previously it was for j=nstart:nstop-1
                for h in range(1, flen + 1):
                    e[j] = e[j] + numpy.square(datai[(j - nstart) * fsh10 + h])
                if numpy.less_equal(e[j], ENERGYFLOOR):
                    e[j] = ENERGYFLOOR

            e[nstop] = e[nstop - 1]

            eY = numpy.sort(e[range(nstart, nstop + 1)])
            eY = numpy.insert(eY, 0, 'inf')  # as [0] is discarding

            emin = eY[int(numpy.floor((nstop - nstart + 1) * 0.1))]

            for j in range(nstart + 1, nstop + 1):

                postsnr[j] = math.log10(e[j]) - math.log10(emin)

                if numpy.less(postsnr[j], 0):
                    postsnr[j] = 0

                D[j] = math.sqrt(numpy.abs(e[j] - e[j - 1]) * postsnr[j])

            D[nstart] = D[nstart + 1]

            tm1 = numpy.hstack((numpy.ones(Dexpl) * D[nstart], D[range(nstart, nstop + 1)]))
            Dexp = numpy.hstack((tm1, numpy.ones(Dexpr) * D[nstop]))

            Dexp = numpy.insert(Dexp, 0, 'inf')

            for j in range(0, nstop - nstart + 1):
                Dsmth[nstart + j] = sum(Dexp[range(j + 1, j + Dexpl + Dexpr + 1)])

            Dsmth_thres = sum(Dsmth[range(nstart, nstop + 1)] * pv01_[range(nstart, nstop + 1)]) / sum(
                pv01_[range(nstart, nstop + 1)])

            for j in range(nstart, nstop + 1):
                if numpy.greater(Dsmth[j], Dsmth_thres * vadThres):
                    snre_vad[j] = 1

                    #
    pv_vad = deepcopy(snre_vad)

    nexpl = 33
    nexpr = 47  # % 29 and 39, estimated statistically, 95% ; 33, 47 %98 for voicebox pitch
    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            nstop = i - 1
            if i == nfr10:
                nstop = i
            sign_vad = 0
            for j in range(nstart, nstop + 1):
                if pv01_[j] == 1:
                    break

            pv_vad[range(nstart, numpy.max([j - nexpl - 1, 1]) + 1)] = 0

            for j in range(0, nstop - nstart + 1):
                if pv01_[nstop - j] == 1:
                    break

            pv_vad[range(nstop - j + 1 + nexpr, nstop + 1)] = 0

    nexpl = 5;
    nexpr = 12  # ; % 9 and 13, estimated statistically 5%; 5, 12 %2 for voicebox pitch
    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (snre_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((snre_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            nstop = i - 1
            if i == nfr10:
                nstop = i
            sign_vad = 0

            if numpy.greater(sum(pv01_[range(nstart, nstop + 1)]), 4):
                for j in range(nstart, nstop + 1):
                    if pv01_[j] == 1:
                        break

                pv_vad[range(numpy.maximum(j - nexpl, 1), j - 1 + 1)] = 1
                for j in range(0, nstop - nstart + 1):
                    if pv01_[nstop - j] == 1:
                        break
                pv_vad[range(nstop - j + 1, min(nstop - j + nexpr, nfr10) + 1)] = 1

            esegment = sum(e[range(nstart, nstop + 1)]) / (nstop - nstart + 1)
            if numpy.less(esegment, 0.001):
                pv_vad[range(nstart, nstop + 1)] = 0

            if numpy.less_equal(sum(pv01_[range(nstart, nstop + 1)]), 2):
                pv_vad[range(nstart, nstop + 1)] = 0

    sign_vad = 0
    esum = 0
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            nstop = i - 1
            if i == nfr10:
                nstop = i
            sign_vad = 0
            esum = esum + sum(e[range(nstart, nstop + 1)])

    #
    eps = numpy.finfo(float).eps

    eave = esum / (sum(pv_vad[1:len(pv_vad)]) + eps)  # except [0] index 'inf'

    sign_vad = 0
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            nstop = i - 1
            if i == nfr10:
                nstop = i
            sign_vad = 0

            # if numpy.less(sum(e[range(nstart,nstop+1)])/(nstop-nstart+1), eave*0.05):
            # pv_vad[range(nstart,nstop+1)] = 0

    #
    sign_vad = 0
    vad_seg = numpy.zeros((nfr10, 2), dtype="int64")
    n_vad_seg = -1  # for indexing array
    for i in range(1, nfr10 + 1):
        if (pv_vad[i] == 1) and (sign_vad == 0):
            nstart = i
            sign_vad = 1
        elif ((pv_vad[i] == 0) or (i == nfr10)) and (sign_vad == 1):
            nstop = i - 1
            sign_vad = 0
            n_vad_seg = n_vad_seg + 1
            # print i, n_vad_seg, nstart, nstop
            vad_seg[n_vad_seg, :] = numpy.array([nstart, nstop])

    vad_seg = vad_seg[:n_vad_seg + 1, ]

    # syn  from [0] index
    vad_seg = vad_seg - 1

    # print vad_seg

    # make one dimension array of (0/1)
    xYY = numpy.zeros(nfr10, dtype="int64")
    for i in range(len(vad_seg)):
        k = range(vad_seg[i, 0], vad_seg[i, 1] + 1)
        xYY[k] = 1

    vad_seg = xYY

    return vad_seg


def pitchblockdetect(pv01, pitch, nfr10, opts):
    pv01_ = deepcopy(pv01)

    if nfr10 == len(pv01_) + 1:
        numpy.append(pv01_, pv01_[nfr10 - 1])
    if opts == 0:
        sign_pv = 0
        for i in range(0, nfr10):

            if (pv01_[i] == 1) and (sign_pv == 0):

                nstart, sign_pv = i, 1

            elif ((pv01_[i] == 0) or (i == nfr10 - 1)) and (sign_pv == 1):

                nstop = i
                if i == nfr10 - 1:
                    nstop = i + 1
                sign_pv = 0
                pitchseg = numpy.zeros(nstop - nstart)
                # print len(pitchseg)
                for j in range(nstart, nstop):
                    pitchseg[j - nstart] = pitch[j];

                if (sum(numpy.abs(numpy.round(pitchseg - numpy.average(pitchseg)))) == 0) and (
                        nstop - nstart + 1 >= 10):
                    pv01_[range(nstart, nstop)] = 0
                    #
    sign_pv = 0
    pvblk = deepcopy(pv01_)

    # print i
    for i in range(0, nfr10):

        if (pv01_[i] == 1) and (sign_pv == 0):
            # print("i=%s " %(i))
            nstart, sign_pv = i, 1
            pvblk[range(max([nstart - 60, 0]), nstart + 1)] = 1
            # print("fm P2: i=%s %s % " %(i,max([nstart-60,0]), nstart+1))

        elif ((pv01_[i] == 0) or (i == nfr10 - 1)) and (sign_pv == 1):

            nstop, sign_pv = i, 0

            pvblk[range(nstop, numpy.amin([nstop + 60, nfr10 - 1]) + 1)] = 1
            # print("fm P2: i=%s %s %s " %(i,nstop, numpy.amin([nstop+60,nfr10-1])+1 ))

    return pvblk

