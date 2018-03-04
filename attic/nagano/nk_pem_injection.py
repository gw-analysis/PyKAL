#! /usr/bin/env python
# -*- coding: utf-8 -*-

from glue.lal import Cache
import os          # os and sys modules are required to load gwf file using cache
import sys
from gwpy.timeseries import TimeSeries 
import numpy as np
import matplotlib
matplotlib.use('agg')                                    
import matplotlib.pylab as plt
from scipy import signal


def get_lsd(cache, channel, GPSstart, duration, stride, overlap,
        pltflg=0, filename='', yunit=''):
    '''
    This function calculates linear spectral density.
    If pltflg==1, make plot.
    Return: lsd, freq, mu(meanvalue)
    '''
    data = TimeSeries.read(cache, channel, GPSstart, GPSstart + duration, format='lalframe')
    fs = 1./data.dt.value  # sampling frequency
    mu = np.mean(data.value)  # mean value of data
    # Define parameters for FFT
    st = stride  # FFT stride in seconds
    ov = st*overlap # overlap in seconds 
    nfft = int(st*fs)
    freq, t, Sxx = signal.spectrogram(data.value-mu, window=signal.hann(nfft), nperseg=nfft,
                                      fs=fs, noverlap=int(ov*fs))
    lspg = np.sqrt(Sxx)
    lsd = np.mean(lspg, axis=1)
    if pltflg==1:
        plt.figure()
        plt.loglog(freq, lsd)
        plt.grid(True)
        plt.title(channel)
        plt.xlim(freq[1:].min(), freq[1:].max())
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('LSD ['+yunit+'/rtHz]')
        plt.tight_layout()
        fname = filename
        plt.savefig(fname)
        plt.close()
    return lsd, freq, mu


def get_spectrogram(cache, channel, GPSstart, duration, stride, overlap,
        pltflg=0, filename='',rtnflg=0):
    '''
    This function calculated linear spectrogram.
    If pltflg==1, makes plot.
    If rtnflg==1, returns lspg, time, freq, mu
    '''
    data = TimeSeries.read(cache, channel, GPSstart, GPSstart + duration, format='lalframe')
    fs = 1./data.dt.value  # sampling frequency
    mu = np.mean(data.value)  # mean value of data
    # Define parameters for FFT
    st = stride  # FFT stride in seconds
    ov = st*overlap # overlap in seconds 
    nfft = int(st*fs)
    freq, t, Sxx = signal.spectrogram(data.value-mu, window=signal.hann(nfft), nperseg=nfft,
                                      fs=fs, noverlap=int(ov*fs))
    lspg = np.sqrt(Sxx)
    if pltflg==1:
        T, FREQ = np.meshgrid(t, freq)
        fig2, ax = plt.subplots()
        pc = ax.pcolor(T, FREQ, 10*np.log10(lspg), cmap='rainbow')
        cb = fig2.colorbar(pc)
        fname = filename
        plt.sacefig(fname)
        plt.close()
    if rtnflg==1:
        return lspg, t, freq, mu


def compare_lsds(cache, filename, valueunit, channel, GPSstart1, GPSstart2, duration, stride,
        excesslevel=100, pltspectrumflg=0, pltexcessflg=0, returnflg=0):
    '''
    This function compares two lsds.
    '''
    data1 = TimeSeries.read(cache, channel, GPSstart1, GPSstart1 + duration, format='lalframe')
    data2 = TimeSeries.read(cache, channel, GPSstart2, GPSstart2 + duration, format='lalframe')
    fs1 = 1./data1.dt.value
    fs2 = 1./data2.dt.value
    mu1 = np.mean(data1.value)
    mu2 = np.mean(data2.value)
    # Define parameters for FFT
    st = stride # FFT stride in seconds
    overlap = st*0.5 # overlap inseconds
    nov1 = int(overlap*fs1)
    nov2 = int(overlap*fs2)
    nfft1 = int(st*fs1)
    nfft2 = int(st*fs2)
    window1 = signal.hann(nfft1)
    window2 = signal.hann(nfft2)
    freq1, t1, Sxx1 = signal.spectrogram(data1.value-mu1,window=window1, nperseg=nfft1,
                                         fs=fs1, noverlap=nov1)
    lspg1 = np.sqrt(Sxx1)
    lsd1 = np.mean(lspg1, axis=1)
    freq2, t2, Sxx2 = signal.spectrogram(data2.value-mu2,window=window2, nperseg=nfft2,
                                         fs=fs2, noverlap=nov2)
    lspg2 = np.sqrt(Sxx2)
    lsd2 = np.mean(lspg2, axis=1)
    # Compensate the difference of sampling frequency
    if fs1 > fs2:
        lsd1_tmp = lsd1*(freq1 <= freq2[-1])
        freq1_tmp = (freq1+10)*(freq1 <= freq2[-1])
        lsd1 = lsd1_tmp[np.nonzero(lsd1_tmp)]
        freq1 = freq1_tmp[np.nonzero(freq1_tmp)]-10
    if fs2 > fs1:
        lsd2_tmp = lsd2*(freq2 <= freq1[-1])
        freq2_tmp = (freq2+10)*(freq2 <= freq1[-1])
        lsd2 = lsd2_tmp[np.nonzero(lsd2_tmp)]
        freq2 = freq2_tmp[np.nonzero(freq2_tmp)]-10

    el = excesslevel
    lsd2_excess_tmp = lsd2*(lsd2 > el*lsd1)
    lsd2_excess = lsd2_excess_tmp[np.nonzero(lsd2_excess_tmp)]

    freq_excess_tmp = (freq2+10)*(lsd2 > el*lsd1)
    freq_excess = freq_excess_tmp[np.nonzero(freq_excess_tmp)]-10

    if pltspectrumflg==1:
        fig1 = plt.figure()
        plt.loglog(freq1, lsd1, '--', lw=2, alpha=1, label=str(GPSstart1))
        plt.loglog(freq2, lsd2, lw=2, alpha=0.8, label=str(GPSstart2))
        if pltexcessflg==1:
            plt.loglog(freq_excess, lsd2_excess, lw=0, markersize=4, marker='o',
                    label='excess (>'+str(el)+')')
        plt.legend(loc='best')
        plt.grid(True)
        plt.title(channel)
        plt.xlim(freq1[1:].min(), freq1[1:].max())
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('LSD [' + valueunit + '/rtHz]')
        plt.tight_layout()
        fname = filename
        plt.savefig(fname)
        plt.close()

    if returnflg == 1:
        return lsd1, lsd2, freq1, lsd2_excess, freq_excess


def couplingfunc_nk(cache, filename, valueunit, channel1, channel2, GPSref,
        GPSstart, duration, stride, excesslevel=100, pltcouplingflg=0, returnflg=0):
    '''
    This function calculates coupling function (channel)/(channel2).
    Calculation frequency was decided based on the excess from stable measurement
    of channel1.
    '''
    # Read data
    data0 = TimeSeries.read(cache, channel1, GPSref, GPSref+duration, format='lalframe')
    data1 = TimeSeries.read(cache, channel1, GPSstart, GPSstart+duration, format='lalframe')
    data2 = TimeSeries.read(cache, channel2, GPSstart, GPSstart+duration, format='lalframe')
    # sampling frequency
    fs0 = 1./data0.dt.value
    fs1 = 1./data1.dt.value
    fs2 = 1./data2.dt.value
    # mean value
    mu0 = np.mean(data0.value)
    mu1 = np.mean(data1.value)
    mu2 = np.mean(data2.value)
    # Define parameters for FFT
    st = stride # FFT stride in seconds
    overlap = st*0.5 # overlap inseconds
    nov0 = int(overlap*fs0)
    nov1 = int(overlap*fs1)
    nov2 = int(overlap*fs2)
    nfft0 = int(st*fs0)
    nfft1 = int(st*fs1)
    nfft2 = int(st*fs2)
    window0 = signal.hann(nfft0)
    window1 = signal.hann(nfft1)
    window2 = signal.hann(nfft2)
    freq0, t0, Sxx0 = signal.spectrogram(data0.value-mu0,window=window0, nperseg=nfft0,
                                         fs=fs0, noverlap=nov0)
    lspg0 = np.sqrt(Sxx0)
    lsd0 = np.mean(lspg0, axis=1)
    freq1, t1, Sxx1 = signal.spectrogram(data1.value-mu1,window=window1, nperseg=nfft1,
                                         fs=fs1, noverlap=nov1)
    lspg1 = np.sqrt(Sxx1)
    lsd1 = np.mean(lspg1, axis=1)
    freq2, t2, Sxx2 = signal.spectrogram(data2.value-mu2,window=window2, nperseg=nfft2,
                                         fs=fs2, noverlap=nov2)
    lspg2 = np.sqrt(Sxx2)
    lsd2 = np.mean(lspg2, axis=1)
    # compensate sampling frequency
    if fs1 > fs2:
        lsd0_tmp = lsd0*(freq0 <= freq2[-1])
        lsd1_tmp = lsd1*(freq1 <= freq2[-1])
        freq0_tmp = (freq0+10)*(freq0 <= freq2[-1])
        freq1_tmp = (freq1+10)*(freq1 <= freq2[-1])
        lsd0 = lsd0_tmp[np.nonzero(lsd0_tmp)]
        lsd1 = lsd1_tmp[np.nonzero(lsd1_tmp)]
        freq0 = freq0_tmp[np.nonzero(freq0_tmp)]-10
        freq1 = freq1_tmp[np.nonzero(freq1_tmp)]-10
    if fs2 > fs1:
        lsd2_tmp = lsd2*(freq2 <= freq1[-1])
        freq2_tmp = (freq2+10)*(freq2 <= freq1[-1])
        lsd2 = lsd2_tmp[np.nonzero(lsd2_tmp)]
        freq2 = freq2_tmp[np.nonzero(freq2_tmp)]-10
    
    el = excesslevel
    lsd1_excess_tmp = lsd1*(lsd1 > el*lsd0)
    lsd1_excess = lsd1_excess_tmp[np.nonzero(lsd1_excess_tmp)]
    lsd2_excess_tmp = lsd2*(lsd1 > el*lsd0)
    lsd2_excess = lsd2_excess_tmp[np.nonzero(lsd2_excess_tmp)]

    freq_excess_tmp = (freq1+10)*(lsd1 > el*lsd0)
    freq_excess = freq_excess_tmp[np.nonzero(freq_excess_tmp)]-10
    
    coupling = lsd1/lsd2
    coupling_excess = lsd1_excess/lsd2_excess
    if pltcouplingflg==1:
        plt.figure()
        plt.loglog(freq1, coupling, lw=2, label='coupling')
        plt.loglog(freq_excess, coupling_excess, lw=0, alpha=0.6, markersize=4, marker='o',
                label='excess (>'+str(el)+')')
        plt.legend(loc='best')
        plt.grid(True)
        plt.title(channel1+'/'+channel2)
        plt.xlim(freq1[1:].min(), freq1[1:].max())
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Coupling [' + valueunit + ']')
        plt.tight_layout()
        fname = filename
        plt.savefig(fname)
        plt.close()
    if returnflg == 1:
        return coupling, freq1, coupling_excess, freq_excess

def projection_nk(cache, filename, yunit, channel1, channel2, channel3, GPSref, GPSstart,
        duration, stride, excesslevel=100, pltspectrumflg=0, returnflg=0):
    '''
    This function perform noise projection to channel1 from channel3.
    Coupling function is calculated from (channel1)/(channel2).
    channel1: interferometer
    channel2: excited sensor
    channel3: stable sensor
    GPSref: stable time
    GPSstart: excitation time
    '''
    # read data
    data0 = TimeSeries.read(cache, channel1, GPSref, GPSref+duration, format='lalframe')
    data1 = TimeSeries.read(cache, channel1, GPSstart, GPSstart+duration, format='lalframe')
    data2 = TimeSeries.read(cache, channel2, GPSstart, GPSstart+duration, format='lalframe')
    data3 = TimeSeries.read(cache, channel3, GPSref, GPSref+duration, format='lalframe')
    # sampling frequency
    fs0 = 1./data0.dt.value
    fs1 = 1./data1.dt.value
    fs2 = 1./data2.dt.value
    fs3 = 1./data3.dt.value
    # mean value
    mu0 = np.mean(data0.value)
    mu1 = np.mean(data1.value)
    mu2 = np.mean(data2.value)
    mu3 = np.mean(data3.value)
    # Define parameters for FFT
    st = stride # FFT stride in seconds
    overlap = st*0.5 # overlap inseconds
    nov0 = int(overlap*fs0)
    nov1 = int(overlap*fs1)
    nov2 = int(overlap*fs2)
    nov3 = int(overlap*fs3)
    nfft0 = int(st*fs0)
    nfft1 = int(st*fs1)
    nfft2 = int(st*fs2)
    nfft3 = int(st*fs3)
    window0 = signal.hann(nfft0)
    window1 = signal.hann(nfft1)
    window2 = signal.hann(nfft2)
    window3 = signal.hann(nfft3)
    freq0, t0, Sxx0 = signal.spectrogram(data0.value-mu0,window=window0, nperseg=nfft0,
                                         fs=fs0, noverlap=nov0)
    lspg0 = np.sqrt(Sxx0)
    lsd0 = np.mean(lspg0, axis=1)
    freq1, t1, Sxx1 = signal.spectrogram(data1.value-mu1,window=window1, nperseg=nfft1,
                                         fs=fs1, noverlap=nov1)
    lspg1 = np.sqrt(Sxx1)
    lsd1 = np.mean(lspg1, axis=1)
    freq2, t2, Sxx2 = signal.spectrogram(data2.value-mu2,window=window2, nperseg=nfft2,
                                         fs=fs2, noverlap=nov2)
    lspg2 = np.sqrt(Sxx2)
    lsd2 = np.mean(lspg2, axis=1)
    freq3, t3, Sxx3 = signal.spectrogram(data3.value-mu3,window=window3, nperseg=nfft3,
                                         fs=fs3, noverlap=nov3)
    lspg3 = np.sqrt(Sxx3)
    lsd3 = np.mean(lspg3, axis=1)
    freq_min = min([freq0[-1], freq1[-1], freq2[-1], freq3[-1]])

    # compensate sampling frequency difference
    lsd0_tmp = lsd0*(freq0 <= freq_min)
    lsd1_tmp = lsd1*(freq1 <= freq_min)
    lsd2_tmp = lsd2*(freq2 <= freq_min)
    lsd3_tmp = lsd3*(freq3 <= freq_min)
    freq0_tmp = (freq0+10)*(freq0 <= freq_min)
    freq1_tmp = (freq1+10)*(freq1 <= freq_min)
    freq2_tmp = (freq2+10)*(freq2 <= freq_min)
    freq3_tmp = (freq3+10)*(freq3 <= freq_min)
    lsd0 = lsd0_tmp[np.nonzero(lsd0_tmp)]
    lsd1 = lsd1_tmp[np.nonzero(lsd1_tmp)]
    lsd2 = lsd2_tmp[np.nonzero(lsd2_tmp)]
    lsd3 = lsd3_tmp[np.nonzero(lsd3_tmp)]
    freq0 = freq0_tmp[np.nonzero(freq0_tmp)]-10
    freq1 = freq1_tmp[np.nonzero(freq1_tmp)]-10
    freq2 = freq2_tmp[np.nonzero(freq2_tmp)]-10
    freq3 = freq3_tmp[np.nonzero(freq3_tmp)]-10

    el = excesslevel
    lsd1_excess_tmp = lsd1*(lsd1 > el*lsd0)
    lsd1_excess = lsd1_excess_tmp[np.nonzero(lsd1_excess_tmp)]
    lsd2_excess_tmp = lsd2*(lsd1 > el*lsd0)
    lsd2_excess = lsd2_excess_tmp[np.nonzero(lsd2_excess_tmp)]
    lsd3_excess_tmp = lsd3*(lsd1 > el*lsd0)
    lsd3_excess = lsd3_excess_tmp[np.nonzero(lsd3_excess_tmp)]

    freq_excess_tmp = (freq1+10)*(lsd1 > el*lsd0)
    freq_excess = freq_excess_tmp[np.nonzero(freq_excess_tmp)]-10
    
    coupling = lsd1/lsd2
    coupling_excess = lsd1_excess/lsd2_excess
    noise_projection = lsd3_excess*coupling_excess
    if pltspectrumflg==1:
        plt.figure()
        plt.loglog(freq0, lsd0, lw=2, label='Stable sensitivity')
        plt.loglog(freq_excess, noise_projection, '--', lw=1, alpha=0.8,
                label='Projection (excess (>'+str(el)+'))')

        plt.legend(loc='best')
        plt.grid(True)
        plt.title('Noise projection ('+channel3+')')
        plt.xlim(freq0[1:].min(), freq0[1:].max())
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('LSD [' + yunit + '/rtHz]')
        plt.tight_layout()
        fname = filename
        plt.savefig(fname)
        plt.close()
    if returnflg == 1:
        return noise_projection, freq_excess
