#!/bin/env python

def CouplingArrange(cache_drc, ach, bch, s, d, a=None, df=None, f0=None, f1=None) :

    ## [1] cache_drc   :: Enter Cache file direction
    ## [2] ach         :: Enter first channel name to compare
    ## [3] bch         :: Enter second channel name to compare
    ## [4] s           :: Enter start GPS time
    ## [5] d           :: Enter time from star GPS time
    ## [6] a           :: Enter Averaging Number, CAN NOT be used with -df at the same time
    ## [7] df          :: Enter Frequency interval, df, CAN NOT be used with -a at the same time.
    ## [8] f0          :: Enter start frequency to obtain amplitude value list
    ## [9] f1          :: Enter end frequency to obtain amplitude value list

    import sys
    import os
    import numpy as np
    from scipy.interpolate import interp1d

    from gwpy.timeseries import TimeSeries 

    from glue.lal import Cache
    from gwpy.segments import Segment


    ########### List of times to want to calculate ###########

    gst = s    # GPS start time
    get = gst + d     # GPS end time
    dur = d     # Time duration

    ach = str(ach)     # Slected A Channel
    bch = str(bch)     # Slected B Chennel

    avg = a     # Slected Averaging Number
    df = df     # Slected Frequency Interval

    if avg == None and df == None :
        avg = 100
        bvg = float(dur/avg)

    elif avg == 0 and df == None:
        avg = 1  
        bvg = float(dur/avg)

    elif avg != None and df == None :
        avg = int(avg)
        bvg = float(dur/avg)

    elif avg == None and df != None :
        df = float(df)
        avg = int(df*dur)
        bvg = float(1/df)

    else :
        avg = 1  
        bvg = float(dur/avg)   


    ########### Reading Cache ###########

    gwf_cache = cache_drc

    with open(gwf_cache, 'r') as fobj:
        cache = Cache.fromfile(fobj)


    ########### Reaiding TimeSeries Data ###########

    data1 = TimeSeries.read(cache, ach , gst, get, format='lalframe')
    data2 = TimeSeries.read(cache, bch , gst, get, format='lalframe')
       
    ########### TimeSeries Data Averaging Process ###########

    data1_psd_seg = TimeSeries.read(cache, ach , gst, gst+(bvg), format='lalframe').psd()** (1/2.)
    for n in range(avg) :
        if gst+(bvg*(n+2)) > gst+(bvg*(n+1)) and gst+(bvg*(n+2)) < get :
            data1_seg = TimeSeries.read(cache, ach , gst+(bvg*(n+1)), gst+(bvg*(n+2)), format='lalframe')
            data1_psd_seg += data1_seg.psd()** (1/2.)
        else :
            pass

    data1_psd = data1_psd_seg / int(avg)

    data2_psd_seg = TimeSeries.read(cache, bch , gst, gst+(bvg), format='lalframe').psd()** (1/2.)
    for n in range(avg) :
        if gst+(bvg*(n+2)) > gst+(bvg*(n+1)) and gst+(bvg*(n+2)) < get :
            data2_seg = TimeSeries.read(cache, bch , gst+(bvg*(n+1)), gst+(bvg*(n+2)), format='lalframe')
            data2_psd_seg += data2_seg.psd()** (1/2.)
        else :
            pass

    data2_psd = data2_psd_seg / int(avg)


    ########### Calculate PSD Ratio of two channels using Interpolation process ###########

    data1_df = float(str(data1_psd.df)[:-2])
    data2_df = float(str(data2_psd.df)[:-2])

    data1_psd_yarr = np.array(data1_psd)
    data1_psd_xarr = np.linspace(data1_psd.xspan[0], data1_psd.xspan[-1], len(data1_psd_yarr) )

    data2_psd_yarr = np.array(data2_psd)
    data2_psd_xarr = np.linspace(data2_psd.xspan[0], data2_psd.xspan[-1], len(data2_psd_yarr) )

    g1 = interp1d(data1_psd_xarr, data1_psd_yarr)
    g2 = interp1d(data2_psd_xarr, data2_psd_yarr)

    h_xlist = []
    h_ylist = []

    if len(data1_psd_xarr) >= len(data2_psd_xarr) :
        for a in range((len(data2_psd_xarr))) :
            h_xlist.append(a*data2_df)
            h_ylist.append( float(g1(a*data2_df)) / float(g2(a*data2_df)) )
        hx = np.array(h_xlist)
        hy = np.array(h_ylist)
        
    else :
        for b in range((len(data1_psd_xarr))) :
            h_xlist.append(b*data1_df)
            h_ylist.append( float(g1(b*data1_df)) / float(g2(b*data1_df)) )
        hx = np.array(h_xlist)
        hy = np.array(h_ylist)

    h = interp1d(hx, hy)


    ############ Find amplitude for selected frequency ############

    freq0 = f0
    freq1 = f1

    if freq0 != None and freq1 != None :
        amp_list = []
        amp_list_x = []
        amp_list_y = []
        for f in range(len(h_xlist)):
            if h_xlist[f] <= freq1 and freq0 <= h_xlist[f] :
                amp_list_x.append(h_xlist[f])
                amp_list_y.append(h(h_xlist[f]) )
        amp_list.append(amp_list_x)
        amp_list.append(amp_list_y)

        return amp_list


    else :
        amp_list = []
        amp_list.append(h_xlist)
        amp_list.append(h_ylist)

        return amp_list




def CouplingPoints(cache_drc, ach, bch, s, d, a=None, df=None, freq_list):

    ## [1] cache_drc   :: Enter Cache file direction
    ## [2] ach         :: Enter first channel name to compare
    ## [3] bch         :: Enter second channel name to compare
    ## [4] s           :: Enter start GPS time
    ## [5] d           :: Enter time from star GPS time
    ## [6] a           :: Enter Averaging Number, CAN NOT be used with -df at the same time
    ## [7] df          :: Enter Frequency interval, df, CAN NOT be used with -a at the same time.
    ## [8] freq_list   :: Enter LIST of frequencies to obtain amplitude value list

    import sys
    import os
    import numpy as np
    from scipy.interpolate import interp1d

    from gwpy.timeseries import TimeSeries 

    from glue.lal import Cache
    from gwpy.segments import Segment


    ########### List of times to want to calculate ###########

    gst = s    # GPS start time
    get = gst + d     # GPS end time
    dur = d     # Time duration

    ach = str(ach)     # Slected A Channel
    bch = str(bch)     # Slected B Chennel

    avg = a     # Slected Averaging Number
    df = df     # Slected Frequency Interval

    if avg == None and df == None :
        avg = 100
        bvg = float(dur/avg)

    elif avg == 0 and df == None:
        avg = 1  
        bvg = float(dur/avg)

    elif avg != None and df == None :
        avg = int(avg)
        bvg = float(dur/avg)

    elif avg == None and df != None :
        df = float(df)
        avg = int(df*dur)
        bvg = float(1/df)

    else :
        avg = 1  
        bvg = float(dur/avg)   

    if type(freq_list) == list :
        pass

    ########### Reading Cache ###########

    gwf_cache = cache_drc

    with open(gwf_cache, 'r') as fobj:
        cache = Cache.fromfile(fobj)


    ########### Reaiding TimeSeries Data ###########

    data1 = TimeSeries.read(cache, ach , gst, get, format='lalframe')
    data2 = TimeSeries.read(cache, bch , gst, get, format='lalframe')
       
    ########### TimeSeries Data Averaging Process ###########

    data1_psd_seg = TimeSeries.read(cache, ach , gst, gst+(bvg), format='lalframe').psd()** (1/2.)
    for n in range(avg) :
        if gst+(bvg*(n+2)) > gst+(bvg*(n+1)) and gst+(bvg*(n+2)) < get :
            data1_seg = TimeSeries.read(cache, ach , gst+(bvg*(n+1)), gst+(bvg*(n+2)), format='lalframe')
            data1_psd_seg += data1_seg.psd()** (1/2.)
        else :
            pass

    data1_psd = data1_psd_seg / int(avg)

    data2_psd_seg = TimeSeries.read(cache, bch , gst, gst+(bvg), format='lalframe').psd()** (1/2.)
    for n in range(avg) :
        if gst+(bvg*(n+2)) > gst+(bvg*(n+1)) and gst+(bvg*(n+2)) < get :
            data2_seg = TimeSeries.read(cache, bch , gst+(bvg*(n+1)), gst+(bvg*(n+2)), format='lalframe')
            data2_psd_seg += data2_seg.psd()** (1/2.)
        else :
            pass

    data2_psd = data2_psd_seg / int(avg)


    ########### Calculate PSD Ratio of two channels using Interpolation process ###########

    data1_df = float(str(data1_psd.df)[:-2])
    data2_df = float(str(data2_psd.df)[:-2])

    data1_psd_yarr = np.array(data1_psd)
    data1_psd_xarr = np.linspace(data1_psd.xspan[0], data1_psd.xspan[-1], len(data1_psd_yarr) )

    data2_psd_yarr = np.array(data2_psd)
    data2_psd_xarr = np.linspace(data2_psd.xspan[0], data2_psd.xspan[-1], len(data2_psd_yarr) )

    g1 = interp1d(data1_psd_xarr, data1_psd_yarr)
    g2 = interp1d(data2_psd_xarr, data2_psd_yarr)

    h_xlist = []
    h_ylist = []

    if len(data1_psd_xarr) >= len(data2_psd_xarr) :
        for a in range((len(data2_psd_xarr))) :
            h_xlist.append(a*data2_df)
            h_ylist.append( float(g1(a*data2_df)) / float(g2(a*data2_df)) )
        hx = np.array(h_xlist)
        hy = np.array(h_ylist)
        
    else :
        for b in range((len(data1_psd_xarr))) :
            h_xlist.append(b*data1_df)
            h_ylist.append( float(g1(b*data1_df)) / float(g2(b*data1_df)) )
        hx = np.array(h_xlist)
        hy = np.array(h_ylist)

    h = interp1d(hx, hy)


    ############ Find amplitude for selected frequency ############

    amp_list = []
    amp_list_y = []

    for f in freq_list :

        amp_list_y.append(h(f))

    amp_list.append(freq_list)
    amp_list.append(amp_list_y)

    return amp_list
