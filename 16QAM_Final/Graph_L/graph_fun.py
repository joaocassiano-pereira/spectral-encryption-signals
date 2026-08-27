# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 17:09:45 2021

@author: pc
"""

import sys
import string

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fftshift

from Data_L import input_data as inp

# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, str(inp.Path(__file__).parents[1]))
###############################################################################


def cleargraph():
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def graph1(signal,
           spec,
           time,
           freq,
           bitdisp,
           titles,
           tag,
           save=False):

    cleargraph()
    bits_lim = time[int(bitdisp*inp.spsymbol-1)]
    maxf = max(freq)
    shiftf = [x - maxf/2 for x in freq]

    axes = plt.gca()
    axes.set_xlim([0, bits_lim])
    # axes.set_ylim([minsig,maxsig])
    plt.plot(time, inp.np.real(signal), linestyle='-',
             color='black', linewidth=1.5)
    plt.stem(time, inp.np.real(signal), use_line_collection='True')
    plt.plot(time, inp.np.imag(signal), linestyle='-',
             color='red', linewidth=1.5)
    plt.stem(time, inp.np.imag(signal), use_line_collection='True')
    plt.axhline()
    plt.title(titles[0])
    plt.xlabel('Time (ps)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/graph_timedomain_"+tag+"_" +
                    inp.modu+"_"+randomnum+'.png', dpi=900)
        plt.show()

    axes = plt.gca()
    axes.set_xlim([-maxf/2, maxf/2])
    plt.plot(shiftf, fftshift(abs(spec)), linestyle='-',
             color='black', linewidth=1.2)
    plt.axhline()
    plt.title(titles[1])
    plt.xlabel('Frequency (GHz)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/graph_freqdomainamp_"+tag+"_" +
                    inp.modu+"_"+randomnum+'.png', dpi=900)
        plt.show()

    plt.plot(shiftf, fftshift(inp.np.angle(spec)),
             linestyle='-', color='black', linewidth=1.2)
    axes = plt.gca()
    axes.set_xlim([-maxf/2, maxf/2])
    plt.axhline()
    plt.title(titles[2])
    plt.xlabel('Frequency (GHz)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)

    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/graph_freqdomainangle_"+tag+"_" +
                    inp.modu+"_"+randomnum+'.png', dpi=900)
        plt.show()


###############################################################################

def constelation(signal_I,
                 signal_Q,
                 title,
                 tag,
                 save=False):
    """

    Plot the constelation diagram of a real or complex signal using its time domain data.

    Parameters
    ----------
    signal_I : TYPE
        DESCRIPTION.
    signal_Q : string
    title : TYPE
        Plot's title.
    tag : TYPE
        Plot's image file.
    save : Boolean, optional
        Enable or disable saving that image's file. The default is False.

    """
    if inp.Symbols_I >= inp.Symbols_Q:
        limit = inp.Symbols_I+2
    elif inp.Symbols_Q > inp.Symbols_I:
        limit = inp.Symbols_Q+2

    sam_sig_I = signal_I[0::inp.spsymbol]
    sam_sig_Q = signal_Q[0::inp.spsymbol]

    cleargraph()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.axis([-limit, limit, -limit, limit])
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.scatter(sam_sig_I, sam_sig_Q, s=6)
    ax.set_title(title)
    ax.set_xlabel('I', fontsize=18)
    ax.xaxis.set_label_coords(1.05, 0.55)
    ax.set_ylabel('Q', fontsize=18, rotation=0)
    ax.yaxis.set_label_coords(0.55, 1)

    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/const_"+tag+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
        plt.show()

###############################################################################


def filter_graph(fil,
                 f,
                 scale,
                 tag,
                 save=False):
    arrayfil = inp.np.array(fil, dtype="complex_")
    arrayfreq = inp.np.array(f)
    maxf = max(arrayfreq*scale)

    cleargraph()

    axes = plt.gca()
    axes.set_xlim([-1.1*maxf/2, 1.1*maxf/2])
    axes.set_ylim([-1.1*min(abs(fil)), 1.1*max(abs(fil))])
    plt.plot(arrayfreq*scale - maxf/2, fftshift(abs(arrayfil)),
             linestyle='-', color='black', linewidth=1.5)
    plt.axhline()
    plt.title('Amplitude Spectrum - '+tag+' Signal')
    plt.xlabel('Frequency (GHz)', fontsize=16)
    plt.ylabel('Amplitude', fontsize=16)

    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/filter_"+tag+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
        plt.show()


###############################################################################

def hist_graph(data, bins, title, save=False):
    cleargraph()
    plt.hist(data, bins)
    plt.title(title, fontsize=16)
    plt.xlabel('Bit error (%)', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/hist_"+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
    plt.show()

###############################################################################


def time_megaplot(data, time, bitdisp, save=False):
    bits_lim = time[int(bitdisp*inp.spsymbol-1)]

    cleargraph()
    cm = 1/2.54
    plt.rcParams["figure.figsize"] = [16*cm, 24*cm]  # [width, height]
    #plt.rcParams["figure.autolayout"] = True
    fig, axs = plt.subplots(len(data), 1)
    fig.tight_layout()
    
    for i, ax in enumerate(axs.flat):
        ax.set_xlim([-5, bits_lim])
        ax.tick_params(labelsize=18)
        ax.set_xticks(inp.np.arange(0, bits_lim+1, 4*inp.spsymbol*bitdisp))
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.stem(time, inp.np.real(data[i]),
                markerfmt="_", use_line_collection='True')
        ax.stem(time, inp.np.imag(data[i]),
                markerfmt="_", use_line_collection='True')
        ax.text(1.05, 0.7,
                "("+string.ascii_lowercase[i]+")",
                transform=ax.transAxes,
                fontsize=18)
    fig.supylabel('Amplitude (u.a.)', x=-0.08, fontsize=20)
    fig.supxlabel('Time (ps)', y=-0.04, fontsize=20)
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/time_mplot_"+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
    plt.show()

###############################################################################


def amp_megaplot(data, freq, save=False):
    cleargraph()
    maxf = max(freq)
    shiftf = [x - maxf/2 for x in freq]
    
    plt.rcParams["figure.figsize"] = [25, 15]
    #plt.rcParams["figure.autolayout"] = True
    figamp, axsamp = plt.subplots(len(data), 1)
    figamp.tight_layout()
    
    for i, axamp in enumerate(axsamp.flat):
        axamp.tick_params(labelsize=18)
        axamp.set_yticks([-100, 0, 500, 1000, 1001])
        axamp.set_xlim([-maxf/2, maxf/2])
        axamp.plot(shiftf, fftshift(
            abs(data[i])), linestyle='-', color='black', linewidth=1.1)
        axamp.text(1.05, 0.7,
                   "("+string.ascii_lowercase[i]+")",
                   transform=axamp.transAxes,
                   fontsize=18)
    figamp.supylabel('Amplitude (u.a.)', x=-0.02, fontsize=20)
    figamp.supxlabel('Frequency (GHz)', y=-0.02, fontsize=20)

    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/amp_mplot_"+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
    plt.show()

###############################################################################


def pha_megaplot(data, freq, save=False):
    cleargraph()
    maxf = max(freq)
    shiftf = [x - maxf/2 for x in freq]
    
    plt.rcParams["figure.figsize"] = [25, 15]
    #plt.rcParams["figure.autolayout"] = True
    plt.subplots_adjust(hspace=0)
    fig, axs = plt.subplots(len(data), 1)
    fig.tight_layout()
    
    for i, ax in enumerate(axs.flat):
        ax.tick_params(labelsize=18)
        ax.set_xlim([-maxf/2, maxf/2])
        ax.plot(shiftf, fftshift(inp.np.angle(
            data[i])), linestyle='-', color='black', linewidth=1.1)
        ax.text(1.05, 0.7,
                "("+string.ascii_lowercase[i]+")",
                transform=ax.transAxes,
                fontsize=18)
    fig.supylabel('Amplitude (u.a.)', fontsize=20)
    fig.supxlabel('Frequency (GHz)', fontsize=20)
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/amp_mplot_"+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
    plt.show()

###############################################################################


def const_megaplot(data_I,
                   data_Q,
                   save=False):

    cleargraph()

    if inp.Symbols_I >= inp.Symbols_Q:
        limit = inp.Symbols_I+2
    elif inp.Symbols_Q > inp.Symbols_I:
        limit = inp.Symbols_Q+2

    fig, axs = plt.subplots(1, len(data_I))
    for i, ax in enumerate(axs.flat):
        sam_sig_I = data_I[i][0::inp.spsymbol]
        sam_sig_Q = data_Q[i][0::inp.spsymbol]

        ax.set_aspect('equal', adjustable='box')
        ax.axis([-limit, limit, -limit, limit])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        ax.scatter(sam_sig_I, sam_sig_Q, s=6)
        ax.set_xlabel('I', fontsize=20)
        ax.xaxis.set_label_coords(1.05, 0.55)
        ax.set_xticks([-3, -1, 0, 1, 3])
        ax.set_ylabel('Q', fontsize=20, rotation=0)
        ax.yaxis.set_label_coords(0.55, 1)
        ax.set_yticks([-3, -1, 1, 3])
        ax.tick_params(labelsize=16)
        ax.text(1.05, 0.7,
                "("+string.ascii_lowercase[i]+")",
                transform=ax.transAxes,
                fontsize=18)

    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["figure.autolayout"] = True
    if save:
        randomnum = str(int(inp.np.random.rand()*10**8))
        plt.savefig("imgs/const_mplot_"+"_"+inp.modu+"_"+randomnum+'.png', dpi=900)
    plt.show()

###############################################################################


def amp_const_graph(freqdata, timedata, freq, time, save=False):
    cleargraph()
    maxf = max(freq)
    shiftf = [x - maxf/2 for x in freq]

    fig = plt.figure(tight_layout=True)
    fdatalen = len(freqdata)
    gs = GridSpec(fdatalen, 2, figure=fig)

    for i in range(fdatalen):
        axamp = fig.add_subplot(gs[i, 0])
        axamp.plot(shiftf, fftshift(
            abs(freqdata[i])), linestyle='-', color='black', linewidth=1.1)
        axamp.tick_params(labelsize=12)
        axamp.set_xlim([-maxf/2*1.1, maxf/2*1.1])
        axamp.text(1.1, 1.1,
                   "("+string.ascii_lowercase[i]+")",
                   transform=axamp.transAxes,
                   fontsize=14)

    if inp.Symbols_I >= inp.Symbols_Q:
        limit = inp.Symbols_I+2
    elif inp.Symbols_Q > inp.Symbols_I:
        limit = inp.Symbols_Q+2

    ideal_amp = [((2*inp.Symbols_I-2)/(inp.Symbols_I-1)*(i)) -
                 inp.Symbols_I+1 for i in range(0, inp.Symbols_I)]

    for i in range(fdatalen):
        axcons = fig.add_subplot(gs[i, 1])
        sam_sig_I = inp.np.real(timedata)[i][0::inp.spsymbol]
        sam_sig_Q = inp.np.imag(timedata)[i][0::inp.spsymbol]

        axcons.set_aspect('equal', adjustable='box')
        axcons.axis([-limit, limit, -limit, limit])
        axcons.scatter(sam_sig_I, sam_sig_Q, s=2)
        axcons.set_xlabel('I', fontsize=14)
        axcons.xaxis.set_label_coords(1.05, 0.1)
        axcons.set_xticks(ideal_amp)
        axcons.set_ylabel('Q', fontsize=14, rotation=0)
        axcons.yaxis.set_label_coords(0.1, 1.05)
        axcons.set_yticks(ideal_amp)
        axcons.tick_params(labelsize=12)

    fig.supylabel('Amplitude (u.a.)', fontsize=14)
    fig.supxlabel('Frequency (GHz)', x=0.35, fontsize=14)
    cm = 1/2.54
    plt.rcParams["figure.figsize"] = [16*cm, 24*cm]  # [width, height]
    plt.tight_layout()
    plt.show()
