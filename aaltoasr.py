# -*- coding: utf-8 -*-
# Overall settings and paths to data files
rootdir = '/homeappl/appl_taito/ling/aaltoasr/1.1/'

import argparse
import math
import os
import re
import select
import shutil
import struct
import sys
import tempfile
import textwrap
from itertools import groupby
from os.path import basename, join
import subprocess
#from subprocess import call, check_output
from subprocess import call,PIPE
#sys.path.append(rootdir+"/lxml-lxml-fd4a5c5/build/lib.linux-x86_64-2.7")
from lxml import etree
from datetime import datetime
from mimetypes import guess_type
import fileinput
import time
import codecs
import io




# Set your decoder swig path in here!
sys.path.append(rootdir+'/lib/site-packages')
import Decoder
import string

from multiprocessing import Pool
import multiprocessing
import copy_reg, functools
from functools import partial


def _reconstruct_partial(f, args, kwds):
    return functools.partial(f, *args, **(kwds or {}))

def _reduce_partial(p):
    return _reconstruct_partial, (p.func, p.args, p.keywords)

copy_reg.pickle(functools.partial, _reduce_partial) 


"""Aalto ASR tools for CSC Hippu environment.

This module contains the required glue code for using the Aalto ASR
tools for simple speech recognition and forced alignment tasks, in the
CSC Hippu environment.
"""


models = {
    'fi': { 'path': 'fi/speecon_all_multicondition_mmi_kld0.002_6',
             'srate': 16000, 'fstep': 128,
             'regtree': True,
             'lm':'fi/morph19k_D20E10_varigram.bin',
             'lookahead':'fi/morph19k_2gram.bin',
             'lexicon':'fi/morph19k.lex',
             'morph-model':True,
             'lang':'Finnish',
             'default': True },
    'fi-conversational': { 'path': 'fi/puhekieli2016c',
             'srate': 16000, 'fstep': 128,
             'regtree': True,
             'lm':'fi/baseline.4bo.bin',
             'lookahead':'fi/baseline.2bo.bin',
             'lexicon':'fi/baseline.lex',
             'morph-model':False,
             'lang':'Finnish',
             'default': True },
    'swe': { 'path': 'swe/swedish_mfcc_14.4.2015_22',
             'srate': 16000, 'fstep': 128,
             'regtree': True,
             'lm':'swe/swedish_6g_120k.bin',
             'lookahead':'swe/swedish_2g_120k.bin',
             'lexicon':'swe/swedish_120k.lex' ,
             'morph-model':False,
             'lang':'Swedish',},
    'am-en': { 'path': 'am-en/wsj0_ml_pronaligned_tied1_gain3500_occ200_20.2.2010_22',
             'srate': 16000, 'fstep': 128,
             'regtree': True,
             'lm':'am-en/gigaword_trigram.bin',
             'lookahead':'am-en/gigaword_bigram.bin',
             'lexicon':'am-en/gigaword_word_60000.dict' ,
             'morph-model':False,
             'lang':'American English'},
    'gb-en': { 'path': 'gb-en/mfcc_wsjcam0_d_16k_12.11.2009_20',
             'srate': 16000, 'fstep': 128,
             'regtree': True,
             'lm':'gb-en/gigaword_trigram.bin',
             'lookahead':'gb-en/gigaword_bigram.bin',
             'lexicon':'gb-en/beep_word_60000.dict' ,
             'morph-model':False,
             'lang':'British English'},
    }

def bin(prog):
    return join(rootdir, 'bin', prog)

default_args = {
    'model': [m for m, s in models.items() if 'default' in s][0],
    'lmscale': 30,
    'align-window': 1000,
    'align-beam': 100.0,
    'align-sbeam': 100,
    }

# Command-line help

help = {
    'rec': {
        'desc': 'Recognize speech from an audio file.',
        'usage': '%(prog)s [options] input [input ...]',
        'modes': ('trans', 'segword', 'segmorph', 'segphone'),
        'defmode': 'trans',
        'extra': ('The MODE parameter specifies which results to include in the generated output.  '
                  'It has the form of a comma-separated list of the terms "trans", "segword", '
                  '"segmorph" and "segphone", denoting a transcript of the recognized text, '
                  'a word-level, statistical-morpheme-level or phoneme-level segmentation, '
                  'respectively. The listed items will be included in the plaintext output. '
                  'The default is "trans". For more details, see the User\'s Guide at: '
                  'http://research.spa.aalto.fi/speech/aaltoasr-guide/')
        },
    'align': {
        'desc': 'Align a transcription to a speech audio file.',
        'usage': '%(prog)s [options] -t transcript [-t transcript ...] input [input ...]',
        'modes': ('segword', 'segphone'),
        'defmode': 'segword',
        'extra': ('The MODE parameter specifies which results to include in the generated output. '
                  'It has the form of a comma-separated list of the terms "segword" and "segphone", '
                  'denoting a word-level or phoneme-level segmentation, respectively. The listed '
                  'items will be included in the plaintext output. The default is "segword".  '
                  'For more details, see the User\'s Guide at: '
                  'http://research.spa.aalto.fi/speech/aaltoasr-guide/')
        }
    }

class ModelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values == 'list':
            sys.stderr.write('supported models:\n')
            for m in sorted(models.keys()):
                if 'default' in models[m].keys():
                    sys.stderr.write('%s: %s [sample rate: %d Hz] (default)\n' % (m,models[m]['lang'], models[m]['srate']))
                else:
                    sys.stderr.write('%s: %s [sample rate: %d Hz]\n' % (m,models[m]['lang'], models[m]['srate']))
            sys.exit(2)
        setattr(namespace, self.dest, values)

# Class for implementing the rec/align/adapt tools

class AaltoASR(object):
    def __init__(self, tool, args=None):
        """Initialize and parse command line attributes."""

        self.tool = tool

        # Ask argparse to grok the command line arguments

        thelp = help[tool]

        parser = argparse.ArgumentParser(description=thelp['desc'], usage=thelp['usage'], epilog=thelp['extra'])

        parser.add_argument('input', help='input audio file', nargs='+')
        parser.add_argument('-t', '--trans', help='provide an input transcript file', metavar='file',
                            required=True if tool == 'align' else False, action='append')
        parser.add_argument('-a', '--adapt', help='provide a speaker adaptation file', metavar='file',
                            default=None)
        parser.add_argument('-o', '--output', help='output results to file [default stdout]', metavar='file',
                            type=argparse.FileType('w'), default=sys.stdout)
        parser.add_argument('-m', '--mode', help='which kind of results to output (see below)', metavar='MODE',
                            default=thelp['defmode'])
        parser.add_argument('-T', '--tg', help='output a Praat TextGrid segmentation to file', metavar='file',
                            default=None)
        parser.add_argument('-E', '--elan', help='output an ELAN (EAF) segmentation to file', metavar='file',
                            action='append')
        if tool == 'rec':
            parser.add_argument('-r', '--raw', help='produce raw recognizer output (with morph breaks)',
                                action='store_true')
            parser.add_argument('-s', '--split',
                                help='split input file to segments of about S seconds [default %(const)s if present]',
                                metavar='S', nargs='?', type=float, default=None, const=60.0)
        parser.add_argument('-c', '--corr-len', help='corrected length of transcript file (in seconds)',
                                metavar='c', nargs='?', type=float, default=None, const=0.0)
        parser.add_argument('-C', '--corr-trans', help='provide a partially corrected transcript file (EAF)', metavar='file', action='append')


        parser.add_argument('-n', '--cores', help='run tasks simultaneously on up to N cores [default 1]',
                            metavar='N', type=int, default=1)
        parser.add_argument('-v', '--verbose', help='print output also from invoked commands', action='store_true')
        parser.add_argument('-q', '--quiet', help='do not print status messages', action='store_true')
        parser.add_argument('--tempdir', help='directory for temporary files', metavar='D')

        params = parser.add_argument_group('speech recognition parameters')
        params.add_argument('-M', '--model', help='ASR model to use; "-M list" for list [default "%(default)s"]',
                            metavar='M', default=default_args['model'], choices=['list']+list(models.keys()),
                            action=ModelAction)
        if tool == 'rec':
            params.add_argument('-L', '--lmscale', help='language model scale factor [default %(default)s]', metavar='L',
                                type=int, default=default_args['lmscale'])
        if tool == 'align':
            params.add_argument('--align-window',
                                help='set the Viterbi window for alignment [default %(default)s]', metavar='W',
                                type=int, default=default_args['align-window'])
            params.add_argument('--align-beam', help='set alignment log-probability beam [default %(default)s]', metavar='B',
                                type=float, default=default_args['align-beam'])
            params.add_argument('--align-sbeam', help='set alignment state beam [default %(default)s]', metavar='S',
                                type=int, default=default_args['align-sbeam'])
        params.add_argument('--noexp', help='disable input transcript expansion', action='store_true')

        parser.add_argument('--keep', help=argparse.SUPPRESS, action='store_true')

        self.args = parser.parse_args(args)

        # Check applicable arguments for validity
        for infile in self.args.input:
            if not os.access(infile, os.R_OK):
                err('input file not readable: {0}'.format(infile), exit=2)

        self.transfiles = None
        if self.args.trans:
            self.transfiles = [open(f, 'rb') for f in self.args.trans]
            if len(self.transfiles) != len(self.args.input):
                err('number of transcript files does not match number of inputs', exit=2)


        self.corr_transfiles = None
        if tool == 'rec':
            if self.args.corr_trans:
                self.corr_transfile_names = self.args.corr_trans
                self.corr_transfiles = [open(f, 'rb') for f in self.args.corr_trans]
                if len(self.corr_transfiles) != len(self.args.input):
                    err('number of transcript files does not match number of inputs', exit=2)

        self.adaptcfg = { 'feature': False, 'model': False }
        if self.args.adapt is not None:
            with open(self.args.adapt) as f:
                for line in f:
                    if line.find('feature cmllr') >= 0: self.adaptcfg['feature'] = True
                    if line.find('model cmllr') >= 0: self.adaptcfg['model'] = True

        self.mode = set()
        for word in self.args.mode.split(','):
            if word not in thelp['modes']:
                err('invalid output mode: %s; valid: %s' % (word, ', '.join(thelp['modes'])), exit=2)
            self.mode.add(word)

        self.tg = self.args.tg is not None
        
        self.elanfiles = []
        if self.args.elan is not None:
            for elanfile in self.args.elan:
                self.elanfiles.append(elanfile)
            if len(self.elanfiles) != len(self.args.input):
                err('number of ELAN transcript files does not match number of inputs', exit=2)

        self.model = models[self.args.model]
        if self.adaptcfg['model'] and 'regtree' not in self.model:
            err('selected model {0} not compatible with -r adaptation'.format(self.args.model), exit=2)
        self.mpath = join(rootdir, 'model', self.model['path'])
        self.mcfg = self.mpath + ('.adapt.cfg' if self.adaptcfg['feature'] else '.cfg')
        self.margs = ['-b', self.mpath, '-c', self.mcfg]
        self.mgcl = self.mpath + '.regtree.gcl' if self.adaptcfg['model'] else self.mpath + '.gcl'

        self.lm = join(rootdir, 'model', self.model['lm'])
        self.lookahead = join(rootdir, 'model', self.model['lookahead'])
        self.lexicon = join(rootdir, 'model', self.model['lexicon'])
        self.morph_model = self.model['morph-model']

        if self.args.adapt is not None:
            self.margs.extend(('-S', self.args.adapt))

        self.cores = self.args.cores


    def __enter__(self):
        """Make a working directory for a single execution."""

        self.workdir = tempfile.mkdtemp(prefix='aaltoasr', dir=self.args.tempdir)

        return self

    def __exit__(self, type, value, traceback):
        """Clean up the working directory and any temporary files."""

        if self.workdir.find('aaltoasr') >= 0 and not self.args.keep: # sanity check
            shutil.rmtree(self.workdir)


    def convert_input(self):
        """Convert input audio to something suitable for the model."""
        self.audiofiles = []

        for idx, infile in enumerate(self.args.input):
            fileid = 'input-{0}'.format(idx)
            base = join(self.workdir, fileid)
            finfo = { 'id': fileid,
                      'path': infile }
            audiofile = base + '.wav'
            if self.tool == 'rec' and self.args.split is not None:
                self.log('splitting input file {0} to {1}-second segments'.format(infile, self.args.split))
                #Check if supervised adaptation activated
                if self.args.corr_len is not None:
                    #split file from corrected part
                    if call([bin('sox'), infile,'-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                         audiofile,'trim',str(self.args.corr_len)]) != 0:
                        err("input conversion of '%s' with sox failed" % self.args.input, exit=1)
                    finfo['files'] = split_audio(self.args.split, audiofile, base, self.model) 
                else:   
                    finfo['files'] = split_audio(self.args.split, infile, base, self.model)
            else:
                self.log('converting input file {0} to {1} Hz mono'.format(infile, self.model['srate']))
                #Check if supervised adaptation activated
                if self.args.corr_len is not None:
                    if call([bin('sox'), infile,'-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                         audiofile,'trim',str(self.args.corr_len)]) != 0:
                        err("input conversion of '%s' with sox failed" % self.args.input, exit=1)
                    finfo['files'] = [{ 'start': 0, 'file': audiofile }]
                else:
                    audiofile = base + '.wav'
                    if call([bin('sox'), infile,'-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                         audiofile]) != 0:
                        err("input conversion of '%s' with sox failed" % self.args.input, exit=1)
                    finfo['files'] = [{ 'start': 0, 'file': audiofile }]

            self.audiofiles.append(finfo)
        nfiles = sum(len(finfo['files']) for finfo in self.audiofiles)

        if self.cores > nfiles:
            self.cores = nfiles
            self.log('using only {0} core{1}; no more audio segments'.format(
                    self.cores, '' if self.cores == 1 else 's'))


    def align(self):
        """Do segmentation with the Viterbi alignment tool."""

        self.convert_input()
        #audiofile = self.audiofiles[0]['file']

        self.log('computing Viterbi alignment')

        recipe = join(self.workdir, 'align.recipe')
        alignfiles = [join(self.workdir, '{0}.phn'.format(f['id'])) for f in self.audiofiles]
        outfiles = [join(self.workdir, '{0}.align'.format(f['id'])) for f in self.audiofiles]

        self.phones = [None] * len(self.audiofiles)

        for fidx, finfo in enumerate(self.audiofiles):
            audiofile = finfo['files'][0]['file'] # never split when aligning
            # Convert the transcription to a phoneme list
            self.log('converting text to phn')
            phones = text2phn(self.transfiles[fidx], self.workdir, expand=not self.args.noexp)
            self.phones[fidx] = phones
            # Write out the cross-word triphone transcript
            f = open(alignfiles[fidx],'w')
            #with io.open(alignfiles[fidx], 'w', encoding='iso-8859-1') as f:
            for pnum, para in enumerate(phones):
                phns = para['phns']
                f.write('__\n')
                for phnum, ph in enumerate(phns):
                    if ph == '_':
                        f.write('_ #{0}:{1}\n'.format(pnum, phnum))
                    else:
                        prevph, nextph = '_', '_'
                        for prev in range(phnum-1, -1, -1):
                            if phns[prev] != '_': prevph = phns[prev]; break
                        for next in range(phnum+1, len(phns)):
                            if phns[next] != '_': nextph = phns[next]; break
                        f.write('{0}-{1}+{2} #{3}:{4}\n'.format(prevph.encode('iso-8859-15'), ph.encode('iso-8859-15'), nextph.encode('iso-8859-15'), pnum, phnum))
            f.write('__\n')

        # Make a recipe for the alignment

        with open(recipe, 'w') as f:
            for fidx, finfo in enumerate(self.audiofiles):
                f.write('audio={0} transcript={1} alignment={2} speaker=UNK\n'.format(
                        finfo['files'][0]['file'], alignfiles[fidx], outfiles[fidx]))

        # Run the Viterbi alignment

        cmd = [bin('align'),
               '-r', recipe, '-i', '1',
               '--swins', str(self.args.align_window),
               '--beam', str(self.args.align_beam),
               '--sbeam', str(self.args.align_sbeam)]
        cmd.extend(self.margs)
        self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))

        self.alignments = outfiles


    


    def phone_probs(self):
        #Check if corrected input segment exits, perform MAP adaptation
        """Create a LNA file for the provided audio."""
        self.convert_input()
        self.log('computing acoustic model likelihoods')

        recipe = join(self.workdir, 'input.recipe')
        lnafiles = []

        # Construct an input recipe
        # audiofiles are independent spoken docments
        for fidx, finfo in enumerate(self.audiofiles):
            f = open(recipe,"w")
            lnas = []
            adap_recipe_files = []
            eval_recipe_files = []
            adap_len = 0.0
            for i, audiofile in enumerate(finfo['files']):   
                lnafile = join(self.workdir, '{0}-{1}.lna'.format(finfo['id'], i))
                lnas.append(lnafile)
                f.write('audio={0} lna={1} speaker=UNK\n'.format(audiofile['file'], lnafile))
            f.close()
            #If supervised in-document adaptation activated, perform MAP adaptation
            if self.args.corr_len is not None:
                #Retrieve corrected part first
                #Original uncut audiofile
                infile = finfo['path'] 
                fileid = finfo['id'] 
                base = join(self.workdir, fileid)
                adap_audiofile = base+"_adap.wav"
                if call([bin('sox'), infile,'-t', 'wav', '-r', str(self.model['srate']), '-b', '16', '-e', 'signed-integer', '-c', '1',
                         adap_audiofile,'trim','0',str(self.args.corr_len)]) != 0:
                    err("input conversion of '%s' with sox failed" % self.args.input, exit=1)
                #Retrieve corrected part from the EAF file
                start_time_slot = ""
                end_time_slot = ""
                corrected_time_slots = []
                prev_line = ""
                corrected_transcript = ""
                for line in fileinput.input(self.corr_transfile_names[fidx]):
                    line = line.strip()                    
                    if line.startswith("<TIME_SLOT"):
                        ext,time_slot_id,time_value =line.split(" ",2)
                        time_slot_id = time_slot_id.replace("TIME_SLOT_ID=","")
                        time_slot_id = time_slot_id.replace("\"","")
                        time_slot_id = time_slot_id.strip()
                        time_value = time_value.replace("TIME_VALUE=","")
                        time_value = time_value.replace("/>","")
                        time_value = time_value.replace("\"","")
                        #Convert to seconds
                        time_value = float(time_value)/1000.0
                        if time_value < self.args.corr_len:
                            corrected_time_slots.append(time_slot_id)   
                    elif line.startswith("<ANNOTATION_VALUE"):
                        ext,annotation_id,time_slot_ref_1,time_slot_ref_2 = prev_line.split(" ",3)
                        time_slot_ref_1 = time_slot_ref_1.replace("TIME_SLOT_REF1=","")
                        time_slot_ref_1 = time_slot_ref_1.replace("\"","")
                        time_slot_ref_2 = time_slot_ref_2.replace("TIME_SLOT_REF2=","")
                        time_slot_ref_2 = time_slot_ref_2.replace("\"","")
                        time_slot_ref_2 = time_slot_ref_2.replace(">","")
                        if time_slot_ref_2 in corrected_time_slots:
                            word = line.replace("<ANNOTATION_VALUE>","")
                            word = word.replace("</ANNOTATION_VALUE>","")
                            corrected_transcript += word+" "
                    elif line.startswith("<ALIGNABLE_ANNOTATION"):
                        prev_line = line
                corrected_transcript = corrected_transcript.strip()             
                #Make alignment between corrected part and text
                phones = transcript2phn(corrected_transcript.decode('utf-8'),self.workdir, expand=not self.args.noexp)
                #phones = text2phn(self.corr_transfiles[fidx], self.workdir, expand=not self.args.noexp)
                align_filename = adap_audiofile.replace(".wav",".align")
                phn_filename = adap_audiofile.replace(".wav",".phn")
                phn_file = io.open(phn_filename, 'w',encoding='iso-8859-15')
                for pnum, para in enumerate(phones):
                    phns = para['phns']
                    phn_file.write(u'__\n')
                    for phnum, ph in enumerate(phns):
                        if ph == '_':
                            phn_file.write(u'_\n')
                        else:
                            prevph, nextph = '_', '_'
                            for prev in range(phnum-1, -1, -1):
                                if phns[prev] != '_': prevph = phns[prev]; break
                            for next in range(phnum+1, len(phns)):
                                if phns[next] != '_': nextph = phns[next]; break
                            phn_file.write(u'{0}-{1}+{2}\n'.format(prevph, ph, nextph))
                phn_file.write(u'__\n')
                phn_file.close()
                hmmnet_filename = adap_audiofile.replace(".wav",".hmmnet")
                recipeline = "audio="+adap_audiofile+" transcript="+phn_filename+" alignment="+align_filename+" hmmnet="+hmmnet_filename+"\n"
                recipe_filename = adap_audiofile.replace(".wav",".recipe")
                recipe_file = open(recipe_filename,"w")
                recipe_file.write(recipeline)
                recipe_file.close()
                #Run the Viterbi alignment, run with default parameters
                #cmd = [bin('align'),
                #'-r', recipe_filename, '-i', '1']
                #cmd.extend(self.margs)
                #self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))
                os.system(rootdir+"/bin/align -r "+recipe_filename+" -i 1 -b "+self.mpath+" -c "+self.mcfg)
                os.system("cp "+align_filename+" "+phn_filename)
                #Adapt baseline HMM
                aku_script_dir = rootdir+"/scripts/aku_scripts/"
                init_model = self.mpath
                base_id = fileid
                tau = 5
                num_ml_train_iter = 1 
                map_type = 2
                num_batches = 1 
                target_hmm = base
                os.system(aku_script_dir+"map.pl "+init_model+" "+base_id+" "+str(tau)+" "+str(num_ml_train_iter)+" "+str(map_type)+" "+recipe_filename+" "+str(num_batches)+" "+target_hmm)
                lnafiles.append(lnas)
                #Run phone_probs on the files with MAP-adapted model
                target_hmm_cfg = target_hmm+".cfg"
                cmd = [bin('phone_probs'),
                '-r', recipe, '-b',target_hmm, '-c',target_hmm_cfg]
                self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))
                self.lna = lnafiles
                self.adapted_hmm = target_hmm
            else: 
                lnafiles.append(lnas)
                #Run phone_probs on the files with baseline model
                cmd = [bin('phone_probs'),
                '-r', recipe, '-C', self.mgcl, '-i', '1',
                '--eval-ming', '0.2']
                cmd.extend(self.margs)
                self.run(cmd, batchargs=lambda i, n: ('-B', str(n), '-I', str(i)))
                self.lna = lnafiles





    def rec(self):
        """Run the recognizer for the generated LNA file."""
        self.log('recognizing speech')
        # Call rec.py on the lna files
        lnamap = {}
        histmap = {}

        if self.args.corr_len is not None:
            #Run with adapted HMM
            recipe = join(self.workdir, 'rec.recipe')
            with open(recipe, 'w') as f:
                for fidx, lnafiles in enumerate(self.lna):
                    for i, lnafile in enumerate(lnafiles):
                        lna_id = basename(lnafile)
                        lnamap[lna_id] = (fidx, i)
                        histmap[(fidx, i)] = len(histmap)
                        f.write('lna={0}\n'.format(lna_id))

            #Multiprocessing decoding
            pool = Pool(processes=self.cores)  
            seg_phone = '1' if 'segphone' in self.mode or self.tg or self.elanfiles else '0'
            morphseg_file = join(self.workdir, 'wordhist') if 'segmorph' in self.mode or 'segword' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None) else ''
            recipe_file = open(recipe,"r")
            recipelines = recipe_file.readlines()
            recipe_file.close()
            rec_output_vector = pool.map(partial(decode_func,model=self.adapted_hmm,lexicon=self.lexicon,ngram=self.lm,lookahead_ngram=self.lookahead,lna_path=self.workdir,lm_scale=str(self.args.lmscale),do_phoneseg=seg_phone,morphseg_file=morphseg_file),recipelines)
            rec_out = ''.join(o for o in rec_output_vector)
        else:
            #Run with baseline HMM
            recipe = join(self.workdir, 'rec.recipe')
            with open(recipe, 'w') as f:
                for fidx, lnafiles in enumerate(self.lna):
                    for i, lnafile in enumerate(lnafiles):
                        lna_id = basename(lnafile)
                        lnamap[lna_id] = (fidx, i)
                        histmap[(fidx, i)] = len(histmap)
                        f.write('lna={0}\n'.format(lna_id))

            #Multiprocessing decoding
            pool = Pool(processes=self.cores)  
            seg_phone = '1' if 'segphone' in self.mode or self.tg or self.elanfiles else '0'
            morphseg_file = join(self.workdir, 'wordhist') if 'segmorph' in self.mode or 'segword' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None) else ''
            recipe_file = open(recipe,"r")
            recipelines = recipe_file.readlines()
            recipe_file.close()
            rec_output_vector = pool.map(partial(decode_func,model=self.mpath,lexicon=self.lexicon,ngram=self.lm,lookahead_ngram=self.lookahead,lna_path=self.workdir,lm_scale=str(self.args.lmscale),do_phoneseg=seg_phone,morphseg_file=morphseg_file),recipelines)
            rec_out = ''.join(o for o in rec_output_vector)
            ##########################
        # Parse the recognizer output to extract recognition result and state segmentation
        rec_file, rec_filepart = -1, -1
        rec_start = 0
        rec_trans = [[[] for i in lnafiles] for lnafiles in self.lna]
        rec_seg = [[[] for i in lnafiles] for lnafiles in self.lna]

        re_lna = re.compile(r'^LNA: (.*)$')
        re_trans = re.compile(r'^REC: (.*)$')
        re_seg = re.compile(r'^(\d+) (\d+) (\d+)$')
        
        ##########################################

        for line in rec_out.splitlines():
            m = re_lna.match(line)
            if m is not None:
                rec_file, rec_filepart = lnamap[m.group(1)]
                rec_start = self.audiofiles[rec_file]['files'][rec_filepart]['start']
                continue
            m = re_trans.match(line)
            if m is not None:
                rec_trans[rec_file][rec_filepart].append(m.group(1))
                continue
            m = re_seg.match(line)
            if m is not None:
                start, end, state = m.group(1, 2, 3)
                rec_seg[rec_file][rec_filepart].append((rec_start+int(start), rec_start+int(end), int(state)))
                continue

        rec_trans = [[i for l in translist for i in l] for translist in rec_trans]
        rec_seg = [[i for l in seglist for i in l] for seglist in rec_seg]

        if not all(trans for trans in rec_trans):
            sys.stderr.write(rec_out)
            err('unable to find recognition transcript in output', exit=1)


        if self.morph_model == True:
            self.rec = [' <w> '.join(trans).strip() for trans in rec_trans]
        else:
            self.rec = [' '.join(trans) for trans in rec_trans]
   
        fstep = self.model['fstep']

        # If necessary, find labels for states and write an alignment file
        if 'segphone' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None):
            labels = get_labels(self.mpath + '.ph')

            self.alignments = []

            for fidx, finfo in enumerate(self.audiofiles):
                alignment = join(self.workdir, '{0}.align'.format(finfo['id']))
                with io.open(alignment, 'w', encoding='iso-8859-1') as f:
                    for start, end, state in rec_seg[fidx]:
                        f.write('%d %d %s\n' % (start*fstep, end*fstep, labels[state]))
                self.alignments.append(alignment)

        # If necessary, parse the generated word history file

        if 'segmorph' in self.mode or 'segword' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None):
            re_line = re.compile(r'^(\S+)\s+(\d+)')
            self.morphsegs = []
            for fidx, finfo in enumerate(self.audiofiles):
                morphseg = []

                for i, audiofile in enumerate(finfo['files']):
                    seg = []
                    file_start = audiofile['start']
                    prev_end = file_start

                    with io.open(join(self.workdir, 'wordhist-{0}'.format(histmap[(fidx,i)])), 'r', encoding='iso-8859-1') as f:
                        for line in f:
                            m = re_line.match(line)
                            if m is None: continue # skip malformed
                            morph, end = m.group(1), file_start + int(m.group(2))
                            seg.append((prev_end*fstep, end*fstep, morph))
                            prev_end = end

                    while len(seg) > 0 and (seg[0][2] == '<s>' or seg[0][2] == '<w>'):
                        seg.pop(0)
                    while len(seg) > 0 and (seg[-1][2] == '</s>' or seg[-1][2] == '<w>'):
                        seg.pop()

                    if len(seg) == 0:
                        continue
                    if len(morphseg) > 0:
                        morphseg.append((morphseg[-1][1], seg[0][0], '<w>'))
                    morphseg.extend(seg)

                self.morphsegs.append(morphseg)


    def gen_output(self):
        """Construct the requested outputs."""

        out = self.args.output
        out_started = [False]

        def hdr(text, suffix='\n\n'):
            if out_started[0]: out.write('\n\n')
            out.write('### {0}{1}'.format(text, suffix))
            out_started[0] = True

        for fidx, finfo in enumerate(self.audiofiles):
            if len(self.audiofiles) > 1:
                hdr('Input file: {0}'.format(finfo['path']), suffix='\n')
            self.gen_output_one(fidx, hdr)

    def gen_output_one(self, fidx, hdr):
        """Construct the requested outputs for one file."""

        # Start by parsing the state alignment into a phoneme segmentation
        if 'segphone' in self.mode or ('segword' in self.mode and self.tool == 'align') or self.tg or self.elanfiles:
            # Parse the raw state-level alignment

            rawseg = []

            re_line = re.compile(r'^(\d+) (\d+) ([^\.]+)\.(\d+)(?: #(\d+):(\d+))?')
            with io.open(self.alignments[fidx], 'r', encoding='iso-8859-1') as f:
                for line in f:
                    m = re_line.match(line)
                    if m is None:
                        err('invalid alignment line: %s' % line, exit=1)
                    phpos = (int(m.group(5)), int(m.group(6))) if m.group(5) else None
                    rawseg.append((int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), phpos))
            limits = (rawseg[0][0], rawseg[-1][1])

            # Recover the phoneme level segments from the state level alignment file

            phseg = []

            cur_ph, cur_state = None, 0

            for start, end, rawph, state, phpos in rawseg:
                ph = trip2ph(rawph)

                if ph == cur_ph and state == cur_state + 1:
                    # Hack: try to fix cases where the first state of a phoneme after a long silence
                    # actually includes (most of) the silence too.  Ditto for last states.
                    if rawph.startswith('_-') and state == 1 and start - phseg[-1][0] > 2*(end-start):
                        phseg[-1][0] = start
                    if rawph.endswith('+_') and state == 2 and end - start > 2*(start - phseg[-1][0]):
                        continue # don't update end
                    phseg[-1][1] = end
                else:
                    phseg.append([start, end, ph, phpos])

                cur_ph, cur_state = ph, state

            # Split into list of utterances

            uttseg = []

            for issep, group in groupby(phseg, lambda item: item[2] == '__'):
                if not issep:
                    uttseg.append(list(group))

        # Merge phoneme segmentation to words in transcript if aligning

        if ('segword' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None)) and self.tool == 'align':
            warned = False

            phonepos = dict((pos, (start, end, ph))
                            for start, end, ph, pos
                            in (i for utt in uttseg for i in utt))
            wordseg = []

            at = 0
            for uttidx, utt in enumerate(self.phones[fidx]):
                phstack = []

                for phidx, ph in enumerate(utt['phns']):
                    if (uttidx, phidx) in phonepos:
                        start, end, aph = phonepos[(uttidx, phidx)]
                        # push things forward if necessary because of gaps
                        if start < at: start = at
                        if end <= start: end = start + 1
                        if aph != ph:
                            err('segmenter confused: phoneme mismatch at {0}/{1}: {2} != {3}'.format(uttidx, phidx, aph, ph), exit=1)
                        at = end
                    else:
                        if not warned:
                            self.log('warning: gaps in aligned output (check transcript?)')
                            warned = True
                        start, end = at, at+1
                        at = at+1
                    phstack.append((start, end, ph))

                wseg = []

                for w in intersperse(self.phones[fidx][uttidx]['words'], '_'):
                    start, end = 0, 0
                    for phnum, ph in enumerate(w.replace('-', '_')):
                        if phnum == 0: start = phstack[0][0]
                        end = phstack[0][1]
                        phstack.pop(0)
                    if w != '_': wseg.append((start, end, w))

                wordseg.append(wseg)

        # Merge morpheme segmentation into words if required
        if ('segword' in self.mode or self.tg or self.elanfiles or (self.args.corr_len is not None)) and self.tool == 'rec':
            if self.morph_model == True:
                wordseg = []
                for issep, group in groupby(self.morphsegs[fidx], lambda item: item[2] == '<s>' or item[2] == '</s>'):
                    if issep: continue
                    utt = []

                    for issep, wordgroup in groupby(group, lambda item: item[2] == '<w>'):
                        if issep: continue
                        word = list(wordgroup)
                        utt.append((word[0][0], word[-1][1], ''.join(m[2] for m in word)))

                    wordseg.append(utt)
            else:
                wordseg = []
                segs = []
                for m in self.morphsegs[fidx]:
                    token = m[2]
                    if token != "<w>":
                        segs.append(m)
                    else:
                        segs.append((m[0],m[1],'.'))    
                wordseg.append(segs)
        # Generate requested plaintext outputs
        out = self.args.output
        srate = float(self.model['srate'])

   
        ####
        if 'trans' in self.mode:
            hdr('Recognizer transcript:')
            if not self.args.raw:
                if self.morph_model:
                    for utt in self.rec[fidx].split('<s>'):
                        utt = utt.replace('</s>', '')
                        utt = utt.replace(' ', '')
                        utt = utt.replace('<w>', ' ').strip()
                        if len(utt) > 0:
                            out.write('%s\n' % utt.encode('utf-8'))
                else:
                    for utt in self.rec[fidx].split('<s>'):
                        utt = utt.replace('</s>', '')
                        utt = utt.strip()
                        if len(utt) > 0:
                            out.write('%s\n' % utt.encode('utf-8'))
                  
            else:
                out.write('%s\n' % self.rec[fidx].encode('utf-8'))

        if 'trans' in self.mode and self.args.trans:
            hdr('Recognition accuracy:')
            phones = text2phn(self.transfiles[fidx], self.workdir, expand=not self.args.noexp)
            sclite(out, self.rec[fidx], phones, self.workdir)

        if 'segword' in self.mode:
            hdr('Word-level segmentation:')
            for utt in intersperse(wordseg, ()):
                if len(utt) == 0: out.write('\n')
                else:
                    for start, end, w in utt:
                        out.write('%.3f %.3f %s\n' % (start/srate, end/srate, w.encode('utf-8')))

        if 'segmorph' in self.mode:
            hdr('Morpheme[*]-level segmentation:   ([*] statistical)')
            for utt in intersperse(self.morphsegs, ()):
                if len(utt) == 0: out.write('\n\n')
                else:
                    for start, end, morph in utt:
                        if morph == '<s>' or morph == '</s>': continue
                        elif morph == '<w>': out.write('\n')
                        else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, morph.encode('utf-8')))

        if 'segphone' in self.mode:
            hdr('Phoneme-level segmentation:')
            for utt in intersperse(uttseg, ()):
                if len(utt) == 0: out.write('\n\n')
                else:
                    for start, end, ph, phpos in utt:
                        if ph == '_': out.write('\n')
                        else: out.write('%.3f %.3f %s\n' % (start/srate, end/srate, ph.encode('utf-8')))

        # Generate Praat TextGrid output
        if self.tg:
            tiers = []
            tiers.append({ 'name': 'utterance',
                           'data': [(utt[0][0],
                                     utt[-1][1],
                                     ' '.join(w[2] for w in utt))
                                    for utt in wordseg] })

            tiers.append({ 'name': 'word',
                           'data': [word for utt in wordseg for word in utt] })

            if self.tool == 'rec':
                tiers.append({ 'name': 'morph',
                               'data': [morph for ms in self.morphsegs for morph in ms if morph[2][0] != '<'] })

            tiers.append({ 'name': 'phone',
                           'data': [ph[:3] for ph in phseg if ph[2][0] != '_'] })

            if len(self.audiofiles) > 1:
                tgfile = '{0}.{1}'.format(self.args.tg, fidx+1)
            else:
                tgfile = self.args.tg
            with open(tgfile, 'w') as tgf:
                tg_write(tgf, tiers, limits, srate)




        # Generate ELAN output
        if self.elanfiles:
            asr_word_segmentations = []
            for utt in intersperse(wordseg, ()):
                if len(utt) == 0: 
                    out.write('\n')
                else:
                    for start, end, w in utt:
                        start_ms = (start/srate)*1000
                        end_ms = (end/srate)*1000
                        token = w.strip()
                        asr_out = str(int(start_ms))+"\t"+str(int(end_ms))+"\t"+token+"\t"+"Speaker"
                        asr_word_segmentations.append(asr_out)
            wavfile = self.args.input[fidx]
            eaffile = self.elanfiles[fidx]
            write_elan(wavfile,asr_word_segmentations,eaffile)  

        #Write corrected part first if supplied
        if self.args.corr_len is not None:
            #Write to corrected EAF file
            start_time_slot = ""
            end_time_slot = ""
            corrected_time_slots = []
            time_slots = dict()
            prev_line = ""
            corrected_transcript = ""
            asr_word_segmentations = []
            for line in fileinput.input(self.corr_transfile_names[fidx]):
                line = line.strip()                    
                if line.startswith("<TIME_SLOT"):
                    ext,time_slot_id,time_value =line.split(" ",2)
                    time_slot_id = time_slot_id.replace("TIME_SLOT_ID=","")
                    time_slot_id = time_slot_id.replace("\"","")
                    time_slot_id = time_slot_id.strip()
                    time_value = time_value.replace("TIME_VALUE=","")
                    time_value = time_value.replace("/>","")
                    time_value = time_value.replace("\"","")
                    time_slots[time_slot_id] = time_value 
                    #Convert to seconds
                    time_value = float(time_value)/1000.0
                    if time_value <= self.args.corr_len:
                        corrected_time_slots.append(time_slot_id)   
                elif line.startswith("<ANNOTATION_VALUE"):
                    ext,annotation_id,time_slot_ref_1,time_slot_ref_2 = prev_line.split(" ",3)
                    time_slot_ref_1 = time_slot_ref_1.replace("TIME_SLOT_REF1=","")
                    time_slot_ref_1 = time_slot_ref_1.replace("\"","")
                    time_slot_ref_2 = time_slot_ref_2.replace("TIME_SLOT_REF2=","")
                    time_slot_ref_2 = time_slot_ref_2.replace("\"","")
                    time_slot_ref_2 = time_slot_ref_2.replace(">","")
                    if time_slot_ref_2 in corrected_time_slots:
                        word = line.replace("<ANNOTATION_VALUE>","")
                        word = word.replace("</ANNOTATION_VALUE>","")
                        start = time_slots[time_slot_ref_1]
                        end = time_slots[time_slot_ref_2]
                        speaker = "Speaker"
                        corrected_transcript += word+" "
                        segmentation = start+"\t"+end+"\t"+word+"\t"+speaker
                        asr_word_segmentations.append(segmentation.decode('utf-8'))
                elif line.startswith("<ALIGNABLE_ANNOTATION"):
                    prev_line = line
            
            #Process the recognized part
            for utt in intersperse(wordseg, ()):
                if len(utt) == 0: 
                    out.write('\n')
                else:
                    for start, end, w in utt:
                        start_ms = (start/srate)*1000+(self.args.corr_len*1000)
                        end_ms = (end/srate)*1000+(self.args.corr_len*1000)
                        token = w.strip()
                        asr_out = str(int(start_ms))+"\t"+str(int(end_ms))+"\t"+token+"\t"+"Speaker"
                        asr_word_segmentations.append(asr_out)
            wavfile = self.args.input[fidx]
            eaffile = self.corr_transfile_names[fidx]
            write_elan(wavfile,asr_word_segmentations,eaffile)  

    def adapt(self, output, modeladapt):
        """Generate a CMLLR adaptation transform from aligned output."""

        if any(len(f['files']) != 1 for f in self.audiofiles):
            err('impossible: adaptation with splitting enabled', exit=1)

        self.log('cleaning up aligned transcriptions')

        for a in self.alignments:
            alignment_fixup(a, a+'.fixup')

        self.log('training CMLLR adaptation matrix')

        recipe = join(self.workdir, 'adapt.recipe')
        with open(recipe, 'w') as f:
            for fidx, finfo in enumerate(self.audiofiles):
                f.write('audio={0} alignment={1}.fixup speaker=UNK\n'.format(finfo['files'][0]['file'], self.alignments[fidx]))

        # prepare speaker configuration file based on previous adaptation

        if (not modeladapt and self.adaptcfg['feature']) or (modeladapt and self.adaptcfg['model']):
            # use existing speaker config directly
            spk = self.args.adapt
        elif self.args.adapt is not None:
            # add current adaptation style to existing speaker config
            spk = join(self.workdir, 'adapt.spk')
            #with open(spk, 'w') as f, open(self.args.adapt) as oldf:
            with open(spk, 'w') as f:
                oldf = open(self.args.adapt)
                done = False
                for line in oldf:
                    f.write(line)
                    if not done and line.find('{') >= 0:
                        f.write('\n'.join(['{0} cmllr'.format('model' if modeladapt else 'feature'),
                                           '{', '}', '']))
                        done = True
        else:
            # make new empty speaker configuration
            spk = join(self.workdir, 'adapt.spk')
            with open(spk, 'w') as f:
                f.write('\n'.join(['speaker UNK', '{',
                                   '{0} cmllr'.format('model' if modeladapt else 'feature'),
                                   '{', '}', '}', '']))

        cmd_out = sys.stderr if self.args.verbose else open(os.devnull, 'w')

        cmd = [bin('mllr'),
               '-b', self.mpath,
               '-c', self.mpath + ('.adapt.cfg' if not modeladapt or self.adaptcfg['feature'] else '.cfg'),
               '-r', recipe, '-S', spk,
               '-o', output,
               '-O', '-i', '1']
        if modeladapt:
            cmd.extend(('-R', self.mpath + '.regtree'))
        else:
            cmd.extend(('-M', 'cmllr'))

        if self.args.verbose:
            self.log('run: {0}'.format(' '.join(cmd)))
        if call(cmd, stdout=cmd_out, stderr=cmd_out) != 0:
            err('mllr failed', exit=1)


    def run(self, cmdline, batchargs=None, output=False):
        if self.args.verbose:
            cmd_out = sys.stderr
            self.log('run: {0}'.format(' '.join(cmdline)))
        else:
            cmd_out = open(os.devnull, 'w')

        if batchargs is None or self.cores == 1:
            cmds = (cmdline,)
        else:
            cmds = [tuple(cmdline) + tuple(batchargs(i, self.cores))
                    for i in range(1, self.cores+1)]

        procs = []

        for cmd in cmds:
            try:
                p = subprocess.Popen(cmd,
                                     stdout=subprocess.PIPE if output else cmd_out,
                                     stderr=cmd_out)
            except Exception as e:
                err('command "{0}" failed: {1}'.format(' '.join(cmd), e), exit=1)

            procs.append(p)

        if output:
            fds = [p.stdout.fileno() for p in procs]
            fdmap = dict((fd, i) for i, fd in enumerate(fds))
            outputs = [bytearray() for i in range(len(cmds))]
            while fds:
                readable = select.select(fds, (), ())[0]
                for fd in readable:
                    out = os.read(fd, 4096)
                    if len(out) == 0:
                        fds.remove(fd)
                    else:
                        outputs[fdmap[fd]].extend(out)

        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0:
                err('command "{0}" failed: non-zero return code: {1}'.format(
                        ' '.join(cmds[i]), p.returncode), exit=1)

        if output:
            return [bytes(out) for out in outputs]


    def log(self, msg):
        if not self.args.quiet:
            sys.stderr.write('%s: %s\n' % (basename(sys.argv[0]), msg))


def transcript2phn(input, workdir, expand=True):
    """Convert a transcription to a list of phonemes."""

    # Read in, split to trimmed paragraphs
    input = filter(None, (re.sub(r'\s+', ' ', para.strip()) for para in input.split('\n\n')))

    # Attempt to expand abbreviations etc., if requested

    if expand:
        expander = join(rootdir, 'lavennin', 'bin', 'lavennin')
        exp_in = join(workdir, 'expander_in.txt')
        exp_out = join(workdir, 'expander_out.txt')

        with io.open(exp_in, 'w', encoding='iso-8859-15') as f:
            f.write('\n'.join(input) + '\n')

        if call([expander, workdir, exp_in, exp_out]) != 0:
            err('transcript expansion script failed', exit=1)

        with io.open(exp_out, 'r', encoding='iso-8859-15') as f:
            input = filter(None, (para.strip() for para in f.readlines()))

    # Go from utterances to list of words

    input = [re.sub(r'\s+', ' ',
                    re.sub('[^a-zåäö -]', '', para.lower())
                    ).strip().split()
             for para in input]

    # Add phoneme lists

    return [{ 'words': para, 'phns': list('_'.join(para).replace('-', '_')) } for para in input]


# "Free-format" transcript to phoneme list conversion
def text2phn(input, workdir, expand=True):
    """Convert a transcription to a list of phonemes."""

    # Read in, split to trimmed paragraphs
    

    if 'read' in dir(input):
        input = input.read()
    
    if type(input) is bytes:
        try: 
            input = input.decode('utf-8')
        except UnicodeError:
            input = input.decode('iso-8859-15')
  
    #if type(input) is not str:
    #    err('unable to understand input: {0}'.format(repr(input)), exit=1) 
    input = filter(None, (re.sub(r'\s+', ' ', para.strip()) for para in input.split('\n\n')))

    # Attempt to expand abbreviations etc., if requested
    if expand:
        expander = join(rootdir, 'lavennin', 'bin', 'lavennin')
        exp_in = join(workdir, 'expander_in.txt')
        exp_out = join(workdir, 'expander_out.txt')

        f = open(exp_in, 'w')
        out = '\n'.join(input) + '\n'
        f.write(out.encode('iso-8859-15'))
        f.close()

        if call([expander, workdir, exp_in, exp_out]) != 0:
            err('transcript expansion script failed', exit=1)

        f = open(exp_out,'r')
        input = filter(None, (para.strip().decode('iso-8859-15') for para in f.readlines()))
        f.close()

    # Go from utterances to list of words
    word_list = []
    snt_list  = []
    for para in input:
        para = re.sub(r'\s+', ' ',re.sub('[^a-zåäö -]', '', para.lower().encode('utf-8'))).strip()
        para = para.decode('utf-8')
        word_list = para.split()
        snt_list.append(word_list)
    
    # Add phoneme lists
    return [{ 'words': para, 'phns': list('_'.join(para).replace('-', '_')) } for para in snt_list]

# Fixup for discontinuity-caused unexpected states in alignment files

def alignment_fixup(infile, outfile):
    re_line = re.compile(r'^(\d+)( \d+ [^\.]+\.(\d+).*)$')

    #with open(infile, 'r', encoding='iso-8859-1') as fi, open(outfile, 'w', encoding='iso-8859-1') as fo:
    with io.open(infile, 'r', encoding='iso-8859-1') as fi:
        fo = io.open(outfile, 'w', encoding='iso-8859-1')
        pstate = -2
        pstart = None

        for line in fi:
            line = line.rstrip('\n')
            m = re_line.match(line)
            if m is None:
                err('invalid alignment line: %s' % line, exit=1)
            start, rest, state = m.group(1), m.group(2), int(m.group(3))

            if pstate == -2 or state == 0 or state == pstate+1:
                if pstart is None:
                    fo.write(line + '\n')
                else:
                    fo.write(pstart + rest + '\n')
                pstate, pstart = state, None
            else:
                pstate = -1
                if pstart is None: pstart = start

# SCLITE recognition transcript scoring

def sclite(out, rec, phones, workdir):
    reffile = join(workdir, 'sclite.ref')
    hypfile = join(workdir, 'sclite.hyp')

    hyptext = ' '.join(word for para in phones for word in para['words'])
    hyptext = hyptext.replace('-', ' ')

    with io.open(reffile, 'w', encoding='iso-8859-1') as f:
        f.write(hyptext + ' (spk-0)\n')
    with io.open(reffile+'.c', 'w', encoding='iso-8859-1') as f:
        f.write(hyptext.replace(' ', '_') + ' (spk-0)\n')

    rectext = rec.replace(' ', '')
    rectext = re.sub(r'</?[sw]>', ' ', rectext)
    rectext = re.sub(r'\s+', ' ', rectext).strip()

    with io.open(hypfile, 'w', encoding='iso-8859-1') as f:
        f.write(rectext + ' (spk-0)\n')
    with io.open(hypfile+'.c', 'w', encoding='iso-8859-1') as f:
        f.write(rectext.replace(' ', '_') + ' (spk-0)\n')

    for mode, suffix, flags in (('Letter', '.c', ['-c']), ('Word', '', [])):
        out.write('{0} error report:\n'.format(mode))

        cmd = [bin('sclite')]
        cmd.extend(flags)
        cmd.extend(['-r', reffile+suffix, '-h', hypfile+suffix,
                    '-s', '-i', 'rm', '-o', 'sum', 'stdout'])
        #scout = check_output(cmd).decode('ascii', errors='ignore')

        for line in scout.split('\n'):
            if line.find('SPKR') >= 0 or line.find('Sum/Avg') >= 0:
                out.write(line + '\n')

# TextGrid output formatting

def tg_write(tgfile, tiers, limits, srate):
    tgfile.write('''
File type = "ooTextFile"
Object class = "TextGrid"

xmin = {0:.3f}
xmax = {1:.3f}
tiers? <exists>
size = {2}
item []:
'''.format(limits[0] / srate, limits[1] / srate, len(tiers)).lstrip())

    #format(min(t['data'][0][0] for t in tiers) / srate,
    #       max(t['data'][-1][1] for t in tiers) / srate,
    #       len(tiers)).lstrip())

    for tierno, tier in enumerate(tiers):
        tdata = []

        at = limits[0]
        for start, end, label in tier['data']:
            if start > at:
                tdata.append((at, start, ''))
            tdata.append((start, end, label))
            at = end
        if at < limits[1]:
            tdata.append((at, limits[1], ''))

        tgfile.write('    item[{0}]:\n'.format(tierno + 1))
        tgfile.write('        class = "IntervalTier"\n')
        tgfile.write('        name = "{0}"\n'.format(tier['name']))
        tgfile.write('        xmin = {0:.3f}\n'.format(tdata[0][0] / srate))
        tgfile.write('        xmax = {0:.3f}\n'.format(tdata[-1][1] / srate))
        tgfile.write('        intervals: size = {0}\n'.format(len(tdata)))

        for num, (start, end, label) in enumerate(tdata):
            tgfile.write('        intervals [{0}]:\n'.format(num + 1))
            tgfile.write('            xmin = {0:.3f}\n'.format(start / srate))
            tgfile.write('            xmax = {0:.3f}\n'.format(end / srate))
            tgfile.write('            text = "{0}"\n'.format(label.encode('utf-8')))

#Decoder functions
def runto(t,frame):
    while (frame <= 0 or t.frame() < frame):
        if (not t.run()):
            break

def recognize(t,start, end):
    st = os.times()
    t.reset(start)
    t.set_end(end)
    runto(t,0)
    et = os.times()
    duration = et[0] + et[1] - st[0] - st[1] # User + system time
    frames = t.frame() - start;
    #sys.stdout.write('DUR: %.2fs (Real-time factor: %.2f)\n' %(duration, duration * 125 / frames))

def morphs2words(hypo_morphs):
    hypo_morphs = hypo_morphs.replace("<s>","")
    hypo_morphs = hypo_morphs.replace("</s>","")
    morphs = hypo_morphs.strip().split("<w>")
    word_sentence = ""
    for m in morphs:
        m = m.replace(" ","")
        if len(m) > 0:
            word_sentence += m+" "
    word_sentence = word_sentence.strip()
    return word_sentence

def decode(rootdir,model,lexicon,ngram,lookahead_ngram,recipefile,lna_path,lm_scale,do_phoneseg,morphseg_file):
    decoder_output = ""
 
    hmms = model+".ph"
    dur = model+".dur"
    #lexicon = sys.argv[3]
    #ngram = sys.argv[4]
    #lookahead_ngram = sys.argv[5]
    #recipefile = sys.argv[6]
    #lna_path = sys.argv[7]
    lm_scale = int(lm_scale)
    global_beam = 250
    do_phoneseg = int(do_phoneseg)
    #morphseg_file = sys.argv[10]

    num_batches = 1
    batch_index = 1
    
    ##################################################
    #Check if LM is morph model
    lex_file = open(lexicon,"r")
    lex_line = lex_file.readline()
    lex_file.close()
    
    morph_model = False
    if lex_line.strip() == "<w>(1.0) __":
        morph_model = True
    else:
        morph_model = False
    
    
    ##################################################
    # Load the recipe
    #
    f=open(recipefile,'r')
    recipelines = f.readlines()
    f.close()
    
    # Extract the lna files
    
    lnafiles=[]
    for line in recipelines:
        result = re.search(r"lna=(\S+)", line)
        if result:
            lnafiles = lnafiles + [result.expand(r"\1")]
    
    # Check LNA path
    if lna_path[-1] != '/':
        lna_path = lna_path + '/'
    
    ##################################################
    # Recognize
    #
    if morph_model:
        sys.stderr.write("loading morph models\n")
        t = Decoder.Toolbox(hmms, dur)
        t.set_optional_short_silence(1)
        t.set_cross_word_triphones(1)
        t.set_require_sentence_end(1)
        t.set_verbose(1)
        t.set_ignore_case(0)
        #t.set_print_text_result(1)
        t.set_print_state_segmentation(do_phoneseg)
        t.set_lm_lookahead(1)
        word_end_beam = int(2*global_beam/3);
        trans_scale = 1
        dur_scale = 3
        t.set_lm_scale(lm_scale)
        t.set_word_boundary("<w>")
        sys.stderr.write("loading lexicon\n")
        try:
            t.lex_read(lexicon)
        except:
            print("phone:", t.lex_phone())
            sys.exit(-1)
        t.set_sentence_boundary("<s>", "</s>")
        sys.stderr.write("loading ngram\n")
        t.ngram_read(ngram, 1)
        t.read_lookahead_ngram(lookahead_ngram)
    
        t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth
        t.set_global_beam(global_beam)
        t.set_word_end_beam(word_end_beam)
        t.set_token_limit(30000)
        t.set_prune_similar(3)
        #t.set_print_probs(0)
        #t.set_print_indices(0)
        #t.set_print_frames(0)
        t.set_duration_scale(dur_scale)
        t.set_transition_scale(trans_scale)
    else:
        sys.stderr.write("loading word models\n")
        t = Decoder.Toolbox(hmms, dur)
        t.set_optional_short_silence(1)
        t.set_cross_word_triphones(1)
        t.set_require_sentence_end(1)
        t.set_silence_is_word(0)
        t.set_verbose(1)
        #t.set_print_text_result(1)
        #t.set_print_word_start_frame(1)
        #t.set_print_state_segmentation(1)
        t.set_print_state_segmentation(do_phoneseg)
        t.set_lm_lookahead(1)
        #t.set_word_boundary("<w>")
        sys.stderr.write("loading lexicon\n")
        try:
            t.lex_read(lexicon)
        except:
            print("phone:", t.lex_phone())
            sys.exit(-1)
        t.set_sentence_boundary("<s>", "</s>")
        sys.stderr.write("loading ngram\n")
        t.ngram_read(ngram, 1)
        t.read_lookahead_ngram(lookahead_ngram)
        t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth
        word_end_beam = int(2*global_beam/3);
        trans_scale = 1
        dur_scale = 3
        t.set_global_beam(global_beam)
        t.set_word_end_beam(word_end_beam)
        t.set_token_limit(30000)
        t.set_prune_similar(3)
        #t.set_print_probs(0)
        #t.set_print_indices(0)
        #t.set_print_frames(0)
        t.set_duration_scale(dur_scale)
        t.set_transition_scale(trans_scale)
        t.set_lm_scale(lm_scale)
        
    
    decoder_output += "BEAM: "+str(global_beam)+"\n"
    decoder_output += "WORD_END_BEAM: "+str(word_end_beam)+"\n"
    decoder_output += "LMSCALE: "+str(lm_scale)+"\n"
    decoder_output += "WORD_END_BEAM: "+str(word_end_beam)+"\n"
    decoder_output += "DURSCALE: "+str(dur_scale)+"\n"
    
    if morphseg_file:
        t.set_generate_word_graph(1)
    #sys.stderr.write("Start recognition\n")
    for idx, lnafile in enumerate(lnafiles):
        if idx % num_batches != batch_index - 1:
            continue # recognized by a different instance
    
        t.lna_open(lna_path + lnafile, 1024)
        #t.lna_open(lnafile, 1024)
    
        decoder_output += "LNA:"+lnafile+"\n"
        decoder_output += "REC: "
        recognize(t,0,-1)
       
        best_hypo = t.best_hypo_string(False,False)
        state_segmentation_filename = lna_path+lnafile+".state.seg"
        t.write_state_segmentation(state_segmentation_filename)
        state_seg_file = open(state_segmentation_filename,"r")
        state_segs = ""
        for line in state_seg_file:
            state_segs += line.strip()+"\n"
        state_seg_file.close()
        os.remove(state_segmentation_filename)
        best_hypo = best_hypo.replace("*","")
        decoder_output += best_hypo+"\n"+state_segs+"\n"
        #hypo_sentence = morphs2words(best_hypo)
        #print "REC: "+hypo_sentence
        if morphseg_file:
            t.write_word_history('%s-%d' % (morphseg_file, idx))
    decoder_output = decoder_output.decode('iso-8859-15')
    return decoder_output

def decode_func(recipeline,model,lexicon,ngram,lookahead_ngram,lna_path,lm_scale,do_phoneseg,morphseg_file):
    decoder_output = ""
    hmms = model+".ph"
    dur = model+".dur"
    #lexicon = sys.argv[3]
    #ngram = sys.argv[4]
    #lookahead_ngram = sys.argv[5]
    #recipefile = sys.argv[6]
    #lna_path = sys.argv[7]
    lm_scale = int(lm_scale)
    global_beam = 250
    do_phoneseg = int(do_phoneseg)
    #morphseg_file = sys.argv[10]

    num_batches = 1
    batch_index = 1

    
    
    ##################################################
    #Check if LM is morph model
    lex_file = open(lexicon,"r")
    lex_line = lex_file.readline()
    lex_file.close()
    
    morph_model = False
    if lex_line.strip() == "<w>(1.0) __":
        morph_model = True
    else:
        morph_model = False
    
    

    # Extract the lna files
    recipeline = recipeline.strip()
    lnafile = recipeline.replace("lna=","")
    lnafiles=[lnafile]
    
    ext,file_fidx = lnafile.rsplit("-",1)
    file_fidx = file_fidx.replace(".lna","")
    # Check LNA path
    if lna_path[-1] != '/':
        lna_path = lna_path + '/'
    
    ##################################################
    # Recognize
    #
    if morph_model:
        t = Decoder.Toolbox(hmms, dur)
        t.set_optional_short_silence(1)
        t.set_cross_word_triphones(1)
        t.set_require_sentence_end(1)
        t.set_verbose(1)
        t.set_ignore_case(0)
        #t.set_print_text_result(1)
        t.set_keep_state_segmentation(do_phoneseg)
        t.set_lm_lookahead(1)
        word_end_beam = int(2*global_beam/3);
        trans_scale = 1
        dur_scale = 3
        t.set_lm_scale(lm_scale)
        t.set_word_boundary("<w>")
        try:
            t.lex_read(lexicon)
        except:
            print("phone:", t.lex_phone())
            sys.exit(-1)
        t.set_sentence_boundary("<s>", "</s>")
        t.ngram_read(ngram, 1)
        t.read_lookahead_ngram(lookahead_ngram)
    
        t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth
        t.set_global_beam(global_beam)
        t.set_word_end_beam(word_end_beam)
        t.set_token_limit(30000)
        t.set_prune_similar(3)
        #t.set_print_probs(0)
        #t.set_print_indices(0)
        #t.set_print_frames(0)
        t.set_duration_scale(dur_scale)
        t.set_transition_scale(trans_scale)
    else:
        t = Decoder.Toolbox(hmms, dur)
        t.set_optional_short_silence(1)
        t.set_cross_word_triphones(1)
        t.set_require_sentence_end(1)
        t.set_silence_is_word(0)
        t.set_verbose(1)
        #t.set_print_text_result(1)
        #t.set_print_word_start_frame(1)
        #t.set_print_state_segmentation(1)
        t.set_keep_state_segmentation(do_phoneseg)
        t.set_lm_lookahead(1)
        #t.set_word_boundary("<w>")
        try:
            t.lex_read(lexicon)
        except:
            print("phone:", t.lex_phone())
            sys.exit(-1)
        t.set_sentence_boundary("<s>", "</s>")
        t.ngram_read(ngram, 1)
        t.read_lookahead_ngram(lookahead_ngram)
        t.prune_lm_lookahead_buffers(0, 4) # min_delta, max_depth
        word_end_beam = int(2*global_beam/3);
        trans_scale = 1
        dur_scale = 3
        t.set_global_beam(global_beam)
        t.set_word_end_beam(word_end_beam)
        t.set_token_limit(30000)
        t.set_prune_similar(3)
        #t.set_print_probs(0)
        #t.set_print_indices(0)
        #t.set_print_frames(0)
        t.set_duration_scale(dur_scale)
        t.set_transition_scale(trans_scale)
        t.set_lm_scale(lm_scale)
        
    
    decoder_output += "BEAM: "+str(global_beam)+"\n"
    decoder_output += "WORD_END_BEAM: "+str(word_end_beam)+"\n"
    decoder_output += "LMSCALE: "+str(lm_scale)+"\n"
    decoder_output += "WORD_END_BEAM: "+str(word_end_beam)+"\n"
    decoder_output += "DURSCALE: "+str(dur_scale)+"\n"
    
    if morphseg_file:
        t.set_generate_word_graph(1)
    #sys.stderr.write("Start recognition\n")
    for idx, lnafile in enumerate(lnafiles):
        if idx % num_batches != batch_index - 1:
            continue # recognized by a different instance
        t.lna_open(lna_path + lnafile, 1024)
        decoder_output += "LNA:"+lnafile+"\n"
        decoder_output += "REC: "
        recognize(t,0,-1)
       
        best_hypo = t.best_hypo_string(False,False)
        state_segmentation_filename = lna_path+lnafile+".state.seg"
        t.write_state_segmentation(state_segmentation_filename)
        state_seg_file = open(state_segmentation_filename,"r")
        state_segs = ""
        for line in state_seg_file:
            state_segs += line.strip()+"\n"
        state_seg_file.close()
        os.remove(state_segmentation_filename)
        best_hypo = best_hypo.replace("*","")
        decoder_output += best_hypo+"\n"+state_segs+"\n"
        if morphseg_file:
            t.write_word_history('%s-%s' % (morphseg_file, file_fidx))
    decoder_output = decoder_output.decode('iso-8859-15')
    return decoder_output


# Miscellaneous helpers
def err(msg, exit=-1):
    sys.stderr.write('%s: error: %s\n' % (basename(sys.argv[0]), msg))
    if exit >= 0: sys.exit(exit)

def intersperse(iterable, delim):
    """Haskell intersperse: return elements of iterable interspersed with delim."""

    it = iter(iterable)
    yield next(it)
    for x in it:
        yield delim
        yield x

def trip2ph(triphone):
    if len(triphone) == 5: return triphone[2]
    elif triphone == '__': return '__'
    elif triphone[0] == '_': return '_'

    err('unknown triphone: %s' % triphone, exit=1)

def get_labels(phfile):
    re_index = re.compile(r'^\d+ (\d+) (.*)')

    labels = {}

    with io.open(phfile, 'r', encoding='iso-8859-1') as ph:
        if ph.readline() != 'PHONE\n': err('bad phoneme file: wrong header', exit=1)

        phcount = int(ph.readline())

        while True:
            line = ph.readline().rstrip()
            if not line: break
            line = line

            m = re_index.match(line)
            if m is None:
                err('bad phoneme file: wrong index line: %s' % line, exit=1)

            phnstates = int(m.group(1))
            phname = m.group(2)

            line = ph.readline().strip()
            phstates = [int(s) for s in line.split()]
            if len(phstates) != phnstates:
                err('bad phoneme file: wrong number of states', exit=1)

            s = 0
            for idx in phstates:
                ph.readline() # skip the transition probs
                if idx < 0: continue # dummy states don't need labels
                labels[idx] = '%s.%d' % (phname, s)
                s += 1

    return labels

def split_audio(seglen, infile, basepath, model):
    """Split an input audio file to approximately seglen-second segments,
    at more or less silent positions if possible.  Frame size used when
    splitting will match the frame size of the model, and the output list
    gives start offsets of the segments in terms of that.
    """

    # compute target segment length in frames

    srate = model['srate']
    framesize = model['fstep']
    segframes = int(seglen * srate / framesize)
    max_offset = segframes / 5

    # generate frame energy mapping for the input audio
     
    raw_filename = infile+".raw"
    cmd = [bin('sox'), infile, '-t', 'raw', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1',raw_filename]
    subprocess.call(cmd)
    raw_file = open(raw_filename,"rb")
    raw_energy = []
    while True:
        frame = raw_file.read(2*framesize)
        if len(frame) < 2*framesize:
            break
        frame = struct.unpack('={0}h'.format(framesize), frame)
        mean = float(sum(frame)) / len(frame)
        raw_energy.append(math.sqrt(sum((s-mean)**2 for s in frame)))
    raw_file.close()
    os.remove(raw_filename)
    '''
    with subprocess.Popen([bin('sox'), infile, '-t', 'raw', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1','-'],stdin=PIPE,stdout=PIPE) as sox:
        while True:
            frame = sox.stdout.read(2*framesize)
            if len(frame) < 2*framesize:
                print "complete"
                break
            frame = struct.unpack('={0}h'.format(framesize), frame)

            mean = float(sum(frame)) / len(frame)
            raw_energy.append(math.sqrt(sum((s-mean)**2 for s in frame)))
    '''

    if not raw_energy:
        err("input conversion of '{0}' with sox resulted in no frames".format(infile), exit=1)

    # moving-average smoothing for the energy
    energy = [0.0]*len(raw_energy)

    for i in range(len(energy)):
        wnd = raw_energy[max(i-10,0):i+11]
        energy[i] = sum(wnd)/len(wnd)

    # determine splitting positions

    segments = []
    at = 0

    while at < len(energy):
        left = len(energy) - at

        if left <= 1.5 * segframes:
            take = left
        else:
            target = at + segframes
            minpos = max(0, int(target - max_offset))
            maxpos = min(len(energy), int(target + max_offset + 1))
            pos = minpos + min(enumerate(energy[minpos:maxpos]),
                               key=lambda v: (1+abs(minpos+v[0]-target)/max_offset)*v[1])[0]
            take = pos - at

        segments.append((at, at+take))
        at += take
    # generate the resulting audio files

    audiofiles = []

    for i, (start, end) in enumerate(segments):
        starts = start*framesize
        lens = (end-start)*framesize
  
        start_seconds = float(1.0*starts/srate)
        dur_seconds = float(1.0*lens/srate)

        audiofile = '{0}-{1}.wav'.format(basepath, i)

        if call([bin('sox'), infile,
                 '-t', 'wav', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1',
                 audiofile,
                 'trim', str(start_seconds), str(dur_seconds)]) != 0:
            err("input conversion of '{0}' (frames {1}-{2}) with sox failed".format(infile, start, end), exit=1)
        #if call([bin('sox'), infile,
        #         '-t', 'wav', '-r', str(srate), '-b', '16', '-e', 'signed-integer', '-c', '1',
        #         audiofile,
        #         'trim', str(starts)+'s', str(lens)+'s']) != 0:
        #    err("input conversion of '{0}' (frames {1}-{2}) with sox failed".format(infile, start, end), exit=1)

        audiofiles.append({ 'start': start, 'file': audiofile })
    
    return audiofiles

#return audio length in seconds
def audio_file_len(audio_filename):
    ms = 0
    audiofile = open(audio_filename,"r")
    audiofile.seek(28)
    a=audiofile.read(4)

    #convert string a into integer/longint value
    #a is little endian, so proper conversion is required
    byteRate=0
    for i in range(4):
        byteRate=byteRate + ord(a[i])*pow(256,i)
    #get the file size in bytes
    fileSize=os.path.getsize(filename)  
    #the duration of the data, in milliseconds, is given by
    if (byteRate+ms) > 0:
        try:
            ms=((fileSize-44)*1000)/byteRate+ms
        except:
            pass
    audiofile.close()

    audio_file_len = float(ms*1.0/1000.0)
    return audio_file_len


def write_elan(media_file,rfile,outf):
    """Write Elan file"""
    ts_count = 1
    an_count = 1
    NS = 'http://www.w3.org/2001/XMLSchema-instance'
    location_attr = '{%s}noNamespaceSchemaLocation' % NS
    doc = etree.Element('ANNOTATION_DOCUMENT',
                        attrib={location_attr: 'http://www.mpi.nl/tools/elan/EAFv2.7.xsd',
                                'AUTHOR': '', 'DATE': dateIso(),
                                'FORMAT': '2.7', 'VERSION': '2.7'})
    header = etree.SubElement(doc, 'HEADER',
                              attrib={'MEDIA_FILE': '',
                                      'TIME_UNITS': 'milliseconds'})
    etree.SubElement(header, 'MEDIA_DESCRIPTOR',
                     attrib={'MEDIA_URL': media_file,
                             'MIME_TYPE': guess_type(media_file)[0],
                             'RELATIVE_MEDIA_URL': ''})
    t = etree.SubElement(header, 'PROPERTY',
                         attrib={'NAME': 'lastUsedAnnotationId'})
    t.text = str(len(rfile))
    time = etree.SubElement(doc, 'TIME_ORDER')
    for line in rfile:
        start,end,token,speaker_id = line.split("\t",3)
        etree.SubElement(time, 'TIME_SLOT',
                         attrib={'TIME_SLOT_ID': 'ts' + str(ts_count),
                                 'TIME_VALUE': start})
        ts_count += 1
        etree.SubElement(time, 'TIME_SLOT',
                         attrib={'TIME_SLOT_ID': 'ts' + str(ts_count),
                                 'TIME_VALUE': end})
        ts_count += 1

    #tier = etree.SubElement(doc, 'TIER',attrib={'DEFAULT_LOCALE': 'fi','LINGUISTIC_TYPE_REF': 'default-lt','TIER_ID': 'Speakers'})
    ts_count = 1
    index = 0
    current_speaker = ""
    seg_count = 1
    speakers = []
    for line in rfile:
        start,end,token,speaker_id = line.split("\t",3)
        if speaker_id != current_speaker:
            #speaker_utf8 = speaker_id.decode('iso-8859-15')
            speaker_utf8 = speaker_id
            speaker_utf8 = speaker_utf8.replace(":","")
            tier_id = speaker_utf8.strip()+" "+str(seg_count)
            #tier_id = speaker_utf8.strip()
            speakers.append(speaker_utf8.strip())
            if speaker_utf8.strip() == "Puhuja":
                speaker_utf8 = speakers[len(speakers)-3]
                tier_id = speaker_utf8.strip()+" "+str(seg_count)    
                #tier_id = speaker_utf8.strip()  
            #tier = etree.SubElement(doc, 'TIER',attrib={'DEFAULT_LOCALE': 'fi','LINGUISTIC_TYPE_REF': 'default-lt','TIER_ID': unicode(tier_id),'PARTICIPANT':unicode(speaker_utf8)})
            tier = etree.SubElement(doc, 'TIER',attrib={'DEFAULT_LOCALE': 'fi','LINGUISTIC_TYPE_REF': 'default-lt','TIER_ID': tier_id,'PARTICIPANT':speaker_utf8})
            current_speaker = speaker_id
            seg_count += 1
             		            
        a = etree.SubElement(tier, 'ANNOTATION')
        aa = etree.SubElement(a, 'ALIGNABLE_ANNOTATION',
                         attrib={'ANNOTATION_ID': 'a' + str(an_count),
                                 'TIME_SLOT_REF1': 'ts' + str(ts_count),
                                 'TIME_SLOT_REF2': 'ts' + str(ts_count + 1)})
        #token = token.decode('utf-8')
        av = etree.SubElement(aa,'ANNOTATION_VALUE')
        av.text = token
        an_count += 1
        ts_count += 2
        index += 1
    etree.SubElement(doc, 'LINGUISTIC_TYPE',
                     attrib={'GRAPHIC_REFERENCES': 'false',
                             'LINGUISTIC_TYPE_ID': 'default-lt',
                             'TIME_ALIGNABLE': 'true'})
    etree.SubElement(doc, 'LOCALE',
                     attrib={'COUNTRY_CODE': 'FI',
                             'LANGUAGE_CODE': 'fi'})
    tree = etree.ElementTree(doc)
    tree.write(outf, pretty_print=True,encoding="utf-8")

def dateIso():
    """ Returns the actual date in the format expected by ELAN. Source:
        http://stackoverflow.com/questions/3401428/how-to-get-an-isoformat-datetime-string-including-the-default-timezone"""
    dtnow = datetime.now()
    dtutcnow = datetime.utcnow()
    delta = dtnow - dtutcnow
    hh, mm = divmod((delta.days * 24 * 60 * 60 + delta.seconds + 30) // 60, 60)
    return '%s%+02d:%02d' % (dtnow.isoformat(), hh, mm)
