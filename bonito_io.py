"""
Basecaller Input/Output
"""

import os
from os.path import realpath, splitext
import sys
from glob import glob
from textwrap import wrap
from logging import getLogger
from multiprocessing import Process, Lock, cpu_count, Manager
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
from mappy import Aligner, revcomp

from util import mean_qscore_from_qstring

__all__ = ["gen_read", "DecoderWriterPool"]
__version__ = "0.2.2"

logger = getLogger('bonito')


class Read:

    def __init__(self, read, filename):
        self.read_id = read.read_id
        self.run_id = read.get_run_id().decode()
        self.filename = os.path.basename(read.filename)

        read_attrs = read.handle[read.raw_dataset_group_name].attrs
        channel_info = read.handle[read.global_key + 'channel_id'].attrs

        self.offset = int(channel_info['offset'])
        self.sampling_rate = channel_info['sampling_rate']
        self.scaling = channel_info['range'] / channel_info['digitisation']

        self.mux = read_attrs['start_mux']
        self.channel = channel_info['channel_number'].decode()
        self.start = read_attrs['start_time'] / self.sampling_rate
        self.duration = read_attrs['duration'] / self.sampling_rate

        # no trimming
        self.template_start = self.start
        self.template_duration = self.duration

        self.raw = read.handle[read.raw_dataset_name][:]


def gen_read(reads_directory):
    """get read generator

    Args:
        reads_directory: reads directory.

    Returns: [(instance of class `Read`)]

    """
    for fast5 in tqdm(glob("%s/*fast5" % reads_directory), ascii=True, ncols=100):
        f5_fh = get_fast5_file(fast5, "r")
        for read in f5_fh.get_reads():
            yield [(Read(read, fast5))]


class DecoderWriterPool:
    """
   Simple pool of decoder writers
   """

    def __init__(self, decode_func, procs=60, reference=None, output="", **kwargs):
        self.lock = Lock()
        self.queue = Manager().Queue()
        self.procs = procs if procs else max(1, cpu_count() - 10)
        self.decoders = []
        self.output = output
        self._check_output_dir()

        if reference:
            sys.stderr.write("> loading reference\n")
            aligner = Aligner(reference, preset='ont-map')
            if not aligner:
                sys.stderr.write("> failed to load/build index\n")
                sys.exit(1)
            write_sam_header(aligner)
        else:
            aligner = None

        with open(summary_file(self.output), 'w') as summary:
            write_summary_header(summary, alignment=aligner)

        for _ in range(self.procs):
            decoder = DecoderWriter(decode_func, self.queue, self.lock, aligner=aligner, output=self.output, **kwargs)
            decoder.start()
            self.decoders.append(decoder)

    def stop(self):
        for decoder in self.decoders:
            self.queue.put(None)
        for decoder in self.decoders:
            decoder.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _check_output_dir(self):
        if self.output:
            output_dir = Path(self.output).parent
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


class DecoderWriter(Process):
    """
    Decoder Process that writes outputs to file
    """

    def __init__(self, decode_func, queue, lock, fastq=False, beamsize=5, wrap=100, output="", aligner=None):
        super().__init__()
        self.queue = queue
        self.lock = lock
        self.decode_func = decode_func
        self.wrap = wrap
        self.fastq = fastq
        self.beamsize = beamsize
        self.output = output
        self.aligner = aligner

    def run(self):
        while True:
            job = self.queue.get()
            if job is None:
                return
            read, predictions = job

            if predictions.shape[0] == 5:
                predictions = np.transpose(predictions, axes=(1, 0)).astype(np.float32)
            else:
                predictions = predictions.astype(np.float32)

            sequence, path = self.decode_func(
                predictions, beamsize=self.beamsize, qscores=True, return_path=True
            )
            sequence, qstring = sequence[:len(path)], sequence[len(path):]
            mean_qscore = mean_qscore_from_qstring(qstring)

            if not self.fastq:  # beam search
                qstring = "*"
                sequence, path = self.decode_func(
                    predictions, beamsize=self.beamsize, qscores=self.fastq, return_path=True
                )

            if not self.aligner:
                mapping = False

            if sequence:
                with self.lock, open(summary_file(self.output), 'a') as summary:
                    if self.aligner:
                        for mapping in self.aligner.map(sequence):
                            write_sam(read.read_id, sequence, qstring, mapping)
                            break
                        else:
                            mapping = None
                            write_sam(read.read_id, sequence, qstring, mapping, unaligned=True)
                    if self.fastq:
                        write_fastq(read.read_id, sequence[:len(path)], sequence[len(path):], output=self.output)
                    else:
                        write_fasta(read.read_id, sequence, maxlen=self.wrap, output=self.output)
                    write_summary_row(summary, read, len(sequence), mean_qscore, alignment=mapping)
            else:
                logger.warning("> skipping empty sequence %s", read.read_id)


def write_fasta(header, sequence, maxlen=100, output=""):
    """Write a fasta record to a file.

    Args:
        header: sequence id.
        sequence: sequence.
        maxlen: max length of a single line.
        output: output file path.

    Returns:

    """
    if output:
        with open(output, "a") as f:
            f.write(">%s\n" % header)
            f.write("%s\n" % os.linesep.join(wrap(sequence, maxlen)))
    else:
        pass


def write_fastq(header, sequence, qstring, output=""):
    """Write a fastq record to a file.

    Args:
        header: sequence id.
        sequence: sequence.
        qstring:
        output: output file path.

    Returns:

    """
    if output:
        with open(output, "a") as f:
            f.write("@%s\n" % header)
            f.write("%s\n" % sequence)
            f.write("+\n")
            f.write("%s\n" % qstring)
    else:
        pass


def write_summary_header(fd, alignment=None, sep='\t'):
    """
    Write the summary tsv header.
    """
    fields = [
        'filename',
        'read_id',
        'run_id',
        'channel',
        'mux',
        'start_time',
        'duration',
        'template_start',
        'template_duration',
        'sequence_length_template',
        'mean_qscore_template',
    ]
    if alignment:
        fields.extend([
            'alignment_genome',
            'alignment_genome_start',
            'alignment_genome_end',
            'alignment_strand_start',
            'alignment_strand_end',
            'alignment_direction',
            'alignment_length',
            'alignment_num_aligned',
            'alignment_num_correct',
            'alignment_num_insertions',
            'alignment_num_deletions',
            'alignment_num_substitutions',
            'alignment_strand_coverage',
            'alignment_identity',
            'alignment_accuracy',
        ])
    fd.write('%s\n' % sep.join(fields))
    fd.flush()


def write_sam_header(aligner, fd=sys.stdout, sep='\t'):
    """
    Write the SQ & PG sam headers to a file descriptor.
    """
    fd.write('%s\n' % os.linesep.join([
        sep.join([
            '@SQ', 'SN:%s' % name, 'LN:%s' % len(aligner.seq(name))
        ]) for name in aligner.seq_names
     ]))

    fd.write('%s\n' % sep.join([
        '@PG',
        'ID:bonito',
        'PN:bonito',
        'VN:%s' % __version__,
        'CL:%s' % ' '.join(sys.argv),
    ]))
    fd.flush()


def summary_file(output_dir):
    """
    Return the filename to use for the summary tsv.
    """
    return '%s_summary.tsv' % splitext(realpath(output_dir))[0]


def write_summary_row(fd, read, seqlen, qscore, alignment=False, sep='\t'):
    """
    Write a summary tsv row.
    """
    fields = [str(field) for field in [
        read.filename,
        read.read_id,
        read.run_id,
        read.channel,
        read.mux,
        read.start,
        read.duration,
        read.template_start,
        read.template_duration,
        seqlen,
        qscore,
    ]]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([str(field) for field in [
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ]])

    elif alignment is None:
        fields.extend([str(field) for field in
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        ])

    fd.write('%s\n' % sep.join(fields))
    fd.flush()


def write_sam(read_id, sequence, qstring, mapping, fd=sys.stdout, unaligned=False, sep='\t'):
    """
    Write a sam record to a file descriptor.
    """
    if unaligned:
        fd.write("%s\n" % sep.join(map(str, [
            read_id, 4, '*', 0, 0, '*', '*', 0, 0, sequence, qstring, 'NM:i:0'
        ])))
    else:
        softclip = [
            '%sS' % mapping.q_st if mapping.q_st else '',
            mapping.cigar_str,
            '%sS' % (len(sequence) - mapping.q_en) if len(sequence) - mapping.q_en else ''
        ]
        fd.write("%s\n" % sep.join(map(str, [
            read_id,
            0 if mapping.strand == +1 else 16,
            mapping.ctg,
            mapping.r_st + 1,
            mapping.mapq,
            ''.join(softclip if mapping.strand == +1 else softclip[::-1]),
            '*', 0, 0,
            sequence if mapping.strand == +1 else revcomp(sequence),
            qstring,
            'NM:i:%s' % mapping.NM,
        ])))
    fd.flush()