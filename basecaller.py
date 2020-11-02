import sys
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np

from bonito_io import DecoderWriterPool, gen_read
from util import decode, stitch, load_model, create_parallel_loader, get_data_processor_name


def main(args):
    reads_directory = args.reads_directory
    beamsize = args.beamsize
    fastq = args.fastq
    read_multiprocess = args.read_multiprocess
    model_directory = args.model_directory
    decode_multiprocess = args.decode_multiprocess
    output = args.output
    queue_size = args.queue_size
    run_device = args.device
    reference = args.reference

    sys.stderr.write("> loading model\n")
    model = load_model(model_directory, run_device)

    writer = DecoderWriterPool(decode,
                               procs=decode_multiprocess,
                               beamsize=beamsize,
                               fastq=fastq,
                               output=output,
                               reference=reference)
    reader = gen_read(reads_directory)
    sys.stderr.write("> calling\n")
    
    batch_data = torch.from_numpy(np.zeros((args.batch_size, 1, 1, args.chunksize)))
    # count of batch_data, from 0 to batch_size
    batch_count = 0
    # [read_id, num_chunks, is_last_batch]
    batch_list = []
    # previous infer output
    concate_log_prob = None
    # total sequence, total length of every raw data
    samples = 0
    # reads count
    reads_count = 0

    start_time = time.time()
    with writer:

        def infer(data):
            nonlocal concate_log_prob

            log_probs = model[0].run(model[2], feed_dict={model[1]: data})
            log_probs = np.transpose(log_probs, axes=(0, 2, 1))

            start = 0

            for read, batch, flag in batch_list:
                if flag:  # the last batch, to be concatenate with the start of next batch_data
                    if concate_log_prob is not None:
                        concate_log_prob = np.concatenate([concate_log_prob, log_probs[start:start+batch]])
                    else:
                        concate_log_prob = log_probs[start:start+batch]
                    continue
                # concatenate with the last batch of the previous batch_data
                if concate_log_prob is not None:
                    log = np.concatenate([concate_log_prob, log_probs[start:start+batch]])
                    log_prob = stitch(log, args.overlap // 3 // 2)
                    concate_log_prob = None
                # whole read in a batch_data
                else:
                    log_prob = stitch(log_probs[start:start+batch], args.overlap // 3 // 2)

                writer.queue.put((read, log_prob))
                start += batch
        
        with create_parallel_loader(read_multiprocess, reader, queue_size, get_data_processor_name(args)) as parallel_loader:
            while parallel_loader.has_next():
                # data: one read, n * 1 * 1 * chunksize
                read = parallel_loader.next()[0]
                data = read.signal
                read.signal = None

                reads_count += 1
                samples += data.shape[0] * data.shape[-1]

                # the current batch is full
                while data.shape[0] + batch_count > args.batch_size:
                    # full the current batch with some of the current data
                    # set the batch_list flag to True
                    if batch_count != args.batch_size:
                        last_batch = args.batch_size - batch_count
                        batch_data[batch_count:, :, :, :data[:last_batch].shape[-1]] = data[:last_batch]
                        batch_list.append((read, last_batch, True))
                        data = data[last_batch:]

                    # infer the current full batch_data
                    infer(batch_data)
                    batch_count = 0
                    batch_list = []
                    batch_data = torch.from_numpy(np.zeros((args.batch_size, 1, 1, args.chunksize)))

                batch_data[batch_count: batch_count + data.shape[0], :, :, :data.shape[-1]] = data
                batch_count += data.shape[0]
                # the current read end
                batch_list.append((read, data.shape[0], False))

            sys.stderr.write("> wait reader\n")
        if batch_count > 0:
            infer(batch_data)
        sys.stderr.write("> wait write\n")

    duration = time.time() - start_time
    sys.stderr.write("> completed reads: %s\n" % reads_count)
    sys.stderr.write("> samples per second %.1E\n" % (samples / duration))
    sys.stderr.write("> time: {:.4f}s\n".format(duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("--model_directory", default="./models/batchsize200_chunksize6000.pb")
    parser.add_argument("--reads_directory", default="", help="fast5 directory")
    parser.add_argument("--beamsize", default=5, type=int)
    parser.add_argument("--fastq", action="store_true", default=False)
    parser.add_argument("--chunksize", default=6000, type=int)
    parser.add_argument("--overlap", default=300, type=int)
    parser.add_argument("--batch_size", default=200, type=int)
    parser.add_argument("--read_multiprocess", default=30, type=int, help="num processing for processing data")
    parser.add_argument("--decode_multiprocess", default=10, type=int, help="num processing for decoding sequence")
    parser.add_argument("--output", default="", type=str, help="output file path")
    parser.add_argument("--queue_size", default=150, type=int, help="size of queue for processing data")
    parser.add_argument("--device", default="cuda", type=str, help="inference device")
    parser.add_argument("--reference", default="", type=str, help="reference file")
    return parser


if __name__ == '__main__':
    parser = argparser()
    args, _ = parser.parse_known_args()
    print(args)

    main(args)
