"""
Basecaller utils
"""
from functools import partial

import tensorflow as tf
try:
    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    from npu_bridge.estimator import npu_ops
except:
    pass
import torch
import numpy as np
from fast_ctc_decode import beam_search, viterbi_search
from scipy.signal import find_peaks

from mp_parallel import create_parallel_loader, register_function_provider

__all__ = ["chunk", "decode", "stitch", "load_model", "mean_qscore_from_qstring",
           "create_parallel_loader", "register_function_provider", "get_data_processor_name"]


@register_function_provider(r'process_data_chunk(\d+)_overlap(\d+)')
def data_processor(match):
    return partial(pre_process_data, chunksize=int(match.group(1)), overlap=int(match.group(2)))


def get_data_processor_name(args):
    return f'process_data_chunk{int(args.chunksize)}_overlap{int(args.overlap)}'


def chunk(raw_data: torch.Tensor, chunksize: int, overlap: int) -> torch.Tensor:
    """Convert a read into overlapping chunks before calling

    Args:
        raw_data: (l, ) shape raw read data.
        chunksize: the length of chunk data
        overlap: overlap between two chunks, chunksize - overlap equals to step.

    Returns: (num_chunks, 1, 1, chunksize) shape torch tensor, num_chunks = l // step + 1

    """
    if 0 < chunksize < raw_data.shape[0]:
        num_chunks = raw_data.shape[0] // (chunksize - overlap) + 1
        tmp = torch.zeros(num_chunks * (chunksize - overlap)).type(raw_data.dtype)
        tmp[:raw_data.shape[0]] = raw_data
        return tmp.unfold(0, chunksize, chunksize - overlap).unsqueeze(1)
    return raw_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)


def decode(x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
    """decode the output of CNN to ACTG

    Args:
        x:
        beamsize:
        threshold:
        qscores:
        return_path:

    Returns:

    """
    alphabet = ["N", "A", "C", "G", "T"]
    if beamsize == 1 or qscores:
        qbias = 2.0
        qscale = 0.7
        seq, path = viterbi_search(x, alphabet, qscores, qscale, qbias)
    else:
        qbias = 0.0
        qscale = 1.0
        seq, path = beam_search(x, alphabet, beamsize, threshold)
    if return_path:
        return seq, path
    return seq


def stitch(predictions, overlap):
    """Stitch predictions together with a given overlap

    Args:
        predictions: [num_chunks, *]
        overlap: delete the head and tail of length `overlap`

    Returns:

    """
    if predictions.shape[0] == 1:
        return predictions.squeeze(0)
    stitched = [predictions[0, 0:-overlap]]
    for i in range(1, predictions.shape[0] - 1):
        stitched.append(predictions[i][overlap:-overlap])
    stitched.append(predictions[-1][overlap:])
    return np.concatenate(stitched)


def load_model(model_directory, device="npu"):
    """load .pb model

    Args:
        model_directory: model directory
        device: "npu" or "cuda"

    Returns: [session, input_node, output_node]

    """
    if "npu" in device.lower():
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
        custom_op.parameter_map["graph_run_mode"].i = 0
        g1 = tf.Graph()

        with g1.as_default():
            graph_def = tf.GraphDef.FromString(open(model_directory, 'rb').read())
            tf.import_graph_def(graph_def, name="")
            sess = tf.Session(graph=g1, config=config)
    elif "cuda" in device.lower():
        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            with open(model_directory, "rb") as f:
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            sess = tf.Session()

            sess.run(tf.global_variables_initializer())
    else:
        raise Exception('run_device can not be identified, should be "npu" or "cuda".')

    input_tensor = sess.graph.get_tensor_by_name("input:0")
    output_tensor = sess.graph.get_tensor_by_name('output:0')
    model = (sess, input_tensor, output_tensor)
    return model


def pre_process_data(read, chunksize, overlap):
    """process

    Args:
        read: instance of class `Read`, which contains summary information
        chunksize: the length of chunk data
        overlap: overlap between two chunks, chunksize - overlap equals to step.

    Returns: (read_id, chunk_data), chunk_data: n * 1 * 1 * chunksize

    """
    scaled = np.array(read.scaling * (read.raw + read.offset), dtype=np.float32)
    scaled = norm_by_noisiest_section(scaled)
    scaled = torch.tensor(scaled.astype(np.float16))
    scaled = chunk(scaled, chunksize, overlap)
    read.signal = scaled.unsqueeze(1)
    read.raw = None
    return read


def norm_by_noisiest_section(signal, samples=100, threshold=6.0):
    """Normalise using the medmad from the longest continuous region where the
        noise is above some threshold relative to the std of the full signal.
    """
    threshold = signal.std() / threshold
    noise = np.ones(signal.shape)

    for idx in np.arange(signal.shape[0] // samples):
        window = slice(idx * samples, (idx + 1) * samples)
        noise[window] = np.where(signal[window].std() > threshold, 1, 0)

    # start and end low for peak finding
    noise[0] = 0
    noise[-1] = 0
    peaks, info = find_peaks(noise, width=(None, None))

    if len(peaks):
        widest = np.argmax(info['widths'])
        med, mad = med_mad(signal[info['left_bases'][widest]: info['right_bases'][widest]])
    else:
        med, mad = med_mad(signal)
    res = (signal - med) / mad
    return res


def med_mad(x, factor=1.4826):
    """Calculate signal median and median absolute deviation

    Args:
        x: signal
        factor: 1.4826

    Returns: signal median and median absolute deviation

    """
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad


def mean_qscore_from_qstring(qstring):
    """Convert qstring into a mean qscore

    Args:
        qstring:

    Returns: a mean qscore

    """
    if len(qstring) == 0:
        return 0.0
    err_probs = [10**((ord(c) - 33) / -10) for c in qstring]
    mean_err = np.mean(err_probs)
    return -10 * np.log10(max(mean_err, 1e-4))
