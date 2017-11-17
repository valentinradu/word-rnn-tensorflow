from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
from tensorflow.python.lib.io import file_io

from word_rnn.utils import TextLoader
from word_rnn.model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=200,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                       help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--freeze_graph', dest='freeze_graph', action='store_true',
                       help='if true, freeze (replace variables with consts), prune (for inference) and save graph')

    args = parser.parse_args()
    sample(args)

def freeze_and_save_graph(sess, folder, out_nodes, as_text=False):
    ## save graph definition
    graph_raw = sess.graph_def
    graph_frz = tf.graph_util.convert_variables_to_constants(sess, graph_raw, out_nodes)
    ext = '.txt' if as_text else '.pb'
    tf.train.write_graph(graph_frz, folder, 'graph_frz'+ext, as_text=as_text)
    tf.train.write_graph(graph_raw, folder, 'graph_raw'+ext, as_text=as_text)

def sample(args):
    with file_io.FileIO(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with file_io.FileIO(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            if(args.freeze_graph):
                freeze_and_save_graph(sess, args.save_dir, ['data_out', 'state_out'], False)
            print(model.sample(sess, words, vocab, args.n, args.prime, args.sample, args.pick, args.width))

if __name__ == '__main__':
    main()
