import json
import os
import sys

import numpy as np
import tensorflow as tf


from gpt2.src import model
from gpt2.src import sample
from gpt2.src import encoder

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--config_path', type=str,
    default='config/default_config.json', help='model config json path'
)
parser.add_argument(
    '--save_path', type=str,
    default='data/', help='save directory'
)


'''
Interactively run the model
:model_name=124M : String, which model to use
:seed=None : Integer seed for random number generators, fix seed to reproduce
    results
:nsamples=1 : Number of samples to return total
:batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
:length=None : Number of tokens in generated text, if None (default), is
    determined by model hyperparameters
:temperature=1 : Float value controlling randomness in boltzmann
    distribution. Lower temperature results in less random completions. As the
    temperature approaches zero, the model will become deterministic and
    repetitive. Higher temperature results in more random completions.
:top_k=0 : Integer value controlling diversity. 1 means only 1 word is
    considered for each step (token), resulting in deterministic completions,
    while 40 means 40 words are considered at each step. 0 (default) is a
    special setting meaning no restrictions. 40 generally is a good value.
:models_dir : path to parent folder containing model subfolders
    (i.e. contains the <model_name> folder)
'''


class ConditionalGenerator:
    def __init__(self, config_path, save_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.model_name = config['model_name']
        self.seed = config['seed']
        self.nsamples = config['nsamples']
        self.batch_size = config['batch_size']
        self.length = config['length']
        self.temperature = config['temperature']
        self.top_k = config['top_k']
        self.models_dir = config['models_dir']
        self.save_path = save_path

        self.models_dir = os.path.expanduser(os.path.expandvars(
            self.models_dir
        ))
        if self.batch_size is None:
            self.batch_size = 1
        assert self.nsamples % self.batch_size == 0

        self.enc = encoder.get_encoder(self.model_name, self.models_dir)
        self.hparams = model.default_hparams()
        with open(os.path.join(
                self.models_dir, self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        if self.length is None:
            self.length = self.hparams.n_ctx // 2
        elif self.length > self.hparams.n_ctx:
            raise ValueError(('Can't get samples longer '
                             'than window size: {}').format(
                                self.hparams.n_ctx
                            ))

        self.sess = tf.Session(graph=tf.Graph())
        self.context = tf.placeholder(tf.int32, [self.batch_size, None])
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.output = sample.sample_sequence(
            hparams=self.hparams, length=self.length,
            context=self.context,
            batch_size=self.batch_size,
            temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(
            self.models_dir, self.model_name
        ))
        saver.restore(self.sess, ckpt)

    def save_text(self, text_list):
        file_path = input('Filename: ')
        while not save_path:
            print('Filename should not by empty')
            file_path = input('Filename: ')
        save_file = os.path.join(self.save_path, file_path)
        with open(save_file, 'w') as f:
            for text in text_list:
                text = text.replace('\n', ' ')
                f.write('{}\n'.format(text))
        print('generated data saved to {}'.format(save_file))

    def generate(self):
        while True:
            raw_text = input('Model prompt >>> ')
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input('Model prompt >>> ')
            context_tokens = self.enc.encode(raw_text)
            generated = 0
            text_list = []
            for _ in range(self.nsamples // self.batch_size):
                out = self.sess.run(self.output, feed_dict={
                    self.context: [context_tokens for _ in
                                   range(self.batch_size)]
                })[:, len(context_tokens):]
                for i in range(self.batch_size):
                    generated += 1
                    text = self.enc.decode(out[i])
                    text_list.append(text)
                    print('=' * 40 + ' SAMPLE ' + str(generated) + ' ' +
                          '=' * 40)
                    print(text)
            print('=' * 80)

            to_save = input('Save generated data [Y]/n ? ').strip()
            while True:
                if to_save in ['', 'y', 'Y']:
                    self.save_text(text_list)
                    break
                elif to_save in ['n', 'N']:
                    break
                else:
                    to_save = input('Save generated data? [Y]/n')


if __name__ == '__main__':
    args = parser.parse_args()
    generator = ConditionalGenerator(args.config_path, args.save_path)
    generator.generate()
