#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import numbers
import re
import six

import model, sample, encoder, hparam
from hparam import HParams
class ChatBot():
    """A class to help engineer simple chatbots with GPT2"""

    def __init__(self, 
    bot="AI", 
    user="Human",
    instructions="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.",
    examples=[("Hello, who are you?", "I am an AI assistant. How can I help you today?")],
    length=50,
    model_name='124M',
    temperature=.9,
    top_p=1,
    top_k=0,
    stop=True,
    batch_size=1,
    seed=None,
    models_dir='models',
    nsamples=1,):

        self.bot = bot + ": "
        self.user = user + ": "
        self.examples = examples
        self.nsamples = nsamples
        self.seed = seed
        self.batch_size = batch_size
        self.models_dir = models_dir
        self.top_k = top_k
        self.instructions = instructions
        self.model_name = model_name
        self.length = length
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

        if self.stop == True:
            self.stop = [self.user, self.bot]
        
        if self.stop == False:
            self.stop = []

        self.prompt = self.instructions + "\r\n\r\n"
        self.intro = self.instructions + "\r\n"
        
        if self.examples != None:
            self.intro = self.prompt + self.user + self.examples[0][0] + "\r\n" + self.bot + self.examples[0][1] + "\r\n\r\n"
            for i in self.examples:
                self.prompt += self.user + i[0] + "\r\n" + self.bot + i[1] + "\r\n\r\n"
        

    def proto_chat(self):
        """
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
        """
        model_name=self.model_name
        seed=self.seed
        nsamples=self.nsamples
        batch_size=self.batch_size
        length=self.length
        temperature=self.temperature
        top_k=self.top_k
        top_p=self.top_p
        models_dir=self.models_dir

        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:
            context = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size, None])
            np.random.seed(seed)
            tf.compat.v1.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k, top_p=top_p
            )

            saver = tf.compat.v1.train.Saver()
            ckpt = tf.compat.v1.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)
        
            print("\n\r***To QUIT, type 'QUIT' into the prompt field***\n\r")
            print(self.intro)
        
            while True:
                raw_text = input(self.user)
                print("\r\n")
                if raw_text.lower() == 'quit':
                    break
                while not raw_text:
                    print(self.bot, 'What would you like to talk about?')
                    raw_text = input(self.user)
                prompt = self.prompt + self.user + raw_text + " " + self.bot
                context_tokens = enc.encode(prompt)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text_ = enc.decode(out[i]).split(" ")
                        text = ""
                        for i in range(len(text_)):
                            # if text_[i] not in self.stop:
                            if text_[i].strip(" ").strip(":") not in ["AI", "Human"]:
                                text += text_[i] + " "
                            else:
                                break
                        print(self.bot, text + "\r\n")
                

if __name__ == '__main__':
    bot = ChatBot()
    bot.proto_chat()

    # bot = ChatBot(model_name='model-205')
    # Jane = ChatBot(bot="Jane",
    #            user="You",
    #            instructions="This is a conversation with Jane, a friendly and futuristic thinker.",
    #            examples=[("How different will the future be?", 
    #                       "Very different due to exponential growth in key areas.")],
    #            temperature=.95)
    # Jane.proto_chat()