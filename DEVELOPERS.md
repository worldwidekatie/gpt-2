# Installation

Git clone this repository, and `cd` into directory for remaining commands
```
git clone https://github.com/worldwidekatie/gpt.git && cd gpt-2
```

Then, follow instructions

## Native Installation

All steps can optionally be done in a virtual environment using tools such as `virtualenv` or `conda`.

Run the Virtual enviornment.
```
pipenv shell
```

If there's anything you don't have, install it.
```
pip3 install -r requirements.txt
```

Download the model data
```
python3 download_model.py 124M
python3 download_model.py 355M
python3 download_model.py 774M
python3 download_model.py 1558M
```

# Running

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

Some of the examples below may include Unicode text characters. Set the environment variable:
```
export PYTHONIOENCODING=UTF-8
```
to override the standard stream settings in UTF-8 mode.


## Fine-tuning

Put your fine-tuning data into the src folder as a .txt document, then run `encode.py` to generate a .npz version of your data.
```
python3 src/encode.py src/your_data.txt src/your_data.npz
```

Run train.py with that new dataset
```
python3 src/train.py --dataset src/your_data.npz
```

When loss and avg are acceptably low, save a checkpoint with ctrl-c
```
^C
```

Move it from the `checkpoints` folder into the `models` folder.

Add a copy of `encoder.json`, `hparams.json` , and `vocab.bpe` from the original model you fine-tuned from to your fine-tuned folder.

Run it like you would any of the GPT-2 models.

## Chatbots

To get started with an 'out-of-the-box' chatbot, just run `chatbot.py`.
```
python3 src/chatbot.py
```

To make changes, change the default parameters in the `chatbot.py` file and run it again. E.g.
```
if __name__ == '__main__':
    Jane = ChatBot(bot="Jane",
               user="You",
               instructions="This is a conversation with Jane, a friendly and futuristic thinker.",
               examples=[("How different will the future be?", 
                          "Very different due to exponential growth in key areas.")],
               temperature=.95)

    Jane.proto_chat()
```

**Pro Tip!** Try out the fine-tuning instructions above to make a model specific to chatbots.


## Conditional sample generation

To give the model custom prompts, you can use:
```
python3 src/interactive_conditional_samples.py --top_k 40
```

To check flag descriptions, use:
```
python3 src/interactive_conditional_samples.py -- --help
```

To quit, type quit into model prompt
```
Model prompt >>> quit
```

## Unconditional sample generation

**This is still pretty buggy with tensorflow 2.2 because I don't use it enough to have needed to fix them yet.**

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py | tee /tmp/samples
```
There are various flags for controlling the samples:
```
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples
```

To check flag descriptions, use:
```
python3 src/generate_unconditional_samples.py -- --help
```
