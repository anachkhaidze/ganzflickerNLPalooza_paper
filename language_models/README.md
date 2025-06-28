# Ganzflicker-CLIP-and-GPT2-Pipeline

## How to run this code
1. Install any dependencies in requirements.txt

2. Run the main file with the models of your choice ```python main.py --model bert clip clap gpt2 siglip```
or ```python main.py --model all``` to run all models

You can specify the metric that we use to compare the embeddings with the --distance option. The options are cosine, euclidean, and pearson.

You can specify whether you want to also include the random models by include random_bert, random_clip, etc. These are also included in --model all

To get participant-level embeddings, set --average False. This is set to True by default.