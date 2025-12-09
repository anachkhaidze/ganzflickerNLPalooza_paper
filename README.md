# From dots to faces: Individual differences in visual imagery capacity predict the content of flicker-induced hallucinations

This repository contains all the data, code, and materials required to reproduce the analyses presented in the above article. The preprint is available [here](https://arxiv.org/abs/2507.09011).

### Contents

This repository is organized to mirror the three main analytical components of the paper.

1. **Topic modeling (MOSAIC pipeline)**
   - Topic modeling script: `topic_modeling.ipynb`
   - Regression analyses: 'topic_lasso_modeling_plots_paper.ipynb'

2. **LLM/VLM pipeline**
   - To get description embeddings, run the main file with the models of your choice python main.py --model bert clip clap gpt2 siglip or python main.py --model all to run all models
   - RDM and stats: `rdm_stats.ipynb`

3. **Sensorimotor content analysis (Lancaster Norms)**
   - Text preprocessing and Lancaster score assignment: `ls_sensorimotor_analysis/hallucination_preprocess_assign_ls_norms.ipynb.ipynb`
   - Regression analyses for sensorimotor dimensions: `ls_regressions.ipynb`
   - Regression analyses for ASCs: `emotions_altered_states.ipynb`
   - Plotting: `ls_plots_sensorimotor_dimensions.ipynb`
