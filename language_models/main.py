# main.py
# from model_classes.process_clip import CLIP_Pipeline
# from model_classes.process_bert import BERT_Pipeline
# from model_classes.process_clap import CLAP_Pipeline
# from model_classes.process_siglip import SigLIP_Pipeline
# from model_classes.process_gpt2 import GPT2_Pipeline
# from model_classes.process_roberta import ROBERTA_Pipeline
# from model_classes.process_blip import BLIP_Pipeline
from model_classes.model_pipelines import BERT_Pipeline, CLIP_Pipeline, CLAP_Pipeline, SigLIP_Pipeline, GPT2_Pipeline, ROBERTA_Pipeline, BLIP_Pipeline
from utils import get_descriptions, get_scores, plot_rdm, display_images_grid, cosine_distance, euclidean_distance, pearson_distance
import argparse


def main():
    parser = argparse.ArgumentParser(description='Process hallucination data.')
    parser.add_argument("-m", "--models", help="Models to use (e.g., 'bert', 'clip', 'clap', 'siglip', 'gpt2', 'roberta', 'random_bert', 'all')", nargs='+', required=True)
    parser.add_argument("--distance", choices=["cosine", "euclidean", "pearson"], default="cosine", help="Distance metric for RDM (cosine, euclidean, or pearson; default: cosine)")
    parser.add_argument("--average", choices=['True', 'False'], default='True', help="If set, average embeddings by category. If not set, use all embeddings.")
    args = parser.parse_args()
    
    args.average = eval(args.average)
    
    if args.models == ['all']:
        args.models = ['bert', 'clip', 'clap', 'siglip', 'gpt2', 'roberta', 'random_bert', 'random_clip', 'random_gpt2', 'random_roberta', 'random_siglip', 'random_clap', 'blip', 'random_blip']
        
    print(f"Models to use: {args.models}")
    print(f"The average is {args.average}")

    hallucination_descriptions = get_descriptions()
    hallucination_scores = get_scores()
    print(f"Vividness Scores found: {list(set(hallucination_scores))}")

    print(f"Starting Analysis of {len(hallucination_descriptions)} Hallucination Descriptions")

    arg_to_class_map = {'bert': BERT_Pipeline, 
                        'clip': CLIP_Pipeline,
                        'clap': CLAP_Pipeline,
                        'siglip': SigLIP_Pipeline,
                        'gpt2': GPT2_Pipeline,
                        'roberta': ROBERTA_Pipeline,
                        'blip': BLIP_Pipeline}
    arg_to_formatted_name = {
                        'bert': 'BERT', 
                        'clip': 'CLIP',
                        'clap': 'CLAP',
                        'gpt2': 'GPT-2',
                        'siglip': 'SigLIP',
                        'roberta': 'RoBERTa',
                        'blip': 'BLIP',

                        'random_bert': 'Randomized BERT',
                        'random_clip': 'Randomized CLIP',
                        'random_clap': 'Randomized CLAP',
                        'random_gpt2': 'Randomized GPT-2',
                        'random_siglip': 'Randomized SigLIP',
                        'random_roberta': 'Randomized RoBERTa',
                        'random_blip': 'Randomized BLIP',

                        'bert_ppt': 'BERT Participant-Level', 
                        'clip_ppt': 'CLIP Participant-Level',
                        'clap_ppt': 'CLAP Participant-Level',
                        'gpt2_ppt': 'GPT-2 Participant-Level',
                        'siglip_ppt': 'SigLIP Participant-Level',
                        'roberta_ppt': 'RoBERTa Participant-Level',
                        'blip_ppt': 'BLIP Participant-Level',

                        # 'random_bert_ppt': 'Randomized BERT',
                        # 'random_clip_ppt': 'Randomized CLIP',
                        # 'random_clap_ppt': 'Randomized CLAP',
                        # 'random_gpt2_ppt': 'Randomized GPT-2',
                        # 'random_siglip_ppt': 'Randomized SigLIP',
                        # 'random_roberta_ppt': 'Randomized RoBERTa',
                        # 'random_blip_ppt': 'Randomized BLIP',
                        }
    str_distance_to_function = {
        "cosine": cosine_distance,
        "euclidean": euclidean_distance,
        "pearson": pearson_distance}

    if args.distance in list(str_distance_to_function.keys()):
        distance_fn = str_distance_to_function[args.distance]
    else:
        raise ValueError(f"Unknown distance metric: {args.distance}")

    for model_name in args.models:
        if "random" in model_name:
            random = True
            base_model_name = model_name.replace("random_", "")
        else:
            random = False
            base_model_name = model_name

        if args.average:
            pretrained_path = f'results/embeddings/{model_name}_output.npy'
        else:
            pretrained_path = f'results/embeddings/{model_name}_unaveraged_output.npy'

        pipeline_object = arg_to_class_map[base_model_name](hallucination_descriptions, hallucination_scores, pretrained_path, random, args.average)
        vectors = pipeline_object.run()

        if args.average:
            unique_scores = sorted(list(set(hallucination_scores)))
            labels_for_plot = [str(s) for s in unique_scores]
            plot_rdm(vectors, arg_to_formatted_name[model_name], labels=labels_for_plot, show=False, distance_fn=distance_fn)
        elif not args.average and 'random' not in model_name:
            model_name_ppt = f"{model_name}_ppt"
            all_scores_sorted = [str(s) for s in sorted(hallucination_scores)]
            plot_rdm(vectors, arg_to_formatted_name[model_name_ppt], labels=all_scores_sorted, show=False, distance_fn=distance_fn)

    if len(args.models) > 0:
        display_images_grid(distance_method=args.distance)

main()


