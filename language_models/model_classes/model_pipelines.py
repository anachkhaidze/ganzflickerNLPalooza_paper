import torch
import os
import numpy as np
from utils import dict_to_ordered_list
from dotenv import load_dotenv
import random
import math
from transformers import (
    BertTokenizer, BertModel,
    BlipProcessor, BlipModel,
    AutoProcessor, ClapModel,
    GPT2Tokenizer, GPT2Model,
    RobertaTokenizer, RobertaModel,
    AutoModel
)
import open_clip

# --- Base Class ---

class ModelPipeline:
    """
    Base class for model pipelines.
    """
    def __init__(self, description_list, category_list, pretrained_path=None, randomize=False, average_mode=True):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.descriptions = list(description_list)
        self.categories = list(category_list)
        self.randomize = randomize
        self.average_mode = average_mode
        self.pretrained_path = pretrained_path
        self.model_name = self.__class__.__name__.replace('_Pipeline', '')

        if self.randomize:
            # Shuffle descriptions and categories together to maintain pairing
            desc_cat_pairs = list(zip(self.descriptions, self.categories))
            random.shuffle(desc_cat_pairs)
            self.descriptions, self.categories = zip(*desc_cat_pairs)
            self.descriptions = list(self.descriptions)
            self.categories = list(self.categories)

        if pretrained_path and os.path.exists(pretrained_path):
            self.pretrained_output = np.load(pretrained_path)
            print(f"Loaded pretrained {self.model_name} output from {pretrained_path}")
        else:
            self.pretrained_output = None

        self.model = None
        self.tokenizer = None

    def get_embeddings(self, descriptions):
        """
        This method should be implemented by each child class.
        """
        raise NotImplementedError("This method should be implemented by each child class.")

    def batch_run(self, descriptions):
        """
        Processes a batch of descriptions to get their embeddings.
        """
        with torch.no_grad():
            return self.get_embeddings(descriptions)

    def run(self, batch_size=64):
        """
        Runs the pipeline to get embeddings for all descriptions.
        """
        # If a pre-computed file exists, load and return it immediately.
        if self.pretrained_output is not None:
            print(f"Using old {self.model_name} output from {self.pretrained_path}")
            return self.pretrained_output

        print(f"Processing {self.model_name} embeddings...")

        # --- Step 1: Generate all embeddings ---
        all_features = []
        for i in range(0, len(self.descriptions), batch_size):
            print(f"On batch {0 if i == 0 else math.ceil(i / batch_size)} / {math.ceil(len(self.descriptions) / batch_size)}")
            batch_descriptions = self.descriptions[i:i + batch_size]
            batch_features = self.batch_run(batch_descriptions)
            all_features.append(batch_features.cpu())
        all_features = torch.cat(all_features, dim=0)

        # --- Step 2: Process based on averaging mode ---
        results_dir = 'results/embeddings'
        os.makedirs(results_dir, exist_ok=True)

        # --- Path for Unaveraged Embeddings ---
        if not self.average_mode:
            # Sort features to match the original category order for plotting
            sorter = np.argsort(self.categories)
            sorted_features = all_features.numpy()[sorter]

            output_filename = f'{self.model_name.lower()}_unaveraged_output.npy'
            if self.randomize:
                output_filename = f'random_{self.model_name.lower()}_unaveraged_output.npy'
            output_path = os.path.join(results_dir, output_filename)
            np.save(output_path, sorted_features)
            print(f"{self.model_name} unaveraged output saved to {output_path}")
            return sorted_features

        # --- Path for Averaged Embeddings ---
        else:
            hidden_size = self.get_hidden_size()
            self.averages = {category: torch.zeros(hidden_size, device=self.device) for category in set(self.categories)}
            self.counts = {category: 0 for category in set(self.categories)}

            # Use the pre-computed 'all_features' tensor for averaging
            for feature, category in zip(all_features, self.categories):
                self.averages[category] += feature.to(self.device)
                self.counts[category] += 1
            
            for category in self.averages:
                if self.counts[category] > 0:
                    self.averages[category] /= self.counts[category]
                    # Apply model-specific post-processing (e.g., normalization)
                    self.averages[category] = self.post_process_average(self.averages[category])

            # FIX: Manually create the ordered list of tensors to avoid the bug in `dict_to_ordered_list`
            # which creates `None` for non-zero-indexed keys.
            ordered_averages_list = []
            for category_key in sorted(self.averages.keys()):
                ordered_averages_list.append(self.averages[category_key])

            final_vectors = np.array([tensor.cpu().numpy() for tensor in ordered_averages_list])
            
            output_filename = f'{self.model_name.lower()}_output.npy'
            if self.randomize:
                output_filename = f'random_{self.model_name.lower()}_output.npy'
            output_path = os.path.join(results_dir, output_filename)
            np.save(output_path, final_vectors)
            print(f"{self.model_name} averaged output saved to {output_path}")
            return final_vectors

    def post_process_average(self, average_vector):
        """
        Optional post-processing for the final averaged vector.
        Base implementation does nothing.
        """
        return average_vector

    def get_hidden_size(self):
        """
        Returns the hidden size of the model.
        This method may need to be overridden by child classes if the hidden size is not easily accessible.
        """
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'text_config'):
            return self.model.config.text_config.hidden_size
        elif hasattr(self.model, 'text_projection') and self.model.text_projection is not None:
            return self.model.text_projection.shape[0]
        else:
            return 768

# --- Child Classes ---

class BERT_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', token=hf_token)
        self.model = BertModel.from_pretrained('bert-base-uncased', token=hf_token)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("==========BERT model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        tokenized_text = self.tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized_text)
        return outputs.last_hidden_state.mean(dim=1)

class BLIP_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base', token=hf_token)
        self.model = BlipModel.from_pretrained('Salesforce/blip-image-captioning-base', token=hf_token)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("==========BLIP model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        inputs = self.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.text_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)
        
    def post_process_average(self, average_vector):
        return average_vector / average_vector.norm(dim=-1, keepdim=True)

class CLAP_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        model_id = "laion/clap-htsat-unfused"
        self.processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        self.model = ClapModel.from_pretrained(model_id, token=hf_token)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("==========CLAP model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        inputs = self.processor(
            text=descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model.get_text_features(**inputs)
        return outputs / outputs.norm(dim=-1, keepdim=True)

    def get_hidden_size(self):
        return 512 # CLAP's text embedding dimension
        
    def post_process_average(self, average_vector):
        return average_vector / average_vector.norm(dim=-1, keepdim=True)

class CLIP_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_name = 'ViT-B-32'
        pretrained = 'laion2b_s34b_b79k'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        print("==========CLIP model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        tokenized_text = self.tokenizer(descriptions).to(self.device)
        return self.model.encode_text(tokenized_text)
        
    def post_process_average(self, average_vector):
        return average_vector / average_vector.norm(dim=-1, keepdim=True)

class GPT2_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        model_id = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_id, token=hf_token)
        self.model = GPT2Model.from_pretrained(model_id, token=hf_token, output_hidden_states=True)
        self.model.eval()
        self.model = self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.layer_index = -1
        print("==========GPT-2 model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        inputs = self.tokenizer(
            descriptions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.layer_index]
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        masked_hidden_states = hidden_states * attention_mask
        sum_embeddings = torch.sum(masked_hidden_states, dim=1)
        seq_lengths = torch.sum(attention_mask, dim=1)
        text_features = sum_embeddings / seq_lengths
        return text_features / text_features.norm(dim=-1, keepdim=True)
        
    def post_process_average(self, average_vector):
        return average_vector / average_vector.norm(dim=-1, keepdim=True)

class ROBERTA_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', token=hf_token)
        self.model = RobertaModel.from_pretrained('roberta-base', token=hf_token)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("==========RoBERTa model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        tokenized_text = self.tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**tokenized_text)
        return outputs.last_hidden_state.mean(dim=1)

class SigLIP_Pipeline(ModelPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.model_id = "google/siglip-base-patch16-224"
        self.model = AutoModel.from_pretrained(self.model_id, token=hf_token)
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)
        self.model.eval()
        self.model = self.model.to(self.device)
        print("==========SigLIP model initialized and ready for processing.==========")

    def get_embeddings(self, descriptions):
        inputs = self.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs)

    def post_process_average(self, average_vector):
        return average_vector / average_vector.norm(dim=-1, keepdim=True)
