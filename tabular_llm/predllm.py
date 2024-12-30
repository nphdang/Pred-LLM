import os
import warnings
import json
import typing as tp
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments)

from tabular_llm.predllm_dataset import PredLLMDataset, GReaTDataCollator
from tabular_llm.predllm_start import GReaTStart, CategoricalStart, ContinuousStart, RandomStart
from tabular_llm.predllm_trainer import GReaTTrainer
from tabular_llm.predllm_utils import _array_to_dataframe, _get_column_distribution, _convert_tokens_to_text, \
    _convert_text_to_tabular_data, _encode_row_partial


class PredLLM:
    """ GReaT Class

    The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(self, llm: str, experiment_dir: str = "trainer_great", epochs: int = 100,
                 batch_size: int = 8, **train_kwargs):
        """ Initializes GReaT.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.data = None
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(self, data: tp.Union[pd.DataFrame, np.ndarray], column_names: tp.Optional[tp.List[str]] = None,
            conditional_col: tp.Optional[str] = None, resume_from_checkpoint: tp.Union[bool, str] = False)\
            -> GReaTTrainer:
        """ Fine-tune GReaT using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        df = _array_to_dataframe(data, columns=column_names)
        self.data = df
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        df_new = pd.concat([df, df], axis=0)
        df_new = df_new.reset_index(drop=True)
        great_ds = PredLLMDataset.from_pandas(df_new)
        great_ds.get_ds_size(df.shape[0])
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(self.experiment_dir, save_strategy="no",
                                          num_train_epochs=self.epochs,
                                          per_device_train_batch_size=self.batch_size,
                                          **self.train_hyperparameters)
        great_trainer = GReaTTrainer(self.model, training_args, train_dataset=great_ds, tokenizer=self.tokenizer,
                                     data_collator=GReaTDataCollator(self.tokenizer))

        # Start training
        logging.info("Start training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return great_trainer

    def sample(self, n_samples: int,
               start_col: tp.Optional[str] = "", start_col_dist: tp.Optional[tp.Union[dict, list]] = None,
               temperature: float = 0.7, k: int = 100, max_length: int = 100, device: str = "cuda") -> pd.DataFrame:
        """ Generate synthetic tabular data samples

        Args:
            n_samples: Number of synthetic samples to generate
            start_col: Feature to use as starting point for the generation process. If not given, the target
             learned during the fitting is used as starting point
            start_col_dist: Feature distribution of the starting feature. Should have the format
             "{F1: p1, F2: p2, ...}" for discrete columns or be a list of possible values for continuous columns.
             If not given, the target distribution learned during the fitting is used as starting point
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process
            k: Sampling Batch Size. Set as high as possible. Speeds up the generation process significantly
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information!
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU

        Returns:
            Pandas DataFrame with n_samples rows of generated data
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        # Move model to device
        self.model.to(device)

        # Init list for generated DataFrames
        dfs = []

        # Start generation process
        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            _cnt = 0
            while n_samples > already_generated:
                start_tokens = great_start.get_start_tokens(k)
                start_tokens = torch.tensor(start_tokens).to(device)

                # Generate tokens
                tokens = self.model.generate(input_ids=start_tokens, max_length=max_length,
                                             do_sample=True, temperature=temperature, pad_token_id=50256)

                # Convert tokens back to tabular data
                text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                df_gen = _convert_text_to_tabular_data(text_data, pd.DataFrame(columns=self.columns))

                # Remove rows where we have not generated anything
                df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]

                # replace NA or inf with 0
                for i_num_cols in range(len(self.num_cols)):
                    df_gen[df_gen.columns[i_num_cols]].replace([np.inf, -np.inf], np.nan, inplace=True)
                    df_gen[df_gen.columns[i_num_cols]].fillna(0, inplace=True)

                # Remove rows with flawed numerical values
                for i_num_cols in self.num_cols:
                    df_gen = df_gen[pd.to_numeric(df_gen[i_num_cols], errors='coerce').notnull()]

                # Convert numerical columns to float
                df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                dfs.append(df_gen)
                already_generated += len(dfs[-1])

                # Update process bar
                pbar.update(len(dfs[-1]))

                # Check if we are actually generating synthetic samples and if not, break everything
                _cnt += 1
                if _cnt > 13 and already_generated == 0:
                    raise Exception("Breaking the generation loop!")

        df_gen = pd.concat(dfs)
        df_gen = df_gen.reset_index(drop=True)

        return df_gen.head(n_samples)

    def sample_new(self, n_samples: int, max_length: int = 100, task: str = "classification") -> pd.DataFrame:
        n_generative = n_samples
        n_feature = len(self.data.columns) - 1
        feature_names = self.data.columns[:-1]
        # select each feature to be a pre-condition
        n_each_feature = int(np.ceil(n_generative / n_feature))
        dfs = []
        for feature in feature_names:
            df_gen = self.sample(n_samples=n_each_feature, max_length=max_length,
                                 start_col=feature, start_col_dist=_get_column_distribution(self.data, feature))
            dfs.append(df_gen)
        X_y_train_new = pd.concat(dfs)
        X_y_train_new = X_y_train_new.reset_index(drop=True)
        X_y_train_new = X_y_train_new.head(n_generative)

        # replace NA or inf with 0
        for i_num_cols in range(n_feature + 1):
            X_y_train_new[X_y_train_new.columns[i_num_cols]].replace([np.inf, -np.inf], np.nan, inplace=True)
            X_y_train_new[X_y_train_new.columns[i_num_cols]].fillna(0, inplace=True)
        # remove rows with flawed numerical values
        for i_num_cols in range(n_feature + 1):
            X_y_train_new = X_y_train_new[pd.to_numeric(X_y_train_new.iloc[:, i_num_cols], errors='coerce').notnull()]

        X_train_new = X_y_train_new.iloc[:, :-1]
        X_train_new = np.around(X_train_new, 2)
        y_train_new = X_y_train_new.iloc[:, -1:]

        print("use llm as classifier")
        prompts = []
        for idx in range(X_train_new.shape[0]):
            encoded_text = _encode_row_partial(X_train_new.iloc[idx], shuffle=False)
            prompts.append(encoded_text)
        y_train_gen = self.great_sample(prompts, max_length=max_length).iloc[:, -1:]

        n_llm_pred = 0
        n_tried = 0
        while n_llm_pred < n_generative and n_tried < 5:
            # find rows with flawed labels
            invalid_indices = np.where(pd.to_numeric(y_train_gen.iloc[:, 0], errors='coerce').notnull() == False)[0]
            n_llm_pred = n_generative - len(invalid_indices)
            print("n_generative: {}, n_llm_pred: {}".format(n_generative, n_llm_pred))
            for idx in invalid_indices:
                y_train_gen.iloc[idx, 0] = self.great_sample(prompts[idx], max_length=max_length).iloc[0, -1]
            n_tried += 1
        # find rows with flawed labels
        invalid_indices = np.where(pd.to_numeric(y_train_gen.iloc[:, 0], errors='coerce').notnull() == False)[0]
        # use previous labels
        for idx in invalid_indices:
            y_train_gen.iloc[idx, 0] = y_train_new.iloc[idx, 0]
        y_train_new = y_train_gen

        X_train_new = X_train_new.to_numpy(dtype=float).reshape(-1, n_feature)
        y_train_new = y_train_new.to_numpy(dtype=float).reshape(-1, )

        # post-process synthetic data
        if task == "classification":
            # match labels of real and synthetic samples
            y_train = self.data.iloc[:, -1:].to_numpy(dtype=float).reshape(-1, )
            real_min, real_max = np.min(y_train), np.max(y_train)
            fake_min, fake_max = np.min(y_train_new), np.max(y_train_new)
            if fake_min < real_min:
                print("y_fake_min < y_real_min")
                y_train_new[y_train_new == fake_min] = real_min
            if fake_max > real_max:
                print("y_fake_max > y_real_max")
                y_train_new[y_train_new == fake_max] = real_max
        # fix error if generating a large value for feature
        # note that each feature is normalized to [0, 1]
        large_value = 10
        small_value = -1 * large_value
        fake_min, fake_max = np.min(X_train_new), np.max(X_train_new)
        if fake_min < small_value:
            print("X_fake_min is too small")
            X_train_new[X_train_new < small_value] = small_value
        if fake_max > large_value:
            print("X_fake_max is too large")
            X_train_new[X_train_new > large_value] = large_value
        X_y_train_new = np.append(X_train_new, y_train_new.reshape(-1, 1), axis=1)
        X_y_train_new = pd.DataFrame(X_y_train_new)

        return X_y_train_new

    def great_sample(self, starting_prompts: tp.Union[str, list[str]], temperature: float = 0.7, max_length: int = 100,
                     device: str = "cuda") -> pd.DataFrame:
        """ Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.

        self.model.to(device)
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_data = []

        # Generate a sample for each starting point
        if len(starting_prompts) > 1:
            loop_iter = tqdm(starting_prompts)
        else:
            loop_iter = starting_prompts

        for prompt in loop_iter:
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length,
                                      do_sample=True, temperature=temperature, pad_token_id=50256)
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        df_gen = _convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))

        return df_gen

    def save(self, path: str):
        """ Save GReaT Model

        Saves the model weights and a configuration file in the given directory.

        Args:
            path: Path where to save the model
        """
        # Make directory
        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        # Save attributes
        with open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer")
            attributes.pop("model")

            # NDArray is not JSON serializable and therefore has to be converted into a list.
            if isinstance(attributes["conditional_col_dist"], np.ndarray):
                attributes["conditional_col_dist"] = list(attributes["conditional_col_dist"])

            json.dump(attributes, f)

        # Save model weights
        torch.save(self.model.state_dict(), path + "/model.pt")

    def load_finetuned_model(self, path: str):
        """ Load fine-tuned model

        Load the weights of a fine-tuned large language model into the GReaT pipeline

        Args:
            path: Path to the fine-tuned model
        """
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(cls, path: str):
        """ Load GReaT class

        Load trained GReaT model from directory.

        Args:
            path: Directory where GReaT model is saved

        Returns:
            New instance of GReaT loaded from directory
        """
        assert os.path.isdir(path), f"Directory {path} does not exist."

        # Load attributes
        with open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        # Create new be_great model instance
        great = cls(attributes["llm"])

        # Set all attributes
        for k, v in attributes.items():
            setattr(great, k, v)

        # Load model weights
        great.model.load_state_dict(torch.load(path + "/model.pt", map_location="cpu"))

        return great

    def _update_column_information(self, df: pd.DataFrame):
        # Update the column names (and numerical columns for some sanity checks after sampling)
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
        assert conditional_col is None or isinstance(conditional_col, str), \
            f"The column name has to be a string and not {type(conditional_col)}"
        assert conditional_col is None or conditional_col in df.columns, \
            f"The column name {conditional_col} is not in the feature names of the given dataset"

        # Take the distribution of the conditional column for a starting point in the generation process
        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)

    def _get_start_sampler(self, start_col: tp.Optional[str],
                           start_col_dist: tp.Optional[tp.Union[tp.Dict, tp.List]]) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(f"Start column {start_col} was given, but no corresponding distribution.")
        if start_col_dist is not None and not start_col:
            raise ValueError(f"Start column distribution {start_col} was given, the column name is missing.")

        assert start_col is None or isinstance(start_col, str), \
            f"The column name has to be a string and not {type(start_col)}"
        assert start_col_dist is None or isinstance(start_col_dist, dict) or isinstance(start_col_dist, list), \
            f"The distribution of the start column on has to be a list or a dict and not {type(start_col_dist)}"

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)
        elif isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)
        else:
            return RandomStart(self.tokenizer, self.columns)

