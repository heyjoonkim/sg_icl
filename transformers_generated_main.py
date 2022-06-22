
import argparse
import logging
import os
import random
import json
import time

from datasets import load_metric, DatasetDict, Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
import torch

from model_wrapper.TransformersModelWrapper import GPT2Wrapper
from utils import save_config
from dataset_utils import generated_task_to_path, task_to_keys, task_to_verbalizer, prepare_generated_incontext_sampling, prepend_incontext_samples

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(generated_task_to_path.keys()),
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default=None,
        help="The name of the benchmark to train on.",
        choices=['glue', 'super_glue', 'huggingface'],
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default=None, 
        help="Path for the generated datasets."
    )
    parser.add_argument(
        '--overwrite_output_dir', 
        default=False, 
        action="store_true",
        help='Overwrite output directory.'
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )

    # for Few-shot inference
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=0, 
        help="Number of samples for in-context learning."
    )
    parser.add_argument(
        '--balance_sample', 
        default=False, 
        action="store_true",
        help='Balance samples per label for in-context learning.'
    )
    # for manual prompt #
    parser.add_argument(
        "--prefix",
        type=str,
        default='',
        help="Prefix prompt.",
    )
    parser.add_argument(
        "--infix",
        type=str,
        default='',
        help="Infix prompt.",
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default='',
        help="Postfix prompt.",
    )

    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)

    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()
            
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.verbalizer = task_to_verbalizer.get(args.task_name)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    # Handle the repository creation & SummaryWriter
    save_config(args)

    raw_datasets = DatasetDict()
    
    # for datasets from file.
    if args.task_name in generated_task_to_path:
        dataset_processor = generated_task_to_path[args.task_name]["dataset_processor"]
        validation_file_path = generated_task_to_path[args.task_name]["validation"]
        validation_file_path = os.path.join(args.dataset_dir, validation_file_path)

        # validation set
        validation_dict = dataset_processor(validation_file_path)
        raw_eval_dataset = Dataset.from_dict(validation_dict)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    raw_datasets['validation'] = raw_eval_dataset

    logger.info('TRAIN / VALIDATION split.')
    for split, dataset in raw_datasets.items():
        logger.info(f'{split} > {len(dataset)}')
    
    # Labels
    if args.task_name in generated_task_to_path:
        label_list = set(raw_datasets["validation"]['label'])
        num_labels = len(label_list)
    else:
        raise NotImplementedError(f'{args.task_name} task is not implemented yet.')

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # TODO: only inject pad_token_id in case of GPT
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, 
        num_labels=num_labels, 
        finetuning_task=args.task_name, 
        pad_token_id=tokenizer.unk_token_id
    )

    logger.info(f'Start loading {args.model_name_or_path} model...')
    model_loading_start_time = time.time()
    model = GPT2Wrapper(config=config, model_name_or_path=args.model_name_or_path, verbalizer=args.verbalizer)
    model_loading_end_time = time.time()
    logger.info(f'Total time for loading model : {model_loading_end_time - model_loading_start_time}')

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    # load generated in-context samples
    # full_train_samples_list : all in-context samples -> for random sampling
    # label2samples_list      : in-context samples for each label -> for balanced sampling
    label2samples_list, full_train_samples_list = prepare_generated_incontext_sampling(
        generated_samples=raw_datasets['validation'],
        verbalizer=args.verbalizer,
        prefix=args.prefix,
        infix=args.infix,
        postfix=args.postfix,
        sentence1_key=sentence1_key,
        sentence2_key=sentence2_key)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

        sample_num = len(texts[0])
        for sample_index in range(sample_num):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = dict()
            sample_num = len(texts[0])
            result['sentence1'] = examples[sentence1_key]
            input_sentences = []

            # for single sentence tasks
            if sentence2_key is None:
                for sample_index in range(sample_num):
                    input_sentence = args.prefix + texts[0][sample_index] + args.infix + args.postfix
                    input_sentences.append(input_sentence)
            else:
                result['sentence2'] = examples[sentence2_key]
                for sample_index in range(sample_num):
                    input_sentence = args.prefix + texts[0][sample_index] + args.infix + texts[1][sample_index] + args.postfix
                    input_sentences.append(input_sentence)

            result['input_sentence'] = input_sentences
            
            # Map labels to IDs (not necessary for GLUE tasks)
            if "label" in examples:
                # for SST-2, SST-5, AGNews
                result["labels"] = examples["label"]
            elif 'label-coarse' in examples:
                # for TREC
                result["labels"] = examples['label-coarse']
            else:
                raise NotImplementedError
            return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
        desc="Preprocessing datasets...",
    )

    eval_dataset = processed_datasets["validation"]

    # Get the metric function  
    if args.benchmark_name == 'huggingface':
        metric = load_metric("accuracy")
    else:
        metric = load_metric(args.benchmark_name, args.task_name)
    
    # Evaluate! 
    logger.info("***** Zero/Few-shot Evaluation *****")
    logger.info(f"  Task name                   = {args.task_name}")
    logger.info(f"  Num EVAL  examples          = {len(eval_dataset)}")
    logger.info(f"  Random Seed                 = {args.seed}")
    logger.info(f"  K                           = {args.n_samples}")
    logger.info(f"  Inference Model             = {args.model_name_or_path}")
    logger.info(f"  Model Device                = {model.device}")
         
    # for analysis
    prediction_dict = {}

    start_time = time.time()
    model.eval()

    # evaluate
    progressbar = tqdm(range(len(eval_dataset)))
    for step, inputs in enumerate(eval_dataset):

        # in-context samples generated conditioned by the input x.
        if args.n_samples > 0:
            incontext_samples, sep = prepend_incontext_samples(
                label2samples=label2samples_list[step],
                full_train_samples=full_train_samples_list[step],
                k=args.n_samples,
                balance_sample=args.balance_sample,
            )
            # prepend in-context samples
            inputs['input_sentence'] = incontext_samples + sep + inputs['input_sentence']
        
        label = torch.tensor(inputs['labels']).to('cuda').unsqueeze(dim=0)

        # logging first sample
        if step == 0:
            logger.info('LOGGING FIRST GENERATED SAMPLE.')
            logger.info(f'LABEL : {label}')
            logger.info(f'INPUT SENTENCE : {inputs["input_sentence"]}')

        # prediction  : predicted label index
        # predictions : logit values for each label
        prediction, predictions = model(**inputs)
        
        metric.add_batch(
            predictions=prediction,
            references=label,
        )

        # for analysis : save predictions
        prediction = prediction.cpu().item()
        prediction_dict[prediction] = prediction_dict.get(prediction, 0) + 1

        progressbar.update(1)

    eval_metric = metric.compute()

    if args.n_samples == 0:
        logger.info(f"** Zero-shot evaluation result : {eval_metric}")
    else:
        logger.info(f"** {args.n_samples}-shot evaluation result : {eval_metric}")

    logger.info(f'Predictions distribution : {prediction_dict}')

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time} sec.')
    logger.info("Done.")
                
if __name__ == "__main__":
    logger.info('\nRunning : transformers_generated_main.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')