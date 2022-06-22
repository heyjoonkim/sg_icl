import argparse
import logging
import os
import random
import time
import csv

from datasets import load_dataset, DatasetDict
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import torch

from utils import save_config
from dataset_utils import task_to_keys, task_to_verbalizer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
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
    # manual prompts for generation #
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
    parser.add_argument(
        "--label_token", 
        type=str, 
        default="[LABEL]", 
        help="Where to store the final model."
    )
    parser.add_argument(
        '--apply_input', 
        default=False, 
        action="store_true",
        help='Apply input sentence.'
    )
    # until here #

    # hyperparams for generation #
    parser.add_argument(
        "--generation_max_length", 
        type=int, 
        default=10, 
        help="Max length for generation."
    )
    parser.add_argument(
        '--generation_min_length', 
        default=10, 
        type=int, 
        help='Min length for generation.'
    )
    parser.add_argument(
        "--no_repeat_ngram_size", 
        type=int, 
        default=2, 
        help="no_repeat_ngram_size."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.5, 
        help="Temperature for sampling."
    )
    # until here #

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

    # mkdir output directory to save logs and configs.
    if args.output_dir is not None:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.overwrite_output_dir:
                logger.info(f'Output directory {args.output_dir} exits. Exit program. (overwrite_output_dir=False)')
                exit()
    
    # file for writing generated demonstrations
    generation_writer = os.path.join(args.output_dir, "test.tsv")
    # prevent from overwriting generated dataset
    if os.path.isfile(generation_writer):
        logger.info('Generated dataset already exists. Exit Program.')
        exit()
            
    logging_output_file = os.path.join(args.output_dir, "output.log")
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler = logging.FileHandler(logging_output_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    args.verbalizer = task_to_verbalizer.get(args.task_name)
    args.label2token = {v:k for k,v in args.verbalizer.items()}

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        random.seed(args.seed)

    # Handle the repository creation & SummaryWriter
    save_config(args)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = DatasetDict()
    if args.task_name is not None and args.benchmark_name is not None:
        if args.benchmark_name == 'huggingface':
            # SST-2, TREC, AGNews
            raw_eval_dataset = load_dataset(args.task_name, split='test')
        else:
            # SST-2
            raw_eval_dataset = load_dataset(args.benchmark_name, args.task_name, split=f'validation')
    else:
        raise NotImplementedError(f'{args.task_name} task not in GLUE benchmark.')

    raw_datasets['validation'] = raw_eval_dataset

    logger.info('Loaded VALIDATION split for generation.')
    for split, dataset in raw_datasets.items():
        logger.info(f'{split} > {len(dataset)}')

    num_labels = len(args.verbalizer)
    
    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # For gpt-2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    logger.info(f'Start loading {args.model_name_or_path} model...')
    model_loading_start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            revision="float16",             # specific model version to use. We use FP16 model
            torch_dtype=torch.float16,  
            low_cpu_mem_usage=True,         # keep RAM usage to 1x
    ).to('cuda')

    model_loading_end_time = time.time()
    logger.info(f'Total time for loading model : {model_loading_end_time - model_loading_start_time}')

    # Preprocessing the datasets
    sentence1_key, sentence2_key = task_to_keys[args.task_name]

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

            # for single sentence tasks
            if sentence2_key is not None:
                result['sentence2'] = examples[sentence2_key]
                            
            # Map labels to IDs (not necessary for GLUE tasks)
            if "label" in examples:
                result["labels"] = examples["label"]
            elif 'label-coarse' in examples:
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

    # Generate! 
    logger.info("***** Generating Demonstrations per Sample *****")
    logger.info(f"  Task name                  = {args.task_name}")
    logger.info(f"  Num EVAL examples          = {len(eval_dataset)}")
    logger.info(f"  Generation per eval sample = {args.n_samples}")
    logger.info(f"  Random Seed                = {args.seed}")
    logger.info(f"  Inference Model            = {args.model_name_or_path}")
    logger.info(f"  Model Device               = {model.device}")
         
    
    # ignore generating comma(,) and new_line(\n)
    ignored_sequences = [',', ' ,', ' \n', '\n', ' \t', '\t']
    bad_words_ids = [ tokenizer.encode(ignored_sequence) for ignored_sequence in ignored_sequences]
    logger.info(f"  Ignored sequences : {ignored_sequences} -> {bad_words_ids}")

    start_time = time.time()
    model.eval()

    with open(generation_writer, 'w') as file_writer:
        tsv_writer = csv.writer(file_writer, delimiter='\t')

        progressbar = tqdm(range(len(eval_dataset)))
        for step, inputs in enumerate(eval_dataset):
            # input sentences
            sentence1 = inputs['sentence1']
            sentence2 = inputs['sentence2'] if 'sentence2' in inputs else ''

            # original_input = args.prefix + sentence1 + args.infix + sentence2 + args.postfix
            original_input = args.prefix + args.infix + sentence2 + args.postfix

            # gold label for the input
            label = inputs['labels']

            # add label and input sentences to write in .tsv file
            # this step is very importance since we use the generated .tsv file for later inference.
            row = [step, label, sentence1]
            if 'sentence2' in inputs:
                row.append(sentence2)
            
            # generate in-context samples for each label
            for index, (label_token, label) in enumerate(args.verbalizer.items()):
                assert index == label, f'index {index} != label {label}'
                
                # replace args.label_toke with label token
                label_dependent_input = original_input.replace(args.label_token, label_token)

                # logging first input
                if step == 0 and index == 0:
                    logger.info(f'LOGGING GENERATION INPUT : {label_dependent_input}')

                l = len(label_dependent_input)

                tokenized_inputs = tokenizer(label_dependent_input, return_tensors='pt').to('cuda')
                # shape : (1, input_length) -> (input_length, )
                input_ids = tokenized_inputs['input_ids'].squeeze(dim=0)
                input_length = len(input_ids)

                generated_ids = model.generate(
                    **tokenized_inputs,
                    do_sample=True,
                    max_length=input_length+args.generation_max_length,
                    min_length=input_length+args.generation_min_length,
                    temperature=args.temperature,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    num_return_sequences=args.n_samples,
                    early_stopping=True,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=tokenizer.eos_token_id,
                )

                # list of length n_samples
                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                generated_outputs = [genenerated_output[l:].replace('\n', '').strip() for genenerated_output in generated_outputs]
                
                row.append(generated_outputs)
            
            tsv_writer.writerow(row)

            progressbar.update(1)

    end_time = time.time()
    logger.info(f'Total time : {end_time - start_time} sec.')
                
if __name__ == "__main__":
    logger.info('\nRunning : transformers_generate.py')
    
    start_time = time.time()
    main()
    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')