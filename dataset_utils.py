
#
# Utils for loading datasets from file (csv, tsv, ...).
# otherwise we use load_dataset() from huggingface library.
#

import csv
import random
import ast


def generated_sst5_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    samples0_list = []
    samples1_list = []
    samples2_list = []
    samples3_list = []
    samples4_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 8, f'Line length {len(line)} does not match the expected length 8.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[3])
            samples1 = ast.literal_eval(line[4])
            samples2 = ast.literal_eval(line[5])
            samples3 = ast.literal_eval(line[6])
            samples4 = ast.literal_eval(line[7])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
            # assert len(samples0) == len(samples3), f'number samples for label 0 {samples0} does not match the number of samples for label 3 {len(samples3)}'
            # assert len(samples0) == len(samples4), f'number samples for label 0 {samples0} does not match the number of samples for label 4 {len(samples4)}'
            
            label_list.append(label)
            sentence1_list.append(sentence1)
            samples0_list.append(samples0)
            samples1_list.append(samples1)
            samples2_list.append(samples2)
            samples3_list.append(samples3)
            samples4_list.append(samples4)

    return_dict = {
        'text' : sentence1_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
        'samples2' : samples2_list,
        'samples3' : samples3_list,
        'samples4' : samples4_list,
    }

    return return_dict

def generated_cb_generate_dataset_dict(filename):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    samples0_list = []
    samples1_list = []
    samples2_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 7, f'Line length {len(line)} does not match the expected length 7.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]
            sentence2 = line[3]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[4])
            samples1 = ast.literal_eval(line[5])
            samples2 = ast.literal_eval(line[6])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
           
            label_list.append(label)
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            samples0_list.append(samples0)
            samples1_list.append(samples1)
            samples2_list.append(samples2)

    return_dict = {
        'premise' : sentence1_list,
        'hypothesis' : sentence2_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
        'samples2' : samples2_list,
    }

    return return_dict

def generated_sst2_generate_dataset_dict(filename):
    sentence1_list = []
    label_list = []
    samples0_list = []
    samples1_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 5, f'Line length {len(line)} does not match the expected length 5.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[3])
            samples1 = ast.literal_eval(line[4])
            
            label_list.append(label)
            sentence1_list.append(sentence1)
            samples0_list.append(samples0)
            samples1_list.append(samples1)

    return_dict = {
        'sentence' : sentence1_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
    }

    return return_dict

def generated_rte_generate_dataset_dict(filename):
    sentence1_list = []
    sentence2_list = []
    label_list = []
    samples0_list = []
    samples1_list = []

    with open(filename) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        for line_index, line in enumerate(tsv_reader):

            assert len(line) == 6, f'Line length {len(line)} does not match the expected length 6.'
            
            index = int(line[0])
            label = int(line[1])
            sentence1 = line[2]
            sentence2 = line[3]

            assert line_index == index, f'index {index} != line_index {line_index}'

            # convert to list
            samples0 = ast.literal_eval(line[4])
            samples1 = ast.literal_eval(line[5])

            # assert len(samples0) == len(samples1), f'number samples for label 0 {samples0} does not match the number of samples for label 1 {len(samples1)}'
            # assert len(samples0) == len(samples2), f'number samples for label 0 {samples0} does not match the number of samples for label 2 {len(samples2)}'
           
            label_list.append(label)
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            samples0_list.append(samples0)
            samples1_list.append(samples1)

    return_dict = {
        'sentence1' : sentence1_list,
        'sentence2' : sentence2_list,
        'label' : label_list,
        'samples0' : samples0_list,
        'samples1' : samples1_list,
    }

    return return_dict

# for using generated datasets.
generated_task_to_path = {
    "SetFit/sst5" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_sst5_generate_dataset_dict,
    },
    "sst2" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_sst2_generate_dataset_dict,
    },
    "rte" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_rte_generate_dataset_dict,
    },
    "cb" : {
        "validation" : "test.tsv",
        "dataset_processor" : generated_cb_generate_dataset_dict,
    },
}

task_to_keys = {
    "sst2": ("sentence", None),     # #labels = 2
    "SetFit/sst5": ("text", None),         # #labels = 5
    "rte": ("sentence1", "sentence2"),
    "cb" : ("premise", "hypothesis"),

}

task_to_verbalizer = {
    "sst2": {
        " negative" : 0,
        " positive" : 1,
    },
    "SetFit/sst5" : {
        ' terrible' : 0,
        ' bad' : 1,
        ' okay' : 2,
        ' good' : 3,
        ' great' : 4,
    },
    "rte" : {
        # verbalizer 1
        " true" : 0,
        " false" : 1,
    },
    "cb" : {
        " yes" : 0,
        " no" : 1,
        " neither" : 2,
    }
}


def prepare_incontext_sampling(train_samples, 
                                verbalizer,
                                sentence1_key, 
                                sentence2_key,
                                prefix,
                                infix,
                                postfix,
                                ):

    label2token = {v:k for k,v in verbalizer.items()}
    label2samples = {}
    full_samples = []

    for sample in train_samples:
        sentence1 = sample[sentence1_key]
        if 'label' in sample:
            label = sample['label']
        elif 'label-coarse' in sample:
            label = sample['label-coarse']
        else:
            raise NotImplementedError
            
        label_token = label2token[label]
        if sentence2_key is not None:
            sentence2 = sample[sentence2_key]
        else:
            sentence2 = ''
        
        full_sentence = prefix + sentence1 + infix + sentence2 + postfix + label_token
        full_samples.append(full_sentence)

        # empty list if first sample
        label_list = label2samples.get(label, [])
        label_list.append(full_sentence)
        label2samples[label] = label_list

    return label2samples, full_samples
        

def prepend_incontext_samples(
                                label2samples,
                                full_train_samples,
                                k,
                                balance_sample,
                            ):

    
    final_sentence = None
    sep = '\n\n\n'
    # sep = '\n\n\n\n'

    # no in-context samples = zero-shot learning
    if k == 0:
        return '', sep

    if balance_sample:
        total_count = 0
        labels = list(label2samples.keys())
        random.shuffle(labels)
        # prevent infinite while-loop
        samples_map = {label:[i for i in range(len(label2samples[label]))] for label in labels}
        while True:
            for label in labels:
                samples = label2samples[label]
                total_length = len(samples)
                not_used_indices = [i for i in range(total_length)]
                while True:
                    samples_list = samples_map[label]
                    random_index = random.randint(0, total_length-1)
                    selected_sample = samples[random_index]

                    # we don't want to use duplicate in-context samples
                    if final_sentence is None:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break
                    if random_index in samples_list:
                        selected_index = samples_list.index(random_index)
                        samples_list.pop(selected_index)
                        samples_map[label] = samples_list
                        break

                if final_sentence is None:
                    final_sentence = selected_sample
                else:
                    final_sentence = final_sentence + sep + selected_sample

                total_count += 1
                if total_count == k:
                    return final_sentence, sep
    else:
        full_train_samples_copy = full_train_samples.copy()
        for index in range(k):
            total_length = len(full_train_samples_copy)
            random_index = random.randint(0, total_length-1)
            selected_sample = full_train_samples_copy.pop(random_index)

            if final_sentence is None:
                final_sentence = selected_sample
            else:
                final_sentence = final_sentence + sep + selected_sample

    return final_sentence, sep



def prepare_generated_incontext_sampling(generated_samples, 
                                verbalizer,
                                prefix,
                                infix,
                                postfix,
                                sentence1_key,
                                sentence2_key,
                                append_label=True
                                ):

    label2token = {v:k for k,v in verbalizer.items()}
    num_labels = len(label2token.keys())
    label2samples_list=[] 
    full_samples_list=[]

    for samples in generated_samples:
        label2samples = {}
        full_samples = []
        # if sentence2_key is not None -> sentence-pair task -> use the first sentence
        sentence1 = samples[sentence1_key] if sentence2_key is not None else None

        for label in range(num_labels):
            label_token = label2token[label]
            if not append_label:
                label_token = ''
            key = f'samples{label}'
            samples_list = samples[key]

            promped_samples_list = []
            for sample_index, sample in enumerate(samples_list):
                if sentence1:
                    promped_samples_list.append(prefix + sentence1 + infix + sample +postfix + label_token)
                else:
                    promped_samples_list.append(prefix + sample + infix + postfix + label_token)
            # samples_list = [prefix + sample + infix + postfix + label_token for sample in samples_list]

            full_samples = full_samples + promped_samples_list
            label2samples[label] = promped_samples_list
        
        label2samples_list.append(label2samples)
        full_samples_list.append(full_samples)


    return label2samples_list, full_samples_list