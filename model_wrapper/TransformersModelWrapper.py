
from typing import Tuple

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path, verbalizer):
        super(GPT2Wrapper, self).__init__()

        self.config = config
        self.device = torch.device("cuda")

        # Main model
        try:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
                revision="float16",             # specific model version to use. We use FP16 model
                torch_dtype=torch.float16,  
                low_cpu_mem_usage=True,         # keep RAM usage to 1x
            ).to("cuda")     
        except:
            self.transformer = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, 
            ).to("cuda")   

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, truncation_side='left')

        self.num_labels = config.num_labels
        # token -> label
        self.verbalizer = verbalizer
        # label -> token
        self.label2token = {v:k for k,v in self.verbalizer.items()}

        assert self.num_labels == len(self.verbalizer.keys()), f'Number of labels({self.num_labels}) and verbalizer({self.verbalizer}) does not match'

        self.ids_list, self.multiple_token_flag = self._convert_verbalizer_to_ids(self.label2token, self.tokenizer)

    # returns list of token ids of the verbalizer
    def _convert_verbalizer_to_ids(self, label2token, tokenizer):
        ids_list = []
        multiple_token_flag = False
        for label_index in range(self.num_labels):
            token = label2token[label_index]
            ids = tokenizer(token)['input_ids']
            print('> label_index', label_index, 'token', token, 'ids', ids)
            ids_list.append(ids)
            # ids_list.append(ids[0])

            if len(ids) > 1:
                multiple_token_flag = True
                print(f'Multiple token for verbalizer {token} -> {ids}')

        if not multiple_token_flag:
            ids_list = [ids[0] for ids in ids_list]

        assert len(ids_list) == self.num_labels

        return ids_list, multiple_token_flag

    # get log-probability of the token of the label
    # for multiple-tokens we normalize by length (for now)
    def _verbalize(
        self,
        logprobs,
        label_index=None,
    ):
        if self.multiple_token_flag:
            # multiple token verbalizer

            # shift log-probabilities
            logprobs = logprobs[:, :-1, :]
            label_tokens = self.ids_list[label_index]
            label_length = len(label_tokens)

            # shape : (1, label_length, vocab_size)
            # shape : (1, label-token-length, vocab_size)
            label_logprobs = logprobs[:, -label_length:, :]
            total_probs=0
            for input_index, token_index in enumerate(label_tokens, start=1):
                token_logprob = label_logprobs[0, -input_index, token_index]
                total_probs += token_logprob
            # TODO : normalize?
            total_probs = total_probs / label_length
        else:
            # single token verbalizer
            # use only the final distribution for prediction
            total_probs = logprobs[0, -1,  self.ids_list]
        return total_probs

    def forward(
        self,
        input_sentence,
        sentence1=None,
        sentence2=None,
        labels=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # print('=' * 50)
        # print('input sentence :\n', input_sentence)

        if self.multiple_token_flag:
            predictions = []

            # same as noisy channel inference
            for label_token, label_index in self.verbalizer.items():
                label_appended_input_sentence = input_sentence + label_token
                # tokenize label specific input sentence 
                tokenized_inputs = self.tokenizer(label_appended_input_sentence, return_tensors='pt').to(self.transformer.device)

                outputs = self.transformer(**tokenized_inputs)
                
                # shape : (1, length, vocab_size)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=2)
                # shape : (1, length, vocab_size)
                logprobs = torch.log(probs)

                verbalizer_logprob = self._verbalize(logprobs, label_index)

                assert label_index == len(predictions), f'label index : {label_index} <-> prediction count : {len(predictions)}'
                predictions.append(verbalizer_logprob)

            predictions = torch.stack(predictions)
        else:
            # tokenize label specific input sentence 
            tokenized_inputs = self.tokenizer(input_sentence, truncation=True, max_length=2048, return_tensors='pt').to("cuda")
            
            with torch.no_grad():
                try:
                    outputs = self.transformer(**tokenized_inputs)
                except:
                    print(f'Error with input : {input_sentence}')
                    print('input ids', len(tokenized_inputs['input_ids'][0]))

            del tokenized_inputs
            torch.cuda.empty_cache()
                
            # shape : (1, length, vocab_size)
            logits = outputs.logits.cpu()
            del outputs

            probs = torch.softmax(logits, dim=2)
            # shape : (1, length, vocab_size)
            logprobs = torch.log(probs)

            predictions = self._verbalize(logprobs)

        prediction = torch.argmax(predictions, dim=-1)

        # shape : (1, )
        return prediction.unsqueeze(dim=0), predictions