import json
import os
import random
import argparse
from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer
from tqdm import tqdm
from config import Arguments
from logger_config import logger
from loader_utils import group_doc_ids

class RetrievalDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        print("[Init] Initializing RetrievalDataLoader...")
        self.args = args
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        self.corpus: Dataset = load_dataset(
            "json",
            data_files=os.path.join(args.data_dir, 'passages.jsonl')
        )['train']

        id_list = self.corpus["id"]                  # a plain Python list of all IDs
        self.id2idx = dict(zip(id_list, range(len(id_list))))

        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None
        print("[Init] RetrievalDataLoader initialized.")


    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        print("[_transform_func] Starting transform...")
        current_epoch = 0

        input_doc_ids: List[str] = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            use_first_positive=self.args.use_first_positive
        )
        print(f"[_transform_func] Grouped {len(input_doc_ids)} document IDs.")

        input_doc_ids = [self.id2idx[doc_id] for doc_id in input_doc_ids]
        input_docs: List[str] = self.corpus.select(input_doc_ids)["text"]

        print("[_transform_func] Tokenizing queries...")
        query_batch_dict = self.tokenizer(examples['query'],
                                          max_length=self.args.q_max_len,
                                          padding=PaddingStrategy.DO_NOT_PAD,
                                          truncation=True)
        print("[_transform_func] Tokenizing documents...")
        doc_batch_dict = self.tokenizer(input_docs,
                                        max_length=self.args.p_max_len,
                                        padding=PaddingStrategy.DO_NOT_PAD,
                                        truncation=True)

        merged_dict = {'q_{}'.format(k): v for k, v in query_batch_dict.items()}
        step_size = self.args.train_n_passages
        for k, v in doc_batch_dict.items():
            k = 'd_{}'.format(k)
            merged_dict[k] = []
            for idx in range(0, len(v), step_size):
                merged_dict[k].append(v[idx:(idx + step_size)])

        if self.args.do_kd_biencoder:
            print("[_transform_func] Generating KD labels...")
            qid_to_doc_id_to_score = {}

            def _update_qid_pid_score(q_id: str, ex: Dict):
                assert len(ex['doc_id']) == len(ex['score'])
                if q_id not in qid_to_doc_id_to_score:
                    qid_to_doc_id_to_score[q_id] = {}
                for doc_id, score in zip(ex['doc_id'], ex['score']):
                    qid_to_doc_id_to_score[q_id][int(doc_id)] = score

            for idx, query_id in enumerate(examples['query_id']):
                _update_qid_pid_score(query_id, examples['positives'][idx])
                _update_qid_pid_score(query_id, examples['negatives'][idx])

            merged_dict['kd_labels'] = []
            for idx in range(0, len(input_doc_ids), step_size):
                qid = examples['query_id'][idx // step_size]
                cur_kd_labels = [qid_to_doc_id_to_score[qid][doc_id] for doc_id in input_doc_ids[idx:idx + step_size]]
                merged_dict['kd_labels'].append(cur_kd_labels)

            assert len(merged_dict['kd_labels']) == len(examples['query_id']), \
                '{} != {}'.format(len(merged_dict['kd_labels']), len(examples['query_id']))
            print("[_transform_func] KD labels added.")

        print("[_transform_func] Finished transformation.")
        return merged_dict

    def _get_transformed_datasets(self) -> Tuple:
        print("[_get_transformed_datasets] Loading dataset from files...")
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)
        print("[_get_transformed_datasets] Dataset loaded.")

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            print("[_get_transformed_datasets] Preparing train dataset...")
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)
            print(f"[_get_transformed_datasets] Train dataset size: {len(train_dataset)}")

        if self.args.do_eval:
            print("[_get_transformed_datasets] Preparing eval dataset...")
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)
            print(f"[_get_transformed_datasets] Eval dataset size: {len(eval_dataset)}")

        return train_dataset, eval_dataset


if __name__ == "__main__":
    print("[Main] Parsing arguments...")


    def parse_args() -> Arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, help="Path to directory with *_passages.json files", default='Data/Raw Datasets')
        parser.add_argument("--train_file", type=str, help="Comma-separated list of training files", default=','.join(['Data/Datasets Info/raw_dataset_'+dataset_name+'.jsonl' for dataset_name in ['msmarco', 'nli', 'nq']]))
        parser.add_argument("--validation_file", type=str, help="Validation file path", default=None)
        parser.add_argument("--model_name_or_path", type=str, default="intfloat/e5-base-unsupervised")
        parser.add_argument("--train_n_passages", type=int, default=8)
        parser.add_argument("--q_max_len", type=int, default=32)
        parser.add_argument("--p_max_len", type=int, default=128)
        parser.add_argument("--max_train_samples", type=int, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--use_first_positive", type=bool, default=True)
        parser.add_argument("--do_train", type=bool, default=True)
        parser.add_argument("--do_eval", type=bool, default=False)
        parser.add_argument("--do_kd_biencoder", type=bool, default=True)
        args = parser.parse_args()
        return Arguments(**vars(args))

    args = parse_args()
    print("[Main] Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path, use_fast=True)
    print("[Main] Creating RetrievalDataLoader...")
    loader = RetrievalDataLoader(args, tokenizer)

    print("\n[Main] Train dataset example:")
    if loader.train_dataset is not None:
        for idx in range(2):
            print(loader.train_dataset[idx])
    else:
        print("No training dataset loaded.")

    print("\n[Main] Eval dataset example:")
    if loader.eval_dataset is not None:
        for idx in range(2):
            print(loader.eval_dataset[idx])
    else:
        print("No validation dataset loaded.")
