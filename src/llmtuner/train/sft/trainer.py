import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        Removes the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None  # backup labels
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:  # truncate the labels instead of padding the inputs (llama2 fp16 compatibility)
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(  # ignore the returned labels (may be truncated)
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Pads the tensor to the same length as the target tensor.
        """
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor  # adopt left-padding
        return padded_tensor.contiguous()  # in contiguous memory
    

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )  # move pad token to last

        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        t=0
        num_decoded_labels_len = 0
        total_TP = total_FP = total_FN = 0  

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
#######################################################################res###########################################################################

            #res直接写到文档里
            res: List[str] = []

            for label, pred in zip(decoded_labels, decoded_preds):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))

######################################################################res1###########################################################################
            res1: List[str] = []
            for label, pred in zip(decoded_labels, decoded_preds):
                res1.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))

            for line in res1:
                data = json.loads(line)

###以下是重点

                decoded_labels = data["label"]
                decoded_preds = data["predict"]

                ####exact match
                if "star" not in output_prediction_file:
                    #print("star not in filename")

                    # Replace commas within <head> tags
                    decoded_labels = re.sub(r'(<head>)(.*?)(</head>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<head>)(.*?)(</head>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # Replace commas within <relation> tags
                    decoded_labels = re.sub(r'(<relation>)(.*?)(</relation>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<relation>)(.*?)(</relation>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # Replace commas within <tail> tags
                    decoded_labels = re.sub(r'(<tail>)(.*?)(</tail>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<tail>)(.*?)(</tail>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # print("####################################")
                    # print("decoded_labels:",decoded_labels)
                    # print("decoded_preds:",decoded_preds)
                    # print("####################################")

                    # #去掉标识
                    decoded_preds  = data["predict"].replace("<triplet>", "").replace("</triplet>", "").replace("<head>", "").replace("</head>", ",").replace("<relation>", "").replace("</relation>", ",").replace("<tail>", "").replace("</tail>", "") 
                    decoded_labels = data["label"].replace("<triplet>", "").replace("</triplet>", "").replace("<head>", "").replace("</head>", ",").replace("<relation>", "").replace("</relation>", ",").replace("<tail>", "").replace("</tail>", "") 

                    decoded_labels = {
                        tuple(triple.split(',')) for triple in decoded_labels.split(';')
                    }
                    decoded_preds = {
                        tuple(triple.split(',')) for triple in decoded_preds.split(';')
                    }
                    
                    # 先对三元组内部元素进行排序
                    sorted_labels = {tuple(sorted(triplet)) for triplet in decoded_labels}
                    sorted_preds = {tuple(sorted(triplet)) for triplet in decoded_preds}

                    # 再对三元组进行排序
                    decoded_labels = sorted(sorted_labels)
                    decoded_preds = sorted(sorted_preds)

                    # 将列表转换为集合
                    decoded_labels = set(decoded_labels)
                    decoded_preds = set(decoded_preds)

                    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
                    TP = len(decoded_labels & decoded_preds)
                    FP = len(decoded_preds - decoded_labels)
                    FN = len(decoded_labels - decoded_preds)

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                    num_decoded_labels_len += len(decoded_labels)

                if "star" in output_prediction_file:
                    #print("star in filename")

                    # Replace commas within <head> tags
                    decoded_labels = re.sub(r'(<head>)(.*?)(</head>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<head>)(.*?)(</head>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # Replace commas within <relation> tags
                    decoded_labels = re.sub(r'(<relation>)(.*?)(</relation>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<relation>)(.*?)(</relation>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # Replace commas within <tail> tags
                    decoded_labels = re.sub(r'(<tail>)(.*?)(</tail>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_labels)
                    decoded_preds = re.sub(r'(<tail>)(.*?)(</tail>)', lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3), decoded_preds)

                    # print("####################################")
                    # print("decoded_labels:",decoded_labels)
                    # print("decoded_preds:",decoded_preds)
                    # print("####################################")
                    # #去掉标识
                    decoded_preds  = data["predict"].replace("<triplet>", "").replace("</triplet>", "").replace("<head>", "").replace("</head>", ",").replace("<relation>", "").replace("</relation>", ",").replace("<tail>", "").replace("</tail>", "") 
                    decoded_labels = data["label"].replace("<triplet>", "").replace("</triplet>", "").replace("<head>", "").replace("</head>", ",").replace("<relation>", "").replace("</relation>", ",").replace("<tail>", "").replace("</tail>", "") 

                    # 处理decoded_labels
                    triples = decoded_labels.split(';')
                    new_triples = []
                    for triple in triples:
                        elements = triple.split(',')
                        new_elements = [elem.split()[-1] if elem.split() else "none" for elem in elements]
                        new_triples.append(','.join(new_elements))
                    decoded_labels = ';'.join(new_triples)

                    # 处理decoded_preds
                    triples = decoded_preds.split(';')
                    new_triples = []
                    for triple in triples:
                        elements = triple.split(',')
                        new_elements = [elem.split()[-1] if elem.split() else "none" for elem in elements]
                        new_triples.append(','.join(new_elements))
                    decoded_preds = ';'.join(new_triples)


                    decoded_labels = {
                        tuple(triple.split(',')) for triple in decoded_labels.split(';')
                    }
                    decoded_preds = {
                        tuple(triple.split(',')) for triple in decoded_preds.split(';')
                    }
                    
                    # 先对三元组内部元素进行排序
                    sorted_labels = {tuple(sorted(triplet)) for triplet in decoded_labels}
                    sorted_preds = {tuple(sorted(triplet)) for triplet in decoded_preds}

                    # 再对三元组进行排序
                    decoded_labels = sorted(sorted_labels)
                    decoded_preds = sorted(sorted_preds)

                    # 将列表转换为集合
                    decoded_labels = set(decoded_labels)
                    decoded_preds = set(decoded_preds)

                    # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
                    TP = len(decoded_labels & decoded_preds)
                    FP = len(decoded_preds - decoded_labels)
                    FN = len(decoded_labels - decoded_preds)

                    total_TP += TP
                    total_FP += FP
                    total_FN += FN

                    num_decoded_labels_len += len(decoded_labels)



            # Calculate precision, recall, and F1 manually
            precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
            recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            accuracy = total_TP / (total_TP + total_FP + total_FN) if total_TP + total_FP + total_FN > 0 else 0

            # Display the results
            print(f'total_TP: {total_TP}')
            print(f'total_FP: {total_FP}')
            print(f'total_FN: {total_FN}')
            print(f'num_decoded_labels_len: {num_decoded_labels_len}')
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1 Score: {f1}')
            print(f'Accuracy: {accuracy}')
            writer.write("\n".join(res))



        # output_prediction_file2 = os.path.join(self.args.output_dir, "generated_predictions2_withnone.jsonl")
        # logger.info(f"Saving prediction results to {output_prediction_file2}")
        # with open(output_prediction_file2, "w", encoding="utf-8") as writer:
        #     res3: List[str] = []
        #     for label, pred in zip(decoded_labels, decoded_preds):
        #         res3.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
        #     writer.write("\n".join(res3))






