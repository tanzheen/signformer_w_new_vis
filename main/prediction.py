#!/usr/bin/env python
import torch

torch.backends.cudnn.deterministic = True

import logging
import numpy as np
import pickle as pickle
import time
import torch.nn as nn

from typing import List
from torch.utils.data import DataLoader
from loss import XentLoss
from helpers import (
    bpe_postprocess,
    load_config,
    get_latest_checkpoint,
    load_checkpoint,
)
from metrics import bleu, chrf, rouge, wer_list
from model import build_model, SignModel
from data import load_data
from vocabulary import PAD_TOKEN, SIL_TOKEN
from tqdm import tqdm

# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model_dir: str,
    steps: int,
    model: SignModel,
    val_dataloader: DataLoader,
    do_translation: bool,
    translation_loss_function: torch.nn.Module,
    translation_loss_weight: int,
    translation_max_output_length: int,
    level: str,
    translation_beam_size: int = 1,
    translation_beam_alpha: int = -1,
    epoch_no: int = 0):
    
    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_ref_texts = []
        all_txt_outputs = []
        all_attention_scores = []
        total_translation_loss = 0

        # Create progress bar
        progress_bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Training epoch {epoch_no+1}",
            unit="batch",
            leave=True
        )

        for valid_batch in progress_bar:
            # Add debug prints before prediction
            ##print("Input batch shape:", valid_batch['txt_input'].shape)
            
            batch_translation_loss = model.get_loss_for_batch(
                batch=valid_batch,
                translation_loss_function=translation_loss_function,
                translation_loss_weight=translation_loss_weight)
    

            if do_translation:
                total_translation_loss += batch_translation_loss

            batch_txt_predictions, batch_attention_scores = model.run_batch(
                batch=valid_batch,
                translation_beam_size=translation_beam_size if do_translation else None,
                translation_beam_alpha=translation_beam_alpha,
                translation_max_output_length=translation_max_output_length
            )

            # Add debug prints after prediction
            ##print("Raw predictions shape:", [p.shape for p in batch_txt_predictions])
            ##print("First prediction tokens:", batch_txt_predictions[0])

            if do_translation:
                all_txt_outputs.extend(batch_txt_predictions)
                all_ref_texts.extend(valid_batch['txt_input'])
                ##print(f"valid_batch['txt_input']: {valid_batch['txt_input']}")
                ##print(f"valid_batch['txt_input'] shape: {valid_batch['txt_input'].shape}")
                ##print(f"batch_txt_predictions: {batch_txt_predictions}")
                ##print(f"batch_txt_predictions shape: {batch_txt_predictions.shape}")

            all_attention_scores.extend(
                batch_attention_scores
                if batch_attention_scores is not None
                else []
            )

        if do_translation:
            assert len(all_txt_outputs) == len(all_ref_texts)
            if (
                translation_loss_function is not None
                and translation_loss_weight != 0
            ):
                # total validation translation loss
                valid_translation_loss = total_translation_loss

            else:
                valid_translation_loss = -1

            # Add debug prints before decoding
            ##print("Number of predictions:", len(all_txt_outputs))
            ##print("Sample prediction before decoding:", all_txt_outputs[0])
            
            decoded_txt = model.txt_vocab.arrays_to_sentences(arrays=all_txt_outputs)
            decoded_ref = model.txt_vocab.arrays_to_sentences(arrays=all_ref_texts)
            
            # Add debug #print after decoding
            ##print("Sample decoded prediction:", decoded_txt[0])
            ##print("Sample reference:", decoded_ref[0])
            
            # evaluate with metric on full dataset
            join_char = " " if level in ["word", "bpe"] else ""
            # Construct text sequences for metrics
            txt_ref = [join_char.join(t) for t in decoded_ref]
            txt_hyp = [join_char.join(t) for t in decoded_txt]
            # post-process
            if level == "bpe":
                txt_ref = [bpe_postprocess(v) for v in txt_ref]
                txt_hyp = [bpe_postprocess(v) for v in txt_hyp]
            assert len(txt_ref) == len(txt_hyp)
            # store_outputs(model_dir, steps, "dev.hyp.txt", valid_batch['txt_input'], txt_hyp)
            # store_outputs(model_dir, steps, "references.dev.txt", valid_batch['txt_input'], txt_ref)

            # TXT Metrics
            txt_bleu = bleu(references=txt_ref, hypotheses=txt_hyp)
            txt_chrf = chrf(references=txt_ref, hypotheses=txt_hyp)
            txt_rouge = rouge(references=txt_ref, hypotheses=txt_hyp)

        valid_scores = {}
        if do_translation:
            valid_scores["bleu"] = txt_bleu["bleu4"]
            valid_scores["bleu_scores"] = txt_bleu
            valid_scores["chrf"] = txt_chrf
            valid_scores["rouge"] = txt_rouge

    results = {
        "valid_scores": valid_scores,
        "all_attention_scores": all_attention_scores,
    }
    if do_translation:
        results["valid_translation_loss"] = valid_translation_loss
        results["decoded_txt"] = decoded_txt
        results["txt_ref"] = txt_ref
        results["txt_hyp"] = txt_hyp

    return results


# def store_outputs(model_dir: str, steps: int, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
#     ) -> None:
#         """
#         Write current validation outputs to file in `self.model_dir.`

#         :param hypotheses: list of strings
#         """
#         if sub_folder:
#             out_folder = os.path.join(model_dir, sub_folder)
#             if not os.path.exists(out_folder):
#                 os.makedirs(out_folder)
#             current_valid_output_file = "{}/{}.{}".format(out_folder, steps, tag)
#         else:
#             out_folder = model_dir
#             current_valid_output_file = "{}/{}".format(out_folder, tag)

#         with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
#             for seq, hyp in zip(sequence_ids, hypotheses):
#                 opened_file.write("{}|{}\n".format(seq, hyp))

# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file, ckpt: str, output_path: str = None, logger: logging.Logger = None
) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                "No checkpoint found in directory {}.".format(model_dir)
            )

    batch_size = cfg["training"]["batch_size"]
    batch_type = cfg["training"].get("batch_type", "sentence")
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    dataset_version = cfg["data"].get("version", "phoenix_2014_trans")
    translation_max_output_length = cfg["training"].get(
        "translation_max_output_length", None
    )

    # load the data
    _, dev_data, test_data, gls_vocab, txt_vocab = load_data(data_cfg=cfg["data"])

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    do_recognition = cfg["training"].get("recognition_loss_weight", 1.0) > 0.0
    do_translation = cfg["training"].get("translation_loss_weight", 1.0) > 0.0
    multimodal = cfg["data"].get("multimodal", 1.0) > 0.0
    model = build_model(
        cfg=cfg["model"],
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        do_recognition=do_recognition,
        do_translation=do_translation,
        multimodal=multimodal,
    )
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # Data Augmentation Parameters
    frame_subsampling_ratio = cfg["data"].get("frame_subsampling_ratio", None)
    # Note (Cihan): we are not using 'random_frame_subsampling' and
    #   'random_frame_masking_ratio' in testing as they are just for training.

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        recognition_beam_sizes = cfg["testing"].get("recognition_beam_sizes", [1])
        translation_beam_sizes = cfg["testing"].get("translation_beam_sizes", [1])
        translation_beam_alphas = cfg["testing"].get("translation_beam_alphas", [-1])
    else:
        recognition_beam_sizes = [1]
        translation_beam_sizes = [1]
        translation_beam_alphas = [-1]

    if "testing" in cfg.keys():
        max_recognition_beam_size = cfg["testing"].get(
            "max_recognition_beam_size", None
        )
        if max_recognition_beam_size is not None:
            recognition_beam_sizes = list(range(1, max_recognition_beam_size + 1))

    if do_recognition:
        recognition_loss_function = torch.nn.CTCLoss(
            blank=model.gls_vocab.stoi[SIL_TOKEN], zero_infinity=True
        )
        if use_cuda:
            recognition_loss_function.cuda()
    if do_translation:
        translation_loss_function = XentLoss(
            pad_index=txt_vocab.stoi[PAD_TOKEN], smoothing=0.0
        )
        if use_cuda:
            translation_loss_function.cuda()

    # NOTE (Cihan): Currently Hardcoded to be 0 for TensorFlow decoding
    assert model.gls_vocab.stoi[SIL_TOKEN] == 0

    if do_recognition:
        # Dev Recognition CTC Beam Search Results
        dev_recognition_results = {}
        dev_best_wer_score = float("inf")
        dev_best_recognition_beam_size = 1
        for rbw in recognition_beam_sizes:
            logger.info("-" * 60)
            valid_start_time = time.time()
            logger.info("[DEV] partition [RECOGNITION] experiment [BW]: %d", rbw)
            dev_recognition_results[rbw] = validate_on_data(
                model=model,
                data=dev_data,
                batch_size=batch_size,
                use_cuda=use_cuda,
                batch_type=batch_type,
                dataset_version=dataset_version,
                sgn_dim=sum(cfg["data"]["feature_size"])
                if isinstance(cfg["data"]["feature_size"], list)
                else cfg["data"]["feature_size"],
                txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                # Recognition Parameters
                do_recognition=do_recognition,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=1,
                recognition_beam_size=rbw,
                # Translation Parameters
                do_translation=do_translation,
                translation_loss_function=translation_loss_function
                if do_translation
                else None,
                translation_loss_weight=1 if do_translation else None,
                translation_max_output_length=translation_max_output_length
                if do_translation
                else None,
                level=level if do_translation else None,
                translation_beam_size=1 if do_translation else None,
                translation_beam_alpha=-1 if do_translation else None,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            logger.info("finished in %.4fs ", time.time() - valid_start_time)
            if dev_recognition_results[rbw]["valid_scores"]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["valid_scores"]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                logger.info("*" * 60)
                logger.info(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: %d\n\t"
                    "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)",
                    dev_best_recognition_beam_size,
                    dev_best_recognition_result["valid_scores"]["wer"],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "del_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "ins_rate"
                    ],
                    dev_best_recognition_result["valid_scores"]["wer_scores"][
                        "sub_rate"
                    ],
                )
                logger.info("*" * 60)

    if do_translation:
        logger.info("=" * 60)
        dev_translation_results = {}
        dev_best_bleu_score = float("-inf")
        dev_best_translation_beam_size = 1
        dev_best_translation_alpha = 1
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                dev_translation_results[tbw][ta] = validate_on_data(
                    model=model,
                    data=dev_data,
                    batch_size=batch_size,
                    use_cuda=use_cuda,
                    level=level,
                    sgn_dim=sum(cfg["data"]["feature_size"])
                    if isinstance(cfg["data"]["feature_size"], list)
                    else cfg["data"]["feature_size"],
                    batch_type=batch_type,
                    dataset_version=dataset_version,
                    do_recognition=do_recognition,
                    recognition_loss_function=recognition_loss_function
                    if do_recognition
                    else None,
                    recognition_loss_weight=1 if do_recognition else None,
                    recognition_beam_size=1 if do_recognition else None,
                    do_translation=do_translation,
                    translation_loss_function=translation_loss_function,
                    translation_loss_weight=1,
                    translation_max_output_length=translation_max_output_length,
                    txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
                    translation_beam_size=tbw,
                    translation_beam_alpha=ta,
                    frame_subsampling_ratio=frame_subsampling_ratio,
                )

                if (
                    dev_translation_results[tbw][ta]["valid_scores"]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta][
                        "valid_scores"
                    ]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    logger.info(
                        "[DEV] partition [Translation] results:\n\t"
                        "New Best Translation Beam Size: %d and Alpha: %d\n\t"
                        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
                        "CHRF %.2f\t"
                        "ROUGE %.2f",
                        dev_best_translation_beam_size,
                        dev_best_translation_alpha,
                        dev_best_translation_result["valid_scores"]["bleu"],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu1"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu2"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu3"
                        ],
                        dev_best_translation_result["valid_scores"]["bleu_scores"][
                            "bleu4"
                        ],
                        dev_best_translation_result["valid_scores"]["chrf"],
                        dev_best_translation_result["valid_scores"]["rouge"],
                    )
                    logger.info("-" * 60)

    logger.info("*" * 60)
    logger.info(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        dev_best_recognition_result["valid_scores"]["wer"] if do_recognition else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        dev_best_recognition_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        dev_best_translation_result["valid_scores"]["bleu"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        dev_best_translation_result["valid_scores"]["chrf"] if do_translation else -1,
        dev_best_translation_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    test_best_result = validate_on_data(
        model=model,
        data=test_data,
        batch_size=batch_size,
        use_cuda=use_cuda,
        batch_type=batch_type,
        dataset_version=dataset_version,
        sgn_dim=sum(cfg["data"]["feature_size"])
        if isinstance(cfg["data"]["feature_size"], list)
        else cfg["data"]["feature_size"],
        txt_pad_index=txt_vocab.stoi[PAD_TOKEN],
        do_recognition=do_recognition,
        recognition_loss_function=recognition_loss_function if do_recognition else None,
        recognition_loss_weight=1 if do_recognition else None,
        recognition_beam_size=dev_best_recognition_beam_size
        if do_recognition
        else None,
        do_translation=do_translation,
        translation_loss_function=translation_loss_function if do_translation else None,
        translation_loss_weight=1 if do_translation else None,
        translation_max_output_length=translation_max_output_length
        if do_translation
        else None,
        level=level if do_translation else None,
        translation_beam_size=dev_best_translation_beam_size
        if do_translation
        else None,
        translation_beam_alpha=dev_best_translation_alpha if do_translation else None,
        frame_subsampling_ratio=frame_subsampling_ratio,
    )

    logger.info(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: %d\n\t"
        "Best Translation Beam Size: %d and Alpha: %d\n\t"
        "WER %3.2f\t(DEL: %3.2f,\tINS: %3.2f,\tSUB: %3.2f)\n\t"
        "BLEU-4 %.2f\t(BLEU-1: %.2f,\tBLEU-2: %.2f,\tBLEU-3: %.2f,\tBLEU-4: %.2f)\n\t"
        "CHRF %.2f\t"
        "ROUGE %.2f",
        dev_best_recognition_beam_size if do_recognition else -1,
        dev_best_translation_beam_size if do_translation else -1,
        dev_best_translation_alpha if do_translation else -1,
        test_best_result["valid_scores"]["wer"] if do_recognition else -1,
        test_best_result["valid_scores"]["wer_scores"]["del_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["ins_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["wer_scores"]["sub_rate"]
        if do_recognition
        else -1,
        test_best_result["valid_scores"]["bleu"] if do_translation else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu1"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu2"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu3"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["bleu_scores"]["bleu4"]
        if do_translation
        else -1,
        test_best_result["valid_scores"]["chrf"] if do_translation else -1,
        test_best_result["valid_scores"]["rouge"] if do_translation else -1,
    )
    logger.info("*" * 60)

    def _write_to_file(file_path: str, sequence_ids: List[str], hypotheses: List[str]):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_gls_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_recognition_result["gls_hyp"],
            )
            test_gls_output_path_set = "{}.BW_{:03d}.{}.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_gls_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["gls_hyp"],
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [s for s in dev_data.sequence],
                dev_best_translation_result["txt_hyp"],
            )
            _write_to_file(
                test_txt_output_path_set,
                [s for s in test_data.sequence],
                test_best_result["txt_hyp"],
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)