#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Friday, April 19th 2024, 11:17:41 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import os
from huggingface_hub import hf_hub_download
import logging
import numpy as np
import torch
import torchaudio
from whisperspeech.vq_stoks import RQBottleneckTransformer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

# llama_model_path = "/home/root/BachVD/model_zoo/llama3.1-s-instruct-2024-08-19-epoch-3/"

def audio_to_sound_tokens(vq_model, audio, target_bandwidth=1.5, device="cuda"):
    array = audio["array"]
    sr = audio["sampling_rate"]

    wav = torch.from_numpy(array).float().unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with torch.no_grad():
        codes = vq_model.encode_audio(wav.to(device))
        codes = codes[0].cpu().tolist()
    
    result = ''.join(f'<|sound_{num:04d}|>' for num in codes)
    return f'<|sound_start|>{result}<|sound_end|>'
def llama3_1_s_model_loader(self):
    if not os.path.exists("whisper-vq-stoks-v3-7lang-fixed.model"):
        hf_hub_download(
            repo_id="jan-hq/WhisperVQ",
            filename="whisper-vq-stoks-v3-7lang-fixed.model",
            local_dir=".",
        )
    self.vq_model = RQBottleneckTransformer.load_model(
        "whisper-vq-stoks-v3-7lang-fixed.model"
    ).to(self.device)
    self.vq_model.ensure_whisper(self.device)

    self.llm_tokenizer           = AutoTokenizer.from_pretrained(self.model_name)
    self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
    self.llm_model               = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to(self.device)
    self.llm_model.eval()

    logging.info(f"Model loaded from {self.model_name}.")


def llama3_1_s_model_generation(self, sample):

    audio_token = audio_to_sound_tokens(self.vq_model, sample['audio'])
    audio_token_with_transcript = f"Transcribe the speech in this audio sample:{audio_token}"
    
    if sample['task_type'] == "ASR":
        batch_input = [audio_token_with_transcript]
        batch_input_templated = []
        for sample in batch_input:    
            messages = [
                {"role": "user", "content": sample},
            ]
            sample_templated = self.llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
            sample_templated += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            # print(sample_templated)
            batch_input_templated.append(sample_templated)

        batch_input = batch_input_templated

        encoded_batch        = self.llm_tokenizer(batch_input, return_tensors="pt").to(self.llm_model.device)
        generated_ids        = self.llm_model.generate(**encoded_batch, max_new_tokens=1024, eos_token_id=[128009, 128001], repetition_penalty=1.15)
        generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
        decoded_batch_output = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(decoded_batch_output)
        return decoded_batch_output
    
    question      = sample['text']

    batch_input = [question]

    if sample['task_type'] == "SI":
        batch_input = [audio_token]

    batch_input_templated = []
    for sample in batch_input:    
        # messages = [
        #     {"role": "user", "content": sample},
        # ]
        # sample_templated = self.llm_tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)
        sample_templated = f"<|start_header_id|>user<|end_header_id|>\n\n{sample}<|start_header_id|>assistant<|end_header_id|>\n\n"
        # print(sample_templated)
        batch_input_templated.append(sample_templated)

    batch_input = batch_input_templated

    encoded_batch        = self.llm_tokenizer(batch_input, return_tensors="pt").to(self.llm_model.device)
    generated_ids        = self.llm_model.generate(**encoded_batch, max_new_tokens=2048, eos_token_id=[128009, 128001])
    generated_ids        = generated_ids[:, encoded_batch.input_ids.shape[-1]:]
    decoded_batch_output = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(decoded_batch_output)
    
    return decoded_batch_output

