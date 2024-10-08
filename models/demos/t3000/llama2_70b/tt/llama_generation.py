# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ConcatMeshToTensor

from loguru import logger

import copy
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None):
        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device

        # Initial model_config is set in decode mode
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            batch=self.max_batch_size,
            seq_len=1,
        )
        self.model_config = model_config

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            self.mesh_device,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            self.params,
            cache_path=tt_args.cache_path,
            read_cache=False,
            paged_attention_config=paged_attention_config,
        )

        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos)
        else:
            return self.prefill_forward(tokens, start_pos)

    def capture_trace(self, tokens: torch.Tensor, start_pos: int):
        tt_inp, start_pos, rot_mat, attn_mask, cache_idxs_tt = self.tt_model.prepare_inputs(tokens, start_pos)

        # Compile model
        tt_inp = ttnn.to_device(tt_inp, self.mesh_device, memory_config=self.model_config["DRAM_MEMCFG"])
        tt_inp_emb = self.tt_model.tt_embd(tt_inp)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
        rot_mat = ttnn.to_device(rot_mat, self.mesh_device, memory_config=self.model_config["ROT_MAT_MM_IN1_MEMCFG"])
        cache_idxs_tt = ttnn.to_device(cache_idxs_tt, self.mesh_device, memory_config=self.model_config["DRAM_MEMCFG"])
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            cache_idxs=cache_idxs_tt,
        )

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Run TT model
        tt_inp_emb = self.tt_model.tt_embd(tt_inp)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            cache_idxs=cache_idxs_tt,
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return trace_id, tt_inp, rot_mat, cache_idxs_tt, tt_logits

    def delete_trace(self, trace_id):
        ttnn.release_trace(self.mesh_device, trace_id)

    def decode_forward_trace(
        self, tokens: torch.Tensor, start_pos: int, trace_id, tt_inp, rot_mat, cache_idxs_tt, tt_logits
    ):
        self._update_model_config("decode", tokens.shape[0], 1)
        batch = tokens.shape[0]

        # Update preallocated tensors
        (
            updated_tt_inp,
            start_pos,
            updated_rot_mat,
            updated_attn_mask,
            updated_cache_idxs_tt,
        ) = self.tt_model.prepare_inputs(tokens, start_pos)
        ttnn.copy_host_to_device_tensor(updated_tt_inp, tt_inp)
        ttnn.copy_host_to_device_tensor(updated_rot_mat, rot_mat)
        ttnn.copy_host_to_device_tensor(updated_cache_idxs_tt, cache_idxs_tt)

        # Run TT model
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        updated_tt_logits = ttnn.from_device(tt_logits)

        logits = self._process_logits(updated_tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users

        return logits

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, page_table=None):
        self._update_model_config("decode", tokens.shape[0], 1)
        batch = tokens.shape[0]
        tt_inp, start_pos, rot_mat, attn_mask, cache_idxs_tt = self.tt_model.prepare_inputs(tokens, start_pos)
        tt_inp = ttnn.to_device(tt_inp, self.mesh_device, memory_config=self.model_config["DRAM_MEMCFG"])
        tt_inp_emb = self.tt_model.tt_embd(tt_inp)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
        rot_mat = ttnn.to_device(rot_mat, self.mesh_device, memory_config=self.model_config["ROT_MAT_MM_IN1_MEMCFG"])
        cache_idxs_tt = ttnn.to_device(cache_idxs_tt, self.mesh_device, memory_config=self.model_config["DRAM_MEMCFG"])
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            cache_idxs=cache_idxs_tt,
            page_table=page_table,
        )

        # del tt_inp_emb
        # del rot_mat
        # del attn_mask

        logits = self._process_logits(tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users
        # del tt_logits

        return logits

    def prefill_forward_single_user(
        self, tokens: torch.Tensor, start_pos: int, user_id: int, last_token_idx=None, page_table=None
    ):
        batch, seq_len = tokens.shape
        assert batch == 1
        assert start_pos == 0, "start_pos must be 0 for prefill_forward_single_user"
        assert seq_len in [128, 2048, 8 * 1024], f"Only prefill up to 128 or 2048 tokens is supported, got {seq_len}"

        self._update_model_config("prefill", batch, seq_len)

        tt_inp_emb, start_pos, rot_mat, attn_mask, _ = self.tt_model.prepare_inputs(
            tokens, start_pos=start_pos, valid_seq_len=seq_len
        )

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            user_id=user_id,
            last_token_idx=last_token_idx,
            page_table=page_table,
        )

        del tt_inp_emb
        del rot_mat
        del attn_mask

        logits = self._process_logits(tt_logits)
        logits = logits.squeeze(1)
        del tt_logits
        return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int):
        batch, seq_len = tokens.shape
        assert seq_len <= 8 * 1024, f"Only prefill up to 2048 tokens is supported, got {seq_len}"

        prefill_seq_len = 128 if seq_len <= 128 else 2048 if seq_len <= 2048 else 8 * 1024
        self._update_model_config("prefill", batch, prefill_seq_len)

        batch, seq_len = tokens.shape
        last_token_idx = seq_len - 1
        output_logits = torch.zeros(batch, seq_len, self.params.vocab_size)
        # pad tokens to 128 or 2048
        prefill_ids = torch.cat([tokens, torch.zeros(batch, prefill_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            logits = self.prefill_forward_single_user(
                prefill_ids[user_id : user_id + 1], start_pos, user_id, last_token_idx=last_token_idx
            )

            # output_logits[user_id] = logits[:, :seq_len, :]
            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits[:, last_token_idx % 32 : last_token_idx % 32 + 1, :]

        logger.info(f"Finished prefill for all users up to {seq_len} tokens, Starting decode...")

        return output_logits

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.mesh_device, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=3)
        )
        return logits[..., : self.params.vocab_size].float()

    def _update_model_config(self, mode, batch, seq_len):
        if self.tt_model.model_config["LLM_MODE"] != mode:
            logger.info(f"Changing mode to {mode}")
            model_config = get_model_config(
                llama_version=self.llama_version,
                max_batch_size=self.max_batch_size,
                max_context_len=self.max_kv_context_len,
                batch=batch,
                seq_len=seq_len,
            )
            self.tt_model.set_model_config(model_config)
