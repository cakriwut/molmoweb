import dataclasses
import math
from typing import Optional, Any, List, Union, Dict

import numpy as np
from olmo.config import BaseConfig

from olmo import tokenizer
import json
from typing import List, Tuple, Any


ATTEND_ALL_SUBSEGMENT_ID = 10000


def build_subsegment_pos_ids(subsegments):
    position_ids = np.zeros_like(subsegments, dtype=np.int32)
    for subsegment_id in np.unique(subsegments):
        mask = subsegments == subsegment_id
        position_ids[mask] = np.arange(mask.sum(), dtype=np.int32)
    return position_ids


@dataclasses.dataclass
class MessageWeight(BaseConfig):
    """
    How to weight a message
    """
    weight: Optional[float] = None
    """Scale by weights by this much"""

    root_subsegments: Optional[bool] = None
    """Divide by 1/sqrt(n_subsegments)"""

    root_length: Optional[bool] = None
    """Divide by 2/sqrt(n_loss_tokens)"""
    
    action_weight: Optional[float] = None
    """Weight multiplier for action tokens in JSON messages"""

    finegrained_action_weight: Optional[bool] = None
    """If True, use finegrained per-field normalized weighting for action tokens"""

    formatting_weight: float = 0.1
    """Weight for formatting tokens (braces, colons, commas) and keys in finegrained mode"""

    def with_overrides(self, other: Union[None, float, 'MessageWeight']) -> 'MessageWeight':
        if other is None:
            return self
        if isinstance(other, (int, float)):
            return MessageWeight(
                weight=other, root_subsegments=self.root_subsegments, root_length=self.root_length,
                action_weight=self.action_weight, finegrained_action_weight=self.finegrained_action_weight,
                formatting_weight=self.formatting_weight,
            )
        return MessageWeight(
            weight=self.weight if other.weight is None else other.weight,
            root_subsegments=self.root_subsegments if other.root_subsegments is None else other.root_subsegments,
            root_length=self.root_length if other.root_length is None else other.root_length,
            action_weight=self.action_weight if other.action_weight is None else other.action_weight,
            finegrained_action_weight=self.finegrained_action_weight if other.finegrained_action_weight is None else other.finegrained_action_weight,
            formatting_weight=other.formatting_weight,
        )


@dataclasses.dataclass
class TextPreprocessorConfig(BaseConfig):
    max_answer_len: Optional[int] = None
    last_message_loss_only: bool = False
    max_text_tokens: Optional[str] = None
    loss_token_weighting: Optional[str] = None

    def build_text_preprocessor(self, tokenizer, max_seq_len):
        if self.loss_token_weighting == "root_subsegments":
            weighting = MessageWeight(root_subsegments=True)
        elif self.loss_token_weighting == "root_subsegments_root_tokens":
            weighting = MessageWeight(root_subsegments=True, root_length=True)
        elif self.loss_token_weighting is None:
            weighting = MessageWeight()
        else:
            raise NotImplementedError(self.loss_token_weighting)
        return InterleavedTextPreprocessor(
            tokenizer,
            self.max_text_tokens,
            max_seq_len,
            self.last_message_loss_only,
            self.max_answer_len,
            default_message_weight=weighting
        )


@dataclasses.dataclass
class InterleavedTextPreprocessor:
    """
    Build batches from text that is interleaved with tokens from other modalities
    """
    tokenizer: Any = None
    max_text_tokens: Optional[int] = None
    max_sequence_length: Optional[int] = None
    last_message_loss_only: bool = False
    max_answer_len: int = None
    default_message_weight: Optional[MessageWeight] = dataclasses.field(default_factory=MessageWeight)

    def encode_message_with_weights(self, message: str, weight: MessageWeight) -> Tuple[List[int], List[float]]:
        """
        Encode a message and return token IDs with corresponding weight multipliers.
        For JSON messages with actions, separately encode parts and assign appropriate weights.
        """
        # Default: regular encoding with uniform weights
        regular_token_ids = self.tokenizer.encode(message)
        weights = [1.0] * len(regular_token_ids)
        
        if weight.action_weight is not None:
            try:
                data = json.loads(message)
                
                # Check if this is an action-containing message
                if isinstance(data, dict) and "action" in data and isinstance(data["action"], dict):
                    finegrained = bool(weight.finegrained_action_weight)
                    token_ids, weights = self._encode_action_message(data, weight.action_weight, finegrained=finegrained, formatting_weight=weight.formatting_weight)
                    # assert token_ids == regular_token_ids, (
                    #     f"Token ID mismatch!\n"
                    #     f"Separate encoding: {token_ids}\n"
                    #     f"Regular encoding: {regular_token_ids}\n"
                    #     f"Separate decoded: {self.tokenizer.decode(token_ids)}\n"
                    #     f"Regular decoded: {self.tokenizer.decode(regular_token_ids)}\n"
                    #     f"Original message: {message}"
                    # )
                    return token_ids, weights
            except (json.JSONDecodeError, ValueError):
                pass
        
        return regular_token_ids, weights

    def _encode_action_message(self, data: dict, action_weight: float, finegrained: bool = False, formatting_weight: float = 0.1) -> Tuple[List[int], List[float]]:
        """
        Encode a JSON message with action field, applying higher weights to action name and values.
        Reconstructs the JSON structure by encoding each part separately.
        """
        all_token_ids = []
        all_weights = []
        
        # Start with opening brace
        tokens, weights = self._encode_with_weight("{", 1.0)
        all_token_ids.extend(tokens)
        all_weights.extend(weights)
        
        # Process each key-value pair in the dictionary
        keys = list(data.keys())
        for idx, key in enumerate(keys):
            value = data[key]
            
            # Encode the key
            key_json = json.dumps(key)
            tokens, weights = self._encode_with_weight(key_json, 1.0)
            all_token_ids.extend(tokens)
            all_weights.extend(weights)
            
            # Encode the colon and any spacing
            tokens, weights = self._encode_with_weight(":", 1.0)
            all_token_ids.extend(tokens)
            all_weights.extend(weights)
            
            # Encode the value based on the key
            if key == "action" and isinstance(value, dict):
                # Special handling for action dictionary
                tokens, weights = self._encode_action_dict(value, action_weight, finegrained=finegrained, formatting_weight=formatting_weight)
                all_token_ids.extend(tokens)
                all_weights.extend(weights)
            else:
                # Regular value encoding
                value_json = json.dumps(value, ensure_ascii=False)
                tokens, weights = self._encode_with_weight(value_json, 1.0)
                all_token_ids.extend(tokens)
                all_weights.extend(weights)
            
            # Add comma if not last item
            if idx < len(keys) - 1:
                tokens, weights = self._encode_with_weight(",", 1.0)
                all_token_ids.extend(tokens)
                all_weights.extend(weights)
        
        # End with closing brace
        tokens, weights = self._encode_with_weight("}", 1.0)
        all_token_ids.extend(tokens)
        all_weights.extend(weights)
        
        return all_token_ids, all_weights

    def _encode_action_dict(self, action: dict, action_weight: float, finegrained: bool = False, formatting_weight: float = 0.1) -> Tuple[List[int], List[float]]:
        """
        Encode the action dictionary, applying higher weights to action name and argument values.

        In finegrained mode, per-step loss is a weighted sum over fields where each
        field's value is normalized by its own token count:
          L_step = sum_field  action_weight / n_value_tokens * sum(per-token losses)
        Formatting tokens ({, }, :, ,) and keys get formatting_weight so the loss
        focuses on values.
        """
        all_token_ids = []
        all_weights = []

        if finegrained:
            # Opening brace
            tokens, weights = self._encode_with_weight("{", formatting_weight)
            all_token_ids.extend(tokens)
            all_weights.extend(weights)

            action_keys = list(action.keys())
            for idx, key in enumerate(action_keys):
                value = action[key]

                # Encode the key with near-zero weight
                key_json = json.dumps(key)
                tokens, weights = self._encode_with_weight(key_json, formatting_weight)
                all_token_ids.extend(tokens)
                all_weights.extend(weights)

                # Encode colon with near-zero weight
                tokens, weights = self._encode_with_weight(":", formatting_weight)
                all_token_ids.extend(tokens)
                all_weights.extend(weights)

                # Encode value: weight normalized by this field's token count
                value_json = json.dumps(value, ensure_ascii=False)
                value_token_ids = self.tokenizer.encode(value_json)
                n_value_tokens = len(value_token_ids)
                normalized_weight = action_weight / max(n_value_tokens, 1)
                all_token_ids.extend(value_token_ids)
                all_weights.extend([normalized_weight] * n_value_tokens)

                # Add comma if not last item
                if idx < len(action_keys) - 1:
                    tokens, weights = self._encode_with_weight(",", formatting_weight)
                    all_token_ids.extend(tokens)
                    all_weights.extend(weights)

            # Closing brace
            tokens, weights = self._encode_with_weight("}", formatting_weight)
            all_token_ids.extend(tokens)
            all_weights.extend(weights)
        else:
            all_token_ids, all_weights = self._encode_with_weight(json.dumps(action), action_weight)
            
        return all_token_ids, all_weights

    def _encode_with_weight(self, text: str, weight: float) -> Tuple[List[int], List[float]]:
        """
        Encode a text snippet and assign the same weight to all its tokens.
        """
        token_ids = self.tokenizer.encode(text)
        weights = [weight] * len(token_ids)
        return token_ids, weights

    def tokenize_message(self, message_list: List[str], weight, bos=True, add_last_eos=True):
        if bos:
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            text_token_ids = [bos]
            loss_mask = [0.0]
            action_weights = [1.0]
        else:
            text_token_ids = []
            loss_mask = []
            action_weights = []
        
        for msg_ix, message in enumerate(message_list):
            # Use the new encoding method that tracks weights
            message_ids, msg_weights = self.encode_message_with_weights(message, weight)
            
            is_model = msg_ix % 2 == 1
            if is_model and (add_last_eos or msg_ix != len(message_list) - 1):
                message_ids.append(self.tokenizer.eos_token_id)
                msg_weights.append(1.0)

            if is_model and self.max_answer_len:
                message_ids = message_ids[:self.max_answer_len]
                msg_weights = msg_weights[:self.max_answer_len]

            has_loss = is_model and (
                not self.last_message_loss_only or (msg_ix == (len(message_list) - 1)))
            loss_mask += [has_loss] * len(message_ids)
            action_weights.extend(msg_weights)
            text_token_ids += message_ids
        
        text_token_ids = np.array(text_token_ids)
        is_prompt = text_token_ids == self.tokenizer.image_prompt_token_id
        loss_mask = np.array(loss_mask, dtype=np.float32)
        action_weights = np.array(action_weights, dtype=np.float32)
        
        # Apply base weighting
        if weight.root_length:
            if loss_mask.sum() > 0:
                loss_mask *= 2 / np.sqrt(loss_mask.sum())
        if weight.weight is not None:
            loss_mask *= weight.weight
        
        # Apply action token weighting
        loss_mask *= action_weights
        
        return text_token_ids, loss_mask

    def tokenize_message_list(
        self,
        message_list: Union[List[str], List[List[str]]],
        n_mm_tokens: int,
        num_images: int = 1,
        weights: List[MessageWeight] = None,
    ):
        """Handle multi-annotation data where we have many annotations for one multi-modal input"""
        assert len(message_list) > 0, "Given empty messages"
        # Multi-annotation data where we have many annotations for one multi-modal input
        before_ids = []
        after_ids = []
        before_losses = []
        after_losses = []
        before_subsegments = []
        after_subsegments = []
        n_tokens = 0
        for message_set_ix, message_tuple in enumerate(message_list):
            add_bos = message_set_ix == 0
            weight = weights[message_set_ix]
            text_ids, text_loss = self.tokenize_message(
                message_tuple, weight, bos=add_bos, add_last_eos=False)

            is_prompt = text_ids == self.tokenizer.image_prompt_token_id
            n_prompts = is_prompt.sum()
            if n_prompts == 1:
                image_idx = np.argmax(is_prompt)
                s, e = image_idx, image_idx+1
            elif n_prompts == 0 and add_bos:
                s, e = 1, 1
            elif n_prompts == 0:
                s, e = 0, 0
            else:
                raise NotImplementedError("Multi-message with multi images")

            if text_loss[e] != 0:
                raise ValueError("Must have a non-loss token after MM data")
            if self.max_sequence_length and message_set_ix != 0:
                if (n_mm_tokens + n_tokens + np.argmax(text_loss != 0)) >= self.max_sequence_length:
                    # This example would get no loss tokens anyway
                    break
            n_tokens += len(text_ids)
            before_ids.append(text_ids[:s])
            after_ids.append(text_ids[e:])
            before_losses.append(text_loss[:s])
            after_losses.append(text_loss[e:])
            before_subsegments.append(np.full(s, ATTEND_ALL_SUBSEGMENT_ID, dtype=np.int32))
            after_subsegments.append(np.full(len(text_ids[e:]), message_set_ix, dtype=np.int32))
            if self.max_text_tokens and (n_tokens >= self.max_text_tokens):
                break

        # Account for root_subsegments, we do this here so it does not account for
        # messages that were truncated
        for ix, weight in enumerate(weights[:len(before_ids)]):
            if weight.root_subsegments:
                before_losses[ix] /= math.sqrt(len(before_ids))
                after_losses[ix] /= math.sqrt(len(before_ids))

        text_token_ids = np.concatenate([
            np.concatenate(before_ids),
            [self.tokenizer.image_prompt_token_id] * num_images,
            np.concatenate(after_ids),
            [self.tokenizer.eos_token_id],
        ])
        text_subsegments = np.concatenate([
            np.concatenate(before_subsegments),
            [ATTEND_ALL_SUBSEGMENT_ID] * num_images,
            np.concatenate(after_subsegments),
            after_subsegments[-1][-1:]  # for EOS
        ])
        text_loss_masks = np.concatenate([
            np.concatenate(before_losses),
            [0] * num_images,
            np.concatenate(after_losses),
            [0]  # for EOS
        ])
        return text_token_ids, text_loss_masks, text_subsegments

    def tokenize_and_interleave(
        self,
        message_list: Union[List[str], List[List[str]]],
        multi_model_tokens: List[np.ndarray],
        multi_model_pos_ids: Optional[List[np.ndarray]]=None,
        weight: Optional[float]=None
    ) -> Dict[str, np.ndarray]:
        """
        Build a batch by interleaving the text tokens from tokenizing `message_list` and the
        multi-modal tokens from `multi_model_tokens`

        `tokenizer.IMAGE_PROMPT` is used to show where the MM tokens should be inserted, if it is
        not present the MM tokens are inserted right after BOS

        If `message_list` is a list of lists, the batch is assumed to contain multiply-annotated
        MM data. The batch will include tokens from all messages but the MM tokens only once, and
        `subsegment_id` will indicate how to cross-attend between the tokens. Attending between
        tokens before the MM tokens will be allowed, but attending between tokens after the MM
        tokens will not.
        """
        if isinstance(message_list[0], str):
            weight = self.default_message_weight.with_overrides(weight)
            text_token_ids, text_loss_masks = self.tokenize_message(message_list, weight)
            text_subsegments = None
            for_inference = len(message_list) % 2 == 1
        else:
            if not isinstance(weight, (list, tuple)):
                weight = [self.default_message_weight.with_overrides(weight)] * len(message_list)
            else:
                weight = [self.default_message_weight.with_overrides(x) for x in weight]
            if len(message_list) == 1:
                text_token_ids, text_loss_masks = self.tokenize_message(message_list[0], weight[0])
                text_subsegments = None
                for_inference = len(message_list[0]) % 2 == 1
            else:
                text_token_ids, text_loss_masks, text_subsegments = self.tokenize_message_list(
                    message_list, sum(len(x) for x in multi_model_tokens),
                    0 if multi_model_tokens is None else len(multi_model_tokens),
                    weight
                )
                for_inference = False

        if len(multi_model_tokens) > 0:
            mm_idx = np.argwhere(text_token_ids == self.tokenizer.image_prompt_token_id)
            if len(mm_idx) == 0:
                if multi_model_tokens is not None:
                    # Assume mm data should go right after BOS
                    mm_idx = [1] * len(multi_model_tokens)
            else:
                mm_idx = mm_idx[:, 0]
        else:
            mm_idx = []

        mm_tokens = []
        mm_loss_masks = []
        mm_subsegments = None if text_subsegments is None else []
        mm_position_ids = []
        on = 0
        on_pos = 0
        for i, token_ix in enumerate(mm_idx):
            mm_tokens.append(text_token_ids[on:token_ix])
            mm_loss_masks.append(text_loss_masks[on:token_ix])
            if text_subsegments is not None:
                mm_subsegments.append(text_subsegments[on:token_ix])
            if multi_model_pos_ids is not None:
                assert len(multi_model_tokens[-1]) == len(multi_model_pos_ids[-1])
                mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
                on_pos += len(mm_tokens[-1])

            vision_tokens = multi_model_tokens[i]
            mm_tokens.append(vision_tokens)
            mm_loss_masks.append(np.zeros_like(vision_tokens))
            if text_subsegments is not None:
                mm_subsegments.append(np.full([len(vision_tokens)], text_subsegments[token_ix]))
            if multi_model_pos_ids is not None:
                mm_position_ids.append(multi_model_pos_ids[i] + on_pos)
                on_pos += multi_model_pos_ids[i].max() + 1
            if text_token_ids[token_ix] == self.tokenizer.image_prompt_token_id:
                on = token_ix + 1  # Skip over the image prompt token
            else:
                on = token_ix

        mm_tokens.append(text_token_ids[on:])
        mm_loss_masks.append(text_loss_masks[on:])
        if text_subsegments is not None:
            mm_subsegments.append(text_subsegments[on:])
            n_pre_mm_tokens = sum(len(x) for x in mm_tokens[:-1])
            if not mm_position_ids:
                mm_position_ids = [np.arange(n_pre_mm_tokens)]
                on_pos = n_pre_mm_tokens
            mm_position_ids.append(on_pos + build_subsegment_pos_ids(text_subsegments[on:]))
        elif mm_position_ids:
            mm_position_ids.append(np.arange(on_pos, on_pos+len(mm_tokens[-1])))
        else:
            mm_position_ids = [np.arange(0, sum(len(x) for x in mm_tokens))]

        mm_tokens = np.concatenate(mm_tokens)
        mm_loss_masks = np.concatenate(mm_loss_masks)
        mm_position_ids = np.concatenate(mm_position_ids)
        if mm_subsegments is not None:
            mm_subsegments = np.concatenate(mm_subsegments)

        target_tokens = mm_tokens

        if not for_inference:
            target_tokens = mm_tokens[1:]
            input_tokens = mm_tokens[:-1]
            mm_loss_masks = mm_loss_masks[1:]
            if mm_subsegments is not None:
                # The targets for subsegments in the middle need to end with EOS,
                # currently they end with whatever starts the next segment
                mm_subsegments = mm_subsegments[:-1]
                target_tokens = np.copy(target_tokens)
                for i in range(len(message_list)):
                    subsegment_mask = mm_subsegments == i
                    if not np.any(mm_subsegments == i):
                        assert (self.max_text_tokens or self.max_sequence_length) and i != 0
                        # Message skipped due hitting `self.max_text_tokens`
                        break
                    segment_end = np.argwhere(mm_subsegments == i)[-1, 0]
                    target_tokens[segment_end] = self.tokenizer.eos_token_id
                    assert mm_subsegments[segment_end-1] == i
                    assert mm_loss_masks[segment_end-1] != 0
                    mm_loss_masks[segment_end] = mm_loss_masks[segment_end-1]

            if mm_loss_masks[-1] == 0:
                raise RuntimeError("EOS should not be masked")
            mm_position_ids = mm_position_ids[:-1]
        else:
            # Presumably doing inference, but give a dummy target anyway for consistency
            assert mm_tokens[-1] != self.tokenizer.eos_token_id
            input_tokens = mm_tokens
            target_tokens = np.pad(mm_tokens[1:], [0, 1], constant_values=0)

        out = {
            "input_tokens": input_tokens,
            "target_tokens": target_tokens,
            "loss_masks": mm_loss_masks,
            "position_ids": mm_position_ids,
        }
        if mm_subsegments is not None:
            out["subsegment_ids"] = mm_subsegments

        # Some sanity checks
        if not all(len(v) == len(input_tokens) for v in out.values()):
            raise RuntimeError("Length mismatch")
        special_tokens = np.array([
            self.tokenizer.image_end_token_id,
            self.tokenizer.image_start_token_id,
            self.tokenizer.image_col_token_id,
            self.tokenizer.image_patch_token_id,
            self.tokenizer.image_low_res_token_id,
        ])[None, :]
        if np.any(target_tokens[mm_loss_masks != 0][:, None] == special_tokens):
            raise RuntimeError("A special token had a loss")
        return out