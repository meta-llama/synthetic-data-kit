# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from pprint import pprint
from typing import Any, Mapping

from torchtune.data import Message
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets import SFTDataset
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class ToolCallMessages(Transform):
    def __init__(self, train_on_input=False):
        self._role_map = {
            "system": "system",
            "human": "user",
            "gpt": "assistant",
            "tool": "ipython",
            "user": "user",  # to avoid key errors
            "assistant": "assistant",  # avoid key errors again, lol
        }
        self.train_on_input = train_on_input

    def __call__(self, sample):
        conversations = sample["cot_conversations"]
        messages = []
        # EOT Logic we agreed on after debating a lot, more notes here: https://github.com/pytorch/torchtune/issues/2405#issuecomment-2670392887
        for i, msg in enumerate(conversations):
            next_is_tool = (
                i < len(conversations) - 1 and conversations[i + 1]["from"] == "tool"
            )
            messages.append(
                Message(
                    role=self._role_map[msg["from"]],
                    content=msg["value"],
                    masked=(
                        False
                        if self.train_on_input
                        else self._role_map[msg["from"]] != "assistant"
                    ),
                    eot=not (
                        msg["from"] == "tool" or (msg["from"] == "gpt" and next_is_tool)
                    ),
                )
            )
        return {"messages": messages}

        return {"messages": messages}


def custom_dataset(
    tokenizer,
    train_on_input=False,
    packed: bool = False,
    **load_dataset_kwargs,
) -> SFTDataset:
    message_transform = ToolCallMessages(train_on_input=train_on_input)

    dataset_path = "/path/to/file/train/"
    arrow_files = [
        os.path.join(dataset_path, x)
        for x in os.listdir(dataset_path)
        if x.endswith(".arrow")
    ]

    ds = SFTDataset(
        source="arrow",
        data_files=arrow_files,
        split="train",
        message_transform=message_transform,
        model_transform=tokenizer,
        **load_dataset_kwargs,
    )

    if packed:
        if tokenizer is None:
            raise ValueError(
                "Needs tokenizer to pack samples."
            )
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
