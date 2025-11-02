# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import lance
from lance.dataset import write_dataset as lance_write_dataset
from lance.dataset import LanceDataset
import pyarrow as pa
from typing import List, Dict, Any, Optional
import os

def create_lance_dataset(
    data: List[Dict[str, Any]],
    output_path: str,
    schema: Optional[pa.Schema] = None
) -> None:
    """Create a Lance dataset from a list of dictionaries.

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries, where each dictionary represents a row.
        output_path (str): The path to save the Lance dataset.
        schema (Optional[pa.Schema], optional): The PyArrow schema. If not provided, it will be inferred. Defaults to None.
    """
    if not data:
        return

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    table = pa.Table.from_pylist(data, schema=schema)
    lance_write_dataset(table, output_path, mode="overwrite")

def load_lance_dataset(
    dataset_path: str
):
    """Load a Lance dataset.

    Args:
        dataset_path (str): The path to the Lance dataset.

    Returns:
        The loaded Lance dataset, or None if the dataset does not exist.
    """
    if not os.path.exists(dataset_path):
        return None
    return LanceDataset(dataset_path)
