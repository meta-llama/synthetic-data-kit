# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Multimodal Question Answering Generator

import os
from typing import Optional

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config
from synthetic_data_kit.utils.text import split_into_chunks
import math
import base64

class MultimodalQAGenerator:
    """Generates Multimodal Question Answering data (text QA from text+image context)"""
    def __init__(self, client: LLMClient, config_path: Optional[str] = None, target_language: Optional[str] = "english"):
        self.client = client
        self.config = load_config(str(config_path) if config_path else None) if config_path else client.config
        self.generation_config = get_generation_config(self.config)
        self.target_language = (target_language or "english").lower()

    def _language_instruction(self) -> str:
        if self.target_language == "english":
            return "Please respond in English."
        if self.target_language == "arabic":
            return "Please respond in Arabic."
        if self.target_language == "source":
            return "Please respond in the same language as the provided text."
        return "Please respond in English."

    def generate_qa_pairs(self, documents, num_pairs=25, verbose=False, difficulty: Optional[str] = None):
        # Concatenate all text and collect all images (if any)
        all_text = " ".join([doc["text"] for doc in documents])
        images = [doc.get("image", None) for doc in documents]
        # Chunk the text
        chunk_size = self.generation_config.get("chunk_size", 4000)
        overlap = self.generation_config.get("overlap", 200)
        chunks = split_into_chunks(all_text, chunk_size=chunk_size, overlap=overlap)
        print(f"Document split into {len(chunks)} chunks")
        # Distribute num_pairs across chunks
        pairs_per_chunk = max(1, math.ceil(num_pairs / len(chunks)))
        # Prepare all message batches
        all_messages = []
        for i, chunk in enumerate(chunks):
            user_content = []
            user_content.append({"type": "text", "text": f"Passage: {chunk}"})
            image = next((img for img in images if img is not None), None)
            if image is not None:
                image_b64 = base64.b64encode(image).decode("utf-8")
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })
            system_prompt = (
                f"You are a helpful assistant. Given the following passage and image, generate {pairs_per_chunk} high-quality question-answer pairs. "
                "Return ONLY valid JSON as a list: [{\"question\": \"...\", \"answer\": \"...\"}, ...]. "
                "Do not include any explanation, markdown, or text outside the JSON."
            )
            if difficulty in {"easy", "medium", "advanced"}:
                system_prompt += f"\n\nDifficulty: {difficulty}. Generate questions that are {difficulty}-level."
            system_prompt = f"{system_prompt}\n\n{self._language_instruction()}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            all_messages.append(messages)
        # Batch LLM calls
        batch_size = self.generation_config.get("batch_size", 32)
        all_qa_pairs = []
        for batch_start in range(0, len(all_messages), batch_size):
            batch_end = min(batch_start + batch_size, len(all_messages))
            batch_messages = all_messages[batch_start:batch_end]
            batch_responses = self.client.batch_completion(
                batch_messages,
                temperature=self.generation_config.get("temperature", 0.7),
                batch_size=batch_size
            )
            for response in batch_responses:
                import json as _json
                try:
                    pairs = _json.loads(response)
                    if isinstance(pairs, dict):
                        pairs = [pairs]
                    for qa in pairs:
                        question = qa.get("question", "")
                        answer = qa.get("answer", "")
                        all_qa_pairs.append({"question": question, "answer": answer})
                except Exception:
                    pass
            if len(all_qa_pairs) >= num_pairs:
                break
        return all_qa_pairs[:num_pairs]

    def process_dataset(self, documents, output_dir: str, num_examples=None, verbose=False, base_name: str = "multimodal_qa_pairs", difficulty: Optional[str] = None) -> str:
        # documents: list of dicts with 'text' and 'image'
        qa_pairs = self.generate_qa_pairs(documents, num_examples or 25, verbose=verbose, difficulty=difficulty)
        suffix = f"_{difficulty}" if difficulty in {"easy", "medium", "advanced"} else ""
        output_path = os.path.join(output_dir, f"{base_name}{suffix}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            import json
            json.dump({"qa_pairs": qa_pairs}, f, indent=2, ensure_ascii=False)
        if verbose:
            print(f"Saved processed multimodal QA pairs to {output_path}")
        return output_path 