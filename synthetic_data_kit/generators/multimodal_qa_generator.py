# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Multimodal Question Answering Generator

import os
from typing import Optional, List, Dict

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.config import load_config, get_generation_config
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs
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
        """Generate multimodal QA pairs, enforcing exact count when feasible.

        Uses multi-round generation with dynamic per-chunk requests, deduplication,
        and robust parsing to approach the requested number of pairs.
        """
        # Concatenate all text and collect all images (if any)
        all_text = " ".join([doc.get("text", "") for doc in documents])
        images = [doc.get("image", None) for doc in documents]

        # Chunk the text
        chunk_size = self.generation_config.get("chunk_size", 4000)
        overlap = self.generation_config.get("overlap", 200)
        chunks = split_into_chunks(all_text, chunk_size=chunk_size, overlap=overlap)
        print(f"Document split into {len(chunks)} chunks")

        temperature = self.generation_config.get("temperature", 0.7)
        batch_size = self.generation_config.get("batch_size", 32)
        max_rounds = max(1, int(self.generation_config.get("max_rounds", 3)))
        max_per_chunk_cap = int(self.generation_config.get("max_per_chunk_cap", 8))

        all_qa_pairs: List[Dict[str, str]] = []
        seen: set = set()

        import re as _re

        def _norm_q(q: str) -> str:
            q = (q or "").strip().lower()
            q = _re.sub(r"\s+", " ", q)
            q = _re.sub(r"[؟?]+$", "", q).strip()
            return q

        # Helper: build messages for a round given requested_per_chunk
        def _build_round_messages(requested_per_chunk: int) -> List[List[Dict[str, object]]]:
            round_messages: List[List[Dict[str, object]]] = []
            # choose first available image if any
            image = next((img for img in images if img is not None), None)
            for chunk in chunks:
                user_content = [{"type": "text", "text": f"Passage: {chunk}"}]
                if image is not None:
                    image_b64 = base64.b64encode(image).decode("utf-8")
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    })

                system_prompt = (
                    f"You are a helpful assistant. Given the following passage and optional image, generate {requested_per_chunk} high-quality question-answer pairs. "
                    "Return ONLY valid JSON as an array with EXACTLY the requested number of items: "
                    "[{\"question\": \"...\", \"answer\": \"...\"}, ...]. "
                    "Do not include any explanation, markdown, or text outside the JSON."
                )
                if difficulty in {"easy", "medium", "advanced"}:
                    system_prompt += f"\n\nDifficulty: {difficulty}. Generate questions that are {difficulty}-level."
                system_prompt = f"{system_prompt}\n\n{self._language_instruction()}"

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                round_messages.append(messages)
            return round_messages

        total_batches = (len(chunks) + batch_size - 1) // batch_size if len(chunks) else 0
        for round_idx in range(max_rounds):
            if len(all_qa_pairs) >= num_pairs:
                break
            remaining = num_pairs - len(all_qa_pairs)
            num_chunks = max(1, len(chunks))
            per_chunk = max(1, remaining // num_chunks)
            requested_per_chunk = min(max(2, per_chunk + 1), max(max_per_chunk_cap, per_chunk))
            if verbose:
                print(f"[Multimodal] Round {round_idx+1}/{max_rounds}: need {remaining}, requesting ~{requested_per_chunk}/chunk")

            round_messages = _build_round_messages(requested_per_chunk)

            for batch_start in range(0, len(round_messages), batch_size):
                if len(all_qa_pairs) >= num_pairs:
                    break
                batch_end = min(batch_start + batch_size, len(round_messages))
                batch_messages = round_messages[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1
                if not verbose:
                    print(f"Processing batch {batch_num}/{total_batches or 1}...", end="\r")
                else:
                    print(f"Processing batch {batch_num}/{total_batches or 1} (round {round_idx+1})")

                try:
                    batch_responses = self.client.batch_completion(
                        batch_messages,
                        temperature=temperature,
                        batch_size=batch_size,
                    )
                    for resp in batch_responses:
                        if len(all_qa_pairs) >= num_pairs:
                            break
                        # Parse with robust parser to handle extra text/markdown
                        pairs = parse_qa_pairs(resp)
                        cleaned: List[Dict[str, str]] = []
                        for qa in pairs:
                            if not isinstance(qa, dict):
                                continue
                            q = qa.get("question", "")
                            a = qa.get("answer", "")
                            nq = _norm_q(q)
                            if not nq or nq in seen:
                                continue
                            seen.add(nq)
                            cleaned.append({"question": q, "answer": a})

                        # Deduplicate within this response just in case
                        unique_cleaned: List[Dict[str, str]] = []
                        local_seen = set()
                        for p in cleaned:
                            nq = _norm_q(p.get("question", ""))
                            if nq in local_seen:
                                continue
                            local_seen.add(nq)
                            unique_cleaned.append(p)

                        # Add up to remaining
                        remaining_pairs = num_pairs - len(all_qa_pairs)
                        if remaining_pairs > 0 and unique_cleaned:
                            to_add = unique_cleaned[:remaining_pairs]
                            all_qa_pairs.extend(to_add)
                except Exception as e:
                    if verbose:
                        print(f"[Multimodal] Error in batch {batch_num} (round {round_idx+1}): {e}")

        # Trim just in case
        if len(all_qa_pairs) > num_pairs:
            all_qa_pairs = all_qa_pairs[:num_pairs]

        return all_qa_pairs

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