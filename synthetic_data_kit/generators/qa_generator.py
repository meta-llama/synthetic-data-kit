# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Create QA Pairs

from typing import Dict, List, Any, Optional, Tuple
import json
import time
import os
from pathlib import Path
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from synthetic_data_kit.models.llm_client import LLMClient
from synthetic_data_kit.utils.text import split_into_chunks
from synthetic_data_kit.utils.llm_processing import parse_qa_pairs, parse_ratings, convert_to_conversation_format
from synthetic_data_kit.utils.config import load_config, get_generation_config, get_curate_config, get_prompt

class QAGenerator:
    def __init__(
        self,
        client: LLMClient,
        config_path: Optional[Path] = None,
        target_language: Optional[str] = "english",
    ):
        """Initialize the QA Generator with an LLM client and optional config"""
        self.client = client

        # Load config
        self.config = load_config(config_path)

        # Get specific configurations
        self.generation_config = get_generation_config(self.config)
        self.curate_config = get_curate_config(self.config)

        # Language control
        self.target_language = (target_language or "english").lower()

    def _language_instruction(self) -> str:
        if self.target_language == "english":
            return "Please respond in English."
        if self.target_language == "arabic":
            return "Please respond in Arabic."
        if self.target_language == "source":
            return "Please respond in the same language as the provided text."
        # Fallback
        return "Please respond in English."
    
    def generate_summary(
        self,
        document_text: str,
        rolling_summary: Optional[bool] = False,
    ) -> str:
        """Generate a summary of the document"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        if verbose:
            print("Generating document summary...")
        
        # Get summary prompt and params from config
        base_prompt = get_prompt(self.config, "summary")
        prompt = f"{base_prompt}\n\n{self._language_instruction()}"
        max_context_length = self.generation_config.get("max_context_length", 8000)
        summary_overlap = self.generation_config.get("summary_overlap", 0)

        if rolling_summary:
            summary_per_chunk = []
            #split text into long chunks for summarization
            chunks = split_into_chunks(document_text,
                                       chunk_size=max_context_length,
                                       overlap=summary_overlap)

            for chunk in chunks:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": chunk}
                ]
                new_summary = self.client.chat_completion(
                    messages, 
                    temperature=0.1  # Use lower temperature for summaries
                )
                summary_per_chunk.append(new_summary)

            summary = " .".join(summary_per_chunk)
            # Summarize again to reduce overall length and redundancy
            summary = self.generate_summary(summary, rolling_summary=False)
        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text[0:max_context_length]},
            ]
            summary = self.client.chat_completion(
                messages,
                temperature=0.1,  # Use lower temperature for summaries
            )
        
        if verbose:
            print(f"Summary generated ({len(summary)} chars)")
        return summary
    
    def generate_qa_pairs(
        self,
        document_text: str,
        summary: Optional[str] = None,
        num_pairs: int = 25,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'

        # Get generation config
        chunk_size = self.generation_config.get("chunk_size", 4000)
        temperature = self.generation_config.get("temperature", 0.7)
        overlap = self.generation_config.get("overlap", 200)
        batch_size = self.generation_config.get("batch_size", 32)

        # Split text into chunks
        chunks = split_into_chunks(document_text, chunk_size=chunk_size, overlap=overlap)
        try:
            import random
            random.shuffle(chunks)
        except Exception:
            pass

        if verbose:
            print("Generating QA pairs...")
            print(f"Document split into {len(chunks)} chunks")
            print(f"Using batch size of {batch_size}")

        all_qa_pairs: List[Dict[str, str]] = []
        num_chunks = max(1, len(chunks))
        pairs_per_chunk = max(1, num_pairs // num_chunks)
        # Over-generate slightly to allow filtering of trivial/meta pairs
        requested_per_chunk = min(max(2, pairs_per_chunk + 2), max(8, pairs_per_chunk))

        # Prompt template and difficulty normalization
        qa_prompt_template = get_prompt(self.config, "qa_generation")
        if difficulty:
            difficulty = difficulty.lower()
            if difficulty not in {"easy", "medium", "advanced"}:
                difficulty = None

        # Prepare message batches
        all_messages: List[List[Dict[str, str]]] = []
        summary_snippet = ""  # ignore summary to avoid bias

        for chunk in chunks:
            lang_instr = self._language_instruction()
            difficulty_rules = {
                "easy": (
                    "Write straightforward, factual questions answerable with a single span from the text. "
                    "Use concrete nouns and exact phrases present in the chunk."
                ),
                "medium": (
                    "Write questions that require combining 2-3 facts from the text. "
                    "Prefer dates, quantities, named entities, and causality explicitly stated."
                ),
                "advanced": (
                    "Write multi-step questions that synthesize multiple details across sentences. "
                    "Cite names, dates, figures, or technical terms directly from the text in the answers."
                ),
            }
            level_text = (
                f"Difficulty: {difficulty}. {difficulty_rules.get(difficulty, '')}"
                if difficulty in {"easy", "medium", "advanced"}
                else ""
            )

            instruction = (
                qa_prompt_template.format(num_pairs=requested_per_chunk, summary=summary_snippet, text="")
                + "\n\nINSTRUCTIONS:\n"
                "- Use ONLY the SOURCE_TEXT between the fences.\n"
                "- Never ask about page numbers, headers/footers, formatting marks, or the difficulty/instructions.\n"
                "- Avoid trivial lists (e.g., enumerate years or page numbers) unless central to a key argument.\n"
                "- Prefer content-focused, specific questions; for advanced, require multi-step reasoning.\n"
                f"{level_text}\n\n"
                "Return JSON array only.\n\n"
                "SOURCE_TEXT (between fences):\n<<BEGIN_SOURCE>>\n"
                f"{chunk}\n"
                "<<END_SOURCE>>"
            )

            messages = [
                {"role": "system", "content": f"You are a careful data creation assistant for LLM training. {lang_instr}"},
                {"role": "user", "content": instruction},
            ]
            all_messages.append(messages)

        print(f"Processing {len(chunks)} chunks to generate QA pairs...")

        # Optional progress bar
        if verbose:
            progress_columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ]
            progress_ctx = Progress(*progress_columns)
            generate_task = progress_ctx.add_task("Generating QA pairs", total=len(chunks))
            progress_ctx.start()
        else:
            progress_ctx = None
            generate_task = None

        for batch_start in range(0, len(chunks), batch_size):
            if len(all_qa_pairs) >= num_pairs:
                if verbose:
                    print(f"Reached target of {num_pairs} pairs. Stopping processing.")
                break

            batch_end = min(batch_start + batch_size, len(chunks))
            batch_messages = all_messages[batch_start:batch_end]
            current_batch_size = len(batch_messages)

            batch_num = batch_start // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size

            if not verbose:
                print(f"Processing batch {batch_num}/{total_batches}...", end="\r")
            else:
                print(f"Processing batch {batch_num}/{total_batches} with {current_batch_size} chunks")

            try:
                batch_responses = self.client.batch_completion(
                    batch_messages, temperature=temperature, batch_size=batch_size
                )

                for j, response in enumerate(batch_responses):
                    if len(all_qa_pairs) >= num_pairs:
                        if verbose:
                            print(f"  Reached target of {num_pairs} pairs. Stopping batch processing.")
                        break

                    chunk_pairs = parse_qa_pairs(response)

                    # Filter trivial/meta pairs (pages, headings, difficulty echoes, parenthesis markers)
                    import re

                    def _is_bad(q: str, a: str) -> bool:
                        qa_l = (q + " " + a).lower()
                        bad_keywords = [
                            "page", "pages", "صفحة", "الصفحة", "الصفحات", "عنوان", "العنوان", "heading",
                            "difficulty", "مستوى الصعوبة", "قوس", "أقواس", "قوسين", "بين القوسين",
                        ]
                        if any(k in qa_l for k in bad_keywords):
                            return True
                        # Numeric-heavy answer heuristic
                        digits = len(re.findall(r"[0-9٠-٩]", a))
                        seps = len(re.findall(r"[\s,،/\-–—]", a))
                        alphas = len(re.findall(r"[A-Za-z\u0600-\u06FF]", a))
                        total = max(1, len(a.strip()))
                        # Loosen for Arabic: allow short dates when some letters present
                        if (digits + seps) / total >= 0.85 and alphas < 2:
                            return True
                        trivial_q_patterns = [
                            r"\byears?\b|السنوات|الأعوام",
                            r"أرقام الصفحات|page numbers",
                            r"عنوان القسم|section title",
                        ]
                        if any(re.search(p, q, flags=re.IGNORECASE) for p in trivial_q_patterns):
                            return True
                        if any(x in qa_l for x in ["advanced", "easy", "medium", "متقدم", "سهل", "متوسط"]):
                            return True
                        return False

                    cleaned_pairs = [
                        p for p in chunk_pairs
                        if isinstance(p, dict) and not _is_bad(p.get("question", ""), p.get("answer", ""))
                    ]

                    remaining_pairs = num_pairs - len(all_qa_pairs)
                    if remaining_pairs > 0:
                        pairs_to_add = cleaned_pairs[:remaining_pairs]
                        all_qa_pairs.extend(pairs_to_add)

                        if verbose:
                            print(f"  Generated {len(pairs_to_add)} pairs (total: {len(all_qa_pairs)}/{num_pairs})")

                    if len(all_qa_pairs) >= num_pairs:
                        break

                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

                if len(all_qa_pairs) >= num_pairs:
                    break

            except Exception as e:
                if verbose:
                    print(f"  Error processing batch {batch_num}: {str(e)}")
                if progress_ctx and generate_task:
                    progress_ctx.update(generate_task, advance=current_batch_size)

        if progress_ctx:
            progress_ctx.stop()

        if not verbose:
            print(" " * 80, end="\r")
            print("Batch processing complete.")

        print(f"Generated {len(all_qa_pairs)} QA pairs total (requested: {num_pairs})")
        return all_qa_pairs
    
    def rate_qa_pairs(self, 
                    qa_pairs: List[Dict[str, str]], 
                    summary: str, 
                    threshold: Optional[float] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Rate and filter QA pairs by quality"""
        verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
        
        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}
        
        # Get threshold from args, then config, then default
        if threshold is None:
            threshold = self.curate_config.get("threshold", 7.0)
            
        if verbose:
            print(f"Evaluating {len(qa_pairs)} pairs...")
        
        # Get rating config
        batch_size = self.curate_config.get("batch_size", 8)
        temperature = self.curate_config.get("temperature", 0.1)
        
        # Get rating prompt template
        rating_prompt_template = get_prompt(self.config, "qa_rating")
        
        # Process in batches
        batches = [qa_pairs[i:i+batch_size] for i in range(0, len(qa_pairs), batch_size)]
        
        rated_pairs = []
        total_score = 0
        
        # Create progress bar
        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        
        with Progress(*progress_columns) as progress:
            rating_task = progress.add_task(f"Rating QA pairs", total=len(batches))
            
            for i, batch in enumerate(batches):
                if verbose:
                    print(f"Rating batch {i+1}/{len(batches)}...")
                batch_json = json.dumps(batch, indent=2)
                
                # Format the rating prompt with pairs
                rating_prompt = rating_prompt_template.format(pairs=batch_json)
                
                messages = [
                    {"role": "system", "content": rating_prompt}
                ]
                
                try:
                    response = self.client.chat_completion(
                        messages, 
                        temperature=temperature
                    )
                    
                    rated_batch = parse_ratings(response)
                    
                    for pair in rated_batch:
                        if "rating" in pair:
                            total_score += pair["rating"]
                            if pair["rating"] >= threshold:
                                rated_pairs.append(pair)
                
                except Exception as e:
                    if verbose:
                        print(f"Error rating batch {i+1}: {str(e)}")
                
                time.sleep(0.5)  # Avoid rate limits
                progress.update(rating_task, advance=1)
        
        # Calculate metrics
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0,
        }

        # Always print summary information, even in non-verbose mode
        print(f"Keeping {len(rated_pairs)} out of {len(qa_pairs)} pairs (threshold: {threshold})")
        print(f"Average score: {metrics['avg_score']}")
        return rated_pairs, metrics
    
    def process_documents(self,
                        documents: List[Dict[str, Any]],
                        num_pairs: int = 25,
                        verbose: bool = False,
                        rolling_summary: Optional[bool] = False,
                        difficulty: Optional[str] = None) -> Dict[str, Any]:
        """Process a list of documents to generate QA pairs without rating.
        Summary generation is intentionally omitted to avoid bias and reduce output size."""
        # Set the verbose environment variable
        if verbose:
            os.environ['SDK_VERBOSE'] = 'true'
        else:
            os.environ['SDK_VERBOSE'] = 'false'

        all_qa_pairs: List[Dict[str, Any]] = []
        full_text = " ".join([doc["text"] for doc in documents])

        # Generate QA pairs grounded in chunk text
        qa_pairs = self.generate_qa_pairs(
            full_text, None, num_pairs=num_pairs, difficulty=difficulty
        )

        all_qa_pairs.extend(qa_pairs)

        # Prepare result - no summary included
        result = {"qa_pairs": all_qa_pairs}

        return result