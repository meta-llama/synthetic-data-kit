# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Output utilities
import re
import json
import os
from typing import List, Dict, Any, Optional

def parse_qa_pairs(text: str) -> List[Dict[str, str]]:
    """Parse QA pairs from LLM output with enhanced error handling.

    Robust to:
    - Markdown code fences (```json ... ```)
    - Arrays wrapped in other text; trailing commas
    - Alternative field names: Arabic equivalents, short keys (q/a), or 'answers' arrays
    - Either array of objects or repeated objects parsed by regex fallback
    """
    verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
    
    if verbose:
        print(f"Parsing response of length {len(text)}")
    
    def _normalize_item(item: Any) -> Optional[Dict[str, str]]:
        """Normalize a single QA dict to {'question','answer'} if possible."""
        if not isinstance(item, dict):
            return None
        # Known key candidates
        q_keys = {
            "question", "Question", "Q", "q",
            "سؤال", "السؤال",
            "prompt", "query",
        }
        a_keys = {
            "answer", "Answer", "A", "a",
            "إجابة", "الاجابة", "الإجابة", "جواب",
            "response", "reply",
        }
        question = None
        answer = None
        # Direct mappings
        for k in item.keys():
            if k in q_keys and question is None:
                question = item.get(k)
            if k in a_keys and answer is None:
                answer = item.get(k)
        # Handle answers list or alt fields
        if answer is None:
            if isinstance(item.get("answers"), list) and item.get("answers"):
                answer = item.get("answers")[0]
            elif "completion" in item:
                answer = item.get("completion")
        if question and isinstance(question, str) and answer and isinstance(answer, str):
            return {"question": question.strip(), "answer": answer.strip()}
        return None

    def _try_load_array(json_text: str) -> Optional[List[Dict[str, str]]]:
        cleaned_text = re.sub(r'(\n\s*|\r\s*)', ' ', json_text)
        cleaned_text = re.sub(r',\s*(\}|\])', r'\1', cleaned_text)
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "qa_pairs" in parsed:
                parsed = parsed["qa_pairs"]
            if isinstance(parsed, list):
                norm = []
                for it in parsed:
                    n = _normalize_item(it)
                    if n:
                        norm.append(n)
                return norm
        except json.JSONDecodeError:
            return None
        return None

    try:
        # Prefer content inside code fences if present
        code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        if code_blocks:
            # Try last block first
            for block in reversed(code_blocks):
                result = _try_load_array(block)
                if result is not None and len(result) > 0:
                    if verbose:
                        print(f"Parsed {len(result)} QA pairs from code fence")
                    return result
        # Fallback: find array brackets in whole text
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            result = _try_load_array(text[start:end])
            if result is not None and len(result) > 0:
                if verbose:
                    print(f"Successfully parsed {len(result)} QA pairs from array")
                return result
    except Exception as e:
        if verbose:
            print(f"Error during JSON extraction: {e}")
    
    # Fallback to regex pattern matching
    if verbose:
        print("Falling back to regex pattern matching")
    qa_pattern = r'"question":\s*"((?:[^"\\]|\\.)*)"\s*,\s*"answer":\s*"((?:[^"\\]|\\.)*)"\s*'
    pairs = []
    
    for match in re.finditer(qa_pattern, text):
        try:
            q = match.group(1).replace('\\"', '"')
            a = match.group(2).replace('\\"', '"')
            pairs.append({"question": q, "answer": a})
        except Exception as e:
            if verbose:
                print(f"Error extracting pair: {e}")

    # If still empty, try to extract dicts with alternative keys and normalize
    if not pairs:
        try:
            obj_matches = re.findall(r'(\{[^\{\}\[\]]+\})', text)
            for om in obj_matches:
                try:
                    obj = json.loads(re.sub(r'\s*\n\s*', ' ', om))
                    norm = _normalize_item(obj)
                    if norm:
                        pairs.append(norm)
                except Exception:
                    continue
        except Exception:
            pass
    
    if verbose:
        if pairs:
            print(f"Extracted {len(pairs)} QA pairs with regex")
        else:
            print("No QA pairs extracted. Check the model output format.")
    
    return pairs

def parse_ratings(text: str, original_items: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """Parse rated items from LLM output
    
    Attempts to parse JSON from LLM response. Will raise an exception if
    parsing fails. Never adds default ratings - either the model returns valid
    ratings or the function will crash.
    
    Args:
        text: LLM response text to parse
        original_items: Original QA pairs (ignored - no defaults used)
    
    Returns:
        List of items with ratings from the LLM
        
    Raises:
        ValueError: If the response cannot be parsed as valid JSON
    """
    verbose = os.environ.get('SDK_VERBOSE', 'false').lower() == 'true'
    
    if verbose:
        print(f"Parsing ratings response of length {len(text)}")
        print(f"Raw response: {repr(text[:500])}")
    
    # The multiple passes are to for edge cases that emerge when using 8B or smaller models for generating synthetic data. This is to make a comprehensive parser for faster protoyping.
    # With 70B or bigger model, `json.load()` should "just work"
    try:
        # Handle the common case of indented JSON with newlines
        # First, remove any markdown or text before/after the JSON
        # Look for standard JSON start/end markers
        json_content = text.strip()
        
        # Try to normalize escape sequences
        json_content = json_content.replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
        
        # Check if we have a JSON object
        if '{' in json_content and '}' in json_content:
            start_idx = json_content.find('{')
            end_idx = json_content.rfind('}') + 1
            json_text = json_content[start_idx:end_idx]
            
            # Clean up the JSON string to handle common issues
            # First, convert newlines to spaces in JSON
            json_text = re.sub(r'\s*\n\s*', ' ', json_text)
            
            # Now, try to parse it
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, dict) and "rating" in parsed:
                    if verbose:
                        print("Successfully parsed single JSON object")
                    return [parsed]
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"JSON parse error for object: {str(e)}")
        
        # Check if we have a JSON array
        if '[' in json_content and ']' in json_content:
            start_idx = json_content.find('[')
            end_idx = json_content.rfind(']') + 1
            json_text = json_content[start_idx:end_idx]
            
            # Clean up the JSON string
            json_text = re.sub(r'\s*\n\s*', ' ', json_text)
            
            try:
                parsed = json.loads(json_text)
                if isinstance(parsed, list):
                    for item in parsed:
                        if not isinstance(item, dict) or "rating" not in item:
                            if verbose:
                                print(f"Array contains invalid item: {item}")
                            return []
                    if verbose:
                        print(f"Successfully parsed {len(parsed)} items in JSON array")
                    return parsed
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"JSON parse error for array: {str(e)}")
    
    except Exception as e:
        if verbose:
            print(f"Error in primary parsing approach: {str(e)}")
    
    # Fallback to more specific methods
    # Method 1: Code block extraction
    try:
        code_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        if code_blocks:
            for block in code_blocks:
                try:
                    # Clean up newlines in the code block
                    clean_block = re.sub(r'\s*\n\s*', ' ', block.strip())
                    parsed = json.loads(clean_block)
                    if isinstance(parsed, dict) and "rating" in parsed:
                        if verbose:
                            print("Successfully parsed from code block (single object)")
                        return [parsed]
                    elif isinstance(parsed, list):
                        valid_items = True
                        for item in parsed:
                            if not isinstance(item, dict) or "rating" not in item:
                                valid_items = False
                                break
                        if valid_items and len(parsed) > 0:
                            if verbose:
                                print(f"Successfully parsed {len(parsed)} items from code block")
                            return parsed
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        if verbose:
            print(f"Error in code block extraction: {str(e)}")
    
    # Method 2: Regex
    try:
        # Look for JSON patterns in the text
        json_patterns = [
            # Single object pattern
            r'(\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"answer"\s*:\s*"[^"]*"\s*,\s*"rating"\s*:\s*\d+(?:\.\d+)?\s*\})',
            # Array pattern
            r'(\[\s*\{\s*"question"\s*:.*"rating"\s*:\s*\d+(?:\.\d+)?\s*\}\s*\])'
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        # Clean up newlines in the match
                        clean_match = re.sub(r'\s*\n\s*', ' ', match)
                        parsed = json.loads(clean_match)
                        if isinstance(parsed, dict) and "rating" in parsed:
                            if verbose:
                                print("Successfully parsed using regex (single object)")
                            return [parsed]
                        elif isinstance(parsed, list) and all("rating" in item for item in parsed):
                            if verbose:
                                print(f"Successfully parsed {len(parsed)} items using regex")
                            return parsed
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        if verbose:
            print(f"Error in regex extraction: {str(e)}")
    
    # Method 3: Try using json5 if available (more lenient parser)
    try:
        import importlib
        json5_spec = importlib.util.find_spec("json5")
        if json5_spec is not None:
            json5 = importlib.import_module("json5")
            try:
                parsed = json5.loads(text)
                if isinstance(parsed, dict) and "rating" in parsed:
                    if verbose:
                        print("Successfully parsed using json5 (single object)")
                    return [parsed]
                elif isinstance(parsed, list) and all("rating" in item for item in parsed):
                    if verbose:
                        print(f"Successfully parsed {len(parsed)} items using json5")
                    return parsed
            except Exception:
                pass
        else:
            if verbose:
                print("json5 not available")
    except Exception:
        if verbose:
            print("json5 check failed")
    
    # If we reach here, try one last aggressive approach
    try:
        # Try line-by-line parsing for each item
        if original_items and len(original_items) > 0:
            # Look for patterns that include both the question and rating
            found_items = []
            for item in original_items:
                # Escape regex special characters in question text
                question_escaped = re.escape(item.get("question", ""))
                pattern = f'.*{question_escaped}.*"rating"\\s*:\\s*(\\d+(?:\\.\\d+)?)'
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    try:
                        rating = float(match.group(1))
                        found_items.append({
                            "question": item.get("question", ""),
                            "answer": item.get("answer", ""),
                            "rating": rating
                        })
                        if verbose:
                            print(f"Found rating {rating} for question: {item.get('question', '')[:30]}...")
                    except:
                        pass
            
            if found_items:
                if verbose:
                    print(f"Extracted {len(found_items)} ratings using pattern matching")
                return found_items
    except Exception as e:
        if verbose:
            print(f"Error in final extraction attempt: {str(e)}")
    
    # If we reach here, we couldn't extract valid JSON
    if verbose:
        print("All parsing methods failed")
    
    # Instead of a generic error message, include part of the response
    error_snippet = text[:100] if len(text) > 100 else text
    raise ValueError(f"Could not parse JSON with ratings: {error_snippet}")

def convert_to_conversation_format(qa_pairs: List[Dict[str, str]], 
                                 system_prompt: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """Convert QA pairs to conversation format"""
    if system_prompt is None:
        system_prompt = "You are a helpful AI assistant that provides accurate, detailed responses."
    
    conversations = []
    for pair in qa_pairs:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]}
        ]
        conversations.append(conversation)
    
    return conversations