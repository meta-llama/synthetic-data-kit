# Unlocking Reasoning in Llama 3

This guide demonstrates how to enhance tool-use conversations with Chain of Thought (CoT) reasoning using Synthetic Data Kit. We'll process the ToolACE dataset by adding reasoning traces before tool calls.

It also raises the significant question: Can Llama-3.3 teach itself how to do reasoning effectively?

## High-level Approach:

We use our CLI to unlock reasoning in Llama-3 models:

- We "unlock" reasoning behaviour in Llama-3 model family
- We use Tool-calling as an example implementation
- With the [tool](https://pypi.org/project/synthetic-data-kit/) we enhance [ToolACE dataset](https://huggingface.co/datasets/Team-ACE/ToolACE) to now include CoT
- We measure performance on [BFCL benchmark](https://gorilla.cs.berkeley.edu/leaderboard.html)
- After using FFT for 2 epochs via TorchTune (configs here), we achieve the following results:

![Improvements](./BFCLv3_Accuracy_Comparison.png)

## Details:

Using approx 10k examples of Tool Calling pairs, we introduce new "reasoning" behaviour in Llama-3 model family. Below is an example of a behaviour change with the model:

## Before: 

**Assistant:** <tool>[calc_binomial_probability(n=20, k=5, p=1/6)]</tool>

## After using synthetic dataset:

**Assistant:** To solve this problem, I need to identify the relevant function that can calculate the probability of getting a certain number of successes in a given number of trials. The function 'calc_binomial_probability' seems to fit this purpose, as it calculates the probability of getting k successes in n trials. Now, let's analyze the parameters required by this function: n, k, and p. In this case, n is the number of trials, which is 20 (the number of times the die is rolled). k is the number of successes, which is 5 (the number of times we want to land on a 6). p is the probability of success, which is 1/6 (the probability of landing on a 6 in a single roll).
With these parameters in mind, I can now make the function call. <tool>[calc_binomial_probability(n=20, k=5, p=1/6)]</tool>

This guide shows you how to use the `cot-enhance` feature of the CLI to "unlock" new behaviour in Llama-3 family.

## Why BFCL?

Why do we measure improvements using BFCL?

Tool calling is one of the fundamental tasks at improving LLMs' agentic performance. BFCL is the default benchmark for measuring tool-calling performance. 

It has an independent score of model performance that we can measure and validate.

The benchmark tests approx 4k prompts on various categories giving a thorough understanding of the challenge.

# Steps to preparing the dataset:

Quick overview:

1. `pip install synthetic-data-kit`
2. Follow the [notebook](./cot_enhancement_tutorial.ipynb) to prepare the dataset
3. This will conver the ToolACE with a new prompt and wrap <tool></tool> tags around tool responses
4. Use the `synthetic-data-kit -c cot_tools_config.yaml create /path/to/input/file --type cot-enhance -o /path/to/output/file`
5. After this we will have to clean a few bad examples
6. Now we are ready to perform FFT using this dataset
7. Finally, we re-evaluate numbers on BFCL


## Step 1: Understanding the Dataset

The ToolLlama dataset contains conversations where an assistant makes tool calls to fulfill user requests. The dataset has been formatted with tool calls wrapped in `<tool></tool>` XML tags for easier identification and processing.

Key structure:
- Each conversation starts with a system prompt
- User messages contain questions or requests
- Assistant messages include tool calls and regular responses
- Tool responses contain the results of tool executions

## Step 2: Creating a Custom CoT Configuration

First, we'll create a custom configuration file specifically for enhancing tool conversations with Chain of Thought reasoning:

```yaml
# cot_tools_config.yaml

vllm:
  api_base: "http://localhost:8000/v1"
  model: "unsloth/Meta-Llama-3.3-70B-Instruct"
  max_retries: 3
  retry_delay: 1.0

generation:
  temperature: 0.2   # Lower temperature for more consistent reasoning
  top_p: 0.95
  max_tokens: 8192   # Allow for longer outputs to accommodate CoT reasoning

# The most important part - our custom Chain of Thought prompt
prompts:
  cot_enhancement: |
    You are a highly intelligent AI with an IQ of 170, and your job is to enhance existing conversation examples. Remember to return the entire conversation as is, BUT

    BUT, we will add Chain of Thought and planning to "Assistant" messages whenever they return a tool call.

    Remember, ONLY when an assistant message returns a tool call will we add thinking and reasoning traces before it to add logic. Otherwise, we don't touch the conversation history.

    Remember to return the entire message, but only enhance the assistant messages whenever a tool is called in the conversation by adding thoughts.

    Please keep in mind that we are not modifying anything in the example, nor are we changing what it does. We are only adding CoT every time a tool gets called in the conversation.

    Think out loud and maximize your tokens when adding CoT.

    For example, if you see:
    
    "from": "assistant",
    "value": "<tool>[Some API(param=\"value\")]</tool>"
    
    Change it to:
    
    "from": "assistant",
    "value": "Let me think about this request. I need to gather X information using Tool Y. 
    To do this, I need to set the parameter to 'value' because of reason Z. 
    <tool>[Some API(param=\"value\")]</tool>"
    
    BEGIN WORK NOW. Enhance the assistant's messages with detailed Chain of Thought reasoning before each tool call:
    {conversations}
```

## Step 3: Extracting a Test Example

Let's extract a single conversation example from the dataset to work with:

```bash
# Create a directory for test files if it doesn't exist
mkdir -p test_files

# Extract a single conversation to a JSON file
python -c "
import json
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('toolllama_formatted_dataset')

# Get first example
example = dataset['train'][0]

# Save conversations to file
with open('test_files/conversation_example.json', 'w') as f:
    json.dump({'conversations': example['conversations']}, f, indent=2)

print('Example saved to test_files/conversation_example.json')
"
```

## Step 4: Applying Chain of Thought Enhancement with the CLI

We've extended the Synthetic Data Kit CLI to support enhancing tool-use conversations with Chain of Thought reasoning through a new `cot-enhance` content type. Now, you can use the CLI to enhance your tool-use conversations:

```bash
# Use the create command with type=cot-enhance and our custom config
synthetic-data-kit -c cot_tools_config.yaml create test_files/conversation_example.json \
  --type cot-enhance \
  -o enhanced_output/
```

This will enhance the conversations in the input file by adding Chain of Thought reasoning before each tool call. The output will be saved as `enhanced_output/conversation_example_enhanced.json`.

### How It Works

When you use the `--type cot-enhance` option:

1. The CLI loads the JSON file containing conversations
2. It extracts the conversations from the file
3. For each assistant message containing a tool call (identified by `<tool>` tags), it adds detailed reasoning
4. The enhanced conversations are saved to the output file

This makes it easy to bulk process tool-use conversations to create high-quality training data with explicit reasoning.

### Using the `cot-enhance` Feature in Your Workflow

The `cot-enhance` feature is designed to be easily integrated into your data processing pipeline. Here's how you might use it in practice:

1. **Direct Command Line Usage**:
   ```bash
   # Process a single file
   synthetic-data-kit -c custom_config.yaml create input.json --type cot-enhance -o enhanced_output/
   
   # Process multiple files with a loop
   for file in input_files/*.json; do
     basename=$(basename "$file" .json)
     synthetic-data-kit -c custom_config.yaml create "$file" --type cot-enhance -o "enhanced_output/${basename}_enhanced.json"
   done
   ```

2. **Python Integration**:
   ```python
   from synthetic_data_kit.core.create import process_file
   from pathlib import Path
   
   # Process a batch of files
   input_files = ["file1.json", "file2.json", "file3.json"]
   for input_file in input_files:
       output_path = process_file(
           file_path=input_file,
           output_dir="enhanced_output",
           config_path=Path("custom_config.yaml"),
           content_type="cot-enhance",
           verbose=True
       )
       print(f"Enhanced {input_file} -> {output_path}")
   ```

3. **Expected Input Format**: The input must be a JSON file with one of these formats:
   - A single conversation in the `conversations` field
   - An array of conversation objects, each with a `conversations` field
   - A direct array of conversation messages with `from` and `value` fields

4. **Output Format**: The output preserves the structure of the input but enhances assistant messages containing tool calls by adding Chain of Thought reasoning.

## Running it on the entire dataset:

After following the short guide above, we can use the output from the [Notebook](./cot_enhancement_tutorial.ipynb) and enhance the examples

## Fine-Tuning configs:

Finally, we can prepare our dataset for fine-tuning and use the [TorchTune configs here](./tt_configs/)


Following these on 8xH100 takes about 2-3 hours and we can finally run our new checkpoints on BFCL.

## Troubleshooting and Tips

When working with the CoT enhancement process, you might encounter some challenges:

### VLLM Server Issues

- **Problem**: The enhancement process runs but doesn't add CoT reasoning
- **Solution**: 
  - Ensure your VLLM server is running with `vllm serve your-model --port 8000`
  - Check connectivity with `synthetic-data-kit system-check`
  - Verify your model is capable of following complex prompts (Llama-3-70B-Instruct or similar recommended)

### Prompt Engineering

- **Problem**: The CoT reasoning is too brief or not insightful
- **Solution**:
  - Try adjusting the prompt in your configuration file
  - Experiment with different temperature settings (0.2-0.7 works well)
  - Add specific instructions about reasoning depth or steps to include

### Dataset Format Issues

- **Problem**: The enhancement process fails with JSON errors
- **Solution**:
  - Ensure your input follows one of the expected formats
  - Use the verbose flag (`--verbose`) to see detailed processing information
  - Check that tool calls are properly wrapped in `<tool></tool>` tags
