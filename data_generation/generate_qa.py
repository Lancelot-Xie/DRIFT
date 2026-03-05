#!/usr/bin/env python3
import pandas as pd
import json
import sys
import argparse
from tqdm import tqdm
import random
import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_json(text: str) -> dict:
    """Parse JSON from text containing JSON blocks or raw JSON."""
    # Check for JSON code blocks
    if "```json" in text:
        start = text.find("```json")
        end = text.find("```", start + 7)
        if start != -1 and end != -1:
            json_string = text[start + 7: end]
            try:
                return json.loads(json_string)
            except:
                print(f"Error: parsing JSON block failed")
    # Check for generic code blocks
    elif "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if start != -1 and end != -1:
            json_string = text[start + 3: end]
            try:
                return json.loads(json_string)
            except:
                print(f"Error: parsing generic code block failed")
    # Try parsing raw JSON
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_string = text[start: end + 1]
            try:
                return json.loads(json_string)
            except:
                print(f"Error: parsing raw JSON failed")
    
    return {}

def split_context(context, place):
    """将 context 按长度平均分为三段，返回指定段落"""
    length = len(context)
    if length < 3:
        return context  # 太短直接返回
    one_third = length // 3
    if place == "top":
        return context[:one_third]
    elif place == "middle":
        return context[one_third:2*one_third]
    elif place == "bottom":
        return context[2*one_third:]
    else:
        return context

def judge_answer_quality(question, answer, evidence):
    """Judge whether the answer can be inferred from the evidence"""
    client = OpenAI(api_key='EMPTY', base_url="http://localhost:8000/v1")
    engine = 'vllm-Qwen2.5-72B-Instruct'
    
    judge_prompt = """You are a judge evaluating the quality of question-answer pairs. Your task is to determine whether the given answer can be reasonably inferred from the provided evidence.

Please evaluate based on the following criteria:
1. Can the answer be directly supported by the evidence?
2. Is the evidence sufficient to answer the question?
3. Is the answer logically consistent with the evidence?
4. Are there any contradictions between the answer and evidence?

Question: {question}

Evidence: {evidence}

Answer: {answer}

Please respond with only "true" if the answer can be reasonably inferred from the evidence, or "false" if it cannot.

Your judgment:""".format(question=question, evidence=evidence, answer=answer)

    try:
        messages = [
            {"role": "system", "content": "You are a careful and precise judge who evaluates the quality of question-answer pairs."},
            {"role": "user", "content": judge_prompt}
        ]
        result = client.chat.completions.create(
            model=engine,
            messages=messages,
            temperature=0.0,  # Lower temperature for more consistent judging
            max_tokens=10,
            n=1
        )
        response = result.choices[0].message.content.strip().lower()
        return "true" in response
    except Exception as e:
        print(f"Judge LLM调用失败: {e}")
        return False  # If judge fails, assume false to be safe

def ask_llm(context):
    """调用本地vllm（兼容openai sdk）接口生成问题"""
    # 设置本地 vllm 相关参数
    engine = 'vllm-Qwen2.5-72B-Instruct'
    client = OpenAI(api_key='EMPTY', base_url="http://localhost:8000/v1")
    
    max_attempts = 10  # 最大重试次数
    
    for attempt in range(max_attempts):
        try:
            lst = ["top", "middle", "bottom"]
            place = random.choice(lst)
            type_list = ["true/false question", "multiple choice question", "open-ended question"]
            weights = [0.2, 0.4, 0.4]  # Probabilities: 20% for true/false, 40% for each of the others
            question_type = random.choices(type_list, weights=weights, k=1)[0]
            # 新增切分context
            context_cut = split_context(context, place)
            prompt = """Please generate a question that can be answered based on the provided context. The question should be highly relevant to the context, and the answer must be directly inferable from the given information. Avoid asking questions that cannot be answered using the context. The question should be of the type: {question_type}.
Your response should consist of three parts:
1. Question – the generated question. (a string)
2. Answer – the answer, including how it is reasoned out from the relevant information in the context.  (a string)
3. Evidence – the specific part(s) of the original text that support the answer.  (a string)

Attention: Evidence must be quoted directly from the original text and must include all the information needed to answer the question. If some parts of the evidence involve unclear references (e.g., ambiguous subjects), include the related sentences that clarify them, so that the evidence alone is sufficient for answering the question. Ensure that every sentence remains complete, without the use of ellipses.

Your output format should be:
```json
{{
    "question": "<the generated question (include options if the question type is multiple choice)>",
    "answer": "<the corresponding answer, including how it is inferred from the relevant information in the context>",
    "evidence": "<the specific part(s) taken directly from the original text that support the answer>"
}}
```

Context: 
{context}

Your output:
""".format(question_type=question_type, context=context_cut)

            messages = [
                {"role": "system", "content": "You are a helpful assistant who is good at generating questions."},
                {"role": "user", "content": prompt}
            ]
            
            # Generate QA pair
            result = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                n=1
            )
            response = result.choices[0].message.content.strip()
            
            # Retry parsing if failed
            temp_try_time = 0
            while parse_json(response) == {} and temp_try_time <= 20:
                result = client.chat.completions.create(
                    model=engine,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                    n=1
                )
                response = result.choices[0].message.content.strip()
                temp_try_time += 1
            
            response_dict = parse_json(response)
            
            # Check if all three fields are non-empty
            if (response_dict.get("question", "").strip() == "" or 
                response_dict.get("answer", "").strip() == "" or 
                response_dict.get("evidence", "").strip() == ""):
                print(f"Attempt {attempt + 1}: Empty fields detected, retrying...")
                continue
            
            question = response_dict["question"]
            answer = response_dict["answer"]
            evidence = response_dict["evidence"]
            
            # Judge the quality
            if judge_answer_quality(question, answer, evidence):
                return question, answer, evidence
            else:
                print(f"Attempt {attempt + 1}: Judge rejected the QA pair, retrying...")
                continue
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
    
    # If all attempts failed, return empty strings
    print(f"All {max_attempts} attempts failed for this context")
    return "", "", ""


def main():
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from contexts using LLM')
    parser.add_argument('input_parquet', help='Path to input parquet file with contexts')
    parser.add_argument('output_jsonl', help='Path to output JSONL file for saving QA pairs')
    parser.add_argument('--sample_size', type=float, default=1.0, 
                        help='Ratio of samples to process (0.0-1.0, default: 1.0 for 100%)')
    parser.add_argument('--num_workers', type=int, default=256, help='Number of parallel workers (default: 256)')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input_parquet}")
    df = pd.read_parquet(args.input_parquet)
    
    # Validate sample_size is between 0 and 1
    if args.sample_size < 0 or args.sample_size > 1:
        raise ValueError("Sample size must be between 0.0 and 1.0")
    
    # Calculate actual sample size based on ratio
    actual_sample_size = int(len(df) * args.sample_size)
    if actual_sample_size < len(df):
        df = df.sample(n=actual_sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.sample_size:.1%} of data ({actual_sample_size} entries)")
    else:
        print(f"Processing all {len(df)} entries")
    
    if "context" not in df.columns:
        raise ValueError("Input file does not contain 'context' column")

    contexts = df["context"].tolist()
    questions = [None] * len(contexts)
    answers = [None] * len(contexts)
    evidences = [None] * len(contexts)
    
    print(f"Generating Q&A pairs using {args.num_workers} workers")
    print("Note: Each QA pair will be validated and may require multiple attempts")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_idx = {
            executor.submit(ask_llm, context): idx for idx, context in enumerate(contexts)
        }
        for f in tqdm(as_completed(future_to_idx), total=len(contexts), desc="Generating questions"):
            idx = future_to_idx[f]
            try:
                questions[idx], answers[idx], evidences[idx] = f.result()
                if not isinstance(questions[idx], str):
                    questions[idx] = str(questions[idx])
                if not isinstance(answers[idx], str):
                    answers[idx] = str(answers[idx])
                if not isinstance(evidences[idx], str):
                    evidences[idx] = str(evidences[idx])
            except Exception as e:
                print(f"Processing failed at index {idx}: {e}")
                questions[idx], answers[idx], evidences[idx] = "", "", ""

    # Filter out samples where any of question, answer, or evidence are empty strings
    filtered_data = [
        (q, c, a, e) for q, c, a, e in zip(questions, contexts, answers, evidences)
        if not (q.strip() == "" or a.strip() == "" or e.strip() == "")
    ]

    if filtered_data:
        # Create list of dictionaries with renamed keys
        output_data = [
            {
                "Document": c,
                "Question": q,
                "Answer": a,
                "Evidence": e
            }
            for q, c, a, e in filtered_data
        ]
        
        # Save as JSONL
        with open(args.output_jsonl, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Successfully saved {len(output_data)} QA pairs to {args.output_jsonl}")
        print(f"Success rate: {len(output_data)}/{len(contexts)} ({len(output_data)/len(contexts)*100:.1f}%)")
        
        # Generate example JSON file
        output_dir = os.path.dirname(args.output_jsonl)
        example_json_path = os.path.join(output_dir, "qa_example.json")
        
        # Sample 10 examples (or less if fewer are available)
        sample_size = min(10, len(output_data))
        examples = random.sample(output_data, sample_size)
        
        # Remove Document field from examples
        examples_no_doc = [
            {
                "Question": item["Question"],
                "Answer": item["Answer"],
                "Evidence": item["Evidence"]
            }
            for item in examples
        ]
        
        # Write examples to JSON file
        with open(example_json_path, "w", encoding="utf-8") as f:
            json.dump(examples_no_doc, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {sample_size} example Q&A pairs to {example_json_path}")
    else:
        print("All samples resulted in empty question, answer, or evidence. No output file generated.")


if __name__ == "__main__":
    main()