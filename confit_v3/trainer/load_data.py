
import argparse
import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from openai import OpenAI
import anthropic
from pydantic import BaseModel
import requests
import re
import diskcache
import itertools

def load_all_resume_texts(file_path: str) -> Dict[str, str]:
    try:
        resume_df = pd.read_csv(file_path)
        resume_texts_dict = pd.Series(resume_df['resume_text'].fillna('').values, 
                                      index=resume_df['user_id'].astype(str)).to_dict()
        print(f"successfully loaded {len(resume_texts_dict)} resume texts from {file_path}")
        return resume_texts_dict
    except Exception as e:
        raise ValueError(f"Error loading all resume texts: {str(e)}")


def load_job_descriptions(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} job descriptions from {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading job descriptions: {str(e)}")

def load_labels(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            labels = json.load(f)
        print(f"Loaded labels for {len(labels)} job descriptions from {file_path}")
        return labels
    except Exception as e:
        raise ValueError(f"Error loading labels: {str(e)}")

def load_rank_resume(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            rank_resume = json.load(f)
        print(f"Loaded rank resume data for {len(rank_resume)} job descriptions")
        return rank_resume
    except Exception as e:
        raise ValueError(f"Error loading rank resume data: {str(e)}")

def load_all_labels_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded all labels data: {len(df)} entries")
        return df
    except Exception as e:
        raise ValueError(f"Error loading all labels data: {str(e)}")

def build_all_labels_dict(rank_resume: Dict, all_labels_df: pd.DataFrame) -> Dict:
    all_labels_dict = {}
    for _, row in all_labels_df.iterrows():
        job_id = row['jd_no']
        user_id = row['user_id']
        if job_id not in rank_resume:
            continue
            
        if job_id in all_labels_dict:
            all_labels_dict[job_id]["user_ids"].append(user_id)
            
            satisfied_value = row["satisfied"]
            if satisfied_value == 0:
                satisfied_value = 0.5
                
            all_labels_dict[job_id]["satisfied"].append(satisfied_value)
        
        else:
            all_labels_dict[job_id] = {
                "user_ids": [],
                "satisfied": []
            }
            
            all_labels_dict[job_id]["user_ids"].append(user_id)

            satisfied_value = row["satisfied"]
            if satisfied_value == 0:
                satisfied_value = 0.5
                
            all_labels_dict[job_id]["satisfied"].append(satisfied_value)
    return all_labels_dict

def save_all_labels_dict(all_labels_dict: Dict, output_path: str):
    with open(output_path, 'w') as f:
        json.dump(all_labels_dict, f, indent=2)
    print(f"Saved all labels data to {output_path}")

def load_prompt_template(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            template = json.load(f)
        return template
    except Exception as e:
        raise ValueError(f"Error loading prompt template: {str(e)}")

def load_few_shot_examples(file_path: str, max_examples: int) -> List[Dict]:
    examples = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                example_json = json.loads(line.strip())
                if "query" in example_json and "llm_output" in example_json["query"]:
                    example = {
                        "jid": example_json.get("jd_no", ""),
                        "keyword_example": example_json["query"]["llm_output"]
                    }
                    examples.append(example)
                    if len(examples) >= max_examples:
                        break
        print(f"Loaded {len(examples)} few-shot examples from {file_path}")
        return examples
    except Exception as e:
        print(f"Error loading few-shot examples: {str(e)}")
        return []

def enrich_prompt_with_fewshot(template: Dict, examples: List[Dict], jd_df: pd.DataFrame) -> Dict:
    if not examples:
        return template
    
    new_template = template.copy()
    
    output_fields_emphasis = (
        "Please ensure your output includes the following fields:\n"
        "- 'title': List of relevant job titles\n"
        "- 'location': List of relevant locations\n"
        "- 'industryKeywords': List of relevant industry keywords\n"
        "- 'jobFunctionKeywords': List of relevant job function keywords\n"
        "- 'languages': List of required languages\n"
        "- 'yearOfWork': Work experience range in years\n"
        "- 'skillBooleanString': List of boolean expressions combining skills with AND/OR operators. Create multiple expressions to cover different skill areas. Use AND when both skills are required together, and OR when any of the skills is acceptable.\n"
    )
    
    few_shot_text = f"\n\n{output_fields_emphasis}\n\nHere are some examples of well-formatted keyword queries:\n\n"
    
    for i, example in enumerate(examples):
        if "jid" in example and "keyword_example" in example:
            jid = example["jid"]
            matching_rows = jd_df[jd_df['jd_no'] == jid]
            job_text = None
            if not matching_rows.empty:
                job_text = matching_rows.iloc[0]['job_text']
            
            if job_text:
                few_shot_text += f"Example {i+1} Job Description:\n{job_text}\n\n"
                few_shot_text += f"Example {i+1} Keywords:\n{json.dumps(example['keyword_example'], indent=2)}\n\n"
            else:
                few_shot_text += f"Example {i+1}:\n{json.dumps(example['keyword_example'], indent=2)}\n\n"
    
    if "system_message" in new_template:
        new_template["system_message"] += few_shot_text
    
    return new_template