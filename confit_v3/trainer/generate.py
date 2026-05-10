

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

def get_accepted_resume_texts(job_id: str, labels: Dict, all_resume_data: Dict[str, str]) -> str:
    accepted_resumes_texts_list = []
    if job_id in labels:
        user_ids = labels[job_id].get('user_ids', [])
        satisfied_scores = labels[job_id].get('satisfied', [])

        for i, user_id_from_label in enumerate(user_ids):
            user_id_str = str(user_id_from_label)
            if i < len(satisfied_scores):
                if satisfied_scores[i] == 1.0: 
                    resume_text = all_resume_data.get(user_id_str)
                    if resume_text and isinstance(resume_text, str) and resume_text.strip():
                        accepted_resumes_texts_list.append(f"--- Accepted Resume ---\n{resume_text}\n--- End of Resume ---")
            else:
                break

    if not accepted_resumes_texts_list:
        return "no accepted resumes found in records"
    return "\n\n".join(accepted_resumes_texts_list)

def run_elasticsearch_query(
    es_client: Elasticsearch,
    query: Dict,
    
    index: str = "resume_0303_2025",
    random_select: int = -1,
    cache_dir: diskcache.Cache = None
) -> Dict:
    key= (json.dumps(query),index,random_select)
    if cache_dir is not None:
        if key in cache_dir:
            return cache_dir[key]
    
    try:
        response = es_client.search(index=index, body=query)
        if not hasattr(response, 'body'):
            return {'hits': {'total': {'value': 0}, 'hits': []}}
        else:
            search_result = response.body
            if random_select>0:
                search_result["hits"]["total"]["value"]=random_select
                search_result["hits"]["hits"]=random.sample(search_result["hits"]["hits"], random_select)
                search_result["hits"]["max_score"]=max([item["_score"] for item in search_result["hits"]["hits"]])
            if cache_dir is not None:
                cache_dir[key]=search_result
        return search_result
    
    except Exception as e:
        print(query)
        print(f"Error running Elasticsearch query: {str(e)}")
        return {'hits': {'total': {'value': 0}, 'hits': []}}
    

from typing import List, Optional, Union
from pydantic import BaseModel


class YearOfWork(BaseModel):
    gte: Optional[int]
    lte: Optional[int]


class KeywordResult(BaseModel):
    title: List[str]
    location: List[str]
    industryKeywords: List[str]
    jobFunctionKeywords: List[str]
    languages: Optional[List[str]] = []
    yearOfWork: Optional[YearOfWork] = None
    skillBooleanString: List[str]
    
def generate_search_keywords_openai(
    client: OpenAI, 
    job_description: str, 
    template: Dict,
    model: str
) -> Dict:
    
    prompt = template["prompt_template"].replace("{job_description}", job_description)
    
    system_message = template.get("system_message", 
                                "You are a helpful assistant that extracts relevant search keywords from job descriptions to find matching candidates.")
    
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
    }
    
    if "qwen" in model.lower():
        json_schema = KeywordResult.model_json_schema()
        params["extra_body"] = {"guided_json": json_schema}
    else:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "extract_keywords",
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "array", "items": {"type": "string"}},
                        "location": {"type": "array", "items": {"type": "string"}},
                        "industryKeywords": {"type": "array", "items": {"type": "string"}},
                        "jobFunctionKeywords": {"type": "array", "items": {"type": "string"}},
                        "languages": {"type": "array", "items": {"type": "string"}},
                        "yearOfWork": {"type": "object", 
                                    "properties": {
                                        "gte": {"type": ["integer", "null"]},
                                        "lte": {"type": ["integer", "null"]}
                                    }},
                        "skillBooleanString": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Boolean expressions combining skills with AND/OR operators. Each expression represents a search query where terms with AND require both terms to be present, and terms with OR require any term to be present. Example: 'Russian AND Chinese OR Russian AND English' is interpreted as (Russian AND Chinese) OR (Russian AND English). Create multiple expressions to cover different skill areas."
                        }
                    },
                    "required": ["title", "location", "industryKeywords", "jobFunctionKeywords", "skillBooleanString"]
                    
                }
            } 
        }
    
    response = client.chat.completions.create(**params)
    
    try:
        content = response.choices[0].message.content
        keyword_dict = json.loads(content)
        return keyword_dict
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"Error parsing structured response: {str(e)}")
        return {}

def generate_search_keywords_anthropic(
    client: anthropic.Anthropic,
    job_description: str,
    template: Dict,
    model: str = "claude-3-7-sonnet-latest",
    temperature: float = 0.2
) -> Dict:
    prompt = template["prompt_template"].replace("{job_description}", job_description)
    
    system_message = template.get("system_message", 
                                 "You are a helpful assistant that extracts relevant search keywords from job descriptions to find matching candidates.")
    
    params = {
        "model": model,
        "system": system_message,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 500
    }

    params["tools"] = [{
        "name":"extract_keywords",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "array", "items": {"type": "string"}},
                "location": {"type": "array", "items": {"type": "string"}},
                "industryKeywords": {"type": "array", "items": {"type": "string"}},
                "jobFunctionKeywords": {"type": "array", "items": {"type": "string"}},
                "languages": {"type": "array", "items": {"type": "string"}},
                "yearOfWork": {"type": "object", 
                               "properties": {
                                   "gte": {"type": ["integer", "null"]},
                                   "lte": {"type": ["integer", "null"]}
                               }},
                "skillBooleanString": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "Boolean expressions combining skills with AND/OR operators. Each expression represents a search query where terms with AND require both terms to be present, and terms with OR require any term to be present. Example: 'Russian AND Chinese OR Russian AND English' is interpreted as (Russian AND Chinese) OR (Russian AND English). Create multiple expressions to cover different skill areas."
                }
            },
            "required": ["title", "location", "industryKeywords", "jobFunctionKeywords", "skillBooleanString"]
        }
    }]

    try:
        response = client.messages.create(**params)

        for content_block in response.content:
            if content_block.type == "tool_use":
                keyword_dict = content_block.input
                return keyword_dict
        content = response.content[0].text if response.content else ""
        try:
            keyword_dict = json.loads(content)
            return keyword_dict
        except json.JSONDecodeError:
            print(f"Error parsing structured response: {content}")
            return {}
    except Exception as e:
        print(f"Error generating keywords: {str(e)}")
        return {}

def generate_search_keywords_sft(
    client: OpenAI,
    job_description: str,
    model: str
) -> Dict:
    instruction = "You are a helpful assistant that extracts relevant search keywords from the following job descriptions to find matching candidates.\n"
    input_text = "[[Job Description]]\n" + job_description
    
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
    }
    response = client.chat.completions.create(**params)
    try:
        content = response.choices[0].message.content
        try:
            import ast
            keyword_dict = ast.literal_eval(content)
            return keyword_dict
        except (SyntaxError, ValueError):
            pass
        
    except Exception as e:
        print(response)
        print(f"Error parsing response: {str(e)}")
        return {}

def generate_keywords(
    job_id: str,
    job_description: str,
    model_provider: str,
    model_name: str,
    prompt_template: Dict,
    openai_client: Optional[OpenAI] = None,
    anthropic_client: Optional[anthropic.Anthropic] = None,
    num_prompts: int = 1,
    sft_model: bool = False,
    accepted_resumes_text: str = ""
) -> List[Dict]:
    results = []
    keyword_dict = {}
    for i in range(num_prompts):
        # try:
            if sft_model:
                if model_provider in ["openai", "local"]:
                    if not openai_client:
                        raise ValueError(f"{model_provider} client not provided")
                    keyword_dict = generate_search_keywords_sft(
                        openai_client, job_description, model_name, accepted_resumes_text
                    )
                else:
                    raise ValueError(f"SFT mode not supported for {model_provider}")
            else:
                if model_provider == "openai":
                    if not openai_client:
                        raise ValueError("OpenAI client not provided")
                    keyword_dict = generate_search_keywords_openai(
                        openai_client, job_description, prompt_template, 
                        model_name,accepted_resumes_text
                    )
                elif model_provider == "anthropic":
                    if not anthropic_client:
                        raise ValueError("Anthropic client not provided")
                    keyword_dict = generate_search_keywords_anthropic(
                        anthropic_client, job_description, prompt_template, 
                        model_name, temperature=0.2
                    )
                elif model_provider == "local":
                    if not openai_client:
                        raise ValueError("Local OpenAI client not provided")
                    keyword_dict = generate_search_keywords_openai(
                        openai_client, job_description, prompt_template, 
                        model_name, accepted_resumes_text
                    )
                else:
                    raise ValueError(f"Unsupported model provider: {model_provider}")
            
            results.append({
                "job_id": job_id,
                "prompt_num": i + 1,
                "keywords": keyword_dict
            })
            
        # except Exception as e:
            # print(f"Error generating keywords for job {job_id}, prompt {i+1}: {str(e)}")
    
    return results
# def generate_keywords(
#     job_id: str,
#     job_description: str,
#     model_provider: str,
#     model_name: str,
#     prompt_template: Dict,
#     openai_client: Optional[OpenAI] = None,
#     anthropic_client: Optional[anthropic.Anthropic] = None,
#     num_prompts: int = 1,
#     sft_model: bool = False
# ) -> List[Dict]:
#     results = []
#     keyword_dict = {}
#     for i in range(num_prompts):
#         # try:
#             if sft_model:
#                 if model_provider in ["openai", "local"]:
#                     if not openai_client:
#                         raise ValueError(f"{model_provider} client not provided")
#                     keyword_dict = generate_search_keywords_sft(
#                         openai_client, job_description, model_name
#                     )
#                 else:
#                     raise ValueError(f"SFT mode not supported for {model_provider}")
#             else:
#                 if model_provider == "openai":
#                     if not openai_client:
#                         raise ValueError("OpenAI client not provided")
#                     keyword_dict = generate_search_keywords_openai(
#                         openai_client, job_description, prompt_template, 
#                         model_name
#                     )
#                 elif model_provider == "anthropic":
#                     if not anthropic_client:
#                         raise ValueError("Anthropic client not provided")
#                     keyword_dict = generate_search_keywords_anthropic(
#                         anthropic_client, job_description, prompt_template, 
#                         model_name, temperature=0.2
#                     )
#                 elif model_provider == "local":
#                     if not openai_client:
#                         raise ValueError("Local OpenAI client not provided")
#                     keyword_dict = generate_search_keywords_openai(
#                         openai_client, job_description, prompt_template, 
#                         model_name
#                     )
#                 else:
#                     raise ValueError(f"Unsupported model provider: {model_provider}")
            
#             results.append({
#                 "job_id": job_id,
#                 "prompt_num": i + 1,
#                 "keywords": keyword_dict
#             })
            
#         # except Exception as e:
#             # print(f"Error generating keywords for job {job_id}, prompt {i+1}: {str(e)}")
    
#     return results

# def parse_skill_boolean_string(skill_list):
#     clauses = []
#     for skill in skill_list:
#         # Split by 'AND' and 'OR'
#         and_clauses = [s.strip().lower()
#                         for s in re.split(r'\s+and\s+', skill, flags=re.I)]
#         should_clauses = []
#         for and_clause in and_clauses:
#             or_clauses = [s.strip().lower() for s in re.split(
#                 r'\s+or\s+', and_clause, flags=re.I)]
#             if len(or_clauses) == 1:
#                 should_clauses.append(
#                     {"match_phrase": {"searchContent": or_clauses[0]}})
#             else:
#                 should_clauses.append({"bool": {"should": [
#                                         {"match_phrase": {"searchContent": clause}} for clause in or_clauses]}})
#         if len(should_clauses) == 1:
#             clause = should_clauses[0]
#             if 'bool' in clause:
#                 clauses.extend(clause["bool"]["should"])
#             else:
#                 clauses.append(clause)
#         else:
#             clauses.append({"bool": {"must": should_clauses}})
#     return clauses


def parse_skill_boolean_string(skill_list):
    clauses = []
    for skill in skill_list:
        # Split by 'AND' and 'OR'
        and_clauses = [s.strip().lower()
                        for s in re.split(r'\s+and\s+', skill, flags=re.I)]
        should_clauses = []
        for and_clause in and_clauses:
            or_clauses = [s.strip().lower() for s in re.split(
                r'\s+or\s+', and_clause, flags=re.I)]
            if len(or_clauses) == 1:
                should_clauses.append(
                    {"match_phrase": {"searchContent": or_clauses[0]}})
            else:
                should_clauses.append({"bool": {"should": [
                                        {"match_phrase": {"searchContent": clause}} for clause in or_clauses]}})
        if len(should_clauses) == 1:
            clause = should_clauses[0]
            if 'bool' in clause:
                clauses.extend(clause["bool"]["should"])
            else:
                clauses.append(clause)
        else:
            clauses.append({"bool": {"must": should_clauses}})
    return clauses
# def build_es_query_from_dict(keyword_dict: Dict) -> Dict:
#     _titles = keyword_dict.get('title', [])
#     _location = keyword_dict.get('location', [])
#     _industry_keywords = keyword_dict.get('industryKeywords', [])
#     _jf_keywords = keyword_dict.get('jobFunctionKeywords', [])
#     _languages = [lang_.upper() for lang_ in keyword_dict.get('languages', [])]
#     _skills_boolean = keyword_dict.get('skillBooleanString', [])
#     _year_of_work = keyword_dict.get('yearOfWork')
#     if isinstance(_year_of_work, dict) and "gte" in _year_of_work and _year_of_work["gte"] is None:
#         del _year_of_work["gte"]
#     if isinstance(_year_of_work, dict) and "lte" in _year_of_work and _year_of_work["lte"] is None:
#         del _year_of_work["lte"]
#     title_should_conditions = [
#         {"match_phrase": {"titles": title.lower()}} for title in _titles]
#     location_should_conditions = [
#         {"match_phrase": {"location": location.lower()}} for location in _location]
#     year_of_work_condition = {
#         "range": {
#             "yearOfWork": _year_of_work
#         }
#     }
#     languages_condition = [
#         {"match_phrase": {"languages": language}} for language in _languages]
#     skill_key_conditions = parse_skill_boolean_string(
#         _skills_boolean)
#     keyword_conditions = []
#     for keyword in itertools.chain(_industry_keywords, _jf_keywords):
#         if isinstance(keyword, str):
#             keyword_conditions.append(
#                 {"match_phrase": {"searchContent": keyword.lower()}})
#         elif isinstance(keyword, list):
#             keyword_conditions.extend(
#                 {"match_phrase": {"searchContent": k_.lower()}} for k_ in keyword)
#         else:
#             print(keyword)
#     must_conditions = [
#         {"bool": {"should": conditions_}} for conditions_ in
#         (title_should_conditions, location_should_conditions, languages_condition, skill_key_conditions, keyword_conditions) if conditions_
#     ]
#     must_conditions.append(year_of_work_condition)
#     query = {
#         "from": 0,
#         "size": 10000,
#         "query": {
#             "constant_score": {
#                 "filter": {
#                     "bool": {
#                         "must": must_conditions
#                     }
#                 }    
#             }
#         }
#     }
#     return query


def build_es_query_from_dict(keyword_dict: Dict) -> Dict:
    '''for finetuned model using search_train_data.json'''
    query = {
        "from": 0,
        "size": 10000,
        "query": {
            "bool": {
                "should": []
            }
        }
    }

    field_mapping = {
        "title": "titles",
        "location": "location",
        "industryKeywords": "industries",
        "jobFunctionKeywords": "jobFunctions",
        "languages": "languages",
        "yearOfWork": "yearOfWork"
    }

    def clean_term(term: str) -> str:
        term = term.replace("\\", "")
        term = term.strip()
        if term.startswith('('):
            term = term[1:]
        if term.endswith(')'):
            term = term[:-1]
        term = term.strip()
        if (term.startswith('"') and term.endswith('"')) or (term.startswith("'") and term.endswith("'")):
            term = term[1:-1]
        return term.strip().lower() 

    for model_field, value in keyword_dict.items():
        if model_field == "skillBooleanString":
            continue

        es_field = field_mapping.get(model_field)
        if es_field is None:
            continue

        if model_field == "yearOfWork":
            if isinstance(value, str):
                value = value.strip().strip("[]")
                parts = re.split(r'(?i)\s+to\s+', value.strip())
                range_query = {"range": {es_field: {}}}
                if len(parts) == 2:
                    min_years = parts[0].strip()
                    max_years = parts[1].strip()

                    if min_years != "" and min_years.isdigit():
                        range_query["range"][es_field]["gte"] = int(min_years)

                    if max_years != "" and max_years.isdigit():
                        range_query["range"][es_field]["lte"] = int(max_years)

                if range_query["range"][es_field] != {}:
                    query["query"]["bool"]["filter"] = range_query
            elif isinstance(value, dict):
                range_query = {"range": {es_field: {}}}
                if "gte" in value and value["gte"] is not None:
                    range_query["range"][es_field]["gte"] = value["gte"]
                if "lte" in value and value["lte"] is not None:
                    range_query["range"][es_field]["lte"] = value["lte"]

                if range_query["range"][es_field] != {}:
                    query["query"]["bool"]["filter"] = range_query

        elif isinstance(value, list) and value:
            or_clauses = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    cleaned_term = clean_term(item)
                    if cleaned_term:
                        or_clauses.append({"match": {es_field: cleaned_term}})

            if or_clauses:
                query["query"]["bool"]["should"].append({
                    "bool": {
                        "should": or_clauses,
                        "minimum_should_match": 1
                    }
                })

        elif isinstance(value, str) and value.strip():
            query["query"]["bool"]["should"].append({"match": {es_field: clean_term(value)}})

    if "skillBooleanString" in keyword_dict and isinstance(keyword_dict["skillBooleanString"], list):
        search_content_clauses = []

        for boolean_expr in keyword_dict["skillBooleanString"]:
            if not boolean_expr or not isinstance(boolean_expr, str):
                continue

            def parse_boolean_expr(expr: str):
                if " AND " in expr.upper():
                    and_parts = re.split(r'\s+(?i:AND)\s+', expr)
                    and_clauses = []

                    for part in and_parts:
                        cleaned_part = clean_term(part)
                        if cleaned_part:
                            and_clauses.append({"match": {"searchContent": cleaned_part}})

                    if and_clauses:
                        return {
                            "bool": {
                                "must": and_clauses
                            }
                        }
                elif " OR " in expr.upper():
                    or_parts = re.split(r'\s+(?i:OR)\s+', expr)
                    or_clauses = []

                    for part in or_parts:
                        cleaned_part = clean_term(part)
                        if cleaned_part:
                            or_clauses.append({"match": {"searchContent": cleaned_part}})

                    if or_clauses:
                        return {
                            "bool": {
                                "should": or_clauses,
                                "minimum_should_match": 1
                            }
                        }
                else:
                    cleaned_expr = clean_term(expr)
                    if cleaned_expr:
                        return {"match": {"searchContent": cleaned_expr}}

                return None

            def analyze_complex_expr(expr: str):
                if " AND " in expr.upper() and " OR " in expr.upper():
                    or_blocks = re.split(r'\s+(?i:OR)\s+', expr)
                    or_clauses = []

                    for block in or_blocks:
                        if " AND " in block.upper():
                            and_parts = re.split(r'\s+(?i:AND)\s+', block)
                            and_clauses = []

                            for part in and_parts:
                                cleaned_part = clean_term(part)
                                if cleaned_part:
                                    and_clauses.append({"match": {"searchContent": cleaned_part}})

                            if and_clauses:
                                or_clauses.append({
                                    "bool": {
                                        "must": and_clauses
                                    }
                                })
                        else:
                            cleaned_block = clean_term(block)
                            if cleaned_block:
                                or_clauses.append({"match": {"searchContent": cleaned_block}})

                    if or_clauses:
                        return {
                            "bool": {
                                "should": or_clauses,
                                "minimum_should_match": 1
                            }
                        }
                else:
                    return parse_boolean_expr(expr)

                return None

            parsed_expr = analyze_complex_expr(boolean_expr)
            if parsed_expr:
                search_content_clauses.append(parsed_expr)

        if search_content_clauses:
            query["query"]["bool"]["should"].append({
                "bool": {
                    "should": search_content_clauses,
                    "minimum_should_match": 1,
                    # "boost": 2.0  
                }
            })

    if len(query["query"]["bool"]["should"]) == 0:
        print("Warning: No valid query conditions were created from keywords")

    return query

