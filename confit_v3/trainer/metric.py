
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

def calculate_query_length(keyword_dict: Dict) -> int:
    total_length = 0
    
    normal_fields = ["title", "location", "industryKeywords", "jobFunctionKeywords", "languages"]
    
    for field in normal_fields:
        if field in keyword_dict:
            if isinstance(keyword_dict[field], list):
                total_length += len(keyword_dict[field])
            elif isinstance(keyword_dict[field], str) and keyword_dict[field].strip():
                total_length += 1
    
    if "yearOfWork" in keyword_dict and keyword_dict["yearOfWork"]:
        if isinstance(keyword_dict["yearOfWork"], dict):
            if "gte" in keyword_dict["yearOfWork"] and keyword_dict["yearOfWork"]["gte"] is not None:
                total_length += 1
            if "lte" in keyword_dict["yearOfWork"] and keyword_dict["yearOfWork"]["lte"] is not None:
                total_length += 1
    
    if "skillBooleanString" in keyword_dict and isinstance(keyword_dict["skillBooleanString"], list):
        for boolean_expr in keyword_dict["skillBooleanString"]:
            if not boolean_expr or not isinstance(boolean_expr, str):
                continue
            
            def count_terms(expr: str) -> int:
                expr = expr.upper()
                if " AND " in expr and " OR " in expr:
                    or_blocks = re.split(r'\s+(?i:OR)\s+', expr)
                    count = 0
                    
                    for block in or_blocks:
                        if " AND " in block.upper():
                            and_parts = re.split(r'\s+(?i:AND)\s+', block)
                            count += len(and_parts)
                        else:
                            count += 1
                    
                    return count
                elif " AND " in expr:
                    and_parts = re.split(r'\s+(?i:AND)\s+', expr)
                    return len(and_parts)
                elif " OR " in expr:
                    or_parts = re.split(r'\s+(?i:OR)\s+', expr)
                    return len(or_parts)
                else:
                    return 1
            
            total_length += count_terms(boolean_expr)
    return total_length


def calculate_top_k_avg_bm25(search_results: Dict, k: int = 10) -> Optional[float]:
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    hits = search_results['hits']['hits']
    top_k_hits = hits[:min(k, len(hits))]
    
    if not top_k_hits:
        return 0.0
    
    total_score = sum(hit.get('_score', 0) for hit in top_k_hits)
    avg_score = total_score / len(top_k_hits) if len(top_k_hits) > 0 else 0.0
    
    return avg_score

def calculate_precision_recall_f1(
    search_results: Dict,
    job_id: str,
    labels: Dict,
    k: int = 10
) -> Dict:
    """Calculate precision, recall, and F1 score for the top k search results."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    if job_id not in labels:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Get the top k search results
    hits = search_results['hits']['hits']
    top_k_hits = hits[:min(k, len(hits))]
    
    # If no results, return zeros
    if not top_k_hits:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Get the IDs of the top k search results
    top_k_ids = [hit.get('_id') for hit in top_k_hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zeros
    if not relevant_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
    
    # Calculate true positives (resumes that are both relevant and retrieved)
    true_positives = len(set(top_k_ids).intersection(set(relevant_ids)))
    
    # Calculate precision, recall, and F1
    precision = true_positives / len(top_k_ids) if top_k_ids else 0.0
    recall = true_positives / len(relevant_ids) if relevant_ids else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_map1(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate MAP considering only documents with satisfied == 1.0"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate average precision
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    # Normalize by the total number of relevant documents
    if len(relevant_ids) > 0:
        return precision_sum / len(relevant_ids)
    else:
        return 0.0

def calculate_map2(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate MAP considering documents with satisfied == 1.0 OR satisfied == 0.5"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate average precision
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    # Normalize by the total number of relevant documents
    if len(relevant_ids) > 0:
        return precision_sum / len(relevant_ids)
    else:
        return 0.0

def calculate_mrr1(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate MRR considering only documents with satisfied == 1.0"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0

    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    
    # If no relevant document is found
    return 0.0

def calculate_mrr2(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate MRR considering documents with satisfied == 1.0 OR satisfied == 0.5"""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0

    hits = search_results['hits']['hits']

    result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Find the rank of the first relevant document
    for i, doc_id in enumerate(result_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    
    # If no relevant document is found
    return 0.0


def calculate_recall_at_all1(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate recall over all search results, only considering satisfied == 1.0."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']
    
    if not hits:
        return 0.0
    
    all_result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0:  # Use 1.0 for "interviewed" status
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate recall at all (how many relevant docs are in the entire result set)
    found_relevant = len(set(all_result_ids).intersection(set(relevant_ids)))
    total_relevant = len(relevant_ids)
    
    recall_at_all = found_relevant / total_relevant if total_relevant > 0 else 0.0
    
    return recall_at_all

def calculate_recall_at_all2(search_results: Dict, job_id: str, labels: Dict) -> float:
    """Calculate recall over all search results, considering satisfied == 1.0 OR satisfied == 0.5."""
    if not search_results or 'hits' not in search_results or 'hits' not in search_results['hits']:
        return 0.0
    
    if job_id not in labels:
        return 0.0
    
    hits = search_results['hits']['hits']
    
    if not hits:
        return 0.0
    
    all_result_ids = [hit.get('_id') for hit in hits]
    
    # Get the relevant resumes for this job (satisfied == 1.0 OR satisfied == 0.5)
    relevant_ids = []
    for i, satisfied in enumerate(labels[job_id]['satisfied']):
        if satisfied == 1.0 or satisfied == 0.5:
            relevant_ids.append(labels[job_id]['user_ids'][i])
    
    # If no relevant documents, return zero
    if not relevant_ids:
        return 0.0
    
    # Calculate recall at all (how many relevant docs are in the entire result set)
    found_relevant = len(set(all_result_ids).intersection(set(relevant_ids)))
    total_relevant = len(relevant_ids)
    
    recall_at_all = found_relevant / total_relevant if total_relevant > 0 else 0.0
    
    return recall_at_all

def calculate_ndcg_at_k(search_results: Dict, job_id: str, labels: Dict, 
                       k: int = None, total_doc_count: int = 3112, 
                       use_alternative_relevance: bool = False) -> float:

    all_result_ids = [hit.get('_id') for hit in search_results['hits']['hits']]

    result_ids = all_result_ids
    if k is not None:
        result_ids = all_result_ids[:min(k, len(all_result_ids))]

    position_map = {doc_id: pos + 1 for pos, doc_id in enumerate(all_result_ids)}

    relevance_map = {}
    
    for i, user_id in enumerate(labels[job_id]['user_ids']):
        satisfied_value = labels[job_id]['satisfied'][i]

        if use_alternative_relevance:
            if satisfied_value == 1.0:
                relevance_map[user_id] = 1 
            elif satisfied_value == 0.5:
                relevance_map[user_id] = 0.5 
            else:
                relevance_map[user_id] = 0 
        else:
            if satisfied_value == 1.0:
                relevance_map[user_id] = 1 
            else:
                relevance_map[user_id] = 0 
    
    dcg = 0.0
    
    for doc_id in result_ids:
        relevance = relevance_map.get(doc_id, 0)
        position = position_map[doc_id]
        dcg += relevance / np.log2(position + 1)
    
    sorted_relevance = sorted([relevance_map.get(doc_id, 0) for doc_id in labels[job_id]['user_ids'] 
                               if relevance_map.get(doc_id, 0) > 0], reverse=True)
    
    if k is not None:
        sorted_relevance = sorted_relevance[:min(k, len(sorted_relevance))]
    
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevance):
        position = i + 1
        idcg += relevance / np.log2(position + 1)
    
    if idcg > 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    
    return ndcg


def calculate_metrics(search_result: Dict, job_id: str, labels: Dict, 
                      bm25_k_values: List[int], eval_k_values: List[int]) -> Dict:
    metrics_data = {
        "bm25_scores": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg1": {},
        "ndcg2": {}
    }
    
    # Calculate BM25 scores
    for k in bm25_k_values:
        k_key = f"top{k}"
        metrics_data["bm25_scores"][k_key] = calculate_top_k_avg_bm25(search_result, k)
    
    # Calculate precision, recall, F1 for each k value
    for k in eval_k_values:
        k_key = f"top{k}"
        
        # Calculate precision, recall, F1
        precision_recall_f1 = calculate_precision_recall_f1(search_result, job_id, labels, k)
        metrics_data["precision"][k_key] = precision_recall_f1["precision"]
        metrics_data["recall"][k_key] = precision_recall_f1["recall"]
        metrics_data["f1"][k_key] = precision_recall_f1["f1"]
        ndcg_key = f"@{k}"
        metrics_data["ndcg1"][ndcg_key] = calculate_ndcg_at_k(
            search_result, job_id, labels, k=k, use_alternative_relevance=False
        )
    
        metrics_data["ndcg2"][ndcg_key] = calculate_ndcg_at_k(
            search_result, job_id, labels, k=k, use_alternative_relevance=True
        )
    
    metrics_data["map1"] = calculate_map1(search_result, job_id, labels)
    metrics_data["mrr1"] = calculate_mrr1(search_result, job_id, labels)
    
    metrics_data["map2"] = calculate_map2(search_result, job_id, labels)
    metrics_data["mrr2"] = calculate_mrr2(search_result, job_id, labels)
    
    metrics_data["recall_at_all1"] = calculate_recall_at_all1(search_result, job_id, labels)
    metrics_data["recall_at_all2"] = calculate_recall_at_all2(search_result, job_id, labels)
    
    metrics_data["recallmap1"] = metrics_data["map1"] * metrics_data["recall_at_all1"]
    metrics_data["recallmap2"] = metrics_data["map2"] * metrics_data["recall_at_all2"]
    
    metrics_data["ndcg1@all"] = calculate_ndcg_at_k(
        search_result, job_id, labels, k=None, use_alternative_relevance=False
    )
    metrics_data["ndcg2@all"] = calculate_ndcg_at_k(
        search_result, job_id, labels, k=None, use_alternative_relevance=True
    )
    
    return metrics_data


def save_intermediate_results(
    output_dir: str,
    keyword_results: List[Dict],
    query_results: List[Dict],
    search_results: List[Dict],
    metrics: List[Dict]
):
    # Save keywords
    with open(os.path.join(output_dir, "keywords.json"), "w") as f:
        json.dump(keyword_results, f, indent=2)
    
    # Save queries
    with open(os.path.join(output_dir, "queries.json"), "w") as f:
        json.dump(query_results, f, indent=2)
    
    # Save search results (might be large, so save separately)
    with open(os.path.join(output_dir, "search_results.json"), "w") as f:
        json.dump(search_results, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def save_final_results(
    output_dir: str,
    metrics: List[Dict],
    model_provider: str,
    model_name: str,
    bm25_k_values: List[int],
    eval_k_values: List[int]
):
    df = pd.DataFrame(metrics) if metrics else pd.DataFrame()
    
    final_results = {
        "model_provider": model_provider,
        "model_name": model_name,
        "num_jobs": len(df["job_id"].unique()) if not df.empty else 0,
        "bm25_scores": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "ndcg1": {},
        "ndcg2": {},
        "map1": 0.0,
        "mrr1": 0.0,
        "map2": 0.0,
        "mrr2": 0.0,
        "recall_at_all1": 0.0,
        "recall_at_all2": 0.0,
        "recallmap1": 0.0,
        "recallmap2": 0.0,
        "ndcg1@all": 0.0,
        "ndcg2@all": 0.0,
        "average Retrieval Number of Resumes": 0.0,
        "average_query_length": 0.0
    }
    
    for k in bm25_k_values:
        k_key = f"top{k}"
        bm25_values = []
        for m in metrics:
            if "bm25_scores" in m and k_key in m["bm25_scores"]:
                v = m["bm25_scores"][k_key]
                if v is not None:
                    bm25_values.append(v)
        
        if bm25_values:
            final_results["bm25_scores"][k_key] = sum(bm25_values) / len(bm25_values)
        else:
            final_results["bm25_scores"][k_key] = 0.0
    
    for k in eval_k_values:
        k_key = f"top{k}"
        
        for metric_name in ["precision", "recall", "f1"]:
            values = []
            for m in metrics:
                if metric_name in m and k_key in m[metric_name]:
                    v = m[metric_name][k_key]
                    if v is not None:
                        values.append(v)
                elif f"{metric_name}_{k_key}" in m:
                    v = m[f"{metric_name}_{k_key}"]
                    if v is not None:
                        values.append(v)
            
            if values:
                final_results[metric_name][k_key] = sum(values) / len(values)
            else:
                final_results[metric_name][k_key] = 0.0
                
        ndcg_key = f"@{k}"
        
        for metric_name in ["ndcg1", "ndcg2"]:
            values = []
            for m in metrics:
                if metric_name in m and ndcg_key in m[metric_name]:
                    v = m[metric_name][ndcg_key]
                    if v is not None:
                        values.append(v)
            
            if values:
                final_results[metric_name][ndcg_key] = sum(values) / len(values)
            else:
                final_results[metric_name][ndcg_key] = 0.0
    
    for metric_name in ["map1", "mrr1", "map2", "mrr2", "recall_at_all1", "recall_at_all2", "recallmap1", "recallmap2", "ndcg1@all", "ndcg2@all"]:
        values = []
        for m in metrics:
            if metric_name in m and isinstance(m[metric_name], (int, float)):
                v = m[metric_name]
                if v is not None:
                    values.append(v)
        
        if values:
            final_results[metric_name] = sum(values) / len(values)
        else:
            final_results[metric_name] = 0.0
    
    values = []
    for m in metrics:
        if "hits_count" in m:
            v = m["hits_count"]
            if v is not None:
                values.append(v)
    if values:
        final_results["average Retrieval Number of Resumes"] = sum(values) / len(values)
    else:
        final_results["average Retrieval Number of Resumes"] = 0.0
        
    values = []
    for m in metrics:
        if "query_length" in m:
            v = m["query_length"]
            if v is not None:
                values.append(v)
    if values:
        final_results["average_query_length"] = sum(values) / len(values)
    else:
        final_results["average_query_length"] = 0.0

    # with open(os.path.join(output_dir, "final_results.json"), "w") as f:
    #     json.dump(final_results, f, indent=2)
    if os.path.exists(os.path.join(output_dir, "final_results.json")):
            with open(os.path.join(output_dir, "final_results.json"), "r") as f:
                existing_results = json.load(f)
            if isinstance(existing_results, list):
                existing_results.append(final_results)
            else:
                existing_results = [existing_results, final_results]
            with open(os.path.join(output_dir, "final_results.json"), "w") as f:
                json.dump(existing_results, f, indent=2)
    else:
        with open(os.path.join(output_dir, "final_results.json"), "w") as f:
           json.dump([final_results], f, indent=2)
    print("\nFinal Results:")
    print(f"Model: {final_results['model_provider']}/{final_results['model_name']}")
    print(f"Number of jobs: {final_results['num_jobs']}")
    
    print("\nBM25 Scores:")
    for k in bm25_k_values:
        k_key = f"top{k}"
        print(f"Average BM25@{k}: {final_results['bm25_scores'][k_key]:.4f}")
    
    print("\nRelevance Metrics:")
    for k in eval_k_values:
        k_key = f"top{k}"
        print(f"Precision@{k}: {final_results['precision'][k_key]:.4f}")
        print(f"Recall@{k}: {final_results['recall'][k_key]:.4f}")
        print(f"F1@{k}: {final_results['f1'][k_key]:.4f}")
        ndcg_key = f"@{k}"
        print(f"NDCG1@{k}: {final_results['ndcg1'][ndcg_key]:.4f}")
        print(f"NDCG2@{k}: {final_results['ndcg2'][ndcg_key]:.4f}")
        print("")
    
    
    print(f"MAP1 (only sat=1.0): {final_results['map1']:.4f}")
    print(f"MRR1 (only sat=1.0): {final_results['mrr1']:.4f}")
    print(f"Recall@All1 (only sat=1.0): {final_results['recall_at_all1']:.4f}")
    print(f"RecallMAP1 (only sat=1.0): {final_results['recallmap1']:.4f}")
    
    print(f"MAP2 (sat=1.0 or 0.5): {final_results['map2']:.4f}")
    print(f"MRR2 (sat=1.0 or 0.5): {final_results['mrr2']:.4f}")
    print(f"Recall@All2 (sat=1.0 or 0.5): {final_results['recall_at_all2']:.4f}")
    print(f"RecallMAP2 (sat=1.0 or 0.5): {final_results['recallmap2']:.4f}")
    print(f"NDCG1@all: {final_results['ndcg1@all']:.4f}")
    print(f"NDCG2@all: {final_results['ndcg2@all']:.4f}")
    print(f"Average Retrieval Number of Resumes: {final_results['average Retrieval Number of Resumes']:.4f}")
    print(f"Average Query Length: {final_results['average_query_length']:.4f}")