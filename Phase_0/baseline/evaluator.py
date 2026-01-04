#!/usr/bin/env python3
"""
RAGAS Evaluation Script for Baseline RAG System (Phase 0)
Updated for DeepSeek API Judge.
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings 
# deepseek is openai compatible, so we use ChatOpenAI
# from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

load_dotenv()

# --- CONFIGURATION ---
# 1. Generator Config (Local - Free)
GEN_MODEL_NAME = "llama3.1:8b"
EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"

# # 2. Judge Config (DeepSeek)
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# JUDGE_MODEL_NAME = "deepseek-chat" 
# DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Initialize components
GENERATOR_LLM = OllamaLLM(model=GEN_MODEL_NAME, temperature=0.0)
EMBEDDING = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# # Initialize Judge
# if not DEEPSEEK_API_KEY:
#     print("❌ Error: DEEPSEEK_API_KEY not found in .env file")
#     print("Please add DEEPSEEK_API_KEY=your_key_here to your .env file")
#     sys.exit(1)

# Configure Judge (DeepSeek via OpenAI compatible client)
JUDGE_LLM = OllamaLLM(model="nemotron-mini:4b", temperature=0.0, format = "json")

os.environ["RAGAS_DEBUG"] = "false"

try:
    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
    )
    from ragas.run_config import RunConfig
except ImportError as e:
    print(f"❌ RAGAS import error: {e}")
    print("Install with: pip install ragas datasets langchain-openai")
    sys.exit(1)

# Import your Simple RAG
try:
    from simple_rag import Simple_RAG 
except ImportError:
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
        from simple_rag import Simple_RAG
    except ImportError:
        print("❌ Could not import Simple_RAG. Check your paths.")
        sys.exit(1)

class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems using RAGAS.
    """
    
    def __init__(self):
        self.judge_llm = JUDGE_LLM
        self.embedding = EMBEDDING
        self.results = {}
        
    def load_golden_dataset(self, dataset_path: Path, max_samples: int = None) -> List[Dict]:
        """Load and validate golden dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if max_samples and max_samples < len(data):
            print(f"⚠️  Limiting to {max_samples} samples for quick evaluation")
            data = data[:max_samples]
        
        print(f"✅ Loaded {len(data)} samples from golden dataset")
        return data
    
    def run_rag_pipeline(self, rag_system: Simple_RAG, questions: List[str]) -> Tuple[List[str], List[List[str]], List[float]]:
        """Run questions through RAG system and collect outputs."""
        answers = []
        contexts_list = []
        latencies = []
        
        print(f"🔍 Running {len(questions)} queries through RAG pipeline...")
        
        for i, question in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] Processing: {question[:60]}...")
            start_time = time.time()
            try:
                result = rag_system.ask_question(question)
                answer = result.get('answer', '')
                context_docs = result.get('context', [])
                contexts = []
                
                if context_docs:
                    first_doc = context_docs[0]
                    if hasattr(first_doc, 'page_content'):
                        contexts = [doc.page_content for doc in context_docs]
                    elif isinstance(first_doc, str):
                        contexts = context_docs
                    else:
                        contexts = [str(c) for c in context_docs]
                
                answers.append(answer)
                contexts_list.append(contexts)
                latencies.append(time.time() - start_time)
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                answers.append("Error processing query")
                contexts_list.append([""]) 
                latencies.append(0.0)
        
        return answers, contexts_list, latencies
    
    def calculate_ragas_metrics(self, dataset: Dataset) -> Dict[str, float]:
        """
        Calculate RAGAS metrics.
        """
        print("📊 Calculating RAGAS metrics (DeepSeek Judge)...")
        
        # ESSENTIAL TRIO METRICS
        metrics = [
            faithfulness,       # Hallucination check 
            answer_relevancy,   # Quality check 
            context_precision   # Retrieval check 
        ]
        
        aggregated_scores = {m.name: [] for m in metrics}
        total_samples = len(dataset)
        
        for i in range(total_samples):
            print(f"    ⏳ Evaluating sample {i+1}/{total_samples}...", end="", flush=True)
            
            sample_data = dataset.select([i])
            
            try:
# In evaluator.py --> calculate_ragas_metrics method

                # ... inside the loop ...
                result = evaluate(
                    dataset=sample_data,
                    metrics=metrics,
                    llm=self.judge_llm,
                    embeddings=self.embedding,
                    run_config=RunConfig(timeout=1200, max_retries=3),
                    raise_exceptions=False 
                )
                
                # --- PATCH STARTS HERE ---
                print(f"    🔍 Raw Result Keys: {list(result.keys())}") # Debug print
                
                # 1. Safely convert RAGAS Result to a standard Python Dictionary
                # This bypasses the buggy __getitem__ method in the RAGAS class
                try:
                    # Try newer RAGAS version attribute first
                    if hasattr(result, 'scores'):
                        res_dict = result.scores.to_dict() 
                    else:
                        # Fallback for older versions
                        res_dict = dict(result)
                except Exception as conversion_error:
                    print(f"    ⚠️ Could not convert result to dict: {conversion_error}")
                    res_dict = {}

                # 2. Safe Extraction Loop
                for metric_name in aggregated_scores.keys():
                    # Use .get() which never raises KeyError
                    score = res_dict.get(metric_name, np.nan)
                    
                    if not np.isnan(score):
                        aggregated_scores[metric_name].append(score)
                    else:
                        print(f"    ⚠️ Metric '{metric_name}' failed (got NaN)")
                    # --- PATCH ENDS HERE ---
                    
            except Exception as e:
                print(f" Failed! {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate Averages
        final_scores = {}
        for metric, values in aggregated_scores.items():
            if values:
                final_scores[metric] = float(np.mean(values))
            else:
                final_scores[metric] = 0.0
                
        return final_scores
    
    def prepare_ragas_dataset(self, questions, answers, contexts_list, ground_truths) -> Dataset:
        dataset_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths 
        }
        return Dataset.from_dict(dataset_dict)

    def calculate_custom_metrics(self, answers, contexts_list, latencies) -> Dict[str, float]:
        print("📈 Calculating custom metrics...")
        custom_metrics = {}
        
        answer_lengths = [len(a.split()) for a in answers]
        custom_metrics["avg_answer_length"] = float(np.mean(answer_lengths)) if answer_lengths else 0.0
        
        retrieved_counts = [len(ctx) for ctx in contexts_list]
        custom_metrics["avg_retrieved_chunks"] = float(np.mean(retrieved_counts)) if retrieved_counts else 0.0
        
        if latencies:
            custom_metrics["avg_latency"] = float(np.mean(latencies))
        
        return custom_metrics
    
    def evaluate_system(self, system_name: str, rag_system: Simple_RAG, golden_data: List[Dict]) -> Dict[str, Any]:
        print(f"\n{'='*70}")
        print(f"EVALUATING: {system_name.upper()}")
        print(f"{'='*70}")
        
        questions = [item["query"] for item in golden_data]
        ground_truths = [item["answer"] for item in golden_data]
        
        answers, contexts_list, latencies = self.run_rag_pipeline(rag_system, questions)
        
        dataset = self.prepare_ragas_dataset(questions, answers, contexts_list, ground_truths)
        
        ragas_scores = self.calculate_ragas_metrics(dataset)
        custom_metrics = self.calculate_custom_metrics(answers, contexts_list, latencies)
        
        results = {
            "system_name": system_name,
            "timestamp": datetime.now().isoformat(),
            "ragas_metrics": ragas_scores,
            "custom_metrics": custom_metrics
        }
        
        self.results[system_name] = results
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY: {results['system_name'].upper()}")
        print(f"{'='*70}")
        
        print("\n📊 RAGAS METRICS:")
        print("-" * 40)
        if results.get('ragas_metrics'):
            for metric, score in results['ragas_metrics'].items():
                print(f"  {metric.replace('_', ' ').title():<25}: {score:.3f}")
        else:
            print("  No RAGAS metrics generated.")
            
        print("\n⚡ PERFORMANCE METRICS:")
        print("-" * 40)
        if results.get('custom_metrics'):
            metrics = results['custom_metrics']
            print(f"  Avg Latency:              {metrics.get('avg_latency', 0):.2f}s")
    
    def save_results(self, output_dir: Path, system_name: str):
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.results.get(system_name)
        if result:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{system_name}_results_{timestamp}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            print(f"💾 Saved results to: {filepath}")

def main():
    print("🚀 PHASE 0: BASELINE RAG EVALUATION (DEEPSEEK JUDGE)")
    print("=" * 70)
    
    BASE_DIR = Path(__file__).parent.parent
    GOLDEN_DATASET_PATH = BASE_DIR / "benchmark" / "golden_dataset.json"
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "benchmark" / "results"
    
    # Define files data
    FILES_DATA = [
        {"path": str(DATA_DIR / "ai_infra_compute.txt"), "metadata": {"source": "technical_report"}},
        {"path": str(DATA_DIR / "emerging_ai_hardware.txt"), "metadata": {"source": "article"}},
        {"path": str(DATA_DIR / "enterprise_ai.txt"), "metadata": {"source": "market_report"}},
        {"path": str(DATA_DIR / "llm_training_alignment.txt"), "metadata": {"source": "research_summary"}},
        {"path": str(DATA_DIR / "rag_and_evaluation.txt"), "metadata": {"source": "financial_report"}}
    ]
    
    evaluator = RAGEvaluator()  
    
    try:
        # Load 10 samples (Good enough for testing)
        golden_data = evaluator.load_golden_dataset(GOLDEN_DATASET_PATH, max_samples=10)
    except Exception as e:
        print(f"❌ Failed to load golden dataset: {e}")
        return
    
    print("\n🔧 Initializing Baseline RAG System...")
    valid_files = [f for f in FILES_DATA if os.path.exists(f["path"])]
    if not valid_files:
        print(f"❌ No valid data files found in {DATA_DIR}.")
        return
        
    try:
        baseline_rag = Simple_RAG(valid_files)
        print("✅ Baseline RAG initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    results = evaluator.evaluate_system("baseline_phase0", baseline_rag, golden_data)
    evaluator.print_summary(results)
    evaluator.save_results(OUTPUT_DIR, "baseline_phase0")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# #!/usr/bin/env python3
# #!/usr/bin/env python3
# """
# RAGAS Evaluation Script for Baseline RAG System (Phase 0)
# Optimized for Google Gemini Free Tier Limits (15 RPM).
# """
# import sys
# import os
# import json
# import time
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Any, Tuple
# import numpy as np

# # LangChain Imports
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv

# # Add project root to path
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# load_dotenv()

# # --- CONFIGURATION ---
# # 1. Generator Config (Local - Free)
# GEN_MODEL_NAME = "llama3.1:8b"
# EMBEDDING_MODEL_NAME = "nomic-embed-text:latest"

# # 2. Judge Config (Gemini Free Tier)
# # FIX: Use 'gemini-1.5-flash-latest' to resolve the 404 Error
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# JUDGE_MODEL_NAME = "gemini-1.5-flash-latest" 

# # Initialize components
# GENERATOR_LLM = OllamaLLM(model=GEN_MODEL_NAME, temperature=0.0)
# EMBEDDING = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)

# # Initialize Judge
# if not GEMINI_API_KEY:
#     print("❌ Error: GEMINI_API_KEY not found in .env file")
#     sys.exit(1)

# # Configure Judge with retry logic
# JUDGE_LLM = ChatGoogleGenerativeAI(
#     model=JUDGE_MODEL_NAME,
#     temperature=0,
#     google_api_key=GEMINI_API_KEY,
#     max_retries=3,
#     request_timeout=60
# )

# os.environ["RAGAS_DEBUG"] = "false"

# try:
#     from datasets import Dataset
#     from ragas import evaluate, RunConfig
#     from ragas.metrics import (
#         faithfulness,
#         answer_relevancy,
#         context_precision,
#     )
# except ImportError as e:
#     print(f"❌ RAGAS import error: {e}")
#     print("Install with: pip install ragas datasets langchain-google-genai")
#     sys.exit(1)

# # Import your Simple RAG
# try:
#     from simple_rag import Simple_RAG 
# except ImportError:
#     try:
#         sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'baseline'))
#         from simple_rag import Simple_RAG
#     except ImportError:
#         print("❌ Could not import Simple_RAG. Check your paths.")
#         sys.exit(1)

# class RAGEvaluator:
#     """
#     Comprehensive evaluator for RAG systems using RAGAS.
#     """
    
#     def __init__(self):
#         self.judge_llm = JUDGE_LLM
#         self.embedding = EMBEDDING
#         self.results = {}
        
#     def load_golden_dataset(self, dataset_path: Path, max_samples: int = None) -> List[Dict]:
#         """Load and validate golden dataset."""
#         if not dataset_path.exists():
#             raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
        
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if max_samples and max_samples < len(data):
#             print(f"⚠️  Limiting to {max_samples} samples for quick evaluation")
#             data = data[:max_samples]
        
#         print(f"✅ Loaded {len(data)} samples from golden dataset")
#         return data
    
#     def run_rag_pipeline(self, rag_system: Simple_RAG, questions: List[str]) -> Tuple[List[str], List[List[str]], List[float]]:
#         """Run questions through RAG system and collect outputs."""
#         answers = []
#         contexts_list = []
#         latencies = []
        
#         print(f"🔍 Running {len(questions)} queries through RAG pipeline...")
        
#         for i, question in enumerate(questions, 1):
#             print(f"  [{i}/{len(questions)}] Processing: {question[:60]}...")
#             start_time = time.time()
#             try:
#                 result = rag_system.ask_question(question)
#                 answer = result.get('answer', '')
#                 context_docs = result.get('context', [])
#                 contexts = []
                
#                 if context_docs:
#                     first_doc = context_docs[0]
#                     if hasattr(first_doc, 'page_content'):
#                         contexts = [doc.page_content for doc in context_docs]
#                     elif isinstance(first_doc, str):
#                         contexts = context_docs
#                     else:
#                         contexts = [str(c) for c in context_docs]
                
#                 answers.append(answer)
#                 contexts_list.append(contexts)
#                 latencies.append(time.time() - start_time)
                
#             except Exception as e:
#                 print(f"    ✗ Error: {e}")
#                 answers.append("Error processing query")
#                 contexts_list.append([""]) 
#                 latencies.append(0.0)
        
#         return answers, contexts_list, latencies
    
#     def calculate_ragas_metrics(self, dataset: Dataset) -> Dict[str, float]:
#         """
#         Calculate RAGAS metrics with STRICT THROTTLING for Free Tier.
#         """
#         print("📊 Calculating RAGAS metrics (Free Tier Optimized Mode)...")
        
#         # ESSENTIAL TRIO METRICS (Saves API calls)
#         metrics = [
#             faithfulness,       # Hallucination check (Expensive: ~2 calls)
#             answer_relevancy,   # Quality check (Medium: ~1 call)
#             context_precision   # Retrieval check (Medium: ~1 call)
#         ]
        
#         aggregated_scores = {m.name: [] for m in metrics}
#         total_samples = len(dataset)
        
#         for i in range(total_samples):
#             print(f"    ⏳ Evaluating sample {i+1}/{total_samples}...", end="", flush=True)
            
#             sample_data = dataset.select([i])
            
#             try:
#                 result = evaluate(
#                     dataset=sample_data,
#                     metrics=metrics,
#                     llm=self.judge_llm,
#                     embeddings=self.embedding,
#                     raise_exceptions=False 
#                 )
                
#                 # Store scores safely
#                 for metric_name in aggregated_scores.keys():
#                     if metric_name in result:
#                         score = result[metric_name]
#                         # CRITICAL: Filter out NaNs (failed evaluations)
#                         if not np.isnan(score):
#                             aggregated_scores[metric_name].append(score)
                
#                 print(" Done.")
                
#                 # CRITICAL THROTTLING LOGIC
#                 # Free Tier = 15 Requests Per Minute.
#                 # 1 Sample = ~4 requests.
#                 # We can handle approx 3 samples per minute.
#                 # Sleep 25s ensures we never exceed ~2.4 samples/min.
#                 if i < total_samples - 1:
#                     wait_time = 25
#                     print(f"       💤 Throttling: Sleeping {wait_time}s to preserve Free Tier quota...")
#                     time.sleep(wait_time)
                    
#             except Exception as e:
#                 print(f" Failed! {e}")
#                 import traceback
#                 traceback.print_exc()
        
#         # Calculate Averages
#         final_scores = {}
#         for metric, values in aggregated_scores.items():
#             if values:
#                 final_scores[metric] = float(np.mean(values))
#             else:
#                 final_scores[metric] = 0.0
                
#         return final_scores
    
#     def prepare_ragas_dataset(self, questions, answers, contexts_list, ground_truths) -> Dataset:
#         dataset_dict = {
#             "question": questions,
#             "answer": answers,
#             "contexts": contexts_list,
#             "ground_truth": ground_truths 
#         }
#         return Dataset.from_dict(dataset_dict)

#     def calculate_custom_metrics(self, answers, contexts_list, latencies) -> Dict[str, float]:
#         print("📈 Calculating custom metrics...")
#         custom_metrics = {}
        
#         answer_lengths = [len(a.split()) for a in answers]
#         custom_metrics["avg_answer_length"] = float(np.mean(answer_lengths)) if answer_lengths else 0.0
        
#         retrieved_counts = [len(ctx) for ctx in contexts_list]
#         custom_metrics["avg_retrieved_chunks"] = float(np.mean(retrieved_counts)) if retrieved_counts else 0.0
        
#         if latencies:
#             custom_metrics["avg_latency"] = float(np.mean(latencies))
        
#         return custom_metrics
    
#     def evaluate_system(self, system_name: str, rag_system: Simple_RAG, golden_data: List[Dict]) -> Dict[str, Any]:
#         print(f"\n{'='*70}")
#         print(f"EVALUATING: {system_name.upper()}")
#         print(f"{'='*70}")
        
#         questions = [item["query"] for item in golden_data]
#         ground_truths = [item["answer"] for item in golden_data]
        
#         answers, contexts_list, latencies = self.run_rag_pipeline(rag_system, questions)
        
#         dataset = self.prepare_ragas_dataset(questions, answers, contexts_list, ground_truths)
        
#         ragas_scores = self.calculate_ragas_metrics(dataset)
#         custom_metrics = self.calculate_custom_metrics(answers, contexts_list, latencies)
        
#         results = {
#             "system_name": system_name,
#             "timestamp": datetime.now().isoformat(),
#             "ragas_metrics": ragas_scores,
#             "custom_metrics": custom_metrics
#         }
        
#         self.results[system_name] = results
#         return results
    
#     def print_summary(self, results: Dict[str, Any]):
#         print(f"\n{'='*70}")
#         print(f"EVALUATION SUMMARY: {results['system_name'].upper()}")
#         print(f"{'='*70}")
        
#         print("\n📊 RAGAS METRICS:")
#         print("-" * 40)
#         if results.get('ragas_metrics'):
#             for metric, score in results['ragas_metrics'].items():
#                 print(f"  {metric.replace('_', ' ').title():<25}: {score:.3f}")
#         else:
#             print("  No RAGAS metrics generated.")
            
#         print("\n⚡ PERFORMANCE METRICS:")
#         print("-" * 40)
#         if results.get('custom_metrics'):
#             metrics = results['custom_metrics']
#             print(f"  Avg Latency:              {metrics.get('avg_latency', 0):.2f}s")
    
#     def save_results(self, output_dir: Path, system_name: str):
#         output_dir.mkdir(parents=True, exist_ok=True)
#         result = self.results.get(system_name)
#         if result:
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             filepath = output_dir / f"{system_name}_results_{timestamp}.json"
#             with open(filepath, 'w', encoding='utf-8') as f:
#                 json.dump(result, f, indent=2, default=str, ensure_ascii=False)
#             print(f"💾 Saved results to: {filepath}")

# def main():
#     print("🚀 PHASE 0: BASELINE RAG EVALUATION (GEMINI FLASH OPTIMIZED)")
#     print("=" * 70)
    
#     BASE_DIR = Path(__file__).parent.parent
#     GOLDEN_DATASET_PATH = BASE_DIR / "benchmark" / "golden_dataset.json"
#     DATA_DIR = BASE_DIR / "data"
#     OUTPUT_DIR = BASE_DIR / "benchmark" / "results"
    
#     # Define files data
#     FILES_DATA = [
#         {"path": str(DATA_DIR / "ai_infra_compute.txt"), "metadata": {"source": "technical_report"}},
#         {"path": str(DATA_DIR / "emerging_ai_hardware.txt"), "metadata": {"source": "article"}},
#         {"path": str(DATA_DIR / "enterprise_ai.txt"), "metadata": {"source": "market_report"}},
#         {"path": str(DATA_DIR / "llm_training_alignment.txt"), "metadata": {"source": "research_summary"}},
#         {"path": str(DATA_DIR / "rag_and_evaluation.txt"), "metadata": {"source": "financial_report"}}
#     ]
    
#     evaluator = RAGEvaluator()  
    
#     try:
#         # Load 10 samples (Good enough for testing)
#         golden_data = evaluator.load_golden_dataset(GOLDEN_DATASET_PATH, max_samples=10)
#     except Exception as e:
#         print(f"❌ Failed to load golden dataset: {e}")
#         return
    
#     print("\n🔧 Initializing Baseline RAG System...")
#     valid_files = [f for f in FILES_DATA if os.path.exists(f["path"])]
#     if not valid_files:
#         print(f"❌ No valid data files found in {DATA_DIR}.")
#         return
        
#     try:
#         baseline_rag = Simple_RAG(valid_files)
#         print("✅ Baseline RAG initialized successfully")
#     except Exception as e:
#         print(f"❌ Failed to initialize RAG: {e}")
#         return
    
#     results = evaluator.evaluate_system("baseline_phase0", baseline_rag, golden_data)
#     evaluator.print_summary(results)
#     evaluator.save_results(OUTPUT_DIR, "baseline_phase0")

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# """
# RAGAS Evaluation Script for Baseline RAG System (Phase 0)
# Evaluates against golden dataset and produces metrics for A/B testing.
# """
# import sys
# import os
# import json
# import time
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Any, Tuple
# import numpy as np
# from langchain_ollama import OllamaLLM , OllamaEmbeddings
# from langchain_openai import OpenAI
# from dotenv import load_dotenv
# # Add project root to path
# project_root = Path(__file__).parent.parent
# sys.path.append(str(project_root))

# load_dotenv()

# OPENAI_API = os.getenv("OPEN_AI_API_KEY")
# MODEL_NAME = "llama3.1:8b"
# LLM = OllamaLLM(model=MODEL_NAME,temperature=0.0)
# EMBEDDING_MODEL = "nomic-embed-text:latest"
# EMBEDDING = OllamaEmbeddings(model=EMBEDDING_MODEL)
# # RAG_LLM= OllamaLLM(model="llama3.2:1b",temperature=0.0)
# RAG_LLM= OpenAI(
#             model="gpt-3.5-turbo-instruct",
#             temperature=0,
#             max_retries=2,
#             api_key="..."
# )
# os.environ["RAGAS_DEBUG"] = "true"

# try:
#     from datasets import Dataset
#     from ragas import evaluate, RunConfig
#     from ragas.metrics import (
#         faithfulness,
#         answer_relevancy,
#         context_recall,
#         context_precision,
#         answer_correctness,
#         answer_similarity
#     )
#     # from ragas.metrics.numeric import harmfulness
# except ImportError as e:
#     print(f"❌ RAGAS import error: {e}")
#     print("Install with: pip install ragas datasets")
#     sys.exit(1)

# # Import your Simple RAG
# try:
#     from baseline.simple_rag import Simple_RAG
# except ImportError:
#     print("❌ Could not import Simple_RAG. Check your paths.")
#     sys.exit(1)

# class RAGEvaluator:
#     """
#     Comprehensive evaluator for RAG systems using RAGAS.
#     Supports baseline evaluation and future A/B testing.
#     """
    
#     def __init__(self):
#         """
#         Initialize evaluator.
        
#         Args:
#             eval_llm: LLM to use for RAGAS evaluation (default: gpt-3.5-turbo)
#         """
#         self.eval_llm = LLM
#         self.rag_llm = RAG_LLM
#         self.embedding = EMBEDDING
#         self.results = {}
#         self.query_times = []
        
#     def load_golden_dataset(self, dataset_path: Path, max_samples: int = None) -> List[Dict]:
#         """
#         Load and validate golden dataset.
        
#         Args:
#             dataset_path: Path to golden_dataset.json
#             max_samples: Limit number of samples for quick testing
            
#         Returns:
#             List of evaluation samples
#         """
#         if not dataset_path.exists():
#             raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
        
#         with open(dataset_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         # Validate structure
#         required_keys = {"query", "context", "answer"}
#         for i, item in enumerate(data):
#             missing = required_keys - set(item.keys())
#             if missing:
#                 raise ValueError(f"Item {i} missing keys: {missing}")
        
#         # Limit samples if specified
#         if max_samples and max_samples < len(data):
#             print(f"⚠️  Limiting to {max_samples} samples for quick evaluation")
#             data = data[:max_samples]
        
#         print(f"✅ Loaded {len(data)} samples from golden dataset")
#         return data
    
#     def run_rag_pipeline(self, rag_system: Simple_RAG, questions: List[str]) -> Tuple[List[str], List[List[str]], List[float]]:
#         """
#         Run questions through RAG system and collect outputs.
        
#         Args:
#             rag_system: Initialized Simple_RAG instance
#             questions: List of questions to evaluate
            
#         Returns:
#             Tuple of (answers, contexts_list, latencies)
#         """
#         answers = []
#         contexts_list = []
#         latencies = []
        
#         print(f"🔍 Running {len(questions)} queries through RAG pipeline...")
        
#         for i, question in enumerate(questions, 1):
#             print(f"  [{i}/{len(questions)}] Processing: {question[:60]}...")
            
#             start_time = time.time()
            
#             try:
#                 result = rag_system.ask_question(question)
                
#                 # Extract answer and contexts
#                 answer = result.get('answer', '')
                
#                 # Extract context texts from Document objects
#                 context_docs = result.get('context', [])
#                 if hasattr(context_docs[0], 'page_content'):
#                     contexts = [doc.page_content for doc in context_docs]
#                 else:
#                     print(f"No page Content found#### in {context_docs}")
#                     contexts = []
#                 answers.append(answer)
#                 contexts_list.append(contexts)
                
#                 # Calculate latency
#                 latency = time.time() - start_time
#                 latencies.append(latency)
                
#                 print(f"    ✓ Latency: {latency:.2f}s, Retrieved chunks: {len(contexts)}")
                
#             except Exception as e:
#                 print(f"    ✗ Error: {e}")
#                 answers.append("")
#                 contexts_list.append([])
#                 latencies.append(0.0)
        
#         return answers, contexts_list, latencies
    
#     def prepare_ragas_dataset(self, 
#                             questions: List[str], 
#                             answers: List[str], 
#                             contexts_list: List[List[str]], 
#                             ground_truths: List[str]) -> Dataset:
#         """
#         Prepare data in RAGAS format.
        
#         Args:
#             questions: List of questions
#             answers: List of generated answers
#             contexts_list: List of context lists
#             ground_truths: List of ground truth answers
            
#         Returns:
#             Dataset object for RAGAS evaluation
#         """
#         # Validate lengths
#         lengths = [len(questions), len(answers), len(contexts_list), len(ground_truths)]
#         if len(set(lengths)) != 1:
#             raise ValueError(f"Length mismatch: questions={lengths[0]}, answers={lengths[1]}, contexts={lengths[2]}, ground_truths={lengths[3]}")
        
#         # Convert to RAGAS format
#         dataset_dict = {
#             "question": questions,
#             "answer": answers,
#             "contexts": contexts_list,
#             "ground_truth": ground_truths
#         }
        
#         return Dataset.from_dict(dataset_dict)
    
#     def calculate_ragas_metrics(self, dataset: Dataset) -> Dict[str, float]:
#         """
#         Calculate RAGAS metrics.
        
#         Args:
#             dataset: RAGAS Dataset object
            
#         Returns:
#             Dictionary of metric scores
#         """
#         print("📊 Calculating RAGAS metrics...")
        
#         # Define metrics based on your requirements
#         metrics = [
#             faithfulness,           # Hallucination rate
#             answer_relevancy,      # Answer addresses query
#             context_recall,        # Retrieved relevant info
#             context_precision,     # Precision of retrieved contexts
#             answer_correctness,    # Correctness vs ground truth
#             answer_similarity    # Semantic similarity to ground truth
#                   # Safety metric
#         ]
# # Judge: Needs to be a smart model (Llama 3.1 8B is good)
#         # ollama_judge = self.eval_llm
        
#         # # Embeddings: MUST match what you used for ingestion (nomic-embed-text)
#         # ollama_embeddings = self.embedding
        
#         # # 3. Wrap the LLM so it becomes a 'BaseRagasLLM'
#         # ragas_judge = llm_factory(ollama_judge)
#         run_config = RunConfig(timeout=600, max_workers=5, log_tenacity=True) 
        
#         try:
#             print("    ⏳ Starting RAGAS evaluation (Sequential Mode)... this will take time.")
#             # Run evaluation
#             result = evaluate(
#                 dataset=dataset,
#                 metrics=metrics,
#                 llm=self.rag_llm,
#                 embeddings=self.embedding,
#                 run_config=run_config 
#             )
            
#             # Extract scores
#             scores = {}
#             for metric in metrics:
#                 metric_name = metric.name
#                 if metric_name in result:
#                     scores[metric_name] = float(result[metric_name])
            
#             return scores
            
#         except Exception as e:
#             print(f"❌ RAGAS evaluation failed: {e}")
#             return {}
    
#     def calculate_custom_metrics(self, 
#                                answers: List[str], 
#                                contexts_list: List[List[str]], 
#                                latencies: List[float]) -> Dict[str, float]:
#         """
#         Calculate custom metrics beyond RAGAS.
        
#         Args:
#             answers: Generated answers
#             contexts_list: Retrieved contexts
#             latencies: Query latencies
            
#         Returns:
#             Dictionary of custom metrics
#         """
#         print("📈 Calculating custom metrics...")
        
#         custom_metrics = {}
        
#         # 1. Answer Length Analysis
#         answer_lengths = [len(a.split()) for a in answers]
#         custom_metrics["avg_answer_length"] = np.mean(answer_lengths)
#         custom_metrics["answer_length_std"] = np.std(answer_lengths)
        
#         # 2. Retrieval Metrics
#         retrieved_counts = [len(ctx) for ctx in contexts_list]
#         custom_metrics["avg_retrieved_chunks"] = np.mean(retrieved_counts)
#         custom_metrics["retrieval_coverage"] = sum(1 for cnt in retrieved_counts if cnt > 0) / len(retrieved_counts)
        
#         # 3. Latency Metrics (as per your requirements)
#         latencies_array = np.array(latencies)
#         custom_metrics["avg_latency"] = float(np.mean(latencies_array))
#         custom_metrics["p95_latency"] = float(np.percentile(latencies_array, 95))
#         custom_metrics["first_token_time"] = float(np.mean(latencies_array) * 0.3)  # Approximation
        
#         # 4. Throughput Estimation
#         custom_metrics["estimated_qps"] = 1.0 / custom_metrics["avg_latency"] if custom_metrics["avg_latency"] > 0 else 0
        
#         # 5. Error Rate
#         error_rate = sum(1 for a in answers if "don't know" in a.lower() or len(a.strip()) < 10) / len(answers)
#         custom_metrics["error_rate"] = error_rate
        
#         # 6. Consistency (run a sample query multiple times - future enhancement)
#         custom_metrics["consistency_score"] = 0.0  # Placeholder
        
#         return custom_metrics
    
#     def evaluate_system(self, 
#                        system_name: str, 
#                        rag_system: Simple_RAG, 
#                        golden_data: List[Dict]) -> Dict[str, Any]:
#         """
#         Complete evaluation pipeline for a RAG system.
        
#         Args:
#             system_name: Name of the system (e.g., "baseline")
#             rag_system: Initialized RAG system
#             golden_data: Loaded golden dataset
            
#         Returns:
#             Complete evaluation results
#         """
#         print(f"\n{'='*70}")
#         print(f"EVALUATING: {system_name.upper()}")
#         print(f"{'='*70}")
        
#         # Extract questions and ground truths
#         questions = [item["query"] for item in golden_data]
#         ground_truths = [item["answer"] for item in golden_data]
        
#         # Run RAG pipeline
#         answers, contexts_list, latencies = self.run_rag_pipeline(rag_system, questions)
        
#         # Prepare for RAGAS
#         dataset = self.prepare_ragas_dataset(questions, answers, contexts_list, ground_truths)
        
#         # Calculate metrics
#         ragas_scores = self.calculate_ragas_metrics(dataset)
#         custom_metrics = self.calculate_custom_metrics(answers, contexts_list, latencies)
        
#         # Combine all results
#         results = {
#             "system_name": system_name,
#             "timestamp": datetime.now().isoformat(),
#             "evaluation_config": {
#                 "num_samples": len(golden_data),
#                 "eval_llm": self.eval_llm,
#                 "rag_llm": self.rag_llm
#             },
#             "ragas_metrics": ragas_scores,
#             "custom_metrics": custom_metrics,
#             "per_query_results": [
#                 {
#                     "question": q,
#                     "generated_answer": a,
#                     "retrieved_contexts": ctx,
#                     "ground_truth": gt,
#                     "latency": lt
#                 }
#                 for q, a, ctx, gt, lt in zip(questions, answers, contexts_list, ground_truths, latencies)
#             ]
#         }
        
#         self.results[system_name] = results
#         return results
    
#     def print_summary(self, results: Dict[str, Any]):
#         """Print formatted evaluation summary."""
#         print(f"\n{'='*70}")
#         print(f"EVALUATION SUMMARY: {results['system_name'].upper()}")
#         print(f"{'='*70}")
        
#         # RAGAS Metrics
#         print("\n📊 RAGAS METRICS:")
#         print("-" * 40)
#         if results.get('ragas_metrics'):
#             for metric, score in results['ragas_metrics'].items():
#                 print(f"  {metric.replace('_', ' ').title():<25}: {score:.3f}")
#         else:
#             print("  No RAGAS metrics available")
        
#         # Custom Metrics
#         print("\n⚡ PERFORMANCE METRICS:")
#         print("-" * 40)
#         if results.get('custom_metrics'):
#             metrics = results['custom_metrics']
#             print(f"  Avg Latency:              {metrics.get('avg_latency', 0):.2f}s")
#             print(f"  P95 Latency:              {metrics.get('p95_latency', 0):.2f}s")
#             print(f"  First Token Time:         {metrics.get('first_token_time', 0):.2f}s")
#             print(f"  Estimated QPS:            {metrics.get('estimated_qps', 0):.2f}")
#             print(f"  Error Rate:               {metrics.get('error_rate', 0):.1%}")
#             print(f"  Avg Retrieved Chunks:     {metrics.get('avg_retrieved_chunks', 0):.1f}")
#             print(f"  Retrieval Coverage:       {metrics.get('retrieval_coverage', 0):.1%}")
        
#         # Targets (from your requirements)
#         print("\n🎯 TARGETS (From Your Spec):")
#         print("-" * 40)
#         print("  Faithfulness:              > 0.85")
#         print("  Answer Relevance:          > 0.90")
#         print("  Context Recall:            > 0.80")
#         print("  P95 Latency:               < 15s")
#         print("  Error Rate:                < 2%")
        
#         print(f"\n📅 Evaluated: {results['timestamp']}")
#         print(f"📊 Samples:   {results['evaluation_config']['num_samples']}")
    
#     def save_results(self, output_dir: Path, system_name: str = None):
#             """
#             Save evaluation results to JSON file.
#             Automatically converts non-serializable objects (like OllamaLLM) to strings.
#             """
#             output_dir.mkdir(exist_ok=True)
            
#             # --- Helper Function to Sanitize Data ---
#             def make_serializable(obj):
#                 if isinstance(obj, (str, int, float, bool, type(None))):
#                     return obj
#                 if isinstance(obj, dict):
#                     return {k: make_serializable(v) for k, v in obj.items()}
#                 if isinstance(obj, (list, tuple)):
#                     return [make_serializable(i) for i in obj]
#                 if isinstance(obj, set):
#                     return [make_serializable(i) for i in list(obj)]
#                 # For complex objects (OllamaLLM, Path, etc.), return string representation
#                 return str(obj)
#             # ----------------------------------------

#             if system_name:
#                 results_to_save = {system_name: self.results.get(system_name)}
#             else:
#                 results_to_save = self.results
            
#             for name, result in results_to_save.items():
#                 if result:
#                     # Sanitize the entire result dictionary before saving
#                     safe_result = make_serializable(result)
                    
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = f"{name}_results_{timestamp}.json"
#                     filepath = output_dir / filename
                    
#                     with open(filepath, 'w', encoding='utf-8') as f:
#                         json.dump(safe_result, f, indent=2, ensure_ascii=False)
                    
#                     print(f"💾 Saved {name} results to: {filepath}")
    
#     def compare_systems(self, system_names: List[str]):
#         """
#         Compare multiple systems (for A/B testing).
        
#         Args:
#             system_names: List of system names to compare
#         """
#         print(f"\n{'='*70}")
#         print("A/B TESTING COMPARISON")
#         print(f"{'='*70}")
        
#         # Ensure all systems have results
#         for name in system_names:
#             if name not in self.results:
#                 print(f"⚠️  No results for system: {name}")
#                 return
        
#         # Comparison table
#         print("\n📈 COMPARISON TABLE")
#         print("-" * 80)
        
#         # Header
#         header = ["Metric", "Target"] + system_names
#         print(f"{header[0]:<25} {header[1]:<10} " + " ".join(f"{sys:>10}" for sys in header[2:]))
#         print("-" * 80)
        
#         # Key metrics to compare
#         comparison_metrics = [
#             ("Faithfulness", 0.85, "ragas_metrics", "faithfulness"),
#             ("Answer Relevance", 0.90, "ragas_metrics", "answer_relevance"),
#             ("Context Recall", 0.80, "ragas_metrics", "context_recall"),
#             ("P95 Latency", 15.0, "custom_metrics", "p95_latency"),
#             ("Error Rate", 0.02, "custom_metrics", "error_rate"),
#             ("Avg Latency", 10.0, "custom_metrics", "avg_latency"),
#         ]
        
#         for metric_name, target, category, metric_key in comparison_metrics:
#             row = [metric_name, f"{target}"]
#             for sys_name in system_names:
#                 result = self.results[sys_name]
#                 score = result.get(category, {}).get(metric_key, 0)
                
#                 # Format based on metric type
#                 if "Rate" in metric_name or metric_key == "error_rate":
#                     formatted = f"{score:.1%}"
#                 elif "Latency" in metric_name:
#                     formatted = f"{score:.2f}s"
#                 else:
#                     formatted = f"{score:.3f}"
                
#                 # Add indicator if meets target
#                 if score >= target:
#                     formatted = f"✅ {formatted}"
#                 elif score > 0:
#                     formatted = f"⚠️  {formatted}"
#                 else:
#                     formatted = f"❌ {formatted}"
                
#                 row.append(formatted)
            
#             print(f"{row[0]:<25} {row[1]:<10} " + " ".join(f"{val:>10}" for val in row[2:]))

# def main():
#     """Main evaluation pipeline for Phase 0."""
#     print("🚀 PHASE 0: BASELINE RAG EVALUATION")
#     print("=" * 70)
    
#     # ===== CONFIGURATION =====
#     GOLDEN_DATASET_PATH = Path(r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\benchmark\golden_dataset.json")
#     OUTPUT_DIR = project_root / "benchmark" / "results"
#     MAX_SAMPLES = 5  # Start with 5 for quick testing, increase later
    
#     # Files data matching test_baseline.py
#     FILES_DATA = [
#         {
#             "path": r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\ai_infra_compute.txt",
#             "metadata":{

#         "key_theme": "The mathematics of AI infrastructure"
#     }
#         },
#         {
#             "path": r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\emerging_ai_hardware.txt", 
#             "metadata": {

#         "key_theme": "All about hardwares for AI model training"

#     }
#         },
#         {
#             "path": r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\enterprise_ai.txt",
#             "metadata":{

#         "key_theme": "enterprise_ai"

#     }
#         },
#         {
#             "path": r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\llm_training_alignment.txt",
#             "metadata": {

#         "key_theme": "Facts of llm training"

#     }
#         },
#         {
#             "path": r"C:\Users\Abhi9\OneDrive\Desktop\Multi-Agent RAG\Phase_0\data\rag_and_evaluation.txt",
#             "metadata": {

#         "key_theme": "Rag evaluation"

#     }
#         }
#     ]
    
#     # ===== INITIALIZATION =====
#     evaluator = RAGEvaluator()  
    
#     # Load golden dataset
#     try:
#         golden_data = evaluator.load_golden_dataset(GOLDEN_DATASET_PATH, max_samples=20)
#     except Exception as e:
#         print(f"❌ Failed to load golden dataset: {e}")
#         return
    
#     # Initialize Baseline RAG
#     print("\n🔧 Initializing Baseline RAG System...")
#     try:
#         baseline_rag = Simple_RAG(FILES_DATA)
#         print("✅ Baseline RAG initialized successfully")
#     except Exception as e:
#         print(f"❌ Failed to initialize RAG: {e}")
#         return
    
#     # ===== EVALUATION =====
#     print("\n" + "=" * 70)
#     print("STARTING EVALUATION")
#     print("=" * 70)
    
#     results = evaluator.evaluate_system("baseline_phase0", baseline_rag, golden_data)
    
#     # ===== RESULTS =====
#     evaluator.print_summary(results)
    
#     # Save results
#     evaluator.save_results(OUTPUT_DIR, "baseline_phase0")
    
#     # ===== NEXT STEPS =====
#     print(f"\n{'='*70}")
#     print("NEXT STEPS FOR PHASE 0")
#     print(f"{'='*70}")
#     print("1. Review the saved JSON file for detailed per-query results")
#     print("2. Analyze which queries performed well/poorly")
#     print("3. Adjust RAG parameters (chunk_size, k in retriever)")
#     print("4. Run full evaluation (increase MAX_SAMPLES to 20)")
#     print("5. Proceed to Phase 1: Multi-Agent System")
#     print(f"{'='*70}")

# if __name__ == "__main__":
#     main()