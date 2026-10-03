
import polars as pl
from huggingface_hub import hf_hub_download

HF_REPO = "PatronusAI/financebench"
HF_FILE = "financebench_merged.jsonl"
PDF_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/{doc_name}.pdf"


def load_questions() -> pl.DataFrame:
    path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset")
    return pl.read_ndjson(path)

# questions = load_questions()
# print(questions)

x = ["kundan","sai","ponugumati"]
for i,x in enumerate(x,2):
    print(i,x)