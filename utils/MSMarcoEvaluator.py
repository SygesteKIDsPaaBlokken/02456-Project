from pathlib import Path
from collections import defaultdict

from sentence_transformers.evaluation import InformationRetrievalEvaluator

from utils.config import DATA_FOLDER, VERBOSE


def make_evaluator():
    dev_queries_path = DATA_FOLDER / 'queries.dev.tsv'
    qrels_path = DATA_FOLDER / 'qrels.dev.tsv'
    collection_path = DATA_FOLDER / 'collection.tsv'

    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    needed_pids = set()     #Passage IDs we need

    # Load the 6980 dev queries
    dev_queries = get_dev_queries(dev_queries_path)

    # Load which passages are relevant for which queries
    dev_rel_docs, needed_pids = get_relevant_passages_for_queries(qrels_path, dev_queries)

    # Read passages
    corpus = get_corpus(collection_path, needed_pids)

    ir_evaluator = InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=VERBOSE,
                                                            corpus_chunk_size=100_000,
                                                            precision_recall_at_k=[10, 100],
                                                            name="msmarco dev")
    return ir_evaluator


def get_dev_queries(dev_queries: Path) -> dict[str, str]:
    dev_queries = {} 
    with open(dev_queries, encoding='utf8') as dev_queries_file:
        for line in dev_queries_file:
            qid, query = extract_data_from_line(line)
            dev_queries[qid] = format_query(query)
    
    return dev_queries


def get_relevant_passages_for_queries(qrels: Path, dev_queries: dict[str, str]) ->tuple[dict, set]:
    dev_rel_docs = defaultdict(set())
    needed_pids = set()

    with open(qrels) as qrels_file:
        for line in qrels_file:
            qid, _, pid, _ = extract_data_from_line(line)

            if qid not in dev_queries: # Do not include queries not in dev set
                continue

            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
    
    return dict(dev_rel_docs), needed_pids

def get_corpus(corpus: Path, needed_pids: set) -> dict[str, str]:
    corpus = dict()
    with open(corpus, encoding='utf8') as corpus_file:
        for line in corpus_file:
            pid, passage = extract_data_from_line(line)

            if pid in needed_pids:
                corpus[pid] = format_passage(passage)

    return corpus
    
def extract_data_from_line(line: str):
    return line.strip().split("\t")

def format_query(query: str):
    return query.strip()

format_passage = format_query