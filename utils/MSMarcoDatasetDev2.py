from sentence_transformers import  evaluation

from utils.config import DATA_FOLDER

def make_evaluator():
    dev_queries_file = DATA_FOLDER / 'queries.dev.tsv'
    qrels_filepath = DATA_FOLDER / 'qrels.dev.tsv'
    collection_filepath = DATA_FOLDER / 'collection.tsv'

    corpus = {}             #Our corpus pid => passage
    dev_queries = {}        #Our dev queries. qid => query
    dev_rel_docs = {}       #Mapping qid => set with relevant pids
    needed_pids = set()     #Passage IDs we need
    needed_qids = set()     #Query IDs we need

    # Load the 6980 dev queries
    with open(dev_queries_file, encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            dev_queries[qid] = query.strip()


    # Load which passages are relevant for which queries
    with open(qrels_filepath) as fIn:
        for line in fIn:
            qid, _, pid, _ = line.strip().split('\t')

            if qid not in dev_queries:
                continue

            if qid not in dev_rel_docs:
                dev_rel_docs[qid] = set()
            dev_rel_docs[qid].add(pid)

            needed_pids.add(pid)
            needed_qids.add(qid)


    # Read passages
    with open(collection_filepath, encoding='utf8') as fIn:
        for line in fIn:
            pid, passage = line.strip().split("\t")

            if pid in needed_pids:
                corpus[pid] = passage.strip()

    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs,
                                                            show_progress_bar=True,
                                                            corpus_chunk_size=100000,
                                                            precision_recall_at_k=[10, 100],
                                                            name="msmarco dev")
    
    return ir_evaluator