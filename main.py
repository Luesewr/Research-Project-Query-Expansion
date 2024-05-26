import pyterrier as pt
from pyterrier.measures import *
from ir_measures import *
from pathlib import Path
import pandas as pd

if not pt.started():
    pt.init(tqdm="auto", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

def get_index_of_dataset(dataset, index_name):
    index_path = str(Path.cwd().joinpath('index', index_name))

    try:
        indexer = pt.TRECCollectionIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    except ValueError:
        index = pt.IndexRef.of(index_path)
    
    return index

dataset_name = "irds:msmarco-passage/trec-dl-2020/judged"
index_name = "trec-dl-2020"

dataset = pt.get_dataset(dataset_name)

models = {}

index = get_index_of_dataset(dataset, index_name)
rm3_index = get_index_of_dataset(dataset, index_name)
axiomatic_index = get_index_of_dataset(dataset, index_name)

models['BM25'] = pt.BatchRetrieve(index, wmodel="BM25")
models['Bo1'] = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "Bo1", "qe" : "on"})
models['KL'] = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "KL", "qe" : "on"})
models['RM3'] = models['BM25'] >> pt.rewrite.RM3(rm3_index) >> models['BM25']
models['Axiomatic'] = models['BM25'] >> pt.rewrite.AxiomaticQE(axiomatic_index) >> models['BM25']

measures = [RR(rel=2, cutoff=10), nDCG@10, MAP(rel=2)@100]

results = pt.Experiment(
    list(models.values()),
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=measures,
    names=list(models.keys()),
    perquery=True,
)

results = results.merge(dataset.get_topics(), on='qid', how='left')
# print(results)
for name in models.keys():
    name_results = results[results['name'] == name]
    # print(name_results)

    for measure in measures:
        measure_str = str(measure)

        measure_results = name_results[name_results['measure'] == measure_str]
        measure_results.sort_values(by='value', axis=0, ascending=False, inplace=True)
        print(measure_results)
