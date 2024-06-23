import pyterrier as pt
from pyterrier.measures import *
from ir_measures import *
from pathlib import Path
import pandas as pd
from datetime import datetime

if not pt.started():
    pt.init(tqdm="auto", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

# Generates or retrieval an already generated index from a dataset
def get_index_of_dataset(dataset, index_name):
    index_path = str(Path.cwd().joinpath('index', index_name))
    # Try to create a new index
    try:
        indexer = pt.index.IterDictIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    # If anything goes wrong or an index already exists an already generated index will be attempted to use
    except ValueError as e:
        print(e, 'Using existing index instead')
        index = pt.IndexRef.of(index_path)
    
    return index

# Will print extra information while the experiments are running
debug = False

# State the different setups of datasets
datasets = [
    {
        "dataset_name": "irds:msmarco-passage/trec-dl-2019/judged",
        "index_name": "trec-dl",
        "variant": None,
        "rel": 2,
    },
    {
        "dataset_name": "irds:msmarco-passage/trec-dl-2020/judged",
        "index_name": "trec-dl",
        "variant": None,
        "rel": 2,
    },
    {
        "dataset_name": "irds:msmarco-passage/trec-dl-hard",
        "index_name": "trec-dl",
        "variant": None,
        "rel": 2,
    },
    {
        "dataset_name": "irds:beir/arguana",
        "index_name": "arguana",
        "variant": None,
        "rel": 1,
    },
    {
        "dataset_name": "irds:antique/test",
        "index_name": "antique",
        "variant": None,
        "rel": 3,
    },
    {
        "dataset_name": "trec-deep-learning-docs",
        "index_name": "terrier_stemmed",
        "variant": 'train',
        "rel": 1,
    }
]

# Limit the amount of topics that will be tested
query_limit = 1000

# Give different parameter sets for the fb_docs and fb_terms for all query expansion models
fb_setups = [(3, 10), (10, 40), (50, 50)]

if debug:
    print('Loading models...')

# Run the set of experiments for every dataset
for dataset_info in datasets:
    dataset_name = dataset_info["dataset_name"]
    index_name = dataset_info["index_name"]
    variant = dataset_info["variant"]
    rel = dataset_info["rel"]

    print("Dataset:", dataset_name)

    # Get the dataset, with the variant if it's defined
    if variant:
        dataset = pt.get_dataset(dataset_name, variant=variant)
    else:
        dataset = pt.get_dataset(dataset_name)

    models = {}

    # Get ths index
    index = get_index_of_dataset(dataset, index_name)

    # Run experiments for every query expansion setup
    for fb_docs, fb_terms in fb_setups:
        print("fb_docs:", fb_docs, ", fb_terms:", fb_terms)

        # Create the query expansion pipelines
        models['BM25'] = pt.BatchRetrieve(index, wmodel="BM25")
        models['Bo1'] = models['BM25'] >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
        models['KL'] = models['BM25'] >> pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
        models['RM3'] = models['BM25'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.8) >> models['BM25']
        models['Axiomatic'] = models['BM25'] >> pt.rewrite.AxiomaticQE(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']

        # Set up the measures
        measures = [RR(rel=rel)@10, RR(rel=rel)@100, RR(rel=rel)@1000, nDCG@10, nDCG@100, nDCG@1000, MAP(rel=rel)@10, MAP(rel=rel)@100, MAP(rel=rel)@1000]

        if debug:
            print('Models loaded')
            print('Starting experiment...')

        # Retrieve the topics and qrels
        if variant:
            topics = dataset.get_topics(variant=variant)
        else:
            topics = dataset.get_topics()

        if query_limit:
            topics = topics.head(query_limit)

        if variant:
            qrels = dataset.get_qrels(variant='train')
        else:
            qrels = dataset.get_qrels()

        result_rows = []

        # Run the experiment per model so the execution time can be kept track of
        for model_name, model_obj in models.items():
            start_time = datetime.now()
            results = pt.Experiment(
                [model_obj],
                topics,
                qrels,
                eval_metrics=measures,
                names=[model_name],
                verbose=debug,
                batch_size=1000,
            )
            end_time = datetime.now()
            time_passed = (end_time - start_time).total_seconds()
            results['time'] = time_passed

            if debug:
                print(model_name, time_passed)

                print('Finished experiment')
                print('Formatting results...')
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                    print(results)
            result_rows.append(results)

        # Merge the results of the models
        results = pd.concat(result_rows)

        bm25_time = results.loc[results['name'] == 'BM25', 'time'].values[0]
        results['corrected_time'] = results['time'] - 2 * bm25_time

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(results.round(4))
