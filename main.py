import pyterrier as pt
from pyterrier.measures import *
from ir_measures import *
from pathlib import Path
import pandas as pd
# import matplotlib
# import matplotlib.pyplot as plt
# import mpld3
# import numpy as np
# from sklearn.metrics import r2_score
from datetime import datetime

if not pt.started():
    pt.init(tqdm="auto", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

matplotlib.use('Agg')

def get_index_of_dataset(dataset, index_name):
    index_path = str(Path.cwd().joinpath('index', index_name))

    try:
        indexer = pt.index.IterDictIndexer(index_path)
        # indexer = pt.TRECCollectionIndexer(index_path)
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    except ValueError as e:
        print('Using existing index instead', e)
        index = pt.IndexRef.of(index_path)
    
    return index

debug = False

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

query_limit = 1000

fb_setups = [(3, 10), (10, 40), (50, 50)]

if debug:
    print('Loading models...')

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

        models['BM25'] = pt.BatchRetrieve(index, wmodel="BM25")
        models['Bo1'] = models['BM25'] >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
        models['KL'] = models['BM25'] >> pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
        models['RM3'] = models['BM25'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.8) >> models['BM25']
        models['Axiomatic'] = models['BM25'] >> pt.rewrite.AxiomaticQE(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']

        measures = [RR(rel=rel)@10, RR(rel=rel)@100, RR(rel=rel)@1000, nDCG@10, nDCG@100, nDCG@1000, MAP(rel=rel)@10, MAP(rel=rel)@100, MAP(rel=rel)@1000]

        if debug:
            print('Models loaded')
            print('Starting experiment...')

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

        for model_name, model_obj in models.items():
            start_time = datetime.now()
            results = pt.Experiment(
                [model_obj],
                topics,
                qrels,
                eval_metrics=measures,
                names=[model_name],
                # perquery=True,
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
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):  # more options can be specified also
                    print(results)
            result_rows.append(results)

        results = pd.concat(result_rows)

        bm25_time = results.loc[results['name'] == 'BM25', 'time'].values[0]
        results['corrected_time'] = results['time'] - 2 * bm25_time

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):  # more options can be specified also
                print(results.round(4))

# results = results.merge(topics, on='qid', how='left')

# for name in models.keys():
#     print(name, ':')
#     name_results = results[results['name'] == name]
# #     # print(name_results)

#     # for measure in measures:
#     measure = measures[1]
#     measure_str = str(measure)

#     measure_results = name_results[name_results['measure'] == measure_str]

#     word_scores = {}

#     for index, row in measure_results.iterrows():
#         split = row['query'].split()
#         weight = 1 / len(split)

#         for word in split:
#             value = row['value']

#             if word in word_scores:
#                 cur_value, count = word_scores[word]
#                 word_scores[word] = (((cur_value * count) + (value * weight)) / (count + weight), count + weight)
#             else:
#                 word_scores[word] = ((value * weight), weight)

#     print(sorted([word_score_item for word_score_item in word_scores.items() if word_score_item[1][1] >= 1], key=lambda entry: entry[1][0], reverse=True))
    # x = measure_results['query'].apply(lambda x: len(x.split()))
    # y = measure_results['value']

    # z = np.polyfit(x, y, 2)
    # p = np.poly1d(z)
    # r2 = r2_score(y, p(x))
    # print(r2)
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(x, y)
    # ax.plot(x, p(x))
    # ax.set_title('Value vs Word Count')
    # ax.set_xlabel('Values')
    # ax.set_ylabel('Word Count')
    # html_str = mpld3.fig_to_html(fig)
    # with open('plot.html', 'w') as f:
    #     f.write(html_str)
    # exit()



#         measure_results.sort_values(by='value', axis=0, ascending=False, inplace=True)
        # print(measure_results)

# print('Results formatted')

# results_wide = results.pivot_table(
#     index='name',
#     columns='measure',
#     values='value',
#     aggfunc='mean'
# ).reset_index()

# print(results_wide)