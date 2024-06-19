import pyterrier as pt
from pyterrier.measures import *
from ir_measures import *
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
from sklearn.metrics import r2_score

if not pt.started():
    pt.init(tqdm="auto", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

matplotlib.use('Agg')

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
variant = None
rel = 2
max_rel = 3

# dataset_name = "trec-deep-learning-docs"
# index_name = "terrier_stemmed"
# variant = 'train'
# rel = 1
# max_rel = 1

query_limit = 200

fb_docs = 3
fb_terms = 10

print('Loading models...')

if variant:
    dataset = pt.get_dataset(dataset_name, variant=variant)
else:
    dataset = pt.get_dataset(dataset_name)

models = {}

index = get_index_of_dataset(dataset, index_name)

models['BM25'] = pt.BatchRetrieve(index, wmodel="BM25")
models['DPH'] = pt.BatchRetrieve(index, wmodel="DPH")
# models['DLH'] = pt.BatchRetrieve(index, wmodel="DLH")
models['Bo1'] = models['BM25'] >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
models['Bo1_DPH'] = models['DPH'] >> pt.rewrite.Bo1QueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
models['KL'] = models['BM25'] >> pt.rewrite.KLQueryExpansion(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']
models['RM3'] = models['BM25'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.8) >> models['BM25']
# models['RM3_DPH_0.7'] = models['DPH'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.7) >> models['BM25']
# models['RM3_DLH_0.7'] = models['DLH'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.7) >> models['BM25']
# models['RM3_DPH_0.6'] = models['DPH'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.6) >> models['BM25']
# models['RM3_DLH_0.6'] = models['DLH'] >> pt.rewrite.RM3(index, fb_docs=fb_docs, fb_terms=fb_terms, fb_lambda=0.6) >> models['BM25']
models['Axiomatic'] = models['BM25'] >> pt.rewrite.AxiomaticQE(index, fb_docs=fb_docs, fb_terms=fb_terms) >> models['BM25']

measures = [RR(rel=rel)@10, nDCG@10, MAP(rel=rel)@10, NERR11(max_rel=max_rel)]

print('Models loaded')
print('Starting experiment...')

if variant:
    topics = dataset.get_topics(variant=variant).head(query_limit)
else:
    topics = dataset.get_topics().head(query_limit)

if variant:
    qrels = dataset.get_qrels(variant='train')
else:
    qrels = dataset.get_qrels()

results = pt.Experiment(
    list(models.values()),
    topics,
    qrels,
    eval_metrics=measures,
    names=list(models.keys()),
    # perquery=True,
    verbose=True,
    batch_size=1000,
    # baseline=0,
)

print('Finished experiment')
print('Formatting results...')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(results)

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