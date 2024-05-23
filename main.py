import pyterrier as pt
from pyterrier.measures import *
from ir_measures import *

if not pt.started():
    pt.init(tqdm="auto", boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

dataset = pt.get_dataset("vaswani")
index = dataset.get_index(variant="terrier_stemmed")

bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
pipelineBo1QE = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "Bo1", "qe" : "on"})
pipelineKLQE = pt.BatchRetrieve(index, wmodel="BM25", controls={"qemodel" : "KL", "qe" : "on"})
rm3_index = dataset.get_index(variant="terrier_stemmed")
pipelineRM3QE = bm25 >> pt.rewrite.RM3(rm3_index) >> bm25
axiomatic_index = dataset.get_index(variant="terrier_stemmed")
pipelineAxiomaticQE = bm25 >> pt.rewrite.AxiomaticQE(axiomatic_index) >> bm25

print(pt.Experiment(
    [bm25, pipelineBo1QE, pipelineKLQE, pipelineRM3QE, pipelineAxiomaticQE],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[RR(rel=1), nDCG@10, nDCG@100, AP(rel=1)],
    names=["BM25", "Bo1QE", "KLQE", "RM3QE", "Axiomatic"],
))