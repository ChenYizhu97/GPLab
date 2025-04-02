# GPLab

GPLab is designed to make the evaluation of graph pooling approaches with different settings less laborious. With GPLab, it is easy to compare the behavior of pooling methods under various GNN backbones, hyperparameters, and tasks. The evaluation settings and results are trackable and friendly to query operations.

## Experiment Mapping

The core of GPLab is how it treats, organizes, and tracks evaluations. In GPLab, an execution of the main Python script is recognized as an evaluation, which trains and evaluates a model on a given task several times and produces some results. The configurations, settings, and results are persisted together as a JSON object to record information about this evaluation.

Precisely, an execution of the main script conducts an evaluation and produces a record represented by a JSON object that tracks the configurations, settings, and results. A series of evaluations produces a JSON object stream, on which we can query and analyze multiple results.

## Query Design

The implementation of queries on records exploits the advantages of the JSON object data structure. A query operation is decomposed into two steps on a JSON object, which are **filter** and **read**. The filter step compares the items to filter the records satisfying conditions, and the read step returns the related keys and values of the required records.

The figure below shows an example of a query and the structure of a record.
![A query.](GPLab.png)