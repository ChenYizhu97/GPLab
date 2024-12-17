
# GPLab
GPLab is implemented to make evaluation on graph pooling approches with different settings less laborious. With GPLab, It is easy to compare the behavior of pooling methods under various GNN backbones, hyperparameters and tasks. The evaluation settings and results are trackable and friendly to query operations. 

## Experiment Mapping
The core of GPLab is how it *treats*, *organizes* and *tracks* the evaluation. In GPLab, an execution of the main python script is recognized as an evaluation, which trains and evaluates a model on a given task several runs and produces some results. The configurations, settings and results are persisted together as a json object to record information about this evaluation. 

Precisely, an execution of main script conducts an evaluation and produces a record represented by a json object that tracks the configurations, settings and results. A series of evaluations produces a json object stream, on which we could query and analysis multiple results.

## Query Design
The implementation of query on records exploiting the advantages of json object data structure.A query operation is decomposed into two steps on json object, which are **filter** and **read**. The filter step compares the items to filter the records satisfying conditions, and the read step returns the related keys and values of required records. 

The figure below shows an example of query and the structure of a record.
![A query.](GPLab.png)
