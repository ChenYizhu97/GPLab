# Graph Pooling Lab

GPLab is a CLI app which offers a playground where you can quickly test, evaluate and analysis graph pooling layers. 

## Why GPLab
When carrying on experiments regarding to graph pooling, It could be messy to adjust the model architecture and tune the experiments setting, and keep the track of them at the same time.
GPLab makes some effort to eliminate this kind of mess. 

First, the model architecture and experiment setting could be changed by simply modifying the config file, which is a toml file readable to user, so you do not need to change all setting respectively in the code and have a top-level sense about the experiment.

Then, when you chose to save the experiment results, this config information is also save with them so that you can easily keep track of each experiment. The results are saved as the json object stream, so that you can easily save and read all the information that you are insterested in.

Besides, GPLab help you make the experiments reproducible and provide simple script to query the experiments results that you are interested in.

Moreover, as a playground for evaluating graph pooling, we provides several default models designed to evaluate certain properties of graph pooling and two naive pooling layers to have a basic validation on the graph pooling layers.


## The precedure
GPLab use the common procedure to evaulate and analysis the graph pooling layers. In this procedure, a basic GNN model is defined to evaluate certain properties of graph pooling. The different graph pooling layers are inserted into the model and the model is tested on several graph datasets. The results of different graph poolings are compared to reveal the performance of the desired graph pooling with respect to certain properties.

For each experiment, if the dataset doesn't have its division, the dataset is randomly divided into training, validation and test dataset by the ratio 8:1:1. The randomness is maintained by GPLab using a fixed stream of seeds, which could be automatically produced by GPLab or be set manually. This design makes sure that when you have two or more series of experiments, the experiments of the same position use the same seed, which reduces the influence of randomness when comparing different graph poolings. The fixed seeds also ensures the reproducibility of the experiments.

For the training, we use the early stopping with the patience. If the performance of model on validation dataset doesn't increase for some epochs (the patience), we think that the model is overfitting on training dataset and stop the training. The model take the weights from the epoch where overfitting does not happend, which is (current epoch - patience). Then the model performance are evaluated on the test dataset.

GPLab will print the experiment information and outputs on the console, but you can also chose to save it into file. The experiment information contains the model and training setting, the graph pooling and the dataset. The outputs include the validation loss, test accuracy and epochs run. There is a query.py script which is also a simple CLI app that you could query the results from the saved file.

### Default pooling and model
The graph poolings fall into different types. Depending on how many original nodes contribute to each pooled node, the graph pooling could be sparse or dense. If there are constant nodes that contribute to each pooled node (O(1)), the graph pooling is sparse. If the number of nodes that contribute to each pooled node depends on the number of original nodes (O(n)), the graph pooling is dense. We proposed two naive pooling for each of these two types, which only pool the graph using the node attributes and do not take the advantage of graph topology. These naive pooling could be a quick tool to validate whether your design of new graph pooling works.

One basic model is designed to evaluate the ability of the graph pooling to preserve task-related information. The model consists of pre-transformed mlp, convolution layers before pooling, inserted pooling layer, convolution layers after pooling and readout mlp. Different graph pooling layers are inserted to the basic model on the specific position. if the pooled graph performs better on given tasks, we say that the corresponding graph pooling is able to preserve more task-related information.

(other model)

### Model and experiment setting
Though the general hierarchy of basic model is fixed, the details could be adjusted to form a refined customized model. You may simple do this by modifying the config file at location config/config.toml, which is readable and self-explanatory. The model is generated from the config file so you don't have to change the code of the model. You may change the number of layers, hidden parameters, activation of the model and so on. 

To have a general perception of the experiments, you may also change the experiment setting at the same location, such as learning rate, batch size, maximum epochs and anything regarding to the experiments. However, you may feed some of the experiments setting as command parameters when you run the main script, depending on your preference.


### Data maintain and querry
The data is mantained as json object stream. There are several benefits to save data as this type:
1. Ultilize the json library to manipulate the data conveniently.
2. Saving data is flexible. You could save anything you interested in by simply add another item in the json object.
3. Data querying is efficient. The data querying could be break into two steps. First, filter the results by the key and value. Second, only read the value of interested key. Composing these two steps, you could quickly focus on the parts of the results that you are insterested in.

### Controling the experiments

Have a try at the properties GPLab provides:
1. Reproducibility. GPLab makes experiments reproducible by using the fixed seeds for each experiment in the sequence. The seeds are automatically generated and maintained, but you can set it mannully in the location config/seeds.
2. Default Models to evaluate the properties of graph pooling layer. For example, the models to evaluate the ability of pooling layer to preserving task-related information, the ability to improve task performance, and the expresiveness (see the paper).
3. Customizing the components of your models, tuning the hyperparameters of trainning and adjusting your setting of experiments by modifying the configuration file easily.
4. Naive pooling layers as a baseline. We proposed two naive pooling layers, one of which is sparse and another is dense, to give an relatively objective baseline to demostrate whether a pooling layer works at least.
5. Data maintaining and querrying. The experiments results are saved as json objects stream, which keeps the information about results, pooling layers that evaluated, trainning and experiments setting and anything helpful. Besides, we provide a querry script that querries the data in SQL style to help analysis the results. For example, selecting the results regarding to certain pooling layer or/and dataset.

# Usage
# Set model and trainning
The preset models have general architechture with some details you may adjust. For example, for the classifier model that evaluates the ability of pooling layer to preserve taske-related information, you may change the type of convolution layers, the number of convolution layers before and after the pooling layer, the activation function, the pre-transform MLPs, the readout MLPs and so on. All you need to do is to modify the setting in the config file, then when the model object is initiated, the model class will read the config file and generate the desired model for you. You may also custom the trainning setting such as the learning rate, patience, 
## Run experiment
Run an experiment using the script main.py. The script takes some arguments as follows:

pool_ratio: float, pooling ratio.
pooling: str. Pooling layer to use.
dataset: str. Dataset to evaluate on.
config: str. Location where config file is. Default value is config/config.toml
logging: Optional[str]. The file to save experiment results. None if don't save results.
comment: Optional[str]. Any comment to save with the experiment results.

For example. The following command evaluates model defined by config/config.toml with pooling layer lspool on the dataset proteins and saves the results in file logs/example.log
`python main.py --pooling lspool --pool_ratio 0.5 --logging logs/example`

You may compose a batch of python command in a shell script to run a batch of experiments.

The results are of experiments are saved as json objects stream. The json objects is flexible which allows you to save any information helpful about the experiments, besides, query such data structure is easy because we can simply break the query operations into a sets of filtering operations on the dict that is loaded from the json objects.
The test accuracy, validation loss and epoches are saved with other descriptive information about the experiments, such as trainning and model setting.  The mean and std of test accuracy is calculated as the metrics.

## Query the results
Run a query using the script querry.py. The script takes some arguments as follows:

file: str. The file that keeps the json objects.
pool: Optional[str]. Query the experiments results about certain pooling layer.
dataset: Optional[str]. Query the experiments results about certain dataset.
comment: Optional[str]. Query the experiments results that contains specific comment.

For example, the follwing command will query the experiments results about lspool on the dataset proteins given the log file logs/example.log
`python querry.py logs/example.log --pool lspool --dataset proteins`

