# gTDR

**gTDR** is a graph-based deep learning toolkit for temporal, dynamic, and relational data. It offers a variety of graph neural network (GNN) tools for solving graph problems, including, for example, node property predictions and link predictions on static or dynamic graphs. The tools can also be used to solve time series problems (e.g., forecasting and anomaly detection) through graph structure modeling and learning. The toolkit implements recently published methods that have demonstrated competitiveness over strong baselines.

The toolkit features a few characteristics that differentiate itself from other GNN libraries: ease of use, scalability, and comprehensive data tasks.

**Ease of use.** gTDR is a toolkit rather than a software library. Graph methods are off-the-shelf from the toolkit and using them is as simple as writing a Python notebook. The toolkit reduces coding efforts and lowers the bar of adoption. It can be used by not only experts, but also those who are not knowledgeable of the method/model that solves their use cases.

**Scalability.** gTDR includes some of the state-of-the-art scalable GNN trainers. It can be used to address large-scale graph problems such as node property prediction.

**Comprehensive data tasks.** gTDR handles not only graph data but also time series data and tabular data. Thanks to graph structure modeling, it is found that learning the (hidden) graph structure among time series or usual tabular data may significantly improve the performance of the downstream task, including forecasting and anomaly detection. Being a graph toolkit, gTDR not only solves graph problems but also uses graphs to solve problems that do not come with graph data in the first place.

## Installation

Download the repository:
```bash
git clone <git repo url>
cd gTDR
```

Create and set up python environment:
```bash
conda env create -f environment.yml
conda activate gtdr
pip install -r requirements.txt
```

Install additional packages (SALIENT's fast sampler):
```bash
cd packages/fast_sampler
python setup.py install
cd ../..
```

Install this toolkit:
```bash
pip install -e .
```

## Demos

The toolkit supports many use scenarios and applications. The graph can be static or dynamic; the graph may even not exist. Other than graphs, the data can also be time series or tabular. The tasks range from node property prediction, time series forecasting, to anomaly detection. The provided tools can run on one or multiple GPUs. Below lists the use cases supported by gTDR and the corresponding demos.

| Use Case                                                                      | Compute       | Demo                                                      | Method                                                          | Demo Dataset |
| ----------------------------------------------------------------------------- | ------------- | --------------------------------------------------------- | --------------------------------------------------------------- | ------------ |
| Static graph, tabular node features, node classification                      | single GPU    | [ipynb](examples/FastGCN_Cora.ipynb)                      | [FastGCN](https://jiechenjiechen.github.io/pub/fastgcn.pdf)     | Cora         |
| Static graph, tabular node features, node classification                      | multiple GPUs | [ipynb](examples/SALIENT_ogbn_arxiv_single_machine.ipynb) | [SALIENT](https://jiechenjiechen.github.io/pub/salient.pdf)     | obgn-arxiv   |
| Dynamic graphs, tabular node features, node classification                    | single GPU    | [ipynb](examples/EvolveGCN_H_Elliptic.ipynb)              | [EvolveGCN](https://jiechenjiechen.github.io/pub/evolvegcn.pdf) | Elliptic     |
| Dynamic graphs, tabular node features, link prediction                        | single GPU    | [ipynb](examples/EvolveGCN_O_sbm50.ipynb)                 | [EvolveGCN](https://jiechenjiechen.github.io/pub/evolvegcn.pdf) | sbm50        |
| Static graph (optional), multiple multivariate time series, forecasting       | single GPU    | [ipynb](examples/GTS_METR_LA.ipynb)                       | [GTS](https://jiechenjiechen.github.io/pub/gts.pdf)             | METR-LA      |
| Static graph (optional), multiple multivariate time series, anomaly detection | single GPU    | [ipynb](examples/GANF_METR_LA.ipynb)                      | [GANF](https://jiechenjiechen.github.io/pub/ganf.pdf)           | METR-LA      |
| Tabular features, DAG structure learning                                      | single GPU    | [ipynb](examples/DAG_GNN_synthetic.ipynb)                 | [DAG-GNN](https://jiechenjiechen.github.io/pub/DAG-GNN.pdf)     | synthetic    |

## References

1. Chen et al.
[FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://jiechenjiechen.github.io/pub/fastgcn.pdf).
In *ICLR*, 2018.

1. Kaler et al.
[Accelerating Training and Inference of Graph Neural Networks with Fast Sampling and Pipelining](https://jiechenjiechen.github.io/pub/salient.pdf).
In *MLSys*, 2022.

1. Pareja et al.
[EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://jiechenjiechen.github.io/pub/evolvegcn.pdf).
In *AAAI*, 2020.

1. Shang et al.
[Discrete Graph Structure Learning for Forecasting Multiple Time Series](https://jiechenjiechen.github.io/pub/gts.pdf).
In *ICLR*, 2021.

1. Dai and Chen.
[Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series](https://jiechenjiechen.github.io/pub/ganf.pdf).
In *ICLR*, 2022.

1. Yu et al.
[DAG-GNN: DAG Structure Learning with Graph Neural Networks](https://jiechenjiechen.github.io/pub/DAG-GNN.pdf).
In *ICML*, 2019.
