# Primer: neural activity forecasting
The goal of this research is to propose improvements/novel model architectures (that are neurobiologically sound/motivated), for predicting how neural activity signal will change over time.
# Intuition: combining temporal and spatial information
At the lowest level, how a neuron's activity might change depends on:
1. Its previous activity
	- e.g., if it just spiked, it can't immediately spike again, because voltage-gated sodium channels can't reopen immediately after being activated
2. Activity of its neighbor
	- e.g., suppose neighbor neurons (with axon connection to this neuron) just fired. Potential difference might build up, causing this neuron to spike

Existing methods extrapolate this idea to the population level:
- A cluster/population of neurons is referred to as a *channel* (i.e., the set of neurons measured by a single µECoG node)
- Channel $i$ at timestep $t$ is represented by an embedding $e_{i,t}$
- Previous work introduces different methods to update $e_{i,t}$, utilizing:
	- temporal information $e_{i,0:t-1}$ (the representation of that channel at previous time steps)
	- spatial information $e_{j,t}$ (the representation of other channels at the current time step)
# Encoding spatial information: GNN and Transformer
## GNN: Core idea
The idea behind GNNs is:
- Suppose we can represent our data using a graph, such as
	- YouTube recommender system: 
		- an user or a video is a node in the graph
		- an edge between a user and a video means the user had watched that video
	- Brain:
		- a neuron cluster is a node
		- an edge between two nodes means there exist a connection/pathway between the two neuron clusters 
- We wish to enforce the inductive bias that a node is only directly influenced by its neighbors
	- a node's neighbor refer to other nodes that are directly connected to it through one edge
	- e.g., information cannot directly flow from left V1 to right V1, because there's no direct cortical connection between them
- In other words, GNNs learn a representation for every node, assuming each node is updated using only information from neighboring nodes
## Two phases of GNN: neighbor selection & aggregation
Hence, a GNN approach can be broken down into:
1. Constructing the graph:
	- 