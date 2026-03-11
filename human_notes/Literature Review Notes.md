# Diffusion Convolution Recurrent Neural Network
## Innovation: Diffusion Convolution (a new aggregation function)
- Models the diffusion of information across neighbor (multiple hops in one aggregation)
	- Equivalent to weighted sum of k-hop neighbor information)
- Allows learning different kernels (each kernel corresponding to a set of weights, each weight applies to information at k-step away)
**Adjacency matrix is fixed**

Temporal information is the result of spatial interaction!! 
## Encoder-decoder for temporal representation learning and prediction
The above diffusion convolution operation basically is a way of updating all the nodes in a graph using neighbor's information.

A RNN with diffusion convolution is built by changing the learned weight matrices to diffusion convolution operations
- e.g., previous update gate might be a sigmoid of MLP(previous hidden state, which looks like one embedding for every node)
- Now, it might be sigmoid of DiffusionConvolution($h_{t-1}$) 
- This means we are strictly enforcing the graph structure and relevant inductive biases about how spatial interaction should be done into the model

Essentially, GRU aggregates temporal information about every node/channel, using diffusion convolution layer.

We then take the final hidden representation extracted by GRU ( $h_{T}$), and use that as the initial hidden state of the decoder GRU.
- A prediction head is learned to map hidden state to output
- GRU uses ground truth $x_{t}$ and previous hidden state $h_{t-1}$ to produce next hidden state $h_{t}$

*Schedule sampling*:
- At test time, GRU is used auto-regressively (output of prediction head is feed into next time step as input)
- At train time, we cannot always provide GRU with the actual ground truth, because perhaps GRU's prediction won't be that accurate, and hence the prediction output at test time will be out of distribution of what GRU sees during training
- Hence, for each step, schedule sampling technique is applied to feed the ground truth $x_t$ into GRU with probability $\epsilon$, its own prediction $y_{t}$ with probability $1 - \epsilon$
# Graph WaveNet
Adjacency matrix: 
- Actually uses multiple ones (like different adajacency matrix for multiplicative and additive module in AMAG)
	- One is forward direction, one is backward (a relic of diffusion convolution and how traffic flow is directional)
	- The third one is a learned adjacency matrix, computed via similarity between learned embeddings of nodes (note that these are fixed, learned node embeddings. They don't adapt to input.)

Temporal convolution + Graph convolution (spacial)
- At each layer, the input is, for each node, one embedding for each time step
- Temporal convolution is first runned
	- It independently act on each node , processing so that the embedding at each time step aggregates information from embedding of previous time steps
- Then, graph convolution is runned
	- Independently act on each time step
	- At each time step, apply diffusion convolution on the graph
# GraphS4 Former
## How to construct adjacency matrix
- Obtain vector embedding of each node in a certain time interval 
	- Done through averaging the embedding learned by S4 across time in that interval
- Use self-attention (query key value) to determine attention weight (i.e., edge strength)
- Apply sparsity constraint
## Aggregation function 
- GIN (graph isomorphism network): weighed sum of neighbors, then pass through MLP
## Potential limitation:
- wasteful... especially given done all this, but at the end it just does temporal and graph pooling (temporal as averaging the embedding of every node across time intervals; graph as in averaging all the nodes' embedding in a graph into a single vector)