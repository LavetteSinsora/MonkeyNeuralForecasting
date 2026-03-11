# Whole process — how is data transformed
**The full sequence for one channel c**

**Step 1 — Temporal Encoding (TE GRU):** The GRU processes the raw input sequence for channel c: `[x_{c,0}, x_{c,1}, ..., x_{c,9}]`, where each `x_{c,t}` is a 9-dimensional vector (the 9 features at timestep t). The GRU unrolls across all 10 timesteps and produces a hidden state at _each_ timestep, giving a sequence of hidden states: `[h_{c,0}, h_{c,1}, ..., h_{c,9}]`, each 64-dimensional. So after TE, every channel has a sequence of 10 hidden states, not just one.

**Step 2 — Spatial Interaction (SI):** This runs once _per timestep_, not once overall. At each t, it takes the hidden states of all channels at that timestep — `h_{0,t}, h_{1,t}, ..., h_{C,t}` — runs the additive and multiplicative message passing, and produces updated states `z_{0,t}, z_{1,t}, ..., z_{C,t}`. So after SI, every channel still has a sequence of 10 hidden states, just spatially updated: `[z_{c,0}, z_{c,1}, ..., z_{c,9}]`.

**Step 3 — Temporal Readout (TR GRU):** Now, exactly as you described. For channel c, the TR GRU takes the sequence `[z_{c,0}, z_{c,1}, ..., z_{c,9}]` as input (each element is 64-dimensional), unrolls across all 10 timesteps, and at the very end we take only the **final hidden state** `h_{TR, c, 9}` (64-dimensional). This single vector is a compressed summary of everything — the temporal dynamics of channel c, plus the spatial interaction signals baked in at each step.

**Step 4 — Linear readout:** A single linear layer maps that 64-dimensional final hidden state to 10 scalar values: `[LMP_{c,10}, LMP_{c,11}, ..., LMP_{c,19}]`. That's your prediction for all 10 future LMP values for channel c, produced in one shot.
# Interesting finding
## 1. Demonstrated that forecasting objective is good for model to learn cross-channel interaction
Paper considered a task with synthetic dataset (so we know ground-truth channel-interaction), and trained the same AMAG model with two loses (reconstruction and prediction). 
- Result shows that single-step forecasting objective results in better learned interaction/adjacency matrix
- Hypothesis is that forecasting requires model to better learn how channels interact with each other (because channel-interaction primarily determines future)
- Reconstruction, on the other hand, don't harshly require that dire desire of learning interaction matrix 
## 2. Transformer consistently underperforms RNNs
Perhaps due to temporal transformers are good at long-range dependency, while for neural activity, only the previous few timestep's information is needed, necessary, and will influence/determine the future activity
## 3. Behavior decoding from forecasted neural signal
Paper trained a linear decoder that takes in neural signal and output behavior property (e.g., velocity of hand)
- Trained using ground truth neural signal + behavior pairs
- Decoding using real signal get $R^2$ of 0.0656, while others consistently underperform (AMAG performs the best — 0.0555)
## 4. Random init and correlation init of adjacency matrix lead to similar performance/learned adjacency
- Nonetheless, random init leads to a less stable training process
Global and local patterns
- The global pattern from different learned adjacency matrices are quite similar, yet the local patterns can be different
	- Suggests that different local patterns/configurations can lead to similar performance/can all explain the same underlying data
	- Interesting point: how to rate different local patterns learned? Is there one local pattern that is better than the other? Some sort of neurobiological grounding of how regions organized?
# Potential issues/future directions
1. Adaptive module
	- Intuition
		- Global learned adjacency matrix represents the brain topology — the macro-level connections between different regions/channels
		- However, a path/connection that exists doesn't necessarily mean that the initiating region of that path will influence the end
			- e.g., at recipient region, local inhibitory is causing hyper-depolarization that means neuron firing is impossible
			- Hence, it is observed that the influence of a certain channel on another channel depends upon the local dynamics of both channels 
		- The limitation with current approach is that it only uses the hidden representation of the two channels as input
			- Hidden representation is the direct output of GRU, which only encodes the temporal information of one specific channel, not containing anything about local/neighboring dynamics
		- Moreover, it utilizes the whole $H^{(u)}=\{h_{1},\dots,h_{t}\}$, while if it assumes that the temporal encoding unit is performing well, then $h_{t}$ should already capture all historical information
2. Temporal Readout GRU
	- Sequentially takes in the spatio-context-infused output of SI module $z$
	- $z$ itself is computed using $h$, and $h$ is assumed to contain all historical information of that channel
	- Hence, because $z_{t=9}$ is computed from $h_{t=9}$, and $h_{t=9}$ contains info of $h_{t=0}$ to $h_{t=9}$, $z_{t=9}$ itself essentially contains a lot of information about that channel's history, that's largely the same or overlaps with what the GRU keeps track of in its hidden state
3. SI
	- Multi-hop to better capture the kind of one channel impacting its neighbor, which influences its neighbor?
	- Under current architecture, spatio-context-infused representation of a channel is essentially not-utilized
		- It's not used to inform the next round of SI
	- An equal tension is that SI's output (spatio-context-infused representation) simply is unnecessary, because what we hope it can cover is exactly what's already captured by the raw input data (the x itself is the product of the cross-channel interaction)
4. Decoupling of temporal and spatial encoding 
# Potential directions
1. Substitute $h_t$ with $z_{t}$ for temporal encoding
	- Perhaps makes GRU harder to learn, as it now has to figure out given a spatially rich representation and input, how to incorporate it
2. Add auxiliary loss (as noticed before, the output of SI, $z_{t}$, should be predictive of $x_{t+1}$, because $x_{t+1}$ is the direct product of the process SI tries to model)
	- Benefits:
		- Added per-step loss signal (before, signal have to flow through temporal readout GRU before reaching adjacency matrix and SI)
		- Parallels predictive coding 

1.Multi-hop
2.Temporal readout — changed to add delta gate from z_t
3.Auxiliary and interleave (integrate spatial and temporal data)