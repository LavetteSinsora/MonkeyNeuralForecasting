# Mechanism behind neuron firing 
## High-level image
1. Dendrites receive signals from previous neurons
2. These signals modify (i.e., increases/decreases) the electric potential in the soma (center cell body)
3. When the local electric potential at the axon hillock (the connection point between soma and axon) exceeds a certain threshold (negative 55mV), a series of self-reinforcing operations will happen that kicks the electric potential to positive 40mV
4. The positive electric potential will propagate down the axon (soma and axon is a continuous fluid space of cytoplasm)
5. Along the axon, there exist *nodes of Ranvier* that will regenerate the signal (positive electric potential will decay as traveling further, and these nodes can regenerate those positive signals/restore them to initial high level)
## Neurobiological implementation — local electric potential & ion channels
### Preliminary — constant potential difference & concentration gradient 
**Core idea**:
A neuron constantly maintains 3 things:
1. Potential difference between neuron inside and outside
	- ~ negative 70mV in the inside relative to the outside
2. High concentration of Potassium ion inside, low concentration outside
3. High concentration of Sodium ion outside, low concentration inside
This is useful because when ion channels open, ions will flow in/out, drived by these differences.

The potential difference is kept by:
- Neuron's membrane is lined with *Sodium-Potassium APTase pumps*
	- These are proteins that moves 3 sodium ions (Na⁺) from the inside of the cell to the outside, and moves 2 potassium ions (K⁺) from outside to inside
		- Physically, these proteins do so by expending APT to transform their shape
- The pumps maintain this constant concentration gradient of potassium
	- With random motion, due to there's higher concentration of potassium on the inside then outside, statistically speaking, there will be a net outflow of potassium ion
- Yet, as potassium ion flows out, they carry away positive charge, making inside more negatively charged, which attracts more potassium ion (positively charged) inflow 
- An equilibrium is reached, where the net flow is zero when the inside is roughly 70 potential less then outside
	- Yet, this equilibrium requires a constant concentration gradient of potassium, which is maintained by the pumps that constantly takes in potassium from outside
### Ion channels — modulating the potential difference
Due to the aforementioned potential/concentration gradient
- Ions like sodium constantly want to flow in (lower concentration inside + negative charge attracting positive sodium ion)
- Cell membrane is impermeable, stopping their inflow
- However, ion channels exist that can be opened to allow their inflow
**Dendrite and sodium channel**:
When a dendrite receives a signal (from its presynaptic neuron), what essentially is happening is that the sodium channel at that dendrite is activated and opened
- Sodium flows in due to concentration gradient and electric attraction
- The region around that dendrite becomes slightly more positive than the rest of neuron (e.g., negative 65mV)
- This potential difference generates a force on molecules toward the rest of neuron, propagating the increased electric potential to the rest and increasing the electric potential
Each dendrite that receives an excitatory signal will contribute positively to the electric potential of the soma/neuron.
**Threshold of voltage-gated sodium channel**
At axon hillock (a point between soma and axon)
- A dense region packed with sodium channels that will open once voltage exceeds a certain threshold (negative 55mV)
- Sodium will flux in, propelling the potential up to positive 40 mV
- When ~40mV is reached, potassium channel will open, opening a way for potassium ions to move out (concentration gradient + repelled by positively charged inside)
- This will make the inside more negative as positive potassium ions move out
This is what is described as a neuron's spike — potential jumping from the usual -70 to +40 due to sudden influx of sodium, and then drop down immediately as sodium channel closes and potassium channel opens.
**Propagating spike along axon via node of Ranvier**
Along the axon
- There periodically are regions with dense voltage-gated sodium channels, similar to configuration at axon hillock
- Positive spike at axon hillock will propagate down the axon, increasing the electric potential
- Once it reaches a threshold, the same thing (that happened before at axon hillock) will happen — sodium channel open, electric potential positive spikes, etc.
- Each node serves as a relay that ensures that the signal don't decay over time as it travels
# Whole process documented
axon hillock spikes → spike propagates down through axon, constantly regenerating → at axon terminal, spike triggers calcium channel open → calcium influx → neurotransmitter release → neurotransmitter diffuses through synaptic cleft and binds to receptor of postsynaptic neuron's dendrite → receptor is basically sodium channel that opens when bind → sodium flows in, this smooth/gentle flow (in contrast to spike, which is a one-time temporary event) produces the postsynaptic current → leads to postsynaptic potential (increase in potential), which passively spread to soma → when enough potential from enough dendrite is spread so that potential is above threshold, a new spike fires/occurs
# How the brain learns
At different time-scales:
- days:
	- The dendrite spine (tip of dendrite that with receptors and receives signal from presynaptic axon terminal) physically grows (i.e., more receptors are grown out) when there are consistent stimulation from presynaptic neuron
- days/weeks/months
	- Axons where presynaptic firing consistently don't trigger postsynaptic neuron firing will retreat/be pruned back, while synapses connecting neurons consistently firing together will see presynaptic extending its axon arbor
# Misconception about dendrite-soma-axon model
Previously, textbooks illustrate neurons as having thousands of dendrites, one soma, and one axon. 
- In reality, the axon can have axon branches/collateral, meaning that it won't only form synaptic connection with a single postsynaptic dendrite
- Moreover, these branches mainly connect with physically neighboring dendrites, but some can go far to other areas of the brain