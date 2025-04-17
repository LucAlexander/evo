# EVOMORPH

## Preface

This repository is a rewrite and continuation of my undergrad capstone project. It aims to generalize neural networks into a higher order residual network system, wherein more complicated structures can be formed naturally through a neuromorphic approach to neural architecture search. This projects proof of concept predecessor was only completed to the minimum requirement during the semester I had to work on it, and included a model description language which is not present in this new system, as it was ultimately unnecessary, and was only used to create initial architecture search seeds in leu of a proper model serialization system.

This new version is more simple in its construction, and generalizes the neural network system even further, making for a cleaner implementation. It is heavily work in progress.

## Generalization

The graph of neurons is first reduced to a graph of layers. For trivial networks this reduced graph is also trivial. For recurrent networks the reduction becomes a simplified directed cyclic graph.

For reference, the equation for the activated output of a layer in a non recurrent network remains untouched. For any given layer, we construct an output $z$ from the bias of its neurons and the weights connected the current and previous layer. This output is then activated using a function of choice $\sigma$ to receive the final output of the layer $a$.

$$z_{i} = b_{i} + \sum_{k} w_{ik}x_{k} $$
$$a_{i} = \sigma(z_{i})$$

The output layer performs an additional loss calculation on the final activated vector using a vector of expected values. A back propagation pass (using a gradient optimizer) then calculates weight, bias, and activation gradients using standard formulas. These gradients are accumulated and averaged over a batch, after which they are applied to their corresponding parameters.

$$\frac{\delta cost}{\delta w} = \frac{\delta z}{\delta w} \frac{\delta a}{\delta z} \frac{\delta cost}{\delta a}$$
$$\frac{\delta cost}{\delta b} = \frac{\delta z}{\delta b} \frac{\delta a}{\delta z} \frac{\delta cost}{\delta a}$$
$$\frac{\delta cost}{\delta a_{i-1}} = \frac{\delta z}{\delta a_{i-1}} \frac{\delta a}{\delta z} \frac{\delta cost}{\delta a}$$

The final gradient is propagated back recursively in place of the cost. For recurrent networks where each layer may have multiple incoming layers, we aggregate them into the layer as if they were all part of one cohesive input layer. 

$$z_{i} = b_{i} + \sum_{p}\sum_{k} w_{ikp}x_{kp} $$

The backward pass receives a reciprocal modification, wherein the gradient is propagated back through time for each recurrent connection, passing over the main body of the model once more per recurrent connection.

## Higher order weights

Each connection in a reduced graph is given its own trainable weight parameter $p$. The layer calculation needs to take this into account.

$$z^{l}_{i} = b^{l}_{i} + \sum_{k} w_{ik}a^{l-1}_{k}p^{l} $$
$$a^{l}_{i} = \sigma(z^{l}_{i})$$

Since these weights effect the final loss, they also effect the equations for the gradients of all other learnable parameters. We also include a calculation for the gradient of the layer weight parameter so that it may also be trained. 

$$\frac{\delta cost}{\delta p} = \frac{\delta z}{\delta p} \frac{\delta a}{\delta z} \frac{\delta cost}{\delta a}$$
