# Torch Brain

## Inspiration
Deep Learning has revolutionized the way we conceive computation and the way we interact with devices on a daily basis. Although very successful, they have succumbed to many limitations; namely:
1. Lack of robustness (adversarial examples)
2. Highly power Inefficient
3. Require a lot of man-hours for building "functional" networks
4. Require large datasets to make intelligent inferences

To overcome these challenges, we gain inspiration from the brain, as it's robust to adversarial perturbations, highly power efficient and require very less labeled data to make useful inferences. One of the key advantages that brains have over existing artificial neural networks are that they're built completely out of spiking neural networks. 

PyTorch has enabled the deep-learning community to build modular intelligent non-spiking networks that can be trained to perform tasks, as well as allow researchers to rapidly prototype their innovative ideas on bettering the state of the art in DL. Similarly, we want to extend PyTorch functionality to enable the Deep learning community as well as computational neuroscientists to build spiking neural network models, in an attempt towards building networks that are highly power efficient (for DL community) or to build sophisticated models of the brain (neuroscience research community). 

## What it does

We are currently building an open-source library for extending PyTorch to implement spiking neural networks.

Our major goal is to provide basic building blocks that would allow the modular, flexible and scalable construction of 
spiking neural networks. Apart from the construction of these networks, we are also writing optimization libraries that would allow users to train these networks for a wide variety of tasks. 

In this version of the spiking neural net library, we provide the following options:
1. The number of layers/reservoirs in the network 
*  An easily configurable geometry of each layer 
    * 2.1\. Square layers
    * 2.2\. Circular layers
    * 2.3\. Torus layers
3. The number of spiking nodes within each layer
* The dynamics that govern the evolution of neural activity
    * 4.1\. Izhikevich neuron model
    * 4.2\. Linear Integrate and fire model

To exhibit the strength and utility of this library, we apply it to grow neural networks that can self-organize useful pooling layers, serve as feature extractors. In this specific implementation, we show that like the visual system in humans have functioning circuits much before eye-opening (do not require data to form networks), we can similarly show that spiking neural networks can self-organize into useful initial architectures without the requirement of any data! We, finally, seamlessly seamlessly stack multiple spiking layers and demonstrate reasonable accuracy on a simple task (MNIST classification).

## How we built it!

Our architecture enables us to build a modular spiking neural networks. 

Each network can be built with a variable number of spiking layers, and to do so, we have instantiated the following set of classes:
1. Every spiking node is imported from a `Node' class. 
2. Every layer of neurons is imported from a `SpikingLayer' class
3. The dynamics of every neuron in the network is imported from a 'dynamics_NN' class
4. The learning rule that the spiking neural net uses is imported from 'learn_spikeRules' class.  


## Demonstrations

Spiking neural network in the first layer spontaneously forming spatiotemporal waves that tile the first layer to self-organize pooling layers.

<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/enMumwvLAug" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

## Challenges I ran into
The majority of us in the group are not computational neuroscientists - although maybe now we are :)
With only one person knowing the depths of the field, we faced several bottlenecks in development that required the group getting up to speed. 

Separately, we faced challenges in designing our implementation. We wanted an architecture that was flexible, yet one that maximally utilized the Pytorch's advanced functionality. 

## Accomplishments that we're proud of

The field of computational neuroscience and the machine-learning community shifting to spiking-neural networks implementation of artificial networks face a major difficulty when it comes to large scale implementation of spiking nets. Although spiking nets have been around since the early 1960's, the simulators present for the same aren't malleable to being built and being trained similar to artificial neural networks. 

Our preliminary success in building a framework for implementing spiking neural nets on PyTorch would allow the community to start building SNN's, and would allow for reproducible research. We believe torchbrain is a promising step in that direction. 


%We're proud of the direction we've set. All too often researches must take tedious efforts to reimplement the code of old and well-established algorithms. For instance, spiking neural networks have been around since the 1950's and our particular SNN model was developed in 1984. While there exist functional libraries in Fortran, we believe present-day researchers should have easy access to modern technologies. More importantly, we believe an open-source ecosystem should exist to serve the computational neuroscience community and develop as it does. 
We believe torchbrain is a promising step in that direction. 

## What I learned
As we've said, we're (mostly) not computational neuroscientists. We've had to learn the fundamentals of spiking neural networks and how they train. By extension, we learned a different paradigm to training neural networks and more about how the brain extracts information. 

## What's next for torchbrain
Ideally, the computational neuroscience community and community of machine-learning engineers who are moving towards building brain-inspired architecture should contribute to this endeavor of implementing libraries and frameworks on PyTorch for spiking neural networks. We believe there's a lot of potential for nailing the core abstractions of the field and providing tools that accelerate research. 


## License:

MIT

# Cite

```
@article{DBLP:journals/corr/abs-1906-01039,
  author    = {Guruprasad Raghavan and
               Matt Thomson},
  title     = {Neural networks grown and self-organized by noise},
  journal   = {CoRR},
  volume    = {abs/1906.01039},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.01039},
  archivePrefix = {arXiv},
  eprint    = {1906.01039},
  timestamp = {Thu, 13 Jun 2019 13:36:00 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1906-01039},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Issues/Feedback:

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **shoaib_ahmed.siddiqui@dfki.de**
