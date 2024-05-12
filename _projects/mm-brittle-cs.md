---
layout: page
title: "Machine learning enabled multiscale simulations for brittle particle cold spray"
description: Atomistic-mesoscale simulation framework to understand size and shape effects of particle feedstock for brittle particle cold spray
img: assets/img/projects/mm-ml-brittle-cs/mm-ml-thumbnail.png
importance: 3
category: Current research
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>
Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute and the Experitial AI institute of Northeastern University. Computational simulations were conducted in part using the Discovery cluster, supported by the Research Computing team at Northeastern University.

This project focuses on an emerging additive manufacturing technique developed by a small company named TTEC LLC. This technique is termed **Brittle particle cold spray (BPCS)**. Conventional cold spray is mainly designed for ductile materials, this new technique demonstrated with brittle materials to make dense coatings on a variety of substrates at a high deposition rate and with unlimited thickness. This new technique is in essence a custom-designed low-pressure cold spray system, which uses the expansion of the gas in the diverging section of the nozzle to create a supersonic gas stream. A few experiments found that BPCS favors small-size (0.1--10 $$\mu$$m) and irregular-shapes partile feedstocks for a successful build versus large sizes (5-50 $$\mu$$m) and spherical shapes used for ductile material cold spray.


<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/csam-sketch.png" title="BPCS sketch" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Schematics of a BPCS system developed by TTEC LLC.
</div>

The bonding mechanics of brittle particle cold spray ranges from atomistic-scale chemical bondings at particle/substrate interface to mesoscale mechanical interlocking for particle/particle packing, necessitating a multiscale approach to understand its fundamentals. I proposed an atomistic-mesoscale simulation approach to address the effects of unique size ranges and shapes suitable for BPCS.


### 1. Single particle size dependent plasticity


The hypothesis is that small-scale particles are more likely to bond at a substrate. It implies a size-dependent plasticity. Plasticity for ceramics can originate from phase transformation, dislocation dynamics and defet structures. Here using a dislocation based mechanism, we will first develop a machine learning potential (MLP) to upsclae atomistic simulations for mechanics of ceramic particles. The MLP will be developed in an active learning manner, incorporating a *MaxVol* uncertainty metric and a *Nearsighted force training* approach to generate small-size data inforamtive to improve the MLP as well as affordable by *ab initio* calculations. Next, we will use the MLP to conduct MD nanocompression tests to generate the stress-strain relationship for the ceramic particles. However, even with MLP, the largest size is limited to the order of 10 nm. To approach the experimental size range around 0.1-10 $$\mu$$m, dislocation characteristics will be utilized to upscale the simulations using discrete dislocation dynamcis (DDD).

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/size-dependent-plasticity.png" title="Single particle plasticity" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Singe particle size dependent plasticity: (a) Active learning loop to develop a MLP, (b) MD nanocompression tests  and (c) Upscaled DDD simulations.
</div>


### 2. Shape dependent mechanical interlocking

The hypothesis is that irredugar particle shapes are more prone to building up the layers during particle deposition, suggesting a mechanical interlocking mechanism. I propose to use discrete element modeling to study the interlocking phenomenon for particles in various shapes. The contact behavior can be complex and varied for irregular shapes, hence a machine learning enabled contact detection algorithm will be employed to relate features of object and cue particles to the contact geometry. The contract geometry will be next fed into a linear parallel bond model to compute the contact forces. Using Netwon's Second law of motion, particle velocity, position and moment can be updated via a Verlet time integration algorithm. A simple mass-volume analysis indicates that for one batch of BPCS load around 5g and if we assume a uniform size of 2.5 $$\mu$$m spherical alumina feedstock particles with a density of 3.98 g/cm$$^3$$, we will have numbers of particles on the order of 10 billion, which is not feasible computationally. So a reduced number of particle 10000 is used to study the layer buildup. We initialize the system by defining the simulation space and boundary conditions. The next step introduces a small number of particles, followed a contact detection and time integration. Once particles are settled on the substrate, more particles are added until all particles are allowed to deposit, after which the packing density and mechanical behavior can be analyzed separately.


<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/shape-dependent-packing.png" title="Shape dependent mechanical interlocking" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Shape dependent mechanical interlocking: (a) Machine learning enabled contact detection and recognition, (b) Linear parallel bond model  and (c) The workflow of DEM for brittle particle cold spray deposition.
</div>


### 3. Size and shape dependent impact behavior

At above, we introduce machine learning enabled simulation methods to understand single particle plasticity and layer buildup during BPCS deposition. The last piece to complete our understanding of BPCS mechanics is the bonding at the particle/substrate interface. We will use a hybrid QM/MM approach to set up a MD simulation single particle impact behavior. The interface will be treated at quantum mechanic level using a machine learning potential whereas the regions in particle and susbtrate away from the interface will be evaluated based on molecular mechanics using a COMB potential. To prevent shockwave bouncing off and propagating through the surface, a three-layer substrate structure will be used.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/ml-mm-single-particle-impact.png" title="Single particle impact: ML/MM hybrid approach and the MD system setup." class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Single particle impact: ML/MM hybrid approach and the MD system setup.
</div>

For multiple particle impact, we will coarse grain the COMB potential for the particle and substrate region to enable mesoscale simulations. Then the coarse-grained and atomistic regions will be coupled via a finite-temperature Quasi-Continuum approach. The coarse graining is achieved by setting multiple conventional crystal cells as one bead and coarse-grained parameters will be found and optimized via an evolutionary algorithm by performing simulations targeting multiple mechanical properties. The impact behavior will be used to focus on the microstructure evolution, dislocation dynamics and phase changes due to a thermo-mechanical coupling effect.

<div class="row justify-content-sm-center">
    <div class="col-sm-11 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/atomistic-mesoscale-approach.png" title="Multiple particle impact: Atomistic-mesoscale coupled framework." class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Multiple particle impact behavior: (a) Atomistic-mesoscale coupled scheme, (b) n-level coarse graining, and (c) The workflow for ML enabled coarse-grained force fields.
</div>