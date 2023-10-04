---
layout: page
title: "A computational framework for brittle particle cold spray"
description: Past, present and perspective of Multiphysics, Multiscale and Machine learning (M3) modeling
img: assets/img/projects/mm-ml-brittle-cs/mm-ml-thumbnail.png
importance: 2
category: Current research
---

Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute and the Experitial AI institute of Northeastern University. Computational simulations were conducted in part using the Discovery cluster, supported by the Research Computing team at Northeastern University.

This project focuses on an emerging additive manufacturing technique developed by a small Virginia-based company named TTEC. This technique is termed **Brittle particle cold spray (BPCS)**. In contrast with conventional cold spray that is mainly designed for ductile materials, this new technique has been demonstrated to fabricate dense coatings on thermally sensitive substrates with unlimited thickness. The TTEC system and an example deposit are shown in the below figure.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/instrument.jpg" title="Hardware and an example deposit" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    The hardware used in BPCS (R&D scale) can fit into a (b) sand blast cabinet.  (c) depicts a fully dense sprayed semiconductor material (Figure courtesy of TTEC LLC).
</div>

This new technique is in essence a custom-designed, type-one, low-pressure cold spray system. As shown in the schematic illustration below, the system uses the expansion of the gas in the diverging section of the nozzle to create a supersonic gas stream.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/csam-sketch.png" title="BPCS sketch" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Schematics of a BPCS system developed by TTEC LLC.
</div>

While this technique has been used to establish correct particle size distributions of a few materials, the mechanics and the bonding mechanism of BPCS is unclear. It is well-acknowledged that bonding of two ductile materials in a high-pressure setting results from plastic deformation (termed as "adiabatic shear instability"). Current understandings for BPCS mechanics include mechanical interlocking followed after the fragmentation of sub-micron brittle particles and metallurgical bonding. Since BPCS processes are impacted by many factors as illustrated in the process diagram below, which span in multiple length- and time-scales, and involve a variety of complex physical processes, multiscale and multiphysics simulations are required to fully understand BPCS mechanics.

A multiscale approach spans from microscale first-principles calculations of minimum inputs, to mesoscale phase-field modeling and macroscale discrete element modeling, as depicted in the following figure.

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/multiscale-modeling.png" title="Multiscale simulations" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Number of inputs for Multiscale simulations: From quantum to continuum.
</div>

Multiphysics simulations are needed to draw insights into the temperature, phase and microstructure evolution for many analyses.


<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/mm-ml-brittle-cs/multiphysics.png" title="Multiphysics modeing" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Multiphysics simulations for BPCS.
</div>

Moreover, as more reliable sensory techniques have been developed and implemented in the manufacturing process, more data become available, enabling a more well-rounded understanding and analysis of this process via data-driven machine learning modeling.
However, machine learning algorithms for additive manufacturing often suffers from challenges in limited, noisy and high-dimensional data. Also, development of physics-informed complex models to understand intricate physical processes in BPCS may require intensive computation and substantial expertise. Opportunities emerge along with those challenges for using machine learning to enhance fundamental research and application of BPCS. For instance, machine learning can be used for fast predictive modeling, which allows for process optimization and real-time control. Besides, machine learning techniques have a potential in integrating multiscale simulations seamlessly, expanding the knowledge body in this area. In addition, interpretable machine learning models can help identify the root cause of build failures, suggesting corrective actions for adaptive control.