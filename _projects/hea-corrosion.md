---
layout: page
title: Discovery of high-entropy alloys for corrosion protection
description: A machine learning framework to evaluate corrosion performance for given compositions of high-entropy alloys
img: assets/img/projects/hea-corrosion/corrosion-three-metrics.png
importance: 1
category: Current research
---

Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute of Northeastern University. Computational simulations are carried out using the Discovery cluster, supported by the Research Computing team at Northeastern University.

I introduced a machine learning framework combining two pipelines to quantify three corrosion metrics using machine learning models.
Corrosion metrics include single-phase formability, FCC111 surface energy and Pilling-Bedworth ratios.
Singe-phase formation indicates whether a homogeneous phase will be formed given the composition of a high-entropy alloy (HEA).
A homogeneous phase suggests the absence of fast corrosion channels and incliness of forming a more protective passive oxide layer against corrosion.
The single-phase formability is predicted by a random forest classifier trained on experimental data whose inputs and outputs are simply chemical compositions and labels to indicate whether single phase or multiple phase structures are formed.
Surface energy relates the tendency of dissolution of a surface atom to the current density during corrosion.
Piling-Bedworth ratios (PBR) are associated with metal/oxide stresses upond oxidation of a surface alloy. Ideally, we would like to see PBR values in the range of one and two, which can form a protective oxide layer while not causing significant stresses in oxide layers (that can result in cracks of oxides).
The surface energy and Pilling-Bedworth ratios are computed by machine learning potentials trained on systematically generated first-principles data.
The machine learning potentials used in this work belong to the group of [momentum tensor potentials](https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe), which uses moment tensor contraction to define atomic energies. Each moment consists of two components---one is radial basis functions (e.g. Chebyshev polynomials) and the other is tensor of a certain rank to describe angular local environments.
The workflow of this framework is shown in the below figure.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-corrosion/workflow.jpg" title="Worflow of ML HEA corrosion" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Workflow of the machine learning framework for discovery of corrosion-resistant high-entropy alloys.
</div>

This methodology was applied to AlCrFeCoNi high-entropy alloys. Compositions of Al and Cr were varied whereas the remaining Fe, Co and Ni have identical compositions. The results are given below. The identified low Al and around 18% Cr compositions for enhanced corrosion resistance agrees well with experiments.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-corrosion/results.png" title="Data centric framework" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Single phase formability (a), Pilling-Bedworth ratios (b) and FCC111 surface energies (c) of AlCrFeCoNi as a function of Al and Cr compositions. In part (a), single phase (SP) and multiple phase training points are indicated by respective red squares and green circles. Decision boundary from <a href='https://www.sciencedirect.com/science/article/pii/S1359645419307050'>Wu et al</a> is represented by a grey dashed line.
</div>

This work is now published and more details can be found in the paper at Computational Materials Science [here](https://www.sciencedirect.com/science/article/pii/S0927025624001460).