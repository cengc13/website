---
layout: page
title: Enabling exascale computing of chemical systems
description: Machine learning potentials for large-size nanoparticle catalysts
img: assets/img/projects/exascale-computing/exascale-thumbnail.png
importance: 1
category: Past research
---

This project is funded by DOE Basic Energy Science with the Award No.  <a href='https://pamspublic.science.energy.gov/WebPAMSExternal/Interface/Common/ViewPublicAbstract.aspx?rv=ea42433a-4522-453d-8d7e-5379ff745b47&rtc=24&PRoleId=10'>SC0019441</a>. Cheng Zeng is also indebted to the financial support from Brown's presidential fellowship and open graduate education program. Computational simulations were undertaken using resources at Brown's Center for Computation and Visualization (CCV).

State-of-the-art nanoparticle catalysts can come up in sizes of more than 5000 atoms while modern standard DFT calculations used to provide physical insights into material properties are limited to around 500 atoms. To bridge this size gap, I first developed a nearsighted force-training (NFT) approach to systematically generate small-size informative training structures for target nanoparticles in a learning-on-the-fly manner.
The key of a successful NFT is to align the nearsightedness of finite-ranged machine learning potentials with that of *ab initio* calculators. The original NFT work discusses the theoretical principles, methods and algorithms, and the approach was benchmarked on pure Pt nanoparticles using the atom-centered neural networks as machine learning models.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/nft.png" title="NFT workflow" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
    <div class="col-sm-4 mt-4 mt-md-0" style="top:30px">
        {% include figure.html path="assets/img/projects/exascale-computing/nft_learning.png" title="NFT learning on Pt260" zoomable="true" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Workflow of structure initialization and nearsighted force training (Left). NFT learning on a Pt260 nanoparticle: Uncertainty and # chunks propagation with iteration steps (Right).
</div>

I further extended the NFT approach to spin-polarized Co--Pt nanoparticles. The trained neural network potentials (NNPs) are robust and transferable in describing energetics of Co--Pt nanoparticles in different sizes, compositions, shapes and atomic arrangements. I addressed some key phase stability questions surrounding Co--Pt nanoparticles, including morphology crossover, order--disorder phase transition and the putative most stable atomic arrangement. Moreover, highly scalable SPARC DFT calculations performed by our collaborators at Georgia Tech validated machine learning predicted stable structures and represented the largest size of Co--Pt nanoparticles ever calculated by a full *ab initio* code.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/packing_pattern.png" title="Shell by shell packing pattern" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    Shell by shell atomic arrangement of a Co3102Pt3164 truncated octahedron optimized by Monte-Carlo simulations.
</div>

To complete the understanding of catalytic performance of nanoparticle catalysts, I combined the NFT approach with a newly-developed eigenforce model, a geometric descriptor using generalized coordination number and microkinetic analysis based on DFT calculated free energy diagram for oxygen reduction reaction. This combined strategy allows for a quantitative description of relative catalytic activities across Co--Pt nanoparticles with different sizes and Co compositions. Besides, the stability of Co--Pt nanoparticles can be described using electrochemical dissolution potentials.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/atom-level-activity.png" title="Atom level limiting potential" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/atom-level-stability.png" title="Atom level dissolution potential" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Atom-level limiting potential for oxygen reduction reaction (Left) and Atom-level dissolution potential (Right) of a full-scale 9 nm Co--Pt truncated octahedron of ~17000 atoms.
</div>

The NFT paper is [available](https://pubs.aip.org/aip/jcp/article/156/6/064104/2840702/A-nearsighted-force-training-approach-to) at J. Chem. Phys.
The Co--Pt phase stability work is available at Journal of Physical Chemistry C [here](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.3c04639). The work regarding Co--Pt catalytic activity and stability is to be submitted. The three works together form a trilogy of nearsighted force training targeting nanoparticle alloy catalysts.