---
layout: page
title: Catalyst design with eigenforce models and ML potentials
description: Phenomenological models for strain and coverage effects, and ML potentials for nanoparticles
img: assets/img/projects/catalyst-design/catalyst-design-thumbnail.png
importance: 3
category: Research
period: Ph.D. · Brown University
---

This work was carried out during my Ph.D. at Brown University. It was funded by DOE Basic Energy Science with the Award No. <a href='https://pamspublic.science.energy.gov/WebPAMSExternal/Interface/Common/ViewPublicAbstract.aspx?rv=ea42433a-4522-453d-8d7e-5379ff745b47&rtc=24&PRoleId=10'>SC0019441</a>. Cheng Zeng also acknowledges financial support from Brown's Presidential Fellowship and Open Graduate Education Program. Computational simulations were undertaken using resources at Brown's Center for Computation and Visualization (CCV).

The project has two parts: eigenforce models, a physics-based force--displacement framework for strain effects, surface relaxation and lateral interactions on catalyst surfaces; and machine learning potentials trained with a nearsighted force-training approach, which scale atomistic simulations to nanoparticle catalysts of thousands of atoms. The two come together in a quantitative description of activity and stability across full-scale Co--Pt nanoparticles.

- [Eigenforce models to accelerate catalyst design](#eigenforce-models-to-accelerate-catalyst-design)
- [Machine learning potentials for large nanoparticle catalysts](#machine-learning-potentials-for-large-nanoparticle-catalysts)

### **Eigenforce models to accelerate catalyst design**

Fundamentally, the bonding between the surface and important reaction intermediates dictates the kinetics of elementary reactions on a heterogeneous catalyst.
Strain is known to modify the surface adsorption of key intermediates during catalytic reactions, offering a tunable approach to rational catalyst design.
A [recent work](https://www.nature.com/articles/s41929-018-0054-0) from the Peterson group introduced the eigenstress model to provide an intuitive understanding of strain effects.
Instead of stress and strain, which are hard to define in atomistic systems, I used atomic forces and displacements to quantify the strain effect for improved oxygen reduction activity of Co--Pt and Fe--Pt alloys. This force--displacement model is termed the "eigenforce model". The work is [published](https://pubs.aip.org/aip/jcp/article-abstract/150/4/041704/1023696/Face-centered-tetragonal-FCT-Fe-and-Co-alloys-of?redirectedFrom=fulltext) in J. Chem. Phys. In collaboration with experimental groups, we used the model to describe anisotropic strain effects and to identify high-performance ternary Pt alloys for oxygen reduction; this work has been [published](https://pubs.acs.org/doi/abs/10.1021/jacs.0c08962) in JACS. In a [recent systematic study](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.2c07246) published in J. Phys. Chem. C, I generalized the treatment of strain effects to different adsorption systems using the simple eigenforce model.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/physical-models/simple-eigenforce-model.png" title="Eigenforce model illustration" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0" style="top:0px">
        {% include figure.html path="assets/img/projects/physical-models/results-eigenforce-model.png" title="Eigenforce model results" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    Schematic illustration of the simple eigenforce model (Left). Eigenforce model predictions <i>versus</i> DFT calculated strain effects for a variety of adsorption systems (Right).
</div>

In the simple eigenforce model, adsorbate-induced eigenforces are considered unchanged during atomic displacements. Despite the success of the simple constant-force model, adsorption- and strain-induced surface relaxation should also be taken into account to achieve a more rigorous description of strain effects. Using a simple ball-and-spring analysis, I introduced an atomistic model to quantify changes of eigenforces during displacements and to account for important surface relaxation terms, including strain-induced relaxation, adsorbate-induced relaxation and relaxation due to strain--adsorption coupling.
I also discussed why the simple constant-force model gives good predictions of strain effects in many systems.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/physical-models/relaxation-coupling.png" title="Coupling of strain and adsorption" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/physical-models/decoupling-spring-constants.png" title="Decoupling spring constants" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Schematic of coupling of strain and adsorption (Left). Strategy to decouple spring constants in atomistic simulations (Right).
</div>

The strain effect and surface relaxation for a single adsorbate are well described by the above two models. In realistic catalytic conditions, however, multiple adsorbates may coexist on the catalyst surface, and a complete picture is out of reach without including lateral interactions between adsorbates.
Inspired by the [pioneering work](https://www.sciencedirect.com/science/article/pii/0039602877904691) of Lau and Kohn, I further extended the eigenforce framework to quantify elastic interactions between adsorbates on a metallic surface, where the elastic component is ascribed to cooperative or frustrated surface relaxation arising from the coexistence of multiple adsorbates. I also found that the non-elastic counterpart can be well approximated by a simple two-body fit using a small number of DFT calculations. Combining the elastic and non-elastic terms gives a full picture of lateral interactions at arbitrary adsorbate coverage.

<div class="row justify-content-sm-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/elastic-li.png" title="Elastic lateral interaction" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/DFT_vs_model.png" title="DFT versus Model" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    One-dimensional schematic of elastic interactions between adsorbates on a surface (Left). DFT-calculated lateral interactions <i>versus</i> model predictions (Right).
</div>

The force--displacement framework enables fast evaluation of strain effects, surface relaxation and lateral interactions with only a small number of DFT calculations. Together, these methods provide a scalable route to the rapid design of electrocatalysts.

### **Machine learning potentials for large nanoparticle catalysts**

State-of-the-art nanoparticle catalysts can contain more than 5000 atoms, while the standard DFT calculations used to provide physical insight into material properties are limited to around 500 atoms. To bridge this size gap, I first developed a nearsighted force-training (NFT) approach to systematically generate small-size informative training structures for target nanoparticles in a learning-on-the-fly manner.
The key to a successful NFT is to align the nearsightedness of finite-ranged machine learning potentials with that of *ab initio* calculators. The original NFT work discusses the theoretical principles, methods and algorithms, and the approach was benchmarked on pure Pt nanoparticles using atom-centered neural networks as the machine learning models.

<div class="row justify-content-sm-center">
    <div class="col-sm-8 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/nft.png" title="NFT workflow" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
    <div class="col-sm-4 mt-4 mt-md-0" style="top:30px">
        {% include figure.html path="assets/img/projects/exascale-computing/nft_learning.png" title="NFT learning on Pt260" zoomable="true" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Workflow of structure initialization and nearsighted force training (Left). NFT learning on a Pt260 nanoparticle: evolution of uncertainty and number of chunks with iteration steps (Right).
</div>

I further extended the NFT approach to spin-polarized Co--Pt nanoparticles. The trained neural network potentials (NNPs) are robust and transferable in describing the energetics of Co--Pt nanoparticles of different sizes, compositions, shapes and atomic arrangements. I addressed some key phase stability questions surrounding Co--Pt nanoparticles, including morphology crossover, the order--disorder phase transition and the putative most stable atomic arrangement. Moreover, highly scalable SPARC DFT calculations performed by our collaborators at Georgia Tech validated the stable structures predicted by machine learning, and represent the largest Co--Pt nanoparticles ever calculated with a full *ab initio* code.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/exascale-computing/packing_pattern.png" title="Shell-by-shell packing pattern" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    Shell-by-shell atomic arrangement of a Co<sub>3102</sub>Pt<sub>3164</sub> truncated octahedron optimized by Monte Carlo simulations.
</div>

To complete the understanding of the catalytic performance of nanoparticle catalysts, I combined the NFT approach with a newly developed eigenforce model, a geometric descriptor based on the generalized coordination number, and microkinetic analysis based on DFT-calculated free energy diagrams for the oxygen reduction reaction. This combined strategy allows for a quantitative description of relative catalytic activities across Co--Pt nanoparticles with different sizes and Co compositions. In addition, the stability of Co--Pt nanoparticles can be described using electrochemical dissolution potentials.

<div class="row justify-content-sm-center">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/exascale-computing/atom-level-activity.png" title="Atom level limiting potential" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/exascale-computing/atom-level-stability.png" title="Atom level dissolution potential" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Atom-level limiting potential for the oxygen reduction reaction (Left) and atom-level dissolution potential (Right) of a full-scale 9 nm Co--Pt truncated octahedron of ~17000 atoms.
</div>

The NFT paper is [available](https://pubs.aip.org/aip/jcp/article/156/6/064104/2840702/A-nearsighted-force-training-approach-to) in J. Chem. Phys.
The Co--Pt phase stability work is available in J. Phys. Chem. C [here](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.3c04639). The work on Co--Pt nanoparticle catalytic activity and stability is in preparation.
