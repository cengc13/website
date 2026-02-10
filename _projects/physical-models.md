---
layout: page
title: Eigenforce models to expedite catalyst design
description: Force-displacement models for strain effect, surface relaxation and lateral interaction
img: assets/img/projects/physical-models/model-thumbnail.png
importance: 2
category: Past research
---

Cheng Zeng's research work at Brown was supported by Brown's Presidential fellowship and Open graduate education program.
Computaitonal modeling uses resources at Brown's Center for Computation and Visualization (CCV).

Fundamentally, the bonding between the surface and important reaction intermediates dictates the kinetics of elementary reactions on a heterogeneous catalyst.
Strain is known to modify the surface adsorption of key intermediates during catalytic reactions, offering a tunable approach to rational catalyst design.
A [recent work](https://www.nature.com/articles/s41929-018-0054-0) from the Peterson group introduced the eigenstress model to provide intuitive  understanding of strain effect.
Instead of stress/strain which is hard to define in atomistic systems, I used atomic forces and displacement to quantify the strain effect for improved oxygen reduction activity of Co--Pt and Fe--Pt alloys. The force--displacement model is termed "Eigenforce model". The work is [published](https://pubs.aip.org/aip/jcp/article-abstract/150/4/041704/1023696/Face-centered-tetragonal-FCT-Fe-and-Co-alloys-of?redirectedFrom=fulltext) at J. Chem. Phys. In collaboration with experiment groups, we used the model to describe anisotropic strain effects and to identify high-performance ternary Pt alloys for oxygen reduction, the work of which has been [published](https://pubs.acs.org/doi/abs/10.1021/jacs.0c08962) on JACS. In a [recent systematic study](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.2c07246) published on J. Phys. Chem. C, I generalized the strain effect for different adsorption systems using the simple eigenforce model.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/simple-eigenforce-model.png" title="Eigenforce model illustration" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0" style="top:0px">
        {% include figure.html path="assets/img/projects/physical-models/results-eigenforce-model.png" title="Eigenforce model results" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    Schematic illustration of the simple eigenforce model (Left). Eigenforce model predictions <i>versus</i> DFT calculated strain effects for a variety of adsorption systems (Right).
</div>

In the simple eigenforce model, adsorbate-induced eigenforces are considered unchanged during atomic displacements. Despite the success of simple constant-force model, adsorption and strain induced surface relaxation should be taken into account as well to achieve a more rigorous description of strain effect. Using simple ball and spring analysis, I introduced an atomistic model to quantify changes of eigenforces during displacements and to account for important surface relaxation terms, including strain induced relaxation, adsorbate induced relaxation and relaxation due to strain and adsorption coupling.
I also discuss why simple constant-force model gives good predictions for strain effect in many systems.

<div class="row justify-content-sm-center">
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/relaxation-coupling.png" title="Coupling of strain and adsorption" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/decoupling-spring-constants.png" title="Decoupling spring constants" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Schematic of coupling of strain and adsorption (Left). Strategy to decouple spring constants in atomistic simulations (Right).
</div>

The strain effect and surface relaxation of single adsorption are now well described by the above two models. Nevertheless, in real catalysis scenario, multiple adsorbates may exist on the catalyst surface. A complete picture is out of reach without inclusion of lateral interactions between adsorbates.
Inspired by the [pioneering work](https://www.sciencedirect.com/science/article/pii/0039602877904691) of Lau and Kohn, I further developed the eigenforce framework to quantify the elastic interaction between adsorbates on a metallic surface and the elastic component is ascribed to the cooperative/frustrated surface relaxation due to coexistence of multiple adsorbates. Besides, I found that non-elastic counterpart can be well approximated by a simple two-body fitting using a small number of DFT calculations. Combining both elastic and non-elastic terms, a full picture of lateral interaction for arbitrary adsorbate coverage can be completed.

<div class="row justify-content-sm-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/elastic-li.png" title="Elastic lateral interation" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/DFT_vs_model.png" title="DFT versus Model" class="img-fluid rounded z-depth-1" zoomable="true"%}
    </div>
</div>
<div class="caption">
    One-dimensional schematic of elastic interactions between adsorbates on a surface (Left). DFT calculated lateral interaction versus Model predictions (Right).
</div>

The force--displacement framework enables fast evaluation of strain effect, surface relaxation and lateral interaction with only a small number of DFT calculations. These methods combined provide a scalable solution to rapid design of electrocatalysts.