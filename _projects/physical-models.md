---
layout: page
title: Atomistic models to expedite catalyst design
description: Simple force-displacement models to quantify strain effect, surface relaxation and adsorbate-adsorbate interaction
img: assets/img/projects/physical-models/model-thumbnail.png
importance: 2
category: Past research
---

Cheng Zeng's research work at Brown was supported by Brown's Presidential fellowship and Open graduate education program.
Computaitonal modeling uses resources at Brown's Center for Computation and Visualization (CCV).

Strain is known to modify the surface adsorption of key intermediates during catalytic reactions, offering a tunable approach to rational catalyst design.
A [recent work](https://www.nature.com/articles/s41929-018-0054-0) from the Peterson group introduced the eigenstress model to provide intuitive  understanding of strain effect.
Instead of stress/strain which is hard to define in atomistic systems, I used atomic forces and displacement to quantify the strain effect for improved oxygen reduction activity of Co--Pt and Fe--Pt alloys. The force--displacement model is termed "Eigenforce model". The work is [published](https://pubs.aip.org/aip/jcp/article-abstract/150/4/041704/1023696/Face-centered-tetragonal-FCT-Fe-and-Co-alloys-of?redirectedFrom=fulltext) at J. Chem. Phys. In collaboration with experiment groups, we used the model to describe anisotropic strain effects and to identify high-performance ternary Pt alloys for oxygen reduction, the work of which has been [published](https://pubs.acs.org/doi/abs/10.1021/jacs.0c08962) on JACS. In a [recent systematic study](https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.2c07246) published on J. Phys. Chem. C, I generalized the strain effect for different adsorption systems using the simple eigenforce model.

<div class="row justify-content-sm-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/simple-eigenforce-model.png" title="Eigenforce model illustration" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0" style="top:15px">
        {% include figure.html path="assets/img/projects/physical-models/results-eigenforce-model.png" title="Eigenforce model results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Schematic illustration of the simple eigenforce model (Left). Eigenforce model predictions <i>versus</i> DFT calculated strain effects for a variety of adsorption systems (Right).
</div>

In the simple eigenforce model, adsorbate-induced eigenforces are considered unchanged during atomic displacements. Despite the success of simple constant-force model, adsorption and strain induced surface relaxation should be taken into account as well to achieve a more rigorous description of strain effect. Using simple ball and spring analysis, I introduced an atomistic model to quantify changes of eigenforces during displacements and to account for important surface relaxation terms, including strain induced relaxation, adsorbate induced relaxation and relaxation due to strain and adsorption coupling.
I also discuss why simple constant-force model gives good predictions for strain effect in many systems.

<div class="row justify-content-sm-center">
    <div class="col-sm-7 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/physical-models/relaxation-coupling.png" title="Coupling of strain and adsorption" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm-5 mt-3 mt-md-0" style="top:10px">
        {% include figure.html path="assets/img/projects/physical-models/results-relaxation.png" title="Surface relaxation model results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Schematic of coupling of strain and adsorption (Left). Model predicted net surface relaxation due to strain <i>versus</i> DFT calculated values (Right).
</div>
