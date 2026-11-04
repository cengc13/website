---
layout: page
title: Generative design of molecular materials
description: Flow matching models for property-guided molecule generation and molecular crystal structure prediction
img: assets/img/projects/generative-molecular-materials/propmolflow_toc.png
importance: 1
category: Current research
---

We acknowledge funding from NSF Grant OAC-2311632 and from the AI and Complex Computational Research Award of the University of Florida. We also acknowledge UFIT Research Computing for computational resources from HiPerGator and consultation.

Designing new molecules and molecular materials with target properties requires searching a vast chemical and structural space. At the University of Florida, I develop **flow matching** generative models that sample 3D molecules and molecular crystals directly, and that can be steered toward desired properties. The work so far has three parts: property-guided molecule generation (PropMolFlow), guidance strategies for conditional generation (MolGuidance), and crystal packing prediction for molecular crystals (MolCrystalFlow).

### **PropMolFlow: property-guided molecule generation**

PropMolFlow uses flow matching parameterized by an SE(3)-equivariant graph neural network to generate 3D molecules conditioned on target molecular properties. It is competitive with state-of-the-art diffusion models across a range of molecular properties while maintaining high structural validity, stability and sampling efficiency.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/publication_preview/propmolflow.png" title="PropMolFlow" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    PropMolFlow: property-guided molecule generation with geometry-complete flow matching.
</div>

The paper is published at [Nature Computational Science](https://www.nature.com/articles/s43588-025-00946-y) (see also the [NYU news release](https://www.nyu.edu/about/news-publications/news/2026/january/scientists-design-molecules--backward--to-speed-up-discovery.html)). The preprint is on [arXiv](https://arxiv.org/abs/2505.21469) and the dataset is on [zenodo](https://zenodo.org/records/17726328).

### **MolGuidance: guidance strategies for conditional generation**

MolGuidance extends PropMolFlow by integrating three advanced property-guidance strategies (classifier-free guidance, autoguidance and model guidance) into an SE(3)-equivariant flow matching framework for conditional molecule generation.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/publication_preview/molguidance.png" title="MolGuidance" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    MolGuidance: advanced guidance strategies for conditional molecular generation with flow matching.
</div>

An early version was accepted at the [ICML 2025 GenBio Workshop](https://openreview.net/forum?id=WhZrYbwrKb), and the full paper is published at [Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/full/10.1021/acs.jcim.6c00964). The dataset is on [zenodo](https://zenodo.org/records/16847162).

### **MolCrystalFlow: molecular crystal structure prediction**

How molecules pack in a crystal determines many properties of molecular materials. MolCrystalFlow generates crystal packings for given rigid-body molecular conformers, combining a hierarchical representation with equivariant Riemannian flow matching. We benchmark it on two open-source datasets against a state-of-the-art generative model for large periodic crystals and against the rule-based Genarris-3 method.

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/publication_preview/molcrystalflow.png" title="MolCrystalFlow" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    MolCrystalFlow: molecular crystal structure prediction via flow matching.
</div>

The preprint is available at [arXiv](https://arxiv.org/abs/2602.16020). Code is on [GitHub](https://github.com/Liu-Group-UF/MolCrystalFlow) and data is on [zenodo](https://zenodo.org/records/18879915). I presented this work at the ACS Spring 2026 meeting in the session "Deep Generative Models in Computational Chemistry".
