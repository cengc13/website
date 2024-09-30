---
layout: page
title: Inverse design of single-phase high-entropy alloys
description: Disentangled representation of compositions/structures and properties in a semi-supervised variational autoencoder (VAE)
img: assets/img/projects/hea-xai-inverse-materials-design/xai-inverse-mater.png
importance: 2
category: Current research
---

Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute of Northeastern University. Computational simulations, if any, are carried out using the Discovery cluster, supported by the Research Computing team at Northeastern University.

Conventional inverse materials design use unsupervised machine learning where the input and output are both chemical compositions, crystal structures or microstructure images without considering the composition/structure-property relationship explicitly. The unsupervised learning project high-dimentional features encoding compositions/structures into a low-dimensional latent space and reconstruct the features in the output. Sampling in the latent space generates a new composition/structure. This unsupervised scheme is theorized to learn the entangled composition/structure-property relationship in an implicit manner, which can only be modeled through a seperate surrogate model. The surrogate model and latent space representation combined enables the sampling of materials with desired properties. However, this entangled and implicit scheme has three drawbacks. First, the entangled representation of composition/structure-property relationship is by nature lacking interpretability. Second, without inclusion of target property, the learned latent space can fail to form reliable composition/structure-property relationship. Lastly, this unsupervised approach is not suitable for inverse design of materials with multiple desired properties as all these info can be highly entangled in the latent space.

We propose to use a custom semi-supervised variational autoencoder to disentangle the relationship between compositions and multiple materials property. This approach includes a recognition (encoder) and a genreative (decoder) model. By using informative priors for the target properties and inputs, we can learn the composition-property relationship in a disentangled and data-efficient way. Besides, it allows us to generate a target material with pre-defined mutliple materials properties. Adding post-hoc analysis using existing techniques such as shapley values (SHAP), it innovates a data-efficient and interpretable paradigm for inverse materials design with multi-objectives. Human-AI collaboration corraborated by an interative intelligent user interface will assist to complement the missing info in the inverse design loop.

This methodology is initially framed to search for high-entropy alloys that tend to form single-phase structures using an experimental dataset.
The dataset takes input and output as the chemical formulas and binary phase formation respectively, as detailed in our [previous work](https://www.sciencedirect.com/science/article/pii/S0927025624001460). Although it is demonstrated for single property, it is trivial to extend the framework for optimizing multiple properties all at once.

The disentangled variational encoder consists of a recognition model (encoder) and a generative model (decoder). It leverages all useful information---both labelled and unlabelled data---to learn a probablistic relationship between the features, latent variables and target properties. An schematic illustration of the semi-supervised learning approach is shown below. The key to this approach is that the latent variables are now conditioned on the binary phase formation and the generated compositions at the decoder side also depends on the probability of single-phase formation.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/ss_vae.png" title="Disentangled VAE" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Disentangled variational autoencoder (VAE) for learning single-phase formation of high-entropy alloys: Generative model (Left) and Recognition model (Right).
</div>

Using a well-trained VAE, we can now map the latent variables for all data points and we color it by the ground truth label (binary phase formation), as shown in the left panel of Figure below. One can see that the high-dimensional composition space is transformed to a compact low-dimensional latent representation where most data points are concentrated in a small region.
More importantly, the target property is explicitly liberated from the latent space as points at the same latent positions can be either multiple-phase or single-phase alloys.
In constrast, the latent space is implicitly associated with other properties. For instance, in the right panel of Figure below, we created three groups of element lists and we combined four elements from each list and made four-element equimolar high-entropy alloys. We found that different types of alloys are actually located in well-separated regions in the latent space.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/latent_summary.png" title="Latent representation" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Latent representation for all data points: Disentangled single-phase formation (Left) and Associated alloy types (Right).
</div>

With this disentangled represenation, we devised an iterative procedure to find single-phase alloys starting from a multi-phase alloy, in hope that alloys with similar element constitutes will be identified in the end. The workflow is shown below and an example is given for benchmarking.
The start alloy is Fe14Ni16Cr22Co14Al22Cu8 and the inverted alloy is Fe21Ni22Cr22Co35. It makes intuitive sense as Cu, Cr and Al are not desired elements for the formation of a single-phase alloy in high-entropy alloys comprising of base element Fe, Ci, Co and Ni.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/iterative_search_workflow.png" title="Inversion of a material" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    An iterative process to design single-phase alloys with similar constitutes using the disentangled VAE.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/inversion_sequence.png" title="Step-by-step inversion" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    The chemical formula, predicted single-phase probability and latent variables at each step of the iterative inversion process.
</div>

One can also carry out post-hoc analysis to extract additional interpretation of the inversion process. Figure below shows the overall feature importance evaluated on the test dataset (left) and the each feature value for the start and inverted alloy (right).
One can see that the trained VAE pushes the initial multi-phase alloy towards an alloy with smaller size difference and mixing entropy as well as a higher molar volume and melting temperature. This is in good agreement with the overall contribution of each feature.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/shapley_interpretability.png" title="shapley analysis" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Post-hoc analysis for improved interpretability: Overall feature importance on the test data (Left) and the individual feature values for both start and inverted alloys (Right).
</div>