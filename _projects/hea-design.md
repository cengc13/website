---
layout: page
title: Machine learning high-entropy alloys
description: Predictive models and inverse design for corrosion-resistant high-entropy alloys
img: assets/img/projects/ml-hea-design/ml-hea-thumbnail.png
importance: 1
category: Past research
---

Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute of Northeastern University. Computational simulations are carried out using the Discovery cluster, supported by the Research Computing team at Northeastern University.

This project covers two related lines of work on high-entropy alloys (HEAs): a machine learning framework for discovering corrosion-resistant HEAs, and a semi-supervised variational autoencoder for inverse design of single-phase HEAs.

- [Discovery of high-entropy alloys for corrosion protection](#discovery-of-high-entropy-alloys-for-corrosion-protection)
- [Inverse design of single-phase high-entropy alloys](#inverse-design-of-single-phase-high-entropy-alloys)

### **Discovery of high-entropy alloys for corrosion protection**

I introduced a machine learning framework combining two pipelines to quantify three corrosion metrics using machine learning models.
Corrosion metrics include single-phase formability, FCC111 surface energy and Pilling-Bedworth ratios.
Single-phase formation indicates whether a homogeneous phase will be formed given the composition of a high-entropy alloy (HEA).
A homogeneous phase suggests the absence of fast corrosion channels and inclination toward forming a more protective passive oxide layer against corrosion.
The single-phase formability is predicted by a random forest classifier trained on experimental data whose inputs and outputs are simply chemical compositions and labels to indicate whether single phase or multiple phase structures are formed.
Surface energy relates the tendency of dissolution of a surface atom to the current density during corrosion.
Pilling-Bedworth ratios (PBR) are associated with metal/oxide stresses upon oxidation of a surface alloy. Ideally, we would like to see PBR values in the range of one and two, which can form a protective oxide layer while not causing significant stresses in oxide layers (that can result in cracks of oxides).
The surface energy and Pilling-Bedworth ratios are computed by machine learning potentials trained on systematically generated first-principles data.
The machine learning potentials used in this work belong to the group of [moment tensor potentials](https://iopscience.iop.org/article/10.1088/2632-2153/abc9fe), which uses moment tensor contraction to define atomic energies. Each moment consists of two components---one is radial basis functions (e.g. Chebyshev polynomials) and the other is tensor of a certain rank to describe angular local environments.
The workflow of this framework is shown in the below figure.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-corrosion/workflow.jpg" title="Workflow of ML HEA corrosion" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Workflow of the machine learning framework for discovery of corrosion-resistant high-entropy alloys.
</div>

This methodology was applied to AlCrFeCoNi high-entropy alloys. Compositions of Al and Cr were varied whereas the remaining Fe, Co and Ni have identical compositions. The results are given below. The identified low Al and around 18% Cr compositions for enhanced corrosion resistance agrees well with experiments.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-corrosion/results.png" title="Data centric framework" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Single phase formability (a), Pilling-Bedworth ratios (b) and FCC111 surface energies (c) of AlCrFeCoNi as a function of Al and Cr compositions. In part (a), single phase (SP) and multiple phase training points are indicated by respective red squares and green circles. Decision boundary from <a href='https://www.sciencedirect.com/science/article/pii/S1359645419307050'>Wu et al</a> is represented by a grey dashed line.
</div>

This work is now published and more details can be found in the paper at Computational Materials Science [here](https://www.sciencedirect.com/science/article/pii/S0927025624001460).

### **Inverse design of single-phase high-entropy alloys**

Conventional inverse materials design use unsupervised machine learning where the input and output are both chemical compositions, crystal structures or microstructure images without considering the composition/structure-property relationship explicitly. The unsupervised learning project high-dimensional features encoding compositions/structures into a low-dimensional latent space and reconstruct the features in the output. Sampling in the latent space generates a new composition/structure. This unsupervised scheme is theorized to learn the entangled composition/structure-property relationship in an implicit manner, which can only be modeled through a separate surrogate model. The surrogate model and latent space representation combined enables the sampling of materials with desired properties. However, this entangled and implicit scheme has three drawbacks. First, the entangled representation of composition/structure-property relationship is by nature lacking interpretability. Second, without inclusion of target property, the learned latent space can fail to form reliable composition/structure-property relationship. Lastly, this unsupervised approach is not suitable for inverse design of materials with multiple desired properties as all these info can be highly entangled in the latent space.

We propose to use a custom semi-supervised variational autoencoder to disentangle the relationship between compositions and multiple materials property. This approach includes a recognition (encoder) and a generative (decoder) model. By using informative priors for the target properties and inputs, we can learn the composition-property relationship in a disentangled and data-efficient way. Besides, it allows us to generate a target material with pre-defined multiple materials properties. Adding post-hoc analysis using existing techniques such as shapley values (SHAP), it innovates a data-efficient and interpretable paradigm for inverse materials design with multi-objectives. Human-AI collaboration corroborated by an interactive intelligent user interface will assist to complement the missing info in the inverse design loop.

This methodology is initially framed to search for high-entropy alloys that tend to form single-phase structures using an experimental dataset.
The dataset takes input and output as the chemical formulas and binary phase formation respectively, as detailed in our [previous work](https://www.sciencedirect.com/science/article/pii/S0927025624001460). Although it is demonstrated for single property, it is trivial to extend the framework for optimizing multiple properties all at once.

The disentangled variational encoder consists of a recognition model (encoder) and a generative model (decoder). It leverages all useful information---both labelled and unlabelled data---to learn a probabilistic relationship between the features, latent variables and target properties. An schematic illustration of the semi-supervised learning approach is shown below. The key to this approach is that the latent variables are now conditioned on the binary phase formation and the generated compositions at the decoder side also depends on the probability of single-phase formation.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-xai-inverse-materials-design/ss_vae.png" title="Disentangled VAE" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Disentangled variational autoencoder (VAE) for learning single-phase formation of high-entropy alloys: Generative model (Left) and Recognition model (Right).
</div>

Using a well-trained VAE, we can now map the latent variables for all data points and we color it by the ground truth label (binary phase formation), as shown in the left panel of Figure below. One can see that the high-dimensional composition space is transformed to a compact low-dimensional latent representation where most data points are concentrated in a small region.
More importantly, the target property is explicitly liberated from the latent space as points at the same latent positions can be either multiple-phase or single-phase alloys.
In contrast, the latent space is implicitly associated with other properties. For instance, in the right panel of Figure below, we created two groups of element lists and we combined four elements from each list and made four-element equimolar high-entropy alloys. We found that different types of alloys are actually located in well-separated regions in the latent space.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-xai-inverse-materials-design/latent_summary.png" title="Latent representation" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Latent representation for all data points: Disentangled single-phase formation (Left) and Associated alloy types (Right).
</div>

With this disentangled representation, we devised an iterative procedure to find single-phase alloys starting from a multi-phase alloy, in hope that alloys with similar element constituents will be identified in the end. The workflow is shown below and an example is given for benchmarking.
The start alloy is Fe14Ni16Cr22Co14Al22Cu8 and the inverted alloy is Fe21Ni22Cr22Co35. It makes intuitive sense as Cu and Al are not desired elements for the formation of a single-phase alloy in high-entropy alloys comprising of base elements Fe, Cr, Co and Ni.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-xai-inverse-materials-design/iterative_search_workflow.png" title="Inversion of a material" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    An iterative process to design single-phase alloys with similar constituents using the disentangled VAE.
</div>

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-xai-inverse-materials-design/inversion_sequence.png" title="Step-by-step inversion" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    The chemical formula, predicted single-phase probability and latent variables at each step of the iterative inversion process.
</div>

One can also carry out post-hoc analysis to extract additional interpretation of the inversion process. Figure below shows the overall feature importance evaluated on the test dataset (left) and the each feature value for the start and inverted alloy (right).
One can see that the trained VAE pushes the initial multi-phase alloy towards an alloy with smaller size difference and mixing entropy as well as a higher molar volume and melting temperature. This is in good agreement with the overall contribution of each feature.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/hea-xai-inverse-materials-design/shapley_interpretability.png" title="shapley analysis" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Post-hoc analysis for improved interpretability: Overall feature importance on the test data (Left) and the individual feature values for both start and inverted alloys (Right).
</div>
