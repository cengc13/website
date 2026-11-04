---
layout: page
title: Machine learning high-entropy alloys
description: Predictive models and inverse design for corrosion-resistant high-entropy alloys
img: assets/img/projects/ml-hea-design/ml-hea-thumbnail.png
importance: 1
category: Past research
---

Cheng Zeng is grateful to Alfond Post Doc Research Fellowship for supporting the research work at the Roux institute of Northeastern University. Computational simulations are carried out using the Discovery cluster, supported by the Research Computing team at Northeastern University.

This project covers three related lines of work on high-entropy alloys (HEAs): a machine learning framework for discovering corrosion-resistant HEAs, a semi-supervised variational autoencoder for inverse design of single-phase HEAs, and a perspective on machine learning methods for HEA design.

- [Discovery of high-entropy alloys for corrosion protection](#discovery-of-high-entropy-alloys-for-corrosion-protection)
- [Inverse design of single-phase high-entropy alloys](#inverse-design-of-single-phase-high-entropy-alloys)
- [Perspective of materials design with machine learning](#perspective-of-materials-design-with-machine-learning)

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
        {% include figure.html path="assets/img/projects/hea-corrosion/workflow.jpg" title="Workflow of ML HEA corrosion" class="img-fluid rounded z-depth-1"%}
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

### **Inverse design of single-phase high-entropy alloys**

Conventional inverse materials design use unsupervised machine learning where the input and output are both chemical compositions, crystal structures or microstructure images without considering the composition/structure-property relationship explicitly. The unsupervised learning project high-dimensional features encoding compositions/structures into a low-dimensional latent space and reconstruct the features in the output. Sampling in the latent space generates a new composition/structure. This unsupervised scheme is theorized to learn the entangled composition/structure-property relationship in an implicit manner, which can only be modeled through a separate surrogate model. The surrogate model and latent space representation combined enables the sampling of materials with desired properties. However, this entangled and implicit scheme has three drawbacks. First, the entangled representation of composition/structure-property relationship is by nature lacking interpretability. Second, without inclusion of target property, the learned latent space can fail to form reliable composition/structure-property relationship. Lastly, this unsupervised approach is not suitable for inverse design of materials with multiple desired properties as all these info can be highly entangled in the latent space.

We propose to use a custom semi-supervised variational autoencoder to disentangle the relationship between compositions and multiple materials property. This approach includes a recognition (encoder) and a generative (decoder) model. By using informative priors for the target properties and inputs, we can learn the composition-property relationship in a disentangled and data-efficient way. Besides, it allows us to generate a target material with pre-defined multiple materials properties. Adding post-hoc analysis using existing techniques such as shapley values (SHAP), it innovates a data-efficient and interpretable paradigm for inverse materials design with multi-objectives. Human-AI collaboration corroborated by an interactive intelligent user interface will assist to complement the missing info in the inverse design loop.

This methodology is initially framed to search for high-entropy alloys that tend to form single-phase structures using an experimental dataset.
The dataset takes input and output as the chemical formulas and binary phase formation respectively, as detailed in our [previous work](https://www.sciencedirect.com/science/article/pii/S0927025624001460). Although it is demonstrated for single property, it is trivial to extend the framework for optimizing multiple properties all at once.

The disentangled variational encoder consists of a recognition model (encoder) and a generative model (decoder). It leverages all useful information---both labelled and unlabelled data---to learn a probabilistic relationship between the features, latent variables and target properties. An schematic illustration of the semi-supervised learning approach is shown below. The key to this approach is that the latent variables are now conditioned on the binary phase formation and the generated compositions at the decoder side also depends on the probability of single-phase formation.

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
In contrast, the latent space is implicitly associated with other properties. For instance, in the right panel of Figure below, we created two groups of element lists and we combined four elements from each list and made four-element equimolar high-entropy alloys. We found that different types of alloys are actually located in well-separated regions in the latent space.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/latent_summary.png" title="Latent representation" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Latent representation for all data points: Disentangled single-phase formation (Left) and Associated alloy types (Right).
</div>

With this disentangled representation, we devised an iterative procedure to find single-phase alloys starting from a multi-phase alloy, in hope that alloys with similar element constituents will be identified in the end. The workflow is shown below and an example is given for benchmarking.
The start alloy is Fe14Ni16Cr22Co14Al22Cu8 and the inverted alloy is Fe21Ni22Cr22Co35. It makes intuitive sense as Cu and Al are not desired elements for the formation of a single-phase alloy in high-entropy alloys comprising of base elements Fe, Cr, Co and Ni.

<div class="row justify-content-sm-center">
    <div class="col-sm-12 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/hea-xai-inverse-materials-design/iterative_search_workflow.png" title="Inversion of a material" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    An iterative process to design single-phase alloys with similar constituents using the disentangled VAE.
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

### **Perspective of materials design with machine learning**

High-entropy alloys, defined as alloys with no less than five elements with almost equiatomic compositions, are emerging materials with superior corrosion resistance, irradiation resistance, strength and catalytic activity.
The complexity of defect structure is shown in the below figure.
However, design of high-entropy alloys is challenging as many factors should be considered such as compositions, configuration, defect structures and processing of those materials. Thanks to the advance of computer hardware and algorithms, the past decade has witnessed success of ML enabled discovery of high-entropy alloys.
Purposes of machine learning high-entropy alloys include prediction, knowledge mining and optimization.
Machine learning methods used for high-entropy alloys can essentially be grouped into three categories. Each category, its current state-of-the-art implementation and future challenges are outlined below.

<div class="row justify-content-sm-center">
    <div class="col-sm-5 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/ml-hea-design/hea-structures.jpg" title="HEA various defects" class="img-fluid rounded z-depth-1"%}
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/ml-hea-design/hea-ml-ability.png" title="HEA ML purposes" class="img-fluid rounded z-depth-1" top=30%}
    </div>
</div>
<div class="caption">
    Possible defects in high-entropy alloys (Left), source:<a href="https://www.nature.com/articles/s41578-019-0121-4">George et al</a>. Machine learning complex multidimensional space of high-entropy alloys (Right), source: <a href="https://www.science.org/doi/10.1126/science.abn3103?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed">Yao et al</a>.
</div>

The mainstream is machine learning inter-atomic potentials (MLIP) that aim to fit inter-atomic interactions based on first-principles data. The fitted machine leaning models can accelerate atomistic simulations by orders of magnitude compared to first-principles calculations. This category belongs to the forward design with machine leaning models. Schematic of typical finite-ranged machine learning potentials is depicted in the below figure. Atomic positions are first transformed to features via representations invariant to rotation, translation and reflection. Each atom has its own feature vector to describe the atom interacting with its neighbors. Atomic energies are approximated by machine learning algorithms, which in the end are summed to yield the total energy. The training iterates by minimizing the energy (and forces) differences between machine learning models and *ab initio* calculations. Since high-entropy materials live in high-dimensional configuration space, a large amount of training data and features are needed. In addition, current MLIPs mostly use element-specific descriptors which significantly limit the transferability of MLIP when predicting materials comprising elements not existent in the training structures. Besides, the MLIPs only perform well in regions where training data are sufficient, necessitating effective sampling of informative structures. **Future research directions in this line include develop a universal element-agnostic representation of atomic structures, robust machine learning architecture for inter-atomic interactions and efficient sampling methods.**

<div class="row justify-content-sm-center">
    {% include figure.html path="assets/img/projects/ml-hea-design/mlip-scheme.png" title="Scheme of MLIP" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Schematic of machine learning inter-atomic potential, source: <a href='https://www.sciencedirect.com/science/article/pii/S0010465516301266?via%3Dihub'>Khorshidi and Peterson</a>
</div>

The second category features generative models for inverse materials design. The basic idea of generative models is to learn a probabilistic mapping between continuous latent space variables and materials properties. A typical generative model uses variational autoencoder where inputs are projected to a low dimensional space by an encoder and then a decoder maps the latent space back to the same inputs. Nevertheless, general models often suffer from the overfitting issue, leading to a high failure rate when predicted materials are validated through first-principles calculations or experiments. Generative models also strongly depend on massive data, in particular for complex materials like high-entropy alloys. In addition, the inverse of materials from atomic representation is very challenging because the mapping between configuration space and latent space is not invertible for most machine learning methods. As a result, **Incorporation of uncertainty quantification and active learning is required to balance the exploitation and exploration of inverse materials design using generative models. Future development of data-efficient and interpretable algorithms are needed to unleash the full potential of generative methods. Future advance in invertible representation is necessary to expand use cases of generative models.**

<div class="row justify-content-sm-center">
        {% include figure.html path="assets/img/projects/ml-hea-design/generative_design.jpg" title="Variational autoencoder" class="img-fluid rounded z-depth-1"%}
</div>
<div class="caption">
    Inverse materials design with variational autoencoder, source: <a href='https://www.nature.com/articles/s41578-021-00340-w'>Hart et al</a>.
</div>

The last category represents machine learning models based on experimental data. This usually takes input as chemical compositions and output as materials properties. The machine learning models can be generic machine learning models (e.g. random forest classifier) for property prediction. Alternatively, generative models can be built for inverse materials design or materials optimization. The issue with this machine learning method is rooted in the quality of experimental data. It is of central importance to ensure the compiled data follow the <a href='https://www.nature.com/articles/sdata201618'>FAIR principle</a> because of the "Garbage in, Garbage out" principle. More generally, all the data used in materials design loop (see the below figure) should be Findable, Accessible, Interoperable and Reusable. For materials dataset, it is crucial to examine the processing conditions of how materials are made, especially for high-entropy alloys which can be manufactured  either via conventional arc melting and casting or through additive manufacturing with high cooling rates. Moreover, the curation of new data should strictly follow the data principle as well. **Collective efforts are needed among the materials researcher community to safeguard the data quality. Researchers who collect the data should take the caution of data provenance and data governance. Design algorithms that are resistant to data noise are also promising future directions.**

<div class="row justify-content-sm-center">
    <div class="col-sm-10 mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/ml-hea-design/data-centric-materials-design.jpg" title="Data centric framework" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Data-centric framework for materials design, source: <a href='https://www.sciencedirect.com/science/article/pii/S209580992200056X'>Chen et al</a>.
</div>
