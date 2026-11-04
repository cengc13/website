---
layout: page
title: Data science course projects
description: Final projects from my M.S. in Data Science at Brown, covering machine learning, data engineering, statistics and AI ethics
img: assets/img/projects/course-projects/interdisciplinary.jpg
importance: 2
category: Data science
---

Through Brown's Open Graduate Education Program, which lets selected doctoral students earn a master's degree in a second field, I completed an **M.S. in Data Science** alongside my Ph.D. My doctoral research sits at the boundary between physical chemistry, machine learning and solid mechanics, and the degree broadened its data science foundation. Below are final projects from four of the courses.

---

### **Energy performance of residential buildings**

**Hands-on Data Science** · Individual project

<div class="row align-items-center mt-3">
    <div class="col-md-4 mb-3 mb-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/course-projects/green-buildings.jpg" title="Energy-efficient buildings" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-md-8">
        <p>I used machine learning to predict the energy efficiency of residential buildings, measured as heating and cooling loads, from building geometry features. The project follows the full workflow: exploratory data analysis, a pipeline with hyperparameter tuning and cross-validation, and global and local feature importance analysis.</p>
        <p>Links: <a href="https://github.com/cengc13/DATA1030_final_project/blob/master/reports/final_report/final_report.pdf">final report</a> · <a href="https://github.com/cengc13/DATA1030_final_project">code</a></p>
    </div>
</div>

---

### **US Covid-19 tracker**

**Data Engineering** · Team "Old Boys": Cheng Zeng (leader), Tianqi Tang, Zhi Wang

We built an interactive web app that tracks Covid-19 cases and deaths across the United States at national, state and county levels, using daily data from the New York Times.

- **Data pipeline:** the app scrapes the NYT data daily with incremental, de-duplicated updates and stores it in MongoDB. Static data on mask-wearing propensity, state population and state area are added to study how face coverings and population density relate to the spread.
- **Architecture:** a database abstraction layer handles all queries, and a Plotly Dash app serves the pages and updates the interactive components.
- **Findings:** infection rate is strongly correlated with state population density, and mask-wearing propensity is strongly negatively correlated with the case fatality rate. A simple regression model projects near-term case and death counts.

<div class="row justify-content-sm-center">
    <div class="col-md-7 mt-3 mt-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/course-projects/webapp_architecture.png" title="Web application architecture" class="img-fluid rounded z-depth-1"%}
    </div>
</div>
<div class="caption">
    Architecture of the Covid-19 tracker web application.
</div>

Links: [code](https://github.com/cengc13/data1050-final-project) · [ETL & EDA notebook](https://github.com/cengc13/data1050-final-project/blob/main/ETL_EDA.ipynb) · [visualization notebook](https://github.com/cengc13/data1050-final-project/blob/main/Visualization.ipynb) · [enhancement notebook](https://github.com/cengc13/data1050-final-project/blob/main/Enhancement.ipynb)

---

### **African economic crisis analysis**

**Statistical Learning** · Team: Hanxiao Chen, Zeyan Du, Yiwei Sang, Zhi Wang, Cheng Zeng, Yue Zhuang

<div class="row align-items-center mt-3">
    <div class="col-md-4 mb-3 mb-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/course-projects/economic-crisis.png" title="African economic crisis" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-md-8">
        <p>We applied a range of statistical learning methods to the African Economic Crisis dataset: logistic regression, models with engineered and time-series features, and multilevel models. Using residual plots, confusion matrices and ROC curves, we identified the most informative indicators of systemic and currency crises.</p>
    </div>
</div>

---

### **AI ethics in criminal justice**

**Data and Society** · Team: Ben Xiong, Jillian Green, Christopher Rohlicek, Akshay Shah, Qingyan Guo, Cheng Zeng, Emmanuel Peters (no author order)

<div class="row align-items-center mt-3">
    <div class="col-md-4 mb-3 mb-md-0">
        {% include figure.html zoomable="true" path="assets/img/projects/course-projects/algorithm-justice.png" title="Algorithmic justice" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-md-8">
        <p>As AI takes on tasks once thought to require human judgment, we asked what it should and should not do within the justice system. The project covers the roles of AI and humans in the judicial system, the risks of over-relying on AI, and ethical perspectives. It includes a case study of the COMPAS risk-assessment algorithm, focusing on algorithmic fairness and bias, and ends with recommended solutions and their limitations.</p>
    </div>
</div>
