---
layout: page
permalink: /publications/
title: Publications
description: An up-to-date full publication list lives at <a href='https://scholar.google.com/citations?user=3s8uxxkAAAAJ&hl=en'>Google Scholar</a>.
years: [2024, 2023, 2022, 2020, 2019, 2018, 2017, 2016, 2015]
nav: true
nav_order: 1
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f {{ site.scholar.bibliography }} -q @*[year={{y}}]* %}
{% endfor %}

</div>
