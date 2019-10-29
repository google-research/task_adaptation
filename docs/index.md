---
layout: home
---

<!-- Include abstract -->
{% capture abstract %}{% include abstract.md %}{% endcapture %}
{{ abstract | markdownify }}

<h2>Leaderboard summary</h2>
<div class="benchmark summary"></div>
[Detailed leaderboard](benchmark)<br/>

<script src="https://d3js.org/d3.v4.min.js"></script>
<script src="benchmark_data.js"></script>
<link rel="stylesheet" type="text/css" href="benchmark_data.css" />
