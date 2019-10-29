/** Opacity of the colored cell background */
const CELL_BACKGROUND_OPACITY = 0.3;

/** Use a global color scale from [0-100] rather than per-column scales. */
const USE_GLOBAL_COLOR_SCALE = false;

/** Formatting function for numbers. */
const formatToNumber = d3.format('.3n');

/** Keeps track of included dateset. */
const includedDatasets = {};

/** Sort column per dataset */
const sortKeys = {
  'summary': 'Mean (1000 examples)',
  'full': 'Mean (selected datasets)',
  'sampled': 'Mean (selected datasets)',
};

/**
 * Formats a cell value as number if it can be converted.
 * @param {string} textOrNumber string value of the cell.
 * @return {string}
 */
function formatCell(textOrNumber) {
  if (isNaN(+textOrNumber)) {
    return textOrNumber;
  }
  return formatToNumber(textOrNumber);
}

/**
 * Generate a color scale used for the given array of values.
 * @param {!Array<number>} data
 * @return {*}
 */
function computeColorScale(data) {
  const min = d3.min(data);
  const max = d3.max(data);
  const center = d3.mean([min, max]);
  return d3.scaleLinear()
      .domain([min, center, max])
      .range(['#d73027', '#fee08b', '#1a9850'])
      .interpolate(d3.interpolateHcl);
}

/** Default color scale with domain in [0-100]. */
const GLOBAL_COLOR_SCALE = computeColorScale([0, 50, 100]);

/**
 * Return the color associated with a cell value.
 * @param {string|number} textOrNumber string value of the cell.
 * @param {*} colorScale
 * @return {string}
 */
function computeCellColor(textOrNumber, colorScale) {
  const value = +textOrNumber;
  if (isNaN(value)) {
    return d3.color('white');
  }
  const color = d3.color(colorScale(value));
  color.opacity = CELL_BACKGROUND_OPACITY;
  return color;
}

/**
 * Draws the benchmark table.
 * @param {string} name Reference name for the table.
 * @param {!Element} container
 * @param {!Array<*>} models
 * @param {!Array<*>} metrics
 * @param {!Array<*>} data
 */
function drawBenchmarkTable(name, container, models, metrics, data) {
  const sortKey = sortKeys[name];

  // Create the table structure.
  const table =
      container.selectAll('table').data([null]).enter().append('table');
  table.append('thead').append('tr');
  table.append('tbody');

  // Sort function called in response to header interaction.
  const sortBy = (d, i) => {
    if (i === 0) {
      sortKeys[name] = undefined;  // Clear on first column click.
    } else if (sortKeys[name] === d.name) {
      sortKeys[name] = undefined;  // Disable sort when clicked again.
    } else {
      sortKeys[name] = d.name;
    }
    drawBenchmarkTable(name, container, models, metrics, data);
  };

  // Create dynamic data.
  // Note the merge() to update both new and existing elements in one call.
  const columnHeaders =
      container.select('thead tr').selectAll('th').data([{}, ...metrics]);
  columnHeaders.enter()
      .append('th')
      .merge(columnHeaders)
      .text(d => d.name || '')
      .attr('title', d => d.name || '')
      .attr('role', 'button')
      .attr('tabindex', 0)
      .classed('sort-key', d => d.name === sortKey)
      .on('click', (d, i) => sortBy(d, i))
      .on('keypress', (d, i) => {
        if (d3.event.keyCode === 32 || d3.event.keyCode === 13) {
          sortBy(d, i);
        }
      });
  columnHeaders.exit().remove();

  const rows = container.select('tbody').selectAll('tr').data(data);
  rows.enter().append('tr');
  rows.exit().remove();

  // Compute color scales by considering each column separately.
  const colorScales =
      metrics.map((m, i) => computeColorScale(data.map((r, j) => r[i])));

  // Sort model and data based on sortKey.
  const sortKeyIndex = metrics.findIndex(m => m.name === sortKey);
  // Get the desired order of IDs based on the sortKeys.
  const sortedIds = models.map((d, i) => i);
  sortedIds.sort(
      (i, j) => sortKeyIndex >= 0 ?
          data[j][sortKeyIndex] - data[i][sortKeyIndex] :  // Descending.
          0);
  // Sort models and data.
  const sortedModels = sortedIds.map(i => models[i]);
  const sortedData = sortedIds.map(i => data[i]);

  const modelCells = container.selectAll('tbody tr')
                         .selectAll('td.model')
                         .data((d, i) => [sortedModels[i]]);
  modelCells.enter()
      .append('td')
      .classed('model', true)
      .append('a')
      .merge(modelCells.select('a'))
      .attr('href', d => d.url)
      .attr('title', d => d.overtext)
      .text(d => d.name);

  const dataCells = container.selectAll('tbody tr')
                        .selectAll('td.data')
                        .data((d, i) => sortedData[i]);
  dataCells.enter()
      .append('td')
      .classed('data', true)
      .merge(dataCells)
      .text(d => formatCell(d))
      .style(
          'background-color',
          (d, i) => computeCellColor(
              d, USE_GLOBAL_COLOR_SCALE ? GLOBAL_COLOR_SCALE : colorScales[i]));
  dataCells.exit().remove();
}

/**
 * Preprocesses the CSV data and returns a structured object describing the
 * benchmark tables.
 * @param {*} rows
 * @return {*}
 */
function extractData(rows) {
  const metadataNumber = rows[0].filter(e => e == "metadata").length;
  // Assume same datasets for full and 1000.
  const datasetCount = rows[0].filter(e => e == "full").length;
  const dataRows = rows.splice(3);
  const models = dataRows.map(
      row => ({
        name: row[0],
        url: row[row.length - metadataNumber],
        overtext: (`Architecture: ${row[row.length - metadataNumber + 1]} \n` +
                   `Hyper sweep size: ${row[row.length - metadataNumber + 2]} \n` +
                   `Preprocessing: ${row[row.length - metadataNumber + 3]}`),
      }));
  const headerWithDatasetNames = rows[1];

  // Leave out the first and last column.
  // Assume names are the same for sampled and full data.

  const datasets =
      headerWithDatasetNames.slice(1, 1 + datasetCount).map(name => ({name}));

  const dataSampled =
      dataRows.map(row => row.slice(1, 1 + datasetCount).map(v => +v));
  const dataFull = dataRows.map(
      row => row.slice(1 + datasetCount, 1 + datasetCount * 2).map(v => +v));

  // Enable all datasets.
  for (const d of datasets) {
    includedDatasets[d.name] = true;
  }

  return {
    models,
    datasets,
    dataSampled,
    dataFull,
  };
}

/**
 * Draws checkboxes to allow excluding datasets.
 * @param {*} extractedData
 */
function drawCheckboxes(extractedData) {
  const datasetCheckboxes =
      d3.select('div.datasets').selectAll('input').data(extractedData.datasets);
  const inputContainers = datasetCheckboxes.enter().append('div');
  const inputLabels = inputContainers.append('label');
  inputLabels.append('input')
      .attr('type', 'checkbox')
      .attr('name', d => d.name)
      .attr('checked', d => !!includedDatasets[d.name])
      .on('change', (d, i, elements) => {
        includedDatasets[d.name] = elements[i].checked;
        redrawBenchmarks(extractedData);
      });
  inputLabels.append('text').text(d => d.name);
}

/**
 * Filter data based on the selected benchmarks and display a table with
 * selected datasets and means.
 * @param {string} name Reference name for the table.
 * @param {!Element} container
 * @param {!Array<*>} models
 * @param {!Array<*>} datasets
 * @param {!Array<*>} data
 */
function drawFilteredBenchmarkTable(name, container, models, datasets, data) {
  const filteredDatasets = datasets.filter(d => includedDatasets[d.name]);
  const filteredData =
      data.map(r => r.filter((d, i) => includedDatasets[datasets[i].name]));
  const metrics = [{name: 'Mean (selected datasets)'}, ...filteredDatasets];
  const filteredDataWithMean = filteredData.map(r => [d3.mean(r), ...r]);
  drawBenchmarkTable(name, container, models, metrics, filteredDataWithMean);
}

/**
 * Recomputes metrics based on selected datasets and redraws all benchmarks.
 * @param {*} extractedData
 */
function redrawBenchmarks(extractedData) {
  drawFilteredBenchmarkTable(
      'sampled', d3.select('div.benchmark.sampled'), extractedData.models,
      extractedData.datasets, extractedData.dataSampled);

  drawFilteredBenchmarkTable(
      'full', d3.select('div.benchmark.full'), extractedData.models,
      extractedData.datasets, extractedData.dataFull);

  // Compute other metrics from the raw data.
  const meansMetrics = [
    {
      name: 'Mean (1000 examples)',
      compute: (i) => d3.mean(extractedData.dataSampled[i]),
    },
    {
      name: 'Mean (full)',
      compute: (i) => d3.mean(extractedData.dataFull[i]),
    }
  ];
  const meansData = extractedData.models.map(
      (m, i) => meansMetrics.map(metric => metric.compute(i)));

  drawBenchmarkTable(
      'summary', d3.select('div.benchmark.summary'), extractedData.models,
      meansMetrics, meansData);
}

// Load the CSV data and draw the benchmark table.
d3.text('data/results.csv', csv => {
  const rows = d3.csvParseRows(csv);
  const extractedData = extractData(rows);

  drawCheckboxes(extractedData);
  redrawBenchmarks(extractedData);
});
