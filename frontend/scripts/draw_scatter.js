// draw_scatter.js - Scatterplot visualization module

function drawScatter(svg, data, settings) {
  if (!data || !data.instances || data.instances.length === 0) {
    console.warn("No data available for scatter plot");
    return;
  }

  const margin = { top: 40, right: 40, bottom: 40, left: 40 };
  const width = 800 - margin.left - margin.right;
  const height = 600 - margin.top - margin.bottom;

  // Get data bounds
  const xExtent = d3.extent(data.instances, d => d.x);
  const yExtent = d3.extent(data.instances, d => d.y);

  // Add padding to bounds
  const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
  const yPadding = (yExtent[1] - yExtent[0]) * 0.1;

  const xScale = d3.scaleLinear()
    .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
    .range([margin.left, width - margin.right]);

  const yScale = d3.scaleLinear()
    .domain([yExtent[1] + yPadding, yExtent[0] - yPadding])
    .range([margin.top, height - margin.bottom]);

  const colorScale = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3[settings.colorScheme]);

  const sizeScale = d3.scaleLinear()
    .domain([0, 1])
    .range([3, settings.pointSize]);

  const labelColor = d3.scaleOrdinal(d3.schemeCategory10);

  // Create main group
  const g = svg.append('g');

  // Add points
  const points = g.selectAll('.point')
    .data(data.instances)
    .enter()
    .append('g')
    .attr('class', 'point');

  // Add circles
  points.append('circle')
    .attr('cx', d => xScale(d.x))
    .attr('cy', d => yScale(d.y))
    .attr('r', d => sizeScale(d.fitness))
    .attr('fill', d => labelColor(d.label))
    .attr('stroke', 'white')
    .attr('stroke-width', 1)
    .attr('opacity', 0.8)
    .on('mouseover', (event, d) => showTooltip(event, d))
    .on('mouseout', () => hideTooltip());

  // Add axes
  drawAxes(g, xScale, yScale, margin, width, height);

  function showTooltip(event, d) {
    const tooltip = d3.select('#tooltip');
    tooltip.html(`
      <strong>Genoma:</strong> ${d.id}<br>
      <strong>Geração:</strong> ${d.generation}<br>
      <strong>Fitness:</strong> ${d.fitness.toFixed(3)}<br>
      <strong>Posição:</strong> (${d.x.toFixed(2)}, ${d.y.toFixed(2)})
    `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px')
      .classed('show', true);
  }

  function hideTooltip() {
    d3.select('#tooltip').classed('show', false);
  }

  function drawAxes(g, xScale, yScale, margin, width, height) {
    // X axis
    g.append('g')
      .attr('transform', `translate(0, ${height / 2})`)
      .call(d3.axisBottom(xScale))
      .attr('class', 'axis');

    // Y axis
    g.append('g')
      .attr('transform', `translate(${width / 2}, 0)`)
      .call(d3.axisLeft(yScale))
      .attr('class', 'axis');

    // Axis labels
    g.append('text')
      .attr('x', width / 2)
      .attr('y', height + margin.bottom - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#666')
      .text('Componente 1');

    g.append('text')
      .attr('x', -height / 2)
      .attr('y', margin.left - 10)
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('font-size', '14px')
      .attr('fill', '#666')
      .text('Componente 2');
  }
}
