// draw_trajectories.js - Trajectory visualization module

function drawTrajectories(svg, data, settings) {
  if (!data || !data.trajectories || data.trajectories.length === 0) {
    console.warn("No trajectory data available");
    return;
  }

  const margin = { top: 40, right: 40, bottom: 40, left: 40 };
  const width = 800 - margin.left - margin.right;
  const height = 600 - margin.top - margin.bottom;

  // Get data bounds from all instances for consistent scaling
  const allInstances = data.instances || [];
  const xExtent = d3.extent(allInstances, d => d.x);
  const yExtent = d3.extent(allInstances, d => d.y);

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

  const labelColor = d3.scaleOrdinal(d3.schemeCategory10);

  // Create main group
  const g = svg.append('g');

  // Create line generator with fallback for bundling
  let curveBundle = d3.curveBasis;
  if (d3.curveBundle && typeof d3.curveBundle.beta === 'function') {
    curveBundle = d3.curveBundle.beta(0.85);
  }
  const line = d3.line()
    .x(d => d && d.x !== undefined ? xScale(d.x) : null)
    .y(d => d && d.y !== undefined ? yScale(d.y) : null)
    .curve(curveBundle);

  // Defensive: filter out trajectories with malformed points
  const safeTrajectories = data.trajectories.map(traj => ({
    ...traj,
    points: (traj.points || []).filter(p => p && typeof p.x === 'number' && typeof p.y === 'number')
  })).filter(traj => traj.points.length > 1);

  // --- EDGE BUNDLING REAL ---
  // Converter trajetórias para formato de edges para o bundling
  const edges = safeTrajectories.map(traj => ({
    source: traj.points[0],
    target: traj.points[traj.points.length - 1],
    points: traj.points
  }));

  // Usar d3.ForceEdgeBundling
  if (typeof d3.ForceEdgeBundling === 'function') {
    const bundle = d3.ForceEdgeBundling()
      .step_size(0.1)
      .compatibility_threshold(0.6);
    const bundled = bundle(edges);
    // O resultado é um array de arrays de pontos suavizados
    // Substituir os pontos das trajetórias pelos pontos suavizados
    safeTrajectories.forEach((traj, i) => {
      if (bundled[i]) {
        traj.points = bundled[i].map(p => ({ x: p[0], y: p[1] }));
      }
    });
  }

  // Debug: log trajectories and points
  console.log('Trajetórias para desenhar:', safeTrajectories);

  // Add trajectories
  g.selectAll('.trajectory')
    .data(safeTrajectories)
    .enter()
    .append('path')
    .attr('class', 'trajectory')
    .attr('d', d => {
      console.log('Desenhando trajetória:', d.points);
      return line(d.points);
    })
    .attr('fill', 'none')
    .attr('stroke', d => labelColor(d.label))
    .attr('stroke-width', settings.trajectoryWidth)
    .attr('opacity', 0.25)
    .on('mouseover', (event, d) => showTrajectoryTooltip(event, d))
    .on('mouseout', () => hideTooltip());

  // Add trajectory points (optional)
  if (settings.showTrajectoryPoints) {
    g.selectAll('.trajectory-point')
      .data(safeTrajectories.flatMap(t => t.points.map(p => ({ ...p, trajectoryId: t.id }))))
      .enter()
      .append('circle')
      .attr('class', 'trajectory-point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', 2)
      .attr('fill', '#666')
      .attr('opacity', 0.5);
  }

  function showTrajectoryTooltip(event, d) {
    const tooltip = d3.select('#tooltip');
    const avgFitness = d3.mean(d.points, p => p.fitness || 0.5);
    const generations = d.points.map(p => p.generation).sort((a, b) => a - b);

    tooltip.html(`
      <strong>Trajetória:</strong> ${d.id}<br>
      <strong>Gerações:</strong> ${generations[0]} → ${generations[generations.length - 1]}<br>
      <strong>Pontos:</strong> ${d.points.length}<br>
      <strong>Fitness Médio:</strong> ${avgFitness.toFixed(3)}
    `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px')
      .classed('show', true);
  }

  function hideTooltip() {
    d3.select('#tooltip').classed('show', false);
  }
}
