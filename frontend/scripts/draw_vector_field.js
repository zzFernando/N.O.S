// draw_vector_field.js - Vector field visualization module

function drawVectorField(svg, data, settings) {
  if (!data || !data.vector_field || data.vector_field.length === 0) {
    console.warn("No vector field data available");
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

  // Create main group
  const g = svg.append('g');

  // Create arrow marker
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 8)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#999');

  // Scale for arrow opacity based on magnitude
  const magnitudeExtent = d3.extent(data.vector_field, d => d.magnitude);
  const opacityScale = d3.scaleLinear()
    .domain(magnitudeExtent)
    .range([0.1, 0.4]);

  // Scale for arrow width based on magnitude
  const widthScale = d3.scaleLinear()
    .domain(magnitudeExtent)
    .range([0.5, 2]);

  // Add vector arrows
  /*
  g.selectAll('.vector-arrow')
    .data(data.vector_field)
    .enter()
    .append('line')
    .attr('class', 'vector-arrow')
    .attr('x1', d => xScale(d.x))
    .attr('y1', d => yScale(d.y))
    .attr('x2', d => xScale(d.x + d.dx * 0.8)) // Scale down for visibility
    .attr('y2', d => yScale(d.y + d.dy * 0.8))
    .attr('stroke', '#bbb')
    .attr('stroke-width', d => widthScale(d.magnitude))
    .attr('opacity', d => opacityScale(d.magnitude))
    .attr('marker-end', 'url(#arrowhead)')
    .on('mouseover', (event, d) => showVectorTooltip(event, d))
    .on('mouseout', () => hideTooltip());
  */

  // --- STREAMLINES ---
  drawStreamlines(g, data.vector_field, xScale, yScale);

  function showVectorTooltip(event, d) {
    const tooltip = d3.select('#tooltip');
    const angle = Math.atan2(d.dy, d.dx) * 180 / Math.PI;

    tooltip.html(`
      <strong>Campo Vetorial</strong><br>
      <strong>Posição:</strong> (${d.x.toFixed(2)}, ${d.y.toFixed(2)})<br>
      <strong>Direção:</strong> (${d.dx.toFixed(3)}, ${d.dy.toFixed(3)})<br>
      <strong>Magnitude:</strong> ${d.magnitude.toFixed(3)}<br>
      <strong>Ângulo:</strong> ${angle.toFixed(1)}°
    `)
      .style('left', (event.pageX + 10) + 'px')
      .style('top', (event.pageY - 10) + 'px')
      .classed('show', true);
  }

  function hideTooltip() {
    d3.select('#tooltip').classed('show', false);
  }
}

// Função para desenhar streamlines reais
function drawStreamlines(g, vectorField, xScale, yScale) {
  // Parâmetros
  const nSeeds = 80; // número de linhas de fluxo (aumentado)
  const steps = 40; // passos por linha
  const stepSize = 0.18; // tamanho do passo

  // Extrair bounds
  const xVals = vectorField.map(v => v.x);
  const yVals = vectorField.map(v => v.y);
  const xMin = Math.min(...xVals), xMax = Math.max(...xVals);
  const yMin = Math.min(...yVals), yMax = Math.max(...yVals);

  // Função para interpolar vetor local
  function getVectorAt(x, y) {
    // Busca o vetor mais próximo (pode ser melhorado para interpolação bilinear)
    let minDist = Infinity, best = null;
    for (const v of vectorField) {
      const dist = (v.x - x) ** 2 + (v.y - y) ** 2;
      if (dist < minDist) {
        minDist = dist;
        best = v;
      }
    }
    return best ? [best.dx, best.dy] : [0, 0];
  }

  // Gerar pontos de semente em grade regular (fixo)
  const seeds = [];
  const nX = 8, nY = 10; // ajuste para densidade desejada
  for (let i = 0; i < nX; i++) {
    for (let j = 0; j < nY; j++) {
      const sx = xMin + (i + 0.5) * (xMax - xMin) / nX;
      const sy = yMin + (j + 0.5) * (yMax - yMin) / nY;
      seeds.push([sx, sy]);
    }
  }

  // Para cada semente, traçar streamline
  for (const [sx, sy] of seeds) {
    let points = [[sx, sy]];
    let x = sx, y = sy;
    for (let s = 0; s < steps; s++) {
      const [dx, dy] = getVectorAt(x, y);
      const norm = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      x += (dx / norm) * stepSize;
      y += (dy / norm) * stepSize;
      points.push([x, y]);
      // Parar se sair dos limites
      if (x < xMin || x > xMax || y < yMin || y > yMax) break;
    }
    g.append('path')
      .datum(points)
      .attr('fill', 'none')
      .attr('stroke', '#bbb')
      .attr('stroke-width', 1.2)
      .attr('opacity', 0.3)
      .attr('d', d3.line()
        .x(d => xScale(d[0]))
        .y(d => yScale(d[1]))
        .curve(d3.curveBasis));
  }
}
