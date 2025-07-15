// main.js - Entrada principal da visualização D3.js modular

let data = null;
let settings = {
  showTrajectories: true,
  showVectorField: true,
  showLabels: true,
  maxGeneration: 15,
  minFitness: 0.0,
  pointSize: 8,
  trajectoryWidth: 2,
  colorScheme: 'viridis'
};

const DATA_PATH = 'assets/projections.json';

const labelColor = d3.scaleOrdinal(d3.schemeCategory10);

async function loadData() {
  try {
    const response = await fetch(DATA_PATH);
    if (!response.ok) throw new Error('Erro ao carregar dados: ' + response.status);
    let rawData = await response.json();

    // Adaptação para novo formato unificado (genomes, vector_field)
    if (rawData.genomes) {
      // instances: último ponto da trajetória de cada genoma
      rawData.instances = rawData.genomes.map(g => ({
        id: g.id,
        x: g.projection[0],
        y: g.projection[1],
        generation: g.generation,
        fitness: g.fitness,
        label: g.label || '',
        vector: g.vector || [0, 0]
      }));
      // trajectories: cada genoma vira uma trajetória
      rawData.trajectories = rawData.genomes.map(g => ({
        id: g.id,
        label: g.label || '',
        points: g.trajectory.map((p, i) => ({
          x: p[0],
          y: p[1],
          generation: i, // ou g.generation se for único
          fitness: g.fitness,
          label: g.label || ''
        }))
      }));
    }

    // Converter campos numéricos (retrocompatibilidade)
    if (rawData.instances) {
      rawData.instances.forEach(d => {
        d.x = Number(d.x);
        d.y = Number(d.y);
        d.fitness = Number(d.fitness);
        d.generation = Number(d.generation);
      });
    }
    if (rawData.trajectories) {
      rawData.trajectories.forEach(traj => {
        traj.points.forEach(p => {
          p.x = Number(p.x);
          p.y = Number(p.y);
          p.generation = Number(p.generation);
        });
      });
    }
    if (rawData.vector_field) {
      rawData.vector_field.forEach(v => {
        v.x = Number(v.x);
        v.y = Number(v.y);
        v.dx = Number(v.dx);
        v.dy = Number(v.dy);
        if (v.magnitude === undefined) {
          v.magnitude = Math.sqrt(v.dx * v.dx + v.dy * v.dy);
        }
      });
    }

    data = rawData;
    console.log('Dados carregados:', data);
    return data;
  } catch (error) {
    console.error('Erro ao carregar dados:', error);
    // Fallback para dados mock
    return generateMockData();
  }
}

function generateMockData() {
  const instances = [];
  const trajectories = [];
  const vector_field = [];

  // Gerar dados mock
  for (let gen = 0; gen <= 15; gen++) {
    for (let i = 0; i < 5; i++) {
      instances.push({
        id: `gen_${gen}_${i}`,
        x: Math.random() * 4 - 2,
        y: Math.random() * 4 - 2,
        generation: gen,
        fitness: Math.random() * 0.7 + 0.2
      });
    }
  }

  // Gerar trajetórias mock
  for (let i = 0; i < 3; i++) {
    const points = [];
    for (let gen = 0; gen <= 15; gen++) {
      points.push({
        x: Math.cos(gen * 0.2 + i) * (1 + gen * 0.1),
        y: Math.sin(gen * 0.2 + i) * (1 + gen * 0.1),
        generation: gen
      });
    }
    trajectories.push({
      id: `traj_${i}`,
      points: points
    });
  }

  // Gerar campo vetorial mock
  for (let x = -2; x <= 2; x += 0.5) {
    for (let y = -2; y <= 2; y += 0.5) {
      vector_field.push({
        x: x,
        y: y,
        dx: (Math.random() - 0.5) * 0.3,
        dy: (Math.random() - 0.5) * 0.3,
        magnitude: Math.random() * 0.5 + 0.1
      });
    }
  }

  return { instances, trajectories, vector_field };
}

function filterData() {
  if (!data) return { instances: [], trajectories: [], vector_field: [] };

  const filteredInstances = data.instances.filter(d =>
    d.generation <= settings.maxGeneration &&
    d.fitness >= settings.minFitness
  );

  const filteredTrajectories = data.trajectories.filter(traj => {
    const maxGen = Math.max(...traj.points.map(p => p.generation));
    return maxGen <= settings.maxGeneration;
  });

  return {
    instances: filteredInstances,
    trajectories: filteredTrajectories,
    vector_field: data.vector_field
  };
}

function updateStatistics(filteredData) {
  const instances = filteredData.instances;

  document.getElementById('genomeCount').textContent = instances.length;
  document.getElementById('generationCount').textContent =
    instances.length > 0 ? Math.max(...instances.map(d => d.generation)) + 1 : 0;
  document.getElementById('avgFitness').textContent =
    instances.length > 0 ? (d3.mean(instances, d => d.fitness) || 0).toFixed(3) : '0.000';
  document.getElementById('maxFitness').textContent =
    instances.length > 0 ? (d3.max(instances, d => d.fitness) || 0).toFixed(3) : '0.000';
}

function setupEventListeners() {
  // Generation range
  document.getElementById('genRange').addEventListener('input', (event) => {
    settings.maxGeneration = parseInt(event.target.value);
    document.getElementById('genValue').textContent = `0-${settings.maxGeneration}`;
    renderAll();
  });

  // Fitness minimum
  document.getElementById('fitnessMin').addEventListener('input', (event) => {
    settings.minFitness = parseFloat(event.target.value);
    document.getElementById('fitnessValue').textContent = settings.minFitness.toFixed(2);
    renderAll();
  });

  // Toggle trajectories
  document.getElementById('toggleTraj').addEventListener('change', (event) => {
    settings.showTrajectories = event.target.checked;
    renderAll();
  });

  // Toggle vector field
  document.getElementById('toggleField').addEventListener('change', (event) => {
    settings.showVectorField = event.target.checked;
    renderAll();
  });

  // Toggle labels
  document.getElementById('toggleLabels').addEventListener('change', (event) => {
    settings.showLabels = event.target.checked;
    renderAll();
  });

  // Color scheme
  document.getElementById('colorScheme').addEventListener('change', (event) => {
    settings.colorScheme = event.target.value;
    renderAll();
  });

  // Point size
  document.getElementById('pointSize').addEventListener('input', (event) => {
    settings.pointSize = parseInt(event.target.value);
    renderAll();
  });

  // Trajectory width
  document.getElementById('trajectoryWidth').addEventListener('input', (event) => {
    settings.trajectoryWidth = parseFloat(event.target.value);
    renderAll();
  });
}

function drawLegend(svg, labels) {
  // Remove legend if exists
  svg.selectAll('.legend').remove();

  const legendG = svg.append('g')
    .attr('class', 'legend')
    .attr('transform', 'translate(830, 60)'); // Ajuste conforme layout

  const legendItemSize = 22;
  labels.forEach((label, i) => {
    legendG.append('rect')
      .attr('x', 0)
      .attr('y', i * legendItemSize)
      .attr('width', 18)
      .attr('height', 18)
      .attr('fill', labelColor(label));
    legendG.append('text')
      .attr('x', 26)
      .attr('y', i * legendItemSize + 14)
      .attr('font-size', '14px')
      .attr('fill', '#333')
      .text(label);
  });
}

function renderAll() {
  const svg = d3.select('#chart');
  svg.selectAll('*').remove();

  const filteredData = filterData();

  // Renderizar camadas na ordem correta
  if (settings.showVectorField) {
    drawVectorField(svg, filteredData, settings);
  }
  if (settings.showTrajectories) {
    drawTrajectories(svg, filteredData, settings);
  }
  drawScatter(svg, filteredData, settings);

  // Adicionar legenda de labels
  const uniqueLabels = Array.from(new Set(filteredData.instances.map(d => d.label))).sort();
  drawLegend(svg, uniqueLabels);

  updateStatistics(filteredData);
}

async function main() {
  try {
    data = await loadData();
    setupEventListeners();
    renderAll();
  } catch (error) {
    console.error('Erro na inicialização:', error);
  }
}

// Inicializar quando a página carregar
document.addEventListener('DOMContentLoaded', main);
