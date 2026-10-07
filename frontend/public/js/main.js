import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const state = {
    oldPapers: [],
    newPapers: [],
    clusters: [],
    clusterNames: [],
    trends: [],
    pcaResults: { query_papers: [], all_papers: [] },
    visibleClusters: {},
};

const CLUSTER_COLORS = [
    0xe74c3c, 0x3498db, 0x2ecc71, 0xf39c12, 0x9b59b6,
    0x1abc9c, 0xe67e22, 0x34495e, 0x16a085, 0xc0392b,
];

function showSpinner(on) {
    document.getElementById('loading-spinner').style.display = on ? 'block' : 'none';
}

async function postJson(url, body) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
        const message = data.error || response.statusText || 'Request failed';
        const error = new Error(message);
        error.status = response.status;
        throw error;
    }
    return data;
}

function renderPapers(containerId, papers, caption) {
    const container = document.getElementById(containerId);
    if (!papers.length) {
        container.innerHTML = `<p>${caption}: none found.</p>`;
        return;
    }
    const rows = papers.map((paper) => {
        const authors = Array.isArray(paper.authors) ? paper.authors.join(', ') : '';
        const title = paper.title || 'Untitled';
        const abstract = paper.abstract || '';
        return `<tr title="${escapeAttr(abstract)}">
            <td title="${escapeAttr(title)}">${escapeHtml(title)}</td>
            <td>${escapeHtml(paper.pub_date || '')}</td>
            <td title="${escapeAttr(authors)}">${escapeHtml(authors)}</td>
        </tr>`;
    }).join('');
    container.innerHTML = `
        <table class="articles-table">
            <caption>${caption} (${papers.length})</caption>
            <thead><tr><th>Title</th><th>Year</th><th>Authors</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function escapeAttr(value) {
    return escapeHtml(value).replace(/\n/g, ' ');
}

function clusterColor(index) {
    return CLUSTER_COLORS[index % CLUSTER_COLORS.length];
}

function hexToCss(hex) {
    return `#${hex.toString(16).padStart(6, '0')}`;
}

function renderClusterNames() {
    const container = document.getElementById('clusterNames');
    if (!state.clusters.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = state.clusters.map((cluster, index) => {
        const name = state.clusterNames[index] || `Cluster ${index + 1}`;
        const paperCount = Array.isArray(cluster[2]) ? cluster[2].length : 0;
        const queryCount = Array.isArray(cluster[0]) ? cluster[0].length : 0;
        const checked = state.visibleClusters[index] !== false ? 'checked' : '';
        return `
            <div class="cluster-name" data-index="${index}" style="border-left: 6px solid ${hexToCss(clusterColor(index))}; padding-left: 8px; margin-bottom: 8px;">
                <label class="switch">
                    <input type="checkbox" class="cluster-toggle" data-index="${index}" ${checked}>
                    <span class="slider"></span>
                </label>
                <strong>${escapeHtml(String(name).replace(/^"|"$/g, ''))}</strong>
                <span> — ${queryCount} new / ${paperCount} related</span>
            </div>`;
    }).join('');

    container.querySelectorAll('.cluster-toggle').forEach((input) => {
        input.addEventListener('change', () => {
            state.visibleClusters[Number(input.dataset.index)] = input.checked;
            renderScene();
            renderTrends();
        });
    });
}

let renderer, scene, camera, controls, animationId;

function initThree() {
    const container = document.getElementById('threejs-container');
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 400;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeaeaea);
    camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    camera.position.set(9, 6.5, 9);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const light = new THREE.DirectionalLight(0xffffff, 0.6);
    light.position.set(5, 10, 7);
    scene.add(light);

    const axes = new THREE.AxesHelper(7);
    scene.add(axes);

    window.addEventListener('resize', onResize);
    // The container shrinks as the cluster legend fills up, which a window
    // resize listener never sees — without this the canvas stays too tall and
    // the plot gets clipped.
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(onResize).observe(container);
    }
    animate();
}

function onResize() {
    const container = document.getElementById('threejs-container');
    if (!renderer || !camera || !container) return;
    const width = container.clientWidth || 400;
    const height = container.clientHeight || 400;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function animate() {
    animationId = requestAnimationFrame(animate);
    if (controls) controls.update();
    if (renderer && scene && camera) renderer.render(scene, camera);
}

function coordsLookup() {
    const map = new Map();
    (state.pcaResults.all_papers || []).forEach((item) => map.set(item.title, item.coords));
    (state.pcaResults.query_papers || []).forEach((item) => {
        if (!map.has(item.title)) map.set(item.title, item.coords);
    });
    return map;
}

const VIEW_RADIUS = 6;

function renderScene() {
    if (!scene) initThree();
    const keep = new Set(['AmbientLight', 'DirectionalLight', 'AxesHelper']);
    [...scene.children].forEach((child) => {
        if (!keep.has(child.type)) scene.remove(child);
    });

    const lookup = coordsLookup();
    const groups = [];
    state.clusters.forEach((cluster, index) => {
        if (state.visibleClusters[index] === false) return;
        const queries = (Array.isArray(cluster[0]) ? cluster[0] : [])
            .map((title) => lookup.get(title))
            .filter(Boolean);
        const related = (Array.isArray(cluster[2]) ? cluster[2] : [])
            .map((paper) => lookup.get(paper[0]))
            .filter(Boolean);
        if (queries.length || related.length) {
            groups.push({ color: clusterColor(index), queries, related });
        }
    });

    const all = groups.flatMap((group) => group.queries.concat(group.related));
    if (!all.length) return;

    // PCA output has an arbitrary scale, so normalise it into the camera's frame.
    // Centre on the bounding box and divide by the largest deviation, so every
    // paper is guaranteed to land inside the view without distorting distances.
    const center = [0, 1, 2].map((axis) => {
        const values = all.map((point) => point[axis] || 0);
        return (Math.min(...values) + Math.max(...values)) / 2;
    });
    const extent = Math.max(
        ...all.map((point) => Math.max(...[0, 1, 2].map((axis) => Math.abs((point[axis] || 0) - center[axis])))),
        1e-6,
    );
    const scale = VIEW_RADIUS / extent;
    const place = (point) => new THREE.Vector3(
        ((point[0] || 0) - center[0]) * scale,
        ((point[1] || 0) - center[1]) * scale,
        ((point[2] || 0) - center[2]) * scale,
    );

    groups.forEach((group) => {
        const relatedPositions = group.related.map(place);
        const queryPositions = group.queries.map(place);

        // Faint hub lines from the cluster centre make group membership readable in 3D.
        const hub = new THREE.Vector3();
        queryPositions.concat(relatedPositions).forEach((position) => hub.add(position));
        hub.divideScalar(queryPositions.length + relatedPositions.length || 1);

        relatedPositions.forEach((position) => {
            const geometry = new THREE.BufferGeometry().setFromPoints([hub, position]);
            const material = new THREE.LineBasicMaterial({
                color: group.color,
                transparent: true,
                opacity: 0.22,
            });
            scene.add(new THREE.Line(geometry, material));
            addPoint(position, group.color, 0.13, 0.9);
        });

        queryPositions.forEach((position) => addPoint(position, group.color, 0.32, 1));
    });

    fitCamera(groups.flatMap((group) => group.queries.concat(group.related)).map(place));
}

// A perspective frustum narrows towards the camera, so a cloud that fits inside
// the coordinate box can still spill out of view. Pull back to the distance that
// contains its bounding sphere.
function fitCamera(positions) {
    if (!positions.length || !camera || !controls) return;
    const radius = Math.max(...positions.map((position) => position.length()), 1e-6);
    const halfFov = THREE.MathUtils.degToRad(camera.fov) / 2;
    const distance = (radius / Math.sin(halfFov)) * 1.12;
    const direction = new THREE.Vector3(1, 0.72, 1).normalize();
    camera.position.copy(direction.multiplyScalar(distance));
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
}

function addPoint(position, color, size, opacity) {
    const geometry = new THREE.SphereGeometry(size, 20, 20);
    const material = new THREE.MeshLambertMaterial({
        color,
        transparent: opacity < 1,
        opacity,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    scene.add(mesh);
}

function trendMatchesFilters(html) {
    const importance = document.getElementById('importanceDegreeFilter').value;
    const category = document.getElementById('categoryFilter').value;
    const text = html.replace(/<[^>]+>/g, ' ');
    if (importance !== 'All') {
        const re = new RegExp(`importance degree[^\\n]*${importance}`, 'i');
        if (!re.test(text)) return false;
    }
    if (category !== 'All') {
        if (!text.toLowerCase().includes(category.toLowerCase())) return false;
    }
    return true;
}

function renderTrends() {
    const container = document.getElementById('trendResults');
    if (!state.trends.length) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = state.trends.map((trend, index) => {
        if (state.visibleClusters[index] === false) return '';
        const html = typeof trend === 'string' ? trend : JSON.stringify(trend);
        if (!trendMatchesFilters(html)) return '';
        const name = (state.clusterNames[index] || `Cluster ${index + 1}`).replace(/^"|"$/g, '');
        return `
            <div class="trend-box">
                <span class="trend-dot" style="background:${hexToCss(clusterColor(index))}"></span>
                <div>
                    <strong>${escapeHtml(name)}</strong>
                    <div>${html}</div>
                </div>
            </div>`;
    }).join('');
}

document.getElementById('fetchForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    showSpinner(true);
    try {
        const payload = {
            query: document.getElementById('query').value,
            presentdate: document.getElementById('presentdate').value,
            max_records: document.getElementById('max_records').value,
        };
        const grouped = await postJson('/fetch', payload);
        state.oldPapers = grouped[0] || [];
        state.newPapers = grouped[1] || [];
        state.clusters = [];
        state.clusterNames = [];
        state.trends = [];
        renderPapers('oldArticlesContainer', state.oldPapers, 'Older papers');
        renderPapers('newArticlesContainer', state.newPapers, `Papers in ${payload.presentdate}`);
        renderClusterNames();
        renderTrends();
        if (scene) renderScene();
    } catch (error) {
        alert(`Could not fetch articles: ${error.message}`);
    } finally {
        showSpinner(false);
    }
});

document.getElementById('clusterAndNameButton').addEventListener('click', async () => {
    if (!state.newPapers.length || !state.oldPapers.length) {
        alert('Fetch articles first. You need both older papers and papers in the present year.');
        return;
    }
    showSpinner(true);
    try {
        const simThreshold = Number(document.getElementById('gradientSlider').value);
        const clustered = await postJson('/cluster', {
            new_papers: state.newPapers,
            old_papers: state.oldPapers,
            sim_threshold: simThreshold,
        });
        state.clusters = clustered.results || [];
        state.pcaResults = clustered.pca_results || { query_papers: [], all_papers: [] };
        state.visibleClusters = {};
        state.clusters.forEach((_, index) => { state.visibleClusters[index] = true; });

        try {
            state.clusterNames = await postJson('/generate_cluster_names', { clusters: state.clusters });
        } catch (error) {
            state.clusterNames = state.clusters.map((_, index) => `Cluster ${index + 1}`);
            if (error.status === 503) {
                console.warn(error.message);
            } else {
                console.warn('Cluster naming failed:', error.message);
            }
        }

        renderClusterNames();
        renderScene();
    } catch (error) {
        alert(`Analyze failed: ${error.message}`);
    } finally {
        showSpinner(false);
    }
});

document.getElementById('generateTrendsButton').addEventListener('click', async () => {
    if (!state.clusters.length) {
        alert('Run Analyze first so clusters exist.');
        return;
    }
    showSpinner(true);
    try {
        state.trends = await postJson('/identify_trends', { clusters: state.clusters });
        renderTrends();
    } catch (error) {
        alert(`Could not generate trends: ${error.message}`);
    } finally {
        showSpinner(false);
    }
});

document.getElementById('gradientSlider').addEventListener('input', (event) => {
    document.getElementById('sliderValue').textContent = event.target.value;
});

document.getElementById('importanceDegreeFilter').addEventListener('change', renderTrends);
document.getElementById('categoryFilter').addEventListener('change', renderTrends);

initThree();
