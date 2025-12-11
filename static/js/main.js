// Global state
let uploadedFilename = null;
let analysisResult = null;
let allBeats = [];
let currentPage = 1;
const beatsPerPage = 50;
let modelMetrics = null;

// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const uploadStatus = document.getElementById('upload-status');
const processBtn = document.getElementById('process-btn');
const configModal = document.getElementById('config-modal');
const closeModal = document.querySelector('.close-modal');
const configSubmit = document.getElementById('config-submit');
const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results-section');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadModelMetrics(); // Load metrics on page load
    console.log('ECG Analyzer loaded');
});

// Setup Event Listeners
function setupEventListeners() {
    // File input
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Process button
    processBtn.addEventListener('click', processECG);
    
    // Modal
    closeModal.addEventListener('click', () => configModal.style.display = 'none');
    configSubmit.addEventListener('click', submitConfiguration);
    
    // Table controls
    const searchInput = document.getElementById('table-search');
    const classFilter = document.getElementById('class-filter');
    
    if (searchInput) searchInput.addEventListener('input', filterTable);
    if (classFilter) classFilter.addEventListener('change', filterTable);
    
    // Download buttons
    const downloadJson = document.getElementById('download-json');
    const downloadPdf = document.getElementById('download-pdf');
    
    if (downloadJson) downloadJson.addEventListener('click', downloadJSON);
    if (downloadPdf) downloadPdf.addEventListener('click', generatePDF);
}

// File Selection Handlers
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        uploadFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        uploadFile(file);
    }
}

// Upload File
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        uploadStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading file...';
        uploadStatus.className = 'upload-status';
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log('Upload response:', data);
        
        if (data.success) {
            uploadedFilename = data.filename;
            uploadStatus.innerHTML = `<i class="fas fa-check-circle"></i> File uploaded: ${data.filename}`;
            uploadStatus.className = 'upload-status success';
            
            if (!data.has_header) {
                showConfigModal();
            } else {
                processBtn.style.display = 'block';
            }
        } else {
            uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${data.error}`;
            uploadStatus.className = 'upload-status error';
        }
    } catch (error) {
        console.error('Upload error:', error);
        uploadStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> Upload failed: ${error.message}`;
        uploadStatus.className = 'upload-status error';
    }
}

// Configuration Modal
function showConfigModal() {
    configModal.style.display = 'flex';
}

async function submitConfiguration() {
    const fs = parseInt(document.getElementById('fs-input').value);
    const channels = parseInt(document.getElementById('channels-input').value);
    const dtype = document.getElementById('dtype-input').value;
    
    try {
        const response = await fetch('/configure', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: uploadedFilename,
                fs: fs,
                n_channels: channels,
                dtype: dtype
            })
        });
        
        const data = await response.json();
        console.log('Configuration response:', data);
        
        if (data.success) {
            configModal.style.display = 'none';
            processBtn.style.display = 'block';
        } else {
            alert('Configuration failed: ' + data.error);
        }
    } catch (error) {
        console.error('Configuration error:', error);
        alert('Configuration failed: ' + error.message);
    }
}

// Process ECG
async function processECG() {
    if (!uploadedFilename) {
        alert('Please upload a file first');
        return;
    }
    
    console.log('Starting ECG processing...');
    
    // Hide upload section, show loading
    document.getElementById('upload-section').style.display = 'none';
    loadingSection.style.display = 'block';
    resultsSection.style.display = 'none';
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: uploadedFilename })
        });
        
        console.log('Response status:', response.status);
        
        const data = await response.json();
        console.log('Processing response:', data);
        
        if (data.success) {
            analysisResult = data.result;
            allBeats = data.result.beats;
            
            console.log('Analysis successful!');
            console.log('Number of beats:', allBeats.length);
            console.log('Plots received:', Object.keys(data.plots));
            
            // Hide loading, show results
            loadingSection.style.display = 'none';
            resultsSection.style.display = 'block';
            
            // Display all results
            console.log('Displaying interpretation...');
            displayInterpretation(data.interpretation);
            
            console.log('Displaying summary metrics...');
            displaySummaryMetrics(data.result.summary);
            
            console.log('Displaying plots...');
            displayPlots(data.plots);
            
            console.log('Displaying beat table...');
            displayBeatTable(allBeats);
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
            
            console.log('All displays complete!');
        } else {
            console.error('Analysis failed:', data.error);
            loadingSection.style.display = 'none';
            alert('Analysis failed: ' + data.error);
            document.getElementById('upload-section').style.display = 'block';
        }
    } catch (error) {
        console.error('Processing error:', error);
        loadingSection.style.display = 'none';
        alert('Analysis failed: ' + error.message);
        document.getElementById('upload-section').style.display = 'block';
    }
}

// Display Clinical Interpretation
function displayInterpretation(interpretation) {
    const container = document.getElementById('interpretation-content');
    
    if (!container) {
        console.error('interpretation-content element not found');
        return;
    }
    
    const severityClass = `severity-${interpretation.severity}`;
    const severityText = interpretation.severity.charAt(0).toUpperCase() + interpretation.severity.slice(1);
    
    let html = `
        <div class="interpretation-severity ${severityClass}">
            <i class="fas fa-info-circle"></i> Severity: ${severityText}
        </div>
        
        <div class="findings-list">
            <h3><i class="fas fa-clipboard-list"></i> Findings</h3>
            <ul>
                ${interpretation.findings.map(f => `<li><i class="fas fa-check"></i> ${f}</li>`).join('')}
            </ul>
        </div>
        
        <div class="recommendations-list">
            <h3><i class="fas fa-lightbulb"></i> Recommendations</h3>
            <ul>
                ${interpretation.recommendations.map(r => `<li><i class="fas fa-arrow-right"></i> ${r}</li>`).join('')}
            </ul>
        </div>
    `;
    
    container.innerHTML = html;
    console.log('Interpretation displayed');
}

// Display Summary Metrics
function displaySummaryMetrics(summary) {
    const container = document.getElementById('summary-cards');
    
    if (!container) {
        console.error('summary-cards element not found');
        return;
    }
    
    const metrics = [
        {
            label: 'Total Beats',
            value: summary.total_beats,
            unit: 'beats',
            icon: 'fa-heartbeat'
        },
        {
            label: 'Mean Heart Rate',
            value: summary.mean_hr.toFixed(1),
            unit: 'bpm',
            icon: 'fa-heart',
            alert: summary.mean_hr < 60 || summary.mean_hr > 100
        },
        {
            label: 'SDNN',
            value: summary.sdnn.toFixed(1),
            unit: 'ms',
            icon: 'fa-chart-line',
            alert: summary.sdnn < 50
        },
        {
            label: 'RMSSD',
            value: summary.rmssd.toFixed(1),
            unit: 'ms',
            icon: 'fa-wave-square'
        },
        {
            label: 'pNN50',
            value: summary.pnn50.toFixed(1),
            unit: '%',
            icon: 'fa-percentage'
        },
        {
            label: 'PVC Burden',
            value: summary.pvc_burden.toFixed(1),
            unit: '%',
            icon: 'fa-exclamation-triangle',
            alert: summary.pvc_burden > 5
        },
        {
            label: 'SVE Burden',
            value: summary.sve_burden.toFixed(1),
            unit: '%',
            icon: 'fa-exclamation-circle',
            alert: summary.sve_burden > 5
        },
        {
            label: 'Signal Quality',
            value: summary.signal_quality_label,
            unit: `(${summary.signal_quality.toFixed(0)}/100)`,
            icon: 'fa-signal',
            alert: summary.signal_quality < 60
        }
    ];
    
    let html = metrics.map(m => `
        <div class="metric-card ${m.alert ? 'alert' : ''}">
            <h4><i class="fas ${m.icon}"></i> ${m.label}</h4>
            <div class="metric-value">${m.value}</div>
            <div class="metric-unit">${m.unit}</div>
        </div>
    `).join('');
    
    container.innerHTML = html;
    console.log('Summary metrics displayed');
}

// Display Plots
function displayPlots(plots) {
    console.log('displayPlots called with:', plots);
    
    // Main ECG plot
    const ecgPlotDiv = document.getElementById('ecg-plot');
    if (ecgPlotDiv) {
        console.log('ECG plot div found');
        console.log('ECG waveform HTML length:', plots.ecg_waveform ? plots.ecg_waveform.length : 0);
        ecgPlotDiv.innerHTML = plots.ecg_waveform || '<p style="color: red;">Error: ECG plot not generated</p>';
    } else {
        console.error('ecg-plot div not found!');
    }
    
    // HR Trend
    const hrTrendDiv = document.getElementById('hr-trend-plot');
    if (hrTrendDiv) {
        console.log('HR trend div found');
        hrTrendDiv.innerHTML = plots.hr_trend || '<p style="color: red;">Error: HR trend plot not generated</p>';
    } else {
        console.error('hr-trend-plot div not found!');
    }
    
    // RR Histogram
    const rrHistDiv = document.getElementById('rr-histogram');
    if (rrHistDiv) {
        console.log('RR histogram div found');
        rrHistDiv.innerHTML = plots.rr_histogram || '<p style="color: red;">Error: RR histogram not generated</p>';
    } else {
        console.error('rr-histogram div not found!');
    }
    
    // Arrhythmia Pie
    const pieDiv = document.getElementById('arrhythmia-pie');
    if (pieDiv) {
        console.log('Pie chart div found');
        pieDiv.innerHTML = plots.arrhythmia_pie || '<p style="color: red;">Error: Pie chart not generated</p>';
    } else {
        console.error('arrhythmia-pie div not found!');
    }
    
    // Arrhythmia Timeline
    const timelineDiv = document.getElementById('arrhythmia-timeline');
    if (timelineDiv) {
        console.log('Timeline div found');
        timelineDiv.innerHTML = plots.arrhythmia_timeline || '<p style="color: red;">Error: Timeline not generated</p>';
    } else {
        console.error('arrhythmia-timeline div not found!');
    }
    
    console.log('All plots inserted into DOM');
}

// Display Beat Table
function displayBeatTable(beats, page = 1) {
    const tbody = document.querySelector('#beat-table tbody');
    
    if (!tbody) {
        console.error('beat-table tbody not found');
        return;
    }
    
    tbody.innerHTML = '';
    
    // Pagination
    const startIdx = (page - 1) * beatsPerPage;
    const endIdx = Math.min(startIdx + beatsPerPage, beats.length);
    const pageBeats = beats.slice(startIdx, endIdx);
    
    console.log(`Displaying beats ${startIdx} to ${endIdx} of ${beats.length}`);
    
    pageBeats.forEach(beat => {
        const row = tbody.insertRow();
        row.innerHTML = `
            <td>${beat.index}</td>
            <td>${beat.time.toFixed(2)}</td>
            <td><span class="class-badge class-${beat.class}">${beat.class_name}</span></td>
            <td>${(beat.confidence * 100).toFixed(1)}%</td>
            <td>${(beat.rr_interval * 1000).toFixed(0)}</td>
            <td>${beat.heart_rate.toFixed(0)}</td>
        `;
    });
    
    // Update pagination
    updatePagination(beats.length, page);
    console.log('Beat table displayed');
}

// Update Pagination
function updatePagination(totalBeats, currentPageNum) {
    const totalPages = Math.ceil(totalBeats / beatsPerPage);
    const paginationContainer = document.getElementById('table-pagination');
    
    if (!paginationContainer) {
        console.error('table-pagination not found');
        return;
    }
    
    let html = '';
    
    // Previous button
    if (currentPageNum > 1) {
        html += `<button onclick="changePage(${currentPageNum - 1})"><i class="fas fa-chevron-left"></i></button>`;
    }
    
    // Page numbers
    const maxPages = Math.min(totalPages, 10);
    for (let i = 1; i <= maxPages; i++) {
        const active = i === currentPageNum ? 'active' : '';
        html += `<button class="${active}" onclick="changePage(${i})">${i}</button>`;
    }
    
    if (totalPages > 10) {
        html += `<span>...</span>`;
        html += `<button onclick="changePage(${totalPages})">${totalPages}</button>`;
    }
    
    // Next button
    if (currentPageNum < totalPages) {
        html += `<button onclick="changePage(${currentPageNum + 1})"><i class="fas fa-chevron-right"></i></button>`;
    }
    
    paginationContainer.innerHTML = html;
}

// Change Page
function changePage(page) {
    currentPage = page;
    const filteredBeats = filterBeats(allBeats);
    displayBeatTable(filteredBeats, page);
}

// Filter Table
function filterTable() {
    const filteredBeats = filterBeats(allBeats);
    currentPage = 1;
    displayBeatTable(filteredBeats, 1);
}

// Filter Beats
function filterBeats(beats) {
    const searchInput = document.getElementById('table-search');
    const classFilter = document.getElementById('class-filter');
    
    const searchTerm = searchInput ? searchInput.value.toLowerCase() : '';
    const classFilterValue = classFilter ? classFilter.value : 'all';
    
    let filtered = beats;
    
    // Filter by class
    if (classFilterValue !== 'all') {
        filtered = filtered.filter(b => b.class === classFilterValue);
    }
    
    // Filter by search term
    if (searchTerm) {
        filtered = filtered.filter(b => 
            b.index.toString().includes(searchTerm) ||
            b.class_name.toLowerCase().includes(searchTerm) ||
            b.time.toFixed(2).includes(searchTerm)
        );
    }
    
    return filtered;
}

// Download JSON Report
function downloadJSON() {
    if (!analysisResult) {
        alert('No analysis results available');
        return;
    }
    
    const dataStr = JSON.stringify(analysisResult, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ecg_report_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// Generate PDF Report
function generatePDF() {
    alert('PDF generation: Use your browser print function (Ctrl+P or Cmd+P) and select "Save as PDF"');
    window.print();
}

// Load Model Metrics from Backend
async function loadModelMetrics() {
    try {
        // Show loading state
        const accuracyEl = document.getElementById('accuracy');
        const precisionEl = document.getElementById('precision');
        const recallEl = document.getElementById('recall');
        const f1scoreEl = document.getElementById('f1score');
        
        if (accuracyEl) accuracyEl.textContent = 'Loading...';
        if (precisionEl) precisionEl.textContent = 'Loading...';
        if (recallEl) recallEl.textContent = 'Loading...';
        if (f1scoreEl) f1scoreEl.textContent = 'Loading...';
        
        const response = await fetch('/model_metrics');
        const data = await response.json();
        
        if (data.success && data.metrics) {
            modelMetrics = data.metrics;
            
            // Update UI with real metrics
            if (accuracyEl) {
                accuracyEl.textContent = data.metrics.accuracy || 'N/A';
            }
            if (precisionEl) {
                precisionEl.textContent = data.metrics.precision || 'N/A';
            }
            if (recallEl) {
                recallEl.textContent = data.metrics.recall || 'N/A';
            }
            if (f1scoreEl) {
                f1scoreEl.textContent = data.metrics.f1_score || 'N/A';
            }
            
            console.log('✓ Model metrics loaded:', data.metrics);
            console.log('  Source:', data.metrics.source || 'unknown');
            console.log('  Test Samples:', data.metrics.test_samples || 0);
        } else {
            console.warn('⚠ Failed to load model metrics, using defaults');
            // Keep default values displayed
        }
    } catch (error) {
        console.error('Error loading model metrics:', error);
        // Keep default values on error
    }
}