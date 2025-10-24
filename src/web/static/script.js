// YOLO RKNN Web Interface JavaScript

class YOLOWebInterface {
    constructor() {
        this.socket = null;
        this.autoScroll = true;
        this.connectionStatus = 'disconnected';
        this.processing_active = false;
        this.charts = {};
        this.dataHistory = {
            cpu_cores: [],
            npu_cores: [], 
            gpu: [],
            memory: [],
            temperature: []
        };
        this.coreCount = {
            cpu: 8, // Orange Pi 5 Max has 8 cores
            npu: 3  // RK3588 NPU has 3 cores
        };
        this.maxDataPoints = 60; // 1 minute interval
        this.init();
    }

    init() {
        this.initSocket();
        this.initUI();
        this.loadConfig();
        this.refreshStatus();
        
        // Set up periodic status updates
        setInterval(() => this.refreshStatus(), 500);
        
        // Set up periodic system monitoring updates
        setInterval(() => this.updateSystemMonitor(), 500);
        
        // Initialize monitoring charts with delay to ensure DOM is ready
        setTimeout(() => {
            this.initCharts();
        }, 100);
        
        console.log('Web Interface initialized');
    }

    initSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateConnectionStatus('connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateConnectionStatus('disconnected');
        });
        
        this.socket.on('console_messages', (messages) => {
            this.handleConsoleMessages(messages);
        });
        
        this.socket.on('error', (error) => {
            console.error('Socket error:', error);
            this.addConsoleMessage('ERROR', 'Socket connection error: ' + error);
        });
        
        this.socket.on('config_updated', (data) => {
            console.log('Configuration updated:', data);
            this.addConsoleMessage('INFO', data.message);
            // Reload configuration display
            this.loadConfig();
            this.refreshStatus();
        });
    }

    initUI() {
        // Configuration form handler
        document.getElementById('config-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveConfiguration();
        });

        // Console controls
        document.getElementById('clear-console').addEventListener('click', () => {
            this.clearConsole();
        });

        document.getElementById('toggle-autoscroll').addEventListener('click', () => {
            this.toggleAutoScroll();
        });

        // Video error handling
        const videoStream = document.getElementById('video-stream');
        videoStream.addEventListener('error', () => {
            console.log('Video stream error, retrying...');
            setTimeout(() => {
                videoStream.src = '/video_feed?' + new Date().getTime();
            }, 1000);
        });
    }

    updateConnectionStatus(status) {
        this.connectionStatus = status;
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('processing-status');
        
        statusDot.className = 'status-dot';
        
        switch (status) {
            case 'connected':
                statusDot.classList.add('online');
                statusText.textContent = 'Connected';
                break;
            case 'processing':
                statusDot.classList.add('processing');
                statusText.textContent = 'Processing';
                break;
            case 'disconnected':
            default:
                statusDot.classList.add('offline');
                statusText.textContent = 'Disconnected';
                break;
        }
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            
            // Update form fields with validation
            const benchmarkMode = document.getElementById('benchmark-mode');
            const inferenceDeviceSelect = document.getElementById('inference-device-select');
            const maxCameras = document.getElementById('max-cameras');
            
            if (benchmarkMode) {
                benchmarkMode.value = config.benchmark_mode.toString();
            }
            if (inferenceDeviceSelect) {
                inferenceDeviceSelect.value = config.inference_device;
            }
            if (maxCameras && config.camera_config) {
                maxCameras.value = config.camera_config.max_cameras;
            }
            
            const infoMode = document.getElementById('info-mode');
            const infoDevice = document.getElementById('info-device');
            
            if (infoMode) {
                infoMode.textContent = config.benchmark_mode ? 'Benchmark' : 'Camera';
            }
            if (infoDevice) {
                infoDevice.textContent = config.inference_device;
            }
                
            console.log('Configuration loaded:', config);
            
        } catch (error) {
            console.error('Error loading configuration:', error);
            this.addConsoleMessage('ERROR', 'Failed to load configuration: ' + error.message);
        }
    }

    async saveConfiguration() {
        if (this.processing_active) {
            this.addConsoleMessage('WARNING', 'Cannot save configuration while processing is active');
            return;
        }
        
        try {
            const formData = new FormData(document.getElementById('config-form'));
            const config = {
                benchmark_mode: formData.get('benchmark_mode') === 'true',
                inference_device: formData.get('inference_device'),
                max_cameras: parseInt(formData.get('max_cameras'))
            };
            
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.addConsoleMessage('INFO', 'Configuration saved successfully');
                this.loadConfig(); // Reload to update display
            } else {
                this.addConsoleMessage('ERROR', 'Failed to save configuration: ' + result.message);
            }
            
        } catch (error) {
            console.error('Error saving configuration:', error);
            this.addConsoleMessage('ERROR', 'Failed to save configuration: ' + error.message);
        }
    }

    async refreshStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update info panel
            document.getElementById('info-mode').textContent = 
                status.current_mode.charAt(0).toUpperCase() + status.current_mode.slice(1);
            document.getElementById('info-device').textContent = status.inference_device;
            document.getElementById('info-processing').textContent = 
                status.processing_active ? 'Active' : 'Stopped';
            document.getElementById('info-frame').textContent = 
                status.frame_available ? 'Available' : 'Not Available';
            
            // Update button states
            const startBtn = document.getElementById('start-btn');
            const stopBtn = document.getElementById('stop-btn');
            const saveConfigBtn = document.getElementById('save-config-btn');
            
            // Update internal processing state
            this.processing_active = status.processing_active;
            
            if (status.processing_active) {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                if (saveConfigBtn) saveConfigBtn.disabled = true;
                this.updateConnectionStatus('processing');
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                if (saveConfigBtn) saveConfigBtn.disabled = false;
                if (this.connectionStatus !== 'disconnected') {
                    this.updateConnectionStatus('connected');
                }
            }
            
        } catch (error) {
            console.error('Error refreshing status:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    async startProcessing() {
        try {
            this.addConsoleMessage('INFO', 'Starting processing...');
            
            const response = await fetch('/api/start_processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.addConsoleMessage('INFO', 'Processing started successfully');
                this.refreshStatus();
            } else {
                this.addConsoleMessage('ERROR', 'Failed to start processing: ' + result.message);
            }
            
        } catch (error) {
            console.error('Error starting processing:', error);
            this.addConsoleMessage('ERROR', 'Failed to start processing: ' + error.message);
        }
    }

    async stopProcessing() {
        try {
            this.addConsoleMessage('INFO', 'Stopping processing...');
            
            const response = await fetch('/api/stop_processing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.addConsoleMessage('INFO', 'Processing stopped successfully');
                this.refreshStatus();
            } else {
                this.addConsoleMessage('ERROR', 'Failed to stop processing: ' + result.message);
            }
            
        } catch (error) {
            console.error('Error stopping processing:', error);
            this.addConsoleMessage('ERROR', 'Failed to stop processing: ' + error.message);
        }
    }

    handleConsoleMessages(messages) {
        const consoleOutput = document.getElementById('console-output');
        
        messages.forEach(message => {
            const messageEl = document.createElement('div');
            messageEl.className = 'console-message';
            
            const timestamp = document.createElement('span');
            timestamp.className = 'console-timestamp';
            timestamp.textContent = message.timestamp;
            
            const level = document.createElement('span');
            level.className = `console-level ${message.level}`;
            level.textContent = message.level;
            
            const text = document.createElement('span');
            text.className = 'console-text';
            text.textContent = message.message;
            
            messageEl.appendChild(timestamp);
            messageEl.appendChild(level);
            messageEl.appendChild(text);
            
            consoleOutput.appendChild(messageEl);
        });
        
        // Auto-scroll to bottom if enabled
        if (this.autoScroll) {
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        
        // Limit console messages (keep last 500)
        const messages_elements = consoleOutput.querySelectorAll('.console-message');
        if (messages_elements.length > 500) {
            for (let i = 0; i < messages_elements.length - 500; i++) {
                messages_elements[i].remove();
            }
        }
    }

    addConsoleMessage(level, message) {
        const timestamp = new Date().toLocaleTimeString();
        this.handleConsoleMessages([{
            timestamp: timestamp,
            level: level,
            message: message
        }]);
    }

    clearConsole() {
        const consoleOutput = document.getElementById('console-output');
        consoleOutput.innerHTML = '';
        this.addConsoleMessage('INFO', 'Console cleared');
    }

    toggleAutoScroll() {
        this.autoScroll = !this.autoScroll;
        const btn = document.getElementById('toggle-autoscroll');
        
        if (this.autoScroll) {
            btn.classList.add('active');
            btn.textContent = '📜 Auto-scroll';
            
            // Scroll to bottom when enabling auto-scroll
            const consoleOutput = document.getElementById('console-output');
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        } else {
            btn.classList.remove('active');
            btn.textContent = '📜 Manual scroll';
        }
    }

    // Request console updates from server
    requestConsoleUpdate() {
        if (this.socket && this.socket.connected) {
            this.socket.emit('request_console_update');
        }
    }
    
    initCharts() {
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        min: 0,
                        max: 100,
                        display: true,
                        position: 'left',
                        grid: {
                            display: true,
                            color: 'rgba(128, 128, 128, 0.2)',
                            lineWidth: 0.5
                        },
                        ticks: {
                            stepSize: 10,
                            callback: function(value, index, values) {
                                return value % 10 === 0 ? value + '%' : '';
                            },
                            font: {
                                size: 9
                            },
                            color: 'rgba(128, 128, 128, 0.8)',
                            maxTicksLimit: 11,
                            padding: 2
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                },
                elements: {
                    point: {
                        radius: 0
                    },
                    line: {
                        borderWidth: 2,
                        tension: 0.4
                    }
                }
            }
        };

        // CPU Chart (multi-core)
        const cpuElement = document.getElementById('cpu-chart');
        if (cpuElement) {
            console.log('Initializing CPU chart with', this.coreCount.cpu, 'cores');
            const cpuColors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c', '#34495e'];
            const cpuDatasets = [];
            for (let i = 0; i < this.coreCount.cpu; i++) {
                cpuDatasets.push({
                    label: `CPU Core ${i}`,
                    data: new Array(this.maxDataPoints).fill(0),
                    borderColor: cpuColors[i % cpuColors.length],
                    backgroundColor: cpuColors[i % cpuColors.length] + '20',
                    fill: false,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                });
            }
            
            this.charts.cpu = new Chart(cpuElement.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array(this.maxDataPoints).fill(''),
                    datasets: cpuDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            min: 0,
                            max: 100,
                            display: true,
                            position: 'left',
                            grid: {
                                display: true,
                                color: 'rgba(128, 128, 128, 0.2)',
                                lineWidth: 0.5
                            },
                            ticks: {
                                stepSize: 10,
                                callback: function(value, index, values) {
                                    return value % 10 === 0 ? value + '%' : '';
                                },
                                font: {
                                    size: 9
                                },
                                color: 'rgba(128, 128, 128, 0.8)',
                                maxTicksLimit: 11,
                                padding: 2
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        },
                        line: {
                            borderWidth: 2,
                            tension: 0.4
                        }
                    }
                }
            });
            console.log('CPU chart initialized successfully with datasets:', cpuDatasets.length);
            console.log('CPU chart data structure:', this.charts.cpu.data.datasets.map(d => ({label: d.label, dataLength: d.data.length})));
            
            this.initChartLegend('cpu', this.coreCount.cpu, cpuColors);
        } else {
            console.error('CPU chart element not found');
        }

        // NPU Chart (multi-core)  
        const npuElement = document.getElementById('npu-chart');
        if (npuElement) {
            console.log('Initializing NPU chart with', this.coreCount.npu, 'cores');
            const npuColors = ['#3498db', '#2ecc71', '#e67e22'];
            const npuDatasets = [];
            for (let i = 0; i < this.coreCount.npu; i++) {
                npuDatasets.push({
                    label: `NPU Core ${i}`,
                    data: new Array(this.maxDataPoints).fill(0),
                    borderColor: npuColors[i],
                    backgroundColor: npuColors[i] + '20',
                    fill: false,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0
                });
            }
            
            this.charts.npu = new Chart(npuElement.getContext('2d'), {
                type: 'line',
                data: {
                    labels: Array(this.maxDataPoints).fill(''),
                    datasets: npuDatasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            min: 0,
                            max: 100,
                            display: true,
                            position: 'left',
                            grid: {
                                display: true,
                                color: 'rgba(128, 128, 128, 0.2)',
                                lineWidth: 0.5
                            },
                            ticks: {
                                stepSize: 10,
                                callback: function(value, index, values) {
                                    return value % 10 === 0 ? value + '%' : '';
                                },
                                font: {
                                    size: 9
                                },
                                color: 'rgba(128, 128, 128, 0.8)',
                                maxTicksLimit: 11,
                                padding: 2
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    elements: {
                        point: {
                            radius: 0
                        },
                        line: {
                            borderWidth: 2,
                            tension: 0.4
                        }
                    }
                }
            });
            console.log('NPU chart initialized successfully with datasets:', npuDatasets.length);
            console.log('NPU chart data structure:', this.charts.npu.data.datasets.map(d => ({label: d.label, dataLength: d.data.length})));
            
            this.initChartLegend('npu', this.coreCount.npu, npuColors);
        } else {
            console.error('NPU chart element not found');
        }

        // GPU Chart
        this.charts.gpu = new Chart(document.getElementById('gpu-chart').getContext('2d'), {
            ...chartConfig,
            data: {
                labels: Array(this.maxDataPoints).fill(''),
                datasets: [{
                    data: Array(this.maxDataPoints).fill(0),
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    fill: true
                }]
            }
        });

        // Memory Chart
        this.charts.memory = new Chart(document.getElementById('memory-chart').getContext('2d'), {
            ...chartConfig,
            data: {
                labels: Array(this.maxDataPoints).fill(''),
                datasets: [{
                    data: Array(this.maxDataPoints).fill(0),
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: true
                }]
            }
        });

        // Temperature Chart (different scale)
        this.charts.temperature = new Chart(document.getElementById('temp-chart').getContext('2d'), {
            ...chartConfig,
            data: {
                labels: Array(this.maxDataPoints).fill(''),
                datasets: [{
                    data: Array(this.maxDataPoints).fill(0),
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    fill: true
                }]
            },
            options: {
                ...chartConfig.options,
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        min: 20,
                        max: 90,
                        display: true,
                        position: 'left',
                        grid: {
                            display: true,
                            color: 'rgba(128, 128, 128, 0.2)',
                            lineWidth: 0.5
                        },
                        ticks: {
                            stepSize: 10,
                            callback: function(value, index, values) {
                                return value % 10 === 0 ? value + '°C' : '';
                            },
                            font: {
                                size: 9
                            },
                            color: 'rgba(128, 128, 128, 0.8)',
                            maxTicksLimit: 8,
                            padding: 2
                        }
                    }
                }
            }
        });
    }
    
    async updateSystemMonitor() {
        try {
            const response = await fetch('/api/system_monitor');
            const data = await response.json();
            
            if (response.ok) {
                this.updateMonitoringDisplay(data);
            } else {
                console.error('System monitor error:', data.error);
            }
        } catch (error) {
            console.error('Failed to fetch system monitor data:', error);
        }
    }
    
    updateMonitoringDisplay(data) {
        console.log('Received monitoring data:', data);
        
        // Update CPU
        if (data.cpu && !data.cpu.error) {
            const cpuLoad = data.cpu.average_load.toFixed(1);
            const cpuFreq = Array.isArray(data.cpu.frequencies) ? 
                Math.max(...data.cpu.frequencies) : 
                Math.max(...Object.values(data.cpu.frequencies));
            this.updateValueWithColor('cpu-load', `${cpuLoad}%`, data.cpu.average_load);
            document.getElementById('cpu-freq').textContent = `${cpuFreq} MHz`;
            this.updateCoresDisplay('cpu', data.cpu.loads);
            this.updateMultiCoreChart('cpu', data.cpu.loads);
        } else {
            document.getElementById('cpu-load').textContent = 'Error';
            document.getElementById('cpu-freq').textContent = '--';
        }
        
        // Update NPU
        if (data.npu && !data.npu.error) {
            const npuLoad = data.npu.average_load.toFixed(1);
            const npuFreq = data.npu.frequency;
            this.updateValueWithColor('npu-load', `${npuLoad}%`, data.npu.average_load);
            document.getElementById('npu-freq').textContent = `${npuFreq} MHz`;
            this.updateCoresDisplay('npu', data.npu.loads);
            this.updateMultiCoreChart('npu', data.npu.loads);
        } else {
            document.getElementById('npu-load').textContent = 'N/A';
            document.getElementById('npu-freq').textContent = '--';
        }
        
        // Update GPU
        if (data.gpu && !data.gpu.error && data.gpu.available) {
            const gpuLoad = data.gpu.load.toFixed(1);
            const gpuFreq = data.gpu.frequency;
            this.updateValueWithColor('gpu-load', `${gpuLoad}%`, data.gpu.load);
            document.getElementById('gpu-freq').textContent = `${gpuFreq} MHz`;
            this.updateChart('gpu', data.gpu.load);
        } else {
            document.getElementById('gpu-load').textContent = 'N/A';
            document.getElementById('gpu-freq').textContent = '--';
        }
        
        // Update Memory
        if (data.memory && !data.memory.error) {
            const memUsed = data.memory.used_gb.toFixed(1);
            const memTotal = data.memory.total_gb.toFixed(1);
            const memPercent = data.memory.percentage;
            this.updateValueWithColor('memory-used', `${memUsed} GB`, memPercent);
            document.getElementById('memory-total').textContent = `${memTotal} GB`;
            this.updateChart('memory', memPercent);
        } else {
            document.getElementById('memory-used').textContent = 'Error';
            document.getElementById('memory-total').textContent = '--';
        }
        
        // Update Temperature
        if (data.temperature && !data.temperature.error) {
            const tempAvg = data.temperature.average.toFixed(1);
            const tempMax = data.temperature.max.toFixed(1);
            this.updateTemperatureValue('temp-avg', `${tempAvg}°C`, data.temperature.average);
            this.updateTemperatureValue('temp-max', `${tempMax}°C`, data.temperature.max);
            this.updateChart('temperature', data.temperature.average);
        } else {
            document.getElementById('temp-avg').textContent = 'N/A';
            document.getElementById('temp-max').textContent = '--';
        }
    }
    
    updateChart(chartName, value) {
        if (this.charts[chartName]) {
            // Add new data point
            this.dataHistory[chartName].push(value);
            
            // Keep only last maxDataPoints
            if (this.dataHistory[chartName].length > this.maxDataPoints) {
                this.dataHistory[chartName].shift();
            }
            
            // Update chart data
            this.charts[chartName].data.datasets[0].data = [...this.dataHistory[chartName]];
            
            // Pad with zeros if not enough data
            while (this.charts[chartName].data.datasets[0].data.length < this.maxDataPoints) {
                this.charts[chartName].data.datasets[0].data.unshift(0);
            }
            
            // Update chart colors based on current value
            let color = '#27ae60'; // green
            if (chartName === 'temperature') {
                if (value > 75) color = '#e74c3c'; // red
                else if (value > 60) color = '#f39c12'; // orange
            } else {
                if (value > 70) color = '#e74c3c'; // red
                else if (value > 30) color = '#f39c12'; // orange
            }
            
            this.charts[chartName].data.datasets[0].borderColor = color;
            this.charts[chartName].data.datasets[0].backgroundColor = color + '20';
            this.charts[chartName].update('none');
        }
    }
    
    updateValueWithColor(elementId, text, percentage) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
            element.className = 'monitor-value';
            
            if (percentage < 30) {
                element.classList.add('low');
            } else if (percentage < 70) {
                element.classList.add('medium');
            } else {
                element.classList.add('high');
            }
        }
    }
    
    updateTemperatureValue(elementId, text, temperature) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
            element.className = 'monitor-value';
            
            if (temperature < 60) {
                element.classList.add('temp-normal');
            } else if (temperature < 75) {
                element.classList.add('temp-warm');
            } else {
                element.classList.add('temp-hot');
            }
        }
    }
    
    updateCoresDisplay(elementId, coreLoads) {
        const container = document.getElementById(elementId);
        if (container && coreLoads) {
            container.innerHTML = '';
            
            // Handle both arrays and objects
            const loadsArray = Array.isArray(coreLoads) ? coreLoads : Object.values(coreLoads);
            console.log(`Updating ${elementId} with ${loadsArray.length} cores:`, loadsArray);
            
            loadsArray.forEach((load, coreIndex) => {
                const badge = document.createElement('div');
                badge.className = 'core-badge';
                badge.textContent = `C${coreIndex}:   ${load.toFixed(1)}%`;
                
                // Color based on load
                if (load < 30) {
                    badge.classList.add('low');
                } else if (load < 70) {
                    badge.classList.add('medium');
                } else {
                    badge.classList.add('high');
                }
                
                container.appendChild(badge);
            });
        }
    }
    
    updateMultiCoreChart(chartName, coreValues) {
        const chart = this.charts[chartName];
        if (!chart || !coreValues || !Array.isArray(coreValues)) {
            console.error(`Chart ${chartName} not found or invalid values:`, coreValues);
            return;
        }

        // Update timestamp labels
        const now = new Date();
        const timeLabel = now.getSeconds().toString().padStart(2, '0');
        
        // Shift and add new label
        chart.data.labels.shift();
        chart.data.labels.push(timeLabel);
        
        // Update each core's data
        for (let coreIndex = 0; coreIndex < coreValues.length && coreIndex < chart.data.datasets.length; coreIndex++) {
            const value = parseFloat(coreValues[coreIndex]) || 0;
            
            // Get current dataset
            const dataset = chart.data.datasets[coreIndex];
            if (dataset && dataset.data) {
                // Shift old data and add new value
                dataset.data.shift();
                dataset.data.push(value);
            }
        }
        
        // Update chart
        chart.update('none');
        
        // Update legend with current values
        this.updateChartLegend(chartName, coreValues);
    }
    
    initChartLegend(chartName, coreCount, colors) {
        const legendContainer = document.getElementById(`${chartName}-chart-legend`);
        if (!legendContainer) {
            console.error(`Legend container for ${chartName} not found`);
            return;
        }
        
        legendContainer.innerHTML = '';
        
        for (let i = 0; i < coreCount; i++) {
            const legendItem = document.createElement('div');
            legendItem.className = 'legend-item';
            legendItem.id = `${chartName}-legend-${i}`;
            
            const colorBox = document.createElement('div');
            colorBox.className = 'legend-color';
            colorBox.style.backgroundColor = colors[i % colors.length];
            
            const label = document.createElement('span');
            label.className = 'legend-label';
            label.textContent = `Core ${i}:`;
            
            const value = document.createElement('span');
            value.className = 'legend-value';
            value.id = `${chartName}-legend-value-${i}`;
            value.textContent = '0%';
            
            legendItem.appendChild(colorBox);
            legendItem.appendChild(label);
            legendItem.appendChild(value);
            
            legendContainer.appendChild(legendItem);
        }
        
        console.log(`Initialized legend for ${chartName} with ${coreCount} cores`);
    }
    
    updateChartLegend(chartName, coreValues) {
        if (!coreValues) return;
        
        for (let i = 0; i < coreValues.length; i++) {
            const valueElement = document.getElementById(`${chartName}-legend-value-${i}`);
            if (valueElement) {
                const value = parseFloat(coreValues[i]) || 0;
                valueElement.textContent = `${value.toFixed(1)}%`;
                
                // Add color coding based on load level
                const legendItem = document.getElementById(`${chartName}-legend-${i}`);
                if (legendItem) {
                    legendItem.style.backgroundColor = value > 70 ? 'rgba(231, 76, 60, 0.2)' : 
                                                      value > 30 ? 'rgba(243, 156, 18, 0.2)' : 
                                                      'rgba(46, 204, 113, 0.2)';
                }
            }
        }
    }
}

// Global functions for button onclick handlers
function startProcessing() {
    window.yoloInterface.startProcessing();
}

function stopProcessing() {
    window.yoloInterface.stopProcessing();
}

function refreshStatus() {
    window.yoloInterface.refreshStatus();
}

function clearConsole() {
    window.yoloInterface.clearConsole();
}

function toggleAutoScroll() {
    window.yoloInterface.toggleAutoScroll();
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.yoloInterface = new YOLOWebInterface();
    
    // Request console updates every 2 seconds
    setInterval(() => {
        window.yoloInterface.requestConsoleUpdate();
    }, 500);
});
