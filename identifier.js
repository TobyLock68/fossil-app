let ortSession = null;
let modelConfig = null;
let currentIdentifyBase64 = null;

// 1. Initialize and load model + labels on startup
window.addEventListener('DOMContentLoaded', async () => {
    try {
        console.log("Loading model configuration...");
        const response = await fetch('./model/labels.json');
        modelConfig = await response.json();
        console.log(`Loaded model version: ${modelConfig.version} with ${modelConfig.classes.length} active classes.`);

        // Set WebAssembly execution paths for browser runtime
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
        
        console.log("Loading ONNX model into browser memory...");
        ortSession = await ort.InferenceSession.create('./model/fossil_model.onnx', {
            executionProviders: ['webgl', 'wasm']
        });
        console.log("✅ Fossil identifier ready!");

        const statusDiv = document.getElementById('identifyStatusText');
        if (statusDiv) statusDiv.innerText = "Model ready. Choose an image to identify.";

        // Enable button once loaded
        const runBtn = document.getElementById('runIdentifyBtn');
        if (runBtn) runBtn.removeAttribute('disabled');
    } catch (err) {
        console.error("Failed to initialize fossil identifier:", err);
        const statusDiv = document.getElementById('identifyStatusText');
        if (statusDiv) statusDiv.innerText = "Error loading AI model: " + err.message;
    }
});

// 2. Identify Tab File Picker Handling
const idCameraInput = document.getElementById('identifyCameraInput');
const idLibraryInput = document.getElementById('identifyLibraryInput');
const idSnapBtn = document.getElementById('idSnapBtn');
const idLibraryBtn = document.getElementById('idLibraryBtn');
const identifyImagePreview = document.getElementById('identifyImagePreview');

if (idSnapBtn) idSnapBtn.addEventListener('click', () => idCameraInput.click());
if (idLibraryBtn) idLibraryBtn.addEventListener('click', () => idLibraryInput.click());

function handleIdentifyFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            currentIdentifyBase64 = event.target.result;
            identifyImagePreview.src = currentIdentifyBase64;
            identifyImagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
}

if (idCameraInput) idCameraInput.addEventListener('change', handleIdentifyFileSelect);
if (idLibraryInput) idLibraryInput.addEventListener('change', handleIdentifyFileSelect);

// 3. Preprocess image to match training pipeline (Resize, Center Crop, Normalize NCHW)
async function preprocessImage(imageElement, inputSize = 224) {
    const canvas = document.createElement('canvas');
    canvas.width = inputSize;
    canvas.height = inputSize;
    const ctx = canvas.getContext('2d');

    const minDim = Math.min(imageElement.width, imageElement.height);
    const startX = (imageElement.width - minDim) / 2;
    const startY = (imageElement.height - minDim) / 2;

    ctx.drawImage(imageElement, startX, startY, minDim, minDim, 0, 0, inputSize, inputSize);
    const imgData = ctx.getImageData(0, 0, inputSize, inputSize);
    const pixels = imgData.data;

    const float32Data = new Float32Array(3 * inputSize * inputSize);
    const mean = modelConfig.mean; // [0.485, 0.456, 0.406]
    const std = modelConfig.std;   // [0.229, 0.224, 0.225]

    for (let i = 0; i < inputSize * inputSize; i++) {
        let r = pixels[i * 4] / 255.0;
        let g = pixels[i * 4 + 1] / 255.0;
        let b = pixels[i * 4 + 2] / 255.0;

        float32Data[i] = (r - mean[0]) / std[0];
        float32Data[inputSize * inputSize + i] = (g - mean[1]) / std[1];
        float32Data[2 * inputSize * inputSize + i] = (b - mean[2]) / std[2];
    }

    return new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);
}

// 4. Softmax activation helper
function softmax(arr) {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// 5. Main Identification Execution with Region Soft-Filtering
const runIdentifyBtn = document.getElementById('runIdentifyBtn');
if (runIdentifyBtn) {
    runIdentifyBtn.addEventListener('click', async () => {
        if (!ortSession || !modelConfig) {
            alert("Model is still loading, please wait...");
            return;
        }
        if (!currentIdentifyBase64) {
            alert("Please take or upload an image to identify first.");
            return;
        }

        const statusText = document.getElementById('identifyStatusText');
        const resultsContainer = document.getElementById('identifyResultsContainer');
        const selectedRegion = document.getElementById('locationFilter').value;

        statusText.innerText = "Running local AI inference...";
        resultsContainer.innerHTML = "";

        try {
            // Load image into an HTMLImageElement for canvas processing
            const imgElement = new Image();
            imgElement.src = currentIdentifyBase64;
            await new Promise((resolve, reject) => {
                imgElement.onload = resolve;
                imgElement.onerror = reject;
            });

            const tensor = await preprocessImage(imgElement, modelConfig.input_size);
            const results = await ortSession.run({ input: tensor });
            const logits = Array.from(results[ortSession.outputNames[0]].data);
            
            let probabilities = softmax(logits);

            const classes = modelConfig.classes;
            const speciesRegions = modelConfig.species_regions;
            const penalty = modelConfig.location_penalty; // 0.05 multiplier

            // Apply Location Soft-Filtering Rules
            let adjustedPredictions = classes.map((className, index) => {
                let prob = probabilities[index];
                let allowedRegions = speciesRegions[className] || ['*'];
                
                const isGlobal = allowedRegions.includes('*');
                const isInRegion = isGlobal || allowedRegions.includes(selectedRegion) || selectedRegion === 'ALL';

                if (!isInRegion) {
                    prob *= penalty; // Soft penalty for out-of-region species
                }

                return { className, probability: prob, inRegion: isInRegion };
            });

            // Sort descending by adjusted probability
            adjustedPredictions.sort((a, b) => b.probability - a.probability);
            const top5 = adjustedPredictions.slice(0, 5);

            statusText.innerText = "Identification complete!";

            // Render Results using clean UI classes
            let html = '<ul class="pred-list">';
            top5.forEach((pred, idx) => {
                let pct = (pred.probability * 100).toFixed(1);
                let regionBadgeClass = pred.inRegion ? "in-region" : "out-region";
                let regionBadgeText = pred.inRegion ? "In Region" : "Out of Region (Penalized)";
                
                html += `
                    <li class="pred-item">
                        <div>
                            <div class="pred-name">${idx + 1}. ${pred.className}</div>
                            <small style="color: var(--subtext-color);">${pct}% confidence</small>
                        </div>
                        <span class="pred-badge ${regionBadgeClass}">${regionBadgeText}</span>
                    </li>
                `;
            });
            html += '</ul>';
            resultsContainer.innerHTML = html;

        } catch (err) {
            console.error("Inference error:", err);
            statusText.innerText = "Error during identification.";
            resultsContainer.innerHTML = `<p style="color: var(--danger-color); font-size: 0.85rem; margin-top: 8px;">${err.message}</p>`;
        }
    });
}
