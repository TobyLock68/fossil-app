# Fossil Collection Upload & AI Identifier

An offline-first, browser-based Progressive Web App (PWA) designed for fossil collectors, field researchers, and paleontologists. 

This project solves two primary challenges in paleontology dataset collection and field identification:
1. **Ground-Truth Dataset Capture:** Capturing high-quality, verified fossil photographs directly from field sites or personal collections and uploading them automatically to organized Google Drive directories.
2. **Offline AI Identification:** Identifying fossil species directly inside your web browser without an internet connection using an integrated ONNX neural network model paired with regional geological soft-filtering rules.

---

## System Architecture

```text
                     ┌──────────────────────────────────────────────┐
                     │            User Device (Browser PWA)         │
                     └──────────────────────┬───────────────────────┘
                                            │
                     ┌──────────────────────┴───────────────────────┐
                     │          Tab Switch Navigation               │
                     └──────────────┬───────────────────────┬───────┘
                                    │                       │
           ┌────────────────────────┴────────┐     ┌────────┴─────────────────────────┐
           │        TAB 1: DATA CAPTURE      │     │      TAB 2: AI IDENTIFICATION    │
           └────────────────┬────────────────┘     └────────┬─────────────────────────┘
                            │                               │
                [Camera / Library Upload]               [Selected Fossil Image]
                            │                               │
             ┌──────────────┴──────────────┐       ┌────────┴─────────────────────────┐
             │      Connection Check       │       │       ONNX Runtime Web CDN       │
             └──────┬──────────────┬───────┘       └────────┬─────────────────────────┘
                    │              │                        │
             (Online)          (Offline)            [Loads Model & Labels]
                    │              │                        │
       ┌────────────┴──┐      ┌────┴──────────────┐ [Image Preprocessing & Tensor Normalization]
       │ Google Apps   │      │ IndexedDB Queue   │         │
       │ Script API    │      │ (Local Storage)   │ [Softmax Logits -> Base Probabilities]
       └──────┬────────┘      └────┬──────────────┘         │
              │                    │                [Regional Penalty Soft-Filter (0.05x)]
       ┌──────┴────────┐      (Sync on             │
       │ Google Drive  │       Reconnect)          [Top-5 Matches & Region Badges UI]
       │ Subfolders    │───────────┘
       └───────────────┘
```

---

## File Structure & Detailed Code Guide

Every file in this project plays a specific role in UI layout, background performance, database storage, serverless cloud uploads, or artificial intelligence execution. Below is an in-depth, beginner-friendly explanation of how each file works.

---

### 1. `index.html` — The Main User Interface & App Engine

`index.html` is the core visual interface of the app. It displays the splash screen, the tab navigation bar, image upload controls, and the AI identification results panel.

#### Key Technical Features & Code Choices:
- **Mobile Standalone Shell:**
  * Uses `<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=3.0, user-scalable=yes">` to enable responsive display across mobile devices while allowing user zooming when inspecting detailed specimen features.
  * Apple-specific meta tags (`apple-mobile-web-app-capable`) allow iOS users who tap "Add to Home Screen" to launch the website as a standalone mobile application without standard Safari browser address bars.
- **ONNX Web Library Integration:**
  * Imports the ONNX Runtime Web CDN script inside the `<head>` block (`<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>`). This provides WebAssembly (WASM) and WebGL engines inside your browser, allowing complex AI algorithms to run locally on hardware without a backend server.
- **Tab Layout & UI Flow:**
  * Uses three primary section wrappers (`uploadTab`, `identifyTab`, `queueTab`).
  * The **Upload Tab** groups 67 target species into logical taxonomic categories (*Ammonites*, *Trilobites*, *Sea Urchins*, *Vertebrates*, etc.) to make drop-down selection fast during fieldwork.
  * The **Identify Tab** contains image preview screens, a location/formation filter dropdown, and a live target area where predictions are rendered.
- **Client-Side Offline Storage (`IndexedDB`):**
  * Opens a local browser database called `FossilCollectorDB` using the JavaScript `indexedDB` API.
  * When a photo is uploaded in offline mode, it converts the raw image into a text format called Base64 via `FileReader`. It then stores this Base64 text string, the chosen species label, and a timestamp into the database so no work is lost off the grid.
- **Google Apps Script Cloud Upload Engine:**
  * Sends text-encoded images to a serverless Google Apps Script API endpoint (`SCRIPT_URL`) using JavaScript `fetch()`. If the network request fails due to no signal, the file is automatically saved into the local `IndexedDB` draft queue.

---

### 2. `identifier.js` — Offline AI Neural Network & Regional Soft-Filtering Engine

`identifier.js` connects the user interface to the deep learning AI model. It handles model initialization, image transformation, neural network inference, and regional location adjustments.

#### Key Technical Features & Code Choices:
- **Model Initialization & Asset Fetching:**
  * On page load, `initFossilIdentifier()` fetches `./model/labels.json` to obtain the model metadata, species target array, normalization parameters (mean/standard deviation), and location maps.
  * Initializes an ONNX inference session via `ort.InferenceSession.create('./model/fossil_model.onnx')`, choosing WebGL or WASM hardware acceleration dynamically depending on the user's phone or computer.
- **Image Preprocessing (Canvas NCHW Transformation):**
  * Neural networks cannot read raw image files like `.jpg` directly; they require normalized floating-point numbers arranged in a specific matrix shape.
  * `preprocessImage()` takes an HTML image element, draws it onto an internal $224 \times 224$ pixel HTML5 canvas, and crops it centrally to preserve aspect ratios.
  * It extracts pixel RGB channels ($0-255$), scales them between $0.0$ and $1.0$, and normalizes them using ImageNet standards:
    $$\text{Normalized Channel} = \frac{\text{Pixel Channel} - \text{Mean}}{\text{Std}}$$
  * Re-arranges memory into an NCHW tensor format ($\text{Batch}=1, \text{Channels}=3, \text{Height}=224, \text{Width}=224$) expected by EfficientNet models.
- **Logit Conversion via Softmax:**
  * The raw output of a neural network is an array of raw numerical scores called *logits*.
  * The script applies the **Softmax** mathematical operation to convert these logits into true probabilities that add up to $1.0$ ($100\%$):
    $$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$
- **Regional Soft-Filtering Rule System:**
  * Standard AI models often guess species that are geographically impossible for a given search region.
  * To fix this, `identifier.js` compares each candidate species against `species_regions` rules in `labels.json`.
  * If a predicted species is not naturally found in the region selected by the user in the UI dropdown, its prediction probability is multiplied by a soft penalty weight (`LOCATION_PENALTY = 0.05`).
  * This penalizes out-of-region species while keeping them visible if the AI is extremely confident.
- **Top-5 Match Ranking:**
  * Sorts all species by their adjusted probabilities descending and renders the top 5 matches into clean HTML elements formatted with visual badges indicating whether the match is "In Region" or "Out of Region".

---

### 3. `./model/fossil_model.onnx` — The Trained AI Neural Network Model

This is the exported, binary machine learning model file.

- **Architecture:** Trained on an EfficientNet deep learning architecture tuned for multi-class specimen classification.
- **Format:** Exported to the open **ONNX (Open Neural Network Exchange)** format.
- **Function:** Takes a normalized image tensor ($1 \times 3 \times 224 \times 224$) as input and outputs a class probability vector corresponding to active fossil species. Running directly via WebAssembly inside the client's browser eliminates server costs and API latency.

---

### 4. `./model/labels.json` — Model Configuration & Geographic Mapping Matrix

This JSON configuration file serves as the translation dictionary for the ONNX binary model.

- **Model Metadata:** Holds model version strings, input resolution size ($224 \times 224$), normalization color channel averages (`mean`), and standard deviations (`std`).
- **Class Index Mapping:** Contains the `classes` array mapping output numerical array indices ($0, 1, 2...$) to readable species class names (e.g., `dactylioceras`, `calymene_trilobite`).
- **Geographic Matrix (`species_regions`):** Maps every species class to the specific UK formations or regions where it naturally occurs (e.g., `"arnioceras": ["dorset", "yorkshire"]`).

---

### 5. `sw.js` — Service Worker (Offline Application Caching)

A Service Worker is a background script that runs separately from the web page to provide reliable offline capabilities.

- **Asset Pre-Caching:** Caches core application assets (`index.html`, `identifier.js`, `manifest.json`, `labels.json`, and `fossil_model.onnx`) into browser CacheStorage during the installation phase.
- **Cache-First Strategy:** Intercepts incoming network HTTP `fetch` calls. If the user loses mobile reception while out searching in a quarry or along coastal cliffs, the Service Worker serves all app code and AI models directly from local device memory.

---

### 6. `manifest.json` — Progressive Web App Configuration

Tells mobile operating systems (iOS & Android) how to install and display the web app.

- **Standalone Execution:** Configured with `"display": "standalone"`, launching the website like a native app installed from an App Store without browser frames.
- **Theme Color & Branding:** Sets `"background_color": "#1c1c1e"` to give native-matching iOS dark mode splash transitions.

---

### 7. `Code.gs` — Serverless Google Apps Script Cloud Backend

A serverless script hosted inside Google Workspace that receives uploaded fossil images from your app and saves them into Google Drive.

- **API Endpoint (`doPost(e)`):** Accepts incoming HTTP `POST` requests sent by `index.html`.
- **Dynamic Folder Routing:** Scans your master Google Drive folder (`Fossil_Images`) for a subfolder matching the selected species name. If the subfolder does not exist, it creates it automatically.
- **Base64 File Reconstruction:** Decodes incoming Base64 image payload strings back into binary image files using `Utilities.base64Decode()`, assigns timestamped filenames (e.g., `handshot_1727044257000.jpg`), and saves them into your target species subfolder.

---

## Re-Training Pipeline Summary

When new species images are gathered using the app and uploaded to Google Drive:

1. A Python re-training script hosted in Google Colab recursively scans your Drive directory (`/MyDrive/Fossil_Images/`).
2. Folders meeting the active image threshold ($\ge 27$ images) are automatically compiled into a new training split.
3. Pending folders ($1-26$ images) and empty folders are skipped automatically.
4. An updated model is evaluated, compiled, and automatically exported as new `fossil_model.onnx` and `labels.json` assets under `export/vX/`, ready to be pushed to your web repository!
