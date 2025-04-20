document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM Content Loaded. Initializing application...");

    // --- DOM Element References ---
    const elements = {
        particlesContainer: document.getElementById('particles'),
        tabs: document.querySelectorAll('.tab'),
        fileInput: document.getElementById('image-file-input'),
        urlInput: document.getElementById('image-url-input'),
        previewContainer: document.getElementById('image-preview-container'),
        previewImage: document.getElementById('image-preview'),
        submitButton: document.getElementById('submit-button'),
        loadingContainer: document.getElementById('loading-container'),
        resultsContainer: document.getElementById('results-container'),
        analysisContent: document.getElementById('analysis-content'),
        errorMessage: document.getElementById('error-message'),
        dropzone: document.getElementById('dropzone'),
        stylistCardsContainer: document.getElementById('stylist-cards'),
    };

    // Check essential elements
    const essentialElements = ['particlesContainer', 'fileInput', 'urlInput', 'previewContainer', 'previewImage', 'submitButton', 'loadingContainer', 'resultsContainer', 'analysisContent', 'errorMessage', 'dropzone', 'stylistCardsContainer'];
    let missingElement = false;
    essentialElements.forEach(key => {
        if (!elements[key]) {
            console.error(`Initialization failed: Element with ID/selector for '${key}' not found.`);
            missingElement = true;
        }
    });

    if (missingElement) {
        document.body.innerHTML = `<h1>Error: Application failed to load. Essential page element(s) are missing. Please check the console (F12).</h1>`;
        return;
    }

    // --- Initialization Calls ---
    createParticles();
    initTabs();
    handleFileInput();
    handleUrlInput();
    handleSubmit(); // Sets up the listener
    handleDragAndDrop();

    console.log("Application Initialized.");

    // --- Core Functions ---

    function createParticles() {
        if (!elements.particlesContainer) return;
        const colors = ["#FF6B6B", "#6B66FF", "#A36BFF", "#FFD166"];
        const numParticles = 15;
        for (let i = 0; i < numParticles; i++) {
            const particle = document.createElement("div");
            particle.classList.add("particle");
            const size = Math.random() * 30 + 8;
            const color = colors[Math.floor(Math.random() * colors.length)];
            const left = Math.random() * 100;
            const duration = Math.random() * 25 + 20;
            const delay = Math.random() * 20;
            particle.style.width = `${size}px`;
            particle.style.height = `${size}px`;
            particle.style.backgroundColor = color;
            particle.style.left = `${left}vw`;
            particle.style.top = `${100 + Math.random() * 50}vh`;
            particle.style.animationDuration = `${duration}s`;
            particle.style.animationDelay = `${delay}s`;
            elements.particlesContainer.appendChild(particle);
        }
    }

    function initTabs() {
        if (!elements.tabs || elements.tabs.length === 0) return;
        elements.tabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                if (tab.classList.contains("active")) return;
                document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
                document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
                tab.classList.add("active");
                const tabName = tab.getAttribute("data-tab");
                const contentToShow = document.getElementById(`${tabName}-tab`);
                if (contentToShow) {
                    contentToShow.classList.add("active");
                }
                clearInputsAndPreview();
                hideError();
                hideResults(); // Hide results when switching tabs
            });
        });
        const initialActiveTab = document.querySelector('.tab.active');
        if (initialActiveTab) {
            const initialTabName = initialActiveTab.getAttribute('data-tab');
            const initialContent = document.getElementById(`${initialTabName}-tab`);
            if (initialContent && !initialContent.classList.contains('active')) {
                initialContent.classList.add('active');
            }
        } else if (elements.tabs.length > 0) {
            elements.tabs[0].click();
        }
    }

    function handleFileInput() {
        elements.fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file && file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    elements.previewImage.src = event.target.result;
                    elements.previewContainer.style.display = "block";
                    animatePreviewIn();
                    hideError();
                    hideResults();
                };
                reader.onerror = () => { showError("Could not read the selected file."); clearInputsAndPreview(); };
                reader.readAsDataURL(file);
            } else if (file) {
                clearInputsAndPreview(); showError('Invalid file type. Please upload an image.'); elements.fileInput.value = null;
            } else { clearImagePreviewOnly(); }
        });
    }

    function handleUrlInput() {
        const imgLoader = new Image();
        imgLoader.onload = () => {
            elements.previewImage.src = imgLoader.src;
            elements.previewContainer.style.display = "block";
            animatePreviewIn();
            hideError();
            hideResults();
        };
        imgLoader.onerror = () => {
            const url = elements.urlInput ? elements.urlInput.value.trim() : '';
            if (url) { showError("Invalid or inaccessible image URL. Check link & CORS."); }
            clearImagePreviewOnly();
        };

        elements.urlInput.addEventListener("input", debounce((e) => {
            const url = e.target.value.trim();
            if (url && isValidUrl(url)) {
                elements.previewContainer.style.display = "none";
                elements.previewImage.src = '#';
                imgLoader.src = url;
            } else if (url === "") {
                clearInputsAndPreview();
            } else {
                clearImagePreviewOnly();
            }
        }, 500));

        elements.urlInput.addEventListener('blur', (e) => {
            const url = e.target.value.trim();
            if (url && !isValidUrl(url)) {
                showError('Invalid URL format. Use http:// or https://');
            }
        });
    }

    function animatePreviewIn() {
        elements.previewImage.classList.add('visible');
    }

    function clearImagePreviewOnly() {
        elements.previewImage.src = '#';
        elements.previewImage.classList.remove('visible');
        elements.previewContainer.style.display = 'none';
    }

    function clearInputsAndPreview() {
        if (elements.fileInput) elements.fileInput.value = null;
        if (elements.urlInput) elements.urlInput.value = '';
        clearImagePreviewOnly();
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => { clearTimeout(timeout); func.apply(this, args); };
            clearTimeout(timeout); timeout = setTimeout(later, wait);
        };
    }

    function isValidUrl(string) {
        if (typeof string !== 'string') return false;
        if (!string.match(/^https?:\/\/.{3,}/)) { return false; }
        try { new URL(string); return true; } catch (_) { return false; }
    }

    function showError(message) {
        elements.errorMessage.textContent = message;
        elements.errorMessage.style.display = "block";
        elements.errorMessage.style.opacity = '1';
        clearTimeout(elements.errorMessage.timeoutId);
        elements.errorMessage.timeoutId = setTimeout(hideError, 5000);
    }

    function hideError() {
        elements.errorMessage.style.opacity = '0';
        setTimeout(() => { elements.errorMessage.style.display = "none"; }, 300);
        clearTimeout(elements.errorMessage.timeoutId);
    }

    function handleDragAndDrop() {
        const dropzone = elements.dropzone;
        const fileInput = elements.fileInput;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

        ['dragenter', 'dragover'].forEach(eventName => dropzone.addEventListener(eventName, () => dropzone.classList.add("highlight"), false));
        ['dragleave', 'drop'].forEach(eventName => dropzone.addEventListener(eventName, () => dropzone.classList.remove("highlight"), false));

        dropzone.addEventListener("drop", (e) => {
            dropzone.classList.remove("highlight");
            const dt = e.dataTransfer;
            if (!dt) return;
            const files = dt.files;
            if (files && files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    fileInput.files = files;
                    const event = new Event("change", { bubbles: true });
                    fileInput.dispatchEvent(event);
                    const fileTab = document.querySelector('.tab[data-tab="file"]');
                    if (fileTab && !fileTab.classList.contains('active')) { fileTab.click(); }
                    hideError();
                } else { showError("Invalid file type dropped. Please drop an image."); clearInputsAndPreview(); }
            }
        }, false);
    }

    function showLoading() {
        elements.loadingContainer.style.display = "block";
        setTimeout(() => elements.loadingContainer.classList.add('visible'), 10);
    }

    function hideLoading() {
        elements.loadingContainer.classList.remove('visible');
        setTimeout(() => { elements.loadingContainer.style.display = "none"; }, 300);
    }

    function showResults() {
        elements.resultsContainer.style.display = "block";
        setTimeout(() => {
            elements.resultsContainer.classList.add('visible');
            elements.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 10);
    }

    function hideResults() {
        elements.resultsContainer.classList.remove('visible');
        setTimeout(() => {
            if (!elements.resultsContainer.classList.contains('visible')) {
                elements.resultsContainer.style.display = "none";
            }
        }, 500);
    }

    // --- UI Update Functions ---

    function clearStylistRecommendations() {
        elements.stylistCardsContainer.innerHTML = '';
    }

    function createRecommendationCard(item) {
        if (!item || typeof item !== 'object') {
            console.warn("Skipping invalid item for recommendation card:", item);
            return;
        }

        const card = document.createElement("div");
        card.className = "flash-card";

        let imgElement = null;
        const textContentDiv = document.createElement('div');
        textContentDiv.className = 'flash-card-text-content';

        if (item.image_url && typeof item.image_url === 'string') {
            const img = document.createElement('img');
            img.className = 'stylist-card-image'; // Base class
            img.classList.add('loading'); // Add loading class
            img.alt = `Loading visualization for ${item.option || 'Outfit Idea'}...`;
            img.loading = 'lazy';

            img.onload = () => {
                img.classList.remove('loading'); // Remove loading on success
                img.alt = `Visualization for ${item.option || 'Outfit Idea'}`;
            };

            img.onerror = () => {
                console.warn(`Failed to load image: ${item.image_url}`);
                img.classList.remove('loading'); // Remove loading on error
                img.alt = `Image failed to load`;
                img.style.display = 'none';
                const errorPlaceholder = document.createElement('div');
                errorPlaceholder.textContent = 'Image Unavailable';
                errorPlaceholder.style.cssText = 'height: 220px; display: flex; align-items: center; justify-content: center; background: #eee; color: #888; font-size: 0.9em; border-bottom: 1px solid #ddd;';
                card.insertBefore(errorPlaceholder, textContentDiv);
            };

            img.src = item.image_url; // Set src AFTER listeners defined

            card.appendChild(img);
            imgElement = img;
        } else {
             const noImagePlaceholder = document.createElement('div');
             noImagePlaceholder.textContent = 'No Image Visualization';
             noImagePlaceholder.style.cssText = 'height: 220px; display: flex; align-items: center; justify-content: center; background: #f8f8f8; color: #aaa; font-size: 0.9em; border-bottom: 1px solid #eee; text-align: center; padding: 10px;';
             card.appendChild(noImagePlaceholder);
        }

        const title = document.createElement("div");
        title.className = "flash-card-title";
        title.textContent = (typeof item.option === 'string' ? item.option : "Style Suggestion");
        textContentDiv.appendChild(title);

        const content = document.createElement("div");
        content.className = "flash-card-content";
        let recommendationText = (typeof item.text === 'string' ? item.text : "No details provided.");
        let formattedContent = recommendationText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^\s*[-*]\s+/gm, '<li>')
            .replace(/^\s*\d+\.\s+/gm, '<li>')
            .replace(/<\/li>\s*<li>/g, '</li><li>')
            .replace(/((?:<li>.*?<\/li>\s*)+)/g, '<ul>$1</ul>')
            .replace(/\n/g, '<br>');
        content.innerHTML = formattedContent;
        textContentDiv.appendChild(content);
        card.appendChild(textContentDiv);

        // Add Click Listener to Image
        if (imgElement && typeof item.text === 'string' && item.image_url) {
            imgElement.addEventListener('click', () => {
                try {
                    const encodedImgUrl = encodeURIComponent(item.image_url);
                    // Use original unformatted text for URL parameter if possible
                    const encodedText = encodeURIComponent(recommendationText);
                    // Ensure the path is relative to the domain root if image_viewer is in static
                    const viewerUrl = `/static/image_viewer.html?imgUrl=${encodedImgUrl}&text=${encodedText}`;
                    window.open(viewerUrl, '_blank');
                } catch (e) {
                    console.error("Error creating or opening viewer URL:", e);
                }
            });
        }

        elements.stylistCardsContainer.appendChild(card);
        const cardIndex = elements.stylistCardsContainer.children.length - 1;
        setTimeout(() => {
            card.classList.add("visible");
        }, cardIndex * 100); // Stagger animation
    }


    function processApiResponse(result) {
        console.log("--- Processing API response START ---");
        console.log("Full received result:", JSON.stringify(result, null, 2)); // Keep for debugging if needed

        elements.analysisContent.textContent = result.caption || "Analysis not available.";
        clearStylistRecommendations();

        console.log("Inspecting result.recommendations - Value:", result.recommendations);
        console.log("Inspecting result.recommendations - Type:", typeof result.recommendations);

        if (result.recommendations && Array.isArray(result.recommendations)) {
            console.log("Condition PASSED: result.recommendations is an Array.");
            if (result.recommendations.length > 0) {
                console.log(`Array has length > 0. Displaying ${result.recommendations.length} recommendations.`);
                result.recommendations.forEach(item => {
                    if (item && typeof item === 'object') {
                        createRecommendationCard(item);
                    } else {
                        console.warn("Skipping malformed item in recommendations array:", item);
                    }
                });
            } else {
                console.log("Array is empty.");
                createRecommendationCard({
                    option: "Stylist Note",
                    text: "No specific outfit ideas were generated for this item.",
                    image_url: null
                });
            }
        } else {
            console.log("Condition FAILED: result.recommendations is NOT an Array.");
            console.warn("Recommendations received are not in the expected array format. Value:", result.recommendations, "Type:", typeof result.recommendations);
            let fallbackText = "Could not retrieve outfit recommendations in the expected format.";

            if (typeof result.recommendations === 'string' && result.recommendations) {
                 console.log("Condition PASSED: typeof result.recommendations is 'string' AND it's truthy.");
                 console.log("Value before toLowerCase:", result.recommendations);
                try {
                    // Check if it's an error string, case-insensitive
                    if (result.recommendations.toLowerCase().includes("error") || result.recommendations.toLowerCase().includes("limit reached")) {
                        console.error("Detected error/limit message in recommendations field:", result.recommendations);
                        fallbackText = "An error occurred while generating recommendations (e.g., quota limit).";
                    } else {
                         console.log("String does not contain error/limit. Using it as fallback text.");
                        fallbackText = result.recommendations; // Display other string messages directly
                    }
                } catch (e) {
                     console.error("!!! CRITICAL ERROR calling .toLowerCase() or .includes() !!!");
                     console.error("Error:", e);
                     fallbackText = "An unexpected internal error occurred processing recommendations.";
                }
            } else {
                console.log("Condition FAILED: result.recommendations is either not a string, or it's null/empty string/undefined.");
            }

            createRecommendationCard({
                option: "Stylist Update",
                text: fallbackText,
                image_url: null
            });
        }
        console.log("--- Processing API response END ---");
    }


    // --- Form Submission Handler Setup ---
    function handleSubmit() {
        elements.submitButton.addEventListener("click", async () => {
            hideError();
            const activeTabElement = document.querySelector(".tab.active");
            if (!activeTabElement) { showError("Cannot determine active input tab."); return; }
            const activeTab = activeTabElement.getAttribute("data-tab");

            const formData = new FormData();
            let hasInput = false;
            if (activeTab === "file") {
                if (elements.fileInput.files.length > 0 && elements.fileInput.files[0].type.startsWith('image/')) {
                    formData.append("image_file", elements.fileInput.files[0]); hasInput = true;
                } else if (elements.fileInput.files.length > 0) { showError("Invalid file type selected."); }
                 else { showError("Please select an image file."); }
            } else if (activeTab === "url") {
                const url = elements.urlInput.value.trim();
                if (url && isValidUrl(url)) {
                    formData.append("image_url", url); hasInput = true;
                } else if (url) { showError("Invalid URL format."); }
                 else { showError("Please enter a valid image URL."); }
            }

            if (!hasInput) return;

            elements.submitButton.disabled = true;
            hideResults();
            clearStylistRecommendations();
            elements.analysisContent.textContent = 'Analyzing...';
            showLoading();

            try {
                const response = await fetch("/predict", { method: "POST", body: formData });

                if (!response.ok) {
                    let errorDetail = `Request failed: ${response.status}`;
                    try { const errorData = await response.json(); errorDetail = errorData.detail || errorDetail; }
                    catch (e) { errorDetail += ` (${response.statusText || 'Network error'})`; }
                    throw new Error(errorDetail);
                }

                const result = await response.json();
                processApiResponse(result); // Process response

                // If successful, show results container AFTER processing
                elements.resultsContainer.style.display = "block";
                elements.resultsContainer.style.opacity = "1";
                elements.resultsContainer.classList.add('visible');
                elements.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });


            } catch (error) {
                console.error("Error during submission/fetch:", error);
                showError(`Failed to get style advice: ${error.message}`);
                elements.analysisContent.textContent = 'Analysis failed.';
                hideResults(); // Ensure results stay hidden on error

            } finally {
                hideLoading(); // Always hide loading
                elements.submitButton.disabled = false;
                console.log("Submission process finished.");
            }
        });
    }

}); // End DOMContentLoaded