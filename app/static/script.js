// --- DOM Element References (UPDATED to match new HTML) ---
const elements = {
    particlesContainer: document.getElementById('particles'),
    tabs: document.querySelectorAll('.tab'),
    fileInput: document.getElementById('image-file-input'),
    urlInput: document.getElementById('image-url-input'),
    previewContainer: document.getElementById('image-preview-container'),
    previewImage: document.getElementById('image-preview'),
    submitButton: document.getElementById('submit-button'),
    loadingContainer: document.getElementById('loading-container'),
    // --- Updated IDs ---
    resultsContainer: document.getElementById('results-container'), // Use results container
    analysisContent: document.getElementById('analysis-content'),   // Use analysis content for caption
    // --- ---
    errorMessage: document.getElementById('error-message'),
    dropzone: document.getElementById('dropzone'),
    stylistCardsContainer: document.getElementById('stylist-cards'), // This ID was correct
    welcomeCard: document.querySelector('#stylist-cards .flash-card:first-child'),
};

// Create animated background particles
function createParticles() {
    // ... (Particle creation logic remains the same) ...
    if (!elements.particlesContainer) {
        console.warn("Particles container not found.");
        return;
    }
    const colors = ["#FF6B6B", "#6B66FF", "#A36BFF", "#FFD166"];
    const numParticles = 20;

    for (let i = 0; i < numParticles; i++) {
        const particle = document.createElement("div");
        particle.classList.add("particle");
        const size = Math.random() * 40 + 10;
        const color = colors[Math.floor(Math.random() * colors.length)];
        const left = Math.random() * 100;
        const duration = Math.random() * 20 + 15;
        const delay = Math.random() * 15;
        const drift = Math.random() * 2 - 1;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.backgroundColor = color;
        particle.style.left = `${left}vw`;
        particle.style.top = `${100 + Math.random() * 50}vh`;
        particle.style.animationDuration = `${duration}s`;
        particle.style.animationDelay = `${delay}s`;
        particle.style.setProperty('--drift', drift);
        elements.particlesContainer.appendChild(particle);
    }
     // Ensure CSS for .particle and @keyframes float-particle exists
     // matching the animation used (e.g., the one from previous versions)
}

// Initialize tabs
function initTabs() {
    // ... (Tab initialization logic remains the same) ...
    if (!elements.tabs || elements.tabs.length === 0) return;
    console.log("Initializing tabs...");
    elements.tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            console.log("Handler Started: initTabs click");
            if (tab.classList.contains("active")) return;
            document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
            tab.classList.add("active");
            const tabName = tab.getAttribute("data-tab");
            const contentToShow = document.getElementById(`${tabName}-tab`);
            if (contentToShow) {
                contentToShow.classList.add("active");
            } else {
                 console.warn(`Content for tab '${tabName}' not found.`);
            }
            // ** Behavior Decision: Clear inputs on tab switch? **
            // Comment out if you want inputs to persist across tab switches.
            clearInputsAndPreview();
            hideError();
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

// Handle file input change
function handleFileInput() {
    // ... (File input handling logic remains the same) ...
    if (!elements.fileInput || !elements.previewContainer || !elements.previewImage) return;
    elements.fileInput.addEventListener("change", (e) => {
        console.log("Handler Started: handleFileInput change");
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            console.log("Valid file selected:", file.name);
            const reader = new FileReader();
            reader.onload = (event) => {
                elements.previewImage.src = event.target.result;
                elements.previewContainer.style.display = "block";
                animatePreviewIn();
                hideError();
            };
             reader.onerror = () => {
                console.error("FileReader error reading file.");
                showError("Could not read the selected file.");
                clearInputsAndPreview();
            };
            reader.readAsDataURL(file);
        } else if (file) {
             console.warn("Invalid file type selected via input:", file.type);
             clearInputsAndPreview();
             showError('Invalid file type. Please upload an image.');
             elements.fileInput.value = null;
        } else {
            console.log("File input cleared or cancelled.");
            clearImagePreviewOnly();
        }
    });
}

// Handle URL input
function handleUrlInput() {
    // ... (URL input handling logic remains largely the same, including onerror/onload) ...
     if (!elements.urlInput || !elements.previewContainer || !elements.previewImage) return;

    elements.previewImage.onload = () => {
        const url = elements.previewImage.src; // Get the URL that loaded
        console.log("Handler Started: previewImage onload", url.substring(0, 50) + "...");
        elements.previewContainer.style.display = "block";
        animatePreviewIn();
        hideError();
    };

    elements.previewImage.onerror = () => {
        const currentSrc = elements.previewImage.getAttribute('src'); // Get the literal attribute value
    
        // Check if src is empty or just '#' - This check is crucial!
        if (!currentSrc || currentSrc === '#' || currentSrc === window.location.href + '#') {
            console.log(`previewImage onerror ignored for src='${currentSrc}' (empty/hash).`);
            return; // Stop execution for this specific case
        }
    
        // ... rest of the error handling logic ...
        const url = elements.urlInput ? elements.urlInput.value.trim() : '[URL input not found]';
        console.error(`Handler Started: previewImage onerror. Failed URL: ${url}`);
        if (url) {
             showError("Invalid or inaccessible image URL. Please provide a direct link (check CORS?).");
        }
        // Call the version using '#' now
        clearImagePreviewOnly();
    };

    elements.urlInput.addEventListener("input", debounce((e) => {
            const url = e.target.value.trim();
             console.log("Handler Started: urlInput debounced check:", url);
            if (url && isValidUrl(url)) {
                elements.previewImage.src = url;
                elements.previewContainer.style.display = "none"; // Hide until load/error
            } else if (url === "") {
                clearInputsAndPreview();
            } else {
                clearImagePreviewOnly();
            }
        }, 500),
    );

    elements.urlInput.addEventListener('blur', (e) => {
        const url = e.target.value.trim();
        console.log("Handler Started: urlInput blur");
        if (url && !isValidUrl(url)) {
            showError('Invalid URL format. Use http:// or https://');
        }
     });
}

// Animate image preview in
function animatePreviewIn() {
    // ... (Animation logic remains the same) ...
    if (!elements.previewImage) return;
    console.log("Animating preview in...");
    elements.previewImage.style.opacity = '0';
    elements.previewImage.style.transform = 'scale(0.95)';
    requestAnimationFrame(() => {
        elements.previewImage.style.transition = 'all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)';
        elements.previewImage.style.opacity = '1';
        elements.previewImage.style.transform = 'scale(1)';
    });
}

function clearImagePreviewOnly() {
    console.log("Clearing image preview only using '#'.");
    if (elements.previewImage) {
         // ***** CHANGE THIS LINE BACK *****
         elements.previewImage.src = '#'; // Use '#' instead of a placeholder path
    }
    if (elements.previewContainer) {
        elements.previewContainer.style.display = 'none';
    }
}

function clearInputsAndPreview() {
    console.log("Clearing inputs and preview.");
    if (elements.fileInput) elements.fileInput.value = null;
    if (elements.urlInput) elements.urlInput.value = '';
    clearImagePreviewOnly();
}
// --- ---

// Debounce function
function debounce(func, wait) {
    // ... (Debounce logic remains the same) ...
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func.apply(this, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Validate URL format
function isValidUrl(string) {
    // ... (URL validation logic remains the same) ...
    if (typeof string !== 'string') return false;
    if (!string.startsWith('http://') && !string.startsWith('https://')) {
        return false;
    }
    try {
        new URL(string);
        return true;
    } catch (_) {
        return false;
    }
}

// Show error message
function showError(message) {
    // ... (Error showing logic remains the same) ...
     if (!elements.errorMessage) return;
    console.log("Showing error:", message);
    elements.errorMessage.textContent = message;
    elements.errorMessage.style.display = "block";
    elements.errorMessage.style.animation = 'none';
    requestAnimationFrame(() => {
        elements.errorMessage.style.animation = "errorFadeIn 0.3s ease-out forwards";
    });
    clearTimeout(elements.errorMessage.timeoutId);
    elements.errorMessage.timeoutId = setTimeout(() => {
        hideError();
    }, 5000);
}

// Hide error message
function hideError() {
    // ... (Error hiding logic remains the same) ...
     if (!elements.errorMessage) return;
    if (elements.errorMessage.style.display === 'block' && elements.errorMessage.style.opacity !== '0') {
        console.log("Hiding error message.");
        elements.errorMessage.style.animation = "errorFadeOut 0.3s ease-out forwards";
        setTimeout(() => {
            if (elements.errorMessage.style.opacity === '0') {
                 elements.errorMessage.style.display = "none";
                 elements.errorMessage.style.animation = '';
            }
        }, 300);
    }
     clearTimeout(elements.errorMessage.timeoutId);
}


// Add a recommendation card to the stylist panel
function addStylistRecommendation(title, content, tag) {
    // ... (Adding recommendation logic remains the same) ...
    if (!elements.stylistCardsContainer) return;
    console.log(`Adding stylist recommendation: ${title}`);
    const card = document.createElement("div");
    card.className = "flash-card";
    let formattedContent = content || "";
    formattedContent = formattedContent
        .replace(/</g, "<")
        .replace(/>/g, ">")
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^- (.*?)(?=\n- |\n\n|$)/gm, '<li>$1</li>')
        .replace(/(\<\/li\>)\s*<li>/g, '$1</li><li>')
        .replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>')
        .replace(/\n/g, '<br>');
    card.innerHTML = `
        <div class="flash-card-title">${title}</div>
        <div class="flash-card-content">${formattedContent}</div>
        <div class="flash-card-tag">${tag}</div>
    `;
    elements.stylistCardsContainer.appendChild(card);
    requestAnimationFrame(() => {
        card.classList.add("visible");
    });
    elements.stylistCardsContainer.scrollTo({
        top: elements.stylistCardsContainer.scrollHeight,
        behavior: 'smooth'
    });
}

// Clear only dynamically added recommendation cards
function clearStylistRecommendations() {
    // ... (Clearing logic remains the same) ...
     if (!elements.stylistCardsContainer) return;
     console.log("Clearing added stylist recommendations.");
     const addedCards = Array.from(elements.stylistCardsContainer.children).slice(elements.welcomeCard ? 1 : 0); // Handle if welcome card doesn't exist
     addedCards.forEach(card => elements.stylistCardsContainer.removeChild(card));
     if(elements.welcomeCard) {
         elements.welcomeCard.classList.add('visible');
         elements.welcomeCard.style.opacity = '1';
         elements.welcomeCard.style.transform = 'translateY(0)';
     }
}

// Process the API response and update UI (UPDATED)
function processApiResponse(result) {
    console.log("Processing API response:", result);

    // --- Display the Caption/Analysis ---
    // ***** CHANGE HERE: Target #analysis-content *****
    if (elements.analysisContent) {
        elements.analysisContent.textContent = result.caption || "No analysis generated.";
        console.log("Analysis/Caption updated in #analysis-content.");
    } else {
        console.warn("Analysis content element (#analysis-content) not found.");
    }

    // --- Display Recommendations ---
    clearStylistRecommendations(); // Clear previous recommendations first

    if (result.recommendations && result.recommendations !== "Recommendations unavailable." && !result.recommendations.toLowerCase().includes("error")) {
        addStylistRecommendation(
            "Outfit Suggestions",
            result.recommendations,
            "Style Analysis"
        );
    } else {
        addStylistRecommendation(
            "Stylist Note",
            result.recommendations || "Could not retrieve recommendations.",
            "Info"
        );
    }
}

// Handle form submission (UPDATED)
function handleSubmit() {
    // ***** CHANGE HERE: Check for #results-container *****
    if (!elements.submitButton || !elements.loadingContainer || !elements.resultsContainer || !elements.fileInput || !elements.urlInput) {
        console.error("Submit handling cannot initialize: Required elements missing (check for results-container).");
        return;
    }

    elements.submitButton.addEventListener("click", async () => {
        console.log("Handler Started: handleSubmit click");
        hideError();

        const activeTabElement = document.querySelector(".tab.active");
        if (!activeTabElement) {
            showError("Cannot determine active input tab.");
            return;
        }
        const activeTab = activeTabElement.getAttribute("data-tab");
        console.log("Active tab:", activeTab);

        // --- Validate input and prepare FormData ---
        const formData = new FormData();
        let hasInput = false;

        if (activeTab === "file") {
            if (elements.fileInput.files.length > 0 && elements.fileInput.files[0].type.startsWith('image/')) {
                formData.append("image_file", elements.fileInput.files[0]);
                hasInput = true;
                console.log("Using file input:", elements.fileInput.files[0].name);
            } else if (elements.fileInput.files.length > 0) {
                 showError("Invalid file type selected. Please choose an image.");
            } else {
                showError("Please select an image file.");
            }
        } else if (activeTab === "url") {
            const url = elements.urlInput.value.trim();
            if (url && isValidUrl(url)) {
                 if (/\.(jpg|jpeg|png|gif|webp|bmp)(\?.*)?$/i.test(url)) {
                    formData.append("image_url", url);
                    hasInput = true;
                    console.log("Using URL input:", url);
                } else {
                     showError('URL doesn\'t look like a direct image link (.jpg, .png, etc).');
                 }
            } else if (url) {
                showError("Invalid URL format.");
            } else {
                showError("Please enter a valid image URL.");
            }
        }

        if (!hasInput) {
            console.warn("Submission blocked: No valid input provided.");
            return;
        }

        // --- UI updates: Show loading, hide results, disable button ---
        // ***** CHANGE HERE: Hide #results-container *****
        elements.submitButton.disabled = true;
        elements.resultsContainer.style.display = "none"; // Hide previous results
        elements.resultsContainer.style.opacity = "0";
        elements.loadingContainer.style.display = "block";

        // Animate loading entrance
        elements.loadingContainer.style.opacity = "0";
        elements.loadingContainer.style.transform = "translateY(20px)";
        requestAnimationFrame(() => {
            elements.loadingContainer.style.transition = "opacity 0.3s ease-out, transform 0.3s ease-out";
            elements.loadingContainer.style.opacity = "1";
            elements.loadingContainer.style.transform = "translateY(0)";
        });

        // --- API Call ---
        try {
            console.log("Sending request to /predict");
            const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
            console.log(`Response Status: ${response.status}`);

            // --- Handle Response ---
            if (!response.ok) {
                let errorDetail = `Request failed: ${response.status} ${response.statusText}`;
                try {
                    const errorData = await response.json();
                    errorDetail = errorData.detail || errorDetail;
                    console.error("Parsed error response:", errorData);
                } catch (jsonError) {
                    console.warn("Could not parse error response as JSON.");
                    const textError = await response.text().catch(() => '');
                    if (textError) errorDetail += `\nResponse: ${textError.substring(0,100)}...`;
                }
                throw new Error(errorDetail);
            }

            const result = await response.json();

            // --- Process API response and update UI ---
            processApiResponse(result);

            // --- Hide loading, Show results ---
            elements.loadingContainer.style.transition = "opacity 0.3s ease-out, transform 0.3s ease-out";
            elements.loadingContainer.style.opacity = "0";
            elements.loadingContainer.style.transform = "translateY(-20px)";

            setTimeout(() => {
                elements.loadingContainer.style.display = "none";
                // ***** CHANGE HERE: Show #results-container *****
                elements.resultsContainer.style.display = "block"; // Show results container

                // Animate results entrance
                elements.resultsContainer.style.opacity = "0";
                elements.resultsContainer.style.transform = "translateY(20px)";
                requestAnimationFrame(() => {
                    // Use existing CSS animation 'slideUp' or define a JS one
                    // Assuming 'slideUp' exists in CSS targeting .results-container
                    elements.resultsContainer.style.transition = "opacity 0.5s ease-out, transform 0.5s ease-out"; // Fallback if keyframes missing
                    elements.resultsContainer.style.opacity = "1";
                    elements.resultsContainer.style.transform = "translateY(0)";
                });
                console.log("Results displayed.");
            }, 300);

        } catch (error) {
            console.error("Error during submission/fetch:", error);
            showError(`Failed to get style advice: ${error.message}`);

            // Hide loading on error
            elements.loadingContainer.style.transition = "opacity 0.3s ease-out";
            elements.loadingContainer.style.opacity = "0";
            setTimeout(() => {
                elements.loadingContainer.style.display = "none";
                 // Ensure results remain hidden on error
                 elements.resultsContainer.style.display = "none";
            }, 300);
        } finally {
            elements.submitButton.disabled = false;
            console.log("Submission process finished.");
        }
    });
}

// Handle drag and drop
function handleDragAndDrop() {
    // ... (Drag and drop logic remains the same) ...
    const dropzone = elements.dropzone;
    const fileInput = elements.fileInput;
    if (!dropzone || !fileInput) {
        console.warn("Drag and drop elements not found.");
        return;
    }
    console.log("Initializing Drag and Drop...");

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((eventName) => {
        dropzone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach((eventName) => {
        dropzone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach((eventName) => {
        dropzone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        if (dropzone) {
            dropzone.classList.add("highlight");
             // Use CSS :hover/:focus/:active/.highlight pseudo-classes primarily
             // Inline styles from original script removed for cleaner CSS control
        }
    }

    function unhighlight() {
        if (dropzone) {
             dropzone.classList.remove("highlight");
        }
    }

    dropzone.addEventListener("drop", handleDrop, false);

    function handleDrop(e) {
         console.log("Handler Started: handleDrop");
         unhighlight();
         const dt = e.dataTransfer;
         if (!dt) return;
         const files = dt.files;

        if (files && files.length > 0) {
            const file = files[0];
            if(file.type.startsWith('image/')){
                 console.log("Image file dropped:", file.name);
                fileInput.files = files;
                const event = new Event("change", { bubbles: true });
                fileInput.dispatchEvent(event);
                const fileTab = document.querySelector('.tab[data-tab="file"]');
                if (fileTab && !fileTab.classList.contains('active')) {
                    fileTab.click();
                }
                 hideError();
            } else {
                console.warn("Invalid file type dropped:", file.type);
                showError("Invalid file type dropped. Please drop an image.");
                clearInputsAndPreview();
            }
        } else {
             console.log("Drop event occurred but no files found.");
        }
    }
}

// Add micro-interactions (subtle effects)
function addMicroInteractions() {
    // ... (Micro-interactions logic remains the same) ...
     console.log("Adding micro-interactions...");
    [elements.fileInput, elements.urlInput].forEach(input => {
        if (!input) return;
        const parent = input.closest('.file-upload') || input.closest('.tab-content');

        input.addEventListener('focus', () => {
            input.style.outline = '2px solid rgba(107, 102, 255, 0.3)';
            input.style.outlineOffset = '2px';
             if (parent && input === elements.urlInput) {
                 // parent.style.transform = 'scale(1.01)'; // Removed scaling
             }
        });

        input.addEventListener('blur', () => {
            input.style.outline = '';
             if (parent && input === elements.urlInput) {
                // parent.style.transform = ''; // Removed scaling
            }
        });
    });
}


// Add CSS Keyframes for error animations if not already in style.css
function addErrorAnimationsCSS() {
    // ... (CSS keyframe injection logic remains the same) ...
    const styleSheet = document.styleSheets[0];
    if (!styleSheet) {
        console.warn("Cannot add error animations: No stylesheet found.");
        return;
    }
    try {
        let ruleExists = false;
        for (let i = 0; i < styleSheet.cssRules.length; i++) {
            if (styleSheet.cssRules[i].name === 'errorFadeIn') {
                ruleExists = true;
                break;
            }
        }
        if (!ruleExists) {
            styleSheet.insertRule(`
                @keyframes errorFadeIn {
                    from { opacity: 0; transform: translateY(-10px); }
                    to { opacity: 1; transform: translateY(0); }
                }`, styleSheet.cssRules.length);
            styleSheet.insertRule(`
                @keyframes errorFadeOut {
                    from { opacity: 1; transform: translateY(0); }
                    to { opacity: 0; transform: translateY(-10px); }
                }`, styleSheet.cssRules.length);
             console.log("Error animation keyframes added.");
         } else {
              console.log("Error animation keyframes already exist.");
         }
    } catch (e) {
        console.warn("Could not add/check error animation keyframes (might be CORS issue with stylesheet):", e);
    }
}


// Initialize everything when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
    console.log("DOM Content Loaded. Initializing application...");
    // Check crucial elements referenced in the UPDATED 'elements' object
    if (!elements.resultsContainer || !elements.analysisContent || !elements.submitButton || !elements.loadingContainer) {
        console.error("One or more essential DOM elements could not be found. Check IDs: results-container, analysis-content, submit-button, loading-container.");
        // Display a user-facing error maybe?
        // document.body.innerHTML = "<h1>Error: Application failed to load. Essential page elements are missing.</h1>";
         return; // Stop initialization if core elements are missing
    }
    addErrorAnimationsCSS();
    createParticles();
    initTabs();
    handleFileInput();
    handleUrlInput();
    handleSubmit();
    handleDragAndDrop();
    addMicroInteractions();
    console.log("Application Initialized.");
});