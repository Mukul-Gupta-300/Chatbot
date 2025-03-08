// chat-integration.js
document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const generateBtn = document.getElementById('generateReplyBtn');
    const outputCards = document.getElementById('outputCards');
    const extractedText = document.getElementById('extractedText');
    const styleBtns = document.querySelectorAll('.style-btn');
    const viewSavedBtn = document.getElementById('viewSavedBtn');
    const regenerateBtn = document.getElementById('regenerateBtn');
    
    let selectedStyle = 'flirty';
    let uploadedImage = null;
    let isProcessing = false;
    
    // Style selection handler
    styleBtns.forEach(button => {
        button.addEventListener('click', (e) => {
            // Reset all buttons
            styleBtns.forEach(btn => {
                btn.style.backgroundColor = '#F4F4F4';
                btn.style.color = '#333';
            });
            
            // Highlight selected button
            e.target.style.backgroundColor = '#FFE8FA';
            e.target.style.color = '#6700A8';
            selectedStyle = e.target.textContent.toLowerCase();
            console.log('Selected style:', selectedStyle);
        });
    });
    
    // Initialize file upload (drag & drop + click)
    function initializeFileUpload() {
        // Drag and drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop zone on drag over
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.style.background = '#FFE8FA';
            dropZone.innerHTML = '<p>Drop to Upload!</p>';
        }
        
        function unhighlight() {
            dropZone.style.background = '';
            dropZone.innerHTML = '<p>Drag & Drop or Click to Upload</p>';
        }
        
        // Handle the drop
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            handleFiles(files);
        }
        
        // Handle file input change
        fileInput.addEventListener('change', function() {
            handleFiles(this.files);
        });
        
        // Click on drop zone to trigger file input
        dropZone.addEventListener('click', function() {
            fileInput.click();
        });
        
        function handleFiles(files) {
            if (files.length > 0) {
                const file = files[0];
                if (file.type.match('image.*')) {
                    uploadedImage = file;
                    processImage(file);
                } else {
                    alert('Please upload an image file.');
                }
            }
        }
    }
    
    // Process uploaded image and extract text
    function processImage(file) {
        // Show loading indicator
        dropZone.innerHTML = '<p>Processing image...</p>';
        dropZone.style.background = '#f0f0f0';
        
        // Create FormData and append file
        const formData = new FormData();
        formData.append('image', file);
        
        // Send to backend for OCR processing
        fetch('/api/extract-text', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update UI with extracted text
                extractedText.value = data.messages.join('\n');
                dropZone.innerHTML = '<p>Image processed successfully!</p>';
                dropZone.style.background = '#e6ffe6';
                
                // Show thumbnail of uploaded image
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.style.maxHeight = '60px';
                img.style.maxWidth = '100%';
                img.style.marginTop = '10px';
                dropZone.appendChild(img);
            } else {
                throw new Error(data.error || 'Failed to process image');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            dropZone.innerHTML = '<p>Error processing image. Try again.</p>';
            dropZone.style.background = '#ffe6e6';
            
            // Fallback: Allow manual input
            alert('Image processing failed. You can manually enter your chat text.');
        });
    }
    
    // Generate reply button handler
    // Generate reply button handler
function initializeGenerateButton() {
    generateBtn.addEventListener('click', function() {
        if (isProcessing) return;
        
        const chatText = extractedText.value.trim();
        if (!chatText) {
            alert('Please upload a chat image or enter chat text manually.');
            return;
        }
        
        // Show loading state
        isProcessing = true;
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span>⏳</span> Generating...';
        outputCards.innerHTML = '<div class="output-card">Generating response...</div>';
        
        // Prepare request data
        const requestData = {
            chat_text: chatText,
            style: selectedStyle,
            // Removed num_responses since we're only generating one
        };
        
        // Send to backend API
        fetch('/api/generate-responses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResponses(data.responses); // Still works with a single-item array
            } else {
                throw new Error(data.error || 'Failed to generate response');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            outputCards.innerHTML = '<div class="output-card">Error generating response. Please try again.</div>';
        })
        .finally(() => {
            // Reset UI state
            isProcessing = false;
            generateBtn.disabled = false;
            generateBtn.innerHTML = '<span>🔥</span> Generate';
        });
    });
}
    
    // Display generated responses
    function displayResponses(responses) {
        outputCards.innerHTML = '';
        
        responses.forEach((response, index) => {
            const card = document.createElement('div');
            card.className = 'output-card';
            
            // Response text
            const responseText = document.createElement('div');
            responseText.className = 'response-text';
            responseText.textContent = response;
            
            // Action buttons
            const actionButtons = document.createElement('div');
            actionButtons.className = 'action-buttons';
            actionButtons.style.marginTop = '10px';
            actionButtons.style.display = 'flex';
            actionButtons.style.justifyContent = 'flex-end';
            actionButtons.style.gap = '10px';
            
            // Copy button
            const copyBtn = document.createElement('button');
            copyBtn.textContent = 'Copy';
            copyBtn.style.backgroundColor = '#FFB3F0';
            copyBtn.style.border = 'none';
            copyBtn.style.color = '#6700A8';
            copyBtn.style.padding = '5px 10px';
            copyBtn.style.borderRadius = '4px';
            copyBtn.style.cursor = 'pointer';
            copyBtn.onclick = () => {
                navigator.clipboard.writeText(response);
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                }, 2000);
            };
            
            // Save button
            const saveBtn = document.createElement('button');
            saveBtn.textContent = 'Save';
            saveBtn.style.backgroundColor = '#6700A8';
            saveBtn.style.border = 'none';
            saveBtn.style.color = 'white';
            saveBtn.style.padding = '5px 10px';
            saveBtn.style.borderRadius = '4px';
            saveBtn.style.cursor = 'pointer';
            saveBtn.onclick = () => {
                // Save response to server
                fetch('/api/save-response', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ response })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        saveBtn.textContent = 'Saved!';
                        setTimeout(() => {
                            saveBtn.textContent = 'Save';
                        }, 2000);
                    }
                })
                .catch(error => {
                    console.error('Error saving response:', error);
                    saveBtn.textContent = 'Error';
                    setTimeout(() => {
                        saveBtn.textContent = 'Save';
                    }, 2000);
                });
            };
            
            // Add buttons to action container
            actionButtons.appendChild(copyBtn);
            actionButtons.appendChild(saveBtn);
            
            // Add all elements to card
            card.appendChild(responseText);
            card.appendChild(actionButtons);
            
            // Add card to output
            outputCards.appendChild(card);
            
            const feedbackDiv = document.createElement('div');
        feedbackDiv.className = 'feedback-buttons';
        feedbackDiv.style.marginTop = '10px';
        
        const thumbsUp = document.createElement('button');
        thumbsUp.innerHTML = '👍';
        thumbsUp.className = 'feedback-btn';
        thumbsUp.onclick = () => submitFeedback(response, 'positive');
        
        const thumbsDown = document.createElement('button');
        thumbsDown.innerHTML = '👎';
        thumbsDown.className = 'feedback-btn';
        thumbsDown.onclick = () => submitFeedback(response, 'negative');
        
        feedbackDiv.appendChild(thumbsUp);
        feedbackDiv.appendChild(thumbsDown);
        
        card.appendChild(feedbackDiv);
            
        });
    }

    function submitFeedback(response, feedbackType) {
        fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                response: response,
                feedback: feedbackType
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Thank you for your feedback!');
            }
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
        });
    }
    
    // Initialize all components
    initializeFileUpload();
    initializeGenerateButton();
    
    // Set default selected style
    styleBtns[0].click();
    
    // Example link
    document.querySelector('.view-example').addEventListener('click', function(e) {
        e.preventDefault();
        
        // Load example conversation
        extractedText.value = `Hey there! I noticed we have similar taste in music. What's your favorite band?
        
I'm really into indie rock, especially bands like Arctic Monkeys and The Strokes. What about you?
        
Oh nice! I love Arctic Monkeys too. Have you been to any good concerts lately?`;
        
        // Auto-scroll to the textarea
        extractedText.scrollIntoView({ behavior: 'smooth' });
    });
    
    // View saved responses
    viewSavedBtn.addEventListener('click', function() {
        fetch('/api/saved-responses')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.responses.length > 0) {
                    displayResponses(data.responses);
                } else {
                    outputCards.innerHTML = '<div class="output-card">No saved responses found.</div>';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                outputCards.innerHTML = '<div class="output-card">Error loading saved responses.</div>';
            });
    });
    
    // Regenerate button
    regenerateBtn.addEventListener('click', function() {
        // Trigger generate button
        generateBtn.click();
    });
});