class AGENTIC_RAG {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.isProcessing = false;
        this.typingTimeout = null;
        this.apiBaseUrl = window.location.hostname === 'localhost'
            ? 'http://localhost:8080/api/v1'  // Local testing (matches server port)
            : 'https://agentic-rag-production.up.railway.app/api/v1';  // Production
        this.hasDocument = false; // Track if user has uploaded a document
        // Load API key from sessionStorage (survives refresh, clears when tab closes)
        this.userApiKey = sessionStorage.getItem('userApiKey') || '';
        // Internal backend API key - must match Railway API_SECRET environment variable
        this.internalApiKey = window.API_SECRET || 'test-secret-12345678901234567890';
        this.initializeApp();
    }

    getProviderFallbackModels() {
        return {
            gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro'],
            openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
            anthropic: ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
            nvidia: ['meta/llama-3.1-70b-instruct', 'meta/llama-3.1-405b-instruct', 'meta/llama-3.1-8b-instruct', 'mistralai/mistral-large'],
            groq: ['llama-3.1-70b-versatile', 'llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
            huggingface: ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf', 'google/gemma-7b-it', 'tiiuae/falcon-7b-instruct'],
            local: ['llama3.1:8b', 'llama3:8b', 'mistral:7b', 'qwen2.5:7b', 'phi3:mini']
        };
    }

    initializeElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.fileInput = document.getElementById('fileInput');
        this.fileUploadBtn = document.getElementById('fileUploadBtn');
        this.charCount = document.getElementById('charCount');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.apiStatusBadge = document.getElementById('apiStatusBadge');
    }

    attachEventListeners() {
        if (this.sendBtn) this.sendBtn.addEventListener('click', () => this.sendMessage());
        if (this.messageInput) {
            this.messageInput.addEventListener('input', () => this.handleInputChange());
            this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        }
        if (this.fileUploadBtn && this.fileInput) {
            this.fileUploadBtn.addEventListener('click', () => this.fileInput.click());
            this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }
        if (this.newChatBtn) this.newChatBtn.addEventListener('click', () => this.startNewChat());

        // Search modal event listeners
        const searchFilesBtn = document.getElementById('searchFilesBtn');
        const closeSearchBtn = document.getElementById('closeSearchBtn');
        const searchInput = document.getElementById('searchInput');
        const searchModal = document.getElementById('searchModal');

        if (searchFilesBtn) searchFilesBtn.addEventListener('click', () => this.showSearchModal());
        if (closeSearchBtn) closeSearchBtn.addEventListener('click', () => this.hideSearchModal());
        if (searchInput) {
            searchInput.addEventListener('input', (e) => this.performSearch(e.target.value));
            searchInput.addEventListener('keydown', (e) => this.handleSearchKeydown(e));
        }

        // Close search modal when clicking outside
        if (searchModal) {
            searchModal.addEventListener('click', (e) => {
                if (e.target === searchModal) {
                    this.hideSearchModal();
                }
            });
        }

        // Configuration modal event listeners
        const configBtn = document.getElementById('configBtn');
        const closeConfigBtn = document.getElementById('closeConfigBtn');
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        const testConfigBtn = document.getElementById('testConfigBtn');
        const toggleKeyBtn = document.getElementById('toggleKeyBtn');
        const providerSelect = document.getElementById('providerSelect');
        const tempSlider = document.getElementById('tempSlider');
        const tempValue = document.getElementById('tempValue');
        const apiKeyInput = document.getElementById('apiKeyInput');

        if (configBtn) configBtn.addEventListener('click', () => this.showConfigModal());
        if (closeConfigBtn) closeConfigBtn.addEventListener('click', () => this.hideConfigModal());
        if (saveConfigBtn) saveConfigBtn.addEventListener('click', () => this.saveConfiguration());
        if (testConfigBtn) testConfigBtn.addEventListener('click', () => this.testConfiguration());
        if (toggleKeyBtn) toggleKeyBtn.addEventListener('click', () => this.toggleApiKeyVisibility());
        if (providerSelect) providerSelect.addEventListener('change', (e) => this.updateModelOptions(e.target.value));
        if (tempSlider) tempSlider.addEventListener('input', (e) => {
            tempValue.textContent = e.target.value;
        });

        // Close modal when clicking outside
        document.addEventListener('click', (e) => {
            const modal = document.getElementById('configModal');
            if (e.target === modal) {
                this.hideConfigModal();
            }
        });

        if (this.messageInput) this.messageInput.addEventListener('paste', (e) => this.handlePaste(e));
        
        // Add suggested question click handlers
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-chip')) {
                const question = e.target.getAttribute('data-question');
                if (question) {
                    this.messageInput.value = question;
                    this.handleInputChange();
                    this.sendMessage();
                }
            }
        });

    }

    handleInputChange() {
        const text = this.messageInput.value.trim();
        const charLength = this.messageInput.value.length;
        
        this.charCount.textContent = `${charLength}/2000`;
        this.sendBtn.disabled = text.length === 0 || this.isProcessing;

        this.autoResizeTextarea();
    }

    autoResizeTextarea() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;

        // Check if user has uploaded a document
        if (!this.hasDocument) {
            this.addMessage('Please upload a document first.', 'assistant');
            this.messageInput.value = '';
            this.handleInputChange();
            return;
        }

        // Store current question for display
        this.currentQuestion = message;
        
        this.addMessage(message, 'user');
        this.messageInput.value = '';
        this.handleInputChange();

        try {
            this.showTypingIndicator();
            await this.sendQueryToBackend(message);
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, something went wrong. Please try again.', 'assistant');
        } finally {
            this.hideTypingIndicator();
        }
    }

    async sendQueryToBackend(query) {
        try {
            // Check if user API key is set (from memory or sessionStorage)
            let apiKey = this.userApiKey || sessionStorage.getItem('userApiKey') || '';
            
            // If still no key, show warning
            if (!apiKey) {
                this.addMessage('⚠️ Please configure your AI provider API key in settings first.', 'assistant');
                this.isProcessing = false;
                this.sendBtn.disabled = false;
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey, // Internal backend auth
                    'X-User-Api-Key': apiKey // User's AI provider key
                },
                body: JSON.stringify({
                    query: query,
                    max_context: 5,
                    similarity_threshold: 0.7
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.lastResponseData = data; // Store for detailed display
            
            this.addMessage(data.response || data.answer || 'I processed your query but couldn\'t generate a response.', 'assistant');
            
        } catch (error) {
            console.error('Backend query error:', error);
            throw error;
        }
    }

    addMessage(content, sender, isUpload = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = sender === 'user' 
            ? '<i class="fas fa-user"></i>' 
            : '<i class="fas fa-robot"></i>';

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-wrapper';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // For assistant messages with structured content, add full system display
        if (sender === 'assistant' && this.lastResponseData) {
            const data = this.lastResponseData;
            
            // Build complete response with user question and system proof
            let fullResponse = '';
            
            // User question
            if (data.user_question) {
                fullResponse += `User: ${data.user_question}\n\n`;
            }
            
            // Main answer
            fullResponse += content;
            
            // Agent execution
            if (data.agent_execution) {
                fullResponse += '\n\n⚙️ Agent Execution:\n';
                fullResponse += `- Query Agent → ${data.agent_execution.query_agent.action}\n`;
                fullResponse += `- Retrieval Agent → ${data.agent_execution.retrieval_agent.action}\n`;
                fullResponse += `- Generation Agent → ${data.agent_execution.generation_agent.action}\n`;
                fullResponse += `- Validation Agent → ${data.agent_execution.validation_agent.action}`;
            }
            
            // Retrieved chunks
            if (data.sources && data.sources.length > 0) {
                fullResponse += '\n\n📚 Top Retrieved Chunks:\n';
                data.sources.forEach((source, index) => {
                    fullResponse += `${index + 1}. ${source.filename} (${source.chunk_id}) – ${source.similarity}\n`;
                });
            }
            
            const p = document.createElement('p');
            p.style.whiteSpace = 'pre-wrap';
            p.textContent = fullResponse;
            contentDiv.appendChild(p);
        } else {
            const p = document.createElement('p');
            p.textContent = content;
            contentDiv.appendChild(p);
        }
        
        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.getCurrentTime();

        if (sender === 'user') {
            contentWrapper.appendChild(contentDiv);
            contentWrapper.appendChild(timestampDiv);
            messageDiv.appendChild(contentWrapper);
            messageDiv.appendChild(avatarDiv);
        } else {
            messageDiv.appendChild(avatarDiv);
            contentWrapper.appendChild(contentDiv);
            contentWrapper.appendChild(timestampDiv);
            
            messageDiv.appendChild(contentWrapper);
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString('en-US', { 
            hour: 'numeric', 
            minute: '2-digit',
            hour12: true 
        });
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant-message typing-message';
        typingDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-wrapper">
                <div class="message-content">
                    <div class="typing-indicator">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        const typingMessage = document.querySelector('.typing-message');
        if (typingMessage) {
            typingMessage.remove();
        }
    }

    async initializeApp() {
        try {
            await this.checkBackendHealth();
            await this.refreshConfigStatus();
            // Backend connected successfully - no message shown
        } catch (error) {
            console.error('Backend health check failed:', error);
            this.updateApiStatusBadge(false, 'API Inactive');
            this.addMessage('⚠️ Backend connection failed. The app will run in demo mode. Please check if the backend server is running at https://agentic-rag-production.up.railway.app', 'assistant');
        }
    }

    updateApiStatusBadge(isActive, text) {
        if (!this.apiStatusBadge) return;
        this.apiStatusBadge.classList.remove('api-active', 'api-inactive');
        this.apiStatusBadge.classList.add(isActive ? 'api-active' : 'api-inactive');
        this.apiStatusBadge.textContent = text || (isActive ? 'API Active' : 'API Inactive');
    }

    async refreshConfigStatus() {
        try {
            // Backward-compatible flow:
            // 1) Prefer new /config-status endpoint
            // 2) Fall back to /config on older backends
            let response = await fetch(`${this.apiBaseUrl}/config-status`, {
                headers: {
                    'X-API-Key': this.internalApiKey
                }
            });
            if (response.ok) {
                const status = await response.json();
                this.updateApiStatusBadge(!!status.active, status.status_text || (status.active ? 'API Active' : 'API Inactive'));
                return;
            }

            response = await fetch(`${this.apiBaseUrl}/config`, {
                headers: {
                    'X-API-Key': this.internalApiKey
                }
            });
            if (!response.ok) {
                this.updateApiStatusBadge(false, 'API Inactive');
                return;
            }

            const config = await response.json();
            if (!config.configured) {
                this.updateApiStatusBadge(false, 'API Inactive');
                return;
            }

            // Older backend compatibility: verify actual connectivity with /test-api.
            const provider = config.provider || 'gemini';
            const model = config.model || '';
            const apiKey = document.getElementById('apiKeyInput')?.value || '';
            if (provider !== 'local' && !apiKey) {
                this.updateApiStatusBadge(true, 'API Configured');
                return;
            }
            const testResp = await fetch(`${this.apiBaseUrl}/test-api`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey
                },
                body: JSON.stringify({
                    provider,
                    model,
                    api_key: apiKey
                })
            });
            const testResult = await testResp.json();
            if (testResult.success) {
                this.updateApiStatusBadge(true, 'API Active');
            } else {
                this.updateApiStatusBadge(false, 'API Inactive');
            }
        } catch (error) {
            console.error('Failed to refresh config status:', error);
            this.updateApiStatusBadge(false, 'API Inactive');
        }
    }

    async checkBackendHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }
            const health = await response.json();
            console.log('Backend health:', health);
            return health;
        } catch (error) {
            throw error;
        }
    }

    handleFileUpload(event) {
        const files = Array.from(event.target.files);
        if (files.length === 0) return;

        files.forEach(file => this.processFile(file));
        event.target.value = '';
    }

    async processFile(file) {
        // Create processing message
        const processingMessageId = this.addProcessingMessage(file.name, file.size);
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            // Get API key for upload - works with any provider (OpenAI, NVIDIA, Gemini, etc.)
            const apiKey = this.userApiKey || sessionStorage.getItem('userApiKey') || '';
            
            if (!apiKey) {
                this.updateMessageStatus(processingMessageId, 'error', 'Please configure API key in AI Config first');
                this.isProcessing = false;
                return;
            }

            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                headers: {
                    'X-API-Key': this.internalApiKey, // Internal backend auth
                    'X-User-Api-Key': apiKey // OpenAI key for embeddings
                },
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
            // Mark that user has uploaded a document
            this.hasDocument = true;
            
            // Update to success state and show assistant message in one function
            this.updateProcessingMessage(processingMessageId, 'success', file.name);
            
        } catch (error) {
            // Update to error state and show error message in one function
            console.error('Upload error:', error);
            this.updateProcessingMessage(processingMessageId, 'error');
        }
    }

    addProcessingMessage(filename, fileSize) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message user-message`;
        const messageId = `processing-${Date.now()}`;
        messageDiv.id = messageId;
        
        // Store file size for later use
        messageDiv.dataset.fileSize = fileSize;

        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-user"></i>';

        const contentWrapper = document.createElement('div');
        contentWrapper.className = 'message-wrapper';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <p>Processing: ${filename} (${this.formatFileSize(fileSize)})</p>
        `;

        // Add processing class to the entire message content
        contentDiv.classList.add('file-processing');

        // Add timestamp
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'message-timestamp';
        timestampDiv.textContent = this.getCurrentTime();

        contentWrapper.appendChild(contentDiv);
        contentWrapper.appendChild(timestampDiv);
        messageDiv.appendChild(contentWrapper);
        messageDiv.appendChild(avatarDiv);

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageId;
    }

    updateProcessingMessage(messageId, status, filename = null) {
        const messageElement = document.getElementById(messageId);
        if (!messageElement) return;

        const contentDiv = messageElement.querySelector('.message-content');
        if (!contentDiv) return;

        // Remove existing classes
        contentDiv.classList.remove('file-processing', 'file-success', 'file-error');
        
        if (status === 'success') {
            contentDiv.classList.add('file-success');
            
            // Update the same message to show "Processed" instead of creating new message
            const fileSize = this.formatFileSize(messageElement.dataset.fileSize || 0);
            contentDiv.innerHTML = `
                <p>✅ Processed: ${filename} (${fileSize})</p>
            `;
            
            // Show assistant message after a short delay
            setTimeout(() => {
                this.showTypingIndicator();
                setTimeout(() => {
                    this.hideTypingIndicator();
                    this.addMessage(`I've processed your document "${filename}" and it's now ready for queries. You can ask me questions about its content.`, 'assistant');
                }, 1000);
            }, 500);
        } else if (status === 'error') {
            contentDiv.classList.add('file-error');
            
            // Update the same message to show "Error"
            contentDiv.innerHTML = `
                <p>❌ Upload Failed</p>
            `;
            
            // Show error message after a short delay
            setTimeout(() => {
                this.addMessage(`Sorry, I encountered an error. Please try again.`, 'assistant');
            }, 500);
        }
    }

    handlePaste(event) {
        const items = event.clipboardData.items;
        for (let item of items) {
            if (item.type.indexOf('image') !== -1) {
                event.preventDefault();
                const file = item.getAsFile();
                if (file) {
                    this.processFile(file);
                }
                break;
            }
        }
    }

    async startNewChat() {
        // Clear runtime config when chat is refreshed
        try {
            const response = await fetch(`${this.apiBaseUrl}/clear-config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey
                },
            });
            if (response.ok) {
                console.log('✅ Runtime config cleared on chat refresh');
                this.updateApiStatusBadge(false, 'API Inactive');
            }
        } catch (error) {
            console.error('Failed to clear config on chat refresh:', error);
        }
        
        // Clear all messages except the initial greeting
        this.messagesContainer.innerHTML = `
            <div class="message assistant-message">
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-wrapper">
                    <div class="message-content">
                        <p>Hello! I'm your AI assistant. How can I help you with your documents today?</p>
                    </div>
                    <div class="message-timestamp">${this.getCurrentTime()}</div>
                </div>
            </div>
        `;
        
        // Clear input field
        this.messageInput.value = '';
        this.handleInputChange();
        
        // Scroll to top
        this.scrollToBottom();
        
        // Focus on input
        this.messageInput.focus();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }

    // Configuration Modal Methods
    showConfigModal() {
        const modal = document.getElementById('configModal');
        modal.style.display = 'block';
        this.loadCurrentConfig();
    }

    hideConfigModal() {
        const modal = document.getElementById('configModal');
        modal.style.display = 'none';
    }

    loadCurrentConfig() {
        // Load current configuration from backend
        this.fetchCurrentConfig();
    }

    async fetchCurrentConfig() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/config`, {
                headers: {
                    'X-API-Key': this.internalApiKey
                }
            });
            const config = await response.json();
            
            // Set provider
            document.getElementById('providerSelect').value = config.provider || 'gemini';
            
            // Update model options based on provider
            this.updateModelOptions(config.provider);
            
            // Set temperature
            const temp = config.temperature || 0.7;
            document.getElementById('tempSlider').value = temp;
            document.getElementById('tempValue').textContent = temp;
            
            // Show masked API key if configured
            const apiKeyInput = document.getElementById('apiKeyInput');
            if (config.has_api_key) {
                apiKeyInput.value = config.api_key || '••••••••';
                apiKeyInput.placeholder = 'API key configured (••••••••)';
            } else {
                apiKeyInput.value = '';
                apiKeyInput.placeholder = 'Enter your API key';
            }
            
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    async updateModelOptions(provider) {
        const modelSelect = document.getElementById('modelSelect');
        const apiKeyGroup = document.getElementById('apiKeyInput').parentElement;
        const apiKeyInput = document.getElementById('apiKeyInput');
        
        modelSelect.innerHTML = '<option value="">Auto-select from API (Recommended)</option>';
        
        // Hide API key for local/offline models
        const isLocal = ['phi35', 'local-qwen3', 'local'].includes(provider);
        apiKeyGroup.style.display = isLocal ? 'none' : 'block';
        
        // Add a text input option for custom model names
        const customOption = document.createElement('option');
        customOption.value = 'custom';
        customOption.textContent = 'Custom Model Name...';
        modelSelect.appendChild(customOption);
        
        // If API key is entered (or local provider selected), load available models dynamically
        if (isLocal || apiKeyInput.value) {
            modelSelect.innerHTML = '<option value="">Loading models...</option>';
            
            try {
                const response = await fetch(`${this.apiBaseUrl}/list-models`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-Key': this.internalApiKey
                    },
                    body: JSON.stringify({
                        provider: provider,
                        api_key: apiKeyInput.value
                    })
                });
                
                const result = await response.json();
                
                modelSelect.innerHTML = '<option value="">Auto-select from API (Recommended)</option>';
                modelSelect.appendChild(customOption);
                
                if (result.success && result.models && result.models.length > 0) {
                    result.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        if (model === result.first_model) {
                            option.textContent += ' (Auto-selected)';
                        }
                        modelSelect.appendChild(option);
                    });
                } else {
                    // Fallback to common suggestions if API fails
                    const fallbackModels = this.getProviderFallbackModels();
                    
                    if (fallbackModels[provider]) {
                        fallbackModels[provider].forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model + ' (Fallback)';
                            modelSelect.appendChild(option);
                        });
                    }
                }
            } catch (error) {
                console.error('Failed to load models:', error);
                modelSelect.innerHTML = '<option value="">Auto-select from API (Recommended)</option>';
                modelSelect.appendChild(customOption);
                
                // Fallback suggestions
                const fallbackModels = this.getProviderFallbackModels();
                
                if (fallbackModels[provider]) {
                    fallbackModels[provider].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model + ' (Fallback)';
                        modelSelect.appendChild(option);
                    });
                }
            }
        } else {
            // Show fallback suggestions if no API key yet
            const fallbackModels = this.getProviderFallbackModels();
            
            if (fallbackModels[provider]) {
                fallbackModels[provider].forEach(model => {
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model + ' (Enter API key to load all)';
                    modelSelect.appendChild(option);
                });
            }
        }
        
        // Handle custom model input
        modelSelect.addEventListener('change', function handleCustomModel(e) {
            if (e.target.value === 'custom') {
                const customModel = prompt('Enter custom model name:');
                if (customModel) {
                    const option = document.createElement('option');
                    option.value = customModel;
                    option.textContent = customModel;
                    option.selected = true;
                    modelSelect.appendChild(option);
                } else {
                    e.target.value = '';
                }
            }
            modelSelect.removeEventListener('change', handleCustomModel);
        });
    }

    async saveConfiguration() {
        const provider = document.getElementById('providerSelect').value;
        let apiKey = document.getElementById('apiKeyInput').value;
        const model = document.getElementById('modelSelect').value;
        const temperature = parseFloat(document.getElementById('tempSlider').value);
        
        // Check if API key is masked (contains • or ... or looks like a placeholder)
        const isMaskedKey = apiKey.includes('•') || apiKey.includes('...') || /^\*+$/.test(apiKey);
        
        // If masked, don't send the key - backend will use existing runtime config
        if (isMaskedKey) {
            apiKey = null;  // Signal to backend to keep existing key
        }
        
        try {
            // Save to runtime config only (no .env file needed)
            const response = await fetch(`${this.apiBaseUrl}/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey
                },
                body: JSON.stringify({
                    provider,
                    api_key: apiKey,
                    model,
                    temperature
                })
            });
            
            if (response.ok) {
                // Store API key in sessionStorage
            if (apiKey && apiKey !== 'null' && apiKey !== 'undefined') {
                this.userApiKey = apiKey;
                sessionStorage.setItem('userApiKey', apiKey);
            }
                
                // Close modal immediately - don't wait for API test
                this.hideConfigModal();
                this.addMessage(`✅ Configuration saved for ${provider}.`, 'assistant');

                // Test API connection in background (non-blocking)
                this.testApiInBackground(provider, apiKey, model);
            } else {
                this.updateApiStatusBadge(false, 'API Inactive');
                this.addMessage('❌ Failed to save configuration.', 'assistant');
            }
        } catch (error) {
            console.error('Save config error:', error);
            this.updateApiStatusBadge(false, 'API Inactive');
            this.addMessage('❌ Error saving configuration.', 'assistant');
        }
    }

    async testApiInBackground(provider, apiKey, model) {
        // Test API in background without blocking UI
        try {
            // If apiKey is null, fetch existing config first
            let testKey = apiKey;
            if (!testKey) {
                try {
                    const configResponse = await fetch(`${this.apiBaseUrl}/config`, {
                        headers: {
                            'X-API-Key': this.internalApiKey
                        }
                    });
                    const config = await configResponse.json();
                    if (config.has_api_key) {
                        // Key exists but is masked, backend will use it
                        testKey = 'existing';  // Signal that backend should use existing
                    }
                } catch (e) {
                    console.log('Could not fetch existing config for test');
                }
            }

            const testResponse = await fetch(`${this.apiBaseUrl}/test-api`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey
                },
                body: JSON.stringify({
                    provider,
                    api_key: testKey,
                    model
                })
            });
            const testResult = await testResponse.json();

            await this.refreshConfigStatus();

            if (testResult.success) {
                this.updateApiStatusBadge(true, 'API Active');
                this.addMessage(`✅ ${provider} API is now active.`, 'assistant');
            } else {
                this.updateApiStatusBadge(false, 'API Inactive');
                const reason = testResult.error || 'Connection test failed';
                this.addMessage(`⚠️ Configuration saved but API test failed: ${reason}`, 'assistant');
            }
        } catch (error) {
            console.error('Background API test error:', error);
        }
    }

    async testConfiguration() {
        const provider = document.getElementById('providerSelect').value;
        const apiKey = document.getElementById('apiKeyInput').value;
        const model = document.getElementById('modelSelect').value;
        
        if (!apiKey && provider !== 'local') {
            this.addMessage('Please enter an API key first', 'assistant');
            return;
        }
        
        try {
            // Show testing message
            this.addMessage(`Testing ${provider} API connection...`, 'assistant');
            
            const response = await fetch(`${this.apiBaseUrl}/test-api`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.internalApiKey
                },
                body: JSON.stringify({
                    provider,
                    api_key: apiKey,
                    model: model
                })
            });
            
            const result = await response.json();
            await this.refreshConfigStatus();
            
            if (result.success) {
                this.addMessage(`✅ ${provider} API connection successful!`, 'assistant');
            } else {
                // Show error in red color based on error type
                let errorMessage = `❌ ${provider} API connection failed: ${result.error}`;
                if (result.error_type === 'quota_exceeded') {
                    errorMessage = `<span style="color: #ff4444; font-weight: bold;">❌ QUOTA EXCEEDED: ${result.error}</span>`;
                } else if (result.error_type === 'connection_error') {
                    errorMessage = `<span style="color: #ff4444; font-weight: bold;">❌ CONNECTION ERROR: ${result.error}</span>`;
                } else if (result.error_type === 'model_not_found') {
                    errorMessage = `<span style="color: #ff6b6b; font-weight: bold;">❌ MODEL NOT FOUND: ${result.error}</span>`;
                } else if (result.error_type === 'invalid_api_key') {
                    errorMessage = `<span style="color: #ff6b6b; font-weight: bold;">❌ INVALID API KEY: ${result.error}</span>`;
                }
                this.addMessage(errorMessage, 'assistant');
            }
        } catch (error) {
            console.error('Test config error:', error);
            this.addMessage(`<span style="color: #ff4444; font-weight: bold;">❌ Error testing API connection</span>`, 'assistant');
        }
    }

    toggleApiKeyVisibility() {
        const apiKeyInput = document.getElementById('apiKeyInput');
        const toggleBtn = document.getElementById('toggleKeyBtn');

        if (apiKeyInput.type === 'password') {
            apiKeyInput.type = 'text';
            toggleBtn.innerHTML = '<i class="fas fa-eye-slash"></i>';
        } else {
            apiKeyInput.type = 'password';
            toggleBtn.innerHTML = '<i class="fas fa-eye"></i>';
        }
    }

    // Search Methods
    showSearchModal() {
        const modal = document.getElementById('searchModal');
        const searchInput = document.getElementById('searchInput');
        modal.style.display = 'block';
        searchInput.focus();
        this.searchMatches = [];
        this.currentMatchIndex = -1;
    }

    hideSearchModal() {
        const modal = document.getElementById('searchModal');
        const searchInput = document.getElementById('searchInput');
        modal.style.display = 'none';
        searchInput.value = '';
        this.clearSearchHighlights();
        this.searchMatches = [];
        this.currentMatchIndex = -1;
    }

    clearSearchHighlights() {
        const highlights = document.querySelectorAll('.search-highlight, .search-highlight-current');
        highlights.forEach(highlight => {
            const parent = highlight.parentNode;
            parent.replaceChild(document.createTextNode(highlight.textContent), highlight);
            parent.normalize();
        });
    }

    performSearch(query) {
        this.clearSearchHighlights();
        this.searchMatches = [];
        this.currentMatchIndex = -1;

        const resultsInfo = document.getElementById('searchResultsInfo');

        if (!query || query.trim() === '') {
            resultsInfo.textContent = '';
            return;
        }

        const messagesContainer = document.getElementById('messagesContainer');
        const messageContents = messagesContainer.querySelectorAll('.message-content');

        const searchRegex = new RegExp(this.escapeRegex(query), 'gi');

        messageContents.forEach((contentDiv, messageIndex) => {
            const walker = document.createTreeWalker(
                contentDiv,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                if (node.nodeValue.match(searchRegex)) {
                    textNodes.push(node);
                }
            }

            textNodes.forEach(textNode => {
                const text = textNode.nodeValue;
                const matches = [...text.matchAll(searchRegex)];

                if (matches.length > 0) {
                    const fragment = document.createDocumentFragment();
                    let lastIndex = 0;

                    matches.forEach((match, matchIndex) => {
                        // Add text before match
                        if (match.index > lastIndex) {
                            fragment.appendChild(document.createTextNode(text.substring(lastIndex, match.index)));
                        }

                        // Create highlighted span
                        const highlight = document.createElement('span');
                        highlight.className = 'search-highlight';
                        highlight.textContent = match[0];
                        fragment.appendChild(highlight);

                        // Store match reference
                        this.searchMatches.push({
                            element: highlight,
                            messageIndex: messageIndex
                        });

                        lastIndex = match.index + match[0].length;
                    });

                    // Add remaining text
                    if (lastIndex < text.length) {
                        fragment.appendChild(document.createTextNode(text.substring(lastIndex)));
                    }

                    textNode.parentNode.replaceChild(fragment, textNode);
                }
            });
        });

        if (this.searchMatches.length > 0) {
            resultsInfo.textContent = `${this.searchMatches.length} match${this.searchMatches.length !== 1 ? 'es' : ''} found. Press Enter to navigate.`;
            this.currentMatchIndex = 0;
            this.highlightCurrentMatch();
            this.scrollToCurrentMatch();
        } else {
            resultsInfo.textContent = 'No matches found';
        }
    }

    escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    handleSearchKeydown(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            if (this.searchMatches.length > 0) {
                this.currentMatchIndex = (this.currentMatchIndex + 1) % this.searchMatches.length;
                this.highlightCurrentMatch();
                this.scrollToCurrentMatch();
            }
        } else if (e.key === 'Escape') {
            this.hideSearchModal();
        }
    }

    highlightCurrentMatch() {
        // Remove current class from all highlights
        document.querySelectorAll('.search-highlight-current').forEach(el => {
            el.classList.remove('search-highlight-current');
        });

        // Add current class to current match
        if (this.searchMatches[this.currentMatchIndex]) {
            this.searchMatches[this.currentMatchIndex].element.classList.add('search-highlight-current');
        }
    }

    scrollToCurrentMatch() {
        if (this.searchMatches[this.currentMatchIndex]) {
            const matchElement = this.searchMatches[this.currentMatchIndex].element;
            matchElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new AGENTIC_RAG();
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        document.title = 'Chat Application';
    } else {
        document.title = '💬 New messages';
    }
});
