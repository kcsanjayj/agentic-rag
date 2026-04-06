class AGENTIC_RAG {
    constructor() {
        this.initializeElements();
        this.attachEventListeners();
        this.isProcessing = false;
        this.typingTimeout = null;
        this.apiBaseUrl = '/api/v1';
        this.initializeApp();
    }

    initializeElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.fileInput = document.getElementById('fileInput');
        this.fileUploadBtn = document.getElementById('fileUploadBtn');
        this.charCount = document.getElementById('charCount');
        this.newChatBtn = document.getElementById('newChatBtn');
        
        // Execution trace elements
        this.executionTracePanel = document.getElementById('executionTracePanel');
        this.traceContent = document.getElementById('traceContent');
        this.traceStats = document.getElementById('traceStats');
        this.toggleTraceBtn = document.getElementById('toggleTraceBtn');
        this.traceScore = document.getElementById('traceScore');
        this.traceIterations = document.getElementById('traceIterations');
        this.traceDocs = document.getElementById('traceDocs');
        this.traceTime = document.getElementById('traceTime');
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

        // Configuration modal event listeners
        const configBtn = document.getElementById('configBtn');
        const closeConfigBtn = document.getElementById('closeConfigBtn');
        const saveConfigBtn = document.getElementById('saveConfigBtn');
        const testConfigBtn = document.getElementById('testConfigBtn');
        const toggleKeyBtn = document.getElementById('toggleKeyBtn');
        const providerSelect = document.getElementById('providerSelect');
        const tempSlider = document.getElementById('tempSlider');
        const tempValue = document.getElementById('tempValue');

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

        // Toggle execution trace panel
        if (this.toggleTraceBtn) {
            this.toggleTraceBtn.addEventListener('click', () => this.toggleExecutionTrace());
        }
    }

    toggleExecutionTrace() {
        if (this.executionTracePanel) {
            this.executionTracePanel.classList.toggle('collapsed');
            const icon = this.toggleTraceBtn.querySelector('i');
            if (this.executionTracePanel.classList.contains('collapsed')) {
                icon.className = 'fas fa-chevron-right';
            } else {
                icon.className = 'fas fa-chevron-left';
            }
        }
    }

    updateExecutionTrace(data) {
        if (!this.traceContent || !this.traceStats) return;

        // Show stats section
        this.traceStats.style.display = 'block';

        // Update stats
        if (this.traceScore) this.traceScore.textContent = data.evaluation_score ? data.evaluation_score.toFixed(1) : '-';
        if (this.traceIterations) this.traceIterations.textContent = data.iterations || '-';
        if (this.traceDocs) this.traceDocs.textContent = data.retrieved_docs || '-';
        if (this.traceTime) this.traceTime.textContent = data.processing_time ? `${data.processing_time.toFixed(1)}s` : '-';

        // Build execution trace steps
        let traceHTML = '';
        const latencies = data.agent_latencies || {};
        
        // Step 1: Retrieval with latency
        const retrievalTime = latencies.retrieval ? `(${latencies.retrieval.toFixed(0)}ms)` : '';
        traceHTML += `
            <div class="trace-step">
                <div class="trace-step-icon">1</div>
                <div class="trace-step-content">
                    <div class="trace-step-title">Retrieval ${retrievalTime}</div>
                    <div class="trace-step-desc">${data.retrieved_docs || 0} documents fetched</div>
                </div>
            </div>
        `;

        // Step 2: Generation with latency
        const genTime = latencies.generation ? `(${latencies.generation.toFixed(0)}ms)` : '';
        traceHTML += `
            <div class="trace-step">
                <div class="trace-step-icon">2</div>
                <div class="trace-step-content">
                    <div class="trace-step-title">Generation ${genTime}</div>
                    <div class="trace-step-desc">Initial answer created</div>
                </div>
            </div>
        `;

        // Step 3: Evaluation/Iterations with latency
        const iterations = data.iterations || 1;
        const criticTime = latencies.critic ? `(${latencies.critic.toFixed(0)}ms)` : '';
        if (iterations > 1) {
            traceHTML += `
                <div class="trace-step warning">
                    <div class="trace-step-icon">3</div>
                    <div class="trace-step-content">
                        <div class="trace-step-title">Evaluation ${criticTime}</div>
                        <div class="trace-step-desc">Weak answer, ${iterations - 1} refinement(s)</div>
                    </div>
                </div>
            `;
            
            // Show retry reason if available
            if (data.retry_reason) {
                traceHTML += `
                    <div class="trace-retry-reason">
                        <div class="retry-title">🔍 Why retry happened:</div>
                        <div class="retry-detail">Score: ${data.retry_reason.score}/10</div>
                        <div class="retry-detail">Issue: ${data.retry_reason.issue}</div>
                        <div class="retry-detail">Missing: ${data.retry_reason.missing ? data.retry_reason.missing.join(', ') : 'N/A'}</div>
                    </div>
                `;
            }
        } else {
            traceHTML += `
                <div class="trace-step success">
                    <div class="trace-step-icon">3</div>
                    <div class="trace-step-content">
                        <div class="trace-step-title">Evaluation ${criticTime}</div>
                        <div class="trace-step-desc">Passed first attempt</div>
                    </div>
                </div>
            `;
        }

        // Step 4: Critic Validation
        traceHTML += `
            <div class="trace-step success">
                <div class="trace-step-icon">4</div>
                <div class="trace-step-content">
                    <div class="trace-step-title">Critic Validation</div>
                    <div class="trace-step-desc">✓ Answer validated</div>
                </div>
            </div>
        `;

        // Sources section (clickable)
        if (data.sources && data.sources.length > 0) {
            traceHTML += `
                <div class="trace-sources">
                    <div class="sources-title">📚 Sources:</div>
            `;
            data.sources.forEach((source, index) => {
                traceHTML += `
                    <a href="#" class="source-link" data-source-id="${source.id || index}">
                        [${index + 1}] ${source.filename || 'document'} (chunk ${source.chunk_count || index + 1})
                    </a>
                `;
            });
            traceHTML += `</div>`;
        }

        this.traceContent.innerHTML = traceHTML;
        
        // Add click handlers for source links
        this.traceContent.querySelectorAll('.source-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const sourceId = e.target.dataset.sourceId;
                this.showSourceDetails(sourceId, data.sources[sourceId]);
            });
        });
    }
    
    showSourceDetails(index, source) {
        // Could show a modal or expand the source details
        console.log(`Source ${index}:`, source);
        // For now, just alert - could be enhanced to show a modal
        alert(`Source [${parseInt(index) + 1}]: ${source.filename || 'Document'}\nChunks: ${source.chunk_count || 'N/A'}`);
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
            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
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
            
            // Update execution trace panel
            this.updateExecutionTrace(data);
            
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
            
            contentDiv.innerHTML = `<p style="white-space: pre-wrap;">${this.escapeHtml(fullResponse)}</p>`;
        } else {
            contentDiv.innerHTML = `<p>${this.escapeHtml(content)}</p>`;
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
            // Backend connected successfully - no message shown
        } catch (error) {
            console.error('Backend health check failed:', error);
            this.addMessage('⚠️ Backend connection failed. The app will run in demo mode. Please check if the backend server is running on port 8000.', 'assistant');
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

            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
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

    startNewChat() {
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
            const response = await fetch(`${this.apiBaseUrl}/config`);
            const config = await response.json();
            
            // Set provider
            document.getElementById('providerSelect').value = config.provider || 'gemini';
            
            // Update model options based on provider
            this.updateModelOptions(config.provider);
            
            // Set temperature
            document.getElementById('tempSlider').value = config.temperature || 0.7;
            document.getElementById('tempValue').textContent = config.temperature || 0.7;
            
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    updateModelOptions(provider) {
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '<option value="">Auto-select</option>';
        
        const models = {
            gemini: ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro'],
            openai: ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o'],
            anthropic: ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229'],
            nvidia: ['meta/llama3-70b-instruct', 'meta/llama3-8b-instruct', 'mistralai/Mistral-7B-Instruct-v0.2'],
            groq: ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768'],
            huggingface: ['mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf', 'google/gemma-7b'],
            local: ['llama3.2:1b', 'llama3.2:3b', 'llama3.2:7b']
        };
        
        if (models[provider]) {
            models[provider].forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelSelect.appendChild(option);
            });
        }
    }

    async saveConfiguration() {
        const provider = document.getElementById('providerSelect').value;
        const apiKey = document.getElementById('apiKeyInput').value;
        const model = document.getElementById('modelSelect').value;
        const temperature = parseFloat(document.getElementById('tempSlider').value);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/config`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    provider,
                    api_key: apiKey,
                    model,
                    temperature
                })
            });
            
            if (response.ok) {
                this.addMessage('Configuration saved successfully! Restarting server...', 'assistant');
                this.hideConfigModal();
                
                // Restart server after 2 seconds
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                this.addMessage('Failed to save configuration', 'assistant');
            }
        } catch (error) {
            console.error('Save config error:', error);
            this.addMessage('Error saving configuration', 'assistant');
        }
    }

    async testConfiguration() {
        const provider = document.getElementById('providerSelect').value;
        const apiKey = document.getElementById('apiKeyInput').value;
        
        if (!apiKey) {
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
                },
                body: JSON.stringify({
                    provider,
                    api_key: apiKey
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addMessage(`✅ ${provider} API connection successful!`, 'assistant');
            } else {
                this.addMessage(`❌ ${provider} API connection failed: ${result.error}`, 'assistant');
            }
        } catch (error) {
            console.error('Test config error:', error);
            this.addMessage('Error testing API connection', 'assistant');
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
