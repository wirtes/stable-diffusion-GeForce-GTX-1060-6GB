/**
 * JavaScript client example for Stable Diffusion API
 * Works in both Node.js and browser environments
 */

class StableDiffusionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async generateImage(options = {}) {
        const {
            prompt,
            steps = 20,
            width = 512,
            height = 512,
            seed = null
        } = options;

        if (!prompt) {
            throw new Error('Prompt is required');
        }

        const payload = {
            prompt,
            steps,
            width,
            height
        };

        if (seed !== null) {
            payload.seed = seed;
        }

        try {
            const response = await fetch(`${this.baseUrl}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(`API Error: ${errorData.error} - ${errorData.details || ''}`);
            }

            return await response.json();

        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error: Unable to connect to the API');
            }
            throw error;
        }
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return await response.json();
        } catch (error) {
            return { status: 'error', message: error.message };
        }
    }

    // Browser-specific method to display image
    displayImageInBrowser(imageBase64, containerId) {
        if (typeof document === 'undefined') {
            throw new Error('This method only works in browser environment');
        }

        const container = document.getElementById(containerId);
        if (!container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }

        const img = document.createElement('img');
        img.src = `data:image/png;base64,${imageBase64}`;
        img.style.maxWidth = '100%';
        img.style.height = 'auto';
        
        container.innerHTML = '';
        container.appendChild(img);
    }

    // Node.js-specific method to save image
    async saveImageToFile(imageBase64, filename) {
        if (typeof require === 'undefined') {
            throw new Error('This method only works in Node.js environment');
        }

        const fs = require('fs').promises;
        const buffer = Buffer.from(imageBase64, 'base64');
        await fs.writeFile(filename, buffer);
    }
}

// Example usage for browser
async function browserExample() {
    const client = new StableDiffusionClient();

    // Check health
    const health = await client.healthCheck();
    console.log('Service health:', health);

    if (health.status !== 'healthy') {
        console.error('Service is not healthy');
        return;
    }

    try {
        // Generate image
        const result = await client.generateImage({
            prompt: 'A magical forest with glowing mushrooms',
            steps: 25,
            width: 512,
            height: 512
        });

        console.log('Generation completed in', result.metadata.generation_time_seconds, 'seconds');
        
        // Display in browser (assumes there's a div with id 'image-container')
        client.displayImageInBrowser(result.image_base64, 'image-container');

    } catch (error) {
        console.error('Error:', error.message);
    }
}

// Example usage for Node.js
async function nodeExample() {
    const client = new StableDiffusionClient();

    try {
        // Example 1: Basic generation
        console.log('Generating basic image...');
        const result1 = await client.generateImage({
            prompt: 'A cozy cabin in the woods during winter'
        });

        await client.saveImageToFile(result1.image_base64, 'cabin_winter.png');
        console.log('Saved: cabin_winter.png');

        // Example 2: High quality with more steps
        console.log('Generating high quality image...');
        const result2 = await client.generateImage({
            prompt: 'A detailed portrait of a wise old wizard',
            steps: 40,
            width: 768,
            height: 768
        });

        await client.saveImageToFile(result2.image_base64, 'wizard_portrait.png');
        console.log('Saved: wizard_portrait.png');

        // Example 3: Reproducible with seed
        console.log('Generating reproducible image...');
        const result3 = await client.generateImage({
            prompt: 'A steampunk airship flying through clouds',
            seed: 123456,
            steps: 30
        });

        await client.saveImageToFile(result3.image_base64, 'steampunk_airship.png');
        console.log('Saved: steampunk_airship.png');
        console.log('Seed used:', result3.metadata.seed);

    } catch (error) {
        console.error('Error:', error.message);
    }
}

// HTML example for browser usage
const htmlExample = `
<!DOCTYPE html>
<html>
<head>
    <title>Stable Diffusion API Example</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, button { width: 100%; padding: 8px; box-sizing: border-box; }
        button { background-color: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        #image-container { margin-top: 20px; text-align: center; }
        #status { margin-top: 10px; padding: 10px; border-radius: 4px; }
        .success { background-color: #d4edda; color: #155724; }
        .error { background-color: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>Stable Diffusion Image Generator</h1>
    
    <form id="generate-form">
        <div class="form-group">
            <label for="prompt">Prompt:</label>
            <textarea id="prompt" rows="3" placeholder="Describe the image you want to generate..."></textarea>
        </div>
        
        <div class="form-group">
            <label for="steps">Steps (1-50):</label>
            <input type="number" id="steps" min="1" max="50" value="20">
        </div>
        
        <div class="form-group">
            <label for="width">Width:</label>
            <select id="width">
                <option value="256">256px</option>
                <option value="320">320px</option>
                <option value="384">384px</option>
                <option value="448">448px</option>
                <option value="512" selected>512px</option>
                <option value="576">576px</option>
                <option value="640">640px</option>
                <option value="704">704px</option>
                <option value="768">768px</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="height">Height:</label>
            <select id="height">
                <option value="256">256px</option>
                <option value="320">320px</option>
                <option value="384">384px</option>
                <option value="448">448px</option>
                <option value="512" selected>512px</option>
                <option value="576">576px</option>
                <option value="640">640px</option>
                <option value="704">704px</option>
                <option value="768">768px</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="seed">Seed (optional):</label>
            <input type="number" id="seed" placeholder="Leave empty for random">
        </div>
        
        <button type="submit" id="generate-btn">Generate Image</button>
    </form>
    
    <div id="status"></div>
    <div id="image-container"></div>

    <script>
        // Include the StableDiffusionClient class here
        ${StableDiffusionClient.toString()}

        const client = new StableDiffusionClient();
        const form = document.getElementById('generate-form');
        const generateBtn = document.getElementById('generate-btn');
        const status = document.getElementById('status');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                showStatus('Please enter a prompt', 'error');
                return;
            }

            const options = {
                prompt: prompt,
                steps: parseInt(document.getElementById('steps').value),
                width: parseInt(document.getElementById('width').value),
                height: parseInt(document.getElementById('height').value)
            };

            const seedValue = document.getElementById('seed').value.trim();
            if (seedValue) {
                options.seed = parseInt(seedValue);
            }

            generateBtn.disabled = true;
            generateBtn.textContent = 'Generating...';
            showStatus('Generating image, please wait...', 'success');

            try {
                const result = await client.generateImage(options);
                client.displayImageInBrowser(result.image_base64, 'image-container');
                showStatus(\`Image generated in \${result.metadata.generation_time_seconds.toFixed(2)} seconds\`, 'success');
            } catch (error) {
                showStatus(\`Error: \${error.message}\`, 'error');
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = 'Generate Image';
            }
        });

        function showStatus(message, type) {
            status.textContent = message;
            status.className = type;
        }

        // Check service health on page load
        client.healthCheck().then(health => {
            if (health.status === 'healthy') {
                showStatus('Service is ready', 'success');
            } else {
                showStatus('Service is not available', 'error');
                generateBtn.disabled = true;
            }
        });
    </script>
</body>
</html>
`;

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = { StableDiffusionClient, nodeExample };
} else if (typeof window !== 'undefined') {
    // Browser
    window.StableDiffusionClient = StableDiffusionClient;
    window.browserExample = browserExample;
}

// Run Node.js example if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    nodeExample();
}