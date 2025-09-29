#!/bin/bash

# Stable Diffusion API - cURL Examples
# Make sure the API service is running on localhost:8000

echo "=== Stable Diffusion API cURL Examples ==="
echo

# Check if service is running
echo "1. Health Check"
echo "Command: curl -X GET http://localhost:8000/health"
echo
curl -X GET http://localhost:8000/health | jq '.'
echo
echo "---"
echo

# Basic image generation
echo "2. Basic Image Generation"
echo "Command: curl -X POST http://localhost:8000/generate -H 'Content-Type: application/json' -d '{...}'"
echo
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "steps": 20,
    "width": 512,
    "height": 512
  }' | jq '.metadata'
echo
echo "---"
echo

# Image generation with custom parameters
echo "3. Custom Parameters"
echo "Command: curl with custom steps, dimensions, and seed"
echo
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute robot in a futuristic city",
    "steps": 30,
    "width": 768,
    "height": 512,
    "seed": 42
  }' | jq '.metadata'
echo
echo "---"
echo

# Save image to file
echo "4. Save Generated Image to File"
echo "Command: curl with output redirection to save base64 image"
echo
response=$(curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene lake with mountains",
    "steps": 25
  }')

# Extract base64 image data and save to file
echo "$response" | jq -r '.image_base64' | base64 -d > generated_image.png
echo "Image saved to: generated_image.png"
echo "Metadata:"
echo "$response" | jq '.metadata'
echo
echo "---"
echo

# Error handling examples
echo "5. Error Handling Examples"
echo

echo "5a. Invalid dimensions (not multiple of 64):"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test image",
    "width": 500,
    "height": 500
  }' | jq '.'
echo
echo "---"

echo "5b. Invalid steps (out of range):"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Test image",
    "steps": 100
  }' | jq '.'
echo
echo "---"

echo "5c. Empty prompt:"
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "",
    "steps": 20
  }' | jq '.'
echo
echo "---"
echo

# Batch processing example
echo "6. Batch Processing with Different Prompts"
echo

prompts=(
  "A red apple on a wooden table"
  "A blue butterfly on a flower"
  "A golden sunset over the ocean"
  "A cozy cabin in the woods"
  "A futuristic spaceship in space"
)

for i in "${!prompts[@]}"; do
  echo "Generating image $((i+1))/5: ${prompts[i]}"
  
  response=$(curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d "{
      \"prompt\": \"${prompts[i]}\",
      \"steps\": 15,
      \"seed\": $((i * 100))
    }")
  
  # Save image
  echo "$response" | jq -r '.image_base64' | base64 -d > "batch_image_$((i+1)).png"
  
  # Show metadata
  generation_time=$(echo "$response" | jq -r '.metadata.generation_time_seconds')
  seed=$(echo "$response" | jq -r '.metadata.seed')
  echo "  Generated in ${generation_time}s with seed ${seed}"
  echo "  Saved as: batch_image_$((i+1)).png"
  echo
done

echo "---"
echo

# Performance testing
echo "7. Performance Testing"
echo "Generating 3 images with timing information"
echo

for i in {1..3}; do
  echo "Test $i/3:"
  start_time=$(date +%s.%N)
  
  response=$(curl -s -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
      "prompt": "Performance test image",
      "steps": 20
    }')
  
  end_time=$(date +%s.%N)
  total_time=$(echo "$end_time - $start_time" | bc)
  generation_time=$(echo "$response" | jq -r '.metadata.generation_time_seconds')
  
  echo "  Total request time: ${total_time}s"
  echo "  Generation time: ${generation_time}s"
  echo "  Network overhead: $(echo "$total_time - $generation_time" | bc)s"
  echo
done

echo "=== Examples Complete ==="
echo
echo "Generated files:"
echo "- generated_image.png"
echo "- batch_image_1.png through batch_image_5.png"
echo
echo "To view images:"
echo "- Linux: xdg-open generated_image.png"
echo "- macOS: open generated_image.png"
echo "- Windows: start generated_image.png"