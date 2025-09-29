#!/usr/bin/env python3
"""
Python client example for Stable Diffusion API
"""

import requests
import base64
import json
from PIL import Image
from io import BytesIO
import time

class StableDiffusionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_image(self, prompt, steps=20, width=512, height=512, seed=None):
        """Generate an image from a text prompt"""
        
        payload = {
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def save_image(self, image_base64, filename):
        """Save base64 image to file"""
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        image.save(filename)
        return image
    
    def health_check(self):
        """Check service health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

def main():
    # Initialize client
    client = StableDiffusionClient()
    
    # Check service health
    print("Checking service health...")
    health = client.health_check()
    print(f"Service status: {health.get('status', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("Service is not healthy, exiting...")
        return
    
    # Example 1: Basic image generation
    print("\nExample 1: Basic image generation")
    try:
        result = client.generate_image(
            prompt="A serene lake surrounded by mountains at sunset",
            steps=25
        )
        
        # Save the image
        client.save_image(result["image_base64"], "example1_lake_sunset.png")
        print(f"Generated image in {result['metadata']['generation_time_seconds']:.2f} seconds")
        print(f"Used seed: {result['metadata']['seed']}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Reproducible generation with seed
    print("\nExample 2: Reproducible generation")
    seed = 42
    try:
        result1 = client.generate_image(
            prompt="A cute robot in a garden",
            seed=seed,
            steps=20
        )
        
        result2 = client.generate_image(
            prompt="A cute robot in a garden",
            seed=seed,
            steps=20
        )
        
        client.save_image(result1["image_base64"], "example2_robot_1.png")
        client.save_image(result2["image_base64"], "example2_robot_2.png")
        
        print("Generated two identical images with same seed")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Different dimensions
    print("\nExample 3: Different dimensions")
    try:
        result = client.generate_image(
            prompt="A wide landscape with rolling hills",
            width=768,
            height=512,
            steps=30
        )
        
        client.save_image(result["image_base64"], "example3_landscape.png")
        print(f"Generated {result['metadata']['width']}x{result['metadata']['height']} image")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Batch generation
    print("\nExample 4: Batch generation")
    prompts = [
        "A red apple on a wooden table",
        "A blue butterfly on a flower",
        "A golden sunset over the ocean"
    ]
    
    for i, prompt in enumerate(prompts):
        try:
            result = client.generate_image(prompt, steps=15)
            filename = f"example4_batch_{i+1}.png"
            client.save_image(result["image_base64"], filename)
            print(f"Generated: {filename}")
            
        except Exception as e:
            print(f"Error generating image {i+1}: {e}")

if __name__ == "__main__":
    main()