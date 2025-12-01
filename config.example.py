"""
Configuration Example for Vector Engine API

This file shows how to configure the Vector Engine API key.
The actual API key should be set via environment variable for security.

DO NOT hardcode API keys in your code!
"""

# ==========================================
# API Key Configuration (RECOMMENDED METHOD)
# ==========================================

# Set the environment variable before starting ComfyUI:
#
# Linux/macOS:
#   export VECTOR_ENGINE_API_KEY="sk-your-api-key-here"
#   python main.py
#
# Windows (CMD):
#   set VECTOR_ENGINE_API_KEY=sk-your-api-key-here
#   python main.py
#
# Windows (PowerShell):
#   $env:VECTOR_ENGINE_API_KEY="sk-your-api-key-here"
#   python main.py

# ==========================================
# Permanent Configuration
# ==========================================

# Linux/macOS - Add to ~/.bashrc or ~/.zshrc:
#   export VECTOR_ENGINE_API_KEY="sk-your-api-key-here"
#
# Windows - Use setx for permanent environment variable:
#   setx VECTOR_ENGINE_API_KEY "sk-your-api-key-here"

# ==========================================
# Docker Configuration
# ==========================================

# In docker-compose.yml:
#   environment:
#     - VECTOR_ENGINE_API_KEY=sk-your-api-key-here
#
# Or with docker run:
#   docker run -e VECTOR_ENGINE_API_KEY="sk-your-api-key-here" ...

# ==========================================
# API Configuration
# ==========================================

# API Endpoint (configured in the node, no need to change)
VECTOR_ENGINE_API_HOST = "api.vectorengine.ai"

# Default Model (configured in the node)
DEFAULT_MODEL = "gemini-3-pro-image-preview"

# Default Generation Settings
DEFAULT_SYSTEM_PROMPT = "You are an AI assistant skilled in generating images and editing pictures."
DEFAULT_ASPECT_RATIO = "1:1"
DEFAULT_IMAGE_SIZE = "1K"

# ==========================================
# Security Notes
# ==========================================

# ✓ DO: Use environment variables for API keys
# ✓ DO: Add sensitive files to .gitignore
# ✓ DO: Use secrets management in production

# ✗ DON'T: Hardcode API keys in code
# ✗ DON'T: Commit API keys to version control
# ✗ DON'T: Share API keys in public repositories
