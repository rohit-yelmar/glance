# Glance - Visual Semantic Search for Fashion E-commerce

> Intuitive product discovery through natural language and visual understanding

## Overview

Glance is a visual semantic search system designed for fashion e-commerce platforms. It solves the challenge of poor search relevance for descriptive, vibe-specific queries by combining textual metadata with AI-derived visual attributes.

Instead of relying on exact keyword matches, Glance understands queries like:
- "summer solid light blue linen shirt"
- "cozy oversized knit sweater"
- "minimalist black leather boots"

## How It Works

Glance operates through two main pipelines:

**Processing Pipeline (Indexing)**
- Extracts product metadata from your catalog
- Analyzes product images using vision AI to identify materials, colors, fit, and texture
- Generates embeddings from both text and visual data
- Stores everything in a vector database for fast semantic search

**Retrieval Pipeline (Search)**
- Encodes user queries into semantic embeddings
- Performs parallel searches across text and image modalities
- Merges results using Reciprocal Rank Fusion (RRF)
- Returns the most relevant products ranked by semantic similarity

## Technology Stack

- **Backend**: FastAPI (Python 3.10+)
- **Vector Database**: Qdrant (cloud-hosted)
- **Multimodal Model**: Jina AI CLIP v2 (free API)
- **Vision Model**: LLaVA or GPT-4o-mini (free tier)
- **Testing**: Hypothesis for property-based testing

## Project Status

🚧 **In Development** - Currently in the specification phase

This repository contains detailed requirements and design documents for the Glance system. Implementation is planned to follow a spec-driven development approach with comprehensive property-based testing.

## Documentation

- [Requirements Document](.kiro/specs/glance-visual-semantic-search/requirements.md) - Detailed functional requirements and acceptance criteria
- [Design Document](.kiro/specs/glance-visual-semantic-search/design.md) - System architecture, components, and correctness properties

## Key Features

- **Multimodal Search**: Combines text and visual understanding for better relevance
- **Natural Language Queries**: Search using descriptive, conversational language
- **Fast Performance**: Sub-2-second response times for 95% of queries
- **Cost-Effective**: Built on free-tier AI models and services
- **Robust Error Handling**: Comprehensive fault tolerance and retry logic
- **Property-Based Testing**: Formal correctness guarantees through executable properties

## Use Cases

- Fashion e-commerce platforms (Shopify, custom stores)
- Product discovery for apparel and accessories
- Visual search for style-based queries
- Semantic filtering and recommendations


[Add contribution guidelines here]
