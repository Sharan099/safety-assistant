---
title: Safety Copilot
emoji: 🛡️
colorFrom: blue
colorTo: teal
sdk: streamlit
sdk_version: "1.29.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🛡️ Safety Copilot - AI-Powered Automotive Safety Assistant

An AI-powered safety copilot that answers automotive safety questions using verified documents (UNECE regulations, Euro NCAP protocols, ISO 26262, etc.) with full source traceability.

## 🎯 Features

- **Trusted Knowledge Retrieval (RAG)**: Only uses verified safety documents
- **Explainable Answers**: Each answer includes document name, section, and page reference
- **Multiple Regulation Support**: UNECE R94, R127, R155, R156, Euro NCAP, ISO 26262, and more
- **Structured Output**: Clean, readable answers with proper formatting
- **Source Citations**: Direct links to source documents

## 🚀 Getting Started

1. **Initialize**: Click the "🚀 Initialize Safety Copilot" button in the sidebar
2. **Ask Questions**: Type your safety-related question in the chat
3. **Get Answers**: Receive structured, source-backed answers

## 📋 What You Can Ask

- **Definition Questions**: "What is HIC (Head Injury Criterion)?"
- **Requirement Questions**: "What is the maximum HIC value for UN R94?"
- **Comparison Questions**: "Compare UN R94 and Euro NCAP requirements"
- **Compliance Questions**: "Does HIC 850 meet UN R94 requirements?"

## 📚 Regulations Included

- UNECE Regulations (R16, R17, R29, R94, R127, R137, R152, R155, R156)
- EU Regulations (General Safety Regulation, Type Approval Directives)
- Euro NCAP Protocols
- Functional Safety (ISO 26262)
- ADAS Guidelines
- Passive Safety Standards

## ⚙️ Configuration

The app requires API keys to be set in Hugging Face Spaces secrets:
- `ANTHROPIC_API_KEY` - Required for Claude models
- `OPENAI_API_KEY` - Optional, for OpenAI models

## ⚠️ Disclaimer

This information is for decision support only. Always consult qualified safety engineers and follow your organization's safety processes.
