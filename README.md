---
license: apache-2.0
tags:
  - multimodal
  - audio
  - image
  - video
  - text
  - meena
  - inference
---
# Meena (Named inputs)

**Summary**
Meena is a multimodal inference model that accepts Audio, Images, Video and Text as *inputs* and returns **Text** as the single output.

---

## Capabilities & Limits

- **Inputs:** Audio, Images, Video, Text
- **Output:** Text

### Token limits
- **Input token limit:** 2,097,152
- **Output token limit:** 8,192

### Audio / Visual specs
- **Maximum images per prompt:** 7,200
- **Maximum video length:** 2 hours
- **Maximum audio length:** ~19 hours

### Features
- System instructions: Supported
- JSON mode: Supported
- JSON schema: Supported
- Adjustable safety settings: Supported
- Caching: Supported
- Tuning: **Not supported**
- Function calling: Supported
- Code execution: Supported
- Live API: Not supported

---

## Intended use & safety
- Only use datasets and content you are fully licensed to use.
- Remove all private or PII-containing data before training/fine-tuning.
- Set safety and usage instructions under **Model Card** and use GitHub Environments for manual approvals before public publishing.

## License
apache-2.0
