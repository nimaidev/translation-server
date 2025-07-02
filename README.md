# IndicTrans2 Language Server

A high-performance FastAPI-based translation server powered by [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) models for seamless translation between 22 scheduled Indian languages and English.

## 🌟 Features

- **Multilingual Support**: Translate between 12 major Indian languages and English
- **High-Quality Translations**: Powered by state-of-the-art IndicTrans2 models
- **REST API**: Simple HTTP API for easy integration
- **GPU Acceleration**: CUDA support for faster inference
- **Memory Optimization**: Efficient model loading and GPU memory management
- **Graceful Shutdown**: Proper cleanup of resources on server termination

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- At least 8GB GPU memory for optimal performance

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AI4Bharat/IndicTrans2
   cd IndicTrans2
   ```

2. **Install dependencies**
   ```bash
   # Install IndicTrans2 dependencies
   source install.sh
   
   # Install additional requirements for the server
   pip install fastapi uvicorn torch transformers
   ```

3. **Install IndicTransToolkit**
   ```bash
   cd huggingface_interface/IndicTransToolkit
   pip install -e .
   cd ../..
   ```

4. **Run the server**
   ```bash
   python lang_server.py
   ```

The server will start on `http://0.0.0.0:9000`

## 📋 Supported Languages

The server supports translation between the following languages:

| Language | Code | Script |
|----------|------|--------|
| English | `eng_Latn` | Latin |
| Bengali | `ben_Beng` | Bengali |
| Punjabi | `pan_Guru` | Gurmukhi |
| Assamese | `asm_Beng` | Bengali |
| Konkani | `gom_Deva` | Devanagari |
| Gujarati | `guj_Gujr` | Gujarati |
| Hindi | `hin_Deva` | Devanagari |
| Kannada | `kan_Knda` | Kannada |
| Malayalam | `mal_Mlym` | Malayalam |
| Odia | `ory_Orya` | Odia |
| Tamil | `tam_Taml` | Tamil |
| Telugu | `tel_Telu` | Telugu |

## 🔧 API Usage

### Translation Endpoint

**POST** `/language-server/translate`

#### Request Body

```json
{
    "input_sentence": "Hello, how are you?",
    "source_lan": "eng_Latn",
    "target_lang": "hin_Deva"
}
```

#### Response

```json
{
    "translation": "नमस्ते, आप कैसे हैं?"
}
```

#### Error Response

```json
{
    "message": "Not a valid dialect"
}
```

### Example Usage

#### cURL
```bash
curl -X POST "http://localhost:9000/language-server/translate" \
     -H "Content-Type: application/json" \
     -d '{
       "input_sentence": "Good morning!",
       "source_lan": "eng_Latn",
       "target_lang": "hin_Deva"
     }'
```

#### Python
```python
import requests

url = "http://localhost:9000/language-server/translate"
data = {
    "input_sentence": "Good morning!",
    "source_lan": "eng_Latn",
    "target_lang": "hin_Deva"
}

response = requests.post(url, json=data)
print(response.json())
```

#### JavaScript
```javascript
const response = await fetch('http://localhost:9000/language-server/translate', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        input_sentence: 'Good morning!',
        source_lan: 'eng_Latn',
        target_lang: 'hin_Deva'
    })
});

const result = await response.json();
console.log(result);
```

## ⚡ Performance Optimization

The server is optimized for production use with several performance features:

### Model Configuration
- **Distilled Models**: Uses 200M parameter distilled models for faster inference
- **Memory Efficient**: Automatic GPU memory cleanup after each request
- **Batch Processing**: Supports batch translation for multiple sentences

### Recommended Settings for Speed
To optimize performance, you can modify the following in `lang_server.py`:

```python
# Enable quantization for faster inference
quantization = "4-bit"  # or "8-bit"

# Reduce generation parameters for speed
max_length = 128        # Reduced from 256
num_beams = 1          # Greedy decoding for fastest results
```

## 🏗️ Architecture

The server uses a dual-model architecture:

1. **English → Indic Model**: `ai4bharat/indictrans2-en-indic-dist-200M`
2. **Indic → English Model**: `ai4bharat/indictrans2-indic-en-dist-200M`

The appropriate model is automatically selected based on the target language:
- If target is English (`eng_Latn`): Uses Indic→English model
- If target is any Indic language: Uses English→Indic model

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `4` | Batch size for translation |
| `DEVICE` | `cuda` | Device for model inference |

### Model Selection

You can switch between different model variants by modifying the checkpoint directories:

```python
# For base models (higher quality, slower)
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-1B"

# For distilled models (faster, good quality)
en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
```

## 🐳 Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3 python3-pip git
WORKDIR /app

# Clone and setup IndicTrans2
RUN git clone https://github.com/AI4Bharat/IndicTrans2 .
RUN source install.sh
RUN pip install fastapi uvicorn

# Install IndicTransToolkit
WORKDIR /app/huggingface_interface/IndicTransToolkit
RUN pip install -e .
WORKDIR /app

# Copy your server file
COPY lang_server.py .

# Expose port
EXPOSE 9000

# Run the server
CMD ["python3", "lang_server.py"]
```

Build and run:
```bash
docker build -t indictrans2-server .
docker run --gpus all -p 9000:9000 indictrans2-server
```

## 📊 Benchmarks

The IndicTrans2 models achieve state-of-the-art performance on various benchmarks:

- **FLORES-22**: Comprehensive evaluation across 22 languages
- **IN22**: New benchmark with 1024 sentences across multiple domains
- **chrF++**: Primary evaluation metric for translation quality

For detailed benchmark results, refer to the [IndicTrans2 paper](https://arxiv.org/abs/2305.16307).

## 🛠️ Development

### Running in Development Mode

```bash
# Install development dependencies
pip install fastapi[all] uvicorn[standard]

# Run with auto-reload
uvicorn lang_server:app --host 0.0.0.0 --port 9000 --reload
```

### Testing

```bash
# Test the translation endpoint
python -c "
import requests
response = requests.post('http://localhost:9000/language-server/translate', 
                        json={'input_sentence': 'Hello', 'source_lan': 'eng_Latn', 'target_lang': 'hin_Deva'})
print(response.json())
"
```

## 🚦 Production Deployment

### Using Gunicorn

```bash
pip install gunicorn

# Run with multiple workers
gunicorn lang_server:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:9000
```

### Nginx Configuration

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` in the code
   - Enable quantization: `quantization = "4-bit"`
   - Use smaller model variant

2. **Slow Performance**
   - Ensure GPU is available and being used
   - Enable quantization for faster inference
   - Reduce `max_length` and `num_beams` parameters

3. **Model Loading Issues**
   - Check internet connection for model downloading
   - Verify sufficient disk space (models are ~2GB each)
   - Ensure proper CUDA installation

### Monitoring

```python
# Add to your server for monitoring
import psutil
import GPUtil

@app.get("/health")
def health_check():
    gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
    return {
        "status": "healthy",
        "gpu_memory": f"{gpu.memoryUsed}/{gpu.memoryTotal}MB" if gpu else "No GPU",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent
    }
```

## 📄 License

This project uses the IndicTrans2 models which are released under the MIT License. See the [LICENSE](https://github.com/AI4Bharat/IndicTrans2/blob/main/LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicTrans2 models
- [Hugging Face](https://huggingface.co/) for model hosting and transformers library
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework

## 📚 Citation

If you use this server in your research, please cite the IndicTrans2 paper:

```bibtex
@article{gala2023indictrans,
    title={IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages},
    author={Jay Gala and Pranjal A Chitale and A K Raghavan and Varun Gumma and Sumanth Doddapaneni and Aswanth Kumar M and Janki Atul Nawale and Anupama Sujatha and Ratish Puduppully and Vivek Raghavan and Pratyush Kumar and Mitesh M Khapra and Raj Dabre and Anoop Kunchukuttan},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=vfT4YuzAYA},
}
```

## 🔗 Links

- [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2)
- [IndicTrans2 Paper](https://arxiv.org/abs/2305.16307)
- [AI4Bharat Website](https://ai4bharat.iitm.ac.in/)
- [Demo](https://models.ai4bharat.org/#/nmt/v2)
- [Colab Notebook](https://colab.research.google.com/github/AI4Bharat/IndicTrans2/blob/main/huggingface_interface/colab_inference.ipynb)
