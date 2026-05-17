# Sequential Visual Feature Refinement (SVFR)

A memory-efficient, training-free Video Question Answering (VideoQA) framework designed for long-video understanding on consumer-grade GPUs.

---

# 📌 Overview

State-of-the-art Video Question Answering systems often require extremely high GPU memory to process long videos with dense temporal sampling. This creates a major limitation for researchers and developers working on consumer hardware.

**Sequential Visual Feature Refinement (SVFR)** addresses this challenge through a lightweight, training-free inference strategy that enables efficient long-video understanding while maintaining a stable VRAM footprint.

SVFR processes videos sequentially in small chunks, selects only the most relevant visual features using question-aware relevance scoring, and maintains a compact evolving memory bank for inference.

The framework is designed to work efficiently with modern Video-LLMs such as LLaVA-OneVision while remaining feasible on hardware such as RTX 3050 / RTX 4090 GPUs.

---

# 🚀 Key Features

- ✅ Training-free inference framework
- ✅ Memory-efficient long-video processing
- ✅ Sequential chunk-based feature refinement
- ✅ Dynamic relevance-based feature selection
- ✅ Fixed-size evolving memory bank
- ✅ 8-bit quantization support using `bitsandbytes`
- ✅ Multiple Choice VideoQA evaluation support
- ✅ Open-ended VideoQA generation support
- ✅ Length-bias corrected MCQ scoring
- ✅ Consumer GPU friendly

---

# 🧠 Core Idea

Instead of processing the entire video simultaneously, SVFR:

1. Splits videos into manageable chunks
2. Extracts visual embeddings sequentially
3. Computes relevance scores between:
   - visual features
   - question embeddings
4. Retains only the top-k most relevant features
5. Updates a compact memory bank over time

This significantly reduces memory usage while preserving temporally important information.

---

# 🏗️ Architecture Pipeline

```text
Video Stream
     ↓
Frame Chunking
     ↓
Visual Feature Extraction
     ↓
Question-Feature Relevance Scoring
     ↓
Top-K Feature Selection
     ↓
Memory Bank Update
     ↓
VideoQA Inference
```

---

# 📂 Repository Structure

```text
SVFR/
│
├── data/
│   └── cgbench/
│       ├── filtered_cgbench_rekv_stream_relative.json
│       └── videos/
│
├── video_qa/
│   ├── model_loader.py
│   ├── video_llm_wrapper.py
│   ├── svfr_stream_vqa.py
│   └── svfr_mcq_vqa.py
│
├── results/
│
├── run_evaluation.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Saniya-712006/SVFR.git
cd SVFR
```

---

## 2️⃣ Create Environment

Using Conda:

```bash
conda create -n svfr python=3.11
conda activate svfr
```

---

## 3️⃣ Install Dependencies

```bash
pip install torch transformers accelerate logzero
```

---

## 4️⃣ Install Quantization Support

```bash
pip install bitsandbytes
```

---

# 📊 Dataset Structure

Expected dataset layout:

```text
data/
└── your_dataset/
    ├── your_dataset_annotations_file.json
    └── videos/
        ├── video1.mp4
        ├── video2.mp4
        └── ...
```

---

# 🚀 Running Evaluations

The evaluation pipeline is controlled using:

```bash
run_evaluation.py
```

The framework supports:

- Multiple Choice VideoQA
- Open-Ended VideoQA

---

# 📝 Multiple Choice Evaluation

```bash
python run_evaluation.py \
    --task mcq \
    --model_alias "llava_onevision_0.5b" \
    --anno_path "your_dataset/your_dataset_annotations_file.json" \
    --video_dir_path "data/your_dataset/videos/" \
    --save_dir "results/your_dataset_mcq_run" \
    --sample_fps 1.0 \
    --block_size 32 \
    --retrieve_per_chunk 4 \
    --num_chunks 1
```

---

# 💬 Open-Ended Evaluation

```bash
python run_evaluation.py \
    --task open-ended \
    --model_alias "llava_onevision_0.5b" \
    --anno_path "your_dataset/your_dataset_annotations_file.json" \
    --video_dir_path "data/your_dataset/videos/" \
    --save_dir "results/your_dataset_open_ended_run" \
    --sample_fps 1.0 \
    --block_size 16 \
    --retrieve_per_chunk 4 \
    --num_chunks 1
```

---

# ⚙️ Important Parameters

| Parameter | Description |
|---|---|
| `sample_fps` | Frame sampling rate |
| `block_size` | Frames processed per chunk |
| `retrieve_per_chunk` | Number of top features retained |
| `num_chunks` | Number of sequential chunks |
| `task` | `mcq` or `open-ended` |

---

# 🔧 Technical Optimizations Implemented

## ✅ VRAM Optimization

Integrated:

```python
load_in_8bit=True
```

using `bitsandbytes` to significantly reduce GPU memory usage.

---

## ✅ Stable Sequential Inference

Implemented chunk-wise streaming inference with fixed-size memory bank updates to maintain stable VRAM consumption even for long videos.

---

# 🖥️ Hardware Compatibility

Tested on:

- NVIDIA RTX 3050 4GB
- NVIDIA RTX 4090

---

# 📈 Applications

SVFR can be applied to:

- Long-video understanding
- Streaming VideoQA
- Efficient multimodal reasoning
- Resource-constrained VideoLLM deployment
- Surveillance video analysis
- Educational video understanding

---

# 🔮 Future Improvements

- Adaptive chunk sizing
- Dynamic retrieval mechanisms
- Query complexity-aware retrieval
- Semantic block merging
- Streaming real-time VideoQA
- Reinforcement-based memory refinement

---

# 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{svfr2026,
  title={Sequential Visual Feature Refinement for Memory-Efficient Video Question Answering},
  author={Shaikh Saniya Ali and Contributors},
  year={2026}
}
```

---

# 👩‍💻 Author

Developed by:

**Shaikh Saniya Ali ,**
**Vishal P**

GitHub:  
https://github.com/Saniya-712006

---

# ⭐ Acknowledgements

This project builds upon advancements in:

- LLaVA-OneVision
- Video-LLaVA
- Long-context VideoQA
- Efficient multimodal retrieval systems

---

# 📄 License

This project is released under the MIT License.
