import warnings
import random
import json
import os
import math
import argparse
import torch
import csv
from tqdm import tqdm
from logzero import logger

from video_llm_wrapper import VideoLLMWrapper, MODEL_MAP
from memory_bank import MemoryBank

class BaseVQA:
    def __init__(self, anno, save_dir, sample_fps, video_dir_path,
                 model_alias, args,
                 num_chunks=None, chunk_idx=None,
                 block_size=32, memory_bank_size=256):

        self.args = args
        self.save_dir = save_dir
        self.sample_fps = sample_fps
        self.block_size = block_size
        self.args.video_dir_path = video_dir_path 
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx
        
        model_cfg = MODEL_MAP.get(model_alias)
        if model_cfg is None:
            raise ValueError(f"Unknown model alias '{model_alias}'.")
        
        model_path = model_cfg["path"]
        model_type = model_cfg["type"]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}.")

        logger.info(f"Loading model: {model_path} (type={model_type})")
        self.qa_model = VideoLLMWrapper(
            model_path=model_path, model_type=model_type, args=self.args
        )
        self.memory_bank = MemoryBank(max_size=memory_bank_size, device=self.qa_model.device)

        if self.num_chunks is not None and self.chunk_idx is not None:
            self.anno = self.get_chunk(anno, self.num_chunks, self.chunk_idx)
        else:
            self.anno = anno

        self.csv_file = None
        self.csv_writer = None
        self._setup_csv_logging()

    def _setup_csv_logging(self):
        filename = f"chunk_{self.chunk_idx}.csv"
        out_path = os.path.join(self.save_dir, filename)
        
        try:
            self.csv_file = open(out_path, 'w', newline='', encoding='utf-8')
            # fieldnames = ["video_id", "question", "gt_answer", "pred_answer"]

            # -------------- latency calculation modification --------------
            fieldnames = ["video_id", "question", "gt_answer", "pred_answer", "video_duration_s", "inference_time_ms"]
            # ----------------------------- end of modification -----------------------------
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
        except IOError as e:
            logger.error(f"Failed to open CSV file: {e}")
            raise

    def write_csv_row(self, **kwargs):
        if self.csv_writer:
            self.csv_writer.writerow(kwargs)
            self.csv_file.flush()

    def get_chunk(self, data_list, n_chunks, chunk_idx):
        chunk_size = math.ceil(len(data_list) / n_chunks)
        start = chunk_idx * chunk_size
        end = start + chunk_size
        logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} with {len(data_list[start:end])} samples.")
        return data_list[start:end]

    def analyze_a_video(self, video_sample):
        raise NotImplementedError("Subclasses must implement this.")

    def analyze(self, debug=False):
        video_annos = self.anno[:5] if debug else self.anno
        if not video_annos:
            logger.warning("Annotation list is empty.")
            return

        try:
            desc = f"Chunk {self.chunk_idx}"
            for video_sample in tqdm(video_annos, desc=desc):
                self.analyze_a_video(video_sample)
        except Exception as e:
            logger.error(f"Unhandled error in analysis loop: {e}", exc_info=True)
        finally:
            if self.csv_file:
                self.csv_file.close()

def work(QA_CLASS):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_alias", type=str, required=True)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--video_dir_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--memory_bank_size", type=int, default=256)
    parser.add_argument("--retrieve_per_chunk", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, required=True)
    parser.add_argument("--chunk_idx", type=int, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.debug:
        warnings.filterwarnings("ignore")
    random.seed(42)
    torch.manual_seed(42)
    
    try:
        with open(args.anno_path, 'r', encoding='utf-8') as f:
            anno = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load annotation file: {e}")
        return

    vqa_runner = QA_CLASS(
        anno=anno,
        save_dir=args.save_dir,
        sample_fps=args.sample_fps,
        video_dir_path=args.video_dir_path,
        model_alias=args.model_alias,
        block_size=args.block_size,
        memory_bank_size=args.memory_bank_size,
        num_chunks=args.num_chunks,
        chunk_idx=args.chunk_idx,
        args=args
    )
    vqa_runner.analyze(debug=args.debug)